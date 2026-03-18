# drone_controller.py
# Stable low-altitude hover → wait for user (1–4) → fly to selected farm INIT → lawnmower survey with
# light real-time A* avoidance (using yellow "pest" detections) → return → land.
# World: gravity is along Z, farms lie in X–Y plane at z≈0 (from your Supervisor).

import math, heapq, json, os, tempfile
import numpy as np, cv2
from controller import Robot, Motor, GPS, Gyro, InertialUnit, Camera, Keyboard

# --------------------- Webots setup ---------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())
dt = max(1e-3, timestep / 1000.0)
kb = Keyboard(); kb.enable(timestep)

imu  = robot.getDevice("inertial unit"); imu.enable(timestep)
gps  = robot.getDevice("gps");           gps.enable(timestep)
gyro = robot.getDevice("gyro");          gyro.enable(timestep)

camera = robot.getDevice("camera")
W = H = 0
if camera:
    camera.enable(timestep)
    W, H = camera.getWidth(), camera.getHeight()

# Evaluation output directory (user-requested absolute path)
EVAL_DIR = r'D:\my_project2\worlds\evaluation\outputs'
try:
    os.makedirs(EVAL_DIR, exist_ok=True)
except:
    pass
'''
# Optional: try to tilt a gimbal/tilt motor down if present (safe no-op otherwise)
def try_set_gimbal_down():
    names = ["gimbal pitch","gimbal_pitch","camera pitch","camera_pitch","tilt","camera tilt"]
    for n in names:
        try:
            dev = robot.getDevice(n)
            if isinstance(dev, Motor):
                dev.setPosition(-1.22)  # ~ -70° down
                dev.setVelocity(1.0)
                return
        except:
            pass
try_set_gimbal_down()
'''
def get_motor(name):
    try:
        m = robot.getDevice(name)
        if isinstance(m, Motor):
            m.setPosition(float('inf')); m.setVelocity(0.0)
            return m
    except:
        pass
    return None

mFL = get_motor("front left propeller")
mFR = get_motor("front right propeller")
mRL = get_motor("rear left propeller")
mRR = get_motor("rear right propeller")

# --------------------- Gains (stable baseline) ---------------------
k_vertical_thrust = 68.5
k_vertical_offset = 0.6
k_vertical_p      = 3.0
k_roll_p          = 50.0
k_pitch_p         = 30.0

def clamp(v, a, b): 
    return a if v < a else b if v > b else v

# --------------------- Load farm layout from Supervisor ---------------------
TMP = tempfile.gettempdir()
LAYOUT_PATH = os.path.join(TMP, 'farm_layout.json')
while not os.path.exists(LAYOUT_PATH):
    if robot.step(timestep) == -1:
        pass

with open(LAYOUT_PATH, 'r') as f:
    L = json.load(f)

FARMS = {}
for k, info in L["farms"].items():
    FARMS[int(k)] = {
        "bbox": (tuple(info["bbox"]["min"]), tuple(info["bbox"]["max"])),  # (x,y) bounds
        "init": tuple(info["init"]),
        "size": info["size"]
    }
LAND_PT = (L["land"]["x"], L["land"]["y"]) if "land" in L else None

# --------------------- Planning helpers (2D in X–Y) ---------------------
def heuristic(a, b): 
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def a_star(start, goal, grid):
    H = len(grid); Wg = len(grid[0])
    pq = []; heapq.heappush(pq, (heuristic(start,goal), 0, start))
    came = {}; g = {start: 0}
    while pq:
        _, c, u = heapq.heappop(pq)
        if u == goal:
            path = [u]
            while u in came:
                u = came[u]; path.append(u)
            return list(reversed(path))
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            v = (u[0]+dx, u[1]+dy)
            if 0<=v[0]<Wg and 0<=v[1]<H and grid[v[1]][v[0]]==0:
                nc = c+1
                if v not in g or nc < g[v]:
                    g[v] = nc; came[v] = u
                    heapq.heappush(pq, (nc + heuristic(v,goal), nc, v))
    return []

def gen_grid_and_lawn(bbox, cell, row_step_cells=2):
    (xmin, ymin), (xmax, ymax) = bbox
    Wg = max(1, int((xmax-xmin)/cell))
    Hg = max(1, int((ymax-ymin)/cell))
    grid = [[0 for _ in range(Wg)] for _ in range(Hg)]
    lawn = []
    for r in range(0, Hg, row_step_cells):
        row = [(c,r) for c in range(0,Wg)] if (r//row_step_cells)%2==0 else [(c,r) for c in range(Wg-1,-1,-1)]
        lawn.extend(row)
    return grid, lawn, Wg, Hg

def cell_to_world(i, j, bbox, cell):
    (xmin, ymin), _ = bbox
    return xmin + (i + 0.5)*cell, ymin + (j + 0.5)*cell

def world_to_cell(x, y, bbox, cell):
    (xmin, ymin), (xmax, ymax) = bbox
    return int((x - xmin)/cell), int((y - ymin)/cell)

def in_bbox(x, y, bbox):
    (xmin, ymin), (xmax, ymax) = bbox
    return xmin <= x <= xmax and ymin <= y <= ymax

def inflate(grid, r=1):
    H=len(grid); Wg=len(grid[0])
    ones=[(i,j) for j in range(H) for i in range(Wg) if grid[j][i]==1]
    for i,j in ones:
        for di in range(-r,r+1):
            for dj in range(-r,r+1):
                ii=i+di; jj=j+dj
                if 0<=ii<Wg and 0<=jj<H: grid[jj][ii]=1

# --------------------- Simple "pest" detector (yellow) ---------------------
def detect_pests(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([18, 80, 120])
    upper = np.array([35, 255, 255])
    mask  = cv2.inRange(hsv, lower, upper)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
    cnts,_= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out=[]
    for c in cnts:
        a = cv2.contourArea(c)
        if a > 30:
            x,y,w,h = cv2.boundingRect(c)
            out.append((x+w/2, y+h/2, a))
    return out

def project_to_ground(px, py, alt, fov_deg=60.0):
    if W == 0 or H == 0: return 0.0, 0.0
    gw = 2.0 * alt * math.tan(math.radians(fov_deg/2.0))
    sx = (px/W - 0.5) * gw
    sy = (py/H - 0.5) * gw
    return sx, sy

# --------------------- Per-farm runtime ---------------------
farm_runtime = {}
for k, finfo in FARMS.items():
    cell = 0.5 if finfo["size"] == 10.0 else 1.0
    grid, lawn, Wg, Hg = gen_grid_and_lawn(finfo["bbox"], cell, row_step_cells=2)
    farm_runtime[k] = {
        "cell": cell, "grid": grid, "W": Wg, "H": Hg,
        "path_world": [cell_to_world(i,j,finfo["bbox"],cell) for (i,j) in lawn]
    }

# --------------------- Mission state ---------------------
phase = "takeoff"          # → idle_hover → (on key 1–4) nav_to_init → descend_to_survey → follow_path → return → land
selected_farm = None
target_hover_alt = 2.5     # low hover (don’t go high)
target_survey_alt = 1.5
target_alt = target_hover_alt
target_x = 0.0
target_y = 0.0
home_xy = None
land_xy = None
path_idx = 0
replan_timer = 0.0
detections = []

# ---------- ADDED: pest counting & unique coordinate tracking ----------
unique_cells = set()   # (farm, cell_i, cell_j)
pest_records = []      # list of dicts with time/farm/world coords/cell indices

# ---------- ADDED: robust navigator (rate-limited + local segments) ----------
NAV_MAX_SPEED = 1.0          # m/s cap for commanded XY target motion
NAV_MIN_SPEED = 0.25         # reduce near waypoints to prevent jerks
WAYPOINT_RADIUS = 0.6        # arrive threshold (scaled by cell below)
LEASH_EXTRA = 0.9            # extra radius factor per-cell
ACTIVE_SEGMENT = []          # list of (x,y) local waypoints from A* (if any)
ACTIVE_INDEX = 0

def rate_limit_move(curr_xy, goal_xy, max_speed, dt):
    """Move 'curr_xy' toward 'goal_xy' by at most max_speed*dt (meters)."""
    cx, cy = curr_xy; gx, gy = goal_xy
    dx = gx - cx; dy = gy - cy
    d = math.hypot(dx, dy)
    if d <= 1e-6:
        return gx, gy
    step = max_speed * dt
    if d <= step:
        return gx, gy
    s = step / d
    return (cx + dx*s, cy + dy*s)

def next_goal(run, path, path_idx, x, y):
    """Return the next desired goal (x,y) considering ACTIVE_SEGMENT first."""
    global ACTIVE_SEGMENT, ACTIVE_INDEX
    if ACTIVE_SEGMENT and ACTIVE_INDEX < len(ACTIVE_SEGMENT):
        return ACTIVE_SEGMENT[ACTIVE_INDEX]
    if path_idx < len(path):
        return path[path_idx]
    return None

def start_local_segment(seg_world):
    """Start following a local A* segment without mutating the global lawn path."""
    global ACTIVE_SEGMENT, ACTIVE_INDEX
    ACTIVE_SEGMENT = seg_world[:]  # copy
    ACTIVE_INDEX = 0

def advance_if_reached(goal_xy, x, y, arrive_r):
    """Check arrival at goal; if local segment active, advance within it, else signal path advance."""
    global ACTIVE_SEGMENT, ACTIVE_INDEX
    if goal_xy is None: 
        return False
    if math.hypot(x - goal_xy[0], y - goal_xy[1]) <= arrive_r:
        if ACTIVE_SEGMENT:
            ACTIVE_INDEX += 1
            if ACTIVE_INDEX >= len(ACTIVE_SEGMENT):
                ACTIVE_SEGMENT = []
                ACTIVE_INDEX = 0
        else:
            return True  # caller should increment global path_idx
    return False

# ---------- ADDED: safety guard (altitude) ----------
ALT_HOLD_MIN = 0.8   # if z falls below this, freeze XY target (no tilt push) until we climb back

# ---------- ADDED: flight metrics ----------
metrics = {
    "start_time": None,
    "end_time": None,
    "total_distance_xy": 0.0,
    "avg_speed_xy": 0.0,
    "min_alt": float("inf"),
    "max_alt": 0.0,
    "max_abs_roll_deg": 0.0,
    "max_abs_pitch_deg": 0.0,
    "phase_times": {p:0.0 for p in ["takeoff","idle_hover","nav_to_init","descend_to_survey","follow_path","return_home","nav_to_land","land"]},
    "replan_count": 0,
    "a_star_fail_count": 0,
    "alt_freeze_count": 0,
    "unique_pest_cells": 0,
    "events": []
}
last_x = None; last_y = None
last_phase = "takeoff"
def note_event(t, msg):
    metrics["events"].append({"t": round(t,2), "msg": msg})

print("Boot: taking off to a LOW hover and holding. Press 1/2/3/4 to start at a farm INIT.")

# --------------------- Main loop ---------------------
# Navigator command point (smoothed XY command that our controller tracks)
cmd_x = None
cmd_y = None

while robot.step(timestep) != -1:
    t = robot.getTime()
    if metrics["start_time"] is None:
        metrics["start_time"] = t

    # Sensors
    rpy   = imu.getRollPitchYaw()
    roll  = rpy[0]; pitch = rpy[1]; yaw = rpy[2]
    gvals = gyro.getValues()
    roll_vel  = gvals[0] if len(gvals)>0 else 0.0
    pitch_vel = gvals[1] if len(gvals)>1 else 0.0
    x, y, z = gps.getValues()  # z is altitude in this world

    # Metrics: altitude & tilt
    metrics["min_alt"] = min(metrics["min_alt"], z)
    metrics["max_alt"] = max(metrics["max_alt"], z)
    metrics["max_abs_roll_deg"]  = max(metrics["max_abs_roll_deg"],  abs(math.degrees(roll)))
    metrics["max_abs_pitch_deg"] = max(metrics["max_abs_pitch_deg"], abs(math.degrees(pitch)))

    # Metrics: distance
    if last_x is not None:
        metrics["total_distance_xy"] += math.hypot(x-last_x, y-last_y)
    last_x, last_y = x, y

    # Phase time accounting
    if last_phase != phase:
        note_event(t, f"phase→{phase}")
        last_phase = phase
    if phase in metrics["phase_times"]:
        metrics["phase_times"][phase] += dt

    if home_xy is None:
        home_xy = (x, y)
        land_xy = LAND_PT if LAND_PT else (home_xy[0] + 5.0, home_xy[1])
        target_x, target_y = home_xy
        cmd_x, cmd_y = home_xy

    # Gentle XY corrections (keep stable) — unchanged controller math
    dx_ctrl = (cmd_x if cmd_x is not None else target_x) - x
    dy_ctrl = (cmd_y if cmd_y is not None else target_y) - y
    pitch_dist = clamp(-1.0 * dx_ctrl, -0.35, 0.35)
    roll_dist  = clamp( 1.0 * dy_ctrl, -0.35, 0.35)
    yaw_dist   = 0.0

    # --- Phase machine / navigation goals ---
    key = kb.getKey()
    if phase == "idle_hover" and key in [ord('1'),ord('2'),ord('3'),ord('4')]:
        selected_farm = int(chr(key))
        phase = "nav_to_init"
        path_idx = 0
        ACTIVE_SEGMENT = []; ACTIVE_INDEX = 0
        note_event(t, f"farm_selected {selected_farm}")
        print(f"Farm {selected_farm} selected. Navigating to INIT…")

    if phase == "takeoff":
        if abs(z - target_alt) < 0.25:
            phase = "idle_hover"
            print("Hovering. Press 1/2/3/4 to choose a farm.")
        # keep cmd at home
        target_x, target_y = home_xy
    elif phase == "idle_hover":
        # hold at home_xy
        target_x, target_y = home_xy
    elif phase == "nav_to_init":
        finfo = FARMS[selected_farm]
        ix, iy = finfo["init"]
        target_alt = target_hover_alt
        goal_xy = (ix, iy)
        # Arrive logic
        arrive_r = max(WAYPOINT_RADIUS, 0.6)
        if math.hypot(x - ix, y - iy) < arrive_r:
            phase = "descend_to_survey"
            print("At INIT. Descending to survey altitude…")
        target_x, target_y = goal_xy
    elif phase == "descend_to_survey":
        target_alt = target_survey_alt
        target_x, target_y = x, y  # hold XY while descending to prevent tilt
        if abs(z - target_alt) < 0.2:
            path_idx = 0
            print("Starting survey path.")
            phase = "follow_path"
    elif phase == "follow_path":
        finfo = FARMS[selected_farm]
        run   = farm_runtime[selected_farm]
        path  = run["path_world"]; cell = run["cell"]; bbox = finfo["bbox"]

        # If altitude gets too low, freeze XY to stop wild tilting
        if z < ALT_HOLD_MIN:
            target_x, target_y = x, y
            metrics["alt_freeze_count"] += 1
        else:
            # choose next goal (prefer active A* segment if any)
            goal = next_goal(run, path, path_idx, x, y)
            if goal is None:
                target_alt = target_hover_alt
                phase = "return_home"
                print("Survey complete. Returning home…")
            else:
                target_x, target_y = goal
                # Arrive handling
                arrive_r = max(WAYPOINT_RADIUS, 0.8*cell)
                if advance_if_reached(goal, x, y, arrive_r):
                    path_idx += 1

        # Downward detection → mark grid (obstacles)
        if camera and W>0 and H>0 and (int(t*5) % 5 == 0):
            img_b = camera.getImage()
            if img_b:
                img = np.frombuffer(img_b, np.uint8).reshape((H, W, 4))[:, :, :3]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                for (px,py,area) in detect_pests(img):
                    sx, sy = project_to_ground(px, py, z, 60.0)
                    wx = x + math.cos(yaw)*sx - math.sin(yaw)*sy
                    wy = y + math.sin(yaw)*sx + math.cos(yaw)*sy
                    if in_bbox(wx, wy, bbox):
                        ci, cj = world_to_cell(wx, wy, bbox, cell)
                        ci = clamp(ci, 0, run["W"]-1); cj = clamp(cj, 0, run["H"]-1)
                        run["grid"][cj][ci] = 1
                        detections.append((t, selected_farm, wx, wy, ci, cj))
                        key_uc = (selected_farm, ci, cj)
                        if key_uc not in unique_cells:
                            unique_cells.add(key_uc)
                            pest_records.append({
                                "time": round(t,2),
                                "farm": selected_farm,
                                "x": round(wx,3),
                                "y": round(wy,3),
                                "cell_i": ci,
                                "cell_j": cj
                            })

        # Lightweight local replanning every ~2s (only if not altitude-frozen and not already on a local segment)
        replan_timer += dt
        if replan_timer > 2.0 and z >= ALT_HOLD_MIN and not ACTIVE_SEGMENT and path_idx < len(farm_runtime[selected_farm]["path_world"]):
            replan_timer = 0.0
            grid = [row[:] for row in run["grid"]]
            inflate(grid, r=1)
            ci, cj = world_to_cell(x, y, bbox, cell)
            curr_goal = farm_runtime[selected_farm]["path_world"][path_idx]
            ti, tj = world_to_cell(curr_goal[0], curr_goal[1], bbox, cell)
            ci = clamp(ci,0,run["W"]-1); cj = clamp(cj,0,run["H"]-1)
            ti = clamp(ti,0,run["W"]-1); tj = clamp(tj,0,run["H"]-1)
            seg = a_star((ci,cj), (ti,tj), grid)
            if seg and len(seg) > 1:
                repl = [cell_to_world(i,j,bbox,cell) for (i,j) in seg]
                start_local_segment(repl)  # follow locally; DO NOT mutate global path
                metrics["replan_count"] += 1
                note_event(t, f"replan len={len(repl)}")
            else:
                metrics["a_star_fail_count"] += 1
                note_event(t, "replan_failed")
    elif phase == "return_home":
        target_x, target_y = home_xy
        target_alt = target_hover_alt
        if math.hypot(x - home_xy[0], y - home_xy[1]) < 0.7:
            phase = "nav_to_land"
            print("At home. Proceeding to LAND marker…")
    elif phase == "nav_to_land":
        target_x, target_y = land_xy
        if math.hypot(x - land_xy[0], y - land_xy[1]) < 0.7:
            phase = "land"; target_alt = 0.0
            print("Landing…")
    elif phase == "land":
        target_x, target_y = x, y  # no XY commands while landing

    # --------------------- Navigator rate-limit (key stability fix) ---------------------
    # Compute desired goal (target_x,target_y) chosen above, but move cmd_x/cmd_y toward it with limited speed.
    if cmd_x is None or cmd_y is None:
        cmd_x, cmd_y = x, y
    # Slow down when close to goal to avoid snap
    dist_to_goal = math.hypot(target_x - x, target_y - y)
    max_speed = NAV_MAX_SPEED if dist_to_goal > 1.0 else max(NAV_MIN_SPEED, NAV_MAX_SPEED * dist_to_goal)
    cmd_x, cmd_y = rate_limit_move((cmd_x, cmd_y), (target_x, target_y), max_speed, dt)

    # --------------------- Stable mixer (unchanged math) ---------------------
    clamped_alt_diff = clamp(target_alt - z + k_vertical_offset, -1.0, 1.0)
    vertical_input   = k_vertical_p * (clamped_alt_diff ** 3)

    roll_input  = k_roll_p  * clamp(roll,  -1.0, 1.0) + roll_vel  + clamp( 1.0 * (cmd_y - y), -0.35, 0.35)
    pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + pitch_vel + clamp(-1.0 * (cmd_x - x), -0.35, 0.35)
    yaw_input   = 0.0

    base = k_vertical_thrust + vertical_input
    fl = base - roll_input + pitch_input - yaw_input
    fr = base + roll_input + pitch_input + yaw_input
    rl = base - roll_input - pitch_input + yaw_input
    rr = base + roll_input - pitch_input - yaw_input

    # very soft landing
    if phase == "land" and z < 1.0:
        ramp = max(0.0, z / 1.0)
        fl *= ramp; fr *= ramp; rl *= ramp; rr *= ramp

    if mFL: mFL.setVelocity(fl)
    if mFR: mFR.setVelocity(-fr)
    if mRL: mRL.setVelocity(-rl)
    if mRR: mRR.setVelocity(rr)

    # End mission when on ground
    if phase == "land" and z < 0.15:
        metrics["end_time"] = t
        elapsed = max(1e-6, metrics["end_time"] - metrics["start_time"])
        metrics["avg_speed_xy"] = metrics["total_distance_xy"] / elapsed
        metrics["unique_pest_cells"] = len(unique_cells)

        try:
            # detections
            out_csv = os.path.join(EVAL_DIR, 'detections.csv')
            with open(out_csv, 'w') as f:
                f.write('time,farm,x,y,cell_i,cell_j\n')
                for (tt,ff,wx,wy,ci,cj) in detections:
                    f.write(f'{tt:.2f},{ff},{wx:.3f},{wy:.3f},{ci},{cj}\n')

            # pest summary
            total_unique = len(unique_cells)
            sum_json = os.path.join(EVAL_DIR, 'pest_summary.json')
            with open(sum_json, 'w') as f:
                json.dump({
                    "total_unique_pests": total_unique,
                    "detections": pest_records
                }, f, indent=2)

            sum_txt = os.path.join(EVAL_DIR, 'pest_summary.txt')
            with open(sum_txt, 'w') as f:
                f.write(f"Total unique pests detected: {total_unique}\n")
                for rec in pest_records:
                    f.write(f"Farm {rec['farm']} | t={rec['time']}s | x={rec['x']} y={rec['y']} | cell=({rec['cell_i']},{rec['cell_j']})\n")

            # flight metrics
            met_json = os.path.join(EVAL_DIR, 'flight_metrics.json')
            with open(met_json, 'w') as f:
                json.dump(metrics, f, indent=2)

            met_txt = os.path.join(EVAL_DIR, 'flight_metrics.txt')
            with open(met_txt, 'w') as f:
                f.write("==== FLIGHT METRICS ====\n")
                f.write(f"Elapsed (s): {(metrics['end_time']-metrics['start_time']):.2f}\n")
                f.write(f"Min Alt (m): {metrics['min_alt']:.2f}\n")
                f.write(f"Max Alt (m): {metrics['max_alt']:.2f}\n")
                f.write(f"Dist XY (m): {metrics['total_distance_xy']:.2f}\n")
                f.write(f"Avg Speed (m/s): {metrics['avg_speed_xy']:.2f}\n")
                f.write(f"Max |roll| (deg): {metrics['max_abs_roll_deg']:.1f}\n")
                f.write(f"Max |pitch| (deg): {metrics['max_abs_pitch_deg']:.1f}\n")
                for p,sec in metrics["phase_times"].items():
                    f.write(f"Time[{p}] (s): {sec:.2f}\n")
                f.write(f"Replans: {metrics['replan_count']}\n")
                f.write(f"A* fails: {metrics['a_star_fail_count']}\n")
                f.write(f"Alt-freeze events: {metrics['alt_freeze_count']}\n")
                f.write(f"Unique pest cells: {metrics['unique_pest_cells']}\n")
                f.write("Events:\n")
                for e in metrics["events"]:
                    f.write(f"  t={e['t']}: {e['msg']}\n")
        except:
            pass
        break

    # Light console
    if int(t*2) % 4 == 0:
        print(f"phase={phase} | z={z:.2f}/{target_alt:.2f} | pos=({x:.1f},{y:.1f})→({cmd_x:.1f},{cmd_y:.1f})")
