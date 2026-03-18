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
kb = Keyboard(); kb.enable(timestep)

imu  = robot.getDevice("inertial unit"); imu.enable(timestep)
gps  = robot.getDevice("gps");           gps.enable(timestep)
gyro = robot.getDevice("gyro");          gyro.enable(timestep)

camera = robot.getDevice("camera")
W = H = 0
if camera:
    camera.enable(timestep)
    W, H = camera.getWidth(), camera.getHeight()

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

print("Boot: taking off to a LOW hover and holding. Press 1/2/3/4 to start at a farm INIT.")

# --------------------- Main loop ---------------------
while robot.step(timestep) != -1:
    t = robot.getTime()

    # Sensors
    rpy   = imu.getRollPitchYaw()
    roll  = rpy[0]; pitch = rpy[1]; yaw = rpy[2]
    gvals = gyro.getValues()
    roll_vel  = gvals[0] if len(gvals)>0 else 0.0
    pitch_vel = gvals[1] if len(gvals)>1 else 0.0
    x, y, z = gps.getValues()  # z is altitude in this world

    if home_xy is None:
        home_xy = (x, y)
        land_xy = LAND_PT if LAND_PT else (home_xy[0] + 5.0, home_xy[1])
        target_x, target_y = home_xy

    # Gentle XY corrections (keep stable)
    dx = target_x - x
    dy = target_y - y
    pitch_dist = clamp(-1.0 * dx, -0.35, 0.35)   # smaller = more stable
    roll_dist  = clamp( 1.0 * dy, -0.35, 0.35)
    yaw_dist   = 0.0

    # --- Phase machine ---
    key = kb.getKey()
    if phase == "idle_hover" and key in [ord('1'),ord('2'),ord('3'),ord('4')]:
        selected_farm = int(chr(key))
        phase = "nav_to_init"
        path_idx = 0
        print(f"Farm {selected_farm} selected. Navigating to INIT…")

    if phase == "takeoff":
        if abs(z - target_alt) < 0.25:
            phase = "idle_hover"
            print("Hovering. Press 1/2/3/4 to choose a farm.")
    elif phase == "idle_hover":
        # hold position over home_xy at hover altitude, wait for key
        pass
    elif phase == "nav_to_init":
        finfo = FARMS[selected_farm]
        ix, iy = finfo["init"]
        target_x, target_y = ix, iy
        target_alt = target_hover_alt
        if math.hypot(x - ix, y - iy) < 0.7:
            phase = "descend_to_survey"
            print("At INIT. Descending to survey altitude…")
    elif phase == "descend_to_survey":
        target_alt = target_survey_alt
        if abs(z - target_alt) < 0.2:
            path_idx = 0
            print("Starting survey path.")
            phase = "follow_path"
    elif phase == "follow_path":
        finfo = FARMS[selected_farm]
        run   = farm_runtime[selected_farm]
        path  = run["path_world"]; cell = run["cell"]; bbox = finfo["bbox"]

        if path_idx < len(path):
            target_x, target_y = path[path_idx]
            if math.hypot(x - target_x, y - target_y) < max(0.4, 0.8*cell):
                path_idx += 1
        else:
            target_alt = target_hover_alt
            phase = "return_home"
            print("Survey complete. Returning home…")

        # Downward detection (assumes camera looks down; optional gimbal set earlier)
        if camera and W>0 and H>0 and (int(t*5) % 5 == 0):  # ~0.2s cadence
            img_b = camera.getImage()
            if img_b:
                img = np.frombuffer(img_b, np.uint8).reshape((H, W, 4))[:, :, :3]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                for (px,py,area) in detect_pests(img):
                    sx, sy = project_to_ground(px, py, z, 60.0)
                    # rotate offsets by yaw into world X–Y
                    wx = x + math.cos(yaw)*sx - math.sin(yaw)*sy
                    wy = y + math.sin(yaw)*sx + math.cos(yaw)*sy
                    if in_bbox(wx, wy, bbox):
                        ci, cj = world_to_cell(wx, wy, bbox, cell)
                        ci = clamp(ci, 0, run["W"]-1); cj = clamp(cj, 0, run["H"]-1)
                        run["grid"][cj][ci] = 1
                        detections.append((t, selected_farm, wx, wy, ci, cj))
                        # ---------- ADDED: unique counting + coordinate log ----------
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

        # Lightweight local replanning every ~2s
        replan_timer += timestep/1000.0
        if replan_timer > 2.0 and path_idx < len(path):
            replan_timer = 0.0
            grid = [row[:] for row in run["grid"]]
            inflate(grid, r=1)
            ci, cj = world_to_cell(x, y, bbox, cell)
            ti, tj = world_to_cell(path[path_idx][0], path[path_idx][1], bbox, cell)
            ci = clamp(ci,0,run["W"]-1); cj = clamp(cj,0,run["H"]-1)
            ti = clamp(ti,0,run["W"]-1); tj = clamp(tj,0,run["H"]-1)
            seg = a_star((ci,cj), (ti,tj), grid)
            if seg and len(seg) > 1:
                repl = [cell_to_world(i,j,bbox,cell) for (i,j) in seg]
                run["path_world"] = repl + run["path_world"][path_idx+1:]
                path_idx = 0
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
        pass

    # --------------------- Stable mixer (same pattern as your stable code) ---------------------
    clamped_alt_diff = clamp(target_alt - z + k_vertical_offset, -1.0, 1.0)
    vertical_input   = k_vertical_p * (clamped_alt_diff ** 3)

    roll_input  = k_roll_p  * clamp(roll,  -1.0, 1.0) + roll_vel  + roll_dist
    pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + pitch_vel + pitch_dist
    yaw_input   = yaw_dist

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

    # end mission when on ground
    if phase == "land" and z < 0.15:
        try:
            out = os.path.join(TMP, 'detections.csv')
            with open(out, 'w') as f:
                f.write('time,farm,x,y,cell_i,cell_j\n')
                for (tt,ff,wx,wy,ci,cj) in detections:
                    f.write(f'{tt:.2f},{ff},{wx:.3f},{wy:.3f},{ci},{cj}\n')

            # ---------- ADDED: summary display + files ----------
            total_unique = len(unique_cells)
            print("\n===== PEST DETECTION SUMMARY =====")
            print(f"Total unique pests detected: {total_unique}")
            for rec in pest_records:
                print(f"Farm {rec['farm']} | t={rec['time']}s | x={rec['x']} y={rec['y']} | cell=({rec['cell_i']},{rec['cell_j']})")

            sum_json = os.path.join(TMP, 'pest_summary.json')
            with open(sum_json, 'w') as f:
                json.dump({
                    "total_unique_pests": total_unique,
                    "detections": pest_records
                }, f, indent=2)

            sum_txt = os.path.join(TMP, 'pest_summary.txt')
            with open(sum_txt, 'w') as f:
                f.write(f"Total unique pests detected: {total_unique}\n")
                for rec in pest_records:
                    f.write(f"Farm {rec['farm']} | t={rec['time']}s | x={rec['x']} y={rec['y']} | cell=({rec['cell_i']},{rec['cell_j']})\n")
        except:
            pass
        break

    # Light console
    if int(t*2) % 4 == 0:
        print(f"phase={phase} | z={z:.2f}/{target_alt:.2f} | pos=({x:.1f},{y:.1f})→({target_x:.1f},{target_y:.1f})")
