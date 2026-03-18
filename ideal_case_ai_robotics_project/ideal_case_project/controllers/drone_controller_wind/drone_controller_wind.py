# drone_controller_mc_final.py
# 8/10 university capstone version
# - reproducible Monte-Carlo (world reload)
# - auto-batch 30 seeds × 3 wind speeds × 4 farms
# - crash / timeout detection
# - ground-truth recall
# - WCET logging
# - YAML config
# - Initial user selection for wind speed and farm before batch starts
# ------------------------------------------------------------------
import math, heapq, json, os, tempfile, time, random, csv, pathlib
import numpy as np, cv2
from controller import Robot, Motor, GPS, Gyro, InertialUnit, Camera, Keyboard
import pathlib
from yaml_loader import load_yaml
CFG = load_yaml("config.yaml")
# ----------------------------------------------------------
# 0.  CONFIG (YAML) – no more magic numbers scattered
# ----------------------------------------------------------
#CFG = yaml.safe_load(pathlib.Path("config.yaml").read_text())

# ----------------------------------------------------------
# 1.  WEBOTS DEVICE SET-UP
# ----------------------------------------------------------
robot = Robot()
timestep = int(robot.getBasicTimeStep())
kb = Keyboard()
kb.enable(timestep)

imu  = robot.getDevice("inertial unit"); imu.enable(timestep)
gps  = robot.getDevice("gps");           gps.enable(timestep)
gyro = robot.getDevice("gyro");          gyro.enable(timestep)
camera = robot.getDevice("camera")
W = H = 0
if camera:
    camera.enable(timestep)
    W, H = camera.getWidth(), camera.getHeight()

def get_motor(name):
    try:
        m = robot.getDevice(name)
        if isinstance(m, Motor):
            m.setPosition(float('inf'))
            m.setVelocity(0.0)
            return m
    except:
        pass
    return None

mFL = get_motor("front left propeller")
mFR = get_motor("front right propeller")
mRL = get_motor("rear left propeller")
mRR = get_motor("rear right propeller")
motors = [mFL, mFR, mRL, mRR]

# ----------------------------------------------------------
# 2.  CONSTANTS FROM YAML
# ----------------------------------------------------------
k_vertical_thrust = CFG["k_vertical_thrust"]
k_vertical_offset = CFG["k_vertical_offset"]
k_vertical_p      = CFG["k_vertical_p"]
k_roll_p          = CFG["k_roll_p"]
k_pitch_p         = CFG["k_pitch_p"]

WIND_MS_LIST      = CFG["wind_ms"]
FARM_LIST         = CFG.get("farms_mc", [1,2,3,4])
SEEDS_PER_WIND    = CFG["seeds_per_wind"]
MISSION_T_LIMIT   = CFG["mission_timeout"]
CRASH_TILT        = CFG["crash_tilt_deg"] * math.pi / 180
CRASH_Z_MIN       = CFG["crash_z_min"]
YELLOW_LO         = tuple(CFG["yellow_hsv_lo"])
YELLOW_HI         = tuple(CFG["yellow_hsv_hi"])

# ----------------------------------------------------------
# 3.  RUN DIRECTORY
# ----------------------------------------------------------
RUN_DIR = pathlib.Path(CFG["run_dir_root"]) / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------
# 4.  LOAD FARM LAYOUT
# ----------------------------------------------------------
TMP = tempfile.gettempdir()
LAYOUT_PATH = pathlib.Path(TMP) / "farm_layout.json"
while not LAYOUT_PATH.exists():
    if robot.step(timestep) == -1:
        pass
with open(LAYOUT_PATH) as f:
    L = json.load(f)

FARMS = {}
for k, info in L["farms"].items():
    FARMS[int(k)] = {
        "bbox": (tuple(info["bbox"]["min"]), tuple(info["bbox"]["max"])),
        "init": tuple(info["init"]),
        "size": info["size"]
    }
LAND_PT = (L["land"]["x"], L["land"]["y"]) if "land" in L else None
HOME_PT = (L["home"]["x"], L["home"]["y"]) if "home" in L else None

# ----------------------------------------------------------
# 5.  GROUND-TRUTH PEST COUNTS
# ----------------------------------------------------------
def load_gt_counts():
    try:
        with open(pathlib.Path(TMP) / "farm_groundtruth.json") as f:
            gt = json.load(f)
        counts = {}
        for p in gt.get("plants", []):
            if int(p.get("pest", 0)) == 1:
                farm = int(p.get("farm", 1))
                counts[farm] = counts.get(farm, 0) + 1
        return counts
    except:
        return None

GT_COUNTS = load_gt_counts()

# ----------------------------------------------------------
# 6.  FILES
# ----------------------------------------------------------
MISSION_CSV = RUN_DIR / "mission_log.csv"
DETECTIONS_CSV = RUN_DIR / "detections.csv"
PEST_JSON = RUN_DIR / "pest_summary.json"
PEST_TXT = RUN_DIR / "pest_summary.txt"
BATCH_SUMMARY = RUN_DIR / "batch_summary.csv"
RESET_FLAG = pathlib.Path(TMP) / "reset_requested.flag"

# ----------------------------------------------------------
# 7.  HELPER FUNCTIONS
# ----------------------------------------------------------
def clamp(v, a, b):
    return max(a, min(b, v))

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start, goal, grid):
    H = len(grid)  # rows (y_dim)
    W = len(grid[0])  # cols (x_dim)
    pq = []
    heapq.heappush(pq, (heuristic(start, goal), 0, start))
    came = {}
    g = {start: 0}
    while pq:
        _, c, u = heapq.heappop(pq)
        if u == goal:
            path = [goal]
            while u in came:
                u = came[u]
                path.append(u)
            return list(reversed(path))
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            v = (u[0] + dx, u[1] + dy)
            if 0 <= v[0] < W and 0 <= v[1] < H and grid[v[1]][v[0]] == 0:
                nc = c + 1
                if v not in g or nc < g[v]:
                    g[v] = nc
                    came[v] = u
                    heapq.heappush(pq, (nc + heuristic(v, goal), nc, v))
    return None

def build_grid_and_path(bbox, cell, row_step):
    (xmin, ymin), (xmax, ymax) = bbox
    w = max(1, int((xmax - xmin) / cell))  # cols (x)
    h = max(1, int((ymax - ymin) / cell))  # rows (y)
    grid = [[0 for _ in range(w)] for _ in range(h)]
    path = []
    for r in range(0, h, row_step):
        row = [(c, r) for c in range(w)] if (r // row_step) % 2 == 0 else [(c, r) for c in range(w - 1, -1, -1)]
        path.extend(row)
    return grid, path, w, h

def cell_to_world(i, j, bbox, cell):
    xmin, ymin = bbox[0]
    return xmin + (i + 0.5) * cell, ymin + (j + 0.5) * cell

def world_to_cell(x, y, bbox, cell):
    (xmin, ymin), (xmax, ymax) = bbox
    return int((x - xmin) / cell), int((y - ymin) / cell)

def in_bbox(x, y, bbox):
    (xmin, ymin), (xmax, ymax) = bbox
    return xmin <= x <= xmax and ymin <= y <= ymax

def inflate(grid, r):
    H = len(grid)  # rows
    W = len(grid[0])  # cols
    ones = [(i, j) for j in range(H) for i in range(W) if grid[j][i] == 1]
    for i, j in ones:
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                ii, jj = i + di, j + dj
                if 0 <= ii < W and 0 <= jj < H:
                    grid[jj][ii] = 1

def detect_pests(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(YELLOW_LO, dtype=np.uint8), np.array(YELLOW_HI, dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a > 30:
            x, y, w, h = cv2.boundingRect(c)
            out.append((x + w / 2, y + h / 2, a))
    return out

def project_to_ground(px, py, alt, fov_deg=60):
    if W == 0 or H == 0 or alt <= 0.1:
        return 0.0, 0.0
    gw = 2.0 * alt * math.tan(math.radians(fov_deg / 2))
    sx = (px / W - 0.5) * gw
    sy = (py / H - 0.5) * gw
    return sx, sy

def safe_goal(ci, cj, grid, r=1):
    """return nearest free cell if goal occupied"""
    H, W = len(grid), len(grid[0])
    if grid[cj][ci] == 0:  # grid[row][col]
        return ci, cj
    for d in range(1, max(H, W)):
        for dx in range(-d, d + 1):
            for dy in range(-d, d + 1):
                ii, jj = ci + dx, cj + dy
                if 0 <= ii < W and 0 <= jj < H and grid[jj][ii] == 0:
                    return ii, jj
    return ci, cj

def wind_bias(wx, wy, k=0.04):
    r_bias = clamp(-k * wy, -0.12, 0.12)  # roll bias
    p_bias = clamp( k * wx, -0.12, 0.12)  # pitch bias
    return r_bias, p_bias

# ----------------------------------------------------------
# 8.  WIND MODEL
# ----------------------------------------------------------
class Wind:
    def __init__(self, base, tau, sigma):
        self.base = base
        self.tau = tau
        self.sigma = sigma
        self.x = 0.0
        self.y = 0.0

    def update(self, dt):
        for k in range(2):
            w = self.x if k == 0 else self.y
            dw = -(w / self.tau) * dt + self.sigma * math.sqrt(2.0 / self.tau) * math.sqrt(dt) * random.gauss(0, 1)
            if k == 0:
                self.x += dw
            else:
                self.y += dw
        return self.base + self.x, self.y

# ----------------------------------------------------------
# 9.  MONTE-CARLO BATCH CONTROLLER
# ----------------------------------------------------------
class Batch:
    def __init__(self):
        self.wind_idx = 0
        self.seed_idx = 0
        self.farm_idx = 0
        self.active = False
        self.wind_ms = WIND_MS_LIST[0]
        self.wind = Wind(self.wind_ms, CFG["gust_tau"], CFG["gust_sigma"])
        self.initial_selected = False  # Flag for initial user selection

    def next(self):
        if self.seed_idx >= SEEDS_PER_WIND:
            self.seed_idx = 0
            self.farm_idx += 1
            if self.farm_idx >= len(FARM_LIST):
                self.farm_idx = 0
                self.wind_idx += 1
                if self.wind_idx >= len(WIND_MS_LIST):
                    return False
                self.wind_ms = WIND_MS_LIST[self.wind_idx]
                self.wind = Wind(self.wind_ms, CFG["gust_tau"], CFG["gust_sigma"])
        random.seed(self.seed_idx)
        self.seed_idx += 1
        self.active = True
        return True

batch = Batch()

# ----------------------------------------------------------
# 10.  MISSION STATE
# ----------------------------------------------------------
phase = "takeoff"
selected_farm = 1
target_hover_alt = 2.5
target_survey_alt = 1.5
target_alt = target_hover_alt
target_x, target_y = 0.0, 0.0
home_xy = None
land_xy = None
path = []
path_idx = 0
replan_timer = 0.0
replan_count = 0
unique_cells = set()
pest_records = []
mission_started_t = None
loop_wcet = 0.0

# ----------------------------------------------------------
# 11.  PRE-BUILD GRIDS / PATHS
# ----------------------------------------------------------
farm_runtime = {}
for k, finfo in FARMS.items():
    cell = 0.5 if finfo["size"] == 10.0 else 1.0
    grid, lawn, Wg, Hg = build_grid_and_path(finfo["bbox"], cell, 2)
    farm_runtime[k] = {
        "cell": cell,
        "grid": grid,
        "W": Wg,  # cols
        "H": Hg,  # rows
        "path_world": [cell_to_world(i, j, finfo["bbox"], cell) for (i, j) in lawn]
    }

# ----------------------------------------------------------
# 12.  CSV HEADER
# ----------------------------------------------------------
def log_mission(wind, farm, mtime, uniq, repl, crash, frac, wcet_ms=0.0):
    first = not MISSION_CSV.exists()
    with open(MISSION_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if first:
            writer.writerow(["wind_ms", "farm", "mission_time_s", "unique_pests", "replans", "crash", "found_frac", "wcet_ms"])
        writer.writerow([wind, farm, round(mtime, 2), uniq, repl, int(crash), round(frac, 3), round(wcet_ms, 2)])

# ----------------------------------------------------------
# 13.  CRASH / TIMEOUT CHECK
# ----------------------------------------------------------
def is_crashed(roll, pitch, z):
    return abs(roll) > CRASH_TILT or abs(pitch) > CRASH_TILT or z < CRASH_Z_MIN

# ----------------------------------------------------------
# 14.  CONTROL MIXER
# ----------------------------------------------------------
def set_motors(thrust, roll=0, pitch=0, yaw=0):
    if None in motors:
        return
    fl = thrust - roll + pitch - yaw
    fr = thrust + roll + pitch + yaw
    rl = thrust - roll - pitch + yaw
    rr = thrust + roll - pitch - yaw
    mFL.setVelocity(fl)
    mFR.setVelocity(-fr)
    mRL.setVelocity(-rl)
    mRR.setVelocity(rr)

# ==========================================================
# 15.  MAIN LOOP
# ==========================================================
print("[Batch] Starting automatic Monte-Carlo batch...")
print(f"Runs: {len(WIND_MS_LIST)} winds × {SEEDS_PER_WIND} seeds × {len(FARM_LIST)} farms = {len(WIND_MS_LIST)*SEEDS_PER_WIND*len(FARM_LIST)} missions")
print("Initial setup: In idle_hover, press Q/W/E for wind (0/1/2 m/s), then 1/2/3/4 for farm to start first mission. Subsequent missions auto.")

while robot.step(timestep) != -1:
    t_loop = time.perf_counter()
    # ---------- sensors ----------
    rpy = imu.getRollPitchYaw()
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    x, y, z = gps.getValues()
    if home_xy is None:
        home_xy = (x, y)
        land_xy = LAND_PT if LAND_PT else (home_xy[0] + 5.0, home_xy[1])
        target_x, target_y = home_xy

    key = kb.getKey()  # Get key every loop for input handling

    # ---------- batch start ----------
    if phase == "idle_hover":
        if not batch.initial_selected:
            # Initial user selection for wind and farm
            if key in (ord('q'), ord('Q')):
                batch.wind_idx = 0
                batch.wind_ms = WIND_MS_LIST[0]
                batch.wind = Wind(batch.wind_ms, CFG["gust_tau"], CFG["gust_sigma"])
                print(f"[Wind Selected] {batch.wind_ms} m/s (Q=0, W=1, E=2)")
            if key in (ord('w'), ord('W')):
                batch.wind_idx = 1
                batch.wind_ms = WIND_MS_LIST[1]
                batch.wind = Wind(batch.wind_ms, CFG["gust_tau"], CFG["gust_sigma"])
                print(f"[Wind Selected] {batch.wind_ms} m/s (Q=0, W=1, E=2)")
            if key in (ord('e'), ord('E')):
                batch.wind_idx = 2
                batch.wind_ms = WIND_MS_LIST[2]
                batch.wind = Wind(batch.wind_ms, CFG["gust_tau"], CFG["gust_sigma"])
                print(f"[Wind Selected] {batch.wind_ms} m/s (Q=0, W=1, E=2)")
            if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                selected_farm = int(chr(key))
                if selected_farm in FARM_LIST:
                    batch.farm_idx = FARM_LIST.index(selected_farm)
                    batch.initial_selected = True
                    batch.seed_idx = 0  # Start seeds from 0
                    random.seed(batch.seed_idx)  # Seed for first run
                    batch.active = True
                    path_idx = 0
                    replan_count = 0
                    unique_cells.clear()
                    pest_records.clear()
                    mission_started_t = robot.getTime()
                    phase = "nav_to_init"
                    print(f"[Initial Run] Farm {selected_farm} start. Wind={batch.wind_ms} m/s, Seed=0")
                else:
                    print(f"[Invalid Farm] {selected_farm} not in list {FARM_LIST}. Try 1-4.")
        else:
            # Auto batch for subsequent runs
            if not batch.active:
                if not batch.next():
                    # all runs finished → compute summary
                    try:
                        import pandas as pd
                        df = pd.read_csv(MISSION_CSV)
                        summary = df.groupby(["wind_ms", "farm"]).agg(
                            mean_time  =("mission_time_s", "mean"),
                            std_time   =("mission_time_s", "std"),
                            mean_pest  =("unique_pests", "mean"),
                            std_pest   =("unique_pests", "std"),
                            mean_frac  =("found_frac", "mean"),
                            std_frac   =("found_frac", "std"),
                            crash_rate =("crash", "mean")
                        ).round(3)
                        summary.to_csv(BATCH_SUMMARY)
                        print("\n===== BATCH SUMMARY =====")
                        print(summary)
                    except ImportError:
                        print("\n===== BATCH SUMMARY (no pandas) =====")
                        print("See mission_log.csv for details.")
                    break
                # start new mission
                selected_farm = FARM_LIST[batch.farm_idx]
                path_idx = 0
                replan_count = 0
                unique_cells.clear()
                pest_records.clear()
                mission_started_t = robot.getTime()
                phase = "nav_to_init"
                print(f"[Auto Run] Farm {selected_farm}, Wind={batch.wind_ms} m/s, Seed={batch.seed_idx - 1}")

    # ---------- crash / timeout ----------
    if mission_started_t is not None:
        if is_crashed(roll, pitch, z) or (robot.getTime() - mission_started_t) > MISSION_T_LIMIT:
            uniq = len({(ci, cj) for (f, ci, cj) in unique_cells if f == selected_farm})
            tc = (GT_COUNTS or {}).get(selected_farm, 0)
            frac = (uniq / max(1, tc)) if tc else 0.0
            log_mission(batch.wind_ms, selected_farm, robot.getTime() - mission_started_t, uniq, replan_count, True, frac, loop_wcet * 1000)
            # request world reload for next seed
            batch.active = False
            RESET_FLAG.touch()
            phase = "idle_hover"
            continue

    # ---------- wind ----------
    dt = timestep / 1000.0
    wx, wy = batch.wind.update(dt)
    apply_wind = phase not in ("takeoff", "idle_hover", "land")
    r_bias, p_bias = wind_bias(wx, wy) if apply_wind else (0.0, 0.0)

    # ---------- navigation ----------
    dx = target_x - x
    dy = target_y - y
    pitch_dist = clamp(-1.0 * dx + p_bias, -0.35, 0.35)
    roll_dist  = clamp( 1.0 * dy + r_bias, -0.35, 0.35)

    # ---------- phase machine ----------
    if phase == "takeoff":
        target_alt = target_hover_alt
        if abs(z - target_alt) < 0.25:
            phase = "idle_hover"
            if not batch.initial_selected:
                print("Hovering. Select wind (Q=0, W=1, E=2 m/s) then farm (1-4) to start batch.")
    elif phase == "idle_hover":
        pass
    elif phase == "nav_to_init":
        finfo = FARMS[selected_farm]
        ix, iy = finfo["init"]
        target_x, target_y = ix, iy
        target_alt = target_hover_alt
        if math.hypot(x - ix, y - iy) < 0.7:
            phase = "descend_to_survey"
    elif phase == "descend_to_survey":
        target_alt = target_survey_alt
        if abs(z - target_alt) < 0.2:
            path_idx = 0
            phase = "follow_path"
    elif phase == "follow_path":
        finfo = FARMS[selected_farm]
        run = farm_runtime[selected_farm]
        path = run["path_world"]
        cell = run["cell"]
        bbox = finfo["bbox"]
        if path_idx < len(path):
            target_x, target_y = path[path_idx]
            if math.hypot(x - target_x, y - target_y) < max(0.4, 0.8 * cell):
                path_idx += 1
        else:
            target_alt = target_hover_alt
            phase = "return_home"
        # ---- vision ----
        if camera and W > 0 and H > 0 and (int(robot.getTime() * 5) % 5 == 0):
            img_b = camera.getImage()
            if img_b:
                img = np.frombuffer(img_b, np.uint8).reshape((H, W, 4))[:, :, :3]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                for px, py, area in detect_pests(img):
                    sx, sy = project_to_ground(px, py, z)
                    wx = x + math.cos(yaw) * sx - math.sin(yaw) * sy
                    wy = y + math.sin(yaw) * sx + math.cos(yaw) * sy
                    if in_bbox(wx, wy, bbox):
                        ci, cj = world_to_cell(wx, wy, bbox, cell)
                        ci = clamp(ci, 0, run["W"] - 1)
                        cj = clamp(cj, 0, run["H"] - 1)
                        run["grid"][cj][ci] = 1
                        unique_cells.add((selected_farm, ci, cj))
                        pest_records.append({"time": round(robot.getTime(), 2),
                                             "farm": selected_farm,
                                             "x": round(wx, 3),
                                             "y": round(wy, 3),
                                             "cell_i": ci, "cell_j": cj})
        # ---- light replan ----
        replan_timer += dt
        if replan_timer > 2.0 and path_idx < len(path):
            replan_timer = 0.0
            grid = [row[:] for row in run["grid"]]
            inflate(grid, 1)
            ci, cj = world_to_cell(x, y, bbox, cell)
            ti, tj = world_to_cell(path[path_idx][0], path[path_idx][1], bbox, cell)
            ci, cj = clamp(ci, 0, run["W"] - 1), clamp(cj, 0, run["H"] - 1)
            ti, tj = clamp(ti, 0, run["W"] - 1), clamp(tj, 0, run["H"] - 1)
            ti, tj = safe_goal(ti, tj, grid)
            seg = a_star((ci, cj), (ti, tj), grid)
            if seg and len(seg) > 1:
                repl = [cell_to_world(i, j, bbox, cell) for (i, j) in seg]
                run["path_world"] = repl + run["path_world"][path_idx + 1:]
                path_idx = 0
                replan_count += 1
    elif phase == "return_home":
        target_x, target_y = home_xy
        target_alt = target_hover_alt
        if math.hypot(x - home_xy[0], y - home_xy[1]) < 0.7:
            phase = "nav_to_land"
    elif phase == "nav_to_land":
        target_x, target_y = land_xy
        if math.hypot(x - land_xy[0], y - land_xy[1]) < 0.7:
            phase = "land"
            target_alt = 0.0
    elif phase == "land":
        pass

    # ---------- control mixer ----------
    gvals = gyro.getValues()
    roll_vel = gvals[0] if len(gvals) > 0 else 0.0
    pitch_vel = gvals[1] if len(gvals) > 1 else 0.0
    clamped_alt_diff = clamp(target_alt - z + k_vertical_offset, -1.0, 1.0)
    vertical_input = k_vertical_p * (clamped_alt_diff ** 3)
    roll_input  = k_roll_p  * clamp(roll,  -1.0, 1.0) + roll_vel  + roll_dist
    pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + pitch_vel + pitch_dist
    yaw_input = 0.0
    base = k_vertical_thrust + vertical_input
    if phase == "land" and z < 1.0:
        ramp = max(0.0, z / 1.0)
        base *= ramp
    set_motors(base, roll_input, pitch_input, yaw_input)

    # ---------- mission end ----------
    if phase == "land" and z < 0.15:
        uniq = len({(ci, cj) for (f, ci, cj) in unique_cells if f == selected_farm})
        tc = (GT_COUNTS or {}).get(selected_farm, 0)
        frac = (uniq / max(1, tc)) if tc else 0.0
        log_mission(batch.wind_ms, selected_farm, robot.getTime() - mission_started_t, uniq, replan_count, False, frac, loop_wcet * 1000)
        # save files
        first = not DETECTIONS_CSV.exists()
        with open(DETECTIONS_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if first:
                w.writerow(["time", "farm", "x", "y", "cell_i", "cell_j"])
            for rec in pest_records:
                w.writerow([rec["time"], rec["farm"], rec["x"], rec["y"], rec["cell_i"], rec["cell_j"]])
        with open(PEST_JSON, "w") as f:
            json.dump({"total_unique_pests": len(unique_cells), "detections": pest_records}, f, indent=2)
        with open(PEST_TXT, "w") as f:
            f.write(f"Total unique pests detected: {len(unique_cells)}\n")
            for rec in pest_records:
                f.write(f"Farm {rec['farm']} | t={rec['time']}s | x={rec['x']} y={rec['y']} | cell=({rec['cell_i']},{rec['cell_j']})\n")
        # request world reload for next seed
        batch.active = False
        RESET_FLAG.touch()
        phase = "idle_hover"

    # ---------- WCET ----------
    loop_dt = time.perf_counter() - t_loop
    if loop_dt > loop_wcet:
        loop_wcet = loop_dt
    if int(robot.getTime()) % 5 == 0:
        print(f"[{int(robot.getTime())}s] phase={phase} wind={batch.wind_ms:.1f} z={z:.2f} wcet={loop_dt*1000:.2f} ms")

# ---------- never reached ----------