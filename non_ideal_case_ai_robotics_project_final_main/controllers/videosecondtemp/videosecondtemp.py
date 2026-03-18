# drone_controller.py
import math, json, os, tempfile
import numpy as np, cv2
from controller import Robot, Motor, GPS, Gyro, InertialUnit, Camera, Keyboard

def clamp(v,a,b): 
    return a if v<a else b if v>b else v
def ang_wrap(a): 
    return (a + math.pi) % (2*math.pi) - math.pi

NAV_MAX_SPEED     = 0.55
NAV_MIN_SPEED     = 0.20
WAYPOINT_RADIUS   = 0.40
TURN_PAUSE_S      = 0.35
ALT_HOLD_MIN      = 0.8

TARGET_HOVER_ALT  = 2.2
TARGET_SURVEY_ALT = 1.4

k_vertical_thrust = 68.5
k_vertical_offset = 0.60
k_vertical_p      = 3.0
k_roll_p          = 45.0
k_pitch_p         = 26.0
LEAN_CAP_CRUISE   = 0.20

YAW_P      = 3.0
YAW_D      = 0.9
YAW_MAX    = 1.0
YAW_SLEW   = 0.8
YAW_ALIGN_TOL = 0.15

ROW_SPACING   = 0.7
ROW_MARGIN    = 0.01
ORIENT_AXIS   = 'x'
CONNECTOR_FRAC = 0.5

AVOID_TRIP      = 1.0
AVOID_CLEAR     = 1.4
AVOID_LAT       = 0.8
AVOID_FWD       = 1.2
AVOID_COOLDOWN  = 1.5


robot = Robot()
timestep = int(robot.getBasicTimeStep())
dt = max(1e-3, timestep/1000.0)

kb = Keyboard(); kb.enable(timestep)
imu  = robot.getDevice("inertial unit"); imu.enable(timestep)
gps  = robot.getDevice("gps");           gps.enable(timestep)
gyro = robot.getDevice("gyro");          gyro.enable(timestep)

camera = None
CAM_W=CAM_H=0
try:
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    CAM_W, CAM_H = camera.getWidth(), camera.getHeight()
except:
    pass

def get_motor(n):
    try:
        m = robot.getDevice(n)
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

TMP = tempfile.gettempdir()
LAYOUT_PATH = os.path.join(TMP, 'farm_layout.json')
while not os.path.exists(LAYOUT_PATH):
    if robot.step(timestep) == -1:
        pass
with open(LAYOUT_PATH,'r') as f:
    L = json.load(f)

FARMS = {}
for k, info in L["farms"].items():
    FARMS[ int(k) ] = {
        "bbox": (tuple(info["bbox"]["min"]), tuple(info["bbox"]["max"])),
        "init": tuple(info["init"]),
        "size": info["size"]
    }
LAND_PT = (L["land"]["x"], L["land"]["y"]) if "land" in L else None

def serpentine_with_connectors(bbox, row_spacing, start_xy, margin=ROW_MARGIN, orient_axis=ORIENT_AXIS):
    (xmin,ymin),(xmax,ymax) = bbox
    xmin2 = xmin + margin; xmax2 = xmax - margin
    ymin2 = ymin + margin; ymax2 = ymax - margin
    def compress(points):
        out=[]
        for p in points:
            if not out: out.append(p); continue
            if math.hypot(out[-1][0]-p[0], out[-1][1]-p[1])>1e-6:
                out.append(p)
        return out
    path=[]
    if orient_axis.lower() == 'x':
        ys=[]; y=ymin2
        while y <= ymax2+1e-6:
            ys.append(y); y+=row_spacing
        if not ys: ys=[(ymin+ymax)/2.0]
        sy=start_xy[1]
        k0=int(np.argmin([abs(yy-sy) for yy in ys]))
        sign = +1 if k0 < len(ys)-1 else -1
        idxs = list(range(k0, len(ys), sign)) + list(range(k0-1, -1, -sign))
        y0 = ys[idxs[0]]
        L0=(xmin2,y0); R0=(xmax2,y0)
        if abs(start_xy[0]-L0[0]) <= abs(start_xy[0]-R0[0]):
            path += [L0,R0]; last_x = R0[0]
        else:
            path += [R0,L0]; last_x = L0[0]
        for j in range(1,len(idxs)):
            y_prev = ys[idxs[j-1]]
            y_cur  = ys[idxs[j]]
            x_same = last_x
            y_mid  = y_prev + CONNECTOR_FRAC*(y_cur - y_prev)
            path += [(x_same, y_mid), (x_same, y_cur)]
            x_other = xmin2 if x_same==xmax2 else xmax2
            path += [(x_other, y_cur)]
            last_x = x_other
        return compress(path)
    else:
        xs=[]; x=xmin2
        while x <= xmax2+1e-6:
            xs.append(x); x+=row_spacing
        if not xs: xs=[(xmin+xmax)/2.0]
        sx=start_xy[0]
        k0=int(np.argmin([abs(xx-sx) for xx in xs]))
        sign = +1 if k0 < len(xs)-1 else -1
        idxs = list(range(k0, len(xs), sign)) + list(range(k0-1, -1, -sign))
        x0 = xs[idxs[0]]
        B0=(x0,ymin2); T0=(x0,ymax2)
        if abs(start_xy[1]-B0[1]) <= abs(start_xy[1]-T0[1]):
            path += [B0,T0]; last_y = T0[1]
        else:
            path += [T0,B0]; last_y = B0[1]
        for j in range(1,len(idxs)):
            x_prev = xs[idxs[j-1]]
            x_cur  = xs[idxs[j]]
            y_same = last_y
            x_mid  = x_prev + CONNECTOR_FRAC*(x_cur - x_prev)
            path += [(x_mid, y_same), (x_cur, y_same)]
            y_other = ymin2 if y_same==ymax2 else ymax2
            path += [(x_cur, y_other)]
            last_y = y_other
        return compress(path)

def clamp_to_bbox(xy, bbox, margin=ROW_MARGIN):
    (xmin,ymin),(xmax,ymax) = bbox
    return (clamp(xy[0], xmin+margin, xmax-margin),
            clamp(xy[1], ymin+margin, ymax-margin))

def front_clearance_estimates():
    return float('inf'), float('inf'), float('inf')  # placeholder for parity with distance_controller baseline

phase = "takeoff"
selected_farm = None

home_xy = None; land_xy = None
target_x = target_y = 0.0
target_alt = TARGET_HOVER_ALT

path = []; path_idx = 0
yaw_target = None
align_timer = 0.0
turn_pause_timer = 0.0

avoid_queue = []
avoid_cooldown = 0.0

print("Boot: low hover → press 1/2/3/4 to choose farm.")

cmd_x=cmd_y=None
print_t=0.0

while robot.step(timestep) != -1:
    t = robot.getTime()
    rpy = imu.getRollPitchYaw()
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    gvals = gyro.getValues()
    roll_vel  = gvals[0] if len(gvals)>0 else 0.0
    pitch_vel = gvals[1] if len(gvals)>1 else 0.0
    yaw_rate  = gvals[2] if len(gvals)>2 else 0.0

    x,y,z = gps.getValues()
    if home_xy is None:
        home_xy = (x,y)
        land_xy = LAND_PT if LAND_PT else (home_xy[0] + 5.0, home_xy[1])
        target_x, target_y = home_xy
        cmd_x, cmd_y = home_xy
        yaw_target = yaw

    key = kb.getKey()
    if phase == "idle_hover" and key in (ord('1'),ord('2'),ord('3'),ord('4')):
        selected_farm = int(chr(key))
        phase = "nav_to_init"
        fx, fy = FARMS[selected_farm]["init"]
        target_x, target_y = fx, fy
        target_alt = TARGET_HOVER_ALT
        path = []; path_idx = 0
        align_timer = 0.0; turn_pause_timer = 0.0
        avoid_queue=[]; avoid_cooldown=0.0

    if phase == "takeoff":
        target_alt = TARGET_HOVER_ALT
        target_x, target_y = home_xy
        if abs(z - target_alt) < 0.25:
            phase = "idle_hover"

    elif phase == "idle_hover":
        target_x, target_y = home_xy
        target_alt = TARGET_HOVER_ALT

    elif phase == "nav_to_init":
        fx, fy = FARMS[selected_farm]["init"]
        target_x, target_y = fx, fy
        target_alt = TARGET_HOVER_ALT
        if math.hypot(x-fx, y-fy) < max(WAYPOINT_RADIUS,0.5):
            bbox = FARMS[selected_farm]["bbox"]
            path = serpentine_with_connectors(bbox, ROW_SPACING, (fx,fy), ROW_MARGIN, ORIENT_AXIS)
            path_idx = 0
            phase = "align_row"
            align_timer = 0.0

    elif phase == "align_row":
        if path_idx >= len(path):
            phase = "return_home"
        else:
            target_x, target_y = path[path_idx]
            target_alt = TARGET_HOVER_ALT

    elif phase == "descend_to_survey":
        target_alt = TARGET_SURVEY_ALT
        target_x, target_y = x, y
        if abs(z - target_alt) < 0.15:
            phase = "follow_rows"

    elif phase == "follow_rows":
        if path_idx >= len(path):
            target_alt = TARGET_HOVER_ALT
            phase = "return_home"
        else:
            if avoid_queue:
                target_x, target_y = avoid_queue[0]
                if math.hypot(x-target_x, y-target_y) < WAYPOINT_RADIUS:
                    avoid_queue.pop(0)
                if not avoid_queue and avoid_cooldown <= 0.0:
                    avoid_cooldown = AVOID_COOLDOWN
            else:
                gx, gy = path[path_idx]
                target_x, target_y = gx, gy
                if math.hypot(x-gx, y-gy) < WAYPOINT_RADIUS:
                    path_idx += 1
                    turn_pause_timer = TURN_PAUSE_S

    elif phase == "return_home":
        target_x, target_y = home_xy
        target_alt = TARGET_HOVER_ALT
        if math.hypot(x-home_xy[0], y-home_xy[1]) < 0.7:
            phase = "nav_to_land"

    elif phase == "nav_to_land":
        target_x, target_y = land_xy
        if math.hypot(x-land_xy[0], y-land_xy[1]) < 0.7:
            phase = "land"; target_alt = 0.0

    elif phase == "land":
        target_x, target_y = x, y
        target_alt = 0.0

    desired_yaw = math.atan2((target_y - y), (target_x - x))
    if yaw_target is None:
        yaw_target = desired_yaw
    err_to_target = ang_wrap(desired_yaw - yaw_target)
    max_step = YAW_SLEW * dt
    if   err_to_target >  max_step: yaw_target += max_step
    elif err_to_target < -max_step: yaw_target -= max_step
    else:                           yaw_target = desired_yaw
    yaw_target = ang_wrap(yaw_target)

    yaw_err   = ang_wrap(yaw_target - yaw)
    yaw_input = clamp(YAW_P * yaw_err - YAW_D * yaw_rate, -YAW_MAX, YAW_MAX)
    heading_ok = abs(yaw_err) < YAW_ALIGN_TOL

    if phase == "align_row":
        if heading_ok:
            align_timer += dt
            if align_timer >= 0.4:
                phase = "descend_to_survey"
        else:
            align_timer = 0.0

    if turn_pause_timer > 0.0:
        turn_pause_timer = max(0.0, turn_pause_timer - dt)
    if avoid_cooldown > 0.0:
        avoid_cooldown = max(0.0, avoid_cooldown - dt)

    if phase in ("follow_rows","nav_to_init") and z>=ALT_HOLD_MIN:
        # plug in simple front clearance if you add sensors here later
        pass

    if cmd_x is None or cmd_y is None:
        cmd_x, cmd_y = x, y

    can_translate = (z >= ALT_HOLD_MIN) and (turn_pause_timer <= 0.0) and (heading_ok or phase in ("takeoff","idle_hover","nav_to_init","nav_to_land","land","descend_to_survey"))
    goal_x, goal_y = (target_x, target_y) if can_translate else (x, y)

    dist = math.hypot(goal_x - x, goal_y - y)
    max_speed = NAV_MAX_SPEED if dist > 1.0 else max(NAV_MIN_SPEED, NAV_MAX_SPEED*dist)
    if not heading_ok:
        max_speed *= 0.5

    step = max_speed * dt
    if dist <= step or not can_translate:
        cmd_x, cmd_y = goal_x, goal_y
    else:
        s = step / dist
        cmd_x = x + (goal_x - x) * s
        cmd_y = y + (goal_y - y) * s

    clamped_alt_diff = clamp(target_alt - z + k_vertical_offset, -1.0, 1.0)
    vertical_input   = k_vertical_p * (clamped_alt_diff ** 3)
    base = k_vertical_thrust + vertical_input

    px_err = clamp(-(cmd_x - x), -LEAN_CAP_CRUISE, LEAN_CAP_CRUISE)
    py_err = clamp( (cmd_y - y), -LEAN_CAP_CRUISE, LEAN_CAP_CRUISE)

    roll_input  = k_roll_p  * clamp(roll,  -1.0, 1.0) + gyro.getValues()[0] + py_err
    pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + gyro.getValues()[1] + px_err

    if not heading_ok or turn_pause_timer > 0.0:
        roll_input  = clamp(roll_input,  -20.0, 20.0)
        pitch_input = clamp(pitch_input, -20.0, 20.0)

    fl = base - roll_input + pitch_input - yaw_input
    fr = base + roll_input + pitch_input + yaw_input
    rl = base - roll_input - pitch_input + yaw_input
    rr = base + roll_input - pitch_input - yaw_input

    if phase == "land" and z < 1.0:
        ramp = max(0.0, z / 1.0)
        fl *= ramp; fr *= ramp; rl *= ramp; rr *= ramp

    if mFL: mFL.setVelocity(fl)
    if mFR: mFR.setVelocity(-fr)
    if mRL: mRL.setVelocity(-rl)
    if mRR: mRR.setVelocity(rr)

    print_t += dt
    if print_t > 0.6:
        print_t = 0
        print(f"t={t:5.1f} phase={phase:16s} z {z:4.2f}->{target_alt:4.2f} pos ({x:5.2f},{y:5.2f})→({target_x:5.2f},{target_y:5.2f}) avoidQ={len(avoid_queue)}")

    if phase == "land" and z < 0.15:
        break
