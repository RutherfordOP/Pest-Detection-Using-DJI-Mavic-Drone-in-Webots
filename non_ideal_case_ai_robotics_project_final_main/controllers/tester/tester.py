# drone_controller.py
import math, heapq, json, os, tempfile
import numpy as np, cv2
from controller import Robot, Motor, GPS, Gyro, InertialUnit, Camera, Keyboard, Lidar, RangeFinder

robot = Robot()
timestep = int(robot.getBasicTimeStep())
dt = max(1e-3, timestep / 1000.0)
kb = Keyboard(); kb.enable(timestep)
imu  = robot.getDevice("inertial unit"); imu.enable(timestep)
gps  = robot.getDevice("gps");           gps.enable(timestep)
gyro = robot.getDevice("gyro");          gyro.enable(timestep)

def clamp(v, a, b): return a if v<a else b if v>b else v
def ang_wrap(a): return (a + math.pi) % (2*math.pi) - math.pi

camera = None; W=H=0
try:
    camera = robot.getDevice("camera"); camera.enable(timestep)
    W, H = camera.getWidth(), camera.getHeight()
except: pass

front_depth=None; DEPTH_W=DEPTH_H=0; DEPTH_FOV=1.5708
try:
    dev = robot.getDevice("front_depth")
    if isinstance(dev, RangeFinder):
        front_depth=dev; front_depth.enable(timestep)
        DEPTH_W=front_depth.getWidth(); DEPTH_H=front_depth.getHeight(); DEPTH_FOV=float(front_depth.getFov())
except: pass

lidar=None
try:
    ld = robot.getDevice("lidar_front")
    if isinstance(ld, Lidar):
        lidar=ld; lidar.enable(timestep)
except: pass

def get_motor(n):
    try:
        m = robot.getDevice(n)
        if isinstance(m, Motor):
            m.setPosition(float('inf')); m.setVelocity(0.0); return m
    except: pass
    return None

mFL = get_motor("front left propeller")
mFR = get_motor("front right propeller")
mRL = get_motor("rear left propeller")
mRR = get_motor("rear right propeller")

for gn in ("gimbal pitch","gimbal_pitch","camera pitch","camera_pitch","tilt","camera tilt"):
    try:
        gm = robot.getDevice(gn)
        if isinstance(gm, Motor):
            gm.setPosition(-1.22); gm.setVelocity(1.0); break
    except: pass

k_vertical_thrust = 68.5
k_vertical_offset = 0.6
k_vertical_p      = 3.0
k_roll_p          = 50.0
k_pitch_p         = 30.0

YAW_P=3.0; YAW_D=0.9; YAW_SLEW=0.8; YAW_ALIGN_TOL=0.15

NAV_MAX_SPEED   = 1.0
NAV_MIN_SPEED   = 0.25
WAYPOINT_RADIUS = 0.6
ALT_HOLD_MIN    = 0.8
TARGET_HOVER_ALT  = 2.5
TARGET_SURVEY_ALT = 1.5

OBST_TRIP_FRONT = 1.0
STOP_REPLAN_PAUSE = 1.0

EVAL_DIR = r'D:\my_project2\worlds\evaluation\outputs'
try: os.makedirs(EVAL_DIR, exist_ok=True)
except: pass

TMP = tempfile.gettempdir()
LAYOUT_PATH = os.path.join(TMP, 'farm_layout.json')
while not os.path.exists(LAYOUT_PATH):
    if robot.step(timestep) == -1: pass

with open(LAYOUT_PATH,'r') as f: L = json.load(f)

FARMS = {}
for k, info in L["farms"].items():
    FARMS[int(k)] = {"bbox": (tuple(info["bbox"]["min"]), tuple(info["bbox"]["max"])), "init": tuple(info["init"]), "size": info["size"]}
LAND_PT = (L["land"]["x"], L["land"]["y"]) if "land" in L else None

def heuristic_manh(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

def gen_grid_and_lawn(bbox, cell, row_step_cells=2):
    (xmin,ymin),(xmax,ymax)=bbox
    Wg=max(1,int((xmax-xmin)/cell)); Hg=max(1,int((ymax-ymin)/cell))
    grid=[[0 for _ in range(Wg)] for _ in range(Hg)]
    lawn=[]
    for r in range(0,Hg,row_step_cells):
        row=[(c,r) for c in range(0,Wg)] if (r//row_step_cells)%2==0 else [(c,r) for c in range(Wg-1,-1,-1)]
        lawn.extend(row)
    return grid,lawn,Wg,Hg

def cell_to_world(i,j,bbox,cell):
    (xmin,ymin),_=bbox
    return xmin + (i+0.5)*cell, ymin + (j+0.5)*cell

def world_to_cell(x,y,bbox,cell):
    (xmin,ymin),(xmax,ymax)=bbox
    return int((x - xmin)/cell), int((y - ymin)/cell)

def in_bbox(x,y,bbox):
    (xmin,ymin),(xmax,ymax)=bbox
    return xmin<=x<=xmax and ymin<=y<=ymax

def inflate_cells(cells,Wg,Hg,r=1):
    out=set()
    for i,j in cells:
        for di in range(-r,r+1):
            for dj in range(-r,r+1):
                ii=i+di; jj=j+dj
                if 0<=ii<Wg and 0<=jj<Hg: out.add((ii,jj))
    return out

def detect_pests(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([18, 80, 120]); upper = np.array([35,255,255])
    mask  = cv2.inRange(hsv, lower, upper)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
    cnts,_= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out=[]
    for c in cnts:
        a=cv2.contourArea(c)
        if a>30:
            x,y,w,h = cv2.boundingRect(c); out.append((x+w/2,y+h/2,a))
    return out

def project_to_ground(px,py,alt,fov_deg=60.0):
    if W==0 or H==0: return 0.0,0.0
    gw = 2.0*alt*math.tan(math.radians(fov_deg/2.0))
    sx=(px/W - 0.5)*gw; sy=(py/H - 0.5)*gw
    return sx,sy

def lidar_center_min():
    if not lidar: return float('inf')
    arr = lidar.getRangeImage()
    if not arr: return float('inf')
    n=len(arr); c=n//2; span=max(3,n//20)
    vals=[arr[i] for i in range(max(0,c-span), min(n,c+span+1)) if arr[i]==arr[i] and arr[i]!=float('inf')]
    return min(vals) if vals else float('inf')

def depth_center_min():
    if not front_depth or DEPTH_W==0 or DEPTH_H==0: return float('inf')
    img = front_depth.getRangeImage()
    if not img: return float('inf')
    row=DEPTH_H//2; base=row*DEPTH_W; c=DEPTH_W//2; span=max(2,DEPTH_W//20)
    vals=[img[base+u] for u in range(max(0,c-span), min(DEPTH_W,c+span+1)) if img[base+u]==img[base+u] and img[base+u]!=float('inf')]
    return min(vals) if vals else float('inf')

class PriorityQueue:
    def __init__(self): self.h=[]; self.c=0
    def push(self,k,s): heapq.heappush(self.h,(k[0],k[1],self.c,s)); self.c+=1
    def pop(self): return heapq.heappop(self.h)[3]
    def top_key(self): return (float('inf'),float('inf')) if not self.h else (self.h[0][0],self.h[0][1])
    def remove(self,s):
        for i,(_,_,_,x) in enumerate(self.h):
            if x==s: self.h[i]=self.h[-1]; self.h.pop(); heapq.heapify(self.h); return
    def empty(self): return not self.h

class DStarLitePlanner:
    def __init__(self,W,H,grid):
        self.W=W; self.H=H; self.grid=grid
        self.g={}; self.rhs={}; self.U=PriorityQueue()
        self.s_start=None; self.s_goal=None; self.s_last=None; self.km=0.0
    def h(self,a,b): return heuristic_manh(a,b)
    def key(self,s):
        v=min(self.g.get(s,float('inf')), self.rhs.get(s,float('inf')))
        return (v + self.h(self.s_start,s) + self.km, v)
    def nbrs(self,s):
        out=[]; x,y=s
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny=x+dx,y+dy
            if 0<=nx<self.W and 0<=ny<self.H and self.grid[ny][nx]==0: out.append((nx,ny))
        return out
    def initialize(self,s_start,s_goal):
        self.U=PriorityQueue(); self.g.clear(); self.rhs.clear()
        self.s_start=s_start; self.s_goal=s_goal; self.s_last=s_start
        self.rhs[s_goal]=0.0; self.U.push(self.key(s_goal), s_goal)
    def update_vertex(self,u):
        if u!=self.s_goal:
            m=float('inf')
            for s in self.nbrs(u):
                gs=self.g.get(s,float('inf'))
                if gs+1.0<m: m=gs+1.0
            self.rhs[u]=m
        self.U.remove(u)
        if self.g.get(u,float('inf'))!=self.rhs.get(u,float('inf')):
            self.U.push(self.key(u),u)
    def compute_shortest_path(self):
        while (self.U.top_key() < self.key(self.s_start)) or (self.g.get(self.s_start,float('inf'))!=self.rhs.get(self.s_start,float('inf'))):
            u=self.U.pop()
            g_u=self.g.get(u,float('inf')); rhs_u=self.rhs.get(u,float('inf'))
            if g_u>rhs_u:
                self.g[u]=rhs_u
                for p in self.nbrs(u): self.update_vertex(p)
            else:
                self.g[u]=float('inf'); self.update_vertex(u)
                for p in self.nbrs(u): self.update_vertex(p)
    def changed_cell(self,ij):
        self.update_vertex(ij)
        for p in self.nbrs(ij): self.update_vertex(p)
    def move_start(self,new_s): self.km += self.h(self.s_last,new_s); self.s_last=new_s
    def plan(self,s_start,s_goal):
        if (self.s_start is None) or (self.s_goal is None) or (s_goal!=self.s_goal):
            self.initialize(s_start,s_goal)
        else:
            self.move_start(s_start)
        self.compute_shortest_path()
        path=[s_start]; cur=s_start
        for _ in range(self.W*self.H):
            if cur==s_goal: break
            best=None; bestc=float('inf')
            for v in self.nbrs(cur):
                c=1.0 + self.g.get(v,float('inf'))
                if c<bestc: bestc=c; best=v
            if best is None or best==cur: break
            cur=best; path.append(cur)
        return path if path and path[-1]==s_goal else []

class LPAStarPlanner:
    def __init__(self,W,H,grid):
        self.W=W; self.H=H; self.grid=grid
        self.g={}; self.rhs={}; self.U=PriorityQueue(); self.s_start=None; self.s_goal=None
    def h(self,a,b): return heuristic_manh(a,b)
    def key(self,s):
        v=min(self.g.get(s,float('inf')), self.rhs.get(s,float('inf')))
        return (v + self.h(s,self.s_goal), v)
    def nbrs(self,s):
        out=[]; x,y=s
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny=x+dx,y+dy
            if 0<=nx<self.W and 0<=ny<self.H and self.grid[ny][nx]==0: out.append((nx,ny))
        return out
    def initialize(self,s_start,s_goal):
        self.U=PriorityQueue(); self.g.clear(); self.rhs.clear()
        self.s_start=s_start; self.s_goal=s_goal; self.rhs[s_start]=0.0
        self.U.push(self.key(s_start), s_start)
    def update_vertex(self,u):
        if u!=self.s_start:
            m=float('inf')
            for p in self.nbrs(u):
                gp=self.g.get(p,float('inf'))
                if gp+1.0<m: m=gp+1.0
            self.rhs[u]=m
        self.U.remove(u)
        if self.g.get(u,float('inf'))!=self.rhs.get(u,float('inf')):
            self.U.push(self.key(u),u)
    def compute_shortest_path(self):
        while (self.U.top_key() < self.key(self.s_goal)) or (self.g.get(self.s_goal,float('inf'))!=self.rhs.get(self.s_goal,float('inf'))):
            u=self.U.pop()
            g_u=self.g.get(u,float('inf')); rhs_u=self.rhs.get(u,float('inf'))
            if g_u>rhs_u:
                self.g[u]=rhs_u
                for s in self.nbrs(u): self.update_vertex(s)
            else:
                self.g[u]=float('inf'); self.update_vertex(u)
                for s in self.nbrs(u): self.update_vertex(s)
    def changed_cell(self,ij):
        self.update_vertex(ij)
        for p in self.nbrs(ij): self.update_vertex(p)
    def plan(self,s_start,s_goal):
        if (self.s_start is None) or (self.s_goal is None) or (s_start!=self.s_start) or (s_goal!=self.s_goal):
            self.initialize(s_start,s_goal)
        self.compute_shortest_path()
        path=[s_start]; cur=s_start
        for _ in range(self.W*self.H):
            if cur==s_goal: break
            best=None; bestc=float('inf')
            for v in self.nbrs(cur):
                c=1.0 + self.g.get(v,float('inf'))
                if c<bestc: bestc=c; best=v
            if best is None: break
            cur=best; path.append(cur)
        return path if path and path[-1]==s_goal else []

def los_free(grid,a,b):
    x0,y0=a; x1,y1=b
    H=len(grid); W=len(grid[0])
    dx=abs(x1-x0); dy=abs(y1-y0); sx=1 if x1>=x0 else -1; sy=1 if y1>=y0 else -1
    err=dx-dy; x,y=x0,y0
    while True:
        if not (0<=x<W and 0<=y<H) or grid[y][x]==1: return False
        if x==x1 and y==y1: break
        e2=2*err
        if e2>-dy: err-=dy; x+=sx
        if e2< dx: err+=dx; y+=sy
    return True

class FieldDStarPlanner:
    def __init__(self,W,H,grid):
        self.base=DStarLitePlanner(W,H,grid); self.grid=grid
    def plan(self,s_start,s_goal):
        cells=self.base.plan(s_start,s_goal)
        if not cells: return []
        pts=[cells[0]]
        for k in range(2,len(cells)):
            if not los_free(self.grid, pts[-1], cells[k]): pts.append(cells[k-1])
        pts.append(cells[-1]); return pts

def make_planner(mode,W,H,grid):
    m=mode.upper()
    if m=='DSTARLITE': return DStarLitePlanner(W,H,grid),'D* Lite'
    if m=='LPASTAR': return LPAStarPlanner(W,H,grid),'LPA*'
    if m=='FIELDDSTAR': return FieldDStarPlanner(W,H,grid),'Field D*'
    return DStarLitePlanner(W,H,grid),'D* Lite'

PLANNER_MODE = os.environ.get('PLANNER','DSTARLITE').upper()

farm_runtime={}
for k,finfo in FARMS.items():
    cell = 0.5 if finfo["size"]==10.0 else 1.0
    grid,lawn,Wg,Hg = gen_grid_and_lawn(finfo["bbox"], cell, row_step_cells=2)
    planner,pname = make_planner(PLANNER_MODE,Wg,Hg,grid)
    farm_runtime[k]={"cell":cell,"grid":grid,"W":Wg,"H":Hg,
                     "path_world":[cell_to_world(i,j,finfo["bbox"],cell) for (i,j) in lawn],
                     "planner":planner,"planner_name":pname,"changed":set()}

phase="takeoff"
selected_farm=None
target_alt=TARGET_HOVER_ALT
target_x=0.0; target_y=0.0
home_xy=None; land_xy=None
path_idx=0
replan_timer=0.0
detections=[]; unique_cells=set(); pest_records=[]
ACTIVE_SEGMENT=[]; ACTIVE_INDEX=0
cmd_x=None; cmd_y=None
yaw_target=None
obstacle_pause_t=0.0; pending_replan=False

metrics={"start_time":None,"end_time":None,"total_distance_xy":0.0,"avg_speed_xy":0.0,
         "min_alt":float('inf'),"max_alt":0.0,"max_abs_roll_deg":0.0,"max_abs_pitch_deg":0.0,
         "phase_times":{p:0.0 for p in ["takeoff","idle_hover","nav_to_init","descend_to_survey","follow_path","return_home","nav_to_land","land"]},
         "replan_count":0,"planner_fail_count":0,"alt_freeze_count":0,"unique_pest_cells":0,
         "events":[],"planner_name":(farm_runtime[list(farm_runtime.keys())[0]]["planner_name"] if farm_runtime else "N/A"),
         "obstacle_stops":0,"obstacle_stop_time":0.0}
last_x=last_y=None; last_phase="takeoff"

def note(t,msg): metrics["events"].append({"t":round(t,2),"msg":msg})

def next_goal(run, path, path_idx, x, y):
    global ACTIVE_SEGMENT, ACTIVE_INDEX
    if ACTIVE_SEGMENT and ACTIVE_INDEX < len(ACTIVE_SEGMENT): return ACTIVE_SEGMENT[ACTIVE_INDEX]
    if path_idx < len(path): return path[path_idx]
    return None

def start_local_segment(seg_world):
    global ACTIVE_SEGMENT, ACTIVE_INDEX
    ACTIVE_SEGMENT=seg_world[:]; ACTIVE_INDEX=0

def advance_if_reached(goal_xy, x, y, arrive_r):
    global ACTIVE_SEGMENT, ACTIVE_INDEX
    if goal_xy is None: return False
    if math.hypot(x-goal_xy[0], y-goal_xy[1]) <= arrive_r:
        if ACTIVE_SEGMENT:
            ACTIVE_INDEX += 1
            if ACTIVE_INDEX >= len(ACTIVE_SEGMENT): ACTIVE_SEGMENT=[]; ACTIVE_INDEX=0
        else:
            return True
    return False

def add_pest_cell(ci,cj,run):
    if run["grid"][cj][ci]==0:
        run["grid"][cj][ci]=1; run["changed"].add((ci,cj)); return True
    return False

def mark_front_obstacle_into_grid(x,y,yaw,dist,finfo,run):
    if dist<=0.0: return False
    wx=x + dist*math.cos(yaw); wy=y + dist*math.sin(yaw)
    bbox=finfo["bbox"]; cell=run["cell"]
    if not in_bbox(wx,wy,bbox): return False
    ci,cj = world_to_cell(wx,wy,bbox,cell)
    ci = clamp(ci,0,run["W"]-1); cj=clamp(cj,0,run["H"]-1)
    changed=False
    for (ii,jj) in inflate_cells({(ci,cj)},run["W"],run["H"],r=1):
        if run["grid"][jj][ii]==0:
            run["grid"][jj][ii]=1; run["changed"].add((ii,jj)); changed=True
    return changed

print("Boot: low hover; press 1/2/3/4 to choose farm.")

while robot.step(timestep) != -1:
    t = robot.getTime()
    if metrics["start_time"] is None: metrics["start_time"]=t

    rpy = imu.getRollPitchYaw()
    roll,pitch,yaw = rpy[0], rpy[1], rpy[2]
    gvals = gyro.getValues()
    roll_vel = gvals[0] if len(gvals)>0 else 0.0
    pitch_vel= gvals[1] if len(gvals)>1 else 0.0
    yaw_rate = gvals[2] if len(gvals)>2 else 0.0
    x,y,z = gps.getValues()

    metrics["min_alt"]=min(metrics["min_alt"],z); metrics["max_alt"]=max(metrics["max_alt"],z)
    metrics["max_abs_roll_deg"]=max(metrics["max_abs_roll_deg"],abs(math.degrees(roll)))
    metrics["max_abs_pitch_deg"]=max(metrics["max_abs_pitch_deg"],abs(math.degrees(pitch)))
    if last_x is not None: metrics["total_distance_xy"] += math.hypot(x-last_x,y-last_y)
    last_x,last_y=x,y
    if last_phase!=phase: note(t,f"phase→{phase}"); last_phase=phase
    if phase in metrics["phase_times"]: metrics["phase_times"][phase]+=dt

    if home_xy is None:
        home_xy=(x,y); land_xy = LAND_PT if LAND_PT else (home_xy[0]+5.0, home_xy[1])
        target_x,target_y=home_xy; cmd_x,cmd_y=home_xy

    key = kb.getKey()
    if phase=="idle_hover" and key in [ord('1'),ord('2'),ord('3'),ord('4')]:
        selected_farm=int(chr(key)); phase="nav_to_init"; path_idx=0
        ACTIVE_SEGMENT=[]; ACTIVE_INDEX=0; obstacle_pause_t=0.0; pending_replan=False
        note(t,f"farm_selected {selected_farm}")

    if phase=="takeoff":
        target_alt=TARGET_HOVER_ALT; target_x,target_y=home_xy
        if abs(z-target_alt)<0.25: phase="idle_hover"
    elif phase=="idle_hover":
        target_x,target_y=home_xy; target_alt=TARGET_HOVER_ALT
    elif phase=="nav_to_init":
        finfo=FARMS[selected_farm]; ix,iy=finfo["init"]; target_alt=TARGET_HOVER_ALT
        target_x,target_y=ix,iy
        if math.hypot(x-ix,y-iy) < max(WAYPOINT_RADIUS,0.6): phase="descend_to_survey"
    elif phase=="descend_to_survey":
        target_alt=TARGET_SURVEY_ALT; target_x,target_y=x,y
        if abs(z-target_alt)<0.2: path_idx=0; phase="follow_path"
    elif phase=="follow_path":
        finfo=FARMS[selected_farm]; run=farm_runtime[selected_farm]
        path=run["path_world"]; cell=run["cell"]; bbox=finfo["bbox"]
        if z<ALT_HOLD_MIN:
            target_x,target_y=x,y; metrics["alt_freeze_count"]+=1
        else:
            goal = next_goal(run,path,path_idx,x,y)
            if goal is None:
                target_alt=TARGET_HOVER_ALT; phase="return_home"
            else:
                target_x,target_y=goal
                if advance_if_reached(goal,x,y,max(WAYPOINT_RADIUS,0.8*cell)): path_idx+=1

        new_obs=False
        if camera and W>0 and H>0 and (int(t*5)%5==0):
            img_b=camera.getImage()
            if img_b:
                img=np.frombuffer(img_b,np.uint8).reshape((H,W,4))[:,:,:3]
                img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                for (px,py,area) in detect_pests(img):
                    sx,sy=project_to_ground(px,py,z,60.0)
                    wx=x + math.cos(yaw)*sx - math.sin(yaw)*sy
                    wy=y + math.sin(yaw)*sx + math.cos(yaw)*sy
                    if in_bbox(wx,wy,bbox):
                        ci,cj=world_to_cell(wx,wy,bbox,cell)
                        ci=clamp(ci,0,run["W"]-1); cj=clamp(cj,0,run["H"]-1)
                        if add_pest_cell(ci,cj,run): new_obs=True
                        detections.append((t,selected_farm,wx,wy,ci,cj))
                        key_uc=(selected_farm,ci,cj)
                        if key_uc not in unique_cells:
                            unique_cells.add(key_uc)
                            pest_records.append({"time":round(t,2),"farm":selected_farm,"x":round(wx,3),"y":round(wy,3),"cell_i":ci,"cell_j":cj})

        front=min(lidar_center_min(), depth_center_min())
        if front<OBST_TRIP_FRONT and z>=ALT_HOLD_MIN:
            if mark_front_obstacle_into_grid(x,y,yaw,front,finfo,run): new_obs=True

        if new_obs and obstacle_pause_t<=0.0:
            obstacle_pause_t=STOP_REPLAN_PAUSE; pending_replan=True
            ACTIVE_SEGMENT=[]; ACTIVE_INDEX=0
            metrics["obstacle_stops"]+=1; note(t,"obstacle_detected_pause")

        if obstacle_pause_t>0.0:
            obstacle_pause_t=max(0.0, obstacle_pause_t-dt)
            metrics["obstacle_stop_time"]+=dt
            target_x,target_y=x,y
            if obstacle_pause_t<=0.0 and pending_replan:
                s_start = (clamp(world_to_cell(x,y,bbox,cell)[0],0,run["W"]-1), clamp(world_to_cell(x,y,bbox,cell)[1],0,run["H"]-1))
                if path_idx<len(path): gx,gy=path[path_idx]
                else: gx,gy=home_xy
                s_goal  = (clamp(world_to_cell(gx,gy,bbox,cell)[0],0,run["W"]-1), clamp(world_to_cell(gy,gy,bbox,cell)[1],0,run["H"]-1))
                for ij in list(run["changed"]): run["planner"].changed_cell(ij)
                run["changed"].clear()
                seg_cells = run["planner"].plan(s_start, s_goal)
                if seg_cells and len(seg_cells)>1:
                    repl=[cell_to_world(i,j,bbox,cell) for (i,j) in seg_cells]
                    start_local_segment(repl); metrics["replan_count"]+=1; note(t,f"replan[{run['planner_name']}] len={len(repl)}")
                else:
                    metrics["planner_fail_count"]+=1; note(t,f"replan_failed[{run['planner_name']}]")
                pending_replan=False

        replan_timer+=dt
        if replan_timer>2.0 and z>=ALT_HOLD_MIN and not ACTIVE_SEGMENT and path_idx<len(path):
            replan_timer=0.0
            s_start=(clamp(world_to_cell(x,y,bbox,cell)[0],0,run["W"]-1), clamp(world_to_cell(x,y,bbox,cell)[1],0,run["H"]-1))
            gx,gy=path[path_idx]
            s_goal=(clamp(world_to_cell(gx,gy,bbox,cell)[0],0,run["W"]-1), clamp(world_to_cell(gx,gy,bbox,cell)[1],0,run["H"]-1))
            for ij in list(run["changed"]): run["planner"].changed_cell(ij)
            run["changed"].clear()
            seg_cells=run["planner"].plan(s_start,s_goal)
            if seg_cells and len(seg_cells)>1:
                repl=[cell_to_world(i,j,bbox,cell) for (i,j) in seg_cells]
                start_local_segment(repl); metrics["replan_count"]+=1; note(t,f"replan[{run['planner_name']}] len={len(repl)}")
            else:
                metrics["planner_fail_count"]+=1; note(t,f"replan_failed[{run['planner_name']}]")
    elif phase=="return_home":
        target_x,target_y=home_xy; target_alt=TARGET_HOVER_ALT
        if math.hypot(x-home_xy[0], y-home_xy[1])<0.7: phase="nav_to_land"
    elif phase=="nav_to_land":
        target_x,target_y=land_xy
        if math.hypot(x-land_xy[0], y-land_xy[1])<0.7: phase="land"; target_alt=0.0
    elif phase=="land":
        target_x,target_y=x,y

    if cmd_x is None or cmd_y is None: cmd_x,cmd_y=x,y

    desired_yaw = math.atan2((target_y - y), (target_x - x))
    if yaw_target is None: yaw_target=desired_yaw
    err_to_target = ang_wrap(desired_yaw - yaw_target)
    max_step = YAW_SLEW*dt
    if   err_to_target >  max_step: yaw_target += max_step
    elif err_to_target < -max_step: yaw_target -= max_step
    else:                           yaw_target = desired_yaw
    yaw_target = ang_wrap(yaw_target)
    yaw_err = ang_wrap(yaw_target - yaw)
    yaw_input = clamp(YAW_P*yaw_err - YAW_D*yaw_rate, -1.0, 1.0)
    heading_ok = abs(yaw_err) < YAW_ALIGN_TOL

    dist_to_goal = math.hypot(target_x - x, target_y - y)
    max_speed = NAV_MAX_SPEED if dist_to_goal>1.0 else max(NAV_MIN_SPEED, NAV_MAX_SPEED*dist_to_goal)
    if not heading_ok: max_speed *= 0.6
    if obstacle_pause_t>0.0: max_speed = 0.0

    step = max_speed*dt
    if dist_to_goal<=step or max_speed==0.0:
        cmd_x,cmd_y = target_x,target_y
    else:
        s = step/dist_to_goal
        cmd_x = x + (target_x - x)*s
        cmd_y = y + (target_y - y)*s

    clamped_alt_diff = clamp(target_alt - z + k_vertical_offset, -1.0, 1.0)
    vertical_input   = k_vertical_p * (clamped_alt_diff ** 3)

    roll_input  = k_roll_p  * clamp(roll,  -1.0, 1.0) + roll_vel  + clamp( 1.0 * (cmd_y - y), -0.35, 0.35)
    pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + pitch_vel + clamp(-1.0 * (cmd_x - x), -0.35, 0.35)

    base = k_vertical_thrust + vertical_input
    fl = base - roll_input + pitch_input - yaw_input
    fr = base + roll_input + pitch_input + yaw_input
    rl = base - roll_input - pitch_input + yaw_input
    rr = base + roll_input - pitch_input - yaw_input
    if phase=="land" and z<1.0:
        ramp=max(0.0, z/1.0); fl*=ramp; fr*=ramp; rl*=ramp; rr*=ramp

    if mFL: mFL.setVelocity(fl)
    if mFR: mFR.setVelocity(-fr)
    if mRL: mRL.setVelocity(-rl)
    if mRR: mRR.setVelocity(rr)

    if int(t*2)%4==0:
        print(f"phase={phase} | yaw_err={yaw_err:+.2f} | z={z:.2f}/{target_alt:.2f} | pos=({x:.1f},{y:.1f})→({target_x:.1f},{target_y:.1f}) | pause={obstacle_pause_t:.1f}")

    if phase=="land" and z<0.15:
        metrics["end_time"]=t
        elapsed=max(1e-6, metrics["end_time"]-metrics["start_time"])
        metrics["avg_speed_xy"]=metrics["total_distance_xy"]/elapsed
        metrics["unique_pest_cells"]=len(unique_cells)
        try:
            with open(os.path.join(EVAL_DIR,'detections.csv'),'w') as f:
                f.write('time,farm,x,y,cell_i,cell_j\n')
                for (tt,ff,wx,wy,ci,cj) in detections:
                    f.write(f'{tt:.2f},{ff},{wx:.3f},{wy:.3f},{ci},{cj}\n')
            with open(os.path.join(EVAL_DIR,'pest_summary.json'),'w') as f:
                json.dump({"total_unique_pests":len(unique_cells),"detections":pest_records}, f, indent=2)
            with open(os.path.join(EVAL_DIR,'pest_summary.txt'),'w') as f:
                f.write(f"Total unique pests detected: {len(unique_cells)}\n")
                for rec in pest_records:
                    f.write(f"Farm {rec['farm']} | t={rec['time']}s | x={rec['x']} y={rec['y']} | cell=({rec['cell_i']},{rec['cell_j']})\n")
            with open(os.path.join(EVAL_DIR,'flight_metrics.json'),'w') as f:
                json.dump(metrics, f, indent=2)
            with open(os.path.join(EVAL_DIR,'flight_metrics.txt'),'w') as f:
                f.write("==== FLIGHT METRICS ====\n")
                f.write(f"Planner: {metrics.get('planner_name','')}\n")
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
                f.write(f"Planner fails: {metrics['planner_fail_count']}\n")
                f.write(f"Alt-freeze events: {metrics['alt_freeze_count']}\n")
                f.write(f"Unique pest cells: {metrics['unique_pest_cells']}\n")
                f.write(f"Obstacle stops: {metrics['obstacle_stops']}\n")
                f.write(f"Obstacle stop time (s): {metrics['obstacle_stop_time']:.2f}\n")
                f.write("Events:\n")
                for e in metrics["events"]:
                    f.write(f"  t={e['t']}: {e['msg']}\n")
        except: pass
        break
