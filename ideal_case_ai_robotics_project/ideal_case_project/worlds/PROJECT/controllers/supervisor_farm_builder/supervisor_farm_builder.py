# supervisor_farm_builder.py
from controller import Supervisor
import json, random, math

sup = Supervisor()
timestep = int(sup.getBasicTimeStep())

random.seed(42)

SIZES = {1:10.0, 2:10.0, 3:20.0, 4:20.0}
CENTERS = {
    1:(-12.0, -12.0),
    2:(12.0, -12.0),
    3:(-12.0, 12.0),
    4:(12.0, 12.0)
}
INIT_OFFSET = (0.0, 0.0)
HOME = (-20.0, 0.0)
LAND = (20.0, 0.0)

green_mat = 'PBRAppearance { baseColor 0.2 0.6 0.2 metalness 0.0 roughness 1.0 }'
yellow_mat = 'PBRAppearance { baseColor 0.95 0.85 0.2 metalness 0.0 roughness 1.0 }'
red_mat = 'PBRAppearance { baseColor 0.9 0.1 0.1 }'
blue_mat = 'PBRAppearance { baseColor 0.2 0.4 0.9 }'
grey_mat = 'PBRAppearance { baseColor 0.6 0.6 0.6 }'

def box(name, cx, cy, sx, sy, h=0.05, mat=grey_mat):
    return f'DEF {name} Transform {{ translation {cx} 0 {cy} children [ Shape {{ appearance {mat} geometry Box {{ size {sx} {h} {sy} }} }} ] }}'

def cyl(name, cx, cy, r, h, mat):
    return f'DEF {name} Transform {{ translation {cx} 0 {cy} children [ Shape {{ appearance {mat} geometry Cylinder {{ radius {r} height {h} subdivision 24 }} }} ] }}'

def marker(name, cx, cy, color):
    mat = blue_mat if color=="blue" else red_mat
    return cyl(name, cx, cy, 0.25, 0.02, mat)

root = sup.getRoot()
children = root.getField('children')

for k in range(1,5):
    cx, cy = CENTERS[k]
    sz = SIZES[k]
    children.importMFNodeFromString(-1, box(f'FARM{k}_BBOX', cx, cy, sz, sz, 0.02, grey_mat))

children.importMFNodeFromString(-1, marker('COMMON_HOME', HOME[0], HOME[1], 'blue'))
children.importMFNodeFromString(-1, marker('COMMON_LAND', LAND[0], LAND[1], 'red'))

layout = {"farms":{}, "home":{"x":HOME[0],"y":HOME[1]}, "land":{"x":LAND[0],"y":LAND[1]}}

for k in range(1,5):
    cx, cy = CENTERS[k]
    sz = SIZES[k]
    ix = cx + INIT_OFFSET[0]
    iy = cy + INIT_OFFSET[1]
    children.importMFNodeFromString(-1, marker(f'FARM{k}_INIT', ix, iy, 'blue'))
    layout["farms"][str(k)] = {
        "bbox":{"min":[cx - sz/2, cy - sz/2], "max":[cx + sz/2, cy + sz/2]},
        "init":[ix, iy],
        "size":sz
    }

def place_plant(pid, px, py, pest):
    r = 0.12 if not pest else 0.14
    h = 0.5 if not pest else 0.6
    mat = green_mat if not pest else yellow_mat
    children.importMFNodeFromString(-1, cyl(f'PLANT_{pid}', px, py, r, 0.02, mat))

gt = {"plants":[]}

pid = 0
for k in range(1,5):
    cx, cy = CENTERS[k]
    sz = SIZES[k]
    step = 1.0 if sz==10.0 else 1.25
    xmin, zmin = cx - sz/2 + 1.0, cy - sz/2 + 1.0
    xmax, zmax = cx + sz/2 - 1.0, cy + sz/2 - 1.0
    pts = []
    yrows = int(math.floor((zmax - zmin)/step)) + 1
    xcols = int(math.floor((xmax - xmin)/step)) + 1
    for i in range(yrows):
        for j in range(xcols):
            px = xmin + j*step + (0.15* (random.random()-0.5))
            py = zmin + i*step + (0.15* (random.random()-0.5))
            pts.append((px,py))
    pest_prob = 0.12 if sz==10.0 else 0.18
    for (px,py) in pts:
        is_pest = 1 if random.random() < pest_prob else 0
        place_plant(pid, px, py, is_pest==1)
        gt["plants"].append({"farm":k,"x":px,"y":py,"pest":is_pest})
        pid += 1

with open('/tmp/farm_layout.json','w') as f:
    json.dump(layout,f)
with open('/tmp/farm_groundtruth.json','w') as f:
    json.dump(gt,f)

while sup.step(timestep) != -1:
    pass
