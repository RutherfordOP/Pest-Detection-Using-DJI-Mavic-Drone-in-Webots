# world_setup.py  (place it in the same folder as the .wbt file)
import json, os, tempfile
from controller import Supervisor

sup = Supervisor()
root = sup.getRoot()
children = root.getField('children')

poles = []
for i in range(children.getCount()):
    node = children.getMFNode(i)
    name = node.getTypeName().lower()
    if 'pole' in name or 'cylinder' in name:          # catch any pole you placed
        trans = node.getField('translation')
        if trans:
            x, y, _ = trans.getSFVec3f()
            poles.append({'x': x, 'y': y, 'r': 0.12})   # radius ≈ pole radius

# write to the same temp folder the controller uses
tmp = tempfile.gettempdir()
path = os.path.join(tmp, 'poles.json')
with open(path, 'w') as f:
    json.dump({'poles': poles}, f)
print(f'Wrote {len(poles)} poles to {path}')
