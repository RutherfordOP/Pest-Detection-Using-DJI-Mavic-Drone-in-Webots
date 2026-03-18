# world_setup.py
import json, os, tempfile
from controller import Supervisor

sup = Supervisor()
root = sup.getRoot()
children = root.getField("children")

poles = []
for i in range(children.getCount()):
    node = children.getMFNode(i)
    name = node.getTypeName().lower()
    if "pole" in name or "cylinder" in name:
        trans = node.getField("translation")
        if trans:
            x, y, _ = trans.getSFVec3f()
            radius = 0.12
            rad = node.getField("radius")
            if rad: radius = rad.getSFFloat()
            poles.append({"x": x, "y": y, "r": radius})

tmp = tempfile.gettempdir()
path = os.path.join(tmp, "poles.json")
with open(path, "w") as f:
    json.dump({"poles": poles}, f, indent=2)

print(f"[world_setup] Found {len(poles)} poles → {path}")

# Self-delete after writing
sup.getSelf().remove()
