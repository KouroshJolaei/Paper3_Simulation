"""
inspect_scene.py — Read-only scene inspector for the TSF-85 cylinder scene.

Prints every prim's path, type, world position, and bounding box, with special
attention to the graspable object(s). Used to derive the grid center and the
cylinder's geometry (center, long axis, radius, length) for the grid-scan.

RUN:
  cd ~/Paper3_Simulation/TSF-85/examples
  ~/isaacsim/python.sh ~/Paper3_Simulation/sim/inspect_scene.py
"""

import os

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom

SCENE = os.path.join(
    "/home/kourosh/Paper3_Simulation/TSF-85/examples/scenes/scene_cylinder.usd"
)

stage = Usd.Stage.Open(SCENE)
xc = UsdGeom.XformCache()

print("\n\n========== SCENE INSPECTION START ==========\n")

# Pass 1: anything that looks like the graspable object
print("--- OBJECTS (name contains 'object'/'cylinder'/'cyl') ---")
object_prims = []
for prim in stage.Traverse():
    name = prim.GetName().lower()
    path = str(prim.GetPath()).lower()
    if any(k in name for k in ("object", "cylinder", "cyl")) or \
       any(k in path for k in ("object", "cylinder")):
        object_prims.append(prim)

for prim in object_prims:
    tn = str(prim.GetTypeName())
    m = xc.GetLocalToWorldTransform(prim)
    t = m.ExtractTranslation()
    line = f"{prim.GetPath()} | {tn} | pos {[round(v,4) for v in t]}"
    if prim.IsA(UsdGeom.Cylinder):
        c = UsdGeom.Cylinder(prim)
        line += (f" | RADIUS={c.GetRadiusAttr().Get()} "
                 f"HEIGHT={c.GetHeightAttr().Get()} "
                 f"AXIS={c.GetAxisAttr().Get()}")
    if prim.IsA(UsdGeom.Boundable):
        try:
            bb = UsdGeom.Boundable(prim).ComputeWorldBound(
                0, "default").ComputeAlignedRange()
            mn, mx = bb.GetMin(), bb.GetMax()
            size = [round(mx[i]-mn[i], 4) for i in range(3)]
            line += (f"\n      bbox_min {[round(v,4) for v in mn]}"
                     f"\n      bbox_max {[round(v,4) for v in mx]}"
                     f"\n      size(LxWxH) {size}")
        except Exception as e:
            line += f"  (bbox failed: {e})"
    print(line)
    print()

# Pass 2: the two sensor roots, so we know where the pads are
print("\n--- SENSOR ROOTS (TSF_85_right / TSF_85_left) ---")
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if p.endswith("TSF_85_right/TSF_85") or p.endswith("TSF_85_left/TSF_85"):
        m = xc.GetLocalToWorldTransform(prim)
        t = m.ExtractTranslation()
        print(f"{p} | pos {[round(v,4) for v in t]}")

print("\n========== SCENE INSPECTION END ==========\n")

app.close()
