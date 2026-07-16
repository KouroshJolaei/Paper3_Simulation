"""
probe_pad_normal.py — Find the TRUE pad-normal axis in Isaac.

The rotation tipped the pad sideways because we rotated about the wrong axis.
We need the axis that points OUT of the pad face (the pad normal).

This script:
  - reads the pad frame (TSF_85_right) and wrist frame (wrist_3_link)
  - prints the pad's three local axes as world vectors
  - prints the wrist's three local axes as world vectors
  - figures out which direction is the pad NORMAL by using the vector from the
    pad toward the cylinder (the pad must be FACING the cylinder)
  - reports which WRIST-LOCAL axis (x=col0, y=col1, z=col2) best matches that
    normal -> that column index is what rotate_pad_in_air must spin about.

Run:
  cd ~/Paper3_Simulation/TSF-85/examples
  ~/isaacsim/python.sh ~/Paper3_Simulation/sim/probe_pad_normal.py 2>/dev/null
  cat ~/Paper3_Simulation/Data/pad_normal_result.txt
"""

import sys
sys.path.insert(0, "/home/kourosh/Paper3_Simulation/curobo-stable/src")
import os, numpy as np

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "physics_gpu": 0})

import carb
carb.settings.get_settings().set("/physics/enableDeformableBodies", True)
carb.settings.get_settings().set("/physics/enableGpuDynamics",      True)

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import UsdGeom, Usd

EXAMPLES_DIR = "/home/kourosh/Paper3_Simulation/TSF-85/examples"
USD_PATH     = os.path.join(EXAMPLES_DIR, "scenes", "scene_cylinder.usd")
ROBOT_PRIM   = "/World/robot_gripper_adapter_sensor"
CYL_CENTER   = np.array([-0.26806, 0.199, 1.0522])  # world cylinder center

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=USD_PATH, prim_path=ROBOT_PRIM)
stage = world.stage
world.reset()

def find_prim(name_substr):
    hits = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PRIM)):
        p = str(prim.GetPath())
        if p.endswith(name_substr) or ("/"+name_substr in p):
            hits.append(p)
    return hits

def pose(path):
    prim = stage.GetPrimAtPath(path)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    m  = xc.GetLocalToWorldTransform(prim)
    t  = m.ExtractTranslation()
    pos = np.array([t[0], t[1], t[2]])
    R = np.array(m.ExtractRotationMatrix()).T   # columns = local axes in world
    return pos, R

OUT = "/home/kourosh/Paper3_Simulation/Data/pad_normal_result.txt"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
L = []
def w(s=""): L.append(s)

w("================= PAD NORMAL PROBE =================")

wrist_path = min(find_prim("wrist_3_link"), key=len)
pad_path   = min(find_prim("TSF_85_right"), key=len)
w(f"wrist: {wrist_path}")
w(f"pad:   {pad_path}")

pw, Rw = pose(wrist_path)
pp, Rp = pose(pad_path)

w(f"\npad pos (world)   = {pp}")
w(f"cylinder (world)  = {CYL_CENTER}")

# direction from pad toward cylinder = the way the pad SHOULD be facing (its normal)
to_cyl = CYL_CENTER - pp
to_cyl = to_cyl / (np.linalg.norm(to_cyl) + 1e-12)
w(f"\npad -> cylinder unit vector (true normal dir) = {to_cyl.round(3)}")

w("\n--- PAD local axes (world) ---")
for i, nm in enumerate(["pad_X", "pad_Y", "pad_Z"]):
    ax = Rp[:, i]
    d = float(np.dot(ax, to_cyl))
    w(f"  {nm} = {ax.round(3)}   dot(to_cyl) = {d:+.3f}")

w("\n--- WRIST local axes (world) ---  (rotate_pad_in_air uses one of these)")
best_i, best_d = 0, -2
for i, nm in enumerate(["wrist_X(col0)", "wrist_Y(col1)", "wrist_Z(col2)"]):
    ax = Rw[:, i]
    d = float(np.dot(ax, to_cyl))
    w(f"  {nm} = {ax.round(3)}   dot(to_cyl) = {d:+.3f}")
    if abs(d) > abs(best_d):
        best_d, best_i = d, i

names = ["X (column 0)", "Y (column 1)", "Z (column 2)"]
w(f"\n>>> The wrist axis closest to the pad normal is: wrist-{names[best_i]}  (dot={best_d:+.3f})")
w(f">>> rotate_pad_in_air should spin about wrist column {best_i}.")
w(f">>> (current code uses column 0 = X. {'CORRECT' if best_i==0 else 'WRONG -> change to column '+str(best_i)})")

w("\n===================================================")
with open(OUT, "w") as f:
    f.write("\n".join(L))

simulation_app.close()
