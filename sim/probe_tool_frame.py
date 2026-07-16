"""
probe_tool_frame.py — Confirm which Isaac frame matches your real-robot 'tool0'.

Your Paper-2 code measures L = 171 mm from tool0 (UR5e wrist flange = wrist_3_link)
along the tool's v-axis (R_tool0 column 2) to the pad contact point.

This script:
  - finds wrist_3_link and the pad (TSF_85_right) in the Isaac scene
  - prints both world poses
  - prints the distance between them (should be ~0.171 m if wrist_3_link is tool0)
  - prints, in the wrist's LOCAL frame, where the pad sits (so we see the v-axis offset)

Run (no grasp, just loads + reports):
  cd ~/Paper3_Simulation/TSF-85/examples
  ~/isaacsim/python.sh ~/Paper3_Simulation/sim/probe_tool_frame.py
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
from pxr import UsdGeom, Usd, Gf

EXAMPLES_DIR = "/home/kourosh/Paper3_Simulation/TSF-85/examples"
USD_PATH     = os.path.join(EXAMPLES_DIR, "scenes", "scene_cylinder.usd")
ROBOT_PRIM   = "/World/robot_gripper_adapter_sensor"

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=USD_PATH, prim_path=ROBOT_PRIM)
stage = world.stage
world.reset()

def find_prim_by_name(name_substr):
    """Return the first prim whose path ends with name_substr (or contains it)."""
    hits = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PRIM)):
        p = str(prim.GetPath())
        if p.endswith(name_substr) or ("/"+name_substr in p):
            hits.append(p)
    return hits

def world_pose(path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None, None
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    m  = xc.GetLocalToWorldTransform(prim)
    t  = m.ExtractTranslation()
    q  = m.ExtractRotationQuat()
    pos = np.array([t[0], t[1], t[2]])
    # rotation matrix columns = local axes in world
    R = np.array(m.ExtractRotationMatrix()).T  # Gf row-major -> columns are axes
    return pos, R

OUT = "/home/kourosh/Paper3_Simulation/Data/probe_result.txt"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
lines = []
def w(s=""):
    lines.append(s)

w("==================== FRAME PROBE ====================")

# --- find wrist_3_link ---
wrist_hits = find_prim_by_name("wrist_3_link")
w("\n[wrist_3_link candidates]")
for h in wrist_hits:
    w("    " + h)

# --- find the pad (TSF_85_right) ---
pad_hits = find_prim_by_name("TSF_85_right")
w("\n[TSF_85_right candidates]")
for h in pad_hits[:8]:
    w("    " + h)

# pick the shortest path for each (the link/frame itself, not deep children)
wrist_path = min(wrist_hits, key=len) if wrist_hits else None
pad_path   = min(pad_hits, key=len) if pad_hits else None

w("\n[chosen]")
w("  wrist tool0 -> " + str(wrist_path))
w("  pad         -> " + str(pad_path))

if wrist_path and pad_path:
    pw, Rw = world_pose(wrist_path)
    pp, Rp = world_pose(pad_path)
    w("\n[world positions]")
    w(f"  wrist_3_link pos = {pw}")
    w(f"  pad pos          = {pp}")

    d = np.linalg.norm(pp - pw)
    w(f"\n  distance wrist->pad = {d:.4f} m  ({d*1000:.1f} mm)")
    w(f"  (your real-robot L = 171.0 mm)")

    off_local = Rw.T @ (pp - pw)
    w(f"\n[pad offset in wrist-local frame] (x,y,z) = "
      f"({off_local[0]*1000:.1f}, {off_local[1]*1000:.1f}, {off_local[2]*1000:.1f}) mm")
    w("  In your code: n=col0(x), u=col1(y), v=col2(z).")
    w(f"  -> along n (x): {off_local[0]*1000:.1f} mm")
    w(f"  -> along u (y): {off_local[1]*1000:.1f} mm   (your W = 0 mm)")
    w(f"  -> along v (z): {off_local[2]*1000:.1f} mm   (your L = 171 mm)")
    w("\n  If 'along v' ~= 171 mm, wrist_3_link is the right tool0 and L transfers directly.")
else:
    w("\n  WARNING: could not find one of the frames. All wrist/pad path hits:")
    w("  wrist hits: " + str(wrist_hits))
    w("  pad hits:   " + str(pad_hits[:8]))

w("\n====================================================")

with open(OUT, "w") as f:
    f.write("\n".join(lines))

simulation_app.close()
