"""
place_object.py — SET the cylinder's pose and FREEZE it in place.

Opens the scene in a visible window, moves the cylinder to a pose you choose
(position + orientation), and makes it KINEMATIC (frozen) so gravity and the
gripper can't move it. You can watch it sit rock-solid.

The cylinder body is Object_02 (a rigid body). Setting kinematic=True pins it:
it keeps its collider (still grippable) but never moves.

RUN (window opens):
  cd ~/Paper3_Simulation/TSF-85/examples
  ~/isaacsim/python.sh ~/Paper3_Simulation/sim/place_object.py

CHANGE THE POSE: edit the CONFIG block below, then re-run.
"""

import sys
sys.path.insert(0, "/home/kourosh/Paper3_Simulation/curobo-stable/src")
import os

# ============================================================
# CONFIG — edit these, then re-run
# ============================================================
# Position of the cylinder CENTER in world meters.
# Current scene pose is about (-0.26806, 0.199, 1.0522).
POS_X = -0.26806
POS_Y =  0.199
POS_Z =  1.0522

# Orientation preset: "standing" | "sideways_x" | "sideways_y" | "tilted" | "custom"
#   standing    -> long axis vertical (as-is, like a standing can)
#   sideways_x  -> laid down, long axis along world X
#   sideways_y  -> laid down, long axis along world Y
#   tilted      -> tilted by TILT_DEG about world X
#   custom      -> use CUSTOM_QUAT_WXYZ below
ORIENTATION = "standing"
TILT_DEG    = 30.0
CUSTOM_QUAT_WXYZ = [1.0, 0.0, 0.0, 0.0]

# Freeze it? True = pinned (kinematic), can't move/fall. False = normal dynamic.
FREEZE = True

# Hold the window open this many seconds so you can look / orbit.
WATCH_SECONDS = 60
# ============================================================

OBJECT_PATH = "/World/robot_gripper_adapter_sensor/Object_02"

print(f"[place] pose=({POS_X},{POS_Y},{POS_Z}) orient={ORIENTATION} freeze={FREEZE}")

from isaacsim import SimulationApp
app = SimulationApp({"headless": False, "physics_gpu": 0})

import numpy as np, carb
carb.settings.get_settings().set("/physics/enableDeformableBodies", True)
carb.settings.get_settings().set("/physics/enableGpuDynamics",      True)

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf

EXAMPLES_DIR = "/home/kourosh/Paper3_Simulation/TSF-85/examples"
USD_PATH = os.path.join(EXAMPLES_DIR, "scenes", "scene_cylinder.usd")
ROBOT_PRIM_PATH = "/World/robot_gripper_adapter_sensor"


def euler_to_quat_wxyz(rx, ry, rz):
    """ZYX euler (radians) -> quaternion wxyz."""
    cx, sx = np.cos(rx/2), np.sin(rx/2)
    cy, sy = np.cos(ry/2), np.sin(ry/2)
    cz, sz = np.cos(rz/2), np.sin(rz/2)
    w = cx*cy*cz + sx*sy*sz
    x = sx*cy*cz - cx*sy*sz
    y = cx*sy*cz + sx*cy*sz
    z = cx*cy*sz - sx*sy*cz
    return [w, x, y, z]


def get_quat():
    if ORIENTATION == "standing":
        return [1.0, 0.0, 0.0, 0.0]
    if ORIENTATION == "sideways_x":      # tip 90 deg about Y -> long axis along X
        return euler_to_quat_wxyz(0, np.pi/2, 0)
    if ORIENTATION == "sideways_y":      # tip 90 deg about X -> long axis along Y
        return euler_to_quat_wxyz(np.pi/2, 0, 0)
    if ORIENTATION == "tilted":
        return euler_to_quat_wxyz(np.deg2rad(TILT_DEG), 0, 0)
    return CUSTOM_QUAT_WXYZ


world = World(stage_units_in_meters=1.0)
pc = world.get_physics_context()
pc.enable_gpu_dynamics(True)
pc.set_broadphase_type("GPU")
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=USD_PATH, prim_path=ROBOT_PRIM_PATH)
world.reset()
stage = world.stage

# enable GPU dynamics on any physics scene prim (needed by the deformable sensor)
for _p in stage.Traverse():
    if _p.IsA(UsdPhysics.Scene):
        _a = PhysxSchema.PhysxSceneAPI.Apply(_p)
        _a.CreateEnableGPUDynamicsAttr().Set(True)
        try: _a.CreateBroadphaseTypeAttr().Set("GPU")
        except Exception: pass

prim = stage.GetPrimAtPath(OBJECT_PATH)
if not prim.IsValid():
    print(f"[place] ERROR: {OBJECT_PATH} not found.")
    app.close(); sys.exit(1)

# ---- 1) Set the pose (position + orientation) ----
# The cylinder already has xformOp:translate and xformOp:orient defined with
# DOUBLE precision (quatd). We must reuse those exact ops with matching
# precision, NOT add new float ones (that causes a precision-mismatch crash).
q = get_quat()
try:
    existing = {op.GetOpName(): op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()}

    if "xformOp:translate" in existing:
        existing["xformOp:translate"].Set(Gf.Vec3d(POS_X, POS_Y, POS_Z))
    else:
        UsdGeom.Xformable(prim).AddTranslateOp(
            UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(POS_X, POS_Y, POS_Z))

    quat_d = Gf.Quatd(float(q[0]), Gf.Vec3d(float(q[1]), float(q[2]), float(q[3])))
    if "xformOp:orient" in existing:
        existing["xformOp:orient"].Set(quat_d)
    else:
        UsdGeom.Xformable(prim).AddOrientOp(
            UsdGeom.XformOp.PrecisionDouble).Set(quat_d)

    print(f"[place] set pose: pos=({POS_X},{POS_Y},{POS_Z}) quat_wxyz={[round(v,4) for v in q]}")
except Exception as e:
    print(f"[place] WARNING: could not set pose ({e}). Keeping current pose, still freezing.")

# ---- 2) Freeze it (kinematic) ----
if FREEZE:
    rb = UsdPhysics.RigidBodyAPI(prim)
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    k = rb.GetKinematicEnabledAttr()
    if not k:
        k = rb.CreateKinematicEnabledAttr()
    k.Set(True)
    print("[place] cylinder FROZEN (kinematic=True) — gravity/contact can't move it.")
else:
    print("[place] cylinder left DYNAMIC (will fall/move).")

# step physics so the change takes effect, then hold the window open
world.reset()
print(f"[place] holding window {WATCH_SECONDS}s — orbit/zoom to inspect.")
for _ in range(int(WATCH_SECONDS * 60)):
    world.step(render=True)

app.close()
