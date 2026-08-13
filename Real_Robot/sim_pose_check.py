#!/usr/bin/env python3
"""
sim_pose_check.py — put Isaac's arm at the REAL robot's joint angles and print
where tool0 lands, so the two frames can be compared directly.

WHY. Paper 3's grid, canvas and training pairs are all expressed in world
coordinates. Running the same config on the real robot only makes sense if
Berith's simulated station and the physical cell agree about where a given
joint vector puts the tool. A constant offset would silently shift every
stitched map; a rotational difference would shear them.

WHAT IT DOES. Opens the scene, sets the six arm joints, steps physics until
they settle, then reads the flange prim's WORLD transform straight from USD.
No cuRobo, no planning, no gripper, no object contact. Nothing is written
except sim_pose_check.json.

USE. Take q_rad from read_pose.py on the real robot and pass it here:

  cd ~/Paper3_Simulation/TSF-85/examples && GRASP_HEADLESS=1 \
  ~/isaacsim/python.sh ~/Paper3_Simulation/Real_Robot/sim_pose_check.py \
  --q -1.272135 -1.627053 -2.058342 -1.025654 1.568704 4.481452

Then compare its tool0 world mm against the real robot's tool0_pos_mm.
"""

import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--q", nargs=6, type=float, required=True,
                help="six joint angles in RADIANS, canonical order: "
                     "shoulder_pan, shoulder_lift, elbow, wrist_1, "
                     "wrist_2, wrist_3")
ap.add_argument("--project", default=os.path.expanduser("~/Paper3_Simulation"))
ap.add_argument("--usd", default=None,
                help="scene .usd; default is the one collect_from_config uses")
ap.add_argument("--settle", type=int, default=240,
                help="physics steps to let the joints reach the target")
ap.add_argument("--out", default="sim_pose_check.json")
args = ap.parse_args()

Q = np.array(args.q, dtype=float)

# The scene lives with the TSF extension's examples, NOT at the project root:
# collect_from_config.py has
#     EXAMPLES_DIR = ".../Paper3_Simulation/TSF-85/examples"
#     SCENES_DIR   = EXAMPLES_DIR/scenes
# Checked BEFORE Isaac starts, because a wrong path otherwise costs 20 s of
# startup and then surfaces as "no articulation found", which points at the
# robot rather than at the missing file.
USD_PATH = args.usd or os.path.join(
    args.project, "TSF-85", "examples", "scenes", "scene_cylinder.usd")
if not os.path.exists(USD_PATH):
    print(f"FAILED: scene not found: {USD_PATH}")
    print("        pass the right one with --usd /path/to/scene.usd")
    sys.exit(1)
print(f"[sim] scene: {USD_PATH}")

from isaacsim.simulation_app import SimulationApp            # noqa: E402
sim = SimulationApp({"headless": os.environ.get("GRASP_HEADLESS", "1") == "1"})

from isaacsim.core.api import World                          # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.prims import SingleArticulation           # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics                     # noqa: E402

ROBOT_PRIM_PATH = "/World/robot_gripper_adapter_sensor"
FLANGE_PRIM = f"{ROBOT_PRIM_PATH}/robot_gripper_adapter_sensor/ur5e/wrist_3_link"
ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

world = World(stage_units_in_meters=1.0, physics_dt=1 / 120.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=USD_PATH, prim_path=ROBOT_PRIM_PATH)
stage = world.stage

# LET THE REFERENCE FINISH LOADING before looking for the articulation.
# add_reference_to_stage returns before composition is complete, so a
# traversal on the next line finds an empty prim and the articulation search
# silently returns None — which then surfaces much later as
# "'NoneType' object has no attribute 'is_homogeneous'" inside world.reset().
# collect_from_config.py never hit this only because it does a lot of other
# work (object sizing, PhysX scene setup, extension enabling) in between.
for _ in range(60):
    sim.update()

# Same PhysX settings the collector uses, so the articulation is created the
# same way here as it is there.
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Scene):
        from pxr import PhysxSchema
        a = PhysxSchema.PhysxSceneAPI.Apply(prim)
        a.CreateEnableGPUDynamicsAttr().Set(True)
        try:
            a.CreateBroadphaseTypeAttr().Set("GPU")
        except Exception:
            pass


def find_ur5e(s, u, jn):
    rp = s.GetPrimAtPath(u)
    roots = [p for p in Usd.PrimRange(rp)
             if "PhysicsArticulationRootAPI" in p.GetAppliedSchemas()] \
        if rp.IsValid() else []
    for c in roots:
        pp = c.GetParent() if c.IsA(UsdPhysics.Joint) else c
        for x in Usd.PrimRange(pp):
            if x.IsA(UsdPhysics.Joint):
                n = x.GetName()
                if any(n == j or n.endswith("/" + j) for j in jn):
                    return c
    return None


ar = find_ur5e(stage, ROBOT_PRIM_PATH, ARM)
if ar is None:
    # Say WHAT is on the stage instead of failing 80 lines later inside
    # world.reset() with an error that names none of this.
    print(f"\nFAILED: no UR5e articulation under {ROBOT_PRIM_PATH}")
    print(f"  USD: {USD_PATH}  (exists={os.path.exists(USD_PATH)})")
    _roots = [str(p.GetPath()) for p in stage.Traverse()
              if "PhysicsArticulationRootAPI" in p.GetAppliedSchemas()]
    print(f"  articulation roots anywhere on the stage ({len(_roots)}):")
    for r in _roots[:20]:
        print(f"     {r}")
    _top = [str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()]
    print(f"  top-level prims: {_top}")
    sim.close()
    sys.exit(1)

AP = str(ar.GetParent().GetPath()) if ar.IsA(UsdPhysics.Joint) \
    else str(ar.GetPath())
print(f"[sim] articulation root: {AP}")

base_pos = base_quat = None
root_prim = stage.GetPrimAtPath(AP)
if root_prim.IsValid():
    for p in Usd.PrimRange(root_prim):
        if p.GetName() == "base_link":
            xf = UsdGeom.XformCache(Usd.TimeCode.Default()) \
                .GetLocalToWorldTransform(p)
            t, qq = xf.ExtractTranslation(), xf.ExtractRotationQuat()
            base_pos = np.array([t[0], t[1], t[2]], float)
            base_quat = [qq.GetReal(), qq.GetImaginary()[0],
                         qq.GetImaginary()[1], qq.GetImaginary()[2]]
            break

robot = SingleArticulation(prim_path=AP, name="ur5e")
world.scene.add(robot)
world.reset()

dn = list(robot.dof_names)
missing = [j for j in ARM if j not in dn]
if missing:
    print(f"FAILED: joints not in the articulation: {missing}")
    print(f"        available: {dn}")
    sim.close()
    sys.exit(1)
ai = [dn.index(j) for j in ARM]

# Command AND teleport: set_joint_positions places them, the target keeps the
# drives from pulling the arm back toward its previous pose while it settles.
from isaacsim.core.utils.types import ArticulationAction      # noqa: E402
robot.set_joint_positions(Q, joint_indices=ai)
robot.apply_action(ArticulationAction(joint_positions=Q, joint_indices=ai))
for _ in range(args.settle):
    world.step(render=False)

q_reached = np.asarray(robot.get_joint_positions())[ai]
err_deg = np.degrees(q_reached - Q)

# tool0 == the flange prim's world transform, read passively from USD.
_fl = stage.GetPrimAtPath(FLANGE_PRIM)
if not _fl.IsValid():
    print(f"\nFAILED: flange prim not found: {FLANGE_PRIM}")
    _cands = [str(p.GetPath()) for p in stage.Traverse()
              if p.GetName() in ("wrist_3_link", "tool0", "flange")]
    print("  candidates on the stage:")
    for c in _cands[:20]:
        print(f"     {c}")
    sim.close()
    sys.exit(1)
xf = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(_fl)
t = xf.ExtractTranslation()
Rm = np.array(xf.ExtractRotationMatrix()).T      # USD is row-vector convention
tool_world_mm = np.array([t[0], t[1], t[2]], float) * 1000.0

print("\n--- COMMANDED vs REACHED (deg) ------------------------------")
for j, c, r, e in zip(ARM, np.degrees(Q), np.degrees(q_reached), err_deg):
    print(f"  {j:<22} cmd {c:+8.3f}   reached {r:+8.3f}   err {e:+6.3f}")
if np.abs(err_deg).max() > 0.5:
    print(f"  WARNING: worst joint error {np.abs(err_deg).max():.2f} deg — the "
          f"arm has not settled. Raise --settle before trusting the pose.")

print("\n--- ISAAC ---------------------------------------------------")
if base_pos is not None:
    print(f"  base_link world mm : [{base_pos[0]*1000:+9.3f}, "
          f"{base_pos[1]*1000:+9.3f}, {base_pos[2]*1000:+9.3f}]")
    print(f"  base_link quat wxyz: {[round(v, 6) for v in base_quat]}")
print(f"  tool0 world mm     : [{tool_world_mm[0]:+9.3f}, "
      f"{tool_world_mm[1]:+9.3f}, {tool_world_mm[2]:+9.3f}]")
print("  tool0 R (rows)     :")
for r in Rm:
    print(f"                       [{r[0]:+.5f} {r[1]:+.5f} {r[2]:+.5f}]")

if base_pos is not None:
    rel = tool_world_mm - base_pos * 1000.0
    print(f"\n  tool0 RELATIVE to base_link, mm: [{rel[0]:+9.3f}, "
          f"{rel[1]:+9.3f}, {rel[2]:+9.3f}]")
    print("  ^ THIS is the number to compare with the real robot's")
    print("    tool0_pos_mm, which is already expressed in base_link.")
    print("    They should agree to a few mm. NOTE the real cell's ROS frame")
    print("    is base_link; the UR controller's 'base' frame is rotated 180")
    print("    deg about Z from it, so if you compare against a controller")
    print("    reading instead, flip the signs of X and Y first.")

with open(args.out, "w") as f:
    json.dump({"q_rad": Q.tolist(),
               "q_reached_rad": q_reached.tolist(),
               "joint_err_deg": err_deg.tolist(),
               "base_link_world_mm": (base_pos * 1000.0).tolist()
               if base_pos is not None else None,
               "base_link_quat_wxyz": base_quat,
               "tool0_world_mm": tool_world_mm.tolist(),
               "tool0_rel_base_mm": (tool_world_mm - base_pos * 1000.0).tolist()
               if base_pos is not None else None,
               "tool0_R": Rm.tolist(),
               "usd": USD_PATH}, f, indent=2)
print(f"\nsaved {os.path.abspath(args.out)}")

sim.close()
