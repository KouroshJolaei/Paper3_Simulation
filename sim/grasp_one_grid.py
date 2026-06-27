"""
grasp_one_grid.py — Runs ONE grid-point touch in a fresh Isaac Sim session.

This is your proven grasp_one.py with the SMALLEST possible changes for grid
scanning. It reads ONE grid point from environment variables (set by
run_grid.sh) and does the same approach -> descend -> close/record/open ->
ascend motion your working grasp does, just at the shifted grid location.

NEW vs grasp_one.py:
  - reads GRASP_ROT_DEG (finger rotation in degrees about the approach axis)
  - applies that rotation to the tool orientation
Everything else (the motion, the recording, the CSV writing) is UNCHANGED.

LAUNCHED BY: factory/run_grid.sh   (never run this directly)
"""

import sys
sys.path.insert(0, "/home/kourosh/Paper3_Simulation/curobo-stable/src")

import os

# ============================================================
# Read ONE grid point from environment variables
# ============================================================
LABEL        = os.environ.get("GRASP_LABEL",      "grid_default")
GRASP_X      = float(os.environ.get("GRASP_X",        "-0.26806"))
GRASP_Y      = float(os.environ.get("GRASP_Y",         "0.199"))
GRASP_Z      = float(os.environ.get("GRASP_Z",         "1.24244"))
APPROACH_H   = float(os.environ.get("GRASP_APPROACH",  "0.10"))
CLOSE_RAD    = float(os.environ.get("GRASP_CLOSE_RAD", "0.55"))
ROT_DEG      = float(os.environ.get("GRASP_ROT_DEG",   "0.0"))   # NEW: finger rotation
OUTPUT_DIR   = os.environ.get("GRASP_OUTPUT_DIR", "/home/kourosh/Paper3_Simulation/Data/default_grid")
BASENAME     = os.environ.get("GRASP_BASENAME",   "grid_default")

print(f"[grid] label={LABEL}")
print(f"[grid] world=({GRASP_X}, {GRASP_Y}, {GRASP_Z})  approach={APPROACH_H}  rot={ROT_DEG} deg")
print(f"[grid] close={CLOSE_RAD}  output={OUTPUT_DIR}  basename={BASENAME}")

# ============================================================
# Launch Isaac Sim (headless)
# ============================================================
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "physics_gpu": 0})

import numpy as np, carb

carb.settings.get_settings().set("/physics/enableDeformableBodies", True)
carb.settings.get_settings().set("/physics/enableGpuDynamics",      True)
carb.settings.get_settings().set("/exts/TSF_85_Ext/record_active",  False)

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdPhysics, PhysxSchema, Usd, UsdGeom

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import (
    MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric)

# ============================================================
# Constants
# ============================================================
EXAMPLES_DIR      = "/home/kourosh/Paper3_Simulation/TSF-85/examples"
SCENES_DIR        = os.path.join(EXAMPLES_DIR, "scenes")
USD_PATH          = os.path.join(SCENES_DIR, "scene_cylinder.usd")
CUROBO_ROBOT_YAML = os.path.join(SCENES_DIR, "ur5e.yml")
ROBOT_PRIM_PATH   = "/World/robot_gripper_adapter_sensor"
SENSOR_ROOT_RIGHT = f"{ROBOT_PRIM_PATH}/TSF_85_right/TSF_85"
SENSOR_ROOT_LEFT  = f"{ROBOT_PRIM_PATH}/TSF_85_left/TSF_85"
TSF_EXT_SEARCH    = "/home/kourosh/Paper3_Simulation/TSF-85"

ARM_JOINT_NAMES     = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
                        "wrist_1_joint","wrist_2_joint","wrist_3_joint"]
GRIPPER_DRIVE_JOINT = "finger_joint"
GRIPPER_OPEN        = 0.0
INITIAL_JOINTS_RAD  = np.array([-0.992425, -2.179929, -0.865866,
                                  -1.667783,  1.570776, -0.992413])
ROBOT_WORLD_POS       = np.array([0.0, -0.3375, 0.99275])
ROBOT_WORLD_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
TOOL_DOWN_ROTVEC      = np.array([2.2214, 2.2214, 0.0])

GRIPPER_RAMP_FRAMES = 60
WAIT_GRASP_SECONDS  = 1.0
WAIT_HOLD_SECONDS   = 1.0
N_STEPS             = 10
CASE13_WEIGHT       = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Helpers
# ============================================================
def rotvec_to_quat(rv):
    a = float(np.linalg.norm(rv))
    if a < 1e-9: return np.array([1.,0,0,0])
    ax = rv/a; s = np.sin(a/2)
    return np.array([np.cos(a/2), ax[0]*s, ax[1]*s, ax[2]*s])

def quat_mul(a, b):
    """Hamilton product of two wxyz quaternions."""
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])

def rotmat(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)]])

def world_to_base(p):
    return rotmat(ROBOT_WORLD_QUAT_WXYZ).T @ (p - ROBOT_WORLD_POS)

# ============================================================
# Build world
# ============================================================
world = World(stage_units_in_meters=1.0, physics_dt=1/120.,
              rendering_dt=1/60., backend="numpy")
pc = world.get_physics_context()
pc.enable_gpu_dynamics(True)
pc.set_broadphase_type("GPU")
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=USD_PATH, prim_path=ROBOT_PRIM_PATH)
stage = world.stage

for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Scene):
        a = PhysxSchema.PhysxSceneAPI.Apply(prim)
        a.CreateEnableGPUDynamicsAttr().Set(True)
        try: a.CreateBroadphaseTypeAttr().Set("GPU")
        except Exception: pass

# ============================================================
# Enable TSF extension with THIS grid point's output dir + basename
# ============================================================
_tsf = carb.settings.get_settings()
_tsf.set("/exts/TSF_85_Ext/headless",      True)
_tsf.set("/exts/TSF_85_Ext/sensor_root",   SENSOR_ROOT_RIGHT)
_tsf.set("/exts/TSF_85_Ext/sensor_root_2", SENSOR_ROOT_LEFT)
_tsf.set("/exts/TSF_85_Ext/output_dir",    OUTPUT_DIR)
_tsf.set("/exts/TSF_85_Ext/base_name",     BASENAME)
_tsf.set("/exts/TSF_85_Ext/log_dz",        True)
_tsf.set("/exts/TSF_85_Ext/log_pred",      True)
_tsf.set("/exts/TSF_85_Ext/log_mesh",      True)

from omni.kit.app import get_app
_ext_mgr = get_app().get_extension_manager()
_ext_mgr.add_path(TSF_EXT_SEARCH)
_ok = _ext_mgr.set_extension_enabled_immediate("TSF_85_Ext", True)
print(f"[grid] TSF_85_Ext enabled={_ok}")

# ============================================================
# Find robot + base pose
# ============================================================
def find_ur5e(s, u, jn):
    rp = s.GetPrimAtPath(u)
    roots = [p for p in Usd.PrimRange(rp)
             if "PhysicsArticulationRootAPI" in p.GetAppliedSchemas()] if rp.IsValid() else []
    for c in roots:
        pp = c.GetParent() if c.IsA(UsdPhysics.Joint) else c
        for x in Usd.PrimRange(pp):
            if x.IsA(UsdPhysics.Joint):
                n = x.GetName()
                if any(n == j or n.endswith("/"+j) for j in jn):
                    return c
    return None

ar = find_ur5e(stage, ROBOT_PRIM_PATH, ARM_JOINT_NAMES)
AP = str(ar.GetParent().GetPath()) if (ar and ar.IsA(UsdPhysics.Joint)) else (
     str(ar.GetPath()) if ar else ROBOT_PRIM_PATH)

root_prim = stage.GetPrimAtPath(AP)
if root_prim.IsValid():
    for p in Usd.PrimRange(root_prim):
        if p.GetName() == "base_link":
            xfc = UsdGeom.XformCache(Usd.TimeCode.Default())
            xf  = xfc.GetLocalToWorldTransform(p)
            t   = xf.ExtractTranslation()
            q   = xf.ExtractRotationQuat()
            ROBOT_WORLD_POS[:]       = [t[0], t[1], t[2]]
            ROBOT_WORLD_QUAT_WXYZ[:] = [q.GetReal(),
                                         q.GetImaginary()[0],
                                         q.GetImaginary()[1],
                                         q.GetImaginary()[2]]
            print(f"[grid] base_link world pos: {ROBOT_WORLD_POS}")
            break

robot = SingleArticulation(prim_path=AP, name="ur5e")
world.scene.add(robot)
world.reset()

dn = robot.dof_names
def idxs(dn, bn):
    o = []
    for nm in bn:
        if nm in dn:
            o.append(dn.index(nm))
        else:
            c = [d for d in dn if d == nm or d.endswith("/"+nm) or d.endswith(nm)]
            o.append(dn.index(c[0]))
    return np.array(o, dtype=np.int32)

ai = idxs(dn, ARM_JOINT_NAMES)
try:
    gi = np.array([dn.index(GRIPPER_DRIVE_JOINT)], dtype=np.int32)
except ValueError:
    cand = [d for d in dn if d.endswith("/"+GRIPPER_DRIVE_JOINT)
                              or d.endswith(GRIPPER_DRIVE_JOINT)]
    gi = np.array([dn.index(cand[0])], dtype=np.int32) if cand else None

dp = np.array(robot.get_joint_positions(), dtype=np.float32)
dp[ai] = INITIAL_JOINTS_RAD
robot.set_joints_default_state(positions=dp)
robot.set_joint_positions(INITIAL_JOINTS_RAD, joint_indices=ai)
robot.get_articulation_controller().apply_action(
    ArticulationAction(joint_positions=INITIAL_JOINTS_RAD, joint_indices=ai))
for _ in range(10): world.step(render=True)
initial_q = robot.get_joint_positions()[ai].copy()

# ============================================================
# cuRobo
# ============================================================
print("[grid] Loading cuRobo...")
ta = TensorDeviceType()
rc = RobotConfig.from_dict(load_yaml(CUROBO_ROBOT_YAML)["robot_cfg"], ta)
mg = MotionGen(MotionGenConfig.load_from_robot_config(
    rc, world_model=None, tensor_args=ta, interpolation_dt=0.02,
    num_trajopt_seeds=4, project_pose_to_goal_frame=True, use_cuda_graph=False))
mg.warmup(enable_graph=False, warmup_js_trajopt=False)
print("[grid] cuRobo ready.")

# ---- Tool orientation, with OPTIONAL finger rotation ----
# Base tool-down orientation (same as your working grasp):
tq_base = rotvec_to_quat(TOOL_DOWN_ROTVEC)
# Rotation of the fingers about the approach (tool Z / world-down) axis:
if abs(ROT_DEG) > 1e-6:
    rot_rad = np.deg2rad(ROT_DEG)
    # rotation about world Z axis (the vertical approach axis)
    spin = np.array([np.cos(rot_rad/2), 0.0, 0.0, np.sin(rot_rad/2)])
    tq = quat_mul(spin, tq_base)
else:
    tq = tq_base

current_grip = [0.0]

def apply_arm_and_grip(arm_q, grip_val=None):
    if grip_val is not None:
        current_grip[0] = float(grip_val)
    robot.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=arm_q.astype(np.float32), joint_indices=ai))
    if gi is not None:
        robot.get_articulation_controller().apply_action(
            ArticulationAction(
                joint_positions=np.array([current_grip[0]], dtype=np.float32),
                joint_indices=gi))

def run_traj(traj, settle=True):
    for q in traj:
        apply_arm_and_grip(q)
        world.step(render=True)
    if settle:
        fc = traj[-1].astype(np.float32)
        for _ in range(120):
            apply_arm_and_grip(fc)
            world.step(render=True)
            if np.max(np.abs(robot.get_joint_positions()[ai] - fc)) < 0.005:
                break

def fk(q):
    qt = ta.to_device(q.astype(np.float32)).view(1, -1)
    f  = mg.compute_kinematics(JointState.from_position(qt, joint_names=ARM_JOINT_NAMES))
    if hasattr(f, "ee_pose") and f.ee_pose is not None:
        return (f.ee_pose.position.cpu().numpy().flatten(),
                f.ee_pose.quaternion.cpu().numpy().flatten())
    return (f.ee_position.cpu().numpy().flatten(),
            f.ee_quaternion.cpu().numpy().flatten())

def plan_free_move(start_q, target_base, label):
    s = JointState.from_position(
            ta.to_device(start_q.astype(np.float32)).view(1, -1),
            joint_names=ARM_JOINT_NAMES)
    g = Pose(position=ta.to_device(target_base.astype(np.float32)).view(1, 3),
             quaternion=ta.to_device(tq.astype(np.float32)).view(1, 4))
    r = mg.plan_single(s, g, MotionGenPlanConfig(max_attempts=5, enable_graph=False))
    if not r.success.item():
        print(f"  [{label}] free move FAILED ({r.status})")
        return None
    return r.get_interpolated_plan().position.cpu().numpy()

def plan_stitched_z(start_q, dz, label):
    metric = PoseCostMetric(hold_partial_pose=True,
        hold_vec_weight=mg.tensor_args.to_device(
            np.array(CASE13_WEIGHT, dtype=np.float32)))
    cfg = MotionGenPlanConfig(enable_graph=False, max_attempts=4,
                              enable_finetune_trajopt=False, pose_cost_metric=metric)
    step = dz / N_STEPS
    cur_q = start_q.copy()
    stitched = []
    for i in range(N_STEPS):
        cpos, cquat = fk(cur_q)
        tgt = cpos.copy(); tgt[2] += step
        s = JointState.from_position(
                ta.to_device(cur_q.astype(np.float32)).view(1, -1),
                joint_names=ARM_JOINT_NAMES)
        g = Pose(position=ta.to_device(tgt.astype(np.float32)).view(1, 3),
                 quaternion=ta.to_device(cquat.astype(np.float32)).view(1, 4))
        r = mg.plan_single(s, g, cfg)
        if not r.success.item():
            print(f"  [{label}] step {i+1}/{N_STEPS} FAILED ({r.status})")
            return None
        tr = r.get_interpolated_plan().position.cpu().numpy()
        if stitched: tr = tr[1:]
        stitched.extend(list(tr))
        cur_q = tr[-1].copy()
    return np.array(stitched)

def ramp_gripper(arm_q, target, n_frames):
    cur_g = float(robot.get_joint_positions()[gi[0]])
    for k in range(n_frames):
        alpha = (k + 1) / n_frames
        apply_arm_and_grip(arm_q, grip_val=cur_g + alpha*(target - cur_g))
        world.step(render=True)

def hold_for(arm_q, seconds):
    for _ in range(int(seconds * 60)):
        apply_arm_and_grip(arm_q)
        world.step(render=True)

# ============================================================
# Run the single grid-point touch
# (identical motion to grasp_one.py, just at the grid location)
# ============================================================
grasp_world  = np.array([GRASP_X, GRASP_Y, GRASP_Z])
up_world     = grasp_world.copy()
up_world[2] += APPROACH_H
up_base      = world_to_base(up_world)

EXIT_CODE = 0
try:
    q = initial_q.copy()

    print(f"[grid] free move to UP {up_world} ...")
    traj_up = plan_free_move(q, up_base, f"{LABEL}:to-up")
    if traj_up is None:
        print(f"[grid] FAILED to reach UP. Exiting.")
        EXIT_CODE = 2
    else:
        run_traj(traj_up)
        q_up = robot.get_joint_positions()[ai].copy()

        print(f"[grid] straight descent UP->GRASP ...")
        dz_dn = -float(np.linalg.norm(grasp_world - up_world))
        traj_dn = plan_stitched_z(q_up, dz_dn, f"{LABEL}:DOWN")
        if traj_dn is None:
            print(f"[grid] descent FAILED. Exiting.")
            EXIT_CODE = 3
        else:
            run_traj(traj_dn)
            q_grasp = robot.get_joint_positions()[ai].copy()
            hold_qg = q_grasp.astype(np.float32)

            # settle (not recorded)
            hold_for(hold_qg, WAIT_GRASP_SECONDS)

            # RECORD ON -> close -> hold -> open -> RECORD OFF
            _tsf.set("/exts/TSF_85_Ext/record_active", True)
            print(f"[grid] [RECORD ON] close -> hold -> open ...")
            ramp_gripper(hold_qg, CLOSE_RAD, GRIPPER_RAMP_FRAMES)
            hold_for(hold_qg, WAIT_HOLD_SECONDS)
            ramp_gripper(hold_qg, GRIPPER_OPEN, GRIPPER_RAMP_FRAMES)
            _tsf.set("/exts/TSF_85_Ext/record_active", False)
            print(f"[grid] [RECORD OFF]")

            # ascent (not recorded)
            print(f"[grid] straight ascent GRASP->UP ...")
            dz_up = float(np.linalg.norm(up_world - grasp_world))
            traj_up2 = plan_stitched_z(q_grasp, dz_up, f"{LABEL}:UP")
            if traj_up2 is not None:
                run_traj(traj_up2)

            print(f"[grid] SUCCESS. Data in {OUTPUT_DIR} (prefix {BASENAME})")
finally:
    for _ in range(30):
        world.step(render=True)
    simulation_app.close()

sys.exit(EXIT_CODE)
