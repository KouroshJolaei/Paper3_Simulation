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
simulation_app = SimulationApp({"headless": False, "physics_gpu": 0})

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

# ============================================================
# STABILIZE the object with a FIXED JOINT (bolt it to the world).
# The object stays a DYNAMIC body (deformable sensor still reads contact),
# but a fixed joint anchors it so it can't tip or fall.
# Optional explicit pose via OBJ_POS_X/Y/Z and OBJ_ORIENT
#   (standing | horizontal_x | horizontal_y | horizontal_y_tilt | tilt_x | keep).
# OBJ_TILT_DEG sets the tilt angle for the *_tilt / tilt_x options.
# Controlled by GRASP_FREEZE_OBJECT (default on). Set to "0" to disable.
# ============================================================
if os.environ.get("GRASP_FREEZE_OBJECT", "1") != "0":
    _frz = "/World/robot_gripper_adapter_sensor/Object_02"
    try:
        from pxr import Gf
        _obj = stage.GetPrimAtPath(_frz)
        if _obj.IsValid():
            _rb = UsdPhysics.RigidBodyAPI(_obj) if _obj.HasAPI(UsdPhysics.RigidBodyAPI) else UsdPhysics.RigidBodyAPI.Apply(_obj)
            _k = _rb.GetKinematicEnabledAttr()
            if _k: _k.Set(False)   # ensure dynamic so sensor reads contact

            _xc = UsdGeom.XformCache()
            _m = _xc.GetLocalToWorldTransform(_obj)
            _cur_pos = _m.ExtractTranslation()
            _cur_rot = _m.ExtractRotationQuat()

            _px = float(os.environ.get("OBJ_POS_X", _cur_pos[0]))
            _py = float(os.environ.get("OBJ_POS_Y", _cur_pos[1]))
            _pz = float(os.environ.get("OBJ_POS_Z", _cur_pos[2]))
            _orient = os.environ.get("OBJ_ORIENT", "keep")

            def _euler_q(rx, ry, rz):
                cx,sx=np.cos(rx/2),np.sin(rx/2); cy,sy=np.cos(ry/2),np.sin(ry/2); cz,sz=np.cos(rz/2),np.sin(rz/2)
                return Gf.Quatf(float(cx*cy*cz+sx*sy*sz), float(sx*cy*cz-cx*sy*sz),
                                float(cx*sy*cz+sx*cy*sz), float(cx*cy*sz-sx*sy*cz))
            if _orient == "standing":
                _q = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
            elif _orient == "horizontal_x":
                _q = _euler_q(0, np.pi/2, 0)
            elif _orient == "horizontal_y":
                _q = _euler_q(np.pi/2, 0, 0)
            elif _orient == "horizontal_y_tilt":
                _tilt = np.deg2rad(float(os.environ.get("OBJ_TILT_DEG", "20")))
                _q = _euler_q(np.pi/2 + _tilt, 0, 0)
            elif _orient == "tilt_x":
                _tilt = np.deg2rad(float(os.environ.get("OBJ_TILT_DEG", "20")))
                _q = _euler_q(_tilt, 0, 0)
            else:
                _q = Gf.Quatf(_cur_rot.GetReal(), *[float(x) for x in _cur_rot.GetImaginary()])

            try:
                _ex = {op.GetOpName(): op for op in UsdGeom.Xformable(_obj).GetOrderedXformOps()}
                if "xformOp:translate" in _ex:
                    _ex["xformOp:translate"].Set(Gf.Vec3d(_px, _py, _pz))
                if "xformOp:orient" in _ex:
                    _qd = Gf.Quatd(_q.GetReal(), Gf.Vec3d(*[float(x) for x in _q.GetImaginary()]))
                    _ex["xformOp:orient"].Set(_qd)
            except Exception as _pe:
                print(f"[grid] note: could not set object pose ({_pe}); anchoring at current pose")
                _px, _py, _pz = _cur_pos[0], _cur_pos[1], _cur_pos[2]

            _jpath = _frz + "/WorldFixedJoint"
            _joint = UsdPhysics.FixedJoint.Define(stage, _jpath)
            _joint.CreateBody1Rel().SetTargets([_frz])
            _joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(_px), float(_py), float(_pz)))
            _joint.CreateLocalRot0Attr().Set(_q)
            _joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            _joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            print(f"[grid] cylinder BOLTED (dynamic) at pose ({_px:.4f},{_py:.4f},{_pz:.4f}) orient={_orient}")
        else:
            print(f"[grid] WARNING: {_frz} not found, cannot bolt.")
    except Exception as e:
        print(f"[grid] WARNING: fixed joint failed: {e}")
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

# Which axis to spin the fingers about. We want to rotate the pad WITHIN its
# own face plane (Paper-2 style in-plane theta). That means rotating about the
# tool's LOCAL approach axis, applied in the TOOL frame (multiply on the RIGHT).
#
# GRASP_ROT_AXIS picks which tool-local axis:
#   "x" -> spin about tool local X
#   "y" -> spin about tool local Y
#   "z" -> spin about tool local Z  (tool approach axis for this gripper)
# We default to "z" (tool approach axis), which is the in-plane spin you want.
# It is easy to test the others to find which one rotates the contact pattern.
ROT_AXIS = os.environ.get("GRASP_ROT_AXIS", "z").lower()

if abs(ROT_DEG) > 1e-6:
    rot_rad = np.deg2rad(ROT_DEG)
    c, s = np.cos(rot_rad/2), np.sin(rot_rad/2)
    if ROT_AXIS == "x":
        spin_local = np.array([c, s, 0.0, 0.0])
    elif ROT_AXIS == "y":
        spin_local = np.array([c, 0.0, s, 0.0])
    else:  # "z" -> tool approach axis
        spin_local = np.array([c, 0.0, 0.0, s])
    # Apply in the TOOL frame: base FIRST, then local spin (multiply on RIGHT).
    tq = quat_mul(tq_base, spin_local)
    print(f"[grid] in-plane spin {ROT_DEG} deg about TOOL-LOCAL {ROT_AXIS.upper()} axis")
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
# ROTATION MACHINERY — ported from your Paper-2 _incremental_execute
# ============================================================
# Pad pivot offset measured in Isaac (probe_tool_frame.py), in the WRIST-LOCAL
# frame, using YOUR convention n=col0(x), u=col1(y), v=col2(z):
#   along n (x): -14.2 mm   along u (y): 62.4 mm   along v (z): 121.4 mm
# Your method builds p_pivot = p_anchor + v_a*L + u_a*W, so:
PAD_L_M = float(os.environ.get("PAD_L_M", "0.1214"))   # along v_a (col2)
PAD_W_M = float(os.environ.get("PAD_W_M", "0.0624"))   # along u_a (col1)
ROT_TEST_DEG = float(os.environ.get("ROT_TEST_DEG", "30.0"))

grasp_world  = np.array([GRASP_X, GRASP_Y, GRASP_Z])

def quat_to_R(qwxyz):
    """wxyz quaternion -> 3x3 rotation matrix (columns are tool axes in world)."""
    return rotmat(qwxyz)

def get_tool0_pose_world():
    """Current wrist/tool0 pose in WORLD: (p[3], R[3x3]).
    fk() returns EE pose in the robot BASE frame, so convert to world."""
    q = robot.get_joint_positions()[ai].astype(np.float32)
    p_base, quat = fk(q)
    R_base = quat_to_R(quat)
    # base -> world:  p_world = R_world_base @ p_base + p_base_origin
    R_wb = rotmat(ROBOT_WORLD_QUAT_WXYZ)
    p_world = R_wb @ p_base + ROBOT_WORLD_POS
    R_world = R_wb @ R_base
    return p_world, R_world

# Find the pad prim once, so we can read its ACTUAL world position each step.
# (Pivot = real pad point -> the pad rotates about itself, no L/W reconstruction.)
_PAD_PRIM_PATH = None
def _find_pad_prim():
    global _PAD_PRIM_PATH
    if _PAD_PRIM_PATH is not None:
        return _PAD_PRIM_PATH
    hits = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PRIM_PATH)):
        p = str(prim.GetPath())
        if p.endswith("TSF_85_right") or "/TSF_85_right" in p:
            hits.append(p)
    _PAD_PRIM_PATH = min(hits, key=len) if hits else None
    return _PAD_PRIM_PATH

def get_pad_point_world():
    """World position of the pad (rotation pivot)."""
    path = _find_pad_prim()
    if path is None:
        # fallback: tool0 position
        p, _ = get_tool0_pose_world()
        return p
    prim = stage.GetPrimAtPath(path)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    m  = xc.GetLocalToWorldTransform(prim)
    t  = m.ExtractTranslation()
    return np.array([t[0], t[1], t[2]])

def numerical_jacobian6(q_arm, eps=1e-4):
    """6x6 geometric Jacobian at q_arm (world frame), via finite differences on FK.
    Rows 0:3 = linear vel of tool0, rows 3:6 = angular vel (axis-angle rate)."""
    q0 = np.asarray(q_arm, float)
    p0, quat0 = fk(q0.astype(np.float32))
    R_wb = rotmat(ROBOT_WORLD_QUAT_WXYZ)
    p0w = R_wb @ p0 + ROBOT_WORLD_POS
    R0w = R_wb @ quat_to_R(quat0)
    J = np.zeros((6, 6))
    for j in range(6):
        dq = q0.copy(); dq[j] += eps
        pj, quatj = fk(dq.astype(np.float32))
        pjw = R_wb @ pj + ROBOT_WORLD_POS
        Rjw = R_wb @ quat_to_R(quatj)
        # linear part
        J[0:3, j] = (pjw - p0w) / eps
        # angular part: dR = Rj @ R0^T -> axis-angle / eps
        dR = Rjw @ R0w.T
        # skew-symmetric -> vector
        w = np.array([dR[2, 1] - dR[1, 2],
                      dR[0, 2] - dR[2, 0],
                      dR[1, 0] - dR[0, 1]]) * 0.5
        J[3:6, j] = w / eps
    return J

def rotate_pad_in_air(th_deg, n_steps=None, record=False, label="ROT"):
    """Rotate the pad about its own normal, pivoting at the ACTUAL pad point.
    Uses the numerical Jacobian + weighted solve for smooth (no-jerk) motion.
    Pivot = real pad world position (measured), so the pad spins in place rather
    than sweeping an arc about a reconstructed L/W point.
    Pad normal = wrist-local X (column 0), confirmed by probe_pad_normal.py."""
    th_req = float(th_deg)
    N = int(np.clip(max(60, abs(th_req) / 0.2), 2, 1000)) if n_steps is None else n_steps
    dth_seg = np.radians(th_req) / N
    Kp_leash = 0.7

    # anchor: pad point + normal at the START (the fixed pivot)
    pad0 = get_pad_point_world()
    _, R_anchor = get_tool0_pose_world()
    n_a = R_anchor[:, 0]; n_a = n_a / (np.linalg.norm(n_a) + 1e-12)

    print(f"[{label}] rotating {th_req:.1f} deg about pad normal at pad point "
          f"{pad0.round(3)}, N={N} steps")

    if record:
        _tsf.set("/exts/TSF_85_Ext/record_active", True)

    for k in range(1, N):
        q_curr = robot.get_joint_positions()[ai].astype(np.float32)
        p_tool, R_now = get_tool0_pose_world()
        n = R_now[:, 0]; n = n / (np.linalg.norm(n) + 1e-12)
        pad_now = get_pad_point_world()

        # spin about the pad normal
        w_slice = dth_seg * n

        # keep the PAD POINT fixed: tool0 origin moves with w x (p_tool - pad_pivot)
        r_vec  = p_tool - pad0
        v_keep = np.cross(w_slice, r_vec)

        # leash: pull the pad back to its fixed anchor (no drift)
        v_correct = -Kp_leash * (pad_now - pad0)

        v_final = v_keep + v_correct

        xi = np.zeros(6)
        xi[:3] = v_final
        xi[3:] = w_slice

        Jk = numerical_jacobian6(q_curr)
        W = np.diag([1.0, 1.0, 1.0, 50.0, 50.0, 50.0])  # match the rotation strongly
        H = Jk.T @ W @ Jk + (0.05 ** 2) * np.eye(6)
        dq = np.linalg.solve(H, Jk.T @ W @ xi)

        q_next = q_curr + dq.astype(np.float32)
        apply_arm_and_grip(q_next)
        world.step(render=True)

    if record:
        _tsf.set("/exts/TSF_85_Ext/record_active", False)
    moved = np.linalg.norm(get_pad_point_world() - pad0) * 1000
    print(f"[{label}] rotation done. pad drifted {moved:.1f} mm from pivot.")


up_world     = grasp_world.copy()
up_world[2] += APPROACH_H
up_base      = world_to_base(up_world)

def do_grasp_and_record(q_start_up, tag):
    """Descend UP->grasp, record close/hold/open, ascend back to UP.
    Records into files prefixed BASENAME_<tag>. Returns q at UP after ascent."""
    print(f"[rot] descent UP->GRASP ({tag}) ...")
    dz_dn = -float(np.linalg.norm(grasp_world - up_world))
    traj_dn = plan_stitched_z(q_start_up, dz_dn, f"{tag}:DOWN")
    if traj_dn is None:
        print(f"[rot] descent FAILED ({tag}).")
        return None
    run_traj(traj_dn)
    q_grasp = robot.get_joint_positions()[ai].copy()
    hold_qg = q_grasp.astype(np.float32)

    hold_for(hold_qg, WAIT_GRASP_SECONDS)

    # The TSF extension locks base_name at startup, so we DON'T change it mid-run.
    # It writes to {BASENAME}_s1/s2_tactile_maps.csv. After recording we copy those
    # to tagged names ({BASENAME}_{tag}_s1/s2_...) so before/after stay separate.
    _tsf.set("/exts/TSF_85_Ext/record_active", True)
    print(f"[rot] [RECORD ON] {tag}: close -> hold -> open ...")
    ramp_gripper(hold_qg, CLOSE_RAD, GRIPPER_RAMP_FRAMES)
    hold_for(hold_qg, WAIT_HOLD_SECONDS)
    ramp_gripper(hold_qg, GRIPPER_OPEN, GRIPPER_RAMP_FRAMES)
    _tsf.set("/exts/TSF_85_Ext/record_active", False)
    print(f"[rot] [RECORD OFF] {tag}")

    # let the extension flush/close its CSVs before we copy them
    for _ in range(90):
        apply_arm_and_grip(hold_qg)
        world.step(render=True)

    import shutil, glob
    for s in ("s1", "s2"):
        src = os.path.join(OUTPUT_DIR, f"{BASENAME}_{s}_tactile_maps.csv")
        dst = os.path.join(OUTPUT_DIR, f"{BASENAME}_{tag}_{s}_tactile_maps.csv")
        try:
            if os.path.exists(src):
                shutil.copyfile(src, dst)
                sz = os.path.getsize(dst)
                print(f"[rot] saved {tag} {s}: {dst} ({sz} bytes)")
            else:
                print(f"[rot] WARNING: expected {src} not found for {tag} {s}")
        except Exception as e:
            print(f"[rot] WARNING: copy failed for {tag} {s}: {e}")

    print(f"[rot] ascent GRASP->UP ({tag}) ...")
    dz_up = float(np.linalg.norm(up_world - grasp_world))
    traj_up2 = plan_stitched_z(q_grasp, dz_up, f"{tag}:UP")
    if traj_up2 is not None:
        run_traj(traj_up2)

    # The TSF extension locks base_name at startup, so it writes to BASENAME_s1/s2.
    # Copy those to tag-specific names so before/after don't overwrite each other.
    import shutil, glob, time as _t
    _t.sleep(0.5)  # let the logger flush
    for s in ("s1", "s2"):
        for kind in ("tactile_maps", "deformations", "mesh_state"):
            src = os.path.join(OUTPUT_DIR, f"{BASENAME}_{s}_{kind}.csv")
            dst = os.path.join(OUTPUT_DIR, f"{BASENAME}_{tag}_{s}_{kind}.csv")
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"[rot] saved {os.path.basename(dst)} ({os.path.getsize(dst)} bytes)")
    return robot.get_joint_positions()[ai].copy()

EXIT_CODE = 0
try:
    q = initial_q.copy()

    print(f"[rot] free move to UP {up_world} ...")
    traj_up = plan_free_move(q, up_base, "to-up")
    if traj_up is None:
        print("[rot] FAILED to reach UP. Exiting.")
        EXIT_CODE = 2
    else:
        run_traj(traj_up)
        q_up = robot.get_joint_positions()[ai].copy()

        # ===== GRASP #1 : BEFORE rotation =====
        print("\n========== GRASP 1 (BEFORE rotation) ==========")
        q_up = do_grasp_and_record(q_up, "before")
        if q_up is None:
            EXIT_CODE = 3
        else:
            hold_for(robot.get_joint_positions()[ai].astype(np.float32), 1.0)

            # ===== ROTATE PAD IN AIR (no contact), 30 deg =====
            print("\n========== ROTATE PAD IN AIR (watch the window) ==========")
            print(">>> LOOK NOW: pad should spin about its OWN point, smoothly (no jerk) <<<")
            rotate_pad_in_air(ROT_TEST_DEG, record=False, label="ROT-AIR")
            for _ in range(60):
                apply_arm_and_grip(robot.get_joint_positions()[ai].astype(np.float32))
                world.step(render=True)
            q_up_rot = robot.get_joint_positions()[ai].copy()

            # ===== GRASP #2 : AFTER rotation =====
            # Re-descend straight down from the (now rotated) UP pose, so the pad
            # keeps its rotated orientation while grasping -> contact should tilt.
            print("\n========== GRASP 2 (AFTER rotation) ==========")
            q_after = do_grasp_and_record(q_up_rot, "after")

            print("\n[rot] SEQUENCE DONE.")
            print(f"[rot] before: {OUTPUT_DIR}/{BASENAME}_before_s1_tactile_maps.csv (+s2)")
            print(f"[rot] after:  {OUTPUT_DIR}/{BASENAME}_after_s1_tactile_maps.csv (+s2)")
finally:
    print("[rot] holding window 8s before close...")
    for _ in range(8 * 60):
        world.step(render=True)
    simulation_app.close()

sys.exit(EXIT_CODE)

