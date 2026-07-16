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

def plan_stitched_to(start_q, target_base_pos, label, n_steps=None):
    """Straight-line EE move from current position to target_base_pos (in BASE
    frame), in small steps, holding orientation. Forces a DIRECT slide — no
    cuRobo free-plan detour through home. Use for point-to-point moves."""
    metric = PoseCostMetric(hold_partial_pose=True,
        hold_vec_weight=mg.tensor_args.to_device(
            np.array(CASE13_WEIGHT, dtype=np.float32)))
    cfg = MotionGenPlanConfig(enable_graph=False, max_attempts=4,
                              enable_finetune_trajopt=False, pose_cost_metric=metric)
    N = N_STEPS if n_steps is None else n_steps
    cur_q = start_q.copy()
    cpos0, _ = fk(cur_q)
    delta = (np.asarray(target_base_pos, float) - cpos0) / N
    stitched = []
    for i in range(N):
        cpos, cquat = fk(cur_q)
        tgt = cpos + delta
        s = JointState.from_position(
                ta.to_device(cur_q.astype(np.float32)).view(1, -1),
                joint_names=ARM_JOINT_NAMES)
        g = Pose(position=ta.to_device(tgt.astype(np.float32)).view(1, 3),
                 quaternion=ta.to_device(cquat.astype(np.float32)).view(1, 4))
        r = mg.plan_single(s, g, cfg)
        if not r.success.item():
            print(f"  [{label}] step {i+1}/{N} FAILED ({r.status})")
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

# ---- pad frame reader (for verification: position + orientation) ----
_PAD_PATH = None
def _pad_prim_path():
    global _PAD_PATH
    if _PAD_PATH is not None:
        return _PAD_PATH
    hits = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PRIM_PATH)):
        p = str(prim.GetPath())
        if p.endswith("TSF_85_right") or "/TSF_85_right" in p:
            hits.append(p)
    _PAD_PATH = min(hits, key=len) if hits else None
    return _PAD_PATH

def pad_pose_world():
    """Return (pos[3], R[3x3]) of the pad frame in WORLD. R columns = pad axes."""
    path = _pad_prim_path()
    prim = stage.GetPrimAtPath(path)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    m  = xc.GetLocalToWorldTransform(prim)
    t  = m.ExtractTranslation()
    R  = np.array(m.ExtractRotationMatrix()).T
    return np.array([t[0], t[1], t[2]]), R

def solve_ik(target_world, seed_q, label):
    """One cuRobo IK call: joint angles that place the tool at target_world
    (approach-height pose), orientation held = tq. No path planning."""
    tb = world_to_base(target_world)
    s = JointState.from_position(
            ta.to_device(seed_q.astype(np.float32)).view(1, -1),
            joint_names=ARM_JOINT_NAMES)
    g = Pose(position=ta.to_device(tb.astype(np.float32)).view(1, 3),
             quaternion=ta.to_device(tq.astype(np.float32)).view(1, 4))
    r = mg.plan_single(s, g, MotionGenPlanConfig(max_attempts=5, enable_graph=False))
    if not r.success.item():
        print(f"  [{label}] IK FAILED ({r.status})")
        return None
    # take the final joint config of the plan = IK solution at the target
    return r.get_interpolated_plan().position.cpu().numpy()[-1].copy()

def move_joint_line(q_from, q_to, label, max_step=0.01, log=None):
    """Move the arm in a STRAIGHT JOINT-SPACE line from q_from to q_to.
    Direct interpolation, applied step by step. Cannot detour or flail —
    it's the shortest path in joint space. Used for between-points moves.
    If log is a dict, records tool0 world pose at each step (for verification)."""
    q_from = np.asarray(q_from, float); q_to = np.asarray(q_to, float)
    dq = q_to - q_from
    N = int(np.clip(np.ceil(np.max(np.abs(dq)) / max_step), 2, 400))
    print(f"[sess] {label}: joint-line move in {N} steps "
          f"(max joint delta {np.rad2deg(np.max(np.abs(dq))):.1f} deg)")
    if log is not None:
        p0, _ = fk(q_from.astype(np.float32))
        log["q_from"] = q_from.tolist()
        log["q_to"]   = q_to.tolist()
        log["ee_from_base"] = p0.tolist()
        log["ee_path_base"] = [p0.tolist()]
    for i in range(1, N + 1):
        qi = q_from + dq * (i / N)
        apply_arm_and_grip(qi.astype(np.float32))
        world.step(render=True)
        if log is not None:
            pcur, _ = fk(robot.get_joint_positions()[ai].astype(np.float32))
            log["ee_path_base"].append(pcur.tolist())
    # settle
    for _ in range(30):
        apply_arm_and_grip(q_to.astype(np.float32))
        world.step(render=True)
    q_reached = robot.get_joint_positions()[ai].copy()
    if log is not None:
        p_final, _ = fk(q_reached.astype(np.float32))
        p_des,   _ = fk(q_to.astype(np.float32))
        log["ee_desired_base"] = p_des.tolist()
        log["ee_actual_base"]  = p_final.tolist()
        log["q_reached"] = q_reached.tolist()
    return q_reached

# ============================================================
# CONTINUOUS POSE-TO-POSE SESSION (Stage 1 minimal: 2 grasps)
# One Isaac session. Grasp at point 1 -> record -> release -> move RELATIVELY
# to point 2 (no home reset) -> grasp -> record. The arm stays near the object
# the whole time; it never returns to the initial/home pose between grasps.
# ============================================================

# Two test points: the proven grasp point, and one shifted +8mm in Y.
# (Stage 2 will feed the full grid here instead of these two.)
P1_world = np.array([GRASP_X, GRASP_Y,         GRASP_Z])
P2_world = np.array([GRASP_X, GRASP_Y + 0.008, GRASP_Z])   # +8mm in Y
APPROACH = APPROACH_H

def go_to_up_of(point_world, from_q, label, straight=False):
    """Move to APPROACH height above point.
    straight=False: cuRobo free-plan (fine for the first big move from home).
    straight=True : straight-line slide from current pos (direct, no detour)."""
    up = point_world.copy(); up[2] += APPROACH
    up_b = world_to_base(up)
    if straight:
        traj = plan_stitched_to(from_q, up_b, f"{label}:slide")
    else:
        traj = plan_free_move(from_q, up_b, f"{label}:to-up")
    if traj is None:
        print(f"[sess] {label}: move to UP FAILED")
        return None
    run_traj(traj)
    return robot.get_joint_positions()[ai].copy()

def grasp_at(point_world, tag, row_marks):
    """Descend from current UP -> grasp -> record (close/hold/open) -> ascend.
    Slices ONLY this grasp's new rows into BASENAME_<tag>_s1/s2. Returns q at UP.
    row_marks: dict tracking how many rows were in each file before this grasp."""
    q_up = robot.get_joint_positions()[ai].copy()
    grasp = point_world.copy()
    up    = point_world.copy(); up[2] += APPROACH

    print(f"[sess] {tag}: descent UP->GRASP ...")
    dz_dn = -float(np.linalg.norm(grasp - up))
    traj_dn = plan_stitched_z(q_up, dz_dn, f"{tag}:DOWN")
    if traj_dn is None:
        print(f"[sess] {tag}: descent FAILED")
        return None
    run_traj(traj_dn)
    q_grasp = robot.get_joint_positions()[ai].copy()
    hold_qg = q_grasp.astype(np.float32)

    hold_for(hold_qg, WAIT_GRASP_SECONDS)

    _tsf.set("/exts/TSF_85_Ext/record_active", True)
    print(f"[sess] {tag}: [RECORD ON] close -> hold -> open ...")
    ramp_gripper(hold_qg, CLOSE_RAD, GRIPPER_RAMP_FRAMES)
    hold_for(hold_qg, WAIT_HOLD_SECONDS)
    ramp_gripper(hold_qg, GRIPPER_OPEN, GRIPPER_RAMP_FRAMES)
    _tsf.set("/exts/TSF_85_Ext/record_active", False)
    print(f"[sess] {tag}: [RECORD OFF]")

    print(f"[sess] {tag}: ascent GRASP->UP ...")
    dz_up = float(np.linalg.norm(up - grasp))
    traj_up2 = plan_stitched_z(q_grasp, dz_up, f"{tag}:UP")
    if traj_up2 is not None:
        run_traj(traj_up2)

    # The TSF extension APPENDS every grasp to BASENAME_s1/s2. Slice out only
    # the rows added during THIS grasp (from the previous mark to end).
    import time as _t, csv
    _t.sleep(0.6)
    for s in ("s1", "s2"):
        src = os.path.join(OUTPUT_DIR, f"{BASENAME}_{s}_tactile_maps.csv")
        dst = os.path.join(OUTPUT_DIR, f"{BASENAME}_{tag}_{s}_tactile_maps.csv")
        if not os.path.exists(src):
            continue
        with open(src) as f:
            lines = f.readlines()
        header, body = lines[0], lines[1:]
        prev = row_marks.get(s, 0)
        new_body = body[prev:]
        row_marks[s] = len(body)          # update mark to current end
        with open(dst, "w") as f:
            f.write(header); f.writelines(new_body)
        print(f"[sess] saved {os.path.basename(dst)} ({len(new_body)} new rows)")
    return robot.get_joint_positions()[ai].copy()

EXIT_CODE = 0
row_marks = {}   # tracks tactile rows written before each grasp (for clean slicing)

# ---- full-path pad logger: sample pad pose whenever we step the sim ----
PAD_PATH_LOG = {"active": False, "pts": []}
def _sample_pad():
    if PAD_PATH_LOG["active"]:
        p, R = pad_pose_world()
        PAD_PATH_LOG["pts"].append(p.tolist())

# wrap world.step so every simulation step samples the pad position
_orig_step = world.step
def _logged_step(*a, **k):
    _orig_step(*a, **k)
    _sample_pad()
world.step = _logged_step

try:
    q = initial_q.copy()

    # ---- approach point 1 (from home): cuRobo free-plan is fine here ----
    print("\n========== POINT 1 ==========")
    q_up = go_to_up_of(P1_world, q, "P1")
    if q_up is None:
        EXIT_CODE = 2
    else:
        # grasp 1
        q_up = grasp_at(P1_world, "p1", row_marks)
        if q_up is None:
            EXIT_CODE = 3
        else:
            # capture pad frame at START (P1 up, after grasp 1)
            pad_p1, pad_R1 = pad_pose_world()
            print(f"[DIAG] after grasp1 ascent, pad at: {pad_p1.round(4)}")

            # ---- RELATIVE move to point 2 ----
            print("\n========== RELATIVE MOVE P1 -> P2 ==========")
            print(">>> LOGGING FULL PATH so we can SEE any home detour <<<")
            # start logging the pad path through the ENTIRE move
            PAD_PATH_LOG["active"] = True
            PAD_PATH_LOG["pts"] = []

            # 1) IK: joint angles for P2's up-pose (seeded from current pose)
            up2 = P2_world.copy(); up2[2] += APPROACH
            q_up2_target = solve_ik(up2, q_up, "P1->P2:IK")
            if q_up2_target is None:
                EXIT_CODE = 4
            else:
                # DIAG: how big is the joint gap the IK wants us to cross?
                q_now = robot.get_joint_positions()[ai].copy()
                gap = np.rad2deg(np.abs(np.asarray(q_up2_target) - q_now))
                print(f"[DIAG] IK joint gap (deg per joint): {gap.round(1)}")
                print(f"[DIAG] IK max joint gap: {gap.max():.1f} deg")

                # 2) straight joint-space interpolation to it
                move_log = {}
                q_up2 = move_joint_line(q_up, q_up2_target, "P1->P2", log=move_log)

                # capture pad frame at END actual (P2 up)
                pad_p2, pad_R2 = pad_pose_world()
                PAD_PATH_LOG["active"] = False
                print(f"[DIAG] after joint-line, pad at: {pad_p2.round(4)}")
                print(f"[DIAG] pad moved {np.linalg.norm(pad_p2-pad_p1)*1000:.1f} mm "
                      f"(intended ~8mm). path pts: {len(PAD_PATH_LOG['pts'])}")

                # desired pad pose at P2 up = P1 pad pose shifted by the intended move
                intended_shift = (P2_world - P1_world)   # world-frame shift
                pad_p2_desired = pad_p1 + intended_shift

                # save the full verification record
                import json as _json
                move_log["P1_world"] = P1_world.tolist()
                move_log["P2_world"] = P2_world.tolist()
                move_log["robot_base_world"] = ROBOT_WORLD_POS.tolist()
                move_log["pad_start_pos"]    = pad_p1.tolist()
                move_log["pad_start_R"]      = pad_R1.tolist()
                move_log["pad_end_actual_pos"] = pad_p2.tolist()
                move_log["pad_end_actual_R"]   = pad_R2.tolist()
                move_log["pad_end_desired_pos"] = pad_p2_desired.tolist()
                move_log["pad_end_desired_R"]   = pad_R1.tolist()  # orientation unchanged (parallel move)
                move_log["pad_full_path"]    = PAD_PATH_LOG["pts"]  # EVERY step's pad position
                with open(os.path.join(OUTPUT_DIR, "move_p1_to_p2.json"), "w") as f:
                    _json.dump(move_log, f, indent=2)
                print(f"[sess] saved move_p1_to_p2.json "
                      f"({len(PAD_PATH_LOG['pts'])} path points logged)")

                # grasp 2
                print("\n========== POINT 2 ==========")
                grasp_at(P2_world, "p2", row_marks)
                print("\n[sess] CONTINUOUS 2-GRASP SESSION DONE.")
finally:
    print("[sess] holding window 6s before close...")
    for _ in range(6 * 60):
        world.step(render=True)
    simulation_app.close()

sys.exit(EXIT_CODE)

