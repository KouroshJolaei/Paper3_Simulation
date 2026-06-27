"""
data_factory_v2.py  —  Headless tactile data factory for Isaac Sim 5.1 + cuRobo.

FIXES vs v1:
  - Does NOT try to redirect output_dir mid-run (TSF extension locks files at startup)
  - Instead uses a unique base_name per grasp: grasp_001_, grasp_002_, etc.
  - All CSVs land in one flat folder: ~/Paper3_Simulation/Data/run_<timestamp>/
  - Extension is restarted between grasps to pick up the new base_name
  - Skipped grasps (planning failure) are logged but don't crash the factory

HOW TO RUN:
  cd ~/Paper3_Simulation/TSF-85/examples
  ~/isaacsim/python.sh ~/Paper3_Simulation/data_factory_v2.py

OUTPUT STRUCTURE:
  ~/Paper3_Simulation/Data/run_20260626_101221/
    grasp_001_cylinder_center_s1_tactile_maps.csv
    grasp_001_cylinder_center_s1_deformations.csv
    grasp_001_cylinder_center_s1_mesh_state.csv
    grasp_001_cylinder_center_s2_tactile_maps.csv
    ...
    grasp_002_cylinder_left_s1_tactile_maps.csv
    ...
"""

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_BASE_DIR = "/home/kourosh/Paper3_Simulation/Data"

# Five grasp configurations — all world-frame positions (meters)
# The cylinder in the scene sits at approximately x=-0.268, y=0.199, z=1.242
GRASP_CONFIGS = [
    {
        "label":             "cylinder_center",
        "object_pos_world":  [-0.26806, 0.199,  1.24244],
        "approach_height":   0.10,
        "gripper_close_rad": 0.55,
    },
    {
        "label":             "cylinder_left_5cm",
        "object_pos_world":  [-0.31806, 0.199,  1.24244],
        "approach_height":   0.10,
        "gripper_close_rad": 0.55,
    },
    {
        "label":             "cylinder_right_5cm",
        "object_pos_world":  [-0.21806, 0.199,  1.24244],
        "approach_height":   0.10,
        "gripper_close_rad": 0.55,
    },
    {
        "label":             "cylinder_forward_5cm",
        "object_pos_world":  [-0.26806, 0.149,  1.24244],
        "approach_height":   0.10,
        "gripper_close_rad": 0.55,
    },
    {
        "label":             "cylinder_higher_grasp",
        "object_pos_world":  [-0.26806, 0.199,  1.26244],
        "approach_height":   0.12,
        "gripper_close_rad": 0.50,
    },
]

# Physics / timing
GRIPPER_RAMP_FRAMES = 60
WAIT_GRASP_SECONDS  = 1.0
WAIT_HOLD_SECONDS   = 1.0
N_STEPS             = 10
CASE13_WEIGHT       = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]

# ============================================================
# END OF CONFIGURATION
# ============================================================

import sys
sys.path.insert(0, "/home/kourosh/Paper3_Simulation/curobo-stable/src")

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "physics_gpu": 0})

import numpy as np, carb, datetime, os

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
# Paths
# ============================================================
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR      = "/home/kourosh/Paper3_Simulation/TSF-85/examples"
SCENES_DIR        = os.path.join(EXAMPLES_DIR, "scenes")
USD_PATH          = os.path.join(SCENES_DIR, "scene_cylinder.usd")
CUROBO_ROBOT_YAML = os.path.join(SCENES_DIR, "ur5e.yml")
ROBOT_PRIM_PATH   = "/World/robot_gripper_adapter_sensor"
SENSOR_ROOT_RIGHT = f"{ROBOT_PRIM_PATH}/TSF_85_right/TSF_85"
SENSOR_ROOT_LEFT  = f"{ROBOT_PRIM_PATH}/TSF_85_left/TSF_85"
TSF_EXT_SEARCH    = "/home/kourosh/Paper3_Simulation/TSF-85"

RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR   = os.path.join(OUTPUT_BASE_DIR, f"run_{RUN_STAMP}")
os.makedirs(RUN_DIR, exist_ok=True)
print(f"[factory] Output: {RUN_DIR}")

# ============================================================
# Robot config
# ============================================================
ARM_JOINT_NAMES     = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
                        "wrist_1_joint","wrist_2_joint","wrist_3_joint"]
GRIPPER_DRIVE_JOINT = "finger_joint"
GRIPPER_OPEN        = 0.0
INITIAL_JOINTS_RAD  = np.array([-0.992425, -2.179929, -0.865866,
                                  -1.667783,  1.570776, -0.992413])
ROBOT_WORLD_POS       = np.array([0.0, -0.3375, 0.99275])
ROBOT_WORLD_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
TOOL_DOWN_ROTVEC      = np.array([2.2214, 2.2214, 0.0])

# ============================================================
# Helpers
# ============================================================
def rotvec_to_quat(rv):
    a = float(np.linalg.norm(rv))
    if a < 1e-9: return np.array([1.,0,0,0])
    ax = rv/a; s = np.sin(a/2)
    return np.array([np.cos(a/2), ax[0]*s, ax[1]*s, ax[2]*s])

def rotmat(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)]])

def world_to_base(p):
    return rotmat(ROBOT_WORLD_QUAT_WXYZ).T @ (p - ROBOT_WORLD_POS)

# ============================================================
# Build world (once)
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
# Enable TSF_85_Ext ONCE with a placeholder base_name.
# We will update base_name via carb settings before each grasp.
# The extension reads base_name at file-open time (when record_active
# flips to True), so we only need to set it before the gate opens.
# ============================================================
_tsf = carb.settings.get_settings()
_tsf.set("/exts/TSF_85_Ext/headless",      True)
_tsf.set("/exts/TSF_85_Ext/sensor_root",   SENSOR_ROOT_RIGHT)
_tsf.set("/exts/TSF_85_Ext/sensor_root_2", SENSOR_ROOT_LEFT)
_tsf.set("/exts/TSF_85_Ext/output_dir",    RUN_DIR)
_tsf.set("/exts/TSF_85_Ext/base_name",     "grasp_init")
_tsf.set("/exts/TSF_85_Ext/log_dz",        True)
_tsf.set("/exts/TSF_85_Ext/log_pred",      True)
_tsf.set("/exts/TSF_85_Ext/log_mesh",      True)

from omni.kit.app import get_app
_ext_mgr = get_app().get_extension_manager()
_ext_mgr.add_path(TSF_EXT_SEARCH)
_ok = _ext_mgr.set_extension_enabled_immediate("TSF_85_Ext", True)
print(f"[factory] TSF_85_Ext enabled={_ok}")

# ============================================================
# Find robot + auto base
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
            print(f"[factory] base_link world pos: {ROBOT_WORLD_POS}")
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
for _ in range(10): world.step(render=False)
initial_q = robot.get_joint_positions()[ai].copy()

# ============================================================
# cuRobo
# ============================================================
print("[factory] Loading cuRobo...")
ta = TensorDeviceType()
rc = RobotConfig.from_dict(load_yaml(CUROBO_ROBOT_YAML)["robot_cfg"], ta)
mg = MotionGen(MotionGenConfig.load_from_robot_config(
    rc, world_model=None, tensor_args=ta, interpolation_dt=0.02,
    num_trajopt_seeds=4, project_pose_to_goal_frame=True, use_cuda_graph=False))
mg.warmup(enable_graph=False, warmup_js_trajopt=False)
print("[factory] cuRobo ready.")

tq           = rotvec_to_quat(TOOL_DOWN_ROTVEC)
current_grip = [0.0]

def apply_arm_and_grip(arm_q, grip_val=None):
    if grip_val is not None:
        current_grip[0] = float(grip_val)
    robot.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=arm_q.astype(np.float32),
                           joint_indices=ai))
    if gi is not None:
        robot.get_articulation_controller().apply_action(
            ArticulationAction(
                joint_positions=np.array([current_grip[0]], dtype=np.float32),
                joint_indices=gi))

def run_traj(traj, settle=True):
    for q in traj:
        apply_arm_and_grip(q)
        world.step(render=False)
    if settle:
        fc = traj[-1].astype(np.float32)
        for _ in range(120):
            apply_arm_and_grip(fc)
            world.step(render=False)
            if np.max(np.abs(robot.get_joint_positions()[ai] - fc)) < 0.005:
                break

def fk(q):
    qt = ta.to_device(q.astype(np.float32)).view(1, -1)
    f  = mg.compute_kinematics(
            JointState.from_position(qt, joint_names=ARM_JOINT_NAMES))
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
    r = mg.plan_single(s, g, MotionGenPlanConfig(max_attempts=5,
                                                  enable_graph=False))
    if not r.success.item():
        print(f"  [{label}] free move FAILED ({r.status})")
        return None
    return r.get_interpolated_plan().position.cpu().numpy()

def plan_stitched_z(start_q, dz, label):
    metric = PoseCostMetric(
        hold_partial_pose=True,
        hold_vec_weight=mg.tensor_args.to_device(
            np.array(CASE13_WEIGHT, dtype=np.float32)))
    cfg = MotionGenPlanConfig(enable_graph=False, max_attempts=4,
                              enable_finetune_trajopt=False,
                              pose_cost_metric=metric)
    step    = dz / N_STEPS
    cur_q   = start_q.copy()
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
        print(f"  [{label}] step {i+1}/{N_STEPS}: Z={tgt[2]:.4f}")
    return np.array(stitched)

def ramp_gripper(arm_q, target, n_frames):
    cur_g = float(robot.get_joint_positions()[gi[0]])
    for k in range(n_frames):
        alpha = (k + 1) / n_frames
        apply_arm_and_grip(arm_q, grip_val=cur_g + alpha*(target - cur_g))
        world.step(render=False)

def hold_for(arm_q, seconds):
    for _ in range(int(seconds * 60)):
        apply_arm_and_grip(arm_q)
        world.step(render=False)

# ============================================================
# Per-grasp routine
# ============================================================
def run_grasp(cfg, start_q, idx):
    label        = cfg["label"]
    grasp_world  = np.array(cfg["object_pos_world"])
    up_world     = grasp_world.copy()
    up_world[2] += cfg["approach_height"]
    grip_target  = cfg["gripper_close_rad"]
    grasp_prefix = f"grasp_{idx+1:03d}_{label}"

    print(f"\n{'='*55}")
    print(f"[factory] Grasp {idx+1}/{len(GRASP_CONFIGS)}: {label}")
    print(f"  grasp world:  {grasp_world}")
    print(f"  approach:     {up_world}")
    print(f"  output prefix:{grasp_prefix}")

    up_base    = world_to_base(up_world)
    grasp_base = world_to_base(grasp_world)

    # Set the base_name BEFORE flipping record_active ON.
    # The TSF extension reads this when it opens its CSV files,
    # which happens the first time record_active becomes True.
    _tsf.set("/exts/TSF_85_Ext/base_name", grasp_prefix)

    # Free move to UP
    print("  -> free move to UP ...")
    traj_up = plan_free_move(start_q, up_base, f"{label}:to-up")
    if traj_up is None:
        print(f"  SKIP {label}: cannot reach UP point.")
        return start_q, False
    run_traj(traj_up)
    q_up = robot.get_joint_positions()[ai].copy()

    # Straight descent
    print("  -> straight descent UP->GRASP ...")
    dz_dn = -float(np.linalg.norm(grasp_world - up_world))
    traj_dn = plan_stitched_z(q_up, dz_dn, f"{label}:DOWN")
    if traj_dn is None:
        print(f"  SKIP {label}: descent failed.")
        return q_up, False
    run_traj(traj_dn)
    q_grasp = robot.get_joint_positions()[ai].copy()
    hold_qg = q_grasp.astype(np.float32)

    # Settle
    hold_for(hold_qg, WAIT_GRASP_SECONDS)

    # RECORD ON → close → hold → open → RECORD OFF
    _tsf.set("/exts/TSF_85_Ext/record_active", True)
    print("  -> [RECORD ON]  close -> hold -> open ...")
    ramp_gripper(hold_qg, grip_target, GRIPPER_RAMP_FRAMES)
    hold_for(hold_qg, WAIT_HOLD_SECONDS)
    ramp_gripper(hold_qg, GRIPPER_OPEN, GRIPPER_RAMP_FRAMES)
    _tsf.set("/exts/TSF_85_Ext/record_active", False)
    print("  -> [RECORD OFF]")

    # Ascent
    print("  -> straight ascent GRASP->UP ...")
    dz_up = float(np.linalg.norm(up_world - grasp_world))
    traj_up2 = plan_stitched_z(q_grasp, dz_up, f"{label}:UP")
    if traj_up2 is None:
        print(f"  WARNING: ascent failed. Continuing anyway.")
        return q_grasp, True
    run_traj(traj_up2)
    return robot.get_joint_positions()[ai].copy(), True

# ============================================================
# Main loop
# ============================================================
print(f"\n[factory] Starting: {len(GRASP_CONFIGS)} grasps -> {RUN_DIR}\n")

q          = initial_q.copy()
succeeded  = []
failed     = []

for i, cfg in enumerate(GRASP_CONFIGS):
    q, ok = run_grasp(cfg, q, i)
    (succeeded if ok else failed).append(cfg["label"])

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*55}")
print(f"[factory] DONE. Results in: {RUN_DIR}")
print(f"[factory] Succeeded ({len(succeeded)}): {succeeded}")
if failed:
    print(f"[factory] Failed    ({len(failed)}): {failed}")

files = sorted(os.listdir(RUN_DIR))
print(f"\n[factory] Files written ({len(files)}):")
for f in files:
    size = os.path.getsize(os.path.join(RUN_DIR, f))
    print(f"  {f}  ({size//1024} KB)")

simulation_app.close()
