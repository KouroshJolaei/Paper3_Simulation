"""
data_factory.py  —  Headless tactile data factory for Isaac Sim 5.1 + cuRobo.

WHAT IT DOES:
  - Runs Isaac Sim headless (no GUI window)
  - For each grasp configuration (object position + gripper approach angle):
      1. Moves robot to approach position above the object
      2. Descends straight down to grasp position
      3. Closes gripper → holds → opens  (tactile data recorded during this window)
      4. Ascends back up
      5. Saves 6 CSV files per grasp (s1 + s2, deformations + tactile_maps + mesh_state)
  - All output goes to data_generated/run_<timestamp>/

HOW TO RUN (always from terminal, never from PyCharm):
  cd ~/Paper3_Simulation/TSF-85/examples
  ~/isaacsim/python.sh data_factory.py

HOW TO CONFIGURE:
  Edit the GRASP_CONFIGS list below to change object positions and approach angles.
  Each entry is a dict with:
    - object_pos_world:  [x, y, z]  where the object sits on the table (world frame)
    - approach_height:   how high above the object the gripper goes before descending (meters)
    - gripper_close_rad: how far to close the gripper (radians, 0=open, ~0.55=cylinder)
    - label:             name used in the output CSV folder

HOW TO ADD STL OBJECTS (future):
  Place your STL files in:  ~/Paper3_Simulation/TSF-85/examples/objects/
  Use Isaac Sim's asset importer to convert them to USD first, then add their
  USD paths to the GRASP_CONFIGS entries as 'object_usd' (not yet implemented here,
  this version uses the cylinder already in the scene).
"""

# ============================================================
# CONFIGURATION — edit this section to change what gets run
# ============================================================

# Five grasp configurations for the cylinder at different positions/heights
# All positions are in WORLD frame (meters)
GRASP_CONFIGS = [
    {
        "label":            "cylinder_pose_1_center",
        "object_pos_world": [-0.26806, 0.199, 1.24244],   # grasp point (world)
        "approach_height":  0.10,                          # 10 cm above grasp point
        "gripper_close_rad": 0.55,
    },
    {
        "label":            "cylinder_pose_2_left",
        "object_pos_world": [-0.31806, 0.199, 1.24244],   # 5 cm to the left
        "approach_height":  0.10,
        "gripper_close_rad": 0.55,
    },
    {
        "label":            "cylinder_pose_3_right",
        "object_pos_world": [-0.21806, 0.199, 1.24244],   # 5 cm to the right
        "approach_height":  0.10,
        "gripper_close_rad": 0.55,
    },
    {
        "label":            "cylinder_pose_4_forward",
        "object_pos_world": [-0.26806, 0.149, 1.24244],   # 5 cm forward
        "approach_height":  0.10,
        "gripper_close_rad": 0.55,
    },
    {
        "label":            "cylinder_pose_5_higher_grasp",
        "object_pos_world": [-0.26806, 0.199, 1.26244],   # grasp 2 cm higher on object
        "approach_height":  0.12,
        "gripper_close_rad": 0.50,                         # slightly less closed
    },
]

# Physics and timing
GRIPPER_RAMP_FRAMES = 60      # frames to open/close gripper (60 = 0.5s at 120Hz)
WAIT_GRASP_SECONDS  = 1.0     # settle time before closing
WAIT_HOLD_SECONDS   = 1.0     # hold time after closing
N_STEPS             = 10      # straight-line descent steps
CASE13_WEIGHT       = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]  # hold orient+X+Y, free Z

# ============================================================
# END OF CONFIGURATION
# ============================================================

import sys
sys.path.insert(0, "/home/kourosh/Paper3_Simulation/curobo-stable/src")

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "physics_gpu": 0})

import numpy as np, carb, torch, time, datetime, os, csv as csvlib
carb.settings.get_settings().set("/physics/enableDeformableBodies", True)
carb.settings.get_settings().set("/physics/enableGpuDynamics", True)
carb.settings.get_settings().set("/exts/TSF_85_Ext/record_active", False)

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdPhysics, PhysxSchema, Usd, UsdGeom, Sdf

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
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
SCENES_DIR       = os.path.join(SCRIPT_DIR, "scenes")
USD_PATH         = os.path.join(SCENES_DIR, "scene_cylinder.usd")
CUROBO_ROBOT_YAML = os.path.join(SCENES_DIR, "ur5e.yml")
ROBOT_PRIM_PATH  = "/World/robot_gripper_adapter_sensor"
SENSOR_ROOT_RIGHT = f"{ROBOT_PRIM_PATH}/TSF_85_right/TSF_85"
SENSOR_ROOT_LEFT  = f"{ROBOT_PRIM_PATH}/TSF_85_left/TSF_85"

# Output: one timestamped folder per factory run, one subfolder per grasp
RUN_STAMP  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#RUN_DIR    = os.path.join(SCRIPT_DIR, "data_generated", f"run_{RUN_STAMP}")
RUN_DIR = os.path.join("/home/kourosh/Paper3_Simulation/Data", f"run_{RUN_STAMP}")

os.makedirs(RUN_DIR, exist_ok=True)
print(f"[factory] Output folder: {RUN_DIR}")

# ============================================================
# Robot configuration
# ============================================================
ARM_JOINT_NAMES = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
                   "wrist_1_joint","wrist_2_joint","wrist_3_joint"]
GRIPPER_DRIVE_JOINT = "finger_joint"
GRIPPER_OPEN        = 0.0

INITIAL_JOINTS_RAD = np.array([
    -0.992425, -2.179929, -0.865866, -1.667783, 1.570776, -0.992413,
])

ROBOT_WORLD_POS      = np.array([0.0, -0.3375, 0.99275])
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

def base_to_world(p):
    return ROBOT_WORLD_POS + rotmat(ROBOT_WORLD_QUAT_WXYZ) @ p

# ============================================================
# Build Isaac Sim world
# ============================================================
world = World(stage_units_in_meters=1.0, physics_dt=1/120., rendering_dt=1/60., backend="numpy")
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
# Enable TSF_85_Ext (headless, record gate starts OFF)
# ============================================================
_tsf = carb.settings.get_settings()
_tsf.set("/exts/TSF_85_Ext/headless",      True)
_tsf.set("/exts/TSF_85_Ext/sensor_root",   SENSOR_ROOT_RIGHT)
_tsf.set("/exts/TSF_85_Ext/sensor_root_2", SENSOR_ROOT_LEFT)
_tsf.set("/exts/TSF_85_Ext/log_dz",        True)
_tsf.set("/exts/TSF_85_Ext/log_pred",      True)
_tsf.set("/exts/TSF_85_Ext/log_mesh",      True)

from omni.kit.app import get_app
_ext_mgr = get_app().get_extension_manager()
_ext_mgr.add_path(os.path.dirname(SCRIPT_DIR))
_ok = _ext_mgr.set_extension_enabled_immediate("TSF_85_Ext", True)
print(f"[factory] TSF_85_Ext enabled={_ok}")

# ============================================================
# Find robot articulation + auto-detect base pose
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

base_prim = None
root_prim = stage.GetPrimAtPath(AP)
if root_prim.IsValid():
    for p in Usd.PrimRange(root_prim):
        if p.GetName() == "base_link":
            base_prim = p; break
if base_prim is not None:
    xfc = UsdGeom.XformCache(Usd.TimeCode.Default())
    xf  = xfc.GetLocalToWorldTransform(base_prim)
    t   = xf.ExtractTranslation()
    q   = xf.ExtractRotationQuat()
    ROBOT_WORLD_POS[:] = [t[0], t[1], t[2]]
    ROBOT_WORLD_QUAT_WXYZ[:] = [q.GetReal(),
                                  q.GetImaginary()[0],
                                  q.GetImaginary()[1],
                                  q.GetImaginary()[2]]
    print(f"[factory] base_link world pos:  {ROBOT_WORLD_POS}")

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
    cand = [d for d in dn if d.endswith("/"+GRIPPER_DRIVE_JOINT) or d.endswith(GRIPPER_DRIVE_JOINT)]
    gi   = np.array([dn.index(cand[0])], dtype=np.int32) if cand else None

dp = np.array(robot.get_joint_positions(), dtype=np.float32)
dp[ai] = INITIAL_JOINTS_RAD
robot.set_joints_default_state(positions=dp)
robot.set_joint_positions(INITIAL_JOINTS_RAD, joint_indices=ai)
robot.get_articulation_controller().apply_action(
    ArticulationAction(joint_positions=INITIAL_JOINTS_RAD, joint_indices=ai))
for _ in range(10): world.step(render=False)
initial_q = robot.get_joint_positions()[ai].copy()

# ============================================================
# cuRobo motion planner
# ============================================================
print("[factory] Loading cuRobo...")
ta = TensorDeviceType()
rc = RobotConfig.from_dict(load_yaml(CUROBO_ROBOT_YAML)["robot_cfg"], ta)
mg = MotionGen(MotionGenConfig.load_from_robot_config(
    rc, world_model=None, tensor_args=ta, interpolation_dt=0.02,
    num_trajopt_seeds=4, project_pose_to_goal_frame=True, use_cuda_graph=False))
mg.warmup(enable_graph=False, warmup_js_trajopt=False)
print("[factory] cuRobo ready.")

tq = rotvec_to_quat(TOOL_DOWN_ROTVEC)
physics_dt = float(world.get_physics_dt())

# ============================================================
# Motion helpers
# ============================================================
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
    f  = mg.compute_kinematics(JointState.from_position(qt, joint_names=ARM_JOINT_NAMES))
    if hasattr(f, "ee_pose") and f.ee_pose is not None:
        return f.ee_pose.position.cpu().numpy().flatten(), f.ee_pose.quaternion.cpu().numpy().flatten()
    return f.ee_position.cpu().numpy().flatten(), f.ee_quaternion.cpu().numpy().flatten()

def plan_free_move(start_q, target_base, label):
    s = JointState.from_position(
        ta.to_device(start_q.astype(np.float32)).view(1, -1), joint_names=ARM_JOINT_NAMES)
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
                              enable_finetune_trajopt=False,
                              pose_cost_metric=metric)
    step = dz / N_STEPS
    cur_q = start_q.copy()
    stitched = []
    for i in range(N_STEPS):
        cpos, cquat = fk(cur_q)
        tgt = cpos.copy(); tgt[2] += step
        s = JointState.from_position(
            ta.to_device(cur_q.astype(np.float32)).view(1, -1), joint_names=ARM_JOINT_NAMES)
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
        alpha  = (k + 1) / n_frames
        g_val  = cur_g + alpha * (target - cur_g)
        apply_arm_and_grip(arm_q, grip_val=g_val)
        world.step(render=False)

def hold_for(arm_q, seconds):
    n = int(seconds * 60)
    for _ in range(n):
        apply_arm_and_grip(arm_q)
        world.step(render=False)

# ============================================================
# Per-grasp routine
# ============================================================
def run_grasp(cfg, start_q, grasp_index):
    label        = cfg["label"]
    grasp_world  = np.array(cfg["object_pos_world"])
    up_world     = grasp_world.copy()
    up_world[2] += cfg["approach_height"]
    grip_target  = cfg["gripper_close_rad"]

    grasp_dir    = os.path.join(RUN_DIR, f"{grasp_index:03d}_{label}")
    os.makedirs(grasp_dir, exist_ok=True)

    # Point TSF extension output at this grasp's folder
    _tsf.set("/exts/TSF_85_Ext/output_dir", grasp_dir)
    _tsf.set("/exts/TSF_85_Ext/base_name",  "TactileData")

    up_base    = world_to_base(up_world)
    grasp_base = world_to_base(grasp_world)

    print(f"\n{'='*55}")
    print(f"[factory] Grasp {grasp_index+1}/{len(GRASP_CONFIGS)}: {label}")
    print(f"  grasp world: {grasp_world}  approach: {up_world}")

    # Free move to UP point
    print(f"  -> free move to UP ...")
    traj_up = plan_free_move(start_q, up_base, f"{label}:to-up")
    if traj_up is None:
        print(f"  SKIPPING {label}: could not reach UP point.")
        return start_q
    run_traj(traj_up)
    q_up = robot.get_joint_positions()[ai].copy()

    # Straight descent UP -> GRASP
    print(f"  -> straight descent UP->GRASP ...")
    dz = -float(np.linalg.norm(grasp_world - up_world))
    traj_dn = plan_stitched_z(q_up, dz, f"{label}:DOWN")
    if traj_dn is None:
        print(f"  SKIPPING {label}: descent failed.")
        return q_up
    run_traj(traj_dn)
    q_grasp = robot.get_joint_positions()[ai].copy()

    # Settle before close
    hold_qg = q_grasp.astype(np.float32)
    hold_for(hold_qg, WAIT_GRASP_SECONDS)

    # START RECORDING
    carb.settings.get_settings().set("/exts/TSF_85_Ext/record_active", True)
    print(f"  -> [RECORD ON] close -> hold -> open ...")

    ramp_gripper(hold_qg, grip_target, GRIPPER_RAMP_FRAMES)     # close
    hold_for(hold_qg, WAIT_HOLD_SECONDS)                         # hold
    ramp_gripper(hold_qg, GRIPPER_OPEN, GRIPPER_RAMP_FRAMES)    # open

    # STOP RECORDING
    carb.settings.get_settings().set("/exts/TSF_85_Ext/record_active", False)
    print(f"  -> [RECORD OFF]")

    # Straight ascent GRASP -> UP
    print(f"  -> straight ascent GRASP->UP ...")
    dz_up = float(np.linalg.norm(up_world - grasp_world))
    traj_up2 = plan_stitched_z(q_grasp, dz_up, f"{label}:UP")
    if traj_up2 is None:
        print(f"  WARNING: ascent failed for {label}. Continuing anyway.")
        return q_grasp
    run_traj(traj_up2)
    q_final = robot.get_joint_positions()[ai].copy()

    print(f"  -> Done. CSVs in: {grasp_dir}")
    return q_final

# ============================================================
# Main loop
# ============================================================
print(f"\n[factory] Starting data collection: {len(GRASP_CONFIGS)} grasps")
print(f"[factory] Output: {RUN_DIR}\n")

q = initial_q.copy()
for i, cfg in enumerate(GRASP_CONFIGS):
    q = run_grasp(cfg, q, i)

print(f"\n[factory] All done! Data saved to: {RUN_DIR}")
print("[factory] Files per grasp folder:")
for entry in sorted(os.listdir(RUN_DIR)):
    full = os.path.join(RUN_DIR, entry)
    if os.path.isdir(full):
        files = os.listdir(full)
        print(f"  {entry}/  ({len(files)} files: {', '.join(sorted(files))})")

simulation_app.close()
