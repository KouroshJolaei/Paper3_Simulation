"""
watch_translate.py — PROVE translation works (no rotation at all).

Does THREE grasps in one visible session, long holds so you can watch:
  GRASP 1: center           -> hold   [LOOK: baseline]
  GRASP 2: moved RIGHT a bit -> hold   [LOOK: pad shifted sideways, still flat]
  GRASP 3: moved DOWN a bit  -> hold   [LOOK: pad shifted down, still flat]

ALL THREE use the SAME orientation (no rotation). If all three make clean flat
contact, translation is proven and the only remaining problem is rotation.

RUN (window opens):
  cd ~/Paper3_Simulation/TSF-85/examples
  WATCH_SHIFT_MM=8 WATCH_HOLD_SEC=25 \
  ~/isaacsim/python.sh ~/Paper3_Simulation/sim/watch_translate.py
"""

import sys
sys.path.insert(0, "/home/kourosh/Paper3_Simulation/curobo-stable/src")
import os

SHIFT = float(os.environ.get("WATCH_SHIFT_MM", "8")) / 1000.0   # meters
HOLD_SEC = float(os.environ.get("WATCH_HOLD_SEC", "25"))

# Center grasp EE point (proven)
GX, GY, GZ = -0.26806, 0.199, 1.24244
APPROACH_H = 0.10
CLOSE_RAD  = 0.55

print(f"[watch] translation-only test, shift={SHIFT*1000:.0f}mm, hold={HOLD_SEC:.0f}s")

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

EXAMPLES_DIR      = "/home/kourosh/Paper3_Simulation/TSF-85/examples"
SCENES_DIR        = os.path.join(EXAMPLES_DIR, "scenes")
USD_PATH          = os.path.join(SCENES_DIR, "scene_cylinder.usd")
CUROBO_ROBOT_YAML = os.path.join(SCENES_DIR, "ur5e.yml")
ROBOT_PRIM_PATH   = "/World/robot_gripper_adapter_sensor"

ARM_JOINT_NAMES   = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
                     "wrist_1_joint","wrist_2_joint","wrist_3_joint"]
GRIPPER_DRIVE_JOINT = "finger_joint"
GRIPPER_OPEN      = 0.0
INITIAL_JOINTS_RAD = np.array([-0.992425,-2.179929,-0.865866,
                                -1.667783,1.570776,-0.992413])
ROBOT_WORLD_POS       = np.array([0.0,-0.3375,0.99275])
ROBOT_WORLD_QUAT_WXYZ = np.array([1.0,0.0,0.0,0.0])
TOOL_DOWN_ROTVEC      = np.array([2.2214,2.2214,0.0])
N_STEPS = 10
CASE13_WEIGHT = [1.0,1.0,1.0,1.0,1.0,0.0]

def rotvec_to_quat(rv):
    a=float(np.linalg.norm(rv))
    if a<1e-9: return np.array([1.,0,0,0])
    ax=rv/a; s=np.sin(a/2)
    return np.array([np.cos(a/2),ax[0]*s,ax[1]*s,ax[2]*s])
def rotmat(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def world_to_base(p):
    return rotmat(ROBOT_WORLD_QUAT_WXYZ).T @ (p-ROBOT_WORLD_POS)

world = World(stage_units_in_meters=1.0, physics_dt=1/120., rendering_dt=1/60., backend="numpy")
pc=world.get_physics_context(); pc.enable_gpu_dynamics(True); pc.set_broadphase_type("GPU")
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=USD_PATH, prim_path=ROBOT_PRIM_PATH)
stage=world.stage
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Scene):
        a=PhysxSchema.PhysxSceneAPI.Apply(prim); a.CreateEnableGPUDynamicsAttr().Set(True)
        try: a.CreateBroadphaseTypeAttr().Set("GPU")
        except Exception: pass

def find_ur5e(s,u,jn):
    rp=s.GetPrimAtPath(u)
    roots=[p for p in Usd.PrimRange(rp) if "PhysicsArticulationRootAPI" in p.GetAppliedSchemas()] if rp.IsValid() else []
    for c in roots:
        pp=c.GetParent() if c.IsA(UsdPhysics.Joint) else c
        for x in Usd.PrimRange(pp):
            if x.IsA(UsdPhysics.Joint):
                n=x.GetName()
                if any(n==j or n.endswith("/"+j) for j in jn): return c
    return None
ar=find_ur5e(stage,ROBOT_PRIM_PATH,ARM_JOINT_NAMES)
AP=str(ar.GetParent().GetPath()) if (ar and ar.IsA(UsdPhysics.Joint)) else (str(ar.GetPath()) if ar else ROBOT_PRIM_PATH)
root_prim=stage.GetPrimAtPath(AP)
if root_prim.IsValid():
    for p in Usd.PrimRange(root_prim):
        if p.GetName()=="base_link":
            xfc=UsdGeom.XformCache(Usd.TimeCode.Default()); xf=xfc.GetLocalToWorldTransform(p)
            t=xf.ExtractTranslation(); q=xf.ExtractRotationQuat()
            ROBOT_WORLD_POS[:]=[t[0],t[1],t[2]]
            ROBOT_WORLD_QUAT_WXYZ[:]=[q.GetReal(),q.GetImaginary()[0],q.GetImaginary()[1],q.GetImaginary()[2]]
            break
robot=SingleArticulation(prim_path=AP,name="ur5e"); world.scene.add(robot); world.reset()

# ============================================================
# FREEZE the cylinder so it can't wobble or fall during grasping.
# Object_02 is the rigid body; setting kinematic=True pins it in place.
# ============================================================
FREEZE_OBJECT = "/World/robot_gripper_adapter_sensor/Object_02"
try:
    _obj = stage.GetPrimAtPath(FREEZE_OBJECT)
    if _obj.IsValid():
        _rb = UsdPhysics.RigidBodyAPI(_obj) if _obj.HasAPI(UsdPhysics.RigidBodyAPI) else UsdPhysics.RigidBodyAPI.Apply(_obj)
        _k = _rb.GetKinematicEnabledAttr() or _rb.CreateKinematicEnabledAttr()
        _k.Set(True)
        print(f"[watch] cylinder FROZEN (kinematic=True): {FREEZE_OBJECT}")
    else:
        print(f"[watch] WARNING: {FREEZE_OBJECT} not found, cannot freeze.")
except Exception as e:
    print(f"[watch] WARNING: freeze failed: {e}")
world.reset()
dn=robot.dof_names
def idxs(dn,bn):
    o=[]
    for nm in bn:
        if nm in dn: o.append(dn.index(nm))
        else:
            c=[d for d in dn if d==nm or d.endswith("/"+nm) or d.endswith(nm)]; o.append(dn.index(c[0]))
    return np.array(o,dtype=np.int32)
ai=idxs(dn,ARM_JOINT_NAMES)
try: gi=np.array([dn.index(GRIPPER_DRIVE_JOINT)],dtype=np.int32)
except ValueError:
    cand=[d for d in dn if d.endswith("/"+GRIPPER_DRIVE_JOINT) or d.endswith(GRIPPER_DRIVE_JOINT)]
    gi=np.array([dn.index(cand[0])],dtype=np.int32) if cand else None
dp=np.array(robot.get_joint_positions(),dtype=np.float32); dp[ai]=INITIAL_JOINTS_RAD
robot.set_joints_default_state(positions=dp)
robot.set_joint_positions(INITIAL_JOINTS_RAD,joint_indices=ai)
robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=INITIAL_JOINTS_RAD,joint_indices=ai))
for _ in range(10): world.step(render=True)
initial_q=robot.get_joint_positions()[ai].copy()

ta=TensorDeviceType()
rc=RobotConfig.from_dict(load_yaml(CUROBO_ROBOT_YAML)["robot_cfg"],ta)
mg=MotionGen(MotionGenConfig.load_from_robot_config(rc,world_model=None,tensor_args=ta,
    interpolation_dt=0.02,num_trajopt_seeds=4,project_pose_to_goal_frame=True,use_cuda_graph=False))
mg.warmup(enable_graph=False,warmup_js_trajopt=False)

tq=rotvec_to_quat(TOOL_DOWN_ROTVEC)   # SAME orientation for all grasps (no rotation)

cur_grip=[0.0]
def apply(armq,grip=None):
    if grip is not None: cur_grip[0]=float(grip)
    robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=armq.astype(np.float32),joint_indices=ai))
    if gi is not None:
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=np.array([cur_grip[0]],dtype=np.float32),joint_indices=gi))
def run(traj,settle=True):
    for q in traj: apply(q); world.step(render=True)
    if settle:
        fc=traj[-1].astype(np.float32)
        for _ in range(120):
            apply(fc); world.step(render=True)
            if np.max(np.abs(robot.get_joint_positions()[ai]-fc))<0.005: break
def fk(q):
    qt=ta.to_device(q.astype(np.float32)).view(1,-1)
    f=mg.compute_kinematics(JointState.from_position(qt,joint_names=ARM_JOINT_NAMES))
    if hasattr(f,"ee_pose") and f.ee_pose is not None:
        return f.ee_pose.position.cpu().numpy().flatten(),f.ee_pose.quaternion.cpu().numpy().flatten()
    return f.ee_position.cpu().numpy().flatten(),f.ee_quaternion.cpu().numpy().flatten()
def plan_free(start_q,target_base,label):
    s=JointState.from_position(ta.to_device(start_q.astype(np.float32)).view(1,-1),joint_names=ARM_JOINT_NAMES)
    g=Pose(position=ta.to_device(target_base.astype(np.float32)).view(1,3),
           quaternion=ta.to_device(tq.astype(np.float32)).view(1,4))
    r=mg.plan_single(s,g,MotionGenPlanConfig(max_attempts=5,enable_graph=False))
    if not r.success.item():
        print(f"  [{label}] FAILED ({r.status})"); return None
    return r.get_interpolated_plan().position.cpu().numpy()
def plan_z(start_q,dz,label):
    metric=PoseCostMetric(hold_partial_pose=True,hold_vec_weight=mg.tensor_args.to_device(np.array(CASE13_WEIGHT,dtype=np.float32)))
    cfg=MotionGenPlanConfig(enable_graph=False,max_attempts=4,enable_finetune_trajopt=False,pose_cost_metric=metric)
    step=dz/N_STEPS; cur=start_q.copy(); st=[]
    for i in range(N_STEPS):
        cpos,cquat=fk(cur); tgt=cpos.copy(); tgt[2]+=step
        s=JointState.from_position(ta.to_device(cur.astype(np.float32)).view(1,-1),joint_names=ARM_JOINT_NAMES)
        g=Pose(position=ta.to_device(tgt.astype(np.float32)).view(1,3),quaternion=ta.to_device(cquat.astype(np.float32)).view(1,4))
        r=mg.plan_single(s,g,cfg)
        if not r.success.item(): print(f"  [{label}] step {i+1} FAIL"); return None
        tr=r.get_interpolated_plan().position.cpu().numpy()
        if st: tr=tr[1:]
        st.extend(list(tr)); cur=tr[-1].copy()
    return np.array(st)
def ramp(armq,target,n):
    cg=float(robot.get_joint_positions()[gi[0]])
    for k in range(n): apply(armq,cg+(k+1)/n*(target-cg)); world.step(render=True)
def hold(armq,sec,msg):
    print("\n"+"="*52); print(f">>> LOOK NOW: {msg} ({sec:.0f}s) <<<"); print("="*52+"\n")
    for _ in range(int(sec*60)): apply(armq); world.step(render=True)

def grasp_at(ee_xyz, tag):
    gw=np.array(ee_xyz); uw=gw.copy(); uw[2]+=APPROACH_H
    tup=plan_free(initial_q, world_to_base(uw), tag+":up")
    if tup is None: return False
    run(tup); q_up=robot.get_joint_positions()[ai].copy()
    dz=-float(np.linalg.norm(gw-uw)); tdn=plan_z(q_up,dz,tag+":down")
    if tdn is None: return False
    run(tdn); q_g=robot.get_joint_positions()[ai].copy(); hq=q_g.astype(np.float32)
    for _ in range(60): apply(hq); world.step(render=True)
    ramp(hq,CLOSE_RAD,60)
    hold(hq,HOLD_SEC,f"{tag}")
    ramp(hq,GRIPPER_OPEN,60)
    dz2=float(np.linalg.norm(uw-gw)); tup2=plan_z(q_g,dz2,tag+":up2")
    if tup2 is not None: run(tup2)
    return True

try:
    # EE axes: from earlier, Y = across (sideways), Z = up/down.
    # GRASP 1: center
    grasp_at([GX, GY, GZ], "GRASP-1 CENTER")
    # GRASP 2: moved RIGHT (in +Y, across)
    grasp_at([GX, GY + SHIFT, GZ], f"GRASP-2 RIGHT (+{SHIFT*1000:.0f}mm Y)")
    # GRASP 3: moved DOWN (in -Z, down the cylinder)
    grasp_at([GX, GY, GZ - SHIFT], f"GRASP-3 DOWN (-{SHIFT*1000:.0f}mm Z)")
    print("\n[watch] done. Close the window when finished looking.")
    for _ in range(600): world.step(render=True)
finally:
    simulation_app.close()
