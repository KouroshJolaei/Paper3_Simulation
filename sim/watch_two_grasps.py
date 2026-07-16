"""
watch_two_grasps.py — SEE whether the gripper tilts.

Does TWO grasps in one visible session, with long holds so you can zoom/orbit:
  GRASP 1: straight (no tilt)        -> hold 30s  [LOOK: baseline]
  open, lift
  GRASP 2: tilted about an axis you choose -> hold 30s  [LOOK: did it tilt?]

You pass the tilt angle + which axis to tilt about. The script prints big
>>> LOOK NOW <<< markers when each hold starts.

RUN (window opens):
  cd ~/Paper3_Simulation/TSF-85/examples
  WATCH_TILT_DEG=30 WATCH_TILT_AXIS=approach \
  ~/isaacsim/python.sh ~/Paper3_Simulation/sim/watch_two_grasps.py

WATCH_TILT_AXIS choices:
  approach  -> rotate about the horizontal pad->cylinder axis (should tilt the
               pad's long axis = what we want to SEE)
  vertical  -> rotate about world-Z (the wrist spin that does nothing visible)
  toolx / tooly / toolz -> raw tool-local axes (for comparison)
"""

import sys
sys.path.insert(0, "/home/kourosh/Paper3_Simulation/curobo-stable/src")
import os, time

TILT_DEG  = float(os.environ.get("WATCH_TILT_DEG", "30"))
TILT_AXIS = os.environ.get("WATCH_TILT_AXIS", "approach").lower()
HOLD_SEC  = float(os.environ.get("WATCH_HOLD_SEC", "30"))

# Center grasp (proven working point)
GX, GY, GZ = -0.26806, 0.199, 1.24244
APPROACH_H = 0.10
CLOSE_RAD  = 0.55

print(f"[watch] tilt={TILT_DEG} deg about '{TILT_AXIS}', hold={HOLD_SEC}s each")

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
def quat_mul(a,b):
    w1,x1,y1,z1=a; w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
def axis_quat(axis, deg):
    r=np.deg2rad(deg); c,s=np.cos(r/2),np.sin(r/2)
    ax=np.array(axis,dtype=float); ax=ax/ (np.linalg.norm(ax)+1e-12)
    return np.array([c, ax[0]*s, ax[1]*s, ax[2]*s])
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

tq_base=rotvec_to_quat(TOOL_DOWN_ROTVEC)

# ---- build the tilt quaternion for the chosen axis ----
def make_tilt(axis_name, deg):
    if deg==0: return tq_base
    if axis_name=="vertical":      # world Z, applied on LEFT (world frame)
        return quat_mul(axis_quat([0,0,1],deg), tq_base)
    if axis_name=="approach":
        # approach axis = where the tool points (tool local Z) in WORLD.
        # tool local Z in world = rotmat(tq_base) @ [0,0,1]
        app = rotmat(tq_base) @ np.array([0,0,1.0])
        return quat_mul(axis_quat(app, deg), tq_base)
    if axis_name=="toolx": return quat_mul(tq_base, axis_quat([1,0,0],deg))
    if axis_name=="tooly": return quat_mul(tq_base, axis_quat([0,1,0],deg))
    if axis_name=="toolz": return quat_mul(tq_base, axis_quat([0,0,1],deg))
    # also offer the OTHER world axes for completeness
    if axis_name=="worldx": return quat_mul(axis_quat([1,0,0],deg), tq_base)
    if axis_name=="worldy": return quat_mul(axis_quat([0,1,0],deg), tq_base)
    return tq_base

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
def plan_free(start_q,target_base,tq,label):
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
    print("\n"+"="*50); print(f">>> LOOK NOW: {msg} ({sec:.0f}s) <<<"); print("="*50+"\n")
    for _ in range(int(sec*60)): apply(armq); world.step(render=True)

def do_grasp(tq, tag):
    gw=np.array([GX,GY,GZ]); uw=gw.copy(); uw[2]+=APPROACH_H
    tup=plan_free(initial_q, world_to_base(uw), tq, tag+":up")
    if tup is None: return False
    run(tup); q_up=robot.get_joint_positions()[ai].copy()
    dz=-float(np.linalg.norm(gw-uw))
    tdn=plan_z(q_up,dz,tag+":down")
    if tdn is None: return False
    run(tdn); q_g=robot.get_joint_positions()[ai].copy(); hq=q_g.astype(np.float32)
    for _ in range(60): apply(hq); world.step(render=True)
    ramp(hq,CLOSE_RAD,60)
    hold(hq,HOLD_SEC,f"{tag} — gripper CLOSED on cylinder")
    ramp(hq,GRIPPER_OPEN,60)
    dz2=float(np.linalg.norm(uw-gw)); tup2=plan_z(q_g,dz2,tag+":up2")
    if tup2 is not None: run(tup2)
    return True

try:
    # GRASP 1 — straight
    do_grasp(tq_base, "GRASP-1 STRAIGHT")
    # back to start
    s=JointState.from_position(ta.to_device(robot.get_joint_positions()[ai].astype(np.float32)).view(1,-1),joint_names=ARM_JOINT_NAMES)
    # GRASP 2 — tilted
    tq_tilt=make_tilt(TILT_AXIS, TILT_DEG)
    ok=do_grasp(tq_tilt, f"GRASP-2 TILTED {TILT_DEG}deg about {TILT_AXIS}")
    if not ok:
        print(f"\n[watch] tilted grasp about '{TILT_AXIS}' could NOT be planned.\n")
    print("\n[watch] done. Close the window when finished looking.")
    for _ in range(600): world.step(render=True)
finally:
    simulation_app.close()
