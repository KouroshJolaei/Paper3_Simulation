"""
collect_from_config.py — Runs a FULL grid of grasps from a GUI config JSON.

Reads gui_config.json (written by main_gui.py): object pose + tilt, pad offset,
and a list of grasp points (each = pad Y-Z offset from object centre, mm).
For EACH point it does the PROVEN grasp: approach -> descend -> close/record/open
-> ascend, saving tactile per point. This reuses the reliable single-grasp
machinery; the arm lifts to approach height between points (safe, not the fastest).

LAUNCHED BY: main_gui.py  (Save + Run Simulation), or manually:
  cd ~/Paper3_Simulation/TSF-85/examples
  ~/isaacsim/python.sh ~/Paper3_Simulation/sim/collect_from_config.py --config <path>
"""

import sys
sys.path.insert(0, "/home/kourosh/Paper3_Simulation/curobo-stable/src")

import os, json, argparse
import csv as _csv

class _CalNoContact(Exception):
    """Raised to abort a calibrate store when the pads didn't touch the object."""
    pass

# ============================================================
# Read the GUI config (all grid points)
# ============================================================
_ap = argparse.ArgumentParser()
_ap.add_argument("--config", default=os.path.expanduser("~/Paper3_Simulation/Data/gui_config.json"))
_args, _ = _ap.parse_known_args()
with open(_args.config) as _f:
    CONFIG = json.load(_f)

# object centre (world, mm -> m)
_obj_mm = CONFIG["object"]["center_world_mm"]
OBJ_CENTER = [_obj_mm[0]/1000.0, _obj_mm[1]/1000.0, _obj_mm[2]/1000.0]
OBJ_TILT_DEG  = float(CONFIG["object"].get("tilt_deg", 0.0))
OBJ_TILT_AXIS = CONFIG["object"].get("tilt_axis", "X")

# ============================================================
# TOOL_OFFSET_Z: per-object calibration store  (RESTORED)
# ------------------------------------------------------------
# The EE is commanded TOOL_OFFSET_Z above the desired pad Z. That distance
# depends on the object diameter (the fingers swing the pad down as they close),
# so we key a calibration store by diameter and REFUSE to run an uncalibrated
# object rather than silently using a wrong number.
#
#   TOOL_OFFSET_Z(D) = (EE_z - closed_inner_finger_z) + C_ANCHOR
#
# C_ANCHOR = fixed pad-face-below-inner-finger-link distance (rigid geometry,
# object independent), anchored on the verified 26 mm cylinder (offset 0.158).
#
# Modes:
#   GRASP_CALIBRATE=1     -> ONE grasp, MEASURE + STORE this diameter's offset.
#   (normal run)          -> look up this diameter; use it, or REFUSE if absent.
#   GRASP_TOOL_OFFSET=<m> -> manual override (dev escape hatch), skips the store.
# ============================================================
C_ANCHOR          = 0.1332   # m, pad face below inner-finger link

# The sensor 'Case' prim ORIGIN sits at the pad's END, not its centre. Measured
# from the pad node cloud: the pad's up-the-rod length is ~41mm and the Case
# origin is 22.3mm from the pad CENTRE along that axis. Calibration targets the
# pad CENTRE (what the GUI draws), so it shifts the raw Case offset up by this.
# Verify once and trim this ONE number if the pad centre is still off by X mm.


# PAD_CENTER_ABOVE_CASE_M = 0.0223   # m, Case origin -> pad centre, along world Z
# PAD_CENTER_ABOVE_CASE_M = 0.0293
# PAD_CENTER_ABOVE_CASE_M = 0.0329   # measured via diag9 rim-mask census, 2026-07-22
#                                    # (was 0.0223 from node cloud -> pad ran 10.6 mm low;
#                                    #  0.0293 ghost deleted - also wrong by 3.6 mm)


PAD_CENTER_ABOVE_CASE_M = 0.0221   # MEASURED from the extension's own mesh log
                                   # (diag13, 2026-07-23): case origin -> sensing
                                   # array centre = 22.10 mm. Confirms the original
                                   # 0.0223. The 0.0329 census inference was wrong.





CAL_DEFAULT_OFFST = 0.15     # m, provisional offset used ONLY during calibration
CAL_FILE = os.path.expanduser("~/Paper3_Simulation/Data/pad_offset_calibration.json")
OBJ_DIAM_MM = float(CONFIG["object"].get("diameter_mm", 0.0))
CALIBRATE   = os.environ.get("GRASP_CALIBRATE", "0") == "1"

def _load_cal():
    try:
        with open(CAL_FILE) as _cf:
            return json.load(_cf)
    except Exception:
        return {}

_CAL = _load_cal()
_diam_key = f"{OBJ_DIAM_MM:.1f}"

if os.environ.get("GRASP_TOOL_OFFSET"):            # manual override (dev)
    TOOL_OFFSET_Z = float(os.environ["GRASP_TOOL_OFFSET"])
    print(f"[cal] MANUAL override TOOL_OFFSET_Z = {TOOL_OFFSET_Z}")
elif CALIBRATE:
    TOOL_OFFSET_Z = CAL_DEFAULT_OFFST              # provisional; solved after grasp
    print(f"[cal] CALIBRATE mode: provisional TOOL_OFFSET_Z = {TOOL_OFFSET_Z}, "
          f"diameter {OBJ_DIAM_MM} mm")
    _cz_mm = 0.0
    try:
        _cz_mm = float(CONFIG["points"][0].get("pad_offset_z_mm", 0.0))
    except Exception:
        _cz_mm = 0.0
    # ONE grasp: Y centered (so the pads meet the true diameter), Z as chosen
    CONFIG["points"] = [{"index": 0, "pad_offset_y_mm": 0.0, "pad_offset_z_mm": _cz_mm}]
    print(f"[cal] calibrate grasp: Y=0 (centered), Z offset={_cz_mm:+.1f} mm")
elif _diam_key in _CAL:
    TOOL_OFFSET_Z = float(_CAL[_diam_key]["TOOL_OFFSET_Z"])
    print(f"[cal] using calibrated TOOL_OFFSET_Z = {TOOL_OFFSET_Z} for {OBJ_DIAM_MM} mm")
else:
    print(f"\n[cal] REFUSING TO RUN: object diameter {OBJ_DIAM_MM} mm is NOT calibrated.")
    print(f"[cal] Calibrate it first (Calibrate tab, or GRASP_CALIBRATE=1), then collect.")
    print(f"[cal] Calibration file: {CAL_FILE}\n")
    sys.exit(2)


APPROACH_H    = 0.10

# CLOSE_RAD is the finger-joint angle we squeeze to. It is DIAMETER
# DEPENDENT: 0.55 rad closes to ~26 mm, so on a 50 mm object the pads would
# never meet it and on a 13 mm one they would crush it.
#
# The calibration store already RECORDS close_rad per diameter (it is written
# at the end of a calibrate run) -- it was simply never read back. Now it is,
# with the same discipline as TOOL_OFFSET_Z: use the stored value, or fall
# back to the anchored 26 mm value only while CALIBRATING a new diameter.
#   GRASP_CLOSE_RAD=<rad>  -> manual override (needed for the FIRST grasp on
#                             a new diameter, before anything is stored).
CLOSE_RAD_DEFAULT = 0.55                     # the verified 26 mm value
if os.environ.get("GRASP_CLOSE_RAD"):
    CLOSE_RAD = float(os.environ["GRASP_CLOSE_RAD"])
    print(f"[cal] MANUAL override CLOSE_RAD = {CLOSE_RAD:.4f} rad")
elif _diam_key in _CAL and "close_rad" in _CAL[_diam_key]:
    CLOSE_RAD = float(_CAL[_diam_key]["close_rad"])
    print(f"[cal] using calibrated CLOSE_RAD = {CLOSE_RAD:.4f} rad "
          f"for {OBJ_DIAM_MM} mm")
else:
    CLOSE_RAD = CLOSE_RAD_DEFAULT
    if abs(OBJ_DIAM_MM - 26.0) > 0.5:
        print(f"[cal] WARNING: no stored close_rad for {OBJ_DIAM_MM} mm; "
              f"using the 26 mm value {CLOSE_RAD:.4f} rad. Estimate for this "
              f"diameter: {max(0.05, (85.0 - OBJ_DIAM_MM) / 106.0):.3f} rad "
              f"(set GRASP_CLOSE_RAD to use it).")

# ---- PROBLEM 1: stop lifting APPROACH_H between neighbouring grid points ----
# Old behaviour (Berith's single-grasp routine, copied per point): for EVERY
# point the arm rose to grasp+APPROACH_H via a GLOBAL replan (plan_free_move),
# then descended APPROACH_H again. Between two points 8 mm apart that is a
# 100 mm up / 8 mm over / 100 mm down round trip, and the global replan is free
# to pick a different arm branch each time -> the "flies back to home" + jerk.
#
# New: after the first point, move PAD-TO-PAD in a straight line at grasp
# height (gripper is OPEN there: pads ~56 mm apart vs a 26 mm object = ~15 mm
# clearance per side, and the grid moves only in Y/Z, so X clearance is never
# reduced -> collision-free for this scene).
#   POINT_TO_POINT = False  -> exactly the old behaviour (safe fallback).
# The first point still uses the old approach (we start far away at home), the
# last point still retreats, and any failed direct move falls back to the old
# path automatically, so this can only save motion, never strand the arm.
POINT_TO_POINT = True
LINE_STEPS     = 10
# hold_vec_weight for a straight-line move: hold the 3 tool ORIENTATION dofs,
# free the 3 translations (the small steps are what enforce the straight line).
# Compare CASE13_WEIGHT = [1,1,1,1,1,0], which freed ONLY z for the descent.
LINE_WEIGHT    = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

# ============================================================
# REACHABILITY PRE-CHECK   (ported from Paper 2 _evaluate_reachability)
# ------------------------------------------------------------
# Before ANY motion, every grid point is dry-run: solve IK at its EE target, then
# simulate the incremental trajectory q_home -> q_goal and enforce the same gates
# the executor would. NOTHING MOVES. Results -> reachability_report.json;
# unreachable points are SKIPPED and logged, never fatal.
#
# Paper-2 knobs kept verbatim:
MAX_JOINT_STEP = 0.02      # rad, sizes the interpolated trajectory
MAX_STEPS_CAP  = 500       # max interpolation steps
COND_MAX_WARN  = 1e3       # reject if cond(J) exceeds this (near-singular)
POS_TOL_M      = 1e-3      # 1 mm  final FK position tolerance
ROT_TOL_DEG    = 0.5       # 0.5 deg final FK orientation tolerance

# Paper-2 MANUAL_LIMITS, defaults PERMISSIVE on purpose: those gates existed to
# shape a LOCAL regrasp search (3 mm nudges). A grid scan must let the arm travel,
# so tight delta_bounds/one_sided would reject perfectly good points. cuRobo
# already enforces the real UR5e joint limits from ur5e.yml, so abs_limits=None.
# cond(J) IS kept: cuRobo will happily plan through a near-singular pose.
# Turn any gate on here (no GUI needed) if you ever want it.
MANUAL_LIMITS = {
    "frozen_joints":   [],      # e.g. [5] to freeze wrist_3
    "one_sided":       {},      # e.g. {1: "neg"}
    "delta_bounds":    {},      # e.g. {0: (-0.03, 0.03)}
    "per_iter_dq_cap": None,    # e.g. 0.03 rad global per-step cap
    "abs_limits":      None,    # e.g. [(qmin,qmax)]*6  (None -> cuRobo's own)
}
REACH_ONLY  = os.environ.get("GRASP_REACH_ONLY", "0") == "1"  # check + exit, no motion
REACH_CHECK = os.environ.get("GRASP_REACH_CHECK", "1") == "1" # auto-check before a run
REACH_SKIP  = os.environ.get("GRASP_REACH_SKIP", "1") == "1"  # skip unreachable points
# Jacobian conditioning + dry-run convergence are ADVISORY by default: they
# describe a straight-line path the executor never drives. See
# evaluate_reachability. Set 1 for the old veto behaviour.
REACH_STRICT = os.environ.get("GRASP_REACH_STRICT", "0") == "1"

# where to write data — fresh timestamped folder per run (no old data mixing in)
import datetime as _dt
_stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
_base_out = os.environ.get("GRASP_OUTPUT_DIR",
                           os.path.expanduser("~/Paper3_Simulation/Data/gui_run"))
OUTPUT_DIR = os.path.join(_base_out, f"run_{_stamp}")
BASENAME   = os.environ.get("GRASP_BASENAME", "gui")

# ---- self-describing folder name (added 2026-08-04) ------------------------
# Timestamps alone made it impossible to tell a 0 deg run from a 20 deg one
# without opening gui_config_used.json. Append the two angles that define the
# geometry of the test: obj<tilt> = cylinder tilt, pad<roll> = in-plane pad
# roll (GRASP_ROT_DEG). Negatives become 'm' so the name stays shell-safe:
#     run_20260804_160312_obj0_pad20
#     run_20260804_161540_obj35_padm10
def _ang_tag(v):
    v = float(v)
    s = f"{abs(v):g}".replace(".", "p")
    return ("m" + s) if v < 0 else s

try:
    _pad_roll = float(os.environ.get("GRASP_ROT_DEG", "0.0"))
except ValueError:
    _pad_roll = 0.0

# GRASP_RUN_DIR (set by the GUI's SESSION FOLDER) wins outright: the GUI has
# already picked ONE folder for this test, and both the reachability dry-run
# and the real run point at it, so a single test no longer scatters itself
# across two timestamped folders. Without it, fall back to the old behaviour.
_forced_dir = os.environ.get("GRASP_RUN_DIR", "").strip()
if _forced_dir:
    OUTPUT_DIR = os.path.expanduser(_forced_dir)
else:
    OUTPUT_DIR = (f"{OUTPUT_DIR}_obj{_ang_tag(OBJ_TILT_DEG)}"
                  f"_pad{_ang_tag(_pad_roll)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- PHASE LOGGER (file = ground truth; Isaac floods/hides stdout) ----------
# Appends one timestamped line per phase to run_progress.log. Whatever the LAST
# line says is exactly where the run died (move vs descent vs close vs ascent).
def _progress(msg):
    try:
        line = f"{_dt.datetime.now().strftime('%H:%M:%S.%f')[:-3]}  {msg}\n"
        with open(os.path.join(OUTPUT_DIR, "run_progress.log"), "a") as _pf:
            _pf.write(line); _pf.flush()
    except Exception:
        pass
    print(f"[progress] {msg}", flush=True)

# build the list of grasp WORLD targets (EE targets) from pad Y-Z offsets
GRID_POINTS = []
for pt in CONFIG["points"]:
    dy = pt["pad_offset_y_mm"] / 1000.0
    dz = pt["pad_offset_z_mm"] / 1000.0
    gx = OBJ_CENTER[0]                      # X centered (two pads squeeze here)
    gy = OBJ_CENTER[1] + dy                 # pad Y offset
    gz = OBJ_CENTER[2] + dz + TOOL_OFFSET_Z # pad Z offset + tool offset to EE
    GRID_POINTS.append({"index": pt["index"], "world": [gx, gy, gz],
                        "dy_mm": pt["pad_offset_y_mm"], "dz_mm": pt["pad_offset_z_mm"]})

print(f"[cfg] loaded {len(GRID_POINTS)} grid points from {_args.config}")
print(f"[cfg] object center (m): {OBJ_CENTER}  tilt {OBJ_TILT_DEG} about {OBJ_TILT_AXIS}")
print(f"[cfg] output: {OUTPUT_DIR}  basename: {BASENAME}")

# ============================================================
# Launch Isaac Sim (window visible unless GRASP_HEADLESS=1)
# ============================================================
_HEADLESS = os.environ.get("GRASP_HEADLESS", "0") == "1"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": _HEADLESS, "physics_gpu": 0})
print(f"[cfg] headless={_HEADLESS} (window {'hidden' if _HEADLESS else 'visible'})")

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

# ---- SYMMETRIC CLOSE (fixes the ~2 deg gripper tilt / s1>>s2) ---------------
# The 2F-85 is TWO independent 4-bar linkages (finger_joint drives the LEFT
# knuckle, right_outer_knuckle_joint drives the RIGHT). We were commanding ONLY
# finger_joint, so the right side followed passively through a compliant PhysX
# closed loop and settled at a DIFFERENT angle. Measured at closed grip:
#     |left_inner_finger_joint| = 0.53112   vs  |right_inner_finger_joint| = 0.51510
#     -> 0.92 deg apart -> the two fingers sit 1.93 mm apart in Z (2.09 deg tilt)
#     -> the right pad ends up 3.23 mm CLOSER to the rod than the left
#     -> s1 crushes (20023) while s2 barely touches (276).
# Driving BOTH sides to the same angle forces them to close symmetrically.
# Set False to go back to single-joint drive.
DRIVE_BOTH_FINGERS   = True
GRIPPER_MIRROR_JOINT = "right_outer_knuckle_joint"
GRIPPER_OPEN        = 0.0
INITIAL_JOINTS_RAD  = np.array([-0.992425, -2.179929, -0.865866,
                                  -1.667783,  1.570776, -0.992413])
ROBOT_WORLD_POS       = np.array([0.0, -0.3375, 0.99275])
ROBOT_WORLD_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
TOOL_DOWN_ROTVEC      = np.array([2.2214, 2.2214, 0.0])

GRIPPER_RAMP_FRAMES = 60
WAIT_GRASP_SECONDS  = 1.0
# WAIT_HOLD_SECONDS   = 1.0
WAIT_HOLD_SECONDS   = 3.5   # >3s so the +3s temporal snapshot lands during the hold, not after release
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

# ============================================================
# OBJECT SIZE — set from the config, no separate scene file per object.
#
# /World/.../Object_02/Cylinder is a UNIT mesh (extent -0.5..0.5 on every
# axis). Its real size is ENTIRELY the transform:
#     scale     = (diameter_m, diameter_m, length_m)
#     translate = (0, 0, length_m / 2)   -> base sits on Object_02's origin
# The authored 26 x 140 mm rod is scale (0.026, 0.026, 0.14), translate
# z = 0.07. So a new diameter is a number, not a new mesh or an STL.
# ============================================================
_OBJ_MESH_PATH = "/World/robot_gripper_adapter_sensor/Object_02/Cylinder"
_obj_len_mm = float(CONFIG["object"].get("length_mm", 140.0))
try:
    from pxr import Gf as _Gf, UsdGeom as _UsdGeom
    _mesh = stage.GetPrimAtPath(_OBJ_MESH_PATH)
    if not _mesh.IsValid():
        print(f"[obj] WARNING: {_OBJ_MESH_PATH} not found — object left at "
              f"its authored size. Diameter changes will NOT take effect.")
    elif OBJ_DIAM_MM <= 0:
        print("[obj] no diameter_mm in config; leaving object at authored size")
    else:
        _d_m, _l_m = OBJ_DIAM_MM / 1000.0, _obj_len_mm / 1000.0
        _x = _UsdGeom.Xformable(_mesh)
        _s_op = _t_op = None
        for _op in _x.GetOrderedXformOps():
            if _op.GetOpType() == _UsdGeom.XformOp.TypeScale:
                _s_op = _op
            elif _op.GetOpType() == _UsdGeom.XformOp.TypeTranslate:
                _t_op = _op
        _was = tuple(round(float(c), 4) for c in _s_op.Get()) if _s_op else None
        if _s_op is None:
            _s_op = _x.AddScaleOp()
        _s_op.Set(_Gf.Vec3f(_d_m, _d_m, _l_m))
        if _t_op is not None:
            _tv = _t_op.Get()
            _t_op.Set(type(_tv)(_tv[0], _tv[1], _l_m / 2.0))
        print(f"[obj] size set from config: D={OBJ_DIAM_MM:.1f} mm, "
              f"L={_obj_len_mm:.1f} mm -> scale "
              f"({_d_m:.4f}, {_d_m:.4f}, {_l_m:.4f})"
              + (f"  (was {_was})" if _was else ""))
except Exception as _e:
    print(f"[obj] WARNING: could not set object size ({type(_e).__name__}: "
          f"{_e}) — object left at its authored size.")

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
            # ORIENT: env var wins if set; otherwise the GUI CONFIG decides.
            # (Bug fixed: the config tilt was read at the top of this file but
            #  never used here — the sim always ran orient="keep", so the GUI
            #  preview showed a tilt the simulation never applied.)
            if "OBJ_ORIENT" in os.environ:
                _orient = os.environ["OBJ_ORIENT"]
            elif abs(OBJ_TILT_DEG) > 1e-6:
                _orient = "tilt_x" if str(OBJ_TILT_AXIS).upper() == "X" else "tilt_y"
            else:
                _orient = "keep"

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
                _tilt = np.deg2rad(float(os.environ.get("OBJ_TILT_DEG", OBJ_TILT_DEG)))
                _q = _euler_q(np.pi/2 + _tilt, 0, 0)
            elif _orient == "tilt_x":
                _tilt = np.deg2rad(float(os.environ.get("OBJ_TILT_DEG", OBJ_TILT_DEG)))
                _q = _euler_q(_tilt, 0, 0)
            elif _orient == "tilt_y":
                _tilt = np.deg2rad(float(os.environ.get("OBJ_TILT_DEG", OBJ_TILT_DEG)))
                _q = _euler_q(0, _tilt, 0)
            else:
                _q = Gf.Quatf(_cur_rot.GetReal(), *[float(x) for x in _cur_rot.GetImaginary()])

            # Rotate about the object's CENTRE (matching the GUI preview), not
            # the prim origin. The prim origin sits at the cylinder BASE, so a
            # naive rotation would swing the centre ~24 mm sideways at 20 deg.
            # centre-preserving translate:  p_new = C - R @ (C - p_origin)
            if _orient != "keep":
                _Rm = rotmat(np.array([_q.GetReal(),
                                       *[float(x) for x in _q.GetImaginary()]]))
                _v  = np.array(OBJ_CENTER) - np.array([_px, _py, _pz])
                _pn = np.array(OBJ_CENTER) - _Rm @ _v
                _px, _py, _pz = float(_pn[0]), float(_pn[1]), float(_pn[2])

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

            # ---- HOW THE OBJECT IS HELD ---------------------------------
            # Two different things, and Berith asked specifically about the
            # second one (31 Jul 2026):
            #   BOLTED (default) - a DYNAMIC rigid body pinned by a PhysX
            #       FixedJoint. The solver still integrates it; the joint is
            #       compliant, so in principle it can micro-move. Measured
            #       movement has been 0.00 mm on every run so far.
            #   KINEMATIC        - physics:kinematicEnabled = True. The body
            #       is not integrated at all: it is infinitely heavy and
            #       cannot respond to contact. Contact is then resolved
            #       purely against a fixed surface.
            # Set GRASP_OBJECT_KINEMATIC=1 to use the second.
            _kin = os.environ.get("GRASP_OBJECT_KINEMATIC") == "1"
            if _kin:
                try:
                    # The BOLTED path gets its ORIENTATION from the fixed
                    # joint (LocalRot0), not from the prim transform -- the
                    # pose-setting code above only writes xform ops that
                    # ALREADY exist, so a missing xformOp:orient is skipped
                    # silently. With no joint, that left the rod upright.
                    # So for KINEMATIC we author the transform explicitly.
                    _xf = UsdGeom.Xformable(_obj)
                    _xf.ClearXformOpOrder()
                    _xf.AddTranslateOp().Set(Gf.Vec3d(float(_px), float(_py),
                                                      float(_pz)))
                    _xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(
                        Gf.Quatf(_q))

                    _rb = UsdPhysics.RigidBodyAPI.Get(stage, _frz)
                    if not _rb:
                        _rb = UsdPhysics.RigidBodyAPI.Apply(_obj)
                    _rb.CreateKinematicEnabledAttr().Set(True)

                    # READ BACK: prove the tilt actually landed on the prim,
                    # instead of trusting the value we asked for.
                    _m = UsdGeom.Xformable(_obj).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default())
                    _axis_local = {"X": Gf.Vec3d(0, 0, 1),
                                   "Y": Gf.Vec3d(0, 0, 1)}.get(
                        OBJ_TILT_AXIS, Gf.Vec3d(0, 0, 1))
                    _up = _m.TransformDir(_axis_local)
                    _up = _up / (_up.GetLength() or 1.0)
                    _tilt_meas = np.degrees(np.arccos(
                        max(-1.0, min(1.0, float(_up[2])))))
                    print(f"[grid] cylinder KINEMATIC at pose "
                          f"({_px:.4f},{_py:.4f},{_pz:.4f}) orient={_orient} "
                          f"tilt={OBJ_TILT_DEG:.1f} deg about {OBJ_TILT_AXIS} "
                          f"(no fixed joint; body not integrated by PhysX)")
                    print(f"[grid] KINEMATIC pose READBACK: rod axis is "
                          f"{_tilt_meas:.2f} deg from world Z "
                          f"(asked for {abs(OBJ_TILT_DEG):.2f})"
                          + ("   <-- MISMATCH, tilt did NOT apply"
                             if abs(_tilt_meas - abs(OBJ_TILT_DEG)) > 1.0
                             else "   OK"))
                except Exception as _ke:
                    print(f"[grid] WARNING: kinematic setup failed ({_ke}); "
                          f"falling back to BOLTED")
                    _kin = False
            if not _kin:
                _jpath = _frz + "/WorldFixedJoint"
                _joint = UsdPhysics.FixedJoint.Define(stage, _jpath)
                _joint.CreateBody1Rel().SetTargets([_frz])
                _joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(_px), float(_py), float(_pz)))
                _joint.CreateLocalRot0Attr().Set(_q)
                _joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                _joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                print(f"[grid] cylinder BOLTED (dynamic) at pose ({_px:.4f},{_py:.4f},{_pz:.4f}) "
                      f"orient={_orient} tilt={OBJ_TILT_DEG:.1f} deg about {OBJ_TILT_AXIS} "
                      f"(centre kept at {[round(c,4) for c in OBJ_CENTER]})")
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

# add the RIGHT drive joint so both 4-bars are commanded to the same angle
if DRIVE_BOTH_FINGERS and gi is not None:
    _mir = None
    if GRIPPER_MIRROR_JOINT in dn:
        _mir = dn.index(GRIPPER_MIRROR_JOINT)
    else:
        _c = [d for d in dn if d.endswith("/"+GRIPPER_MIRROR_JOINT)
                                or d.endswith(GRIPPER_MIRROR_JOINT)]
        _mir = dn.index(_c[0]) if _c else None
    if _mir is not None:
        gi = np.array([int(gi[0]), int(_mir)], dtype=np.int32)
        print(f"[grid] SYMMETRIC close: driving both '{GRIPPER_DRIVE_JOINT}' and "
              f"'{GRIPPER_MIRROR_JOINT}' (dof {gi.tolist()})")
    else:
        print(f"[grid] WARNING: '{GRIPPER_MIRROR_JOINT}' not a DOF -> "
              f"single-joint drive (expect the ~2 deg tilt / s1>>s2 to persist)")

dp = np.array(robot.get_joint_positions(), dtype=np.float32)
dp[ai] = INITIAL_JOINTS_RAD
robot.set_joints_default_state(positions=dp)
robot.set_joint_positions(INITIAL_JOINTS_RAD, joint_indices=ai)
robot.get_articulation_controller().apply_action(
    ArticulationAction(joint_positions=INITIAL_JOINTS_RAD, joint_indices=ai))
for _ in range(10): world.step(render=True)
initial_q = robot.get_joint_positions()[ai].copy()

# ============================================================
# Collision world for cuRobo  (STEP 1: stop the arm folding into the table)
# ------------------------------------------------------------
# Numbers from scene_probe.py on scene_cylinder.usd (world frame, mm):
#   table/frame AABB top ~997.7 | robot base ~992.8 | support 970-982 |
#   cylinder 982-1122.
# We do NOT model the table as a solid box up to its real AABB top, because
# that AABB is the vention *frame* (uprights) and would swallow the robot
# base, the support, and the lower cylinder -> every plan would fail.
# Instead: ONE protective slab whose TOP sits just BELOW the worktop/base
# (985 mm). It blocks any downward fold toward the table while leaving the
# base clear and never entering the grasp volume (fingertips stay >~1010 mm).
# The CYLINDER is deliberately left OUT: the gripper must reach and squeeze it.
# All numbers are built in WORLD then transformed to the robot BASE frame
# (cuRobo plans in base frame), so it stays correct if ROBOT_WORLD_POS moved.
# ============================================================
from curobo.geom.types import WorldConfig, Cuboid

# CONTROLLED TEST SWITCH:
#   False = original stable behaviour (cuRobo has NO collision world; world_model=None).
#   True  = table collision slab active (the Step-1 change under test).
# Set False first and run the centered 1x1: if it no longer explodes, the
# collision world was the trigger. Then flip True to reconfirm.
USE_COLLISION_WORLD = os.environ.get("GRASP_COLLISION_WORLD", "1") == "1"

TABLE_TOP_Z_M   = 0.985            # slab top (below base 0.99275, below fingertips)
TABLE_CENTER_XY = (0.0009, 0.0)    # table footprint centre  (world, m)
TABLE_DIMS_XY   = (0.732, 0.867)   # table footprint extents (world, m)

# ---- THE TARGET OBJECT AS A PHASE-DEPENDENT OBSTACLE (2026-08-06) --------
# The rod was deliberately left OUT of the collision world, and for a good
# reason: it is the thing the pads are driven INTO. As a plain obstacle the
# planner refuses to bring the gripper anywhere near it and every point turns
# unreachable.
#
# But leaving it out means the free move — which at 90 deg roll swings the
# flange 157 mm sideways across the workspace — is planned as though the rod
# were not there. That is exactly the sweep most likely to hit it, and on a
# real robot it would.
#
# So the obstacle is PHASE-DEPENDENT:
#   free move to the stand-off   -> object ENABLED  (do not sweep through it)
#   stitched descent, pad-to-pad -> object DISABLED (contact is the point)
# The table slab stays enabled throughout; nothing is ever meant to touch it.
#
#   GRASP_OBJECT_COLLISION=0  leaves the object out entirely (old behaviour)
OBJECT_OBSTACLE_NAME = "target_object"
OBJECT_COLLISION = os.environ.get("GRASP_OBJECT_COLLISION", "1") == "1"
# Grown slightly so a plan does not graze the surface. Contact phases have the
# obstacle switched off, so this margin never fights the grasp itself.
OBJECT_INFLATE_M = float(os.environ.get("GRASP_OBJECT_INFLATE_M", "0.005"))
# How close to the goal the object stops being an obstacle. Everything
# outside this is a normal collision-checked move; inside it, touching the
# object is the point. Must exceed OBJECT_INFLATE_M or the inflated shell
# blocks the final approach.
CONTACT_ZONE_M = float(os.environ.get("GRASP_CONTACT_ZONE_M", "0.030"))
# How far a stitched descent may fall behind its commanded travel before it is
# declared blocked, and how far the pad may end up from the grasp pose before
# the fingers are refused. Both exist because a blocked descent used to be
# silent: it just closed on whatever it was jammed against.
STEP_SHORTFALL_ABORT_MM = float(os.environ.get("GRASP_STEP_SHORTFALL_MM", "8.0"))
MAX_DESCENT_RESIDUAL_MM = float(os.environ.get("GRASP_MAX_RESIDUAL_MM", "3.0"))


def _object_obstacle():
    """The target object as an ORIENTED bounding cuboid, or None.

    A cuboid is used rather than a cylinder primitive so this does not depend
    on which geometry types this cuRobo build exposes. For a cylinder it is
    conservative only in the four corners of the cross-section, which costs
    nothing here because the obstacle is off during every contact phase.

    Sized from the run's own config, so replacing the rod with another shape
    needs no code change — only the dims below become more conservative."""
    d = OBJ_DIAM_MM / 1000.0
    L = float(CONFIG["object"].get("length_mm", 140.0)) / 1000.0
    if d <= 0.0 or L <= 0.0:
        print("[grid] object obstacle SKIPPED (no diameter/length in config)")
        return None
    g = 2.0 * OBJECT_INFLATE_M
    dims = [float(d + g), float(d + g), float(L + g)]
    # tilt about world X, matching how the scene and every plot place the rod
    th = np.deg2rad(OBJ_TILT_DEG) if str(OBJ_TILT_AXIS).upper() == "X" else 0.0
    qw, qx = float(np.cos(th / 2.0)), float(np.sin(th / 2.0))
    c_base = world_to_base(np.array(OBJ_CENTER, float))
    # the obstacle pose is in the BASE frame; combine the rod tilt with the
    # world->base rotation the same way world_to_base handles positions
    q_wb = np.asarray(ROBOT_WORLD_QUAT_WXYZ, float)
    q_wb_inv = np.array([q_wb[0], -q_wb[1], -q_wb[2], -q_wb[3]])   # unit -> conj
    q_obj = quat_mul(q_wb_inv, np.array([qw, qx, 0.0, 0.0]))
    pose = [float(c_base[0]), float(c_base[1]), float(c_base[2]),
            float(q_obj[0]), float(q_obj[1]), float(q_obj[2]), float(q_obj[3])]
    print(f"[grid] object obstacle '{OBJECT_OBSTACLE_NAME}': base-frame centre="
          f"{[round(v,4) for v in c_base]} dims={[round(v,4) for v in dims]} "
          f"tilt={OBJ_TILT_DEG:+.1f} deg about {OBJ_TILT_AXIS} "
          f"(inflated {1000*OBJECT_INFLATE_M:.1f} mm)")
    return Cuboid(name=OBJECT_OBSTACLE_NAME, pose=pose, dims=dims)


def _build_collision_world():
    top, bottom = TABLE_TOP_Z_M, 0.0
    dz = top - bottom
    cz = 0.5 * (top + bottom)
    world_center = np.array([TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], cz])
    base_center  = world_to_base(world_center)          # -> robot BASE frame
    dims = [float(TABLE_DIMS_XY[0]), float(TABLE_DIMS_XY[1]), float(dz)]
    pose = [float(base_center[0]), float(base_center[1]), float(base_center[2]),
            1.0, 0.0, 0.0, 0.0]                          # identity orientation
    print(f"[grid] collision world: table slab base-frame center="
          f"{[round(v,4) for v in base_center]} dims={[round(v,4) for v in dims]}")
    cubes = [Cuboid(name="table_surface", pose=pose, dims=dims)]
    if OBJECT_COLLISION:
        _o = _object_obstacle()
        if _o is not None:
            cubes.append(_o)
    return WorldConfig(cuboid=cubes)


# ---- phase toggle --------------------------------------------------------
_obj_coll_state = [None]        # None = unknown, True/False = last set


def set_object_collision(enable, why=""):
    """Enable/disable the target object as an obstacle, mid-run.

    Free moves plan AROUND the object; contact phases plan THROUGH where it
    is. Wrapped and idempotent: if this cuRobo build has no
    enable_obstacle(), it says so once and everything continues with the
    object permanently in the world."""
    if not (USE_COLLISION_WORLD and OBJECT_COLLISION):
        return
    if _obj_coll_state[0] is enable:
        return
    try:
        mg.world_coll_checker.enable_obstacle(OBJECT_OBSTACLE_NAME, bool(enable))
        _obj_coll_state[0] = bool(enable)
        print(f"  [world] object obstacle {'ON ' if enable else 'OFF'}"
              f"{(' (' + why + ')') if why else ''}")
    except Exception as e:
        if _obj_coll_state[0] is None:
            print(f"[world] enable_obstacle unavailable ({type(e).__name__}: "
                  f"{e}) — object stays in the world for ALL phases, which "
                  f"will make contact poses unreachable; set "
                  f"GRASP_OBJECT_COLLISION=0 to remove it")
        _obj_coll_state[0] = "unavailable"

# ============================================================
# cuRobo
# ============================================================
print("[grid] Loading cuRobo...")
ta = TensorDeviceType()
rc = RobotConfig.from_dict(load_yaml(CUROBO_ROBOT_YAML)["robot_cfg"], ta)
_coll_world = _build_collision_world() if USE_COLLISION_WORLD else None
print(f"[grid] USE_COLLISION_WORLD = {USE_COLLISION_WORLD} "
      f"(world_model = {'table slab' if USE_COLLISION_WORLD else 'None'})")
mg = MotionGen(MotionGenConfig.load_from_robot_config(
    rc, world_model=_coll_world, tensor_args=ta, interpolation_dt=0.02,
    num_trajopt_seeds=4, project_pose_to_goal_frame=True, use_cuda_graph=False))
mg.warmup(enable_graph=False, warmup_js_trajopt=False)
print("[grid] cuRobo ready.")


# ---- WHAT DOES cuRobo ACTUALLY COLLISION-CHECK?  (2026-08-07) ------------
# The 90 deg pad y=+80 run planned straight through the rod and then ground
# into it: the planner saw no collision, PhysX resisted, and the descent ended
# 47 mm short. The obstacle was in the world and correctly placed. The thing
# that was missing was on the ROBOT side.
#
# CUROBO_ROBOT_YAML is ur5e.yml — a stock UR5e, which ends at the wrist
# flange. The Robotiq gripper, the fingers and the TSF-85 pads are not in it,
# so cuRobo cannot collide them with anything. At 90 deg roll the flange
# passes 157 mm clear of the rod, so the plan looks perfectly safe while the
# gripper sweeps straight through.
#
# This prints the gap so it is a measured fact, not a suspicion: how far the
# sphere model reaches from the flange, versus how far the pad actually is.
def report_collision_model():
    try:
        q0 = ta.to_device(np.zeros((1, len(ARM_JOINT_NAMES)), np.float32))
        spheres = mg.kinematics.get_robot_as_spheres(q0)
        flat = spheres[0] if (spheres and isinstance(spheres[0], list)) else spheres
        n = len(flat)
        ee_p, _ = fk(np.zeros(len(ARM_JOINT_NAMES), np.float32))
        reach = 0.0
        for sp in flat:
            c = np.asarray(getattr(sp, "position", [0, 0, 0]), float)
            r = float(getattr(sp, "radius", 0.0))
            reach = max(reach, float(np.linalg.norm(c - ee_p)) + r)
        print(f"[grid] collision model: {n} spheres, reaching "
              f"{1000*reach:.0f} mm beyond the flange; the pad sits "
              f"{1000*TOOL_OFFSET_Z:.0f} mm beyond it.")
        if reach < TOOL_OFFSET_Z * 0.6:
            print("[grid] !! THE GRIPPER IS NOT IN THE COLLISION MODEL. cuRobo "
                  "can only collide the arm, so a plan that sweeps the FINGERS "
                  "or the PADS through the object will still look safe. "
                  "Collision results are valid for the arm only.")
            print("[grid] !! To fix properly, add gripper+pad spheres to "
                  f"{os.path.basename(CUROBO_ROBOT_YAML)}.")
        return {"n_spheres": n, "reach_beyond_flange_mm": 1000 * reach,
                "pad_offset_mm": 1000 * TOOL_OFFSET_Z,
                "gripper_modelled": bool(reach >= TOOL_OFFSET_Z * 0.6)}
    except Exception as e:
        print(f"[grid] collision model: could not inspect "
              f"({type(e).__name__}: {e})")
        return {"error": f"{type(e).__name__}: {e}"}


COLLISION_MODEL_INFO = report_collision_model() if USE_COLLISION_WORLD else None

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
# ROT_DEG = 0.0   # no per-grasp finger rotation in the config collector (yet)
ROT_DEG = float(os.environ.get("GRASP_ROT_DEG", "0.0"))
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

# ---- PIVOT CORRECTION: roll the pad about ITSELF, not about the flange ----
# GRID_POINTS was built at load time as
#     EE_z = pad_target_z + TOOL_OFFSET_Z
# i.e. the pad is assumed to hang STRAIGHT DOWN from the flange. That holds
# only while the tool is vertical. Rolling the tool swings the pad on the
# 156.6 mm lever arm instead of spinning it in place: at 20 deg that is
# 156.6*sin20 = 53.5 mm sideways and 156.6*(1-cos20) = 9.4 mm down, which is
# exactly the mismatch seen between the GUI preview and Isaac on 2026-08-03.
#
# Fix: carry the SAME offset vector, expressed in the tool frame, through the
# new orientation.  v_world_new = R(tq) . R(tq_base)^-1 . (0, 0, TOOL_OFFSET_Z)
# With no spin this collapses to the original (0, 0, TOOL_OFFSET_Z), so runs
# without GRASP_ROT_DEG are bit-for-bit unchanged.
if abs(ROT_DEG) > 1e-6:
    _v0 = np.array([0.0, 0.0, TOOL_OFFSET_Z])
    _Rb = rotmat(tq_base)
    _v_new = rotmat(tq) @ (_Rb.T @ _v0)
    for _gp in GRID_POINTS:
        # recover the pad target this point was built from, then re-offset
        _pad = np.array(_gp["world"]) - _v0
        _gp["world"] = list(_pad + _v_new)
    _d = _v_new - _v0
    print(f"[grid] pivot correction: EE target shifted by "
          f"({_d[0]*1000:+.1f}, {_d[1]*1000:+.1f}, {_d[2]*1000:+.1f}) mm "
          f"so the pad rolls about its own centre")

# ---- FLANGE->PAD vector and TOOL APPROACH AXIS, for any roll --------------
# PAD_OFFSET_VEC is the flange->pad vector carried through the CURRENT tool
# orientation. It equals (0, 0, TOOL_OFFSET_Z) when upright, so nothing changes
# without a roll. Having it as one named value stops the "pad = ee_z - TOOL"
# shortcut, which is only true while the tool is vertical and printed the pad
# 157 mm low in the 90 deg report of 2026-08-06.
PAD_OFFSET_VEC = (rotmat(tq) @ (rotmat(tq_base).T @ np.array([0.0, 0.0,
                                                              TOOL_OFFSET_Z]))
                  if abs(ROT_DEG) > 1e-6 else
                  np.array([0.0, 0.0, TOOL_OFFSET_Z]))


def pad_from_ee_target(ee_world):
    """Pad centre implied by an EE target, correct at ANY roll."""
    return np.asarray(ee_world, float) - PAD_OFFSET_VEC


# TOOL APPROACH AXIS (unit, BASE frame). Tool local +Z points along the
# approach; upright that is world -Z, so this is (0, 0, -1) and every formula
# below reduces to the original world-Z arithmetic exactly.
APPROACH_AXIS_BASE = rotmat(tq) @ np.array([0.0, 0.0, 1.0])
APPROACH_AXIS_WORLD = rotmat(ROBOT_WORLD_QUAT_WXYZ) @ APPROACH_AXIS_BASE

# ---- WHY THE ROLLED DESCENT FAILED  (2026-08-06) -------------------------
# plan_stitched_z steps the target along WORLD Z, but the pose metric it plans
# under is CASE13_WEIGHT = [1,1,1,1,1,0] and MotionGen was built with
# project_pose_to_goal_frame=True — so the ONE freed degree of freedom is the
# TOOL's local z, not the world's. Upright the two coincide and nothing is
# wrong. Rolled, they diverge: at 90 deg the planner is asked to travel along
# world Z while the only axis it is allowed to move along is horizontal. The
# request is self-contradictory, which is exactly the 90 deg failure the
# ledger caught (predicted ok -> exec_stage "stitched_descent"), and why 30
# and 45 deg get progressively harder.
#
# Fix: step along the TOOL's approach axis, i.e. the same axis the metric
# frees. Behind a flag so upright behaviour is untouched and the two can be
# A/B'd in one run.
#   GRASP_APPROACH_ALONG_TOOL=0  (default) world Z, exactly as before
#   GRASP_APPROACH_ALONG_TOOL=1            along the tool approach axis
APPROACH_ALONG_TOOL = os.environ.get("GRASP_APPROACH_ALONG_TOOL", "1") == "1"

# GRAPH SEARCH for the free move (see plan_free_move). ON by default: without
# it a rolled pad whose only IK solution sits on another arm branch is
# reported unreachable even though the arm can get there through open air.
ENABLE_GRAPH = os.environ.get("GRASP_ENABLE_GRAPH", "1") == "1"
print(f"[grid] GRASP_ENABLE_GRAPH = {int(ENABLE_GRAPH)} "
      f"(free move {'searches joint space, falls back to local' if ENABLE_GRAPH else 'is LOCAL only (old behaviour)'})")
print(f"[grid] GRASP_APPROACH_ALONG_TOOL = {int(APPROACH_ALONG_TOOL)} "
      f"(approach/descend along "
      f"{'the TOOL axis' if APPROACH_ALONG_TOOL else 'world Z'}; "
      f"tool axis in world = {np.round(APPROACH_AXIS_WORLD, 4).tolist()})")


def approach_axis_base():
    """Unit vector the descent travels along, BASE frame.
    (0,0,-1) upright, so the flag off and roll zero are the same thing."""
    return (APPROACH_AXIS_BASE if APPROACH_ALONG_TOOL
            else np.array([0.0, 0.0, -1.0]))


def approach_axis_world():
    """Same vector in WORLD frame, for building the retreat/UP point."""
    return (APPROACH_AXIS_WORLD if APPROACH_ALONG_TOOL
            else np.array([0.0, 0.0, -1.0]))


def up_point(grasp_world):
    """The stand-off pose APPROACH_H BACK along the approach axis.

    Must use the SAME axis as the descent, or the two do not meet: backing off
    in world Z and then descending along a rolled tool axis lands somewhere
    else entirely. Upright this is grasp_world + (0, 0, APPROACH_H), exactly
    as before."""
    return np.asarray(grasp_world, float) - APPROACH_H * approach_axis_world()

current_grip = [0.0]

def apply_arm_and_grip(arm_q, grip_val=None):
    if grip_val is not None:
        current_grip[0] = float(grip_val)
    robot.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=arm_q.astype(np.float32), joint_indices=ai))
    if gi is not None:
        robot.get_articulation_controller().apply_action(
            ArticulationAction(
                joint_positions=np.full(len(gi), current_grip[0], dtype=np.float32),
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

# ---- pad frame from JOINTS (world), for verification plots + stitching ----
# ROOT CAUSE NOTE (frozen-pose bug of run_20260704_144039):
#   The old reader used UsdGeom.XformCache on the USD stage. In script mode
#   PhysX moves the articulation links in FABRIC only — the USD stage keeps
#   the AUTHORED (startup) transforms. So every read returned the same
#   startup pose, and pad_actual_pos_m came out identical for every grasp.
#   The joints ARE live (robot.get_joint_positions()), so we now compute the
#   pad pose from joints -> cuRobo FK -> EE world -> fixed wrist->pad offset
#   (the Paper-2 logic; offset probed in THIS scene by probe_pad_normal.py).
PAD_OFF_WRIST = np.array([-0.0142, 0.0624, 0.1214])   # m, wrist_3-local (probed)

def pad_pose_from_joints(q_arm):
    """(pos[3], R[3x3]) of the pad frame in WORLD, from the arm joints.
    R columns = pad axes (pad normal = wrist X, confirmed by probing)."""
    ee_pos_b, ee_quat_b = fk(np.asarray(q_arm, dtype=np.float32))
    R_wb   = rotmat(ROBOT_WORLD_QUAT_WXYZ)
    R_ee_w = R_wb @ rotmat(ee_quat_b)
    pos_w  = R_wb @ ee_pos_b + ROBOT_WORLD_POS + R_ee_w @ PAD_OFF_WRIST
    return np.asarray(pos_w, dtype=float), R_ee_w

# ---- PROBE: physics-side TRUE pad pose (MEASURES the finger swing) ----
# pad_pose_from_joints() is BLIND to closing (fixed offset). To measure where the
# pad really ends up after the fingers swing shut, read the sensor prim's LIVE
# world pose from the physics side. We don't know which reader is live in this
# build, so we TRY several and dump ALL results; whichever value CHANGES between
# open and closed is the live one, and that change IS the swing. Fully wrapped so
# it can NEVER break a run.
def _read_prim_world(prim_path):
    """SAFE readers ONLY. We do NOT construct SingleRigidPrim / SingleXFormPrim /
    isaacsim.core prim wrappers here: wrapping a live articulation link in a
    physics view mid-simulation is what corrupted PhysX and exploded the close.
    Both readers below are passive:
      * usdrt  -> LIVE Fabric pose (what PhysX actually moved the link to),
      * UsdGeom.XformCache -> passive USD read (may be 'frozen'/authored for
        Fabric-driven links; logged for comparison).
    Neither creates a physics view, so neither can perturb the sim."""
    out = {}
    # (a) LIVE Fabric read via usdrt -----------------------------------------
    try:
        import omni.usd
        from usdrt import Usd as _RtU, Rt
        srt = _RtU.Stage.Attach(omni.usd.get_context().get_stage_id())
        p = srt.GetPrimAtPath(prim_path)
        val = None
        try:
            rtx = Rt.Xformable(p)
            if rtx.HasWorldXform():
                wp = rtx.GetWorldPosition().Get()
                if wp is not None:
                    val = [float(wp[0]), float(wp[1]), float(wp[2])]
        except Exception:
            pass
        if val is None:  # fall back to Fabric world-position attributes
            for an in ("_worldPosition", "omni:fabric:worldPosition"):
                a = p.GetAttribute(an)
                if a and a.IsValid() and a.Get() is not None:
                    v = a.Get()
                    val = [float(v[0]), float(v[1]), float(v[2])]
                    out["usdrt_attr_used"] = an
                    break
        out["usdrt.world"] = val if val is not None else "no world xform"
    except Exception as e:
        out["usdrt.world"] = f"ERR {e}"
    # (b) passive USD read (frozen-safe reference) ---------------------------
    try:
        from pxr import UsdGeom as _UG, Usd as _U
        prim = world.stage.GetPrimAtPath(prim_path)
        m = _UG.XformCache(_U.TimeCode.Default()).GetLocalToWorldTransform(prim)
        t = m.ExtractTranslation()
        out["UsdGeom.XformCache"] = [float(t[0]), float(t[1]), float(t[2])]
    except Exception as e:
        out["UsdGeom.XformCache"] = f"ERR {e}"
    return out

def _probe_all(prim_paths):
    return {name: _read_prim_world(pp) for name, pp in prim_paths.items()}

def plan_free_move(start_q, target_base, label):
    """Point-to-point move through open air to the stand-off pose.

    GRAPH SEARCH (2026-08-06). This used enable_graph=False, which limits
    cuRobo to LOCAL trajectory optimisation: it takes one straight guess and
    smooths it. That is fast, and it was fine for years because every grid
    point sat on the same arm branch as the previous one.

    Rolling the pad broke that. The IK probe on the failed 45 deg point found
    a perfectly good solution — 0.001 mm pose error — but at
    shoulder_pan = +1.96 rad, while the arm sits near -0.85 rad. The goal is
    real and reachable; it is just ~160 deg away in joint space, and a local
    optimiser cannot cross that gap, so it reported "no IK solution".
    enable_graph=True makes cuRobo search a route through joint space first
    and then smooth it, which is what that swing needs.

    ONLY the free move gets this. plan_stitched_z and plan_stitched_line move
    the pad in 0.8 mm steps while it is near or touching the rod; a graph
    search there would be free to jump branches mid-contact, which is exactly
    what those functions exist to prevent.

    The graph attempt is tried FIRST and falls back to the old local plan if
    it fails, so this cannot do worse than the previous behaviour.
      GRASP_ENABLE_GRAPH=0  restores the old local-only planning
    """
    set_object_collision(True, "free move")
    s = JointState.from_position(
            ta.to_device(start_q.astype(np.float32)).view(1, -1),
            joint_names=ARM_JOINT_NAMES)
    g = Pose(position=ta.to_device(target_base.astype(np.float32)).view(1, 3),
             quaternion=ta.to_device(tq.astype(np.float32)).view(1, 4))
    attempts = ([(True, "graph"), (False, "local")] if ENABLE_GRAPH
                else [(False, "local")])
    last = None
    for use_graph, how in attempts:
        try:
            r = mg.plan_single(s, g, MotionGenPlanConfig(
                max_attempts=5, enable_graph=use_graph))
        except Exception as e:
            print(f"  [{label}] free move ({how}) raised "
                  f"{type(e).__name__}: {e}")
            continue
        last = r
        if r.success.item():
            if how == "local" and ENABLE_GRAPH:
                print(f"  [{label}] free move fell back to LOCAL planning")
            return r.get_interpolated_plan().position.cpu().numpy()
        if ENABLE_GRAPH:
            print(f"  [{label}] free move ({how}) failed ({r.status})")
    print(f"  [{label}] free move FAILED "
          f"({last.status if last is not None else 'no result'})")
    return None

def plan_stitched_z(start_q, dz, label):
    metric = PoseCostMetric(hold_partial_pose=True,
        hold_vec_weight=mg.tensor_args.to_device(
            np.array(CASE13_WEIGHT, dtype=np.float32)))
    cfg = MotionGenPlanConfig(enable_graph=False, max_attempts=4,
                              enable_finetune_trajopt=False, pose_cost_metric=metric)
    # CONTACT ZONE (2026-08-07). Disabling the object for the WHOLE descent
    # was wrong. At 90 deg roll the approach axis is horizontal, so this
    # "descent" is a 100 mm sweep along world Y — and on the pad y=+80 run it
    # went from Y=22 straight through the rod at Y=186..212 with collision
    # switched off. The gripper shoved the cylinder across the table.
    #
    # Contact only legitimately happens in the last few mm. So the object
    # stays an obstacle for the long approach and is released only inside
    # CONTACT_ZONE_M of the goal.
    #
    # Travel direction: upright (or flag off) this is (0,0,-1) and
    # `cpos + (-step)*axis` is identical to the old `cpos[2] += step`, so
    # nothing changes for existing runs. Rolled with the flag on, it follows
    # the tool's approach axis — the one CASE13_WEIGHT actually frees.
    step = dz / N_STEPS
    _axis = approach_axis_base()
    _travel = abs(float(dz))
    _short = 0.0               # accumulated shortfall along the travel axis
    cur_q = start_q.copy()
    stitched = []
    for i in range(N_STEPS):
        # distance still to go AFTER this step
        _left = _travel * (1.0 - float(i + 1) / N_STEPS)
        set_object_collision(_left > CONTACT_ZONE_M,
                             f"descent, {1000*_left:.0f} mm to go")
        cpos, cquat = fk(cur_q)
        tgt = cpos + (-step) * _axis
        s = JointState.from_position(
                ta.to_device(cur_q.astype(np.float32)).view(1, -1),
                joint_names=ARM_JOINT_NAMES)
        # ORIENTATION: fixed tool-down `tq`, NOT the current cquat. Same bug that
        # tilted the line move: targeting cquat makes each step inherit the last
        # step's orientation error, so the tilt compounds down the 10-step
        # descent with nothing to correct it. Evidence: pt00 (the only point that
        # uses THIS descent) came out 72x asymmetric (s1 20023 / s2 276) while
        # every point using the tq-anchored line move was 2-4x.
        g = Pose(position=ta.to_device(tgt.astype(np.float32)).view(1, 3),
                 quaternion=ta.to_device(tq.astype(np.float32)).view(1, 4))
        r = mg.plan_single(s, g, cfg)
        if not r.success.item():
            print(f"  [{label}] step {i+1}/{N_STEPS} FAILED ({r.status})")
            return None
        tr = r.get_interpolated_plan().position.cpu().numpy()

        # ---- DID THE STEP ACTUALLY ARRIVE?  (2026-08-07) -----------------
        # CASE13_WEIGHT = [1,1,1,1,1,0] FREES the tool's own z, which is the
        # very axis this loop travels along. So a step can report success
        # while barely moving: the constraint it satisfied never mentioned
        # the travel direction. Blocked by an obstacle, every step lands a
        # little short, nothing complains, and the shortfall accumulates —
        # 47 mm over ten steps on the pad y=+80, 90 deg run, after which the
        # gripper closed anyway.
        #
        # So measure it: how far along the axis did we really go?
        # The commanded displacement is (-step)*_axis, so project the ACHIEVED
        # displacement onto _axis and compare magnitudes. Works for descent
        # (step<0) and retreat (step>0) alike.
        _p_end, _ = fk(np.asarray(tr[-1], np.float32))
        _moved = float((_p_end - cpos) @ _axis)
        _asked = -float(step)
        _short += max(0.0, abs(_asked) - abs(_moved))
        if _short > STEP_SHORTFALL_ABORT_MM / 1000.0:
            print(f"  [{label}] step {i+1}/{N_STEPS} arrived "
                  f"{1000*_short:.1f} mm short in total (limit "
                  f"{STEP_SHORTFALL_ABORT_MM:.0f} mm) — the path is BLOCKED. "
                  f"Stopping instead of grinding into it.")
            return None

        if stitched: tr = tr[1:]
        stitched.extend(list(tr))
        cur_q = tr[-1].copy()
    return np.array(stitched)

def plan_stitched_line(start_q, delta_world, label, n_steps=LINE_STEPS):
    """Straight-line EE move along delta_world (metres), split into small steps
    with the tool ORIENTATION held. Same incremental idea as plan_stitched_z
    (Paper-2 stitched descent) but along an arbitrary direction, so we can go
    pad-to-pad between grid points instead of lifting APPROACH_H and back.
    Each step is tiny (e.g. 8 mm / 10 = 0.8 mm), so cuRobo cannot swing to a
    different arm branch -> no jerk. Returns None if any step fails (caller
    then falls back to the old approach)."""
    set_object_collision(False, "pad-to-pad move near the object")
    delta = np.asarray(delta_world, dtype=float)
    if float(np.linalg.norm(delta)) < 1e-6:
        return np.array([start_q.copy()])
    # fk() returns BASE-frame positions, so express the step in the base frame.
    # (identity today, but do it properly so a rotated robot mount still works)
    delta = rotmat(ROBOT_WORLD_QUAT_WXYZ).T @ delta
    metric = PoseCostMetric(hold_partial_pose=True,
        hold_vec_weight=mg.tensor_args.to_device(
            np.array(LINE_WEIGHT, dtype=np.float32)))
    cfg = MotionGenPlanConfig(enable_graph=False, max_attempts=4,
                              enable_finetune_trajopt=False, pose_cost_metric=metric)
    step = delta / float(n_steps)
    cur_q = start_q.copy()
    stitched = []
    for i in range(n_steps):
        cpos, cquat = fk(cur_q)
        tgt = cpos + step
        s = JointState.from_position(
                ta.to_device(cur_q.astype(np.float32)).view(1, -1),
                joint_names=ARM_JOINT_NAMES)
        # ORIENTATION: target the FIXED tool-down quaternion `tq`, NOT the current
        # cquat. Using cquat made every step inherit the previous step's small
        # orientation error, so the tilt COMPOUNDED over 10 steps x N points with
        # nothing to re-anchor it (the old lift-to-UP used tq and reset it every
        # point). That drift swung one pad into the rod and the other off it
        # (s1 sum ~21000 vs s2 ~279). Targeting tq re-anchors on every step.
        g = Pose(position=ta.to_device(tgt.astype(np.float32)).view(1, 3),
                 quaternion=ta.to_device(tq.astype(np.float32)).view(1, 4))
        r = mg.plan_single(s, g, cfg)
        if not r.success.item():
            print(f"  [{label}] line step {i+1}/{n_steps} FAILED ({r.status})")
            return None
        tr = r.get_interpolated_plan().position.cpu().numpy()
        if stitched: tr = tr[1:]
        stitched.extend(list(tr))
        cur_q = tr[-1].copy()
    return np.array(stitched)

# ============================================================
# REACHABILITY  (port of Paper 2 _evaluate_reachability)
# ------------------------------------------------------------
# Kept verbatim from Paper 2: the incremental dry-run (steps sized by
# MAX_JOINT_STEP, capped at MAX_STEPS_CAP), every limit gate (frozen_joints,
# one_sided, delta_bounds, per_iter_dq_cap, abs_limits), the cond(J) gate, the
# final FK tolerance (1 mm / 0.5 deg) and the {reachable, reason, ...} contract.
#
# Swapped for this pipeline (MoveIt/real robot -> Isaac + cuRobo):
#   robot.fk.get_fk(...)          -> fk(q)                    (cuRobo)
#   compute_jacobian2(robot.jac)  -> _numeric_jacobian6(q)    (finite diff on FK)
#
# FIXED vs Paper 2: there, the dry-run called get_jacobian6(), which returns J at
# the CURRENT robot q, so cond(J) re-checked the same J every step (the code even
# says "if get_jacobian6() reflects current q"). Here J is evaluated AT each q on
# the trajectory, so the singularity gate actually means something.
# ============================================================
def _quat_to_R(qwxyz):
    """cuRobo quaternion (w,x,y,z) -> 3x3 rotation matrix. numpy only."""
    w, x, y, z = [float(v) for v in qwxyz]
    n = np.sqrt(w*w + x*x + y*y + z*z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)]], dtype=float)

def _rotvec_from_R(Rm):
    """Rotation matrix -> rotation vector (axis*angle). numpy only."""
    c = float(np.clip((np.trace(Rm) - 1.0) * 0.5, -1.0, 1.0))
    ang = float(np.arccos(c))
    v = np.array([Rm[2,1]-Rm[1,2], Rm[0,2]-Rm[2,0], Rm[1,0]-Rm[0,1]], dtype=float)
    if ang < 1e-9:
        return 0.5 * v                       # small-angle
    s = 2.0 * np.sin(ang)
    if abs(s) < 1e-12:                       # ang ~ pi
        return v / (np.linalg.norm(v) + 1e-12) * ang
    return (v / s) * ang

def _numeric_jacobian6(q, eps=1e-5):
    """6xN spatial Jacobian at q by finite differences on cuRobo FK.
    Paper 2 got this from the real robot (compute_jacobian2); cuRobo doesn't
    expose one here, and 7 FK calls is cheap for a pre-check."""
    q = np.asarray(q, float)
    p0, quat0 = fk(q)
    R0 = _quat_to_R(quat0)
    J = np.zeros((6, q.size), dtype=float)
    for i in range(q.size):
        dq = np.zeros(q.size); dq[i] = eps
        p1, quat1 = fk(q + dq)
        R1 = _quat_to_R(quat1)
        J[0:3, i] = (np.asarray(p1, float) - np.asarray(p0, float)) / eps
        J[3:6, i] = _rotvec_from_R(R0.T @ R1) / eps
    return J

def evaluate_reachability(q_start, q_goal, manual_limits=None,
                          max_joint_step=MAX_JOINT_STEP,
                          max_steps_cap=MAX_STEPS_CAP,
                          cond_max_warn=COND_MAX_WARN,
                          pos_tol=POS_TOL_M, rot_tol_deg=ROT_TOL_DEG,
                          check_cond=True):
    """Dry-run q_start -> q_goal. Nothing is commanded. Paper-2 logic.

    ADVISORY vs VETO (2026-08-07). This walks a STRAIGHT LINE in joint space
    and checks the Jacobian along it. The executor never performs that motion:
    it free-moves to the stand-off (cuRobo, free to curve) and then descends in
    0.8 mm steps. So a conditioning failure here says the straight line is
    poorly conditioned, NOT that the pose cannot be reached.

    It was silently costing real data. The same 90 deg pose, same offsets,
    reported min_sigma 0.0060 -> 0.0040 -> 0.0012 on three runs and flipped
    from reachable to 'cond(J)=1749 > 1000' on the third, while cuRobo planned
    to it and the IK solver hit it to 0.002 mm every time. A gate that
    marginal, on a path nobody drives, should not be able to skip a point.

    So the SOFT checks — conditioning and non-convergence of the dry run — are
    now recorded as warnings and leave `reachable` True. The HARD checks stay
    fatal, because they describe the goal itself rather than the straight-line
    path to it: frozen joints, one_sided, delta_bounds, per_iter_dq_cap and
    absolute joint limits. And a missing IK/plan solution still fails outright,
    upstream of this function.

      GRASP_REACH_STRICT=1  restores the old behaviour (soft checks fatal)
    """
    ml = manual_limits or {}
    frozen_joints   = list(ml.get("frozen_joints", []) or [])
    one_sided       = ml.get("one_sided", {}) or {}
    delta_bounds    = ml.get("delta_bounds", {}) or {}
    per_iter_dq_cap = ml.get("per_iter_dq_cap", None)
    abs_limits      = ml.get("abs_limits", None)
    if isinstance(abs_limits, (list, tuple)):
        abs_limits = [(-np.inf, np.inf) if v is None else tuple(v) for v in abs_limits]

    q0 = np.asarray(q_start, float); q1 = np.asarray(q_goal, float)
    rot_tol = float(np.deg2rad(rot_tol_deg))
    limit_hits = []

    # --- same step-count logic as Paper 2's _incremental_execute ---
    dq_total = q1 - q0
    steps_per_joint = np.ceil(np.abs(dq_total) / float(max_joint_step))
    N = int(np.clip(np.max(steps_per_joint) if steps_per_joint.size else 2,
                    2, int(max_steps_cap)))
    traj = [q0 + (i / (N - 1)) * dq_total for i in range(N)]

    def _violates_one_sided(dq_vec):
        for j, mode in (one_sided or {}).items():
            j = int(j)
            if (mode == "pos" and dq_vec[j] < 0) or (mode == "neg" and dq_vec[j] > 0):
                return True
        return False

    q_prev = traj[0].copy()
    traj_q = [q_prev.copy()]
    smin_hist = []
    warnings = []              # soft findings: recorded, not fatal
    reachable, reason = True, "converged"

    for k in range(1, len(traj)):
        q_next = traj[k]; dq = q_next - q_prev

        if frozen_joints:
            bad = [j for j in frozen_joints if abs(dq[int(j)]) > 1e-12]
            if bad:
                reachable = False; reason = f"frozen joint {bad[0]} would move"
                limit_hits.append(reason); break

        if one_sided and _violates_one_sided(dq):
            reachable = False; reason = "one_sided violation"
            limit_hits.append(reason); break

        if delta_bounds:
            hit = None
            for j, (dmin, dmax) in delta_bounds.items():
                j = int(j)
                if not (dmin <= dq[j] <= dmax):
                    hit = f"delta_bounds violation @ joint {j}"; break
            if hit:
                reachable = False; reason = hit; limit_hits.append(hit); break

        if per_iter_dq_cap is not None:
            cap = abs(float(per_iter_dq_cap))
            if np.any(np.abs(dq) > cap + 1e-12):
                reachable = False; reason = "per_iter_dq_cap violation"
                limit_hits.append(reason); break

        if abs_limits:
            hit = None
            for j, (qmin, qmax) in enumerate(abs_limits):
                if not (qmin - 1e-12 <= q_next[j] <= qmax + 1e-12):
                    hit = f"abs_limits violation @ joint {j}"; break
            if hit:
                reachable = False; reason = hit; limit_hits.append(hit); break

        if check_cond:
            try:
                J = _numeric_jacobian6(q_next)      # J AT this q (Paper-2 fix)
                s = np.linalg.svd(J, compute_uv=False)
                smin = max(float(np.min(s)), 1e-12)
                cnd  = float(np.max(s) / smin)
                smin_hist.append(smin)
                if cnd > cond_max_warn:
                    # SOFT: the straight line is badly conditioned here. The
                    # executor does not drive this line, so record it and
                    # carry on. Fatal only under GRASP_REACH_STRICT=1.
                    _w = f"cond(J)={cnd:.1f} > {cond_max_warn}"
                    warnings.append(_w); limit_hits.append(_w)
                    if REACH_STRICT:
                        reachable = False; reason = _w; break
            except Exception as _je:
                pass                                 # no J -> skip the gate

        q_prev = q_next; traj_q.append(q_prev.copy())

    # --- final FK error vs the goal ---
    p_fin, quat_fin = fk(q_prev); R_fin = _quat_to_R(quat_fin)
    p_des, quat_des = fk(q1);     R_des = _quat_to_R(quat_des)
    dp_final  = np.asarray(p_des, float) - np.asarray(p_fin, float)
    ang_final = float(np.linalg.norm(_rotvec_from_R(R_fin.T @ R_des)))

    if reachable:
        if np.linalg.norm(dp_final) < pos_tol and ang_final < rot_tol:
            reason = ("converged (trajectory dry-run)" if not warnings else
                      "converged with warnings (trajectory dry-run)")
        else:
            # SOFT: the straight-line dry run did not land on the goal. That
            # is a property of this integrator, not of the pose — cuRobo has
            # already planned to it. Advisory unless GRASP_REACH_STRICT=1.
            _w = "final FK error above tolerance (trajectory dry-run)"
            warnings.append(_w); limit_hits.append(_w)
            if REACH_STRICT:
                reachable = False; reason = _w
            else:
                reason = "dry-run did not converge, but the planner reached it"

    return {
        "reachable": bool(reachable),
        "reason": reason,
        "advisory": warnings,
        "reach_strict": bool(REACH_STRICT),
        "limit_hits": limit_hits,
        "iters": len(traj_q) - 1,
        "n_steps": N,
        "pos_err_final_m": float(np.linalg.norm(dp_final)),
        "rot_err_final_deg": float(np.rad2deg(ang_final)),
        "min_sigma": float(min(smin_hist)) if smin_hist else None,
    }

def _ik_at(target_world, q_seed, label="ik", prev_world=None):
    """Joint solution at an EE world target, by the route the RUN will take.

    Returns (q_goal | None, path_used) where path_used is "up_then_down",
    "pad_to_pad" or "failed".

    WHY THERE IS NO "direct" ANY MORE (2026-08-07). This used to try a single
    straight plan into the grasp pose first, because one plan is faster than
    two. But the executor NEVER does that: the first point free-moves to
    grasp + APPROACH_H and then descends, and every later point goes
    pad-to-pad along a stitched line. So "reachable" was being proved on a
    path the run does not drive — which is how a point gets certified and then
    fails in execution. Every report up to 2026-08-07 carried the warning
    `certified via the DIRECT plan only`; this removes the cause.

    prev_world is the previous point's EE target. Given, and with
    POINT_TO_POINT on, the pad-to-pad line is checked — exactly the call the
    executor makes. Otherwise up-then-down, exactly what it does for the first
    point. The line falls back to up-then-down because the executor falls back
    the same way."""
    tw = np.asarray(target_world, float)
    q_seed = np.asarray(q_seed, float)

    if prev_world is not None and POINT_TO_POINT:
        delta = tw - np.asarray(prev_world, float)
        tr = plan_stitched_line(q_seed, delta, f"{label}:pad-to-pad")
        if tr is not None:
            return np.asarray(tr[-1], float), "pad_to_pad"

    up_world = up_point(tw)
    tr_up = plan_free_move(q_seed, world_to_base(up_world), f"{label}:to-up")
    if tr_up is None:
        return None, "failed"
    q_up = np.asarray(tr_up[-1], float)
    tr_dn = plan_stitched_z(q_up, -float(APPROACH_H), f"{label}:down")
    if tr_dn is None:
        return None, "failed"
    return np.asarray(tr_dn[-1], float), "up_then_down"


# ------------------------------------------------------- reason codes ----
# "unreachable" was one free-text string covering causes that need completely
# different responses: a wrist limit, a singular straight-line path, a manual
# constraint, and a collision are not the same problem. These stable codes
# separate them so the report can be read at a glance and counted across runs.
#
#   ok               nothing objected
#   ik_no_solution   the planner found no trajectory to the goal pose at all
#                    (workspace or wrist limit, or collision when the world is
#                    on — see collision_untested below)
#   path_infeasible  goal reachable, but the executor's own path could not be
#                    planned
#   singularity      straight-line dry-run passes near a singular
#                    configuration (min_sigma below COND_MAX_WARN)
#   joint_limit      a joint would leave its absolute limits
#   manual_limit     frozen_joints / one_sided / delta_bounds / per_iter_dq_cap
#
# collision_static is NOT emitted yet: with USE_COLLISION_WORLD = False there
# is no world to collide with, so nothing here can distinguish "out of reach"
# from "would hit the table". That separation arrives in Step 3, by planning
# twice — once with the world and once without.
REACH_CODES = ("ok", "ok_advisory", "ik_no_solution", "path_infeasible",
               "singularity", "joint_limit", "manual_limit", "collision_static")


def _reach_reason_code(res, path_used):
    """Map an evaluate_reachability result onto a stable code."""
    if res.get("reachable"):
        # reachable, but the straight-line dry run objected to something the
        # executor never drives — worth seeing, not worth skipping the point
        return "ok_advisory" if res.get("advisory") else "ok"
    hits = [str(h) for h in (res.get("limit_hits") or [])]
    blob = " ".join(hits + [str(res.get("reason", ""))]).lower()
    if path_used == "failed" or "ik_failed" in blob or "ik/plan failed" in blob:
        return "ik_no_solution"
    if "sigma" in blob or "singular" in blob or "cond" in blob:
        return "singularity"
    if "abs_limit" in blob or "joint limit" in blob or "limit @" in blob:
        return "joint_limit"
    if ("frozen" in blob or "one_sided" in blob or "delta_bounds" in blob
            or "per_iter_dq_cap" in blob):
        return "manual_limit"
    if "tol" in blob or "converge" in blob:
        return "path_infeasible"
    return "path_infeasible"


# ------------------------------------------------- predicted vs actual ----
# The pre-check and the executor have disagreed in BOTH directions: points
# certified reachable whose motion later failed, and points skipped that would
# probably have worked. Neither was ever recorded, so the disagreement rate is
# unknown. This ledger writes it down for every point of every run —
# diagnostic only, it changes no decision.
EXEC_LEDGER = {}          # index -> {"stage": ..., "ok": bool}
_REACH_ROWS = {}          # index -> the pre-check row for that point


def _tag_index(tag):
    """'pt07' -> 7. The ledger is keyed by grid index, the executor by tag."""
    try:
        return int(str(tag).strip().lower().replace("pt", ""))
    except Exception:
        return -1


def _ledger(idx, stage, ok):
    """Record how far a point got. Called at every exit of grasp_one_point."""
    try:
        EXEC_LEDGER[int(idx)] = {"stage": str(stage), "ok": bool(ok)}
    except Exception:
        pass


def write_execution_ledger(out_dir):
    """Join what the pre-check PREDICTED against what execution DID.

    Four outcomes, and only two of them are agreements:
      predicted_ok_executed_ok       pre-check right
      predicted_bad_skipped          pre-check right (unverifiable — the point
                                     was skipped, so we never learn)
      predicted_ok_executed_failed   FALSE POSITIVE: certified, then failed
      predicted_bad_executed_ok      FALSE NEGATIVE: only visible with
                                     GRASP_REACH_SKIP=0
    A run with false positives means the pre-check is testing a different path
    from the executor; false negatives mean it is too strict."""
    rows, counts = [], {"predicted_ok_executed_ok": 0,
                        "predicted_ok_executed_failed": 0,
                        "predicted_bad_executed_ok": 0,
                        "predicted_bad_skipped": 0,
                        "not_attempted": 0}
    for idx in sorted(set(_REACH_ROWS) | set(EXEC_LEDGER)):
        pre = _REACH_ROWS.get(idx, {})
        ex = EXEC_LEDGER.get(idx)
        pred_ok = bool(pre.get("reachable", True))
        if ex is None:
            outcome = "predicted_bad_skipped" if not pred_ok else "not_attempted"
        elif pred_ok and ex["ok"]:
            outcome = "predicted_ok_executed_ok"
        elif pred_ok and not ex["ok"]:
            outcome = "predicted_ok_executed_failed"
        elif (not pred_ok) and ex["ok"]:
            outcome = "predicted_bad_executed_ok"
        else:
            outcome = "predicted_bad_skipped"
        counts[outcome] = counts.get(outcome, 0) + 1
        rows.append({"index": idx, "tag": f"pt{idx:02d}",
                     "predicted_reachable": pred_ok,
                     "reason_code": pre.get("reason_code"),
                     "reason": pre.get("reason"),
                     "precheck_path": pre.get("precheck_path"),
                     "executed": (ex or {}).get("ok"),
                     "exec_stage": (ex or {}).get("stage"),
                     "outcome": outcome})
    doc = {"generated": _stamp, "config": _args.config,
           "grasp_rot_deg": float(ROT_DEG),
           "use_collision_world": bool(USE_COLLISION_WORLD),
           "collision_model": COLLISION_MODEL_INFO,
           "reach_skip": bool(REACH_SKIP),
           "counts": counts, "points": rows}
    try:
        p = os.path.join(out_dir, "execution_ledger.json")
        with open(p, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"[ledger] {counts['predicted_ok_executed_ok']} agreed, "
              f"{counts['predicted_ok_executed_failed']} FALSE POSITIVE, "
              f"{counts['predicted_bad_executed_ok']} FALSE NEGATIVE, "
              f"{counts['predicted_bad_skipped']} skipped -> {p}")
        if counts["predicted_ok_executed_failed"]:
            print("[ledger] FALSE POSITIVES mean the pre-check certified a path "
                  "the executor does not take (see precheck_path).")
    except Exception as e:
        print(f"[ledger] write FAILED: {e}")
    return doc


# -------------------------------------------------------- IK PROBE ----
# WHY (2026-08-06): the 45 deg run failed with reason_code "ik_no_solution"
# and precheck_path "failed", i.e. mg.plan_single found no trajectory. That
# does NOT establish that the pose is unreachable — plan_single searches from
# ONE seed and has to find a whole collision-free trajectory, so it can miss a
# pose that IK solves easily from a different arm branch.
#
# This probe asks the narrower question directly: does ANY joint solution
# exist for that flange pose? cuRobo's IKSolver is given many random seeds and
# only has to satisfy the pose, not a path. Three probes, run only on points
# that already failed, so a healthy run costs nothing:
#
#   rolled     the real target      -> if this succeeds, the POSE is fine and
#                                      the problem is the planner/seed
#   upright    same position, no roll -> if this succeeds but rolled does not,
#                                      the ORIENTATION is what cannot be met
#   pad_only   the pad position with the UNROLLED flange offset
#                                   -> tells us whether the 111 mm flange
#                                      swing is what put it out of reach
#
# Everything is wrapped: if this cuRobo build exposes a different IK API, the
# probe reports that and the run continues exactly as before.
IK_PROBE = os.environ.get("GRASP_IK_PROBE", "1") == "1"
IK_PROBE_SEEDS = int(os.environ.get("GRASP_IK_PROBE_SEEDS", "400"))
_ik_solver = [None]          # built lazily, once


def _get_ik_solver():
    """cuRobo IKSolver on the SAME robot + world as the planner, or None."""
    if _ik_solver[0] is not None:
        return _ik_solver[0] if _ik_solver[0] is not False else None
    try:
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        cfg = IKSolverConfig.load_from_robot_config(
            rc, _coll_world, rotation_threshold=0.05, position_threshold=0.002,
            num_seeds=IK_PROBE_SEEDS, self_collision_check=True,
            self_collision_opt=True, tensor_args=ta, use_cuda_graph=False)
        _ik_solver[0] = IKSolver(cfg)
        print(f"[ikprobe] IKSolver ready ({IK_PROBE_SEEDS} seeds, "
              f"world={'table slab' if _coll_world is not None else 'None'})")
    except Exception as e:
        print(f"[ikprobe] unavailable ({type(e).__name__}: {e}) — probe skipped")
        _ik_solver[0] = False
        return None
    return _ik_solver[0]


def _ik_try(pos_base, quat, label):
    """One IK query. Returns a small dict; never raises."""
    solver = _get_ik_solver()
    if solver is None:
        return {"ran": False, "reason": "IKSolver unavailable"}
    try:
        g = Pose(position=ta.to_device(np.asarray(pos_base, np.float32)).view(1, 3),
                 quaternion=ta.to_device(np.asarray(quat, np.float32)).view(1, 4))
        r = solver.solve_single(g)
        ok = bool(r.success.view(-1)[0].item())
        out = {"ran": True, "success": ok}
        try:
            out["n_seeds_converged"] = int(r.success.sum().item())
        except Exception:
            pass
        for key, attr in (("pos_err_mm", "position_error"),
                          ("rot_err_deg", "rotation_error")):
            try:
                v = float(getattr(r, attr).view(-1)[0].item())
                out[key] = v * 1000.0 if attr == "position_error" else \
                    float(np.degrees(v))
            except Exception:
                pass
        if ok:
            try:
                q = r.solution.view(-1, len(ARM_JOINT_NAMES))[0].cpu().numpy()
                out["q_rad"] = [round(float(v), 6) for v in q]
            except Exception:
                pass
        print(f"  [ikprobe] {label}: {'SOLVED' if ok else 'no solution'}"
              + (f"  (pos_err {out['pos_err_mm']:.2f} mm)"
                 if "pos_err_mm" in out else ""))
        return out
    except Exception as e:
        print(f"  [ikprobe] {label}: FAILED ({type(e).__name__}: {e})")
        return {"ran": False, "reason": f"{type(e).__name__}: {e}"}


def ik_probe(ee_world, tag):
    """Run the three probes for one failed point. Returns a dict for the report."""
    if not IK_PROBE:
        return {"ran": False, "reason": "GRASP_IK_PROBE=0"}
    ee_world = np.asarray(ee_world, float)
    pad_world = pad_from_ee_target(ee_world)
    ee_upright = pad_world + np.array([0.0, 0.0, TOOL_OFFSET_Z])
    out = {
        "rolled":   _ik_try(world_to_base(ee_world), tq, f"{tag} rolled"),
        "upright":  _ik_try(world_to_base(ee_world), tq_base, f"{tag} upright-orient"),
        "pad_only": _ik_try(world_to_base(ee_upright), tq_base, f"{tag} unrolled-flange"),
    }
    # 4th probe: the SAME rolled pose with the target object removed. If it
    # solves only then, the object is what blocked it — the one thing that
    # separates "out of reach" from "would hit the rod". Only meaningful once
    # the object is actually in the world.
    no_obj = None
    if USE_COLLISION_WORLD and OBJECT_COLLISION and not out["rolled"].get("success"):
        try:
            set_object_collision(False, "IK probe: is the OBJECT the blocker?")
            no_obj = _ik_try(world_to_base(ee_world), tq, f"{tag} rolled, no object")
        finally:
            set_object_collision(True, "IK probe done")
        out["rolled_no_object"] = no_obj

    # one-line verdict, so the JSON does not have to be read to know what to do
    r, u, p = (out["rolled"].get("success"), out["upright"].get("success"),
               out["pad_only"].get("success"))
    if no_obj is not None and no_obj.get("success") and not r:
        out["verdict"] = ("BLOCKED BY THE OBJECT — the same pose solves with "
                          "the target object removed, so this is a collision, "
                          "not a reach limit")
        print(f"  [ikprobe] {tag} VERDICT: {out['verdict']}")
        return out
    if r:
        v = ("POSE IS REACHABLE — IK solves it, so the PLANNER/seed failed, "
             "not the geometry")
    elif u and not r:
        v = ("position reachable but this ORIENTATION is not — the roll itself "
             "is what cannot be met at this pose")
    elif p and not u:
        v = ("only the UNROLLED flange is reachable — the flange swing from the "
             "roll is what puts it out of reach")
    elif r is None:
        v = "probe did not run"
    else:
        v = "nothing reachable here — the pad position itself is out of workspace"
    out["verdict"] = v
    print(f"  [ikprobe] {tag} VERDICT: {v}")
    return out


def precheck_reachability(grid_points, q_home):
    """Dry-run EVERY grid point BEFORE any motion. Writes reachability_report.json.
    Returns {index: bool}. Nothing moves."""
    print("\n" + "=" * 60)
    print(f"[reach] PRE-CHECK {len(grid_points)} grid points (no motion) ...")
    report = {
        "generated": _stamp,
        "config": _args.config,
        "object_center_mm": [round(v*1000, 2) for v in OBJ_CENTER],
        "diameter_mm": OBJ_DIAM_MM,
        "TOOL_OFFSET_Z": TOOL_OFFSET_Z,
        "settings": {"max_joint_step": MAX_JOINT_STEP, "max_steps_cap": MAX_STEPS_CAP,
                     "cond_max_warn": COND_MAX_WARN, "pos_tol_m": POS_TOL_M,
                     "rot_tol_deg": ROT_TOL_DEG, "manual_limits": MANUAL_LIMITS},
        "points": [],
    }
    ok_map = {}
    # The executor only goes pad-to-pad AFTER a successful grasp, so the
    # pre-check only offers prev_world after a point that passed. None means
    # "approach from where you are", i.e. up-then-down.
    _prev_world = None
    q_seed = np.asarray(q_home, float).copy()
    for gp in grid_points:
        idx = gp["index"]; tag = f"pt{idx:02d}"
        gw = np.array(gp["world"], float)                  # EE target (world)
        pad_world = pad_from_ee_target(gw)      # correct at any roll
        _progress(f"[reach] {tag} IK ...")
        q_goal, path_used = _ik_at(gw, q_seed, f"{tag}:reach-ik",
                                   prev_world=_prev_world)
        if q_goal is None:
            res = {"reachable": False, "reason": "IK/plan failed (out of workspace "
                                                 "or no collision-free solution)",
                   "limit_hits": ["ik_failed"]}
        else:
            res = evaluate_reachability(q_seed, q_goal, MANUAL_LIMITS)




            if not res["reachable"]:
                # direct IK can land a flipped arm branch (e.g. shoulder_lift -3.5 rad)
                # whose straight-line dry-run crosses a singularity. Before giving up,
                # retry via the up-then-down path the real run actually uses.
                up = up_point(gw)
                tr_up = plan_free_move(np.asarray(q_seed, float), world_to_base(up),
                                       f"{tag}:retry-up")
                if tr_up is not None:
                    tr_dn = plan_stitched_z(np.asarray(tr_up[-1], float),
                                            -float(APPROACH_H), f"{tag}:retry-down")
                    if tr_dn is not None:
                        res2 = evaluate_reachability(q_seed,
                                                     np.asarray(tr_dn[-1], float),
                                                     MANUAL_LIMITS)
                        if res2["reachable"]:
                            q_goal, res = np.asarray(tr_dn[-1], float), res2
                            path_used = "up_then_down"

            if res["reachable"]:
                q_seed = q_goal          # chain like the real run does
                _prev_world = np.asarray(gw, float).copy()
            else:
                _prev_world = None
        # Only failed points get probed, so a healthy run costs nothing.
        if not res["reachable"]:
            try:
                res_probe = ik_probe(gw, tag)
            except Exception as _pe:
                res_probe = {"ran": False, "reason": f"{type(_pe).__name__}: {_pe}"}
        else:
            res_probe = None
        ok_map[idx] = bool(res["reachable"])
        row = {"index": idx,
               "pad_offset_y_mm": gp["dy_mm"], "pad_offset_z_mm": gp["dz_mm"],
               "pad_target_world_mm": [round(v*1000, 2) for v in pad_world],
               "ee_target_world_mm":  [round(v*1000, 2) for v in gw.tolist()],
               "q_goal_rad": (q_goal.tolist() if q_goal is not None else None)}
        row.update(res)
        # stable code + which path proved it (see _ik_at). "direct" means the
        # executor's own path was never tested for this point.
        row["reason_code"] = _reach_reason_code(res, path_used)
        row["precheck_path"] = path_used
        if res_probe is not None:
            row["ik_probe"] = res_probe
            row["ik_probe_verdict"] = res_probe.get("verdict")
        _REACH_ROWS[idx] = row
        report["points"].append(row)
        mark = "OK " if res["reachable"] else "XX "
        print(f"  {mark}{tag}  pad(y={gp['dy_mm']:+.1f}, z={gp['dz_mm']:+.1f}) "
              f"-> [{row['reason_code']}] {res['reason']}  "
              f"(path: {path_used})")
        for _a in (res.get("advisory") or []):
            print(f"      advisory (not fatal): {_a}")

    n_ok = sum(1 for v in ok_map.values() if v)
    report["n_points"] = len(grid_points)
    report["n_reachable"] = n_ok
    report["n_unreachable"] = len(grid_points) - n_ok
    # counts per cause, so a run's failures can be read without opening the
    # per-point rows: 12 x singularity is a different problem from 12 x
    # ik_no_solution and needs a different fix.
    by_code, by_path = {}, {}
    for r in report["points"]:
        by_code[r.get("reason_code", "?")] = by_code.get(r.get("reason_code", "?"), 0) + 1
        by_path[r.get("precheck_path", "?")] = by_path.get(r.get("precheck_path", "?"), 0) + 1
    report["by_reason_code"] = by_code
    report["by_precheck_path"] = by_path
    report["grasp_rot_deg"] = float(ROT_DEG)
    report["use_collision_world"] = bool(USE_COLLISION_WORLD)
    report["collision_untested"] = (not USE_COLLISION_WORLD)
    report["collision_model"] = COLLISION_MODEL_INFO
    out = os.path.join(OUTPUT_DIR, "reachability_report.json")
    try:
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[reach] {n_ok}/{len(grid_points)} reachable. report -> {out}")
        print("[reach] by cause: "
              + ", ".join(f"{k}={v}" for k, v in sorted(by_code.items())))
        print("[reach] paths proved: "
              + ", ".join(f"{k}={v}" for k, v in sorted(by_path.items()))
              + "  (these are the routes the executor itself drives)")
        if not USE_COLLISION_WORLD:
            print("[reach] NOTE: USE_COLLISION_WORLD=False, so nothing was "
                  "collision-checked - 'reachable' here does NOT mean lab-safe.")
        _verdicts = [r.get("ik_probe_verdict") for r in report["points"]
                     if r.get("ik_probe_verdict")]
        for _v in sorted(set(_verdicts)):
            print(f"[ikprobe] {_verdicts.count(_v)} point(s): {_v}")
    except Exception as e:
        print(f"[reach] report write FAILED: {e}")
    # also drop a copy next to the config so the GUI always finds the newest
    try:
        side = os.path.join(os.path.dirname(_args.config), "reachability_report.json")
        with open(side, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[reach] GUI copy -> {side}")
    except Exception:
        pass
    print("=" * 60 + "\n")
    return ok_map

# ============================================================
# NaN / physics-blowup WATCHDOG (writes to file every frame so a freeze
# cannot hide the first bad frame). Enabled only around pt00's close.
# ============================================================
import math
_WATCH = {"on": False, "f": None, "bad": False}

def _watch_open(path):
    try:
        _WATCH["f"] = open(path, "w")
        _WATCH["f"].write("phase\tframe\tfinger_joint\tmax_abs_arm\tarm_finite\tgrip_finite\n")
        _WATCH["f"].flush()
        _WATCH["on"] = True
        _WATCH["bad"] = False
    except Exception as e:
        print(f"[watch] open failed: {e}")

def _watch(phase, k):
    if not _WATCH["on"] or _WATCH["f"] is None:
        return
    try:
        qa = np.asarray(robot.get_joint_positions()[ai], dtype=float)
        fj = float(robot.get_joint_positions()[gi[0]]) if gi is not None else float("nan")
        arm_finite = bool(np.all(np.isfinite(qa)))
        grip_finite = bool(math.isfinite(fj))
        max_abs = float(np.max(np.abs(qa))) if arm_finite else float("inf")
        _WATCH["f"].write(f"{phase}\t{k}\t{fj:.6f}\t{max_abs:.4f}\t{arm_finite}\t{grip_finite}\n")
        _WATCH["f"].flush()
        if (not _WATCH["bad"]) and ((not arm_finite) or (not grip_finite) or max_abs > 50.0):
            _WATCH["bad"] = True
            _WATCH["f"].write(f"# FIRST_BAD at phase={phase} frame={k} "
                              f"finger={fj} max_abs_arm={max_abs} "
                              f"arm_finite={arm_finite} grip_finite={grip_finite}\n")
            _WATCH["f"].flush()
    except Exception as e:
        try:
            _WATCH["f"].write(f"# watch read error at {phase} {k}: {e}\n"); _WATCH["f"].flush()
        except Exception:
            pass

def _watch_close():
    if _WATCH["f"] is not None:
        try: _WATCH["f"].close()
        except Exception: pass
    _WATCH["on"] = False; _WATCH["f"] = None

def ramp_gripper(arm_q, target, n_frames):
    cur_g = float(robot.get_joint_positions()[gi[0]])
    for k in range(n_frames):
        alpha = (k + 1) / n_frames
        apply_arm_and_grip(arm_q, grip_val=cur_g + alpha*(target - cur_g))
        world.step(render=True)
        _watch("ramp", k)

def hold_for(arm_q, seconds, _phase="hold"):
    for k in range(int(seconds * 60)):
        apply_arm_and_grip(arm_q)
        world.step(render=True)
        _watch(_phase, k)

# ============================================================
# Run ALL grid points from the config (proven per-point grasp)
# ============================================================
def grasp_one_point(grasp_world, tag, row_marks, pose_hist, dy_m=0.0, dz_m=0.0,
                    direct=False, retreat=True):
    """Grasp at grasp_world: approach -> record -> (optional) retreat.
    Slices this grasp's new tactile rows into BASENAME_<tag>_s1/s2.
    Records the actual reached pad/EE world pose into pose_hist.

    direct=True  -> pad-to-pad straight line from wherever we are (the previous
                    grasp pose, gripper open) to this point. No APPROACH_H lift,
                    no global replan. Falls back to the old path if it fails.
    direct=False -> the original: global free-move to grasp+APPROACH_H, then the
                    Paper-2 stitched descent.
    retreat=True -> ascend APPROACH_H at the end (used for the LAST point, or
                    whenever point-to-point is off)."""
    up_world = up_point(grasp_world)
    up_base  = world_to_base(up_world)

    approached = False
    if direct:
        q_start = robot.get_joint_positions()[ai].copy()
        cpos, _cq = fk(q_start.astype(np.float32))
        cur_world = rotmat(ROBOT_WORLD_QUAT_WXYZ) @ cpos + ROBOT_WORLD_POS
        delta = np.asarray(grasp_world, dtype=float) - np.asarray(cur_world, dtype=float)
        _progress(f"{tag} pad-to-pad PLAN start (|d|={1000*float(np.linalg.norm(delta)):.1f} mm)")
        print(f"[{tag}] pad-to-pad move {np.round(1000*delta,2)} mm (no lift) ...")
        traj_pp = plan_stitched_line(q_start, delta, f"{tag}:pad-to-pad")
        if traj_pp is not None:
            _progress(f"{tag} pad-to-pad EXEC start ({len(traj_pp)} wpts)")
            run_traj(traj_pp)
            _progress(f"{tag} pad-to-pad EXEC done")
            approached = True
        else:
            _progress(f"{tag} pad-to-pad FAILED -> falling back to lift+descend")
            print(f"[{tag}] pad-to-pad failed; using the old approach instead.")

    if not approached:
        q_start = robot.get_joint_positions()[ai].copy()
        _progress(f"{tag} free-move PLAN start")
        print(f"[{tag}] free move to UP {up_world.round(4)} ...")
        traj_up = plan_free_move(q_start, up_base, f"{tag}:to-up")
        if traj_up is None:
            _progress(f"{tag} free-move PLAN FAILED")
            print(f"[{tag}] FAILED to reach UP.")
            _ledger(_tag_index(tag), "free_move_to_up", False); return False
        _progress(f"{tag} free-move EXEC start ({len(traj_up)} wpts, "
                  f"max|dq|={float(np.max(np.abs(traj_up[-1]-traj_up[0]))):.3f} rad)")
        run_traj(traj_up)
        _progress(f"{tag} free-move EXEC done")
        q_up = robot.get_joint_positions()[ai].copy()

        _progress(f"{tag} descent PLAN start")
        print(f"[{tag}] descent UP->GRASP ...")
        dz_dn = -float(np.linalg.norm(grasp_world - up_world))
        traj_dn = plan_stitched_z(q_up, dz_dn, f"{tag}:DOWN")
        if traj_dn is None:
            _progress(f"{tag} descent PLAN FAILED")
            print(f"[{tag}] descent FAILED.")
            _ledger(_tag_index(tag), "stitched_descent", False); return False
        _progress(f"{tag} descent EXEC start ({len(traj_dn)} wpts)")
        run_traj(traj_dn)
        _progress(f"{tag} descent EXEC done")
        # ---- FINAL TRIM (descent path only): close the residual EE error ----
        # The stitched descent settles ~1 mm short (settle-loop tolerance).
        # Measure reached-vs-commanded EE and close the gap with ONE short
        # stitched line (the same move that lands pad-to-pad at +-0.03 mm).
        q_now = robot.get_joint_positions()[ai].copy()
        cpos_t, _ = fk(q_now.astype(np.float32))
        cur_world = rotmat(ROBOT_WORLD_QUAT_WXYZ) @ cpos_t + ROBOT_WORLD_POS
        resid = np.asarray(grasp_world, float) - np.asarray(cur_world, float)
        resid_mm = 1000.0 * float(np.linalg.norm(resid))
        if resid_mm > 0.2:
            _progress(f"{tag} descent TRIM start (residual {resid_mm:.2f} mm)")
            traj_fix = plan_stitched_line(q_now, resid, f"{tag}:trim", n_steps=3)
            if traj_fix is not None:
                run_traj(traj_fix)
                _progress(f"{tag} descent TRIM done")
                q_now = robot.get_joint_positions()[ai].copy()
                cpos_t, _ = fk(q_now.astype(np.float32))
                cur_world = rotmat(ROBOT_WORLD_QUAT_WXYZ) @ cpos_t + ROBOT_WORLD_POS
                resid_mm = 1000.0 * float(np.linalg.norm(
                    np.asarray(grasp_world, float) - np.asarray(cur_world, float)))
            else:
                _progress(f"{tag} descent TRIM plan failed "
                          f"(residual {resid_mm:.2f} mm)")

        # ---- DO NOT CLOSE ON A DESCENT THAT DID NOT ARRIVE (2026-08-07) ----
        # This used to print "keeping residual" and close the fingers anyway.
        # On the pad y=+80, 90 deg run the pads had to pass THROUGH the rod:
        # every stitched step landed short against the obstacle, 47 mm of
        # error accumulated, the trim failed, and the gripper then closed
        # while jammed against the cylinder and shoved it across the table —
        # and the ledger recorded exec_stage "complete", 1/1 grasps OK.
        #
        # A descent that ends far from its target has not reached the grasp
        # pose, so whatever the pads close on is not the pose that was asked
        # for and the tactile map would be mislabelled. Stop, and let the
        # ledger record it as the FALSE POSITIVE it is.
        if resid_mm > MAX_DESCENT_RESIDUAL_MM:
            print(f"[{tag}] descent ARRIVED {resid_mm:.1f} mm SHORT of the "
                  f"grasp pose (limit {MAX_DESCENT_RESIDUAL_MM:.1f} mm) — NOT "
                  f"closing. Most likely something is in the way: check the "
                  f"pad target against the object in gui_preview.png.")
            _progress(f"{tag} descent residual {resid_mm:.1f} mm -> ABORT")
            _ledger(_tag_index(tag), "descent_residual", False)
            return False
    q_grasp = robot.get_joint_positions()[ai].copy()
    hold_qg = q_grasp.astype(np.float32)

    # record the ACTUAL EE + PAD pose reached at grasp (for verification + replay)
    ee_pos, ee_quat = fk(q_grasp.astype(np.float32))
    R_wb = rotmat(ROBOT_WORLD_QUAT_WXYZ)
    ee_world = (R_wb @ ee_pos + ROBOT_WORLD_POS).tolist()
    pad_pos, pad_R = pad_pose_from_joints(q_grasp)
    # GUI TARGET pad pose = what the GUI actually designed = object centre + (0, dy, dz).
    # (dy_m, dz_m are pad offsets FROM THE OBJECT CENTRE.)
    # Old 'desired = INIT_PAD_POS + offset' was WRONG: INIT_PAD_POS is the HOME pad
    # pose, but the offsets are from the OBJECT centre -> it faked a constant
    # ~(0, +150, +199) mm miss on every grasp. This is the bookkeeping bug.
    gui_target = [OBJ_CENTER[0], OBJ_CENTER[1] + dy_m, OBJ_CENTER[2] + dz_m]
    pose_hist.append({
        "tag": tag,
        "ee_world_m": ee_world,
        "joints_rad": q_grasp.tolist(),
        "pad_actual_pos_m": pad_pos.tolist(),   # FK + FIXED wrist->pad const (BLIND to finger swing)
        "pad_actual_R": pad_R.tolist(),
        "pad_gui_target_m": gui_target,         # what the GUI designed
        "pad_desired_pos_m": gui_target,        # same (kept for older tools)
        "pad_initial_pos_m": INIT_PAD_POS.tolist(),
    })

    if tag == "pt00":
        _watch_open(os.path.join(OUTPUT_DIR, "nan_watch.tsv"))
    hold_for(hold_qg, WAIT_GRASP_SECONDS, _phase="settle_pre_close")
    # ---- PAD-TRUTH PROBE (pt00 only): measure the finger swing directly ----
    # We read the RIGID finger links the pads are bolted to (confirmed by the
    # stage dump). NOTE the DOUBLED 'robot_gripper_adapter_sensor' in the path —
    # the earlier probe used the single-path sensor prim, which does not exist,
    # so every reader returned "prim path not valid" / zeros. These finger links
    # are articulation bodies with valid world transforms.
    _GRIP_ROOT = ("/World/robot_gripper_adapter_sensor/robot_gripper_adapter_sensor/"
                  "Robotiq_2F_85_adapter_fixed_v_sibling__1_/Robotiq_2F_85_modified/"
                  "Robotiq_2F_85")
    PROBE = (tag == "pt00")

    # ---- find the REAL sensor bodies (the 'Case' rigid prims) ---------------
    # ".../TSF_85_right/TSF_85" is an EMPTY Xform wrapper -> it never moves.
    # The actual rigid body is TWO levels deeper (".../TSF_85/TSF_85/Case"),
    # bolted to the finger. Path depth differs between raw USD and the referenced
    # stage, so SEARCH for it. THIS is what fills TSF_*_CASE for the calibration.
    def _find_case_prims():
        found = {}
        try:
            for prim in world.stage.Traverse():
                pth = str(prim.GetPath())
                if prim.GetName() != "Case" or "TSF_85" not in pth:
                    continue
                if "TSF_85_right" in pth:
                    found.setdefault("TSF_right_CASE", pth)
                elif "TSF_85_left" in pth:
                    found.setdefault("TSF_left_CASE", pth)
        except Exception as _fe:
            print(f"[{tag}] case-prim search failed: {_fe}")
        return found

    _cases = _find_case_prims()
    if _cases:
        print(f"[{tag}] sensor Case prims found: {list(_cases.values())}")
    else:
        print(f"[{tag}] WARNING: no TSF 'Case' prim found -- pad pose unavailable")

    _probe_prims = {
        "right_inner_finger": f"{_GRIP_ROOT}/right_inner_finger",   # s1 pad link
        "left_inner_finger":  f"{_GRIP_ROOT}/left_inner_finger",    # s2 pad link
        # PALM: the gripper body the rod top can collide with. We add it because
        # the estimated "palm ~ pad + 110mm" was contradicted by reality (a tip
        # grid clears, a centre grid strikes) -> measure it, don't estimate it.
        "gripper_base_link":  f"{_GRIP_ROOT}/base_link",
        # OBJECT: bolted with a COMPLIANT PhysX FixedJoint, so a knock makes it
        # lean and STAY leaning. Logging it proves whether the rod moved.
        "object":  "/World/robot_gripper_adapter_sensor/Object_02",
        # the OLD (wrong) wrapper prims, kept only to prove they are static:
        "TSF_right_wrapper": ("/World/robot_gripper_adapter_sensor/"
                              "robot_gripper_adapter_sensor/TSF_85_right/TSF_85"),
        "TSF_left_wrapper":  ("/World/robot_gripper_adapter_sensor/"
                              "robot_gripper_adapter_sensor/TSF_85_left/TSF_85"),

    }
    _probe_prims.update(_cases)      # <-- the REAL sensor bodies (fills TSF_*_CASE)
    _probe = {}
    if PROBE:
        _probe["open_grip"] = _probe_all(_probe_prims)   # BEFORE closing
    _tsf.set("/exts/TSF_85_Ext/record_active", True)
    _progress(f"{tag} CLOSE start")
    print(f"[{tag}] [RECORD ON] close -> hold -> open ...")
    ramp_gripper(hold_qg, CLOSE_RAD, GRIPPER_RAMP_FRAMES)
    _progress(f"{tag} CLOSE done, holding")
    hold_for(hold_qg, WAIT_HOLD_SECONDS, _phase="hold_closed")
    if PROBE:
        _probe["closed_grip"] = _probe_all(_probe_prims)  # AFTER closing (has the swing)
        _probe["finger_joint_rad"] = (float(robot.get_joint_positions()[gi[0]])
                                      if gi is not None else None)
        # DIAGNOSTIC: the collector only ever commands ONE joint ('finger_joint',
        # which drives the LEFT outer knuckle). The RIGHT side has its own joint
        # (right_outer_knuckle_joint) that nothing commands. If the USD does not
        # mimic it, that side is passive -> one pad presses, the other hangs,
        # which is exactly the s1=20023 / s2=276 pattern we keep seeing.
        # Log EVERY gripper-ish joint so we can see whether both sides moved.
        try:
            _qall = robot.get_joint_positions()
            _probe["all_gripper_joints_rad_closed"] = {
                _n: round(float(_qall[_i]), 5)
                for _i, _n in enumerate(robot.dof_names)
                if any(_k in _n.lower() for _k in
                       ("finger", "knuckle", "grip"))
            }
        except Exception as _je:
            _probe["all_gripper_joints_rad_closed"] = f"ERR {_je}"
        _probe["ee_world_m"] = ee_world
        _probe["pad_actual_fk_m"] = pad_pos.tolist()
        _probe["gui_target_m"] = gui_target
        _probe["TOOL_OFFSET_Z_used_m"] = TOOL_OFFSET_Z

        # ---- IS THE TOOL ACTUALLY PERPENDICULAR TO THE ROD? -----------------
        # Measured directly from the joints, NOT from pad_pose_from_joints()
        # (whose frame is wrong). Reports the commanded vs reached orientation
        # and how far each tool axis is from its intended world axis. A tilt seen
        # while looking ALONG the squeeze axis is a rotation about world X and
        # CANNOT come from the finger swing -> it must be tool orientation.
        try:
            _q_now = robot.get_joint_positions()[ai].copy()
            _p_b, _q_b = fk(_q_now)                       # EE pose in BASE frame
            _R_ee_w = rotmat(ROBOT_WORLD_QUAT_WXYZ) @ rotmat(_q_b)   # -> WORLD
            _R_cmd  = rotmat(ROBOT_WORLD_QUAT_WXYZ) @ rotmat(tq)     # commanded
            _probe["ee_quat_base_reached_wxyz"] = [float(v) for v in _q_b]
            _probe["ee_quat_base_commanded_wxyz"] = [float(v) for v in tq]
            _probe["tool_axes_world"] = {
                "tool_x": [round(float(v), 5) for v in _R_ee_w[:, 0]],
                "tool_y": [round(float(v), 5) for v in _R_ee_w[:, 1]],
                "tool_z": [round(float(v), 5) for v in _R_ee_w[:, 2]],
            }
            # total orientation error, reached vs commanded
            _dR = _R_ee_w.T @ _R_cmd
            _ang = float(np.degrees(np.arccos(
                np.clip((np.trace(_dR) - 1.0) * 0.5, -1.0, 1.0))))
            _probe["orientation_error_vs_commanded_deg"] = round(_ang, 4)
            # the rod is vertical (world Z). the tool axis that should be
            # PARALLEL to the rod is the one the fingers run along.
            for _nm, _v in (("tool_x", _R_ee_w[:, 0]), ("tool_y", _R_ee_w[:, 1]),
                            ("tool_z", _R_ee_w[:, 2])):
                _tilt = float(np.degrees(np.arccos(np.clip(abs(_v[2]), -1.0, 1.0))))
                _probe[f"{_nm}_angle_from_world_Z_deg"] = round(_tilt, 4)
        except Exception as _oe:
            _probe["orientation_measure_error"] = str(_oe)

        # ---- LIVE PAD POSE: is the Case live, and the measured EE->pad offset --
        # THIS writes the TSF_*_CASE_is_live / EE_to_pad fields the calibration
        # reads. Without it, calibration finds nothing and falls back.
        try:
            def _cw(grip, name):
                v = _probe.get(grip, {}).get(name, {}).get("UsdGeom.XformCache")
                return np.array(v, dtype=float) if isinstance(v, list) else None
            _eew = np.array(ee_world, dtype=float)
            for _side in ("TSF_right_CASE", "TSF_left_CASE"):
                _op, _cl = _cw("open_grip", _side), _cw("closed_grip", _side)
                if _cl is None:
                    _probe[f"{_side}_status"] = "prim not found / no xform"
                    continue
                _probe[f"{_side}_closed_world_mm"] = (_cl*1000).round(3).tolist()
                _probe[f"{_side}_EE_to_pad_mm"]   = ((_cl - _eew)*1000).round(3).tolist()
                if _op is not None:
                    _d = (_cl - _op)*1000
                    _probe[f"{_side}_moved_open_to_closed_mm"] = _d.round(3).tolist()
                    _probe[f"{_side}_is_live"] = bool(np.max(np.abs(_d)) > 0.5)
        except Exception as _pe:
            _probe["ee_to_pad_measure_error"] = str(_pe)

        # ---- MEASURED palm clearance + object movement ----------------------
        # Answers two things by measurement instead of estimate:
        #  1) how far above the pad the PALM really sits (my 110mm was a guess),
        #  2) whether the ROD actually moved/leaned during the grasp (the 2F-85
        #     has ONE drive joint, so both fingers move symmetrically about the
        #     gripper centreline -- they CANNOT self-centre. Any rod offset ->
        #     one pad crushed, the other untouched, at ANY close angle.)
        try:
            def _z(grip, name):
                v = _probe.get(grip, {}).get(name, {}).get("UsdGeom.XformCache")
                return float(v[2]) if isinstance(v, list) else None
            def _xyz(grip, name):
                v = _probe.get(grip, {}).get(name, {}).get("UsdGeom.XformCache")
                return [float(c) for c in v] if isinstance(v, list) else None

            _palm_z = _z("closed_grip", "gripper_base_link")
            _pad_z  = float(gui_target[2])
            if _palm_z is not None:
                _probe["measured_palm_above_pad_mm"] = round((_palm_z - _pad_z) * 1000, 1)
                _rod_len_m = float(CONFIG["object"].get("length_mm", 140.0)) / 1000.0
                _rod_top_m = OBJ_CENTER[2] + _rod_len_m / 2.0
                _probe["rod_top_world_m"] = round(_rod_top_m, 4)
                _probe["measured_palm_clearance_mm"] = round((_palm_z - _rod_top_m) * 1000, 1)
                _probe["palm_strikes_rod"] = bool(_palm_z < _rod_top_m)


            _o_open  = _xyz("open_grip",   "object")
            _o_close = _xyz("closed_grip", "object")
            if _o_open and _o_close:
                _d = [round((_o_close[i] - _o_open[i]) * 1000, 2) for i in range(3)]
                _probe["object_moved_during_close_mm"] = _d
                _probe["object_moved"] = bool(max(abs(v) for v in _d) > 0.5)
        except Exception as _me:
            _probe["clearance_measure_error"] = str(_me)

        # ---- OFFSET SOLVER --------------------------------------------------
        # We command  EE_z = pad_target_z + TOOL_OFFSET_Z, and the pad ends up
        # at  pad_z = EE_z - offset_real.  To make pad land on target we need
        #   TOOL_OFFSET_Z := offset_real = EE_z - pad_z.
        # Two estimates of pad_z:
        #   (1) FK pad (open-grip, blind to swing) -> reliable in Z (matches eye).
        #   (2) finger-link Z change open->closed (usdrt, LIVE) -> the swing.
        try:
            _ee_z = float(ee_world[2])
            _pad_fk_z = float(pad_pos[2])
            _off_open = _ee_z - _pad_fk_z            # offset ignoring swing
            _probe["suggested_TOOL_OFFSET_Z_open_m"] = round(_off_open, 5)

            def _live_z(side, grip):  # pull the LIVE usdrt Z if present
                try:
                    v = _probe[grip][side].get("usdrt.world")
                    return float(v[2]) if isinstance(v, list) else None
                except Exception:
                    return None
            _swings = []
            for _s in ("right_inner_finger", "left_inner_finger"):
                zo, zc = _live_z(_s, "open_grip"), _live_z(_s, "closed_grip")
                if zo is not None and zc is not None:
                    _swings.append(zc - zo)
            if _swings:
                _swing_dz = float(np.mean(_swings))     # pad moves with the link
                _probe["measured_finger_swing_dz_m"] = round(_swing_dz, 5)
                _probe["suggested_TOOL_OFFSET_Z_closed_m"] = round(_off_open - _swing_dz, 5)
            else:
                _probe["measured_finger_swing_dz_m"] = "usdrt not live — using open-grip value"
        except Exception as _oe:
            _probe["offset_solver_error"] = str(_oe)

        try:
            with open(os.path.join(OUTPUT_DIR, "pad_truth_probe.json"), "w") as _pf:
                json.dump(_probe, _pf, indent=2)
            print(f"[{tag}] pad_truth_probe.json written")
        except Exception as _pe:
            print(f"[{tag}] pad_truth_probe write FAILED ({_pe})")
    _progress(f"{tag} OPEN start")
    ramp_gripper(hold_qg, GRIPPER_OPEN, GRIPPER_RAMP_FRAMES)
    _tsf.set("/exts/TSF_85_Ext/record_active", False)
    if tag == "pt00":
        _watch_close()
    _progress(f"{tag} OPEN done (record off)")
    print(f"[{tag}] [RECORD OFF]")

    if retreat:
        _progress(f"{tag} ascent start")
        print(f"[{tag}] ascent GRASP->UP ...")
        dz_up = float(np.linalg.norm(up_world - grasp_world))
        traj_up2 = plan_stitched_z(q_grasp, dz_up, f"{tag}:UP")
        if traj_up2 is not None:
            run_traj(traj_up2)
        _progress(f"{tag} ascent done")
    else:
        # stay at grasp height: the next point moves here pad-to-pad
        _progress(f"{tag} no ascent (staying at grasp height for next point)")
        print(f"[{tag}] staying at grasp height (pad-to-pad to next point).")

    # slice out THIS grasp's new tactile rows (extension appends to BASENAME_s1/s2)
    import time as _t
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
        with open(dst, "w") as f:
            f.write(header); f.writelines(body[prev:])
        row_marks[s] = len(body)
        print(f"[{tag}] saved {os.path.basename(dst)} ({len(body)-prev} rows)")
    _ledger(_tag_index(tag), "complete", True)
    return True

EXIT_CODE = 0
row_marks = {}
pose_hist = []
try:
    # Read the INITIAL pad pose ONCE (before any grasp) — this is the reference
    # for all DESIRED poses. desired[i] = initial_pad + grid_offset[i].
    # Computed from the STARTUP JOINTS with the same joints->FK->pad chain as
    # the per-grasp actual pose, so desired-vs-actual differences are meaningful.
    q0 = robot.get_joint_positions()[ai].copy()
    INIT_PAD_POS, INIT_PAD_R = pad_pose_from_joints(q0)
    print(f"[cfg] initial pad pose (world m, from joints/FK): {INIT_PAD_POS.round(4)}")

    n_ok = 0
    _prev_ok = False          # only go pad-to-pad if we KNOW where the arm is
    _n_pts = len(GRID_POINTS)
    print(f"[cfg] POINT_TO_POINT = {POINT_TO_POINT} "
          f"({'pad-to-pad between points, no lift' if POINT_TO_POINT else 'old lift+descend per point'})")

    # ---- REACHABILITY PRE-CHECK (before any motion) ----
    _reach = {}
    if REACH_CHECK or REACH_ONLY:
        _q_home = robot.get_joint_positions()[ai].copy()
        _reach = precheck_reachability(GRID_POINTS, _q_home)
        if REACH_ONLY:
            print("[reach] GRASP_REACH_ONLY=1 -> report written, no motion. Exiting.")
            sys.exit(0)          # the finally: below still closes Isaac cleanly
        _bad = [i for i, v in _reach.items() if not v]
        if _bad and REACH_SKIP:
            print(f"[reach] {len(_bad)} point(s) will be SKIPPED: "
                  f"{['pt%02d' % i for i in _bad]}")
        elif _bad:
            print(f"[reach] {len(_bad)} unreachable, but GRASP_REACH_SKIP=0 -> "
                  f"attempting anyway.")

    for _i, gp in enumerate(GRID_POINTS):
        tag = f"pt{gp['index']:02d}"
        gw  = np.array(gp["world"])
        # reachability pre-check said no -> skip (already logged in the report)
        if REACH_SKIP and (gp["index"] in _reach) and not _reach[gp["index"]]:
            print(f"\n========== {tag}  SKIPPED (unreachable — see "
                  f"reachability_report.json) ==========")
            _prev_ok = False          # next point must re-approach from wherever we are
            continue
        # first point: approach from home the old way. after a good grasp: go
        # straight to the next pad pose. only the last point retreats.
        _direct  = bool(POINT_TO_POINT and _i > 0 and _prev_ok)
        _retreat = bool((not POINT_TO_POINT) or _i == _n_pts - 1)
        _progress(f"===== {tag} START (pad y={gp['dy_mm']:+.1f} z={gp['dz_mm']:+.1f} mm) =====")
        print(f"\n========== {tag}  (pad offset y={gp['dy_mm']:+.1f} z={gp['dz_mm']:+.1f} mm) "
              f"[{'pad-to-pad' if _direct else 'approach'}] ==========")
        ok = grasp_one_point(gw, tag, row_marks, pose_hist,
                             dy_m=gp["dy_mm"]/1000.0, dz_m=gp["dz_mm"]/1000.0,
                             direct=_direct, retreat=_retreat)
        _prev_ok = bool(ok)
        if ok:
            n_ok += 1
        else:
            print(f"[{tag}] skipped (motion failed).")

    # save the pose history (real reached poses) + copy the config into the run folder
    import json as _json, shutil as _sh
    with open(os.path.join(OUTPUT_DIR, "pose_history.json"), "w") as f:
        _json.dump({"points": pose_hist, "config": CONFIG}, f, indent=2)
    # what the pre-check predicted vs what the run actually did (diagnostic)
    try:
        write_execution_ledger(OUTPUT_DIR)
    except Exception as _e:
        print(f"[ledger] skipped ({_e})")
    try:
        _sh.copy(_args.config, os.path.join(OUTPUT_DIR, "gui_config_used.json"))
    except Exception:
        pass
    # The GUI writes its 3-panel preview next to the config every time you press
    # Save Config / Save + Show Run Command. Copy it in so each run folder shows
    # the grid design that produced it, with no screenshot needed.
    try:
        _prev = os.path.join(os.path.dirname(_args.config), "gui_preview.png")
        if os.path.isfile(_prev):
            _sh.copy(_prev, os.path.join(OUTPUT_DIR, "gui_preview.png"))
            print(f"[cfg] gui_preview.png copied into the run folder")
    except Exception:
        pass
    print(f"\n[cfg] DONE. {n_ok}/{len(GRID_POINTS)} grasps OK. Data in {OUTPUT_DIR}")
    print(f"[cfg] pose_history.json written ({len(pose_hist)} poses).")

    # ---- CALIBRATE mode: store the MEASURED live-pad offset ----------------
    # New method: read the LIVE sensor Case poses (proven live) and store the
    # real EE->pad offset. The grasp CENTRE is the midpoint of the two pads; its
    # world offset from the EE is what commanding subtracts. For a symmetric
    # tool-down grasp the x,y of that midpoint offset are ~0, so the scalar
    # TOOL_OFFSET_Z = -offset_z drives the grid unchanged -- but now MEASURED,
    # at a CLEAN (collision-free) grasp, instead of the C_ANCHOR guess.
    # If the Case is not live (older scene), fall back to the old formula so
    # calibration never silently fails.
    if CALIBRATE:
        try:
            with open(os.path.join(OUTPUT_DIR, "pad_truth_probe.json")) as _pf:
                _pt = _json.load(_pf)
            _ee = np.array(_pt["ee_world_m"], dtype=float)

            # ---- CONTACT GATE: did the pads actually touch the object? --------
            # The pad pose reads 'live' even when closing on AIR (the fingers
            # still swing). The only true contact signal is the tactile sum
            # rising (the Roberge-paper test). Read the pt00 tactile peak for
            # BOTH sensors; if neither rises above threshold, REFUSE to store.
            CONTACT_MIN_SUM = 1000.0   # baseline ~250, real contact ~6000
            def _tactile_peak(sensor):
                fn = os.path.join(OUTPUT_DIR, f"{BASENAME}_pt00_{sensor}_tactile_maps.csv")
                if not os.path.exists(fn):
                    fn = os.path.join(OUTPUT_DIR, f"{BASENAME}_{sensor}_tactile_maps.csv")
                if not os.path.exists(fn):
                    return None
                peak = 0.0
                try:
                    with open(fn) as _tf:
                        _r = _csv.reader(_tf); next(_r)
                        for _row in _r:
                            try:
                                peak = max(peak, sum(float(x) for x in _row[2:30]))
                            except Exception:
                                pass
                except Exception:
                    return None
                return peak
            _pk1, _pk2 = _tactile_peak("s1"), _tactile_peak("s2")
            _peak = max([v for v in (_pk1, _pk2) if v is not None] or [0.0])
            _contact = _peak >= CONTACT_MIN_SUM
            print(f"[cal] contact check: tactile peak s1={_pk1} s2={_pk2} "
                  f"-> {'CONTACT' if _contact else 'NO CONTACT'} (thr {CONTACT_MIN_SUM})")
            if not _contact:
                print(f"\n[cal] REFUSING TO STORE: the pads did NOT contact the object "
                      f"(tactile peak {_peak:.0f} < {CONTACT_MIN_SUM:.0f}).")
                print(f"[cal] The grasp closed on air — move the pad ONTO the rod body "
                      f"(a Z on the object, not past its end) and calibrate again.")
                print(f"[cal] Nothing was written; the previous calibration (if any) is kept.")
                EXIT_CODE = 4
                raise _CalNoContact()   # skip the store below

            # ---- PEAK BAND CHECK ------------------------------------------
            # Contact alone is not enough: CLOSE_RAD on a NEW diameter is a
            # hand estimate, and a bad one still makes contact — just far too
            # light or hard enough to crush. Either way the pad face ends up
            # at the wrong height, so the TOOL_OFFSET_Z we are about to store
            # would be wrong AND the bad close_rad would be silently reused
            # for every future run. Compare against the verified 26 mm grasp.
            _ref = 13201.0
            try:
                _ref = float(_CAL.get("26.0", {}).get("tactile_peak_sum", _ref))
            except Exception:
                pass
            _lo, _hi = 0.5 * _ref, 2.0 * _ref
            if not (_lo <= _peak <= _hi) and os.environ.get("GRASP_CAL_FORCE") != "1":
                _too_light = _peak < _lo
                # span ~= 85 - 106*rad (mm), so 0.01 rad ~= 1.06 mm of squeeze
                _step = 0.03 if abs(_peak - _ref) > _ref else 0.015
                _sugg = CLOSE_RAD + (_step if _too_light else -_step)
                print(f"\n[cal] REFUSING TO STORE: tactile peak {_peak:.0f} is "
                      f"{'TOO LOW' if _too_light else 'TOO HIGH'} — outside the "
                      f"sane band {_lo:.0f}..{_hi:.0f} (26 mm reference {_ref:.0f}).")
                print(f"[cal] CLOSE_RAD = {CLOSE_RAD:.4f} rad "
                      f"{'barely touches' if _too_light else 'over-compresses'} "
                      f"the {OBJ_DIAM_MM:.1f} mm object.")
                print(f"[cal] Re-run calibrate with "
                      f"GRASP_CLOSE_RAD={_sugg:.3f}  "
                      f"({'+' if _too_light else '-'}{_step:.3f} rad = "
                      f"{'+' if _too_light else '-'}{_step*106:.1f} mm of squeeze).")
                print(f"[cal] Nothing was written; the previous calibration "
                      f"(if any) is kept.  Override with GRASP_CAL_FORCE=1.")
                EXIT_CODE = 5
                raise _CalNoContact()   # skip the store below

            # ---- OBJECT MUST NOT HAVE MOVED -------------------------------
            # If the rod shifted during the calibrate grasp, the pad pose we
            # are about to measure belongs to a geometry that no longer holds.
            if _pt.get("object_moved") is True and os.environ.get("GRASP_CAL_FORCE") != "1":
                _mv = _pt.get("object_moved_during_close_mm")
                print(f"\n[cal] REFUSING TO STORE: the object MOVED during the "
                      f"calibrate grasp ({_mv} mm).")
                print(f"[cal] The measured pad pose does not correspond to a "
                      f"stable object. Check the fixed joint / supports, then "
                      f"calibrate again.  Override with GRASP_CAL_FORCE=1.")
                EXIT_CODE = 6
                raise _CalNoContact()

            def _case_closed(side):
                v = _pt.get("closed_grip", {}).get(side, {}).get("UsdGeom.XformCache")
                return np.array(v, dtype=float) if isinstance(v, list) else None
            _pr = _case_closed("TSF_right_CASE")
            _pl = _case_closed("TSF_left_CASE")
            _live = (_pt.get("TSF_right_CASE_is_live") is True and
                     _pt.get("TSF_left_CASE_is_live") is True and
                     _pr is not None and _pl is not None)

            if _live:
                _pad_mid   = 0.5 * (_pr + _pl)              # grasp centre = CASE ORIGIN (world)
                _off_world = _pad_mid - _ee                 # EE -> case-origin (world)
                _off_case  = round(float(-_off_world[2]), 5)               # case origin
                # shift UP to the pad CENTRE (Case origin is at the pad's end):
                _offset    = round(_off_case + PAD_CENTER_ABOVE_CASE_M, 5)  # pad CENTRE
                _CAL[_diam_key] = {
                    "diameter_mm": OBJ_DIAM_MM,
                    "method": "measured_live_pad",
                    "TOOL_OFFSET_Z": _offset,                 # targets pad CENTRE (m)
                    "TOOL_OFFSET_Z_case_origin": _off_case,   # raw, targets Case origin
                    "pad_center_above_case_m": PAD_CENTER_ABOVE_CASE_M,
                    "ee_to_grasp_center_offset_world_m": [round(float(v), 5) for v in _off_world],
                    "ee_to_pad_right_world_m": [round(float(v), 5) for v in (_pr - _ee)],
                    "ee_to_pad_left_world_m":  [round(float(v), 5) for v in (_pl - _ee)],
                    "pad_right_closed_world_m": [round(float(v), 5) for v in _pr],
                    "pad_left_closed_world_m":  [round(float(v), 5) for v in _pl],
                    "ee_world_m": [round(float(v), 5) for v in _ee],
                    "close_rad": CLOSE_RAD,
                    "finger_joint_rad": _pt.get("finger_joint_rad"),
                    "tactile_peak_sum": round(float(_peak), 1),
                    "measured_at": _stamp,
                }
                print(f"\n[cal] CALIBRATED (measured live pad) diameter {OBJ_DIAM_MM} mm")
                print(f"[cal]   case-origin offset = {_off_case}  + pad-centre shift "
                      f"{PAD_CENTER_ABOVE_CASE_M} -> TOOL_OFFSET_Z = {_offset}")
                print(f"[cal]   EE->grasp-centre offset (world mm) = "
                      f"{(_off_world*1000).round(2).tolist()}")
                print(f"[cal]   TOOL_OFFSET_Z (scalar, m)          = {_offset}")
            else:
                # ---- fallback: old inner-finger + C_ANCHOR formula ----
                _link_z = float(_pt["closed_grip"]["right_inner_finger"]["UsdGeom.XformCache"][2])
                _offset = round((float(_ee[2]) - _link_z) + C_ANCHOR, 5)
                _CAL[_diam_key] = {
                    "diameter_mm": OBJ_DIAM_MM, "method": "fallback_C_ANCHOR",
                    "TOOL_OFFSET_Z": _offset, "ee_z": round(float(_ee[2]), 5),
                    "closed_inner_finger_z": round(_link_z, 5), "C_ANCHOR": C_ANCHOR,
                    "close_rad": CLOSE_RAD, "measured_at": _stamp,
                }
                print(f"\n[cal] Case NOT live -> FALLBACK formula. "
                      f"diameter {OBJ_DIAM_MM} mm -> TOOL_OFFSET_Z = {_offset}")

            os.makedirs(os.path.dirname(CAL_FILE), exist_ok=True)
            with open(CAL_FILE, "w") as _cf:
                _json.dump(_CAL, _cf, indent=2)
            print(f"[cal] stored in {CAL_FILE}")
        except _CalNoContact:
            pass    # already printed the reason; nothing stored, EXIT_CODE set
        except Exception as _ce:
            print(f"[cal] CALIBRATION FAILED to compute/store: {_ce}")
            EXIT_CODE = 3
finally:
    print("[cfg] holding window 5s before close...")
    for _ in range(5 * 60):
        world.step(render=True)
    simulation_app.close()

sys.exit(EXIT_CODE)
