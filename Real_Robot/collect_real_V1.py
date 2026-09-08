#!/usr/bin/env python3
"""
collect_real.py — run a designed grid on the REAL UR5e and write exactly the
files the existing Block-2 pipeline already reads.

THE WHOLE IDEA. Nothing downstream changes. stitching.py, the pair export,
blob_axis, heatmaps and grid_accuracy all read three things:

    pose_history.json          where the pad actually was, per grasp
    <base>_ptNN_<s>_tactile_maps.csv    the 7x4 stream, per grasp per sensor
    gui_config_used.json       the grid that was asked for

collect_from_config.py produces those from Isaac. This produces the same
three from the real cell. That is the entire contract, and it is why real
runs stitch, export and plot with no new code.

WHAT IS MEASURED AND WHAT IS REFUSED
    TOOL_OFFSET_Z on the real gripper is NOT the sim's 0.15657. The sim value
    came from Berith's simulated contact; the real Robotiq closes under its
    own force control against a real object and lands somewhere else. Since
    the stitcher paints every map at pad = ee + R[:,2]*TOOL_OFFSET_Z, an
    unverified value puts a constant, invisible error into every real map.
    So a (diameter, rig="real") calibration entry is REQUIRED, and its
    absence stops the run. --allow-sim-cal overrides it and records
    calibration_source="sim_fallback" in the run's files, so a run collected
    that way stays identifiable instead of looking like any other.

FRAMES. The GUI config is in Isaac WORLD millimetres; the robot works in
base_link. Verified 2026-08-12 at 0.045 mm on the real arm:

    world_mm = base_link_mm + BASE_IN_WORLD_MM      (no rotation)

BASELINE. stitching.hold_average takes its per-taxel baseline from the
LOWEST-sum frames of the SAME csv. Real taxels rest at 10k-36k counts, so a
file containing only closed-grip frames has no baseline to subtract and every
map comes out as raw capacitance. Each grasp therefore records OPEN-grip
frames first, then closes and records the hold into the same file.

PENDANT: external_control4 must be LOADED AND RUNNING, or goals are accepted
and silently never reach the arm.
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ---------------------------------------------------------------- config --
PROJECT = os.path.expanduser("~/Paper3_Simulation")
REAL_ROOT = os.path.join(PROJECT, "Data", "gui_run", "Real")
CAL_PATH = os.path.join(PROJECT, "Data", "pad_offset_calibration.json")

# Isaac world position of base_link, measured 2026-08-12 by putting the sim
# arm at the real arm's joint angles and comparing tool0 (agreed to 0.045 mm).
BASE_IN_WORLD_MM = np.array([20.930, -337.500, 992.750])

CANON = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

# Two publishers with different reliability; BEST_EFFORT accepts both.
JS_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST, depth=10)
TRAJ_ACTION = "/scaled_joint_trajectory_controller/follow_joint_trajectory"

# uint16 wrap handling; see Tactile._unwrap
WRAP_MIN = 60000.0      # at or above this = a negative reading
WRAP_SUSPECT = 45000.0  # above genuine (max seen 36336), below wrap

DEG_PER_MM = 0.4       # joint-gate slope; see Rig.ik
FLIP_CAP_DEG = 90.0    # never allow a joint to swing this far, at any distance

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(PROJECT, "Data",
                                                 "gui_config.json"))
ap.add_argument("--go", action="store_true", help="actually move")
ap.add_argument("--basename", default="gui")
ap.add_argument("--run-dir", default=None)
ap.add_argument("--approach-mm", type=float, default=60.0)
ap.add_argument("--descent-step-mm", type=float, default=2.0)
ap.add_argument("--speed-mm-s", type=float, default=5.0)
ap.add_argument("--hold-s", type=float, default=3.5)
ap.add_argument("--baseline-s", type=float, default=1.5,
                help="open-grip seconds recorded before closing; these frames "
                     "ARE the baseline hold_average subtracts")
ap.add_argument("--settle-s", type=float, default=0.5)
ap.add_argument("--max-joint-step-deg", type=float, default=15.0)
ap.add_argument("--pose-tol-mm", type=float, default=1.0)
ap.add_argument("--allow-sim-cal", action="store_true",
                help="run with the SIM calibration when no real entry exists. "
                     "Recorded as calibration_source=sim_fallback.")
ap.add_argument("--initial-retries", type=int, default=1)
ap.add_argument("--no-abort-on-initial", action="store_true")
ap.add_argument("--home", nargs=6, type=float, default=None,
                help="home joints in DEGREES (canonical order)")
ap.add_argument("--home-here", action="store_true",
                help="use the arm's CURRENT joints as home")
ap.add_argument("--object-here", action="store_true",
                help="place the object so the FIRST grid point lands exactly "
                     "where the pad is now. Nothing has to be measured and "
                     "the arm does not travel to reach pt00 — it is already "
                     "there. This is both the safe way to test the pipeline "
                     "in free air AND, once a real rod exists, the way to "
                     "MEASURE its position: jog until the pads straddle it, "
                     "then run with this flag and the recorded object centre "
                     "is the true one.")
args = ap.parse_args()

SPEED = float(np.clip(args.speed_mm_s, 1.0, 20.0))

# Jogged to on 2026-08-12 and used as the initial home. The GUI's "Set Home"
# button will overwrite this; it is a starting value, not a constant of the
# cell.
HOME_DEG_DEFAULT = [-72.66, -95.51, -119.57, -55.24, 89.79, 256.76]
HOME_DEG = args.home if args.home else HOME_DEG_DEFAULT
HOME_RAD = np.radians(HOME_DEG)


def w2b(p_world_mm):
    """Isaac world mm -> base_link mm."""
    return np.asarray(p_world_mm, float) - BASE_IN_WORLD_MM


def b2w(p_base_mm):
    """base_link mm -> Isaac world mm."""
    return np.asarray(p_base_mm, float) + BASE_IN_WORLD_MM


# ------------------------------------------------------------------ rig ---
class Rig:
    def __init__(self):
        self.js = Node("paper3_collect_js")
        self._msg = None
        self.js.create_subscription(JointState, "/joint_states", self._cb,
                                    JS_QOS)
        from moveit_msgs.srv import GetPositionFK, GetPositionIK
        self.fk_node = Node("paper3_collect_fk")
        self.ik_node = Node("paper3_collect_ik")
        self.fk_cli = self.fk_node.create_client(GetPositionFK, "/compute_fk")
        self.ik_cli = self.ik_node.create_client(GetPositionIK, "/compute_ik")
        for cli, nm in ((self.fk_cli, "/compute_fk"),
                        (self.ik_cli, "/compute_ik")):
            if not cli.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(f"{nm} unavailable — is ur_moveit running?")
        from control_msgs.action import FollowJointTrajectory
        self.TrajAction = FollowJointTrajectory
        self.exec_node = Node("paper3_collect_exec")
        # The controller BUFFERS an accepted trajectory. Ctrl-C stops this
        # script but not the arm: it keeps executing, and resumes after an
        # e-stop is cleared. So the live goal handle is kept and explicitly
        # cancelled on interrupt.
        self._goal = None
        self.ac = ActionClient(self.exec_node, FollowJointTrajectory,
                               TRAJ_ACTION)
        if not self.ac.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("no trajectory action server. Is "
                               "scaled_joint_trajectory_controller ACTIVE?")

    def _cb(self, m):
        if m.name:
            self._msg = m

    def joints(self, timeout=5.0):
        """Matched BY NAME: this cell publishes shoulder_pan LAST, and with
        the gripper bridge running there is a 7th joint."""
        self._msg = None
        t0 = time.time()
        while rclpy.ok() and self._msg is None and time.time() - t0 < timeout:
            rclpy.spin_once(self.js, timeout_sec=0.05)
        if self._msg is None:
            raise RuntimeError("no /joint_states — is the driver running?")
        by = dict(zip(self._msg.name, self._msg.position))
        miss = [j for j in CANON if j not in by]
        if miss:
            raise RuntimeError(f"joints missing: {miss}")
        return np.array([float(by[j]) for j in CANON])

    def fk(self, q, timeout=5.0):
        """-> (pos_mm in base_link, quat xyzw, R 3x3)."""
        from moveit_msgs.srv import GetPositionFK
        from moveit_msgs.msg import RobotState
        from scipy.spatial.transform import Rotation as R
        req = GetPositionFK.Request()
        req.header.frame_id = "base_link"
        req.fk_link_names = ["tool0"]
        req.robot_state = RobotState(
            joint_state=JointState(name=CANON,
                                   position=[float(v) for v in q]))
        fut = self.fk_cli.call_async(req)
        rclpy.spin_until_future_complete(self.fk_node, fut, timeout_sec=timeout)
        if not fut.done():
            raise RuntimeError("/compute_fk did not answer — restart ur_moveit")
        r = fut.result()
        if r is None or not r.pose_stamped:
            raise RuntimeError("/compute_fk returned nothing")
        ps = r.pose_stamped[0]
        p = np.array([ps.pose.position.x, ps.pose.position.y,
                      ps.pose.position.z]) * 1000.0
        o = ps.pose.orientation
        quat = np.array([o.x, o.y, o.z, o.w])
        return p, quat, R.from_quat(quat).as_matrix()

    def ik(self, p_mm, quat, q_seed, dist_mm=None, timeout=8.0):
        """IK, then VERIFY. Two independent checks:

        1. FK the solution and compare with what was asked. A solver can
           return a perfectly valid pose that is not the requested one.
        2. Gate the joint change against the DISTANCE commanded. A fixed
           limit is wrong at both ends — 15 deg is far too loose for a 2 mm
           descent step and too tight for a 130 mm approach (measured: this
           arm needs ~0.16 deg/mm) — so the allowance scales, with an
           absolute cap that still catches an elbow flip.
        """
        from moveit_msgs.srv import GetPositionIK
        from moveit_msgs.msg import RobotState, PositionIKRequest
        from geometry_msgs.msg import PoseStamped
        req = GetPositionIK.Request()
        r = PositionIKRequest()
        r.group_name = "ur_manipulator"
        r.ik_link_name = "tool0"
        r.robot_state = RobotState(
            joint_state=JointState(name=CANON,
                                   position=[float(v) for v in q_seed]))
        ps = PoseStamped()
        ps.header.frame_id = "base_link"
        ps.pose.position.x = float(p_mm[0] / 1000.0)
        ps.pose.position.y = float(p_mm[1] / 1000.0)
        ps.pose.position.z = float(p_mm[2] / 1000.0)
        ps.pose.orientation.x, ps.pose.orientation.y = float(quat[0]), float(quat[1])
        ps.pose.orientation.z, ps.pose.orientation.w = float(quat[2]), float(quat[3])
        r.pose_stamped = ps
        r.timeout.sec = 2
        r.avoid_collisions = True
        req.ik_request = r
        fut = self.ik_cli.call_async(req)
        rclpy.spin_until_future_complete(self.ik_node, fut, timeout_sec=timeout)
        if not fut.done():
            raise RuntimeError("/compute_ik did not answer — restart ur_moveit")
        res = fut.result()
        if res is None or res.error_code.val != 1:
            code = "no response" if res is None else res.error_code.val
            raise RuntimeError(
                f"IK failed (error_code {code}) for base_link target "
                f"[{p_mm[0]:.1f}, {p_mm[1]:.1f}, {p_mm[2]:.1f}] mm")
        by = dict(zip(res.solution.joint_state.name,
                      res.solution.joint_state.position))
        q_goal = np.array([float(by[j]) for j in CANON])

        p_chk = self.fk(q_goal)[0]
        miss = float(np.linalg.norm(p_chk - p_mm))
        if miss > args.pose_tol_mm:
            raise RuntimeError(f"IK solution misses the target by {miss:.2f} mm")
        d = float(dist_mm if dist_mm is not None
                  else np.linalg.norm(p_mm - self.fk(q_seed)[0]))
        allowed = min(FLIP_CAP_DEG,
                      max(float(args.max_joint_step_deg), DEG_PER_MM * d))
        dq = np.degrees(q_goal - q_seed)
        if np.abs(dq).max() > allowed:
            worst = CANON[int(np.argmax(np.abs(dq)))]
            raise RuntimeError(
                f"IK moved {worst} by {np.abs(dq).max():.1f} deg over a "
                f"{d:.1f} mm move (limit {allowed:.1f}). That is an arm "
                f"reconfiguration, not a move to the point asked for.")
        return q_goal

    def move_to_q(self, q_from, q_to, dist_mm, label=""):
        secs = max(0.4, float(dist_mm) / SPEED)
        n = int(np.clip(round(secs / 0.25), 2, 200))
        dt = secs / n
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        msg = JointTrajectory()
        msg.joint_names = list(CANON)   # explicit: the order can never drift
        for k in range(1, n + 1):
            a = k / float(n)
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in ((1 - a) * q_from + a * q_to)]
            t = dt * k
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t % 1.0) * 1e9)
            msg.points.append(pt)
        goal = self.TrajAction.Goal()
        goal.trajectory = msg
        gf = self.ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.exec_node, gf, timeout_sec=10.0)
        if not gf.done():
            raise RuntimeError("controller never answered the goal request")
        gh = gf.result()
        if not gh.accepted:
            raise RuntimeError("goal REJECTED — is external_control4 running "
                               "on the pendant?")
        self._goal = gh
        try:
            rf = gh.get_result_async()
            rclpy.spin_until_future_complete(self.exec_node, rf,
                                             timeout_sec=secs + 20.0)
        finally:
            self._goal = None
        return secs

    def cancel(self):
        """Stop the arm NOW. Without this, Ctrl-C leaves the controller
        running the trajectory it already accepted."""
        gh = self._goal
        if gh is None:
            return
        try:
            print("    [cancel] stopping the active trajectory ...")
            cf = gh.cancel_goal_async()
            rclpy.spin_until_future_complete(self.exec_node, cf,
                                             timeout_sec=3.0)
            print("    [cancel] done")
        except Exception as e:
            print(f"    [cancel] failed ({e}) — USE THE E-STOP")
        finally:
            self._goal = None

    def move_cart(self, p_target_mm, quat, label=""):
        q_cur = self.joints()
        p_cur = self.fk(q_cur)[0]
        d = float(np.linalg.norm(np.asarray(p_target_mm) - p_cur))
        if d < 0.05:
            return q_cur
        q_to = self.ik(p_target_mm, quat, q_cur, dist_mm=d)
        self.move_to_q(q_cur, q_to, d, label)
        return self.joints()


# -------------------------------------------------------------- tactile ---
class Tactile:
    """Frames from the Qt server on 127.0.0.1:12345 (NOT the USB port —
    the Qt app owns that and serves over TCP)."""

    def __init__(self):
        from tactile_DataReadSave3 import TactileSensorClient
        self.cli = TactileSensorClient()
        self.cli.connect()
        # The protocol is a tiny request followed by a small reply. With
        # Nagle enabled that pattern can stall on delayed ACKs for hundreds
        # of ms per exchange, which is the right thing to rule out first when
        # a LOCAL socket is delivering only ~1.4 Hz.
        try:
            import socket as _sock
            self.cli.client.setsockopt(_sock.IPPROTO_TCP, _sock.TCP_NODELAY, 1)
            print("    [tactile] TCP_NODELAY set")
        except Exception as e:
            print(f"    [tactile] could not set TCP_NODELAY ({e})")
        # Measure the achievable rate ONCE, so the run states the limit it is
        # working under instead of discovering it per grasp.
        _t0 = time.time()
        for _ in range(10):
            self.cli.read_data()
        _hz = 10.0 / max(time.time() - _t0, 1e-9)
        print(f"    [tactile] measured {_hz:.1f} Hz "
              f"({1000.0 / _hz:.0f} ms per frame)")
        self.hz = _hz
        d, _ = self.cli.read_data()
        for k in ("S_0", "S_1"):
            if len(d[k]) != 28:
                raise RuntimeError(f"{k} has {len(d[k])} values, expected 28")

    @staticmethod
    def _unwrap(vals):
        """Undo the UNSIGNED-16-BIT WRAP.

        The server packs taxels as 'H' (uint16). Several taxels rest very
        near zero -- 28, 139, 170 counts were measured on this pair -- so a
        genuinely NEGATIVE reading comes back as 65536+x. Observed directly:
        taxel 26 read 65505, i.e. -31.

        Left alone this is not a cosmetic problem. One taxel at 65505 instead
        of -31 is 2000x every other value in the map, so it dominates the
        frame sum that hold_average uses to pick baseline and hold frames,
        and it swamps every centroid, mean and SSIM downstream. It has to be
        undone AT THE POINT OF READING, before anything else sees it.

        THE THRESHOLD IS NOT HALF-SCALE. Half-scale (32767) is wrong here and
        was caught by testing against a real frame: taxel 10 rests at 36036,
        a perfectly genuine reading, and half-scale would have turned it into
        -29500. Measured on this pair, resting values reach 36336 while every
        observed wrap sits just below 65536 (65395..65512, i.e. -141..-24),
        because the negatives are small sensor noise. WRAP_MIN = 60000 leaves
        ~24k of margin below and ~5k above, and anything landing between the
        two is reported rather than silently converted.
        """
        out = []
        for v in vals:
            v = float(v)
            if v >= WRAP_MIN:
                out.append(v - 65536.0)
            elif v > WRAP_SUSPECT:
                print(f"    [tactile] value {v:.0f} is between the genuine "
                      f"range (<= {WRAP_SUSPECT:.0f}) and the wrap window "
                      f"(>= {WRAP_MIN:.0f}); left AS IS — check the sensor")
                out.append(v)
            else:
                out.append(v)
        return out

    def record(self, seconds, out, period=0.0, t_origin=None):
        """Append (t, 28 floats) rows for each sensor for `seconds`.

        period defaults to 0: read_data() is already a blocking round-trip to
        the Qt server, so it self-paces. The original 10 ms sleep on top of
        that gave 11 frames in 3.2 s (~3 Hz) where the sensor runs at 60 Hz --
        and hold_average splits frames into the lowest 5% and highest 90% by
        taxel sum, which is meaningless with 11 samples. More frames is not a
        refinement here; it is what makes the baseline subtraction work.
        """
        # t_origin lets the caller keep ONE clock across the baseline and
        # hold calls. Without it each call restarted at 0 and the csv time
        # column ran 0.43, 1.10, 1.78, 0.66, ... — non-monotonic, and any
        # later tool that sorts or differences by time would be wrong.
        t0 = time.time()
        origin = t0 if t_origin is None else float(t_origin)
        n = n_wrap = 0
        while time.time() - t0 < seconds:
            try:
                d, _ = self.cli.read_data()
            except Exception as e:
                print(f"    [tactile] read failed: {e}")
                break
            t = time.time() - origin
            s1 = self._unwrap(d["S_0"])
            s2 = self._unwrap(d["S_1"])
            n_wrap += sum(1 for v in s1 + s2 if v < 0)
            out["s1"].append([t] + s1)
            out["s2"].append([t] + s2)
            n += 1
            if period > 0:
                time.sleep(period)
        if n and n / max(seconds, 1e-9) < 20.0:
            print(f"    [tactile] only {n / seconds:.0f} Hz — the baseline "
                  f"split needs many frames; check the Qt server load")
        if n_wrap:
            print(f"    [tactile] unwrapped {n_wrap} negative readings "
                  f"(uint16 wrap)")
        return n

    def close(self):
        try:
            self.cli.close()
        except Exception:
            pass


def write_tactile_csv(path, rows):
    """Header pred_0..pred_27 — the column names stitching.hold_average and
    heatmaps both look for."""
    with open(path, "w") as f:
        f.write("time," + ",".join(f"pred_{i}" for i in range(28)) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.4f}" for v in r) + "\n")


# ------------------------------------------------------------ calibration -
def load_calibration(diameter_mm):
    """(tool_offset_z_m, source). Refuses without a real entry unless
    --allow-sim-cal, because TOOL_OFFSET_Z is what the stitcher paints with."""
    key = f"{float(diameter_mm):.1f}"
    cal = {}
    if os.path.exists(CAL_PATH):
        try:
            with open(CAL_PATH) as f:
                cal = json.load(f)
        except Exception as e:
            print(f"[cal] could not read {CAL_PATH}: {e}")

    # Preferred: an entry recorded on the REAL rig. Two layouts are accepted
    # so the file can gain the rig key without breaking the sim's reader:
    #   {"26.0": {..., "rig": "real"}}     or   {"real": {"26.0": {...}}}
    for k, v in (("real", cal.get("real", {})), ("flat", cal)):
        if isinstance(v, dict) and key in v and isinstance(v[key], dict):
            e = v[key]
            if k == "real" or str(e.get("rig", "")).lower() == "real":
                return float(e["TOOL_OFFSET_Z"]), "real"

    if not args.allow_sim_cal:
        have = sorted(cal.get("real", {}).keys()) or [
            kk for kk, vv in cal.items()
            if isinstance(vv, dict) and str(vv.get("rig", "")).lower() == "real"]
        raise SystemExit(
            f"\nREFUSING TO RUN: no REAL calibration for \u00d8{key} mm.\n"
            f"  file: {CAL_PATH}\n"
            f"  real entries present: {have or 'none'}\n\n"
            f"TOOL_OFFSET_Z is the flange->pad distance the stitcher paints\n"
            f"every map with. The sim's value came from simulated contact and\n"
            f"is not this gripper closing on this object, so borrowing it puts\n"
            f"a constant error into every real map with nothing in the files\n"
            f"to show it.\n\n"
            f"Calibrate this object on the real rig, or pass --allow-sim-cal\n"
            f"to proceed knowingly (recorded as calibration_source=sim_fallback).")

    if key in cal and isinstance(cal[key], dict) and "TOOL_OFFSET_Z" in cal[key]:
        print(f"[cal] !! using SIM calibration for \u00d8{key} — NOT verified "
              f"on hardware")
        return float(cal[key]["TOOL_OFFSET_Z"]), "sim_fallback"
    print(f"[cal] !! no entry at all for \u00d8{key}; using the \u00d826 sim "
          f"value 0.15657 — NOT verified on hardware")
    return 0.15657, "sim_fallback_default"


# ------------------------------------------------------------------ main --
def main():
    with open(args.config) as f:
        cfg = json.load(f)
    obj = cfg["object"]
    diam = float(obj["diameter_mm"])
    obj_c_world = np.array(obj["center_world_mm"], float)
    pad_rot = float((cfg.get("pad") or {}).get("rotation_deg", 0.0))
    pts = cfg.get("points", [])
    if not pts:
        raise SystemExit("config has no points")

    # A rolled pad needs a wrist roll AND the pivot correction for the
    # ~156 mm flange-to-pad lever arm. Neither has been verified on hardware,
    # and getting it silently wrong would rotate every footprint the stitcher
    # paints. Upright only, for now, and said out loud.
    if abs(pad_rot) > 1e-6:
        raise SystemExit(
            f"\nREFUSING: this config has pad rotation_deg={pad_rot}. Wrist "
            f"roll is not yet verified on the real rig (it needs the pivot "
            f"correction for the flange-to-pad lever arm). Use an upright "
            f"config, or verify roll separately first.")

    tool_z, cal_src = load_calibration(diam)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or os.path.join(
        REAL_ROOT, f"run_{stamp}_real_obj{int(diam)}_pad{int(round(pad_rot))}")
    base = args.basename

    print("\n=== REAL COLLECTION ========================================")
    print(f"  config      : {args.config}")
    print(f"  object      : \u00d8{diam:.1f} x {obj.get('length_mm', '?')} mm "
          f"at world {obj_c_world.tolist()}")
    print(f"  object base : {np.round(w2b(obj_c_world), 2).tolist()} mm "
          f"(base_link)")
    print(f"  points      : {len(pts)}")
    print(f"  TOOL_OFFSET_Z: {tool_z:.5f} m   [{cal_src}]")
    print(f"  speed       : {SPEED:.1f} mm/s   approach "
          f"{args.approach_mm:.0f} mm   step {args.descent_step_mm:.1f} mm")
    print(f"  run dir     : {run_dir}")
    print(f"  home (deg)  : {[round(v, 2) for v in HOME_DEG]}")

    rclpy.init()
    rig = Rig()

    q_home = np.array(HOME_RAD, float)
    p_home, quat_home, R_home = rig.fk(q_home)
    # --home-here / --object-here: everything relative to the arm's CURRENT
    # pose, so no long swing across the cell and nothing to measure first.
    if args.home_here:
        q_home = rig.joints()
        p_home, quat_home, R_home = rig.fk(q_home)
        print(f"\n  home taken from the CURRENT pose: "
              f"{[round(math.degrees(v), 2) for v in q_home]} deg")
    if args.object_here:
        pad_now_base = p_home + R_home[:, 2] * (tool_z * 1000.0)
        p0c = pts[0]
        obj_c_world = b2w(pad_now_base) - np.array(
            [0.0, float(p0c["pad_offset_y_mm"]), float(p0c["pad_offset_z_mm"])])
        print(f"  object placed from the CURRENT pad position:")
        print(f"    pad now      : {np.round(b2w(pad_now_base), 1).tolist()} "
              f"mm (world)")
        print(f"    object centre: {np.round(obj_c_world, 1).tolist()} mm "
              f"(world)  <- pt00 will land exactly here")

    print(f"\n  home tool0  : {np.round(p_home, 2).tolist()} mm (base_link)")
    print(f"              = {np.round(b2w(p_home), 2).tolist()} mm (world)")

    # ---- plan every point BEFORE moving anything ------------------------
    # The pad face must sit at object_centre + (0, dy, dz); the flange is
    # therefore tool_z ABOVE it along the pad's own -z. Upright, that is a
    # pure +Z offset.
    # The flange offset uses the MEASURED tool orientation, not an assumed
    # straight-down one. R_home[:, 2] is the tool's own z axis, and on this
    # arm it reads [+0.005, -0.005, -1.000] -- tilted ~0.4 deg. Over the
    # 156 mm flange-to-pad lever that is 1.1 mm of pad displacement, and it
    # would be invisible: the stitcher recovers the pad as
    # ee + R[:,2]*TOOL_OFFSET_Z, so commanding as if the tool were vertical
    # puts every real map 1.1 mm from where the grid asked for it.
    _off_base = -R_home[:, 2] * (tool_z * 1000.0)
    print(f"\n  flange offset from pad: {np.round(_off_base, 3).tolist()} mm "
          f"(from the measured tool axis, not assumed vertical)")

    targets = []
    for p in pts:
        pad_world = obj_c_world + np.array([0.0,
                                            float(p["pad_offset_y_mm"]),
                                            float(p["pad_offset_z_mm"])])
        pad_base = w2b(pad_world)
        ee_base = pad_base + _off_base
        targets.append({"index": int(p["index"]),
                        "tag": f"pt{int(p['index']):02d}",
                        "pad_world": pad_world, "ee_base": ee_base})

    print(f"\n  first point : pad world "
          f"{np.round(targets[0]['pad_world'], 1).tolist()} -> flange base "
          f"{np.round(targets[0]['ee_base'], 1).tolist()} mm")
    zs = [t["ee_base"][2] for t in targets]
    print(f"  flange Z    : {min(zs):.1f} .. {max(zs):.1f} mm; approach at "
          f"{max(zs) + args.approach_mm:.1f} mm")

    print("\n  checking IK for every point ...")
    bad = []
    for t in targets:
        try:
            rig.ik(t["ee_base"], quat_home, q_home,
                   dist_mm=float(np.linalg.norm(t["ee_base"] - p_home)))
        except Exception as e:
            bad.append((t["tag"], str(e)))
    if bad:
        print(f"  {len(bad)} of {len(targets)} points FAILED:")
        for tag, e in bad[:8]:
            print(f"    {tag}: {e}")
        if any(t == targets[0]["tag"] for t, _ in bad):
            print("\nREFUSING: the INITIAL point is unreachable. Without it "
                  "there is no training pair, so the run is pointless.")
            return 1
        print("  (unreachable points will be skipped)")
    else:
        print(f"  all {len(targets)} points reachable")

    if not args.go:
        print("\nDRY RUN — nothing was sent. Re-run with --go.")
        return 0

    d_home = float(np.linalg.norm(
        np.array([targets[0]["ee_base"][0], targets[0]["ee_base"][1],
                  max(zs) + args.approach_mm]) - p_home))
    if d_home > 150.0:
        print(f"\n  !! the first move is {d_home:.0f} mm — home is not above "
              f"the object.")
        print(f"     home  base_link {np.round(p_home, 1).tolist()}")
        print(f"     first base_link "
              f"{np.round([targets[0]['ee_base'][0], targets[0]['ee_base'][1], max(zs) + args.approach_mm], 1).tolist()}")
        print(f"     The arm will swing across the cell to get there. Park it "
              f"above the object and set home there instead, or make sure the "
              f"path is clear.")

    print(f"\n>>> REAL ROBOT: {len(targets)} grasps, "
          f"~{len(targets) * (args.baseline_s + args.hold_s + 12):.0f} s <<<")
    try:
        if input("    type GO to confirm: ").strip() != "GO":
            print("    cancelled.")
            return 0
    except EOFError:
        return 0

    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(run_dir, "gui_config_used.json"))
    # The stitcher looks here for TOOL_OFFSET_Z (stitching._tool_offset_z).
    with open(os.path.join(run_dir, "reachability_report.json"), "w") as f:
        json.dump({"generated": stamp, "config": args.config,
                   "object_center_mm": obj_c_world.tolist(),
                   "diameter_mm": diam,
                   "TOOL_OFFSET_Z": tool_z,
                   "calibration_source": cal_src,
                   "rig": "real",
                   "base_in_world_mm": BASE_IN_WORLD_MM.tolist(),
                   "object_center_source": ("current_pose"
                                           if args.object_here
                                           else "config"),
                   "n_points": len(targets)}, f, indent=2)

    import gripper_io2
    tac = Tactile()
    pose_hist, ledger, n_ok = [], [], 0
    bad_tags = {t for t, _ in bad}
    z_app = max(zs) + float(args.approach_mm)
    aborted, abort_reason, retries_used, retry_needed = False, None, 0, False

    def grasp_one(t, attempt=1):
        """approach -> descend -> record open -> close -> record hold -> open.
        Returns (ok, stage)."""
        tag = t["tag"]
        p_app = np.array([t["ee_base"][0], t["ee_base"][1], z_app])
        try:
            gripper_io2.open_gripper()
            time.sleep(0.4)
            rig.move_cart(p_app, quat_home, f"{tag}:approach")
        except Exception as e:
            print(f"    [{tag}] approach failed: {e}")
            return False, "approach"

        try:
            q_cur = rig.joints()
            z_now = rig.fk(q_cur)[0][2]
            z_goal = float(t["ee_base"][2])
            n_steps = int(math.ceil(abs(z_now - z_goal) /
                                    float(args.descent_step_mm)))
            for k in range(1, n_steps + 1):
                z = max(z_goal, z_now - k * float(args.descent_step_mm))
                q_next = rig.ik(np.array([t["ee_base"][0], t["ee_base"][1], z]),
                                quat_home, q_cur,
                                dist_mm=float(args.descent_step_mm))
                rig.move_to_q(q_cur, q_next, args.descent_step_mm, f"{tag}:d{k}")
                q_cur = rig.joints()
                if z <= z_goal:
                    break
        except Exception as e:
            print(f"    [{tag}] descent failed: {e}")
            return False, "descent"

        rows = {"s1": [], "s2": []}
        try:
            time.sleep(float(args.settle_s))
            # OPEN-grip frames first: hold_average takes its per-taxel
            # baseline from the lowest-sum frames of this same file, and real
            # taxels rest at 10k-36k counts.
            t_grasp = time.time()
            n_base = tac.record(float(args.baseline_s), rows,
                                t_origin=t_grasp)
            gripper_io2.close_gripper()
            n_hold = tac.record(float(args.hold_s), rows, t_origin=t_grasp)
            gripper_io2.open_gripper()
            time.sleep(0.4)
        except Exception as e:
            print(f"    [{tag}] tactile/grip failed: {e}")
            try:
                gripper_io2.open_gripper()
            except Exception:
                pass
            return False, "grasp"

        q_g = rig.joints()
        p_ee_base, _q, R = rig.fk(q_g)
        ee_world_m = (b2w(p_ee_base) / 1000.0).tolist()
        pad_world_m = (b2w(p_ee_base + R[:, 2] * tool_z * 1000.0) / 1000.0)
        for s in ("s1", "s2"):
            write_tactile_csv(
                os.path.join(run_dir, f"{base}_{tag}_{s}_tactile_maps.csv"),
                rows[s])
        pose_hist.append({
            "tag": tag,
            "ee_world_m": ee_world_m,
            "joints_rad": q_g.tolist(),
            "pad_actual_pos_m": pad_world_m.tolist(),
            "pad_actual_R": R.tolist(),
            "pad_gui_target_m": (t["pad_world"] / 1000.0).tolist(),
            "pad_desired_pos_m": (t["pad_world"] / 1000.0).tolist(),
            "attempt": attempt,
        })
        err = float(np.linalg.norm(pad_world_m * 1000.0 - t["pad_world"]))
        print(f"    [{tag}] ok — {n_base} baseline + {n_hold} hold frames, "
              f"pad {err:.2f} mm from target")
        return True, "complete"

    try:
        print("\n[home] going to home ...")
        q_now = rig.joints()
        rig.move_to_q(q_now, q_home,
                      float(np.linalg.norm(rig.fk(q_now)[0] - p_home)), "home")

        for i, t in enumerate(targets):
            tag = t["tag"]
            if tag in bad_tags:
                print(f"\n[{i+1}/{len(targets)}] {tag} SKIPPED (unreachable)")
                ledger.append({"index": t["index"], "tag": tag,
                               "executed": False, "exec_stage": None,
                               "outcome": "predicted_bad_skipped"})
                continue
            print(f"\n[{i+1}/{len(targets)}] {tag} "
                  f"pad world {np.round(t['pad_world'], 1).tolist()}")
            ok, stage = grasp_one(t)
            attempts = [{"attempt": 1, "stage": stage, "ok": ok}]

            # The initial point is not an ordinary point: without it there is
            # no training pair, so it gets a retry and then stops the run.
            if (not ok) and i == 0 and args.initial_retries > 0:
                for a in range(1, args.initial_retries + 1):
                    print(f"    !! INITIAL point failed at '{stage}'. "
                          f"Retry {a}/{args.initial_retries} ...")
                    ok, stage = grasp_one(t, attempt=a + 1)
                    attempts.append({"attempt": a + 1, "stage": stage,
                                     "ok": ok})
                    retries_used = a
                    if ok:
                        retry_needed = True
                        break

            row = {"index": t["index"], "tag": tag, "executed": ok,
                   "exec_stage": stage,
                   "outcome": "ok" if ok else "failed"}
            if len(attempts) > 1:
                row["attempts"] = attempts
            ledger.append(row)

            if ok:
                n_ok += 1
            elif i == 0 and not args.no_abort_on_initial:
                aborted = True
                abort_reason = (
                    f"{tag} is the designed initial point and it failed at "
                    f"'{stage}' after {retries_used} retry(ies); without it "
                    f"the remaining {len(targets)-1} points cannot form a "
                    f"training pair")
                print(f"\n!!!!! RUN ABORTED — {abort_reason}")
                break

    except KeyboardInterrupt:
        aborted, abort_reason = True, "interrupted by operator"
        print("\n!! interrupted — cancelling the trajectory FIRST")
        rig.cancel()
        print("!! opening gripper and returning home")
    except Exception as e:
        aborted, abort_reason = True, f"unhandled: {e}"
        print(f"\nFAILED: {e}")
        rig.cancel()
    finally:
        try:
            gripper_io2.open_gripper()
            time.sleep(0.4)
            q_now = rig.joints()
            p_now = rig.fk(q_now)[0]
            rig.move_cart(np.array([p_now[0], p_now[1], z_app]), quat_home,
                          "retreat")
            q_now = rig.joints()
            rig.move_to_q(q_now, q_home,
                          float(np.linalg.norm(rig.fk(q_now)[0] - p_home)),
                          "home")
            print("[home] returned home")
        except Exception as e:
            print(f"[home] could not return ({e}) — move the arm by hand")
        tac.close()

        with open(os.path.join(run_dir, "pose_history.json"), "w") as f:
            json.dump({"config": cfg, "points": pose_hist,
                       "rig": "real",
                       "TOOL_OFFSET_Z": tool_z,
                       "calibration_source": cal_src}, f, indent=2)
        with open(os.path.join(run_dir, "execution_ledger.json"), "w") as f:
            json.dump({"generated": stamp, "config": args.config, "rig": "real",
                       "calibration_source": cal_src,
                       "initial_point": targets[0]["tag"],
                       "initial_retries_used": retries_used,
                       "initial_retry_required": retry_needed,
                       "aborted": aborted, "abort_reason": abort_reason,
                       "counts": {"ok": n_ok,
                                  "failed": len(ledger) - n_ok},
                       "points": ledger}, f, indent=2)
        print(f"\n=== {n_ok}/{len(targets)} grasps OK ===")
        print(f"    {run_dir}")
        if cal_src != "real":
            print(f"    !! calibration_source = {cal_src}: TOOL_OFFSET_Z was "
                  f"NOT measured on this rig, so every pad position here "
                  f"carries a constant unverified offset.")
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
    return 1 if aborted else 0


if __name__ == "__main__":
    sys.exit(main())
