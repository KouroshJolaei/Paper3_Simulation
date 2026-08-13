#!/usr/bin/env python3
"""
real_motion.py — the MOTION LAYER of the real-robot collector.

One grasp cycle on real hardware, and nothing else: no file writing, no grid
loop, no tactile logging. Those come next, on top of this, once the motion is
trusted.

    home -> above the target -> descend to grasp height -> close -> hold
         -> open -> retreat

WHAT THIS DELIBERATELY DOES NOT DO
    * It does not read the calibration table. TOOL_OFFSET_Z on the real
      gripper has never been measured (the sim's 0.15657 came from Berith's
      simulated contact, and the real Robotiq closes under its own force
      control against a real object), so --grasp-z is given explicitly and
      the operator owns it.
    * It does not know where the object is. Same reason: unmeasured.
    * It does not write pose_history.json or tactile CSVs.
    Each of those is a real measurement that has to be made on hardware
    before it can be trusted, and inventing a plausible number for any of
    them would put a silent error into every stitched map downstream.

SAFETY, in the order it matters
    * DRY RUN BY DEFAULT. --go is required before anything is sent.
    * Every Cartesian target is IK-checked and FK-verified before use, and
      any solution that moves a joint more than --max-joint-step-deg is
      refused as an arm reconfiguration rather than a small move.
    * The descent is stepped, not commanded in one go, and every step is
      re-solved so a bad one is caught at that step's size, not the whole
      descent's.
    * Ctrl-C is caught: the gripper opens and the arm retreats rather than
      stopping wherever it happens to be, on the object.
    * Speed defaults to Paper 2's 5 mm/s.

PENDANT: external_control4 must be LOADED AND RUNNING. A stopped program
still lets the controller accept goals — they simply never reach the arm,
which looks exactly like a successful move that did not happen.
"""

import argparse
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# /joint_states here carries the arm in a NON-canonical order (shoulder_pan
# LAST) and, with the gripper bridge running, a 7th joint. Everything below
# matches BY NAME; indexing it would silently mis-assign five of six joints.
CANON = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

# Two publishers exist on /joint_states with different reliability. A RELIABLE
# subscriber silently receives nothing from the BEST_EFFORT one; BEST_EFFORT
# accepts both.
JS_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST, depth=10)

TRAJ_ACTION = "/scaled_joint_trajectory_controller/follow_joint_trajectory"

ap = argparse.ArgumentParser()
ap.add_argument("--go", action="store_true", help="actually move")
ap.add_argument("--grasp-z", type=float, required=True,
                help="tool0 Z at grasp, mm in base_link. MEASURED, not "
                     "guessed: jog down until the pads straddle the object "
                     "and read it with read_pose.py")
ap.add_argument("--approach-mm", type=float, default=100.0,
                help="stand-off above the grasp (sim uses APPROACH_H=100)")
ap.add_argument("--descent-step-mm", type=float, default=2.0,
                help="descent increment (default 2)")
ap.add_argument("--speed-mm-s", type=float, default=5.0,
                help="travel speed, 1-20. Paper 2 default is 5")
ap.add_argument("--hold-s", type=float, default=3.5,
                help="hold at closed grip (sim uses 3.5)")
ap.add_argument("--max-joint-step-deg", type=float, default=15.0,
                help="FLOOR for the joint gate, deg. The actual limit is "
                     "max(this, 0.4 deg/mm x distance), capped at 90.")

# Measured on this arm: 50 mm vertical needs ~8 deg of wrist_1 (~0.16 deg/mm).
# 0.4 allows margin for other directions without admitting a reconfiguration.
DEG_PER_MM = 0.4
# A genuine elbow/wrist flip moves some joint ~90 deg or more. Nothing below
# that is ever allowed, however long the commanded move.
FLIP_CAP_DEG = 90.0
ap.add_argument("--pose-tol-mm", type=float, default=1.0)
ap.add_argument("--no-gripper", action="store_true",
                help="skip close/open — motion only")
args = ap.parse_args()

SPEED = float(np.clip(args.speed_mm_s, 1.0, 20.0))
if SPEED != args.speed_mm_s:
    print(f"[warn] speed clamped to {SPEED:.1f} mm/s")


class Rig:
    """Read joints, solve FK/IK, send trajectories. Every call bounded."""

    def __init__(self):
        self.js = Node("paper3_real_js")
        self._msg = None
        self.js.create_subscription(JointState, "/joint_states",
                                    self._cb, JS_QOS)
        from moveit_msgs.srv import GetPositionFK, GetPositionIK
        self.fk_node = Node("paper3_real_fk")
        self.ik_node = Node("paper3_real_ik")
        self.fk_cli = self.fk_node.create_client(GetPositionFK, "/compute_fk")
        self.ik_cli = self.ik_node.create_client(GetPositionIK, "/compute_ik")
        for cli, nm in ((self.fk_cli, "/compute_fk"), (self.ik_cli, "/compute_ik")):
            if not cli.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(f"{nm} unavailable — is ur_moveit running?")
        self.exec_node = Node("paper3_real_exec")
        self.ac = ActionClient(self.exec_node, self._traj_type(), TRAJ_ACTION)
        if not self.ac.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(
                "no trajectory action server. Is "
                "scaled_joint_trajectory_controller ACTIVE?")

    @staticmethod
    def _traj_type():
        from control_msgs.action import FollowJointTrajectory
        return FollowJointTrajectory

    def _cb(self, m):
        if m.name:
            self._msg = m

    def joints(self, timeout=5.0):
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
        from moveit_msgs.srv import GetPositionFK
        from moveit_msgs.msg import RobotState
        req = GetPositionFK.Request()
        req.header.frame_id = "base_link"
        req.fk_link_names = ["tool0"]
        req.robot_state = RobotState(
            joint_state=JointState(name=CANON, position=[float(v) for v in q]))
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
        return p, np.array([o.x, o.y, o.z, o.w])

    def ik(self, p_mm, quat, q_seed, dist_mm=None, timeout=8.0):
        """IK, then VERIFY: solutions are checked, never trusted.

        dist_mm is the Cartesian distance being commanded. The joint gate is
        scaled by it, because a fixed limit is wrong at both ends: 15 deg is
        far too loose for a 2 mm descent step and far too tight for the
        130 mm climb to the approach height. Measured on this arm, 50 mm of
        vertical travel needs ~8 deg of wrist_1, i.e. ~0.16 deg/mm; the
        allowance below is 0.4 deg/mm, generous enough for other directions
        and postures while still refusing anything wild.

        The ABSOLUTE cap stays: an elbow or wrist flip reaches the same point
        through a different posture and shows up as a joint moving by ~90 deg
        or more, which no legitimate small Cartesian move produces."""
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
            raise RuntimeError(f"IK failed (error_code {code}) for target "
                               f"[{p_mm[0]:.1f}, {p_mm[1]:.1f}, {p_mm[2]:.1f}]")
        by = dict(zip(res.solution.joint_state.name,
                      res.solution.joint_state.position))
        q_goal = np.array([float(by[j]) for j in CANON])

        p_chk, _ = self.fk(q_goal)
        miss = float(np.linalg.norm(p_chk - p_mm))
        dq = np.degrees(q_goal - q_seed)
        if miss > args.pose_tol_mm:
            raise RuntimeError(
                f"IK solution misses the target by {miss:.2f} mm "
                f"(limit {args.pose_tol_mm}). The solver returned a valid "
                f"pose that is not the one requested.")
        d = float(np.linalg.norm(p_mm - self.fk(q_seed)[0])
                  if dist_mm is None else dist_mm)
        allowed = min(FLIP_CAP_DEG,
                      max(float(args.max_joint_step_deg), DEG_PER_MM * d))
        if np.abs(dq).max() > allowed:
            worst = CANON[int(np.argmax(np.abs(dq)))]
            raise RuntimeError(
                f"IK moved {worst} by {np.abs(dq).max():.1f} deg over a "
                f"{d:.1f} mm move (limit {allowed:.1f} deg = "
                f"{DEG_PER_MM} deg/mm, floor {args.max_joint_step_deg}, "
                f"cap {FLIP_CAP_DEG}). That looks like an arm "
                f"reconfiguration, not a straight move.")
        return q_goal

    def move_to_q(self, q_from, q_to, dist_mm, label=""):
        """Joint-interpolated move, timed from the requested speed."""
        secs = max(0.4, float(dist_mm) / SPEED)
        n = int(np.clip(round(secs / 0.25), 2, 200))
        dt = secs / n
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        msg = JointTrajectory()
        msg.joint_names = list(CANON)      # explicit: order can never drift
        for k in range(1, n + 1):
            a = k / float(n)
            p = JointTrajectoryPoint()
            p.positions = [float(v) for v in ((1 - a) * q_from + a * q_to)]
            t = dt * k
            p.time_from_start.sec = int(t)
            p.time_from_start.nanosec = int((t % 1.0) * 1e9)
            msg.points.append(p)
        goal = self._traj_type().Goal()
        goal.trajectory = msg
        gf = self.ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.exec_node, gf, timeout_sec=10.0)
        if not gf.done():
            raise RuntimeError("controller never answered the goal request")
        gh = gf.result()
        if not gh.accepted:
            raise RuntimeError(
                "goal REJECTED. Usual cause: the pendant's external_control4 "
                "is not running.")
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self.exec_node, rf,
                                         timeout_sec=secs + 20.0)
        if not rf.done():
            print(f"    [{label}] no result within {secs + 20:.0f}s")
        return secs


def gripper(action):
    """open / close via the pendant's digital-out thread."""
    if args.no_gripper:
        print(f"    (gripper {action} skipped: --no-gripper)")
        return
    import gripper_io2
    (gripper_io2.open_gripper if action == "open"
     else gripper_io2.close_gripper)()


def main():
    rclpy.init()
    rig = Rig()

    q0 = rig.joints()
    p0, quat = rig.fk(q0)
    print("\n=== START ==================================================")
    print(f"  tool0   : [{p0[0]:+9.3f}, {p0[1]:+9.3f}, {p0[2]:+9.3f}] mm")
    print(f"  q_deg   : {[round(math.degrees(v), 2) for v in q0]}")

    z_app = float(args.grasp_z) + float(args.approach_mm)
    print(f"\n=== PLAN ===================================================")
    print(f"  speed        : {SPEED:.1f} mm/s")
    print(f"  approach Z   : {z_app:+9.3f} mm   (grasp + "
          f"{args.approach_mm:.0f})")
    print(f"  grasp Z      : {args.grasp_z:+9.3f} mm")
    print(f"  descent      : {args.approach_mm:.0f} mm in "
          f"{args.descent_step_mm:.1f} mm steps  "
          f"({int(math.ceil(args.approach_mm / args.descent_step_mm))} steps)")
    print(f"  X, Y held at : [{p0[0]:+9.3f}, {p0[1]:+9.3f}] mm  "
          f"(pure vertical: park the arm over the object first)")

    if args.grasp_z > p0[2] + 1.0:
        print(f"\nREFUSING: grasp Z {args.grasp_z:.1f} is ABOVE the current "
              f"tool Z {p0[2]:.1f}. This routine only descends; move the arm "
              f"above the object first.")
        return 1

    # Verify the two Cartesian ends BEFORE moving anywhere.
    p_app = np.array([p0[0], p0[1], z_app])
    # Printed BEFORE the IK call: this first leg is usually the largest move
    # of the whole cycle (the arm is parked wherever it was left), so if the
    # joint gate refuses anything it is most often this, and the operator
    # should be able to see the distance that caused it.
    d_app = float(abs(z_app - p0[2]))
    print(f"  first move    : {d_app:.1f} mm "
          f"(from Z {p0[2]:.1f} to {z_app:.1f})")
    q_app = rig.ik(p_app, quat, q0, dist_mm=d_app)
    print(f"\n  IK to approach OK  (largest joint "
          f"{np.abs(np.degrees(q_app - q0)).max():.2f} deg)")
    p_grasp = np.array([p0[0], p0[1], float(args.grasp_z)])
    q_grasp_chk = rig.ik(p_grasp, quat, q_app,
                         dist_mm=float(args.approach_mm))
    print(f"  IK to grasp    OK  (largest joint "
          f"{np.abs(np.degrees(q_grasp_chk - q_app)).max():.2f} deg)")

    if not args.go:
        print("\nDRY RUN — nothing was sent. Re-run with --go.")
        return 0

    print(f"\n>>> REAL ROBOT: descend to Z={args.grasp_z:.1f} mm, "
          f"{'close, hold, open' if not args.no_gripper else 'no gripper'} <<<")
    try:
        if input("    type GO to confirm: ").strip() != "GO":
            print("    cancelled.")
            return 0
    except EOFError:
        print("    non-interactive — cancelled.")
        return 0

    q_cur = q0
    try:
        # 1. up to the approach height
        print(f"\n[1] to approach height ...")
        rig.move_to_q(q_cur, q_app, abs(z_app - p0[2]), "approach")
        q_cur = rig.joints()

        # 2. descend in steps. Each step is re-solved from the CURRENT joints,
        #    so a step that cannot be solved fails at 2 mm rather than 100.
        n_steps = int(math.ceil(args.approach_mm / args.descent_step_mm))
        print(f"[2] descending {n_steps} steps ...")
        gripper("open")
        time.sleep(0.5)
        for k in range(1, n_steps + 1):
            z = max(float(args.grasp_z),
                    z_app - k * float(args.descent_step_mm))
            q_next = rig.ik(np.array([p0[0], p0[1], z]), quat, q_cur,
                            dist_mm=float(args.descent_step_mm))
            rig.move_to_q(q_cur, q_next, args.descent_step_mm, f"down{k}")
            q_cur = rig.joints()
            if k % 10 == 0 or z <= args.grasp_z:
                pc, _ = rig.fk(q_cur)
                print(f"    step {k:>3}/{n_steps}  Z {pc[2]:+9.3f} mm")
            if z <= args.grasp_z:
                break

        # 3. close and hold
        p_at, _ = rig.fk(q_cur)
        print(f"[3] at Z {p_at[2]:+9.3f} mm — closing ...")
        gripper("close")
        print(f"    holding {args.hold_s:.1f} s "
              f"(tactile would be read here)")
        time.sleep(float(args.hold_s))

        # 4. release and retreat
        print("[4] opening and retreating ...")
        gripper("open")
        time.sleep(0.5)
        q_cur = rig.joints()
        rig.move_to_q(q_cur, q_app, abs(z_app - args.grasp_z), "retreat")

    except KeyboardInterrupt:
        # Stopping mid-descent leaves the pads on the object. Open and back
        # off rather than freezing there.
        print("\n!! interrupted — opening gripper and retreating")
        try:
            gripper("open")
            time.sleep(0.5)
            rig.move_to_q(rig.joints(), q_app, args.approach_mm, "abort")
        except Exception as e:
            print(f"   retreat failed ({e}) — move the arm by hand")
        return 130
    except Exception as e:
        print(f"\nFAILED: {e}")
        print("  the arm has been left where it is; open the gripper and "
              "retreat manually if it is on the object")
        return 1

    q_end = rig.joints()
    p_end, _ = rig.fk(q_end)
    print("\n=== DONE ===================================================")
    print(f"  tool0 : [{p_end[0]:+9.3f}, {p_end[1]:+9.3f}, "
          f"{p_end[2]:+9.3f}] mm")
    print(f"  back to approach height, error "
          f"{abs(p_end[2] - z_app):.3f} mm")

    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
