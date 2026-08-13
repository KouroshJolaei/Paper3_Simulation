#!/usr/bin/env python3
"""
move_test.py — STEP A of the real-robot port: move the arm a small, known
distance and check it went where it was asked.

This is the first thing in Paper 3 that moves real hardware, so it is built
to refuse rather than to try:

  * DRY RUN BY DEFAULT. Without --go it plans, checks, prints and exits
    having sent nothing to the robot.
  * IK IS VERIFIED, NOT TRUSTED. The solution is put back through FK and the
    resulting pose compared with what was asked. A solver that returns a
    valid-but-different pose is caught here rather than by watching the arm.
  * ELBOW FLIPS ARE REFUSED. IK can return a configuration that reaches the
    same point through a completely different arm posture. For a 20 mm move
    every joint should barely change, so any joint moving more than
    --max-joint-step-deg aborts.
  * THE PATH IS INTERPOLATED IN JOINT SPACE. Over 20 mm the deviation from a
    straight Cartesian line is negligible, and it cannot produce the sudden
    reconfiguration that a bare IK jump can.

WHY /compute_ik AND NOT THE JACOBIAN. RobotAdapter.move_vertical_slow steps
along a Jacobian, which needs get_jacobian.py and its own KDL setup. MoveIt is
already running for FK, so this uses the same service for IK and adds no new
dependency.

USAGE — dry run first, always:
    python3 move_test.py
    python3 move_test.py --go
"""

import argparse
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# /joint_states has TWO publishers on this cell: joint_state_broadcaster
# (RELIABLE) and the gripper bridge (BEST_EFFORT). A RELIABLE subscriber is
# INCOMPATIBLE with a BEST_EFFORT publisher and silently receives nothing from
# it -- which shows up as the script hanging forever after
#   "offering incompatible QoS. No messages will be received from it."
# A BEST_EFFORT subscriber accepts BOTH, which is exactly what `ros2 topic
# echo` does when it says it is "falling back to BEST_EFFORT as it will
# connect to all publishers".
JS_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                    history=HistoryPolicy.KEEP_LAST, depth=10)

CANON = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

ap = argparse.ArgumentParser()
ap.add_argument("--go", action="store_true",
                help="actually move. Without this nothing is sent.")
ap.add_argument("--dz", type=float, default=20.0,
                help="vertical move in mm, +up (default 20)")
ap.add_argument("--steps", type=int, default=25,
                help="interpolation points (default 25)")
ap.add_argument("--point-time", type=float, default=0.25,
                help="seconds per point; 25 x 0.25 = ~6 s (default 0.25)")
ap.add_argument("--max-joint-step-deg", type=float, default=15.0,
                help="abort if IK moves any joint more than this (default 15)")
ap.add_argument("--pose-tol-mm", type=float, default=1.0,
                help="abort if FK(IK) misses the target by more (default 1)")
args = ap.parse_args()


class _JointGrab(Node):
    def __init__(self):
        super().__init__("paper3_move_test_js")
        self.msg = None
        self.create_subscription(JointState, "/joint_states", self._cb, JS_QOS)

    def _cb(self, m):
        if m.name:
            self.msg = m


def read_joints(node, timeout=5.0):
    """Latest /joint_states, matched BY NAME.

    /joint_states on this cell publishes shoulder_pan LAST. Taking the
    positions as a plain list gives five of six joints wrong and still looks
    plausible, so never index — always match names."""
    node.msg = None
    t0 = time.time()
    while rclpy.ok() and node.msg is None and time.time() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.05)
    if node.msg is None:
        raise RuntimeError("no /joint_states — is the driver running?")
    by = dict(zip(node.msg.name, node.msg.position))
    miss = [j for j in CANON if j not in by]
    if miss:
        raise RuntimeError(f"joints missing from /joint_states: {miss}")
    return np.array([float(by[j]) for j in CANON])


def main():
    rclpy.init()
    js = _JointGrab()

    from scipy.spatial.transform import Rotation as R
    from moveit_msgs.srv import GetPositionFK
    from moveit_msgs.msg import RobotState

    # FK is called here rather than through get_fk2.FKClient because that
    # helper uses spin_until_future_complete with NO timeout: if /compute_fk
    # exists but does not answer (a wedged move_group), the script hangs
    # forever having printed nothing at all. A timeout turns that into a
    # message naming the cause.
    fk_node = Node("paper3_move_test_fk")
    fk_cli = fk_node.create_client(GetPositionFK, "/compute_fk")
    if not fk_cli.wait_for_service(timeout_sec=5.0):
        print("FAILED: /compute_fk not available — is ur_moveit running?")
        return 1

    def fk_pose(q, timeout=5.0):
        req = GetPositionFK.Request()
        req.header.frame_id = "base_link"
        req.fk_link_names = ["tool0"]
        req.robot_state = RobotState(
            joint_state=JointState(name=CANON, position=[float(v) for v in q]))
        fut = fk_cli.call_async(req)
        rclpy.spin_until_future_complete(fk_node, fut, timeout_sec=timeout)
        if not fut.done():
            raise RuntimeError(
                f"/compute_fk did not answer within {timeout:.0f}s. The "
                f"service exists but move_group is not servicing it — "
                f"restart the ur_moveit launch (terminal 2).")
        res = fut.result()
        if res is None or not res.pose_stamped:
            raise RuntimeError("/compute_fk returned no pose")
        if getattr(res.error_code, "val", 1) != 1:
            raise RuntimeError(f"/compute_fk error_code {res.error_code.val}")
        ps = res.pose_stamped[0]
        p = np.array([ps.pose.position.x, ps.pose.position.y,
                      ps.pose.position.z]) * 1000.0
        o = ps.pose.orientation
        return p, np.array([o.x, o.y, o.z, o.w])

    q_now = read_joints(js)
    p_now, quat_now = fk_pose(q_now)

    print("\n=== NOW ====================================================")
    print(f"  q_deg  : {[round(math.degrees(v), 3) for v in q_now]}")
    print(f"  tool0  : [{p_now[0]:+9.3f}, {p_now[1]:+9.3f}, "
          f"{p_now[2]:+9.3f}] mm  (base_link)")

    p_goal = p_now + np.array([0.0, 0.0, float(args.dz)])
    print(f"\n=== GOAL ({args.dz:+.1f} mm in Z) ==========================")
    print(f"  tool0  : [{p_goal[0]:+9.3f}, {p_goal[1]:+9.3f}, "
          f"{p_goal[2]:+9.3f}] mm   (orientation unchanged)")

    # ---- IK via MoveIt, seeded with the current pose --------------------
    from moveit_msgs.srv import GetPositionIK
    from moveit_msgs.msg import RobotState, PositionIKRequest
    from geometry_msgs.msg import PoseStamped

    ik_node = Node("paper3_move_test_ik")
    cli = ik_node.create_client(GetPositionIK, "/compute_ik")
    if not cli.wait_for_service(timeout_sec=5.0):
        print("\nFAILED: /compute_ik not available — is ur_moveit running?")
        return 1

    req = GetPositionIK.Request()
    ikr = PositionIKRequest()
    ikr.group_name = "ur_manipulator"
    ikr.ik_link_name = "tool0"
    ikr.robot_state = RobotState(
        joint_state=JointState(name=CANON, position=list(q_now)))
    ps = PoseStamped()
    ps.header.frame_id = "base_link"
    ps.pose.position.x = float(p_goal[0] / 1000.0)
    ps.pose.position.y = float(p_goal[1] / 1000.0)
    ps.pose.position.z = float(p_goal[2] / 1000.0)
    ps.pose.orientation.x = float(quat_now[0])
    ps.pose.orientation.y = float(quat_now[1])
    ps.pose.orientation.z = float(quat_now[2])
    ps.pose.orientation.w = float(quat_now[3])
    ikr.pose_stamped = ps
    ikr.timeout.sec = 2
    ikr.avoid_collisions = True
    req.ik_request = ikr

    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(ik_node, fut, timeout_sec=8.0)
    if not fut.done():
        print("\nFAILED: /compute_ik did not answer within 8s. Restart the "
              "ur_moveit launch (terminal 2).")
        return 1
    res = fut.result()
    if res is None or res.error_code.val != 1:
        code = "no response" if res is None else res.error_code.val
        print(f"\nFAILED: IK error_code {code}. If this is -31 the goal is "
              f"unreachable; if it is -12 the group name may be wrong "
              f"(tried 'ur_manipulator').")
        return 1

    by = dict(zip(res.solution.joint_state.name,
                  res.solution.joint_state.position))
    q_goal = np.array([float(by[j]) for j in CANON])

    # ---- CHECK 1: does the solution actually reach the target? ---------
    p_chk, _ = fk_pose(q_goal)
    miss = float(np.linalg.norm(p_chk - p_goal))
    # ---- CHECK 2: is it the SAME arm configuration? --------------------
    dq_deg = np.degrees(q_goal - q_now)

    print("\n=== IK SOLUTION ============================================")
    for j, a, b, d in zip(CANON, np.degrees(q_now), np.degrees(q_goal), dq_deg):
        print(f"  {j:<22} {a:+9.3f} -> {b:+9.3f}   d {d:+7.3f} deg")
    print(f"\n  FK(IK) lands at : [{p_chk[0]:+9.3f}, {p_chk[1]:+9.3f}, "
          f"{p_chk[2]:+9.3f}] mm")
    print(f"  miss vs goal    : {miss:.3f} mm  "
          f"(limit {args.pose_tol_mm:.1f})")
    print(f"  largest joint   : {np.abs(dq_deg).max():.3f} deg  "
          f"(limit {args.max_joint_step_deg:.1f})")

    bad = []
    if miss > args.pose_tol_mm:
        bad.append(f"IK solution misses the goal by {miss:.2f} mm")
    if np.abs(dq_deg).max() > args.max_joint_step_deg:
        bad.append(f"IK changed a joint by {np.abs(dq_deg).max():.1f} deg — "
                   f"this is a different arm configuration, not a small move")
    if bad:
        print("\nREFUSING TO MOVE:")
        for b in bad:
            print(f"  - {b}")
        return 1

    if not args.go:
        print("\nDRY RUN — nothing was sent to the robot.")
        print("Checks passed. Re-run with --go to execute.")
        return 0

    # ---- EXECUTE --------------------------------------------------------
    print(f"\n>>> ABOUT TO MOVE THE REAL ROBOT {args.dz:+.1f} mm in Z <<<")
    print(f"    {args.steps} points x {args.point_time:.2f} s = "
          f"~{args.steps * args.point_time:.1f} s")
    try:
        if input("    type GO to confirm: ").strip() != "GO":
            print("    cancelled.")
            return 0
    except EOFError:
        print("    no confirmation possible (non-interactive) — cancelled.")
        return 0

    # ---- EXECUTE --------------------------------------------------------
    # Sent DIRECTLY here rather than through execute_trajectory.py, for two
    # reasons, one of them a safety one.
    #
    # 1. JOINT NAMES. That helper fills msg.joint_names from
    #    define_ur5e_robot.get_ur5e_robot_description(), while the trajectory
    #    was built in CANONICAL order. If those two orders differ — and this
    #    cell already publishes /joint_states with shoulder_pan LAST — the
    #    positions get attached to the wrong joints, which is not a failed
    #    move but a wrong one. Naming them here makes a mismatch impossible.
    # 2. NO TIMEOUTS. It spins on the goal and result futures forever, so a
    #    goal that is never accepted looks identical to a slow move: the
    #    script simply stops printing.
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from control_msgs.action import FollowJointTrajectory
    from rclpy.action import ActionClient
    from get_trajectory import generate_linear_trajectory

    traj = generate_linear_trajectory(q_now.tolist(), q_goal.tolist(),
                                      num_points=int(args.steps))

    exec_node = Node("paper3_move_test_exec")
    ACTION = "/scaled_joint_trajectory_controller/follow_joint_trajectory"
    ac = ActionClient(exec_node, FollowJointTrajectory, ACTION)
    print(f"    connecting to {ACTION} ...")
    if not ac.wait_for_server(timeout_sec=5.0):
        print("FAILED: no action server. Is scaled_joint_trajectory_controller "
              "ACTIVE? (ros2 control list_controllers | grep scaled)")
        return 1

    msg = JointTrajectory()
    msg.joint_names = list(CANON)          # explicit: matches traj point order
    t = 0.0
    for qpt in traj[1:]:                   # skip point 0: already there
        t += float(args.point_time)
        p = JointTrajectoryPoint()
        p.positions = [float(v) for v in qpt]
        p.time_from_start.sec = int(t)
        p.time_from_start.nanosec = int((t % 1.0) * 1e9)
        msg.points.append(p)

    goal = FollowJointTrajectory.Goal()
    goal.trajectory = msg
    print(f"    sending {len(msg.points)} points ...")
    gf = ac.send_goal_async(goal)
    rclpy.spin_until_future_complete(exec_node, gf, timeout_sec=10.0)
    if not gf.done():
        print("FAILED: the controller never answered the goal request.")
        return 1
    gh = gf.result()
    if not gh.accepted:
        print("FAILED: goal REJECTED by the controller. Usual causes: the "
              "pendant's External Control program is not running, or the "
              "joint names do not match the controller's.")
        return 1
    print("    accepted — moving ...")

    rf = gh.get_result_async()
    rclpy.spin_until_future_complete(
        exec_node, rf, timeout_sec=args.steps * args.point_time + 20.0)
    if not rf.done():
        print("WARNING: no result within the expected time. Reading the pose "
              "anyway.")
    else:
        rr = rf.result().result
        if rr.error_code != 0:
            print(f"    controller reported error_code {rr.error_code} "
                  f"{rr.error_string}")
        else:
            print("    trajectory completed.")

    time.sleep(0.5)
    q_end = read_joints(js)
    p_end, _ = fk_pose(q_end)
    err = p_end - p_goal
    moved = p_end - p_now

    print("\n=== RESULT =================================================")
    print(f"  tool0 now  : [{p_end[0]:+9.3f}, {p_end[1]:+9.3f}, "
          f"{p_end[2]:+9.3f}] mm")
    print(f"  moved      : [{moved[0]:+7.3f}, {moved[1]:+7.3f}, "
          f"{moved[2]:+7.3f}] mm   (asked for [0, 0, {args.dz:+.1f}])")
    print(f"  error      : [{err[0]:+7.3f}, {err[1]:+7.3f}, "
          f"{err[2]:+7.3f}] mm   |e| = {np.linalg.norm(err):.3f} mm")
    if abs(moved[0]) > 1.0 or abs(moved[1]) > 1.0:
        print("  NOTE: sideways motion above 1 mm. Joint-space interpolation "
              "bows slightly off the straight line; over 20 mm it should be "
              "well under this.")

    fk_node.destroy_node()
    ik_node.destroy_node()
    js.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
