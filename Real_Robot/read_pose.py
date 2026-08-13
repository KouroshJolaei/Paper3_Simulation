#!/usr/bin/env python3
"""
read_pose.py — print the real robot's joints and tool0 pose. Reads only.

WHY THIS EXISTS. Before any of the Paper-3 collection can run on the real
robot we have to know that Berith's simulated station and the physical cell
are the SAME FRAME: the same joint angles must put tool0 in the same place.
Everything downstream — the grid, the stitched canvas, the training pair —
is expressed in world/base coordinates, so a constant offset here would move
every map without anything complaining.

NOTHING MOVES. No trajectory is sent, no gripper command is issued.

WHY IT DOES NOT USE RobotAdapter. /joint_states on this cell publishes in the
order
    shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, shoulder_pan
i.e. shoulder_pan LAST, not first. Anything that takes those positions as a
plain list and pairs them with the canonical name order gets five of six
joints wrong and still returns a perfectly plausible pose. So this script
matches BY NAME and prints both orders, to make the mismatch impossible to
miss rather than something to remember.

Run:
    source /opt/ros/humble/setup.bash
    source ~/ur5e_ws_Gripper/install/setup.bash
    python3 read_pose.py
"""

import json
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

# canonical order — what URDF/FK/cuRobo all expect
CANON = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


class _JointGrab(Node):
    """Latch one /joint_states message."""

    def __init__(self):
        super().__init__("paper3_pose_reader")
        self.msg = None
        self.create_subscription(JointState, "/joint_states", self._cb, JS_QOS)

    def _cb(self, m):
        if self.msg is None and m.name:
            self.msg = m


def main():
    rclpy.init()
    grab = _JointGrab()
    print("waiting for /joint_states ...")
    t0 = time.time()
    while rclpy.ok() and grab.msg is None and time.time() - t0 < 10.0:
        rclpy.spin_once(grab, timeout_sec=0.1)
    if grab.msg is None:
        print("FAILED: no /joint_states in 10 s. Is the driver running?")
        return 1

    published = list(grab.msg.name)
    by_name = dict(zip(grab.msg.name, grab.msg.position))
    missing = [j for j in CANON if j not in by_name]
    if missing:
        print(f"FAILED: joints missing from /joint_states: {missing}")
        return 1

    q = [float(by_name[j]) for j in CANON]

    print("\n--- JOINTS -------------------------------------------------")
    print(f"published order : {published}")
    print(f"canonical order : {CANON}")
    if published != CANON:
        print("NOTE: the two differ. Everything below is matched BY NAME.")
    print()
    for j, v in zip(CANON, q):
        print(f"  {j:<22} {v:+.6f} rad   {math.degrees(v):+8.2f} deg")
    print(f"\n  q_rad = {[round(v, 6) for v in q]}")
    print(f"  q_deg = {[round(math.degrees(v), 3) for v in q]}")

    # ---- tool0 via MoveIt FK ------------------------------------------
    p = Rm = None
    try:
        from get_fk2 import FKClient
        fk = FKClient()
        ps = fk.get_fk(CANON, q, ee_link="tool0", ref_frame="base_link")
        if ps is None:
            print("\nFK returned nothing — is ur_moveit running?")
        else:
            from scipy.spatial.transform import Rotation as R
            pos, ori = ps.pose.position, ps.pose.orientation
            p = np.array([pos.x, pos.y, pos.z], float)
            Rm = R.from_quat([ori.x, ori.y, ori.z, ori.w]).as_matrix()
            print("\n--- tool0 (MoveIt FK, frame "
                  f"{ps.header.frame_id or '?'}) --------------------")
            print(f"  position mm : [{p[0]*1000:+9.3f}, {p[1]*1000:+9.3f}, "
                  f"{p[2]*1000:+9.3f}]")
            print(f"  quat xyzw   : [{ori.x:+.6f}, {ori.y:+.6f}, "
                  f"{ori.z:+.6f}, {ori.w:+.6f}]")
            print("  R (rows)    :")
            for r in Rm:
                print(f"                [{r[0]:+.5f} {r[1]:+.5f} {r[2]:+.5f}]")
            print(f"  tool x axis : [{Rm[0,0]:+.4f} {Rm[1,0]:+.4f} "
                  f"{Rm[2,0]:+.4f}]   (pads close along this)")
            print(f"  tool z axis : [{Rm[0,2]:+.4f} {Rm[1,2]:+.4f} "
                  f"{Rm[2,2]:+.4f}]   (approach direction)")
        fk.destroy_node()
    except Exception as e:
        print(f"\nFK unavailable ({e}). Start ur_moveit and re-run.")

    # ---- independent cross-check: the driver's own TCP pose ----------
    # The UR driver publishes the CONTROLLER's idea of the TCP. If this and
    # MoveIt FK disagree, one of them is wrong, and finding that out now is
    # much cheaper than after a day of collection. A small difference is
    # expected when a TCP offset is configured on the pendant; a large one
    # means the URDF and the robot do not match.
    try:
        from geometry_msgs.msg import PoseStamped

        class _TcpGrab(Node):
            def __init__(self):
                super().__init__("paper3_tcp_reader")
                self.msg = None
                for t in ("/tcp_pose_broadcaster/pose",
                          "/tcp_pose_broadcaster/pose_stamped"):
                    self.create_subscription(PoseStamped, t, self._cb, 10)

            def _cb(self, m):
                if self.msg is None:
                    self.msg = m

        tg = _TcpGrab()
        t0 = time.time()
        while rclpy.ok() and tg.msg is None and time.time() - t0 < 3.0:
            rclpy.spin_once(tg, timeout_sec=0.1)
        if tg.msg is not None:
            tp = tg.msg.pose.position
            print("\n--- TCP (driver broadcaster, frame "
                  f"{tg.msg.header.frame_id or '?'}) ------------")
            print(f"  position mm : [{tp.x*1000:+9.3f}, {tp.y*1000:+9.3f}, "
                  f"{tp.z*1000:+9.3f}]")
            if p is not None:
                d = np.linalg.norm(np.array([tp.x, tp.y, tp.z]) - p) * 1000.0
                print(f"  |TCP - tool0| = {d:.3f} mm"
                      + ("   (= a TCP offset is set on the pendant)"
                         if d > 1.0 else "   (same point)"))
        else:
            print("\n(no TCP broadcaster message — not required)")
        tg.destroy_node()
    except Exception as e:
        print(f"\n(TCP cross-check skipped: {e})")

    if p is not None:
        out = {"q_rad": q, "q_deg": [math.degrees(v) for v in q],
               "joint_names": CANON,
               "published_order": published,
               "tool0_pos_mm": (p * 1000.0).tolist(),
               "tool0_R": Rm.tolist()}
        with open("real_pose_check.json", "w") as f:
            json.dump(out, f, indent=2)
        print("\nsaved real_pose_check.json")
        print("\nNEXT: set Isaac to q_rad above and compare its tool0 "
              "position with tool0_pos_mm.")

    grab.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
