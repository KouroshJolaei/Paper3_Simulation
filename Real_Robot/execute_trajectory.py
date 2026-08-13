# execute_trajectory.py
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from define_ur5e_robot import get_ur5e_robot_description
class UR5eTrajectoryExecutor(Node):
    def __init__(self, robot_description):
        super().__init__('ur5e_trajectory_executor')

        self.joint_names = robot_description["joint_names"]
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            #"/joint_trajectory_controller/follow_joint_trajectory"
            "/scaled_joint_trajectory_controller/follow_joint_trajectory"
        )

        self.get_logger().info("Waiting for action server...")
        self.trajectory_client.wait_for_server()
        self.get_logger().info("Connected to joint_trajectory_controller")

    def execute_trajectory(self, trajectory, point_time=0.3):
        """Send the full trajectory to the FollowJointTrajectory action server."""
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        time_from_start = 0.0
        for idx, joint_point in enumerate(trajectory):
            point = JointTrajectoryPoint()
            point.positions = joint_point
            time_from_start += point_time
            point.time_from_start.sec = int(time_from_start)
            point.time_from_start.nanosec = int((time_from_start % 1) * 1e9)
            msg.points.append(point)

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = msg

        self.get_logger().info("Sending trajectory to robot...")
        future = self.trajectory_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal was rejected.")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.error_code == 0:
            self.get_logger().info("✅ Trajectory execution completed successfully.")
        else:
            self.get_logger().error(f"❌ Trajectory execution failed with error code: {result.error_code}")
def execute_trajectory_on_robot(trajectory):
    robot_description = get_ur5e_robot_description()
    executor_node = UR5eTrajectoryExecutor(robot_description)

    executor_node.execute_trajectory(trajectory, point_time=0.3)

    executor_node.destroy_node()