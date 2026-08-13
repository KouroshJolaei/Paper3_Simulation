# get_fk.py
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import RobotState

import tf2_ros

class FKClient(Node):
    def __init__(self):
        super().__init__('fk_client_node')
        # FK service
        self.client = self.create_client(GetPositionFK, '/compute_fk')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('⏳ Waiting for /compute_fk ...')

        # TF buffer/listener for frame transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

    def get_fk(self, joint_names, joint_positions, ee_link='tool0', ref_frame=None):
        """
        Returns PoseStamped of ee_link. If ref_frame is provided and TF is available,
        transforms the pose into ref_frame. Otherwise returns in the planning frame.
        """
        # Build FK request
        req = GetPositionFK.Request()
        req.fk_link_names = [ee_link]
        req.robot_state = RobotState(
            joint_state=JointState(name=list(joint_names), position=list(joint_positions))
        )
        # Some MoveIt setups ignore header.frame_id; we still set it for completeness.
        req.header.frame_id = ref_frame if ref_frame else ''

        # Call FK
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        res = future.result()
        if res is None or not res.pose_stamped:
            self.get_logger().error('❌ FK call failed or returned no poses')
            return None

        # Check error code (1 = SUCCESS)
        if hasattr(res, 'error_code') and getattr(res.error_code, 'val', 0) != 1:
            self.get_logger().error(f'❌ FK error_code: {res.error_code.val}')
            return None

        ps = res.pose_stamped[0]  # PoseStamped in the planning frame
        planning_frame = ps.header.frame_id or '(unknown)'
        self.get_logger().debug(f'FK returned in planning frame: {planning_frame}')

        # If a target frame is requested and different, try to transform
        if ref_frame and ref_frame != planning_frame:
            try:
                can = self.tf_buffer.can_transform(
                    ref_frame, planning_frame, rclpy.time.Time(),
                    timeout=Duration(seconds=1.0)
                )
                if not can:
                    self.get_logger().warn(
                        f'⚠️ No TF from {planning_frame} → {ref_frame}. '
                        f'Returning in {planning_frame}.'
                    )
                    return ps

                ps_in_ref = self.tf_buffer.transform(
                    ps, target_frame=ref_frame, timeout=Duration(seconds=1.0)
                )
                ps_in_ref.header.frame_id = ref_frame
                return ps_in_ref
            except Exception as e:
                self.get_logger().warn(
                    f'⚠️ TF transform {planning_frame}→{ref_frame} failed: {e}. '
                    f'Returning in {planning_frame}.'
                )
                return ps

        # No transform requested or already in desired frame
        return ps

# Optional local test
if __name__ == '__main__':
    rclpy.init()
    node = FKClient()
    names = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint']
    positions = [0.0, -1.57, 1.57, 0.0, 1.57, 0.0]
    ps = node.get_fk(names, positions, ee_link='tool0', ref_frame='base_link')
    if ps:
        p = ps.pose.position
        q = ps.pose.orientation
        node.get_logger().info(
            f'Pose in {ps.header.frame_id}: '
            f'pos=({p.x:.3f},{p.y:.3f},{p.z:.3f}) '
            f'quat=({q.x:.3f},{q.y:.3f},{q.z:.3f},{q.w:.3f})'
        )
    node.destroy_node()
    rclpy.shutdown()
