# gripper_io.py
import rclpy
from rclpy.node import Node
from ur_msgs.srv import SetIO

DO_OPEN = 2
DO_CLOSE = 3
DO_STROBE = 1
DO_BITS = {4: 1, 5: 2, 6: 4, 7: 8}

class GripperIOClient(Node):
    def __init__(self):
        super().__init__("gripper_io_client")
        self.cli = self.create_client(SetIO, "/io_and_status_controller/set_io")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /io_and_status_controller/set_io ...")

    def _set_do(self, pin: int, state: float = 1.0) -> bool:
        req = SetIO.Request(fun=1, pin=pin, state=float(state))
        fut = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        return bool(fut.result() and fut.result().success)

    def open(self):
        self.get_logger().info("OPEN (DO2)")
        self._set_do(DO_OPEN, 1.0)

    def close(self):
        self.get_logger().info("CLOSE (DO3)")
        self._set_do(DO_CLOSE, 1.0)

    def set_percent(self, percent: float):
        p = max(0.0, min(100.0, float(percent)))
        byte_val = int(round(p * 255.0 / 100.0))     # 0..255
        nibble = max(0, min(15, int(round(byte_val / 17.0))))  # ~0..15
        self.get_logger().info(f"PARTIAL {p:.1f}% (byte≈{byte_val}, nibble={nibble})")

        # Set DO4..DO7 bits for nibble
        for pin, mask in DO_BITS.items():
            if nibble & mask:
                self._set_do(pin, 1.0)
        # Pulse strobe on DO1 (TP script will clear DO1 and DO4..DO7)
        self._set_do(DO_STROBE, 1.0)

def _make_node():
    """Create a node, initializing/shutting down ROS only if we started it."""
    started_here = False
    if not rclpy.ok():
        rclpy.init()
        started_here = True
    node = GripperIOClient()
    return node, started_here

def open_gripper():
    node, started_here = _make_node()
    node.open()
    node.destroy_node()
    if started_here:
        rclpy.shutdown()

def close_gripper():
    node, started_here = _make_node()
    node.close()
    node.destroy_node()
    if started_here:
        rclpy.shutdown()

def set_gripper_percent(percent: float):
    node, started_here = _make_node()
    node.set_percent(percent)
    node.destroy_node()
    if started_here:
        rclpy.shutdown()
