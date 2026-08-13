# define_ur5e_robot.py
import sys
sys.path.append("/home/kourosh/moveit2_ws/install/pymoveit2/local/lib/python3.10/dist-packages")
from pymoveit2.robots import ur

def get_ur5e_robot_description():
    return {
        "joint_names": ur.joint_names(),
        "base_link_name": ur.base_link_name(),
        "end_effector_name": ur.end_effector_name(),
        "move_group_name": ur.MOVE_GROUP_ARM
    }
