import sys

import numpy as np
import rospy
import moveit_commander 
from moveit_commander import MoveGroupCommander, PlanningSceneInterface, roscpp_initialize
from geometry_msgs.msg import Pose
from MPC_ros_control.casadi_mpc import nmpc_controller

# 初始化ROS节点 
roscpp_initialize(sys.argv)
rospy.init_node('nmpc_moveit_controller', anonymous=True)

# 初始化MoveIt!相关对象
robot = moveit_commander.RobotCommander()
arm = MoveGroupCommander("arm")  # 的MoveIt!配置中机械臂被命名为"arm"
scene = PlanningSceneInterface()
cartesian = rospy.get_param('~cartesian', True) #使用笛卡尔空间规划

# 当运动规划失败后，允许重新规划
arm.allow_replanning(True)
        
# 设置目标位置所使用的参考坐标系
arm.set_pose_reference_frame('world')
current_joint = arm.get_current_joint_values()
current_pose = arm.get_current_pose()
# 定义目标位置为Pose消息（需要根据实际的要求来设置相应的值）
target_pose = Pose()#？？？？
plan=arm.plan()

waypoints=plan.trajectory

while not rospy.is_shutdown:
    pos_error = arm.get_goal_position_tolerance()
    ori_error = arm.get_goal_orientation_tolerance()

    if pos_error > 0.01 or ori_error > 0.01*np.pi:  # 距离大于1厘米,角度大于0.01*pi
        control_command = compute_nmpc_control(waypoints,target_position, current_ee_position)
        apply_control_to_robot(control_command)
    else:
        rospy.loginfo("mission success!!!")
