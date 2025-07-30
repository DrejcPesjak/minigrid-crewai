#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetCartesianPath
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import Pose
from builtin_interfaces.msg import Time

class CartRunner(Node):
    def __init__(self):
        super().__init__('cart_run')
        # Create service client
        self.cli = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.cli.wait_for_service()
        print('Service /compute_cartesian_path is available')

        # Prepare GetCartesianPath request
        req = GetCartesianPath.Request()
        req.group_name = 'ur5_manipulator'

        # Get parameters for the request
        self.declare_parameter('target_x', 0.0)
        self.declare_parameter('target_y', 0.5)
        self.declare_parameter('target_z', 1.0)

        # Create one Pose waypoint and append to list
        waypoint = Pose()
        waypoint.position.x = self.get_parameter('target_x').value
        waypoint.position.y = self.get_parameter('target_y').value
        waypoint.position.z = self.get_parameter('target_z').value
        # waypoint.orientation.w = 1.0
        req.waypoints.append(waypoint)  # use append, not add()
        print('Waypoint added:', waypoint)

        req.max_step = 0.01
        req.jump_threshold = 0.0

        # Call service asynchronously
        self.cli.call_async(req).add_done_callback(self.cb)

    def cb(self, future):
        res = future.result()
        ac = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )
        ac.wait_for_server()
        print('Action server is available')

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = res.solution.joint_trajectory

        # # 1 second from now
        # now = self.get_clock().now().to_msg()
        # goal.trajectory.header.stamp.sec = now.sec + 1
        # goal.trajectory.header.stamp.nanosec = now.nanosec

        print('Sending goal')# with header.stamp =')#, goal.trajectory.header.stamp)
        send_goal_future = ac.send_goal_async(goal)
        send_goal_future.add_done_callback(lambda gh_fut: rclpy.shutdown())


def main():
    rclpy.init()
    node = CartRunner()
    rclpy.spin(node)

if __name__ == '__main__':
    main()