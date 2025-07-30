# agent_actions.py
from typing import Dict, Tuple, Optional
import numpy as np

import rclpy
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, Pose
from trajectory_msgs.msg import JointTrajectory
from moveit.core.robot_state import RobotState
from moveit.core.kinematic_constraints import construct_joint_constraint
from control_msgs.action import FollowJointTrajectory, GripperCommand
from moveit_msgs.srv import GetCartesianPath

# ----------------- helpers (allowed to return) -----------------

def _plan_and_execute(ctx, planning_component):
    if ctx.cancelled():
        raise RuntimeError("cancelled")
    plan = planning_component.plan()
    if not plan:
        raise RuntimeError("planning failed")
    if ctx.cancelled():
        raise RuntimeError("cancelled")
    ctx.robot.execute(planning_component.planning_group_name, plan.trajectory)

def _move_arm_to_posestamped(ctx, frame_id: str, x: float, y: float, z: float,
                             qw: float = 1.0, qx: float = 0.0, qy: float = 0.0, qz: float = 0.0):
    p = PoseStamped()
    p.header.frame_id = frame_id
    p.pose.position.x = float(x)
    p.pose.position.y = float(y)
    p.pose.position.z = float(z)
    p.pose.orientation.w = float(qw)
    p.pose.orientation.x = float(qx)
    p.pose.orientation.y = float(qy)
    p.pose.orientation.z = float(qz)
    ctx.arm.set_start_state_to_current_state()
    ctx.arm.set_goal_state(pose_stamped=p, pose_link=ctx.EEF_LINK)
    _plan_and_execute(ctx, ctx.arm)

def _move_arm_into_jointconstraints(ctx, joint_values: Dict[str, float]):
    model = ctx.robot.get_robot_model()
    group = model.get_joint_model_group(ctx.ARM_GROUP)
    rs: RobotState = ctx.robot.get_current_state()
    for j, v in joint_values.items():
        rs.set_joint_positions(j, [float(v)])
    jc = construct_joint_constraint(robot_state=rs, joint_model_group=group)
    ctx.arm.set_start_state_to_current_state()
    ctx.arm.set_goal_state(motion_plan_constraints=[jc])
    _plan_and_execute(ctx, ctx.arm)

def _canny_edge_detection(ctx, low: int = 50, high: int = 150):
    import cv2
    img = ctx.latest_rgb
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, int(low), int(high))
    return edges

def _closest_point_3d(ctx) -> Optional[Tuple[float, float, float]]:
    depth = ctx.latest_depth
    K = ctx.K_depth if ctx.K_depth is not None else ctx.K_rgb
    if depth is None or K is None:
        return None
    valid = depth > 0.001
    if not np.any(valid):
        return None
    idx = np.argmin(np.where(valid, depth, np.inf))
    v, u = np.unravel_index(idx, depth.shape)
    z = float(depth[v, u])
    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return (x, y, z)  # in depth camera frame

def _follow_trajectory(ctx, traj: JointTrajectory):
    goal = FollowJointTrajectory.Goal()
    goal.trajectory = traj
    if not ctx.follow_traj_ac.server_is_ready():
        ctx.follow_traj_ac.wait_for_server(timeout_sec=2.0)
    fut = ctx.follow_traj_ac.send_goal_async(goal)
    rclpy.spin_until_future_complete(ctx.node, fut)
    res = fut.result()
    if res is None:
        raise RuntimeError("follow_joint_trajectory failed")

def _move_cartesian_srv(ctx, x: float, y: float, z: float,
                        max_step: float = 0.01, jump_threshold: float = 0.0):
    req = GetCartesianPath.Request()
    req.group_name = ctx.ARM_GROUP
    req.max_step = float(max_step)
    req.jump_threshold = float(jump_threshold)
    wp = Pose()
    wp.position.x = float(x)
    wp.position.y = float(y)
    wp.position.z = float(z)
    wp.orientation.w = 1.0
    req.waypoints.append(wp)
    if not ctx.cart_cli.service_is_ready():
        ctx.cart_cli.wait_for_service(timeout_sec=2.0)
    fut = ctx.cart_cli.call_async(req)
    rclpy.spin_until_future_complete(ctx.node, fut)
    res = fut.result()
    if res is None or not res.solution or not res.solution.joint_trajectory.points:
        raise RuntimeError("cartesian path failed")
    _follow_trajectory(ctx, res.solution.joint_trajectory)

# ----------------- actions (no return, ctx + strings only) -----------------

def move_arm_from_home_to_up(ctx):
    if ctx.cancelled():
        raise RuntimeError("cancelled")
    ctx.arm.set_start_state(configuration_name="home")
    ctx.arm.set_goal_state(configuration_name="up")
    _plan_and_execute(ctx, ctx.arm)

def gripper_open(ctx):
    if ctx.cancelled():
        raise RuntimeError("cancelled")
    ctx.gripper.set_start_state_to_current_state()
    ctx.gripper.set_goal_state(configuration_name="open")
    _plan_and_execute(ctx, ctx.gripper)

def gripper_close(ctx):
    if ctx.cancelled():
        raise RuntimeError("cancelled")
    ctx.gripper.set_start_state_to_current_state()
    ctx.gripper.set_goal_state(configuration_name="close")
    _plan_and_execute(ctx, ctx.gripper)
