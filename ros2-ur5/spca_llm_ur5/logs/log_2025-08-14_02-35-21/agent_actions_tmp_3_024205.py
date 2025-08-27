import time
import numpy as np
import cv2
import traceback
import rclpy
from geometry_msgs.msg import PoseStamped, Pose, PointStamped
from moveit.core.robot_state import RobotState
from moveit.core.kinematic_constraints import construct_joint_constraint
from moveit_msgs.srv import GetCartesianPath
from control_msgs.action import FollowJointTrajectory
from moveit.core.robot_state import robotStateToRobotStateMsg
from rclpy.duration import Duration
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped

def move_to_predefined(ctx, done_callback=None):
    """
    Move UR5 to xyz=(0.1, 0.3, 1.5) with orientation 'down' (x,y,z,w)=(1,0,0,0)    """
    ctx.log.info('_move_xyz')
    _move_xyz_down(ctx, 0.1, 0.3, 1.5, done_callback=done_callback)

def _move_xyz_down(ctx, x, y, z, done_callback=None):
    """ Move UR5 to xyz=(x,y,z) with orientation 'down' (x,y,z,w)=(1,0,0,0),
    by reusing the PoseStamped planning path.
    """
    ctx.wait_for_current_state(timeout_s=2.0)
    pose_goal = PoseStamped()
    pose_goal.header.frame_id = ctx.WORLD_FRAME
    pose_goal.header.stamp = ctx.node.get_clock().now().to_msg()
    pose_goal.pose.orientation.x = 1.0
    pose_goal.pose.orientation.y = 0.0
    pose_goal.pose.orientation.z = 0.0
    pose_goal.pose.orientation.w = 0.0
    pose_goal.pose.position.x = float(x)
    pose_goal.pose.position.y = float(y)
    pose_goal.pose.position.z = float(z)
    _move_via_ik(ctx, pose_goal.pose, tip=ctx.EEF_LINK, done_callback=done_callback)

def move_ur5_named(ctx, start_name: str, goal_name: str, done_callback=None):
    """
    Planning UR5 manipulator from a named 'start' to a named 'goal' state.
    Uses SRDF predefined states: 'home', 'up', 'zero' ('zero' is not recommended).
    """
    ctx.arm.set_start_state(configuration_name=start_name)
    ctx.arm.set_goal_state(configuration_name=goal_name)
    _plan_and_execute_with_callback(ctx, ctx.arm, ctx.log, done_callback, sleep_time=5.0)

def gripper_open(ctx, done_callback=None):
    """
    Planning gripper to open state (using SRDF 'open').
    """
    ctx.node.get_logger().info('\n\nPlanning gripper to open state')
    ctx.gripper.set_start_state_to_current_state()
    ctx.gripper.set_goal_state(configuration_name='open')
    _plan_and_execute_with_callback(ctx, ctx.gripper, ctx.node.get_logger(), done_callback, sleep_time=5.0)

def gripper_close(ctx, done_callback=None):
    """
    Planning gripper to close state (using SRDF 'close').
    """
    ctx.node.get_logger().info('\n\nPlanning gripper to close state')
    ctx.gripper.set_start_state_to_current_state()
    ctx.gripper.set_goal_state(configuration_name='close')
    _plan_and_execute_with_callback(ctx, ctx.gripper, ctx.node.get_logger(), done_callback, sleep_time=5.0)

def _plan_and_execute_with_callback(ctx, planning_component, logger, done_callback, sleep_time: float):
    success = False
    err_msg = ''
    trace = ''
    try:
        logger.info('Planning trajectory')
        plan_result = planning_component.plan()
        if plan_result:
            robot_trajectory = plan_result.trajectory
            group_name = planning_component.planning_group_name
            ctx.robot.execute(group_name, robot_trajectory)
            success = True
        else:
            logger.error('Planning failed')
            err_msg = 'Planning failed'
        time.sleep(sleep_time)
    except Exception as e:
        logger.error(f'Execution failed: {e}')
        success = False
        err_msg = str(e)
        trace = traceback.format_exc()
    finally:
        done_callback(success=success, msg=err_msg, trace=trace)

def _move_to_pose_stamped(ctx, pose_goal: PoseStamped, link: str='tool0', done_callback=None):
    """
    Plan to a PoseStamped goal.
    """
    ctx.log.info('\n\nPlanning UR5 manipulator with PoseStamped goal')
    ctx.arm.set_start_state_to_current_state()
    ctx.arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link=link)
    _plan_and_execute_with_callback(ctx, ctx.arm, ctx.log, done_callback, sleep_time=5.0)

def _move_with_joint_constraints(ctx, joint_values: dict, done_callback=None):
    """
    Plan with joint constraints.
    """
    ctx.log.info('\n\nPlanning UR5 manipulator with joint constraints')
    ctx.arm.set_start_state_to_current_state()
    robot_state = RobotState(ctx.robot.get_robot_model())
    robot_state.joint_positions = joint_values
    joint_constraint = construct_joint_constraint(robot_state=robot_state, joint_model_group=ctx.robot.get_robot_model().get_joint_model_group(ctx.ARM_GROUP))
    ctx.arm.set_goal_state(motion_plan_constraints=[joint_constraint])
    _plan_and_execute_with_callback(ctx, ctx.arm, ctx.log, done_callback, sleep_time=5.0)

def _move_via_ik(ctx, pose: Pose, tip: str='tool0', done_callback=None):
    """
    Compute IK for a specified Cartesian goal, unwrap near seed, then plan/execute.
    """
    logger = ctx.log
    logger.info('\n\nPlanning UR5 manipulator by computing IK for a specified Cartesian goal.')
    logger.info(f'Target pose: {pose}, tip: {tip}')
    robot_model = ctx.robot.get_robot_model()
    goal_robot_state = RobotState(robot_model)
    seed_state_for_ik = ctx.arm.get_start_state()
    goal_robot_state.set_joint_group_positions(ctx.arm.planning_group_name, seed_state_for_ik.get_joint_group_positions(ctx.arm.planning_group_name))
    logger.info(f'Seed state for IK:\nJoint positions: {goal_robot_state.get_joint_group_positions(ctx.arm.planning_group_name)}\n')
    ik_found = goal_robot_state.set_from_ik(joint_model_group_name=ctx.arm.planning_group_name, geometry_pose=pose, tip_name=tip, timeout=1.0)
    if not ik_found:
        logger.error(f'Could not find IK solution for target Cartesian pose: {pose}')
        logger.error('This means the desired Cartesian pose might be out of reach, in a singularity, or the IK solver timed out.')
        if callable(done_callback):
            done_callback(success=False, msg='IK solution not found.')
        return
    logger.info(f'IK solution found!\nJoint positions: {goal_robot_state.get_joint_group_positions(ctx.arm.planning_group_name)}\n')
    vals_before = goal_robot_state.get_joint_group_positions(ctx.arm.planning_group_name)
    vals_after = _unwrap_to_seed(ctx, goal_robot_state, seed_state_for_ik, ctx.arm.planning_group_name)
    logger.info(f'Unwrapped IK solution to seed state:\nJoint positions before: {vals_before}\nJoint positions after: {vals_after}\n')
    logger.info(f'Normalized IK solution to joint bounds:\nJoint positions: {goal_robot_state.get_joint_group_positions(ctx.arm.planning_group_name)}\n')
    ctx.arm.set_start_state_to_current_state()
    logger.info('Setting goal state with explicit Joint Constraints from IK solution...')
    ctx.arm.set_goal_state(robot_state=goal_robot_state)
    _plan_and_execute_with_callback(ctx, ctx.arm, logger, done_callback, sleep_time=5.0)

def _unwrap_to_seed(ctx, rs_goal: RobotState, rs_seed: RobotState, jmg_name: str):
    """
    Bring joint angles in rs_goal as close as possible to rs_seed by removing 2π multiples.
    """
    vals = rs_goal.get_joint_group_positions(jmg_name)
    seed = rs_seed.get_joint_group_positions(jmg_name)
    for i in range(len(vals)):
        delta = vals[i] - seed[i]
        vals[i] -= np.round(delta / (2.0 * np.pi)) * (2.0 * np.pi)
    rs_goal.set_joint_group_positions(jmg_name, vals)
    rs_goal.update()
    return vals

def _cartesian_runner_async(ctx, x: float, y: float, z: float, done_callback=None):
    """
    Computes a Cartesian path to a target waypoint and executes it, using callbacks.
    (non-blocking version)
    """
    ctx.log.info('Starting non-blocking Cartesian path computation.')
    req = GetCartesianPath.Request()
    req.group_name = ctx.ARM_GROUP
    waypoint = Pose()
    waypoint.position.x = x
    waypoint.position.y = y
    waypoint.position.z = z
    waypoint.orientation.x = 1.0
    waypoint.orientation.y = 0.0
    waypoint.orientation.z = 0.0
    waypoint.orientation.w = 0.0
    req.waypoints.append(waypoint)
    req.max_step = 0.01
    req.jump_threshold = 0.0
    ctx.cart_cli.wait_for_service()
    future = ctx.cart_cli.call_async(req)
    future.add_done_callback(lambda future: _cartesian_path_cb(ctx, future, done_callback))

def _cartesian_path_cb(ctx, future, done_callback):
    """
    Callback for the GetCartesianPath service.
    """
    res = future.result()
    if res and res.solution.joint_trajectory.points:
        ctx.log.info('Cartesian path computed successfully. Sending to action server.')
        _send_follow_joint_trajectory_async(ctx, res.solution.joint_trajectory, done_callback)
    else:
        ctx.log.error('Cartesian path service returned no trajectory.')
        if callable(done_callback):
            done_callback(success=False, msg='No Cartesian path found.')

def _send_follow_joint_trajectory_async(ctx, traj, done_callback=None):
    """
    Sends a trajectory to the FollowJointTrajectory action server.
    """
    ctx.follow_traj_ac.wait_for_server()
    goal = FollowJointTrajectory.Goal()
    goal.trajectory = traj
    goal.trajectory.header.stamp = ctx.node.get_clock().now().to_msg()
    ctx.node.get_logger().info('Sending goal')
    send_goal_future = ctx.follow_traj_ac.send_goal_async(goal)
    send_goal_future.add_done_callback(lambda gh_fut: _follow_traj_cb(ctx, gh_fut, done_callback))

def _follow_traj_cb(ctx, future, done_callback):
    """
    Callback for the FollowJointTrajectory action server.
    """
    goal_handle = future.result()
    if not goal_handle.accepted:
        ctx.log.error('Goal rejected by action server.')
        if callable(done_callback):
            done_callback(success=False, msg='Goal rejected.')
        return
    ctx.log.info('Goal accepted by action server.')
    get_result_future = goal_handle.get_result_async()
    get_result_future.add_done_callback(lambda gr_fut: _follow_traj_result_cb(ctx, gr_fut, done_callback))

def _follow_traj_result_cb(ctx, future, done_callback):
    """
    Callback for the result of the FollowJointTrajectory action.
    This is the final step, so it calls the main executor's done_callback.
    """
    status = future.result().status
    result = future.result().result
    ctx.log.info(f'Action finished with status: {status}, result: {result}')
    if callable(done_callback):
        success = status == 4
        done_callback(success=success, msg=f'Action finished with status {status}')

def _convert_rgb_msg(ctx, msg):
    """
    ROS Image -> OpenCV BGR8 (from ImageProcessor.image_callback).
    """
    return ctx.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

def _detect_edges(ctx, bgr):
    """
    Canny edges (from ImageProcessor.image_callback).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def _depth_msg_to_colormap(ctx, msg):
    """
    Depth passthrough, mm->m if 16UC1, normalize, apply JET colormap.
    Mirrors RGBDProcessor.depth_callback (keeps its structure).
    Returns (depth_colormap_bgr, (dmin, dmax)).
    """
    depth_raw = ctx.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    depth = np.array(depth_raw, dtype=np.float32)
    valid = depth > 0
    if np.any(valid):
        (dmin, dmax) = (float(depth[valid].min()), float(depth[valid].max()))
    else:
        (dmin, dmax) = (0.0, 0.0)
    if msg.encoding == '16UC1' and dmax > 1000:
        depth = depth * 0.001
        dmin /= 1000
        dmax /= 1000
    if dmax > dmin:
        norm = (depth - dmin) / (dmax - dmin)
    else:
        norm = np.zeros_like(depth)
    norm = np.clip(norm, 0.0, 1.0)
    depth_u8 = (depth * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    return (depth_colormap, (dmin, dmax))

def _stack_rgb_edges_depth(ctx, rgb_bgr, edges_bgr, depth_vis_bgr):
    """
    Stack RGB+edges and depth side-by-side (from RGBDProcessor.display_if_ready).
    """
    left = cv2.hconcat([rgb_bgr, edges_bgr])
    right = depth_vis_bgr
    combined = cv2.hconcat([left, right])
    return combined

def _intrinsics_from_camera_info(ctx, msg):
    """
    Extract 3x3 K (from ClosestPoint3D.info_cb).
    """
    return np.array(msg.k).reshape(3, 3)

def _closest_point_from_depth(ctx, depth_msg, K):
    """
    Find minimum valid depth pixel and back-project (from ClosestPoint3D.depth_cb).
    Returns (x, y, z, u, v).
    """
    depth_image = ctx.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
    depth = np.array(depth_image, dtype=np.float32)
    if depth.max() > 1000:
        depth *= 0.001
    valid = depth > 0.001
    if not np.any(valid):
        return None
    idx = np.argmin(np.where(valid, depth, np.inf))
    (v, u) = np.unravel_index(idx, depth.shape)
    z = float(depth[v, u])
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return (x, y, z, int(u), int(v))

def _make_point_stamped(ctx, x, y, z, header):
    """
    Convenience builder (from ClosestPoint3D.depth_cb).
    """
    pt = PointStamped()
    pt.header = header
    pt.point.x = float(x)
    pt.point.y = float(y)
    pt.point.z = float(z)
    return pt

def touch_blue_cube(ctx, done_callback=None):
    if ctx.cancelled():
        raise RuntimeError('cancelled')
    try:
        (x_w, y_w, z_w) = _get_item_world_point(ctx, 'blue_cube')
    except RuntimeError as e:
        done_callback(success=False, msg=str(e))
        return
    approach = PoseStamped()
    approach.header.frame_id = ctx.WORLD_FRAME
    approach.header.stamp = ctx.node.get_clock().now().to_msg()
    approach.pose.position.x = x_w
    approach.pose.position.y = y_w
    approach.pose.position.z = z_w + 0.2
    approach.pose.orientation.x = 1.0
    approach.pose.orientation.y = 0.0
    approach.pose.orientation.z = 0.0
    approach.pose.orientation.w = 0.0
    if not ctx.wait_for_current_state(timeout_s=2.0):
        done_callback(success=False, msg='failed to get current robot state')
        return

    def _approach_cb(success, msg=None, trace=None):
        if not success:
            done_callback(success=False, msg=msg)
        else:
            _cartesian_runner_async(ctx, x_w, y_w, z_w, done_callback=done_callback)
    _move_via_ik(ctx, approach.pose, tip=ctx.EEF_LINK, done_callback=_approach_cb)

def _segment_item_mask(ctx, item):
    rgb = ctx.latest_rgb
    if rgb is None:
        raise RuntimeError('no rgb image')
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    if item == 'blue_block' or item == 'blue_cube':
        lower = np.array([100, 150, 50], dtype=np.uint8)
        upper = np.array([130, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
    elif item == 'green_cube' or item == 'green_block':
        lower = np.array([40, 100, 50], dtype=np.uint8)
        upper = np.array([80, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
    elif item == 'red_cube' or item == 'red_block':
        lower1 = np.array([0, 150, 50], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, 150, 50], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = mask1 | mask2
    else:
        raise RuntimeError(f'Unknown item: {item}')
    return mask > 0

def _get_item_world_point(ctx, item):
    if ctx.latest_rgb is None or ctx.latest_depth is None or ctx.K_depth is None or (ctx.latest_depth_header is None) or (ctx.depth_frame is None):
        raise RuntimeError('no rgb or depth or intrinsics available')
    mask = _segment_item_mask(ctx, item)
    depth = ctx.latest_depth
    masked_depth = np.where(mask, depth, np.inf)
    if np.isinf(masked_depth).all():
        raise RuntimeError(f'No depth for item {item}')
    (v, u) = np.unravel_index(np.argmin(masked_depth), depth.shape)
    z = float(depth[v, u])
    fx = ctx.K_depth[0, 0]
    fy = ctx.K_depth[1, 1]
    cx = ctx.K_depth[0, 2]
    cy = ctx.K_depth[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pt_cam = PointStamped()
    pt_cam.header = ctx.latest_depth_header
    pt_cam.point.x = float(x)
    pt_cam.point.y = float(y)
    pt_cam.point.z = float(z)
    try:
        pt_world = ctx.tfbuf.transform(pt_cam, ctx.WORLD_FRAME, timeout=Duration(seconds=0.5))
    except Exception as e:
        raise RuntimeError(f'tf transform failed: {e}')
    return (pt_world.point.x, pt_world.point.y, pt_world.point.z)

def touch_item(ctx, g, i, done_callback=None):
    if g != 'gripper':
        raise RuntimeError(f'Unsupported end effector: {g}')
    (x_w, y_w, z_w) = _get_item_world_point(ctx, i)
    approach = PoseStamped()
    approach.header.frame_id = ctx.WORLD_FRAME
    approach.header.stamp = ctx.node.get_clock().now().to_msg()
    approach.pose.position.x = x_w
    approach.pose.position.y = y_w
    approach.pose.position.z = z_w + 0.2
    approach.pose.orientation.x = 1.0
    approach.pose.orientation.y = 0.0
    approach.pose.orientation.z = 0.0
    approach.pose.orientation.w = 0.0

    def _approach_cb(success, msg=None, trace=None):
        if not success:
            done_callback(success=False, msg=msg)
            return
        _cartesian_runner_async(ctx, x_w, y_w, z_w, done_callback=done_callback)
    _move_via_ik(ctx, approach.pose, tip=ctx.EEF_LINK, done_callback=_approach_cb)

def touch_red_cube(ctx, done_callback=None):
    if ctx.cancelled():
        raise RuntimeError('cancelled')
    try:
        (x_w, y_w, z_w) = _get_item_world_point(ctx, 'red_cube')
    except RuntimeError as e:
        done_callback(success=False, msg=str(e))
        return
    _cartesian_runner_async(ctx, x_w, y_w, z_w, done_callback=done_callback)

def touch_green_cube(ctx, done_callback=None):
    if ctx.cancelled():
        raise RuntimeError('cancelled')
    try:
        (x_w, y_w, z_w) = _get_item_world_point(ctx, 'green_cube')
    except RuntimeError as e:
        done_callback(success=False, msg=str(e))
        return
    approach = PoseStamped()
    approach.header.frame_id = ctx.WORLD_FRAME
    approach.header.stamp = ctx.node.get_clock().now().to_msg()
    approach.pose.position.x = x_w
    approach.pose.position.y = y_w
    approach.pose.position.z = z_w + 0.2
    approach.pose.orientation.x = 1.0
    approach.pose.orientation.y = 0.0
    approach.pose.orientation.z = 0.0
    approach.pose.orientation.w = 0.0
    if not ctx.wait_for_current_state(timeout_s=2.0):
        done_callback(success=False, msg='failed to get current robot state')
        return

    def _approach_cb(success, msg=None, trace=None):
        if not success:
            done_callback(success=False, msg=msg)
        else:
            _cartesian_runner_async(ctx, x_w, y_w, z_w, done_callback=done_callback)
    _move_via_ik(ctx, approach.pose, tip=ctx.EEF_LINK, done_callback=_approach_cb)

def move_to_blue_cube(ctx, done_callback=None):
    if ctx.cancelled():
        raise RuntimeError('cancelled')
    try:
        (x_w, y_w, z_w) = _get_item_world_point(ctx, 'blue_cube')
    except RuntimeError as e:
        done_callback(success=False, msg=str(e))
        return
    approach = PoseStamped()
    approach.header.frame_id = ctx.WORLD_FRAME
    approach.header.stamp = ctx.node.get_clock().now().to_msg()
    approach.pose.position.x = x_w
    approach.pose.position.y = y_w
    approach.pose.position.z = z_w + 0.2
    approach.pose.orientation.x = 1.0
    approach.pose.orientation.y = 0.0
    approach.pose.orientation.z = 0.0
    approach.pose.orientation.w = 0.0
    if not ctx.wait_for_current_state(timeout_s=2.0):
        done_callback(success=False, msg='failed to get current robot state')
        return

    def _approach_cb(success, msg=None, trace=None):
        if not success:
            done_callback(success=False, msg=msg)
        else:
            done_callback(success=True)
    _move_via_ik(ctx, approach.pose, tip=ctx.EEF_LINK, done_callback=_approach_cb)

def move_to_red_cube(ctx, done_callback=None):
    if ctx.cancelled():
        raise RuntimeError('cancelled')
    try:
        (x_w, y_w, z_w) = _get_item_world_point(ctx, 'red_cube')
    except RuntimeError as e:
        done_callback(success=False, msg=str(e))
        return
    approach = PoseStamped()
    approach.header.frame_id = ctx.WORLD_FRAME
    approach.header.stamp = ctx.node.get_clock().now().to_msg()
    approach.pose.position.x = x_w
    approach.pose.position.y = y_w
    approach.pose.position.z = z_w + 0.3
    approach.pose.orientation.x = 1.0
    approach.pose.orientation.y = 0.0
    approach.pose.orientation.z = 0.0
    approach.pose.orientation.w = 0.0
    if not ctx.wait_for_current_state(timeout_s=2.0):
        done_callback(success=False, msg='failed to get current robot state')
        return
    _move_via_ik(ctx, approach.pose, tip=ctx.EEF_LINK, done_callback=done_callback)