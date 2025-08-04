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
import cv2
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from moveit_msgs.msg import RobotState as RobotStateMsg

from moveit_msgs.msg import RobotState as RobotStateMsg
from sensor_msgs.msg import JointState as JointStateMsg
from moveit_msgs.srv import GetCartesianPath
import rclpy
import time


def _cartesian_like_cart_runner(ctx, pose):  # pose: geometry_msgs/Pose
    ctx.node.get_logger().info("HEEELEEPPPPPPP")
    # 1) Service request: copy your working code’s fields
    req = GetCartesianPath.Request()
    req.group_name = 'ur5_manipulator'
    # NOTE: do NOT set link_name here (let MoveIt pick the group’s EE link)
    req.waypoints.append(pose)
    req.max_step = 0.01
    req.jump_threshold = 0.0

    if not ctx.cart_cli.service_is_ready():
        ctx.cart_cli.wait_for_service(timeout_sec=5.0)
    fut = ctx.cart_cli.call_async(req)
    rclpy.spin_until_future_complete(ctx.node, fut)
    res = fut.result()
    if res is None or not res.solution or not res.solution.joint_trajectory.points:
        raise RuntimeError("empty cartesian solution")

    jt = res.solution.joint_trajectory
    # DEBUG: make sure this looks sane
    ctx.node.get_logger().info(
        f"[cart_runner_clone] joints={jt.joint_names} points={len(jt.points)} "
        f"t0={jt.points[0].time_from_start.sec}.{jt.points[0].time_from_start.nanosec}"
    )

    # 2) Action goal: create a **fresh** client like your working script
    ac = ActionClient(ctx.node, FollowJointTrajectory,
                      '/joint_trajectory_controller/follow_joint_trajectory')
    ac.wait_for_server()
    goal = FollowJointTrajectory.Goal()
    goal.trajectory = jt

    gh_fut = ac.send_goal_async(goal)
    rclpy.spin_until_future_complete(ctx.node, gh_fut)
    gh = gh_fut.result()
    if gh is None or not gh.accepted:
        raise RuntimeError("trajectory goal rejected by controller")

    # wait for the execution result so we see controller errors
    res_fut = gh.get_result_async()
    rclpy.spin_until_future_complete(ctx.node, res_fut)
    result = res_fut.result().result
    if getattr(result, "error_code", 0) != 0:
        raise RuntimeError(f"trajectory execution failed (code {result.error_code}): "
                           f"{getattr(result, 'error_string', '')}")

def _plan_and_execute(ctx, planning_component):
    if ctx.cancelled():
        raise RuntimeError('cancelled')
    plan = planning_component.plan()
    if not plan:
        raise RuntimeError('planning failed')
    if ctx.cancelled():
        raise RuntimeError('cancelled')
    ctx.robot.execute(planning_component.planning_group_name, plan.trajectory)

def _move_arm_to_posestamped(ctx, frame_id: str, x: float, y: float, z: float, qw: float=1.0, qx: float=0.0, qy: float=0.0, qz: float=0.0):
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
    ctx.arm.set_goal_state(pose_stamped_msg=p, pose_link=ctx.EEF_LINK)
    _plan_and_execute(ctx, ctx.arm)

def _move_arm_into_jointconstraints(ctx, joint_values: dict):
    model = ctx.robot.get_robot_model()
    group = model.get_joint_model_group(ctx.ARM_GROUP)
    rs: RobotState = ctx.robot.get_current_state()
    for (j, v) in joint_values.items():
        rs.set_joint_positions(j, [float(v)])
    jc = construct_joint_constraint(robot_state=rs, joint_model_group=group)
    ctx.arm.set_start_state_to_current_state()
    ctx.arm.set_goal_state(motion_plan_constraints=[jc])
    _plan_and_execute(ctx, ctx.arm)

def _canny_edge_detection(ctx, low: int=50, high: int=150):
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
    (v, u) = np.unravel_index(idx, depth.shape)
    z = float(depth[v, u])
    (fx, fy) = (float(K[0, 0]), float(K[1, 1]))
    (cx, cy) = (float(K[0, 2]), float(K[1, 2]))
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return (x, y, z)

def _follow_trajectory(ctx, traj: JointTrajectory):
    goal = FollowJointTrajectory.Goal()
    goal.trajectory = traj
    if not ctx.follow_traj_ac.server_is_ready():
        ctx.follow_traj_ac.wait_for_server(timeout_sec=2.0)
    fut = ctx.follow_traj_ac.send_goal_async(goal)
    rclpy.spin_until_future_complete(ctx.node, fut)
    res = fut.result()
    if res is None:
        raise RuntimeError('follow_joint_trajectory failed')

def _move_cartesian_srv(ctx, x: float, y: float, z: float, max_step: float=0.01, jump_threshold: float=0.0):
    req = GetCartesianPath.Request()
    req.group_name = ctx.ARM_GROUP
    req.link_name = ctx.EEF_LINK
    req.max_step = float(max_step)
    req.jump_threshold = float(jump_threshold)
    wp = Pose()
    wp.position.x = float(x)
    wp.position.y = float(y)
    wp.position.z = float(z)
    wp.orientation.w = 1.0
    wp.orientation.x = 0.0
    wp.orientation.y = 0.0
    wp.orientation.z = 0.0
    req.waypoints.append(wp)
    if not ctx.cart_cli.service_is_ready():
        ctx.cart_cli.wait_for_service(timeout_sec=2.0)
    fut = ctx.cart_cli.call_async(req)
    rclpy.spin_until_future_complete(ctx.node, fut)
    res = fut.result()
    if res is None or not res.solution or (not res.solution.joint_trajectory.points):
        raise RuntimeError('cartesian path failed')
    _follow_trajectory(ctx, res.solution.joint_trajectory)

def move_arm_from_home_to_up(ctx):
    if ctx.cancelled():
        raise RuntimeError('cancelled')
    ctx.arm.set_start_state(configuration_name='home')
    ctx.arm.set_goal_state(configuration_name='up')
    _plan_and_execute(ctx, ctx.arm)

def move_arm_from_up_to_home(ctx):
    if ctx.cancelled():
        raise RuntimeError('cancelled')
    ctx.arm.set_start_state(configuration_name='up')
    ctx.arm.set_goal_state(configuration_name='home')
    _plan_and_execute(ctx, ctx.arm)

def gripper_open(ctx):
    if ctx.cancelled():
        raise RuntimeError('cancelled')
    ctx.gripper.set_start_state_to_current_state()
    ctx.gripper.set_goal_state(configuration_name='open')
    _plan_and_execute(ctx, ctx.gripper)

def gripper_close(ctx):
    if ctx.cancelled():
        raise RuntimeError('cancelled')
    ctx.gripper.set_start_state_to_current_state()
    ctx.gripper.set_goal_state(configuration_name='close')
    _plan_and_execute(ctx, ctx.gripper)

def touch_item(ctx, hand, item):
    if ctx.cancelled():
        raise RuntimeError('cancelled')
    parts = item.split('_')
    if len(parts) < 2:
        raise RuntimeError(f"cannot parse color from item '{item}'")
    color = parts[-1]
    pt_cam = _segment_color_point(ctx, color)
    if pt_cam is None:
        raise RuntimeError(f"failed to localize item '{item}'")
    ps_cam = PoseStamped()
    if ctx.latest_rgb is not None and ctx.rgb_frame:
        ps_cam.header.frame_id = ctx.rgb_frame
    elif ctx.latest_depth is not None and ctx.depth_frame:
        ps_cam.header.frame_id = ctx.depth_frame
    else:
        raise RuntimeError('no valid camera frame for localization')
    ps_cam.header.stamp = ctx.node.get_clock().now().to_msg()
    (ps_cam.pose.position.x, ps_cam.pose.position.y, ps_cam.pose.position.z) = pt_cam
    ps_cam.pose.orientation.w = 1.0
    ps_cam.pose.orientation.x = 0.0
    ps_cam.pose.orientation.y = 0.0
    ps_cam.pose.orientation.z = 0.0
    try:
        ps_world = ctx.tfbuf.transform(ps_cam, ctx.WORLD_FRAME, timeout=Duration(seconds=1.0))
    except Exception:
        raise RuntimeError('TF failure transforming to world frame')
    x = ps_world.pose.position.x
    y = ps_world.pose.position.y
    z = ps_world.pose.position.z
    (qw, qx, qy, qz) = (0.0, 1.0, 0.0, 0.0)
    approach = PoseStamped()
    approach.header.frame_id = ctx.WORLD_FRAME
    approach.header.stamp = ctx.node.get_clock().now().to_msg()
    approach.pose.position.x = 0.1#x
    approach.pose.position.y = 0.3#y
    approach.pose.position.z = 1.5#z + 0.1
    approach.pose.orientation.w = qw
    approach.pose.orientation.x = qx
    approach.pose.orientation.y = qy
    approach.pose.orientation.z = qz
    move_cartesian_via_controller(ctx, approach)
    # approach
    #approach_pose = approach.pose  # geometry_msgs/Pose
    #_cartesian_like_cart_runner(ctx, approach_pose)
    if ctx.cancelled():
        raise RuntimeError('cancelled')
    target = PoseStamped()
    target.header.frame_id = ctx.WORLD_FRAME
    target.header.stamp = ctx.node.get_clock().now().to_msg()
    target.pose.position.x = x
    target.pose.position.y = y
    target.pose.position.z = z
    target.pose.orientation.w = qw
    target.pose.orientation.x = qx
    target.pose.orientation.y = qy
    target.pose.orientation.z = qz
    move_cartesian_via_controller(ctx, target)
    # target
    #target_pose = target.pose
    #_cartesian_like_cart_runner(ctx, target_pose)


def _segment_color_point(ctx, color_name):
    img = ctx.latest_rgb
    depth = ctx.latest_depth
    K = ctx.K_rgb if ctx.K_rgb is not None else ctx.K_depth
    if img is None or depth is None or K is None:
        raise RuntimeError('no image or depth or intrinsics')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ranges = {'red': ((0, 100, 50), (10, 255, 255)), 'green': ((50, 100, 50), (70, 255, 255)), 'blue': ((100, 100, 50), (130, 255, 255))}
    if color_name not in ranges:
        raise RuntimeError(f"unknown color '{color_name}'")
    (lower, upper) = ranges[color_name]
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    (ys, xs) = np.nonzero(mask)
    if xs.size == 0:
        return None
    idx = xs.size // 2
    (u, v) = (int(xs[idx]), int(ys[idx]))
    z = float(depth[v, u])
    if z <= 0.001:
        zs = depth[ys, xs]
        zs = zs[zs > 0.001]
        if zs.size == 0:
            return None
        z = float(np.median(zs))
    (fx, fy) = (float(K[0, 0]), float(K[1, 1]))
    (cx, cy) = (float(K[0, 2]), float(K[1, 2]))
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return (x, y, z)

def _move_cartesian_pose(ctx, pose_stamped, max_step: float = 0.01, jump_threshold: float = 0.0, use_link_name: bool = False):
    """
    Compute a Cartesian path to the given PoseStamped and execute it via MoveIt.
    This avoids the FollowJointTrajectory action handshake from your node.

    Args:
        ctx: runtime context
        pose_stamped (geometry_msgs/PoseStamped): target pose in its header.frame_id
        max_step (float): eef step in meters
        jump_threshold (float): jump threshold (0.0 disables check)
        use_link_name (bool): if True, force req.link_name = ctx.EEF_LINK; otherwise let MoveIt pick the group's EE link
    """
    ctx.node.get_logger().info(
        f"[cartesian] HEELPEPEOEOE"
    )
    if ctx.cancelled():
        raise RuntimeError("cancelled")

    # ---- Build request
    req = GetCartesianPath.Request()
    req.group_name = ctx.ARM_GROUP
    if use_link_name:
        # Only enable if you're sure ctx.EEF_LINK matches the group's end-effector in SRDF
        req.link_name = ctx.EEF_LINK
    req.max_step = float(max_step)
    req.jump_threshold = float(jump_threshold)
    req.waypoints.append(pose_stamped.pose)

    # Provide start state to silence "Found empty JointState message"
    if ctx.joint_state is not None:
        rs = RobotStateMsg()
        rs.joint_state = ctx.joint_state
        req.start_state = rs

    # ---- Call service
    if not ctx.cart_cli.service_is_ready():
        ctx.cart_cli.wait_for_service(timeout_sec=5.0)

    fut = ctx.cart_cli.call_async(req)
    rclpy.spin_until_future_complete(ctx.node, fut)
    res = fut.result()

    if res is None or (not res.solution) or (not res.solution.joint_trajectory.points):
        raise RuntimeError("compute_cartesian_path returned empty solution")

    jt = res.solution.joint_trajectory
    ctx.node.get_logger().info(
        f"[cartesian] computed path: {len(jt.points)} points for joints {jt.joint_names}"
    )

    # ---- Execute via MoveIt (TrajectoryExecutionManager)
    ok = ctx.robot.execute(ctx.ARM_GROUP, res.solution)
    if not ok:
        raise RuntimeError("MoveIt execute() returned False")

    # (optional) small sync to let state propagate
    rclpy.spin_once(ctx.node, timeout_sec=0.01)



def _filtered_start_state(ctx) -> RobotStateMsg:
    """
    Build a RobotState that only contains variables present in the loaded RobotModel.
    This avoids crashes like:
      Variable 'robotiq_85_left_inner_knuckle_joint_mimic' is not known to model 'ur'
    """
    rs = RobotStateMsg()
    if ctx.joint_state is None:
        return rs

    model = ctx.robot.get_robot_model()
    valid = set(model.get_variable_names())  # all variables MoveIt knows for this robot

    js_in = ctx.joint_state
    js_out = JointStateMsg()
    for name, pos in zip(js_in.name, js_in.position):
        if name in valid:
            js_out.name.append(name)
            js_out.position.append(float(pos))
    # only assign if we kept anything
    if js_out.name:
        rs.joint_state = js_out
    return rs

def _move_to_pose_via_moveit(ctx, pose_stamped, max_step: float = 0.01, jump_threshold: float = 0.0):
    """
    1) Ask MoveIt for a Cartesian path to pose_stamped (no start_state, no link_name).
    2) Execute the returned RobotTrajectory via MoveIt (TrajectoryExecutionManager).
    3) If Cartesian returns empty, fall back to standard pose-goal plan+execute.
    """
    if ctx.cancelled():
        raise RuntimeError("cancelled")
    ctx.node.get_logger().info(f"[move_to_pose] start------------------")

    # ---- Cartesian request (mirror your working cart_runner: no link_name, no start_state) ----
    req = GetCartesianPath.Request()
    req.group_name = ctx.ARM_GROUP
    req.max_step = float(max_step)
    req.jump_threshold = float(jump_threshold)
    req.waypoints.append(pose_stamped.pose)

    if not ctx.cart_cli.service_is_ready():
        ctx.cart_cli.wait_for_service(timeout_sec=5.0)

    fut = ctx.cart_cli.call_async(req)
    rclpy.spin_until_future_complete(ctx.node, fut)
    res = fut.result()

    if res and res.solution and res.solution.joint_trajectory.points:
        jt = res.solution.joint_trajectory
        ctx.node.get_logger().info(f"[move_to_pose] Cartesian OK: {len(jt.points)} points")
        ok = ctx.robot.execute(ctx.ARM_GROUP, res.solution)   # execute via MoveIt
        if not ok:
            raise RuntimeError("MoveIt execute() returned False after Cartesian path")
        rclpy.spin_once(ctx.node, timeout_sec=0.01)
        return

    # ---- Fallback: joint-space plan to the same pose ----
    ctx.node.get_logger().warn("[move_to_pose] Cartesian empty; falling back to plan+execute")
    ctx.arm.set_start_state_to_current_state()
    ctx.arm.set_goal_state(pose_stamped_msg=pose_stamped, pose_link=ctx.EEF_LINK)
    plan = ctx.arm.plan()
    if not plan:
        raise RuntimeError("fallback planning failed")
    ok = ctx.robot.execute(ctx.ARM_GROUP, plan.trajectory)
    if not ok:
        raise RuntimeError("MoveIt execute() returned False for fallback plan")
    rclpy.spin_once(ctx.node, timeout_sec=0.01)


def move_to_pose_goal(ctx, pose_stamped: PoseStamped, pose_link: str = None):
    """
    Plan to a PoseStamped via MoveIt's planning pipeline and execute.
    Mirrors the path used by move_arm_from_home_to_up.
    """
    if ctx.cancelled():
        raise RuntimeError("cancelled")

    link = pose_link or ctx.EEF_LINK  # e.g., 'tool0'
    ctx.arm.set_start_state_to_current_state()
    ctx.arm.set_goal_state(pose_stamped_msg=pose_stamped, pose_link=link)

    # plan + execute (same as your _plan_and_execute helper)
    plan = ctx.arm.plan()
    if not plan:
        raise RuntimeError("pose-goal planning failed")
    ok = ctx.robot.execute(ctx.ARM_GROUP, plan.trajectory)
    if not ok:
        raise RuntimeError("execute() returned False")



def _spin_wait(node, fut, step=0.05):
    while not fut.done():
        rclpy.spin_once(node, timeout_sec=step)


def _wait_future(node, fut, label, timeout=10.0):
    start = time.time()
    while not fut.done():
        rclpy.spin_once(node, timeout_sec=0.05)
        if time.time() - start > timeout:
            raise RuntimeError(f"timeout waiting for {label}")

def _wait_until(cond_fn, node, label, timeout=10.0):
    start = time.time()
    while not cond_fn():
        rclpy.spin_once(node, timeout_sec=0.05)
        if time.time() - start > timeout:
            raise RuntimeError(f"timeout waiting for {label}")

def move_cartesian_via_controller(ctx, pose_stamped, max_step: float = 0.01, jump_threshold: float = 0.0):
    """
    Compute a Cartesian path and send it directly to /joint_trajectory_controller/follow_joint_trajectory
    without blocking the executor thread. This mirrors your working cart_runner.
    """
    if ctx.cancelled():
        raise RuntimeError("cancelled")

    log = ctx.node.get_logger().info
    log("[cart->ctl] enter")

    # 1) Build request (NO start_state, NO link_name)
    req = GetCartesianPath.Request()
    req.group_name = ctx.ARM_GROUP
    req.max_step = float(max_step)
    req.jump_threshold = float(jump_threshold)
    req.waypoints.append(pose_stamped.pose)

    # 2) Ensure service ready (non-blocking)
    log("[cart->ctl] waiting for /compute_cartesian_path")
    _wait_until(ctx.cart_cli.service_is_ready, ctx.node, "compute_cartesian_path service")

    # 3) Call and poll for result
    log("[cart->ctl] calling /compute_cartesian_path")
    cart_fut = ctx.cart_cli.call_async(req)
    _wait_future(ctx.node, cart_fut, "cartesian path response")
    res = cart_fut.result()
    if res is None or not res.solution or not res.solution.joint_trajectory.points:
        raise RuntimeError("compute_cartesian_path returned empty solution")

    jt = res.solution.joint_trajectory
    log(f"[cart->ctl] got trajectory: {len(jt.points)} points for joints {jt.joint_names}")

    # 4) Fresh action client (mirrors your working script)
    ac = ActionClient(ctx.node, FollowJointTrajectory,
                      '/joint_trajectory_controller/follow_joint_trajectory')

    log("[cart->ctl] waiting for controller action server")
    _wait_until(ac.server_is_ready, ctx.node, "follow_joint_trajectory action server")

    # 5) Send goal and wait for goal handle
    goal = FollowJointTrajectory.Goal()
    goal.trajectory = jt
    log("[cart->ctl] sending goal to controller")
    gh_fut = ac.send_goal_async(goal)
    _wait_future(ctx.node, gh_fut, "goal handle")
    gh = gh_fut.result()
    if gh is None or not gh.accepted:
        raise RuntimeError("trajectory goal rejected by controller")

    # 6) Wait for execution result
    log("[cart->ctl] goal accepted; waiting for result")
    res_fut = gh.get_result_async()
    _wait_future(ctx.node, res_fut, "controller result")
    result = res_fut.result().result
    code = getattr(result, "error_code", 0)
    err  = getattr(result, "error_string", "")
    if code != 0:
        raise RuntimeError(f"controller reported failure (code {code}): {err}")

    log("[cart->ctl] execution success")

