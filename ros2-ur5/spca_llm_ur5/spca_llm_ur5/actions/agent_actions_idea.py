# <your_pkg>/actions/agent_actions.py
# Plain Python module imported by Executor. CoderLLM edits this file.
# Keep the API stable: every function is def action(ctx, ...): -> None (raise on failure).

from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration

def move_to_above(ctx, target_frame: str, z_offset_m: float = 0.12):
    """
    Move TCP above a frame using TF and a single pose goal.
    """
    tf = ctx.tfbuf.lookup_transform(
        'world', target_frame, ctx.node.get_clock().now(), timeout=Duration(seconds=1.0)
    )
    p = PoseStamped()
    p.header.frame_id = 'world'
    p.pose.position.x = tf.transform.translation.x
    p.pose.position.y = tf.transform.translation.y
    p.pose.position.z = tf.transform.translation.z + z_offset_m
    p.pose.orientation.w = 1.0  # keep tool z-down in your URDF if that's 'down'
    # plan+execute (short blocking call)
    ctx.arm.set_start_state_to_current_state()
    ctx.arm.set_goal_state(pose_stamped=p)
    plan = ctx.arm.plan()
    if not plan:
        raise RuntimeError("move_to_above: planning failed")
    if ctx.cancelled():
        raise RuntimeError("move_to_above: cancelled")
    ctx.robot.execute(ctx.arm.planning_group_name, plan.trajectory)

def open_gripper(ctx):
    """
    1) set named 'open' on gripper planning group
    """
    # ctx.gripper.set_start_state_to_current_state()
    # ctx.gripper.set_goal_state(configuration_name="open")
    # plan = ctx.gripper.plan()
    # if not plan: raise RuntimeError("open_gripper: planning failed")
    # if ctx.cancelled(): raise RuntimeError("open_gripper: cancelled")
    # ctx.robot.execute(ctx.gripper.planning_group_name, plan.trajectory)
    pass

def close_gripper(ctx):
    """
    1) set named 'close' on gripper group
    """
    pass

def pick_up(ctx, obj_name: str):
    """
    Pseudocode:
      1) move_to_above(ctx, obj_name, 0.12)
      2) compute a lower pose (z - 0.10), plan+execute
      3) close_gripper(ctx)
      4) lift up (z + 0.10), plan+execute
      5) consider using small cartesian path segments; check ctx.cancelled() between each
    """
    pass

def place_on(ctx, plate_name: str):
    """
    Pseudocode:
      1) move_to_above(ctx, plate_name, 0.12)
      2) descend 6-8 cm
      3) open_gripper(ctx)
      4) retract up
    """
    pass
