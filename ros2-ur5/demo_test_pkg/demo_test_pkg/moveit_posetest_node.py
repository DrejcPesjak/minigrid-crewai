
# # All available imports from the moveit_py library
# from moveit.planning import (
#     LockedPlanningSceneContextManagerRO, 
#     LockedPlanningSceneContextManagerRW, 
#     MoveItPy,
#     PlanningComponent,
#     PlanRequestParameters,
#     PlanSolution,
#     PlanningSceneMonitor,
#     TrajectoryExecutionManager
# )
# from moveit.utils import create_params_file_from_dict, get_launch_params_filepaths
# from moveit.core.robot_model import JointModelGroup, RobotModel, VariableBounds
# from moveit.core.robot_state import RobotState, robotStateToRobotStateMsg
# from moveit.core.planning_scene import PlanningScene
# from moveit.core.robot_trajectory import RobotTrajectory
# from moveit.core.controller_manager import ExecutionStatus
# from moveit.core.planning_interface import MotionPlanResponse
# from moveit.core.collision_detection import AllowedCollisionMatrix, CollisionRequest, CollisionResult
# from moveit.core.kinematic_constraints import construct_constraints_from_node, construct_joint_constraint, construct_link_constraint


#!/usr/bin/env python3
"""
A script to outline the fundamentals of the moveit_py motion planning API for UR5 + Robotiq Gripper.
"""

import sys
import time
import os

# generic ros libraries
import rclpy
from rclpy.logging import get_logger

# moveit python library
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from moveit_configs_utils import MoveItConfigsBuilder
from moveit.utils import create_params_file_from_dict

# For PoseStamped
from geometry_msgs.msg import PoseStamped, Pose
from moveit.core.kinematic_constraints import construct_joint_constraint


def plan_and_execute(
    robot,
    planning_component,
    logger,
    single_plan_parameters=None,
    multi_plan_parameters=None,
    sleep_time=0.0,
):
    """Helper function to plan and execute a motion."""
    # plan to goal
    logger.info("Planning trajectory")
    if multi_plan_parameters is not None:
        plan_result = planning_component.plan(
            multi_plan_parameters=multi_plan_parameters
        )
    elif single_plan_parameters is not None:
        plan_result = planning_component.plan(
            single_plan_parameters=single_plan_parameters
        )
    else:
        plan_result = planning_component.plan()

    # execute the plan
    if plan_result:
        logger.info("Executing plan")
        robot_trajectory = plan_result.trajectory
        group_name = planning_component.planning_group_name
        robot.execute(group_name, robot_trajectory)
    else:
        logger.error("Planning failed")

    time.sleep(sleep_time)



# --- Use MoveItConfigsBuilder ---
moveit_config = (
    MoveItConfigsBuilder("custom_robot", package_name="ur5_camera_gripper_moveit_config")
    .robot_description(file_path="config/ur.urdf.xacro")
    .robot_description_semantic(file_path="config/ur.srdf")
    .trajectory_execution(file_path="config/moveit_controllers.yaml") # Contains moveit_controller_manager config
    .robot_description_kinematics(file_path="config/kinematics.yaml")
    .moveit_cpp(file_path="config/moveit_cpp.yaml") 
    .planning_scene_monitor(
        publish_robot_description= True, publish_robot_description_semantic=True, publish_planning_scene=True
    )
    .planning_pipelines(
        pipelines=["ompl", "chomp", "pilz_industrial_motion_planner"] # Matches your existing launch file
    )
    .to_moveit_configs()
)

def main():
    ###################################################################
    # MoveItPy Setup
    ###################################################################
    rclpy.init()
    logger = get_logger("moveit_py.ur5_goal")

    moveit_params_dict = moveit_config.to_dict()
    moveit_params_dict.update({'use_sim_time': True})
    # Disable the perception pipeline that’s spamming errors
    moveit_params_dict.pop('sensors', None)
    moveit_params_dict.pop('default_sensor', None)
    params_file = create_params_file_from_dict(moveit_params_dict, "/**")

    # instantiate MoveItPy instance and get planning component
    ur5_robot = MoveItPy(
        node_name="simple_cartesian_demo",
        launch_params_filepaths=[params_file]  
    ) 
    ur5_manipulator = ur5_robot.get_planning_component("ur5_manipulator")
    robotiq_gripper = ur5_robot.get_planning_component("robotiq_gripper") # Get gripper planning component
    logger.info("MoveItPy instance created for UR5 and Robotiq Gripper")

    ###################################################################
    # Wait for joint_states (and /clock if use_sim_time)
    ###################################################################
    tmp = rclpy.create_node("wait_for_state_tmp")
    psm = ur5_robot.get_planning_scene_monitor()
    psm.wait_for_current_robot_state(tmp.get_clock().now(), 2.0)
    # tmp.destroy_node()

    # tmp = rclpy.create_node("wait_for_state_tmp")
    # psm = ur5_robot.get_planning_scene_monitor()
    # deadline = tmp.get_clock().now() + rclpy.duration.Duration(seconds=2.0)
    # while tmp.get_clock().now() < deadline:
    #     rclpy.spin_once(tmp, timeout_sec=0.1)

    # # This returns False if it timed out — check it.
    # ok = psm.wait_for_current_robot_state(tmp.get_clock().now(), 5.0)
    # if not ok:
    #     logger.error("No fresh /joint_states. Abort.")
    #     # return


    # ###########################################################################
    # # Plan 5 - set goal state with PoseStamped message (UR5 arm)
    # ###########################################################################

    # logger.info("\n\nPlanning UR5 manipulator with PoseStamped goal")

    # current = ur5_manipulator.get_start_state()
    # T = current.get_global_link_transform("tool0") 
    # logger.info(f"Current tool0 pose: {T}")
    # T = current.get_pose("tool0")
    # logger.info(f"Current tool0 pose (from get_pose): {T}")
    # # Current tool0 pose (from get_pose): 
    # # geometry_msgs.msg.Pose(
    # #     position=geometry_msgs.msg.Point(x=0.11476029675701843, y=0.10921541128995416, z=1.5179897809328875), 
    # #     orientation=geometry_msgs.msg.Quaternion(x=0.6331310894798745, y=-0.6328790946623024, z=0.31487303309897896, w=-0.3153792131712247))


    # Tbase = current.get_global_link_transform("base_link")
    # logger.info(f"Current base_link pose: {Tbase}")
    # Tbase = current.get_pose("base_link")
    # logger.info(f"Current base_link pose (from get_pose): {Tbase}")

    # # set plan start state to current state
    # ur5_manipulator.set_start_state_to_current_state()

    # # set pose goal with PoseStamped message
    # pose_goal = PoseStamped()
    # pose_goal.header.frame_id = "base_link"
    # t = tmp.get_clock().now().to_msg()
    # t.sec += 10  # add 10 second to current time
    # pose_goal.header.stamp = t

    # # move to current tool0 pose
    # pose_goal.pose.position.x = T.position.x
    # pose_goal.pose.position.y = T.position.y
    # pose_goal.pose.position.z = T.position.z
    # pose_goal.pose.orientation.x = T.orientation.x
    # pose_goal.pose.orientation.y = T.orientation.y
    # pose_goal.pose.orientation.z = T.orientation.z
    # pose_goal.pose.orientation.w = T.orientation.w

    # # # same as cart_runner_node.py
    # # pose_goal.pose.orientation.w = 1.0
    # # pose_goal.pose.position.x = 0.0 
    # # pose_goal.pose.position.y = 0.5
    # # pose_goal.pose.position.z = 1.0

    # # # home pose example
    # # pose_goal.pose.position.x = 0.115
    # # pose_goal.pose.position.y = 0.109
    # # pose_goal.pose.position.z = 1.518
    # # pose_goal.pose.orientation.x = 0.633
    # # pose_goal.pose.orientation.y = -0.633
    # # pose_goal.pose.orientation.z = 0.315
    # # pose_goal.pose.orientation.w = -0.315
    # logger.info(f"Pose goal set to: {pose_goal}")
    
    # ur5_manipulator.set_goal_state(pose_stamped_msg=pose_goal, pose_link="tool0")

    # # plan to goal
    # plan_and_execute(ur5_robot, ur5_manipulator, logger, sleep_time=10.0)


    # ###########################################################################
    # # Plan 6 - set goal state with constraints (UR5 arm)
    # ###########################################################################


    logger.info("\n\nPlanning UR5 manipulator by computing IK for a specified Cartesian goal.")

    # --- 1. Define your target Cartesian pose ---

    # # Optionally, get current orientation to maintain it
    # current_state_for_orientation = ur5_manipulator.get_start_state()
    # current_tool0_pose = current_state_for_orientation.get_pose("tool0")
    
    # Create the target Pose message
    target_pose_tool0 = Pose()

    target_pose_tool0.position.x = 0.4
    target_pose_tool0.position.y = -0.3
    target_pose_tool0.position.z = 0.75

    # target_pose_tool0.position.x = 0.0  
    # target_pose_tool0.position.y = 0.5
    # target_pose_tool0.position.z = 1.0
    # target_pose_tool0.orientation.x = 0.0  # Identity orientation (no rotation)
    # target_pose_tool0.orientation.y = 0.0
    # target_pose_tool0.orientation.z = 0.0
    # target_pose_tool0.orientation.w = 1.0  # Identity orientation (no rotation

    # target_pose_tool0.position.x = 0.115
    # target_pose_tool0.position.y = 0.109
    # target_pose_tool0.position.z = 1.518
    # target_pose_tool0.orientation.x = 0.633
    # target_pose_tool0.orientation.y = -0.633
    # target_pose_tool0.orientation.z = 0.315
    # target_pose_tool0.orientation.w = -0.315
    
    # Use the current orientation from the robot's tool0 for the goal.
    # If you want a specific orientation, you can define it here.
    # Example: target_pose_tool0.orientation.w = 1.0 (identity/no rotation)
    #          target_pose_tool0.orientation = Quaternion(x=0.0, y=0.707, z=0.0, w=0.707) (90 deg rotation around Y)
    # target_pose_tool0.orientation = current_tool0_pose.orientation 

    logger.info(f"Attempting to reach Cartesian Pose (tool0 in base_link frame):\n"
                f"Position: ({target_pose_tool0.position.x}, {target_pose_tool0.position.y}, {target_pose_tool0.position.z})\n"
                f"Orientation: ({target_pose_tool0.orientation.x}, {target_pose_tool0.orientation.y}, {target_pose_tool0.orientation.z}, {target_pose_tool0.orientation.w})")

    # --- 2. Create a RobotState object to hold the IK solution ---
    robot_model = ur5_robot.get_robot_model()
    goal_robot_state = RobotState(robot_model) 
    
    # --- Set the seed state for IK ---
    # This is crucial for guiding the IK solver to a preferred solution
    # (e.g., closest to current position in joint space).
    # We use the current robot state as the seed.
    seed_state_for_ik = ur5_manipulator.get_start_state()
    goal_robot_state.set_joint_group_positions(
        ur5_manipulator.planning_group_name, 
        seed_state_for_ik.get_joint_group_positions(ur5_manipulator.planning_group_name)
    )
    logger.info(f"Seed state for IK:\n"
                f"Joint positions: {goal_robot_state.get_joint_group_positions(ur5_manipulator.planning_group_name)}\n")
                # f"Joint names: {goal_robot_state.get_joint_group_names()}")

    # --- 3. Call the IK solver ---
    # The set_from_ik method tries to find joint values for the given Cartesian pose.
    ik_found = goal_robot_state.set_from_ik(
        joint_model_group_name=ur5_manipulator.planning_group_name,
        geometry_pose=target_pose_tool0,
        tip_name="tool0",
        timeout=1.0, # Increased timeout for IK (from kinematics.yaml if not set here)
    )

    if not ik_found:
        logger.error(f"Could not find IK solution for target Cartesian pose: {target_pose_tool0}")
        logger.error("This means the desired Cartesian pose might be out of reach, in a singularity, or the IK solver timed out.")
        rclpy.shutdown()
        sys.exit(1)

    logger.info(f"IK solution found!\n"
                f"Joint positions: {goal_robot_state.get_joint_group_positions(ur5_manipulator.planning_group_name)}\n")
                # f"Joint names: {goal_robot_state.get_joint_group_names()}")
    # Optionally, print the IK solution's joint values:
    # logger.info(f"IK Joint Solution: {goal_robot_state.get_joint_group_positions(ur5_manipulator.planning_group_name)}")
    
    # --- 4. Construct Joint Constraints from the IK solution ---
    

    # import numpy as np
    # from math import pi

    # def wrap_to_interval(angle, low, high):
    #     # Map angle to [low, high] by adding/subtracting 2π
    #     width = 2*np.pi
    #     while angle > high:
    #         angle -= width
    #     while angle < low:
    #         angle += width
    #     # If still out (because bounds aren’t symmetric), clamp
    #     return min(max(angle, low), high)

    # def normalize_to_bounds(rs, jmg):
    #     vals = rs.get_joint_group_positions(jmg.name)
    #     bounds = jmg.active_joint_model_bounds  # list of VariableBounds
    #     for i, b in enumerate(bounds):
    #         low  = b[0].min_position
    #         high = b[0].max_position
    #         logger.info(f"Normalizing joint {jmg.active_joint_model_names[i]}: {vals[i]} to bounds ({low}, {high})")
    #         vals[i] = wrap_to_interval(vals[i], low, high)
    #     rs.set_joint_group_positions(jmg.name, vals)
    #     rs.update()

    def unwrap_to_seed(rs_goal, rs_seed, jmg_name):
        import numpy as np
        vals  = rs_goal.get_joint_group_positions(jmg_name)
        seed  = rs_seed.get_joint_group_positions(jmg_name)
        for i in range(len(vals)):
            # bring vals[i] as close as possible to seed[i] by removing 2π multiples
            delta = vals[i] - seed[i]
            vals[i] -= np.round(delta / (2.0 * np.pi)) * (2.0 * np.pi)
        rs_goal.set_joint_group_positions(jmg_name, vals)
        rs_goal.update()
        return vals


    vals_before = goal_robot_state.get_joint_group_positions(ur5_manipulator.planning_group_name)
    vals_after  = unwrap_to_seed(goal_robot_state, seed_state_for_ik, ur5_manipulator.planning_group_name)
    logger.info(f"Unwrapped IK solution to seed state:\n"
                f"Joint positions before: {vals_before}\n"
                f"Joint positions after: {vals_after}\n")

    logger.info(f"Normalized IK solution to joint bounds:\n"
                f"Joint positions: {goal_robot_state.get_joint_group_positions(ur5_manipulator.planning_group_name)}\n")
    
    # # Get the JointModelGroup object for the planning group.
    # joint_model_group_object = ur5_robot.get_robot_model().get_joint_model_group(ur5_manipulator.planning_group_name)
    # # Define the desired tolerance for all joints in the group.
    # # This small tolerance allows the planner some wiggle room around the exact IK solution.
    # joint_tolerance = 0.05 # 0.001 radians (approx 0.057 degrees)

    # # Use construct_joint_constraint to create the full Constraints message.
    # # This function uses the joint positions stored in goal_robot_state
    # # to generate the JointConstraint messages.
    # full_constraints = construct_joint_constraint(
    #     robot_state=goal_robot_state,
    #     joint_model_group=joint_model_group_object,
    #     tolerance=joint_tolerance
    # )

    # --- 5. Set the start state to current and plan ---
    ur5_manipulator.set_start_state_to_current_state()
    logger.info("Setting goal state with explicit Joint Constraints from IK solution...")
    
    # # set_goal_state expects a list of moveit_msgs.msg.Constraints.
    # ur5_manipulator.set_goal_state(motion_plan_constraints=[full_constraints])

    ur5_manipulator.set_goal_state(robot_state=goal_robot_state)

    plan_and_execute(ur5_robot, ur5_manipulator, logger, sleep_time=5.0)



    # ###########################################################################

    # logger.info("\n\nPlanning UR5 manipulator by computing IK and setting Joint Constraints")

    # # 1. Define your target Cartesian pose (e.g., current tool0 pose)
    # start_state_for_ik = ur5_manipulator.get_start_state() # Get current state to use as a seed for IK
    # target_pose_tool0 = start_state_for_ik.get_pose("tool0") 

    # # For demonstration, let's slightly offset the target pose to force IK to find a solution
    # # You would typically set this to your actual desired goal XYZ.
    # target_pose_tool0.position.x += 0.05 # Move 5cm in X
    # # target_pose_tool0.position.y += 0.05
    # # target_pose_tool0.position.z += 0.05
    # # target_pose_tool0.orientation.x += 0.01 # Slightly change orientation (be careful, might make IK unsolvable)

    # logger.info(f"Target Cartesian Pose for IK: {target_pose_tool0}")

    # # 2. Create a RobotState object to hold the IK solution
    # robot_model = ur5_robot.get_robot_model()
    # goal_robot_state = RobotState(robot_model) # This will be populated with the IK solution
    
    # # Set the seed state for IK (important for finding "closest" solution if multiple exist)
    # goal_robot_state.set_joint_group_positions(ur5_manipulator.planning_group_name, start_state_for_ik.get_joint_group_positions(ur5_manipulator.planning_group_name))

    # # 3. Call the IK solver
    # # The set_from_ik method returns True on success and populates goal_robot_state
    # # with the joint values.
    # # Arguments: link_name, pose (geometry_msgs.msg.Pose), group_name, timeout, attempts, IK_solver_info
    # # The last argument is tricky in Python, often best to just rely on kinematics.yaml settings.
    # ik_found = goal_robot_state.set_from_ik(
    #     joint_model_group_name=ur5_manipulator.planning_group_name,
    #     geometry_pose=target_pose_tool0,
    #     tip_name="tool0",
    #     # Optionally, you can specify a timeout here, which overrides kinematics.yaml
    #     timeout=1.0, 
    #     # attempts=10, 
    # )

    # if not ik_found:
    #     logger.error(f"Could not find IK solution for target pose: {target_pose_tool0}")
    #     rclpy.shutdown()
    #     os._exit(1) # Exit if IK failed

    # logger.info("IK solution found!")
    # # You can inspect the joint values here:
    # # logger.info(f"IK Joint Solution: {goal_robot_state.get_joint_group_positions(ur5_manipulator.planning_group_name)}")
    # # ik_joint_positions = goal_robot_state.get_joint_group_positions(ur5_manipulator.planning_group_name)
    
    # # # Make sure to get the list of joint names in the correct order for the group
    # # # from the JointModelGroup object directly.
    # # joint_model_group = ur5_robot.get_robot_model().get_joint_model_group(ur5_manipulator.planning_group_name)
    # # active_joint_names = joint_model_group.active_joint_model_names # Correct way to get names
    # # logger.info(f"Active joint names: {active_joint_names}")


    # # joint_constraints_list = []
    
    # # # Iterate through the joint names and their corresponding values
    # # # Assuming ik_joint_positions is an iterable (e.g., list or tuple) matching the order of active_joint_names
    # # if len(active_joint_names) != len(ik_joint_positions):
    # #     logger.error("Mismatch between number of active joints and IK joint positions!")
    # #     rclpy.shutdown()
    # #     os._exit(1)

    # # for i, joint_name in enumerate(active_joint_names):
    # #     logger.info(f"Setting joint constraint for {joint_name} with value {ik_joint_positions[i]}")
    # #     joint_value = ik_joint_positions[i] # Access by index, assuming ordered
        
    # #     joint_tolerance = 0.001 
        
    # #     joint_constraint = construct_joint_constraint(
    # #         joint_name=joint_name,
    # #         value=joint_value,
    # #         tolerance=joint_tolerance,
    # #     )
    # #     joint_constraints_list.append(joint_constraint)

    # # from moveit_msgs.msg import Constraints
    # # full_constraints = Constraints()
    # # full_constraints.joint_constraints = joint_constraints_list

    # # ur5_manipulator.set_start_state_to_current_state()
    # # logger.info("Setting goal state with explicit Joint Constraints from IK solution...")
    # # ur5_manipulator.set_goal_state(motion_plan_constraints=[full_constraints])

    # # plan_and_execute(ur5_robot, ur5_manipulator, logger, sleep_time=5.0)

    # joint_model_group_object = ur5_robot.get_robot_model().get_joint_model_group(ur5_manipulator.planning_group_name)
    
    # # 2. Define the desired tolerance for all joints in the group
    # joint_tolerance = 0.001 # 0.001 radians

    # # 3. Use construct_joint_constraint with the IK solution (goal_robot_state)
    # # This function will create a Constraints message containing JointConstraints
    # # for all active joints in the specified group, based on the positions
    # # currently stored in goal_robot_state, with the given tolerance.
    # full_constraints = construct_joint_constraint(
    #     robot_state=goal_robot_state,
    #     joint_model_group=joint_model_group_object, # Pass the JointModelGroup object
    #     tolerance=joint_tolerance
    # )

    # # Note: full_constraints is now a moveit_msgs.msg.Constraints object,
    # # already populated with the necessary JointConstraint messages.
    # # No need to manually create `joint_constraints_list` or append.

    # ur5_manipulator.set_start_state_to_current_state()
    # logger.info("Setting goal state with explicit Joint Constraints from IK solution...")
    # # Pass the single Constraints object inside a list, as set_goal_state expects a list
    # ur5_manipulator.set_goal_state(motion_plan_constraints=[full_constraints])

    # plan_and_execute(ur5_robot, ur5_manipulator, logger, sleep_time=5.0)


    # #########################################################################

    # from moveit_msgs.msg import OrientationConstraint, PositionConstraint, Constraints
    # from geometry_msgs.msg import Quaternion, Point, Pose
    # from shape_msgs.msg import SolidPrimitive
    
    # current = ur5_manipulator.get_start_state()
    # target_pose = current.get_pose("tool0") # This is a geometry_msgs.msg.Pose

    # # Log the target pose (useful for debugging)
    # pose_goal_stamped_for_logging = PoseStamped()
    # pose_goal_stamped_for_logging.header.frame_id = "base_link"
    # pose_goal_stamped_for_logging.header.stamp = tmp.get_clock().now().to_msg()
    # pose_goal_stamped_for_logging.pose = target_pose
    # logger.info(f"Attempting to plan to current tool0 pose with constraints: {pose_goal_stamped_for_logging}")


    # ur5_manipulator.set_start_state_to_current_state()

    # # Create an Orientation Constraint
    # oc = OrientationConstraint()
    # oc.link_name = "tool0"
    # oc.header.frame_id = "base_link"
    # oc.orientation = target_pose.orientation # Use the current orientation
    # oc.absolute_x_axis_tolerance = 0.05 # Radians, e.g., ~3 degrees
    # oc.absolute_y_axis_tolerance = 0.05
    # oc.absolute_z_axis_tolerance = 0.05
    # oc.weight = 1.0 # Importance of this constraint

    # # Create a Position Constraint
    # pc = PositionConstraint()
    # pc.link_name = "tool0"
    # pc.header.frame_id = "base_link"
    
    # # Define a sphere around the target point for the position constraint
    # sphere = SolidPrimitive()
    # sphere.type = SolidPrimitive.SPHERE
    # sphere.dimensions = [0.01] # Radius of 1 cm (0.01 meters)
    # pc.constraint_region.primitives.append(sphere)
    
    # sphere_pose = Pose()
    # sphere_pose.position = target_pose.position # Center the sphere at the target
    # sphere_pose.orientation.w = 1.0 # Identity orientation for the sphere (relative to constraint frame)
    # pc.constraint_region.primitive_poses.append(sphere_pose)
    # pc.weight = 1.0

    # full_constraints = Constraints()
    # # full_constraints.header.frame_id = "base_link"
    # # full_constraints.header.stamp = tmp.get_clock().now().to_msg() # Stamp the overall constraints message
    # full_constraints.position_constraints.append(pc)
    # full_constraints.orientation_constraints.append(oc)

    # # set_goal_state expects a list of moveit_msgs::msg::Constraints
    # ur5_manipulator.set_goal_state(motion_plan_constraints=[full_constraints]) 

    
    # plan_and_execute(ur5_robot, ur5_manipulator, logger, sleep_time=5.0)


    # ##########################################################################

    # current = ur5_manipulator.get_start_state()
    # current_joint_values = current.joint_positions
    # logger.info(f"Current joint values: {current_joint_values}")

    # # Create a RobotState for the goal
    # robot_model = ur5_robot.get_robot_model()
    # joint_model_group = robot_model.get_joint_model_group("ur5_manipulator")

    # ur5_manipulator.set_start_state_to_current_state()

    # temp_robot_state = RobotState(robot_model)
    # temp_robot_state.joint_positions = current_joint_values # Assign the actual current joint values
    # logger.info(f"Temporary RobotState set with joint values: {temp_robot_state.joint_positions}")

    # joint_constraint = construct_joint_constraint(
    #     robot_state=temp_robot_state,
    #     joint_model_group=joint_model_group,
    #     tolerance=0.01, 
    # )
    # ur5_manipulator.set_goal_state(motion_plan_constraints=[joint_constraint])
    # plan_and_execute(ur5_robot, ur5_manipulator, logger, sleep_time=5.0)

    # ###########################################################################

    # logger.info("\n\nPlanning UR5 manipulator with joint constraints")
    # robot_model = ur5_robot.get_robot_model()
    # robot_state = RobotState(robot_model)

    # # set plan start state to current state
    # ur5_manipulator.set_start_state_to_current_state()

    # # set constraints message
    # # IMPORTANT: Ensure these joint names match your UR5's joints exactly.
    # # The 'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint' are standard for UR5.
    # joint_values = {
    #     "shoulder_pan_joint": -1.5,
    #     "shoulder_lift_joint": -1.0,
    #     "elbow_joint": 1.0,
    #     "wrist_1_joint": -0.5,
    #     "wrist_2_joint": 0.5,
    #     "wrist_3_joint": 0.0,
    # }
    # robot_state.joint_positions = joint_values
    # joint_constraint = construct_joint_constraint(
    #     robot_state=robot_state,
    #     joint_model_group=ur5_robot.get_robot_model().get_joint_model_group("ur5_manipulator"),
    # )
    # ur5_manipulator.set_goal_state(motion_plan_constraints=[joint_constraint])

    # # plan to goal
    # plan_and_execute(ur5_robot, ur5_manipulator, logger, sleep_time=30.0)

    tmp.destroy_node()
    rclpy.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main()