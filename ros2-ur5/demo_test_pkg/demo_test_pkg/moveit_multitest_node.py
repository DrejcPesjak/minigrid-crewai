
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
from geometry_msgs.msg import PoseStamped
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
    tmp.destroy_node()

    ###########################################################################
    # Plan 1 - set states with predefined string (UR5 arm)
    ###########################################################################

    logger.info("\n\nPlanning UR5 manipulator from 'home' to 'up' state")

    # set plan start state using predefined state
    ur5_manipulator.set_start_state(configuration_name="home") # Using 'home' state from your SRDF

    # set pose goal using predefined state
    ur5_manipulator.set_goal_state(configuration_name="up") # Using 'up' state from your SRDF

    # plan to goal
    plan_and_execute(ur5_robot, ur5_manipulator, logger, sleep_time=30.0)

    ###########################################################################
    # Plan 2 - Gripper action: Open
    ###########################################################################
    logger.info("\n\nPlanning gripper to open state")
    robotiq_gripper.set_start_state_to_current_state()
    robotiq_gripper.set_goal_state(configuration_name="open") # Using 'open' state from your SRDF
    plan_and_execute(ur5_robot, robotiq_gripper, logger, sleep_time=30.0)

    # ###########################################################################
    # # Plan 3 - set goal state with RobotState object (UR5 arm)
    # ###########################################################################

    # logger.info("\n\nPlanning UR5 manipulator with random RobotState object")

    # # instantiate a RobotState instance using the current robot model
    robot_model = ur5_robot.get_robot_model()
    joint_model_group = robot_model.get_joint_model_group("ur5_manipulator")
    robot_state = RobotState(robot_model)

    # # randomize the robot state
    # robot_state.set_to_random_positions(joint_model_group)

    # def state_is_valid(psm, rs, group_name=None):
    #     # Humble’s bindings don’t expose everything nicely. The simplest portable
    #     # trick is to convert to a RobotState msg and ask the PlanningScene to check:
    #     from moveit.core.robot_state import robotStateToRobotStateMsg
    #     rs_msg = robotStateToRobotStateMsg(rs)
    #     # In Humble, PlanningSceneMonitor has `getPlanningScene()` in Python.
    #     ps = psm.get_planning_scene()
    #     # The Python binding usually exposes isStateValid(robot_state_msg, group, verbose)
    #     return ps.is_state_valid(rs_msg, group_name or "", False)

    # def sample_random_in_bounds(rs, jmg, logger, tries=10, margin=0.0):
    #     import numpy as np
    #     group_name = getattr(jmg, "name", "ur5_manipulator")
    #     for i in range(tries):
    #         rs.set_to_random_positions(jmg)
    #         rs.update()
    #         vals = rs.get_joint_group_positions(group_name)
    #         if not jmg.satisfies_position_bounds(vals, margin):
    #             continue
    #         if state_is_valid(psm, rs, group_name):
    #             return True
    #         logger.debug(f"Sample {i+1}/{tries} failed: {vals}")
    #     logger.error(f"Failed to sample a random state in {tries} tries")
    #     return False

    # def state_is_valid_via_service(node, rs, group_name=""):
    #     client = node.create_client(GetStateValidity, "/check_state_validity")
    #     if not client.wait_for_service(timeout_sec=2.0):
    #         raise RuntimeError("/check_state_validity not available")

    #     req = GetStateValidity.Request()
    #     req.robot_state = robotStateToRobotStateMsg(rs)
    #     req.group_name = group_name
    #     # you can also fill req.constraints if you want

    #     future = client.call_async(req)
    #     rclpy.spin_until_future_complete(node, future, timeout_sec=2.0)
    #     if not future.done() or future.result() is None:
    #         return False
    #     return future.result().valid
    
    # def sample_random_collision_free(rs, jmg, node, tries=50, margin=0.0):
    #     group_name = getattr(jmg, "name", "ur5_manipulator")
    #     for _ in range(tries):
    #         rs.set_to_random_positions(jmg)
    #         rs.update()
    #         vals = rs.get_joint_group_positions(group_name)
    #         if not jmg.satisfies_position_bounds(vals, margin):
    #             continue
    #         if state_is_valid_via_service(node, rs, group_name):
    #             return True
    #     return False
    
    # check_node = rclpy.create_node("check_state_validity_node")
    # def state_is_valid(rs, group_name):
    #     return state_is_valid_via_service(check_node, rs, group_name)

    # def sample_random_in_bounds(rs, jmg, logger, tries=10, margin=0.0):
    #     group_name = getattr(jmg, "name", "ur5_manipulator")
    #     for i in range(tries):
    #         rs.set_to_random_positions(jmg)
    #         rs.update()
    #         vals = rs.get_joint_group_positions(group_name)
    #         if not jmg.satisfies_position_bounds(vals, margin):
    #             continue
    #         if state_is_valid(rs, group_name):
    #             return True
    #         logger.debug(f"sample {i+1}/{tries} failed collision/bounds: {vals}")
    #     return False

    # if not sample_random_in_bounds(robot_state, joint_model_group, logger=logger):
    #     logger.error("Couldn't sample a bounded random state")
    # else:
    #     logger.info("Sampled a random state within bounds")

    #     # set plan start state to current state
    #     ur5_manipulator.set_start_state_to_current_state()

    #     # set goal state to the initialized robot state
    #     ur5_manipulator.set_goal_state(robot_state=robot_state)

    #     # plan to goal
    #     plan_and_execute(ur5_robot, ur5_manipulator, logger, sleep_time=30.0)
    
    # check_node.destroy_node()
    ###########################################################################
    # Plan 4 - Gripper action: Close
    ###########################################################################
    logger.info("\n\nPlanning gripper to close state")
    robotiq_gripper.set_start_state_to_current_state()
    robotiq_gripper.set_goal_state(configuration_name="close") # Using 'close' state from your SRDF
    plan_and_execute(ur5_robot, robotiq_gripper, logger, sleep_time=30.0)


    ###########################################################################
    # Plan 5 - set goal state with PoseStamped message (UR5 arm)
    ###########################################################################

    logger.info("\n\nPlanning UR5 manipulator with PoseStamped goal")

    # set plan start state to current state
    ur5_manipulator.set_start_state_to_current_state()

    # set pose goal with PoseStamped message
    pose_goal = PoseStamped()
    planning_frame = ur5_robot.get_robot_model().model_frame
    pose_goal.header.frame_id = planning_frame
    # pose_goal.header.frame_id = "base_link"
    pose_goal.pose.orientation.w = 1.0
    pose_goal.pose.position.x = 0.0 
    pose_goal.pose.position.y = 0.5
    pose_goal.pose.position.z = 0.1
    ur5_manipulator.set_goal_state(pose_stamped_msg=pose_goal, pose_link="tool0") # Changed from panda_link8 to tool0

    # plan to goal
    plan_and_execute(ur5_robot, ur5_manipulator, logger, sleep_time=30.0)

    ###########################################################################
    # Plan 6 - set goal state with constraints (UR5 arm)
    ###########################################################################

    logger.info("\n\nPlanning UR5 manipulator with joint constraints")

    # set plan start state to current state
    ur5_manipulator.set_start_state_to_current_state()

    # set constraints message
    # IMPORTANT: Ensure these joint names match your UR5's joints exactly.
    # The 'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint' are standard for UR5.
    joint_values = {
        "shoulder_pan_joint": -1.5,
        "shoulder_lift_joint": -1.0,
        "elbow_joint": 1.0,
        "wrist_1_joint": -0.5,
        "wrist_2_joint": 0.5,
        "wrist_3_joint": 0.0,
    }
    robot_state.joint_positions = joint_values
    joint_constraint = construct_joint_constraint(
        robot_state=robot_state,
        joint_model_group=ur5_robot.get_robot_model().get_joint_model_group("ur5_manipulator"),
    )
    ur5_manipulator.set_goal_state(motion_plan_constraints=[joint_constraint])

    # plan to goal
    plan_and_execute(ur5_robot, ur5_manipulator, logger, sleep_time=30.0)

    # ###########################################################################
    # # Plan 7 - Planning with Multiple Pipelines simultaneously (UR5 arm)
    # ###########################################################################

    # # set plan start state to current state
    # ur5_manipulator.set_start_state_to_current_state()

    # # set pose goal with PoseStamped message
    # ur5_manipulator.set_goal_state(configuration_name="zero") # Using 'zero' state from your SRDF

    # # initialise multi-pipeline plan request parameters
    # # Make sure 'ompl_rrtc', 'pilz_lin', 'chomp_planner' are actually configured
    # # in your OMPL/Kinematics/Pilz/CHOMP YAML files and loaded by MoveItPy.
    # # Otherwise, simplify to just 'ompl' or whatever you are using.
    # multi_pipeline_plan_request_params = MultiPipelinePlanRequestParameters(
    #     ur5_robot, ["ompl", "pilz_lin"] # Adjusted to simpler, more common planners for example
    # )

    # # plan to goal
    # plan_and_execute(
    #     ur5_robot,
    #     ur5_manipulator,
    #     logger,
    #     multi_plan_parameters=multi_pipeline_plan_request_params,
    #     sleep_time=3.0,
    # )

    rclpy.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main()