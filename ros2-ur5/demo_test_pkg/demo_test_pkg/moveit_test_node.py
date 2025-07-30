#!/usr/bin/env python3
"""
Simplest MoveItPy script: Moves UR5 manipulator from 'up' to 'home' position.
Loads MoveIt configurations directly within the script using MoveItConfigsBuilder.
Assumes Gazebo, controllers, etc., are already running.
"""

import time
import rclpy
from rclpy.time import Time
from moveit.planning import MoveItPy

# Import for MoveItConfigsBuilder
from moveit_configs_utils import MoveItConfigsBuilder
from moveit.utils import create_params_file_from_dict
from ament_index_python.packages import get_package_share_directory
import os

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
    
    print("Starting MoveItPy UR5 Manipulator example...")
    # Get the consolidated dictionary of all MoveIt parameters
    moveit_params_dict = moveit_config.to_dict()

    # Explicitly add use_sim_time, as it's critical for simulation and won't be picked up
    # by MoveItPy automatically via a launch file's parameter loading in this 'ros2 run' scenario.
    moveit_params_dict.update({'use_sim_time' : True})

    params_file = create_params_file_from_dict(moveit_params_dict, "/**")
    print(params_file)

    rclpy.init()
    logger = rclpy.logging.get_logger("moveit_py.pose_goal")

    # Instantiate MoveItPy by passing the generated parameter dictionary
    ur5_robot = MoveItPy(
        node_name="simple_cartesian_demo2", # Your desired node name
        # config_dict=moveit_params_dict # Pass all configurations here
        launch_params_filepaths=[params_file]  # Use the generated params file
    )
    logger.info("MoveItPy instance created for UR5 manipulator.")

    # Get the planning component for the UR5 arm
    ur5_manipulator = ur5_robot.get_planning_component("ur5_manipulator")

    # Wait for joint_states (and /clock if use_sim_time)
    tmp = rclpy.create_node("wait_for_state_tmp")
    psm = ur5_robot.get_planning_scene_monitor()
    psm.wait_for_current_robot_state(tmp.get_clock().now(), 2.0)
    tmp.destroy_node()
    

    logger.info("\n--- Planning to 'home' position ---")
    # print(vars(ur5_manipulator.get_start_state()))
    def dump_robot_state(state, group_name=None, precision=4):
        """Return a dict of joint -> value for the whole robot_state or just a group."""
        # Fast path: if joint_positions is already a dict, just pretty-print it
        if hasattr(state, "joint_positions") and isinstance(state.joint_positions, dict):
            return {k: round(v, precision) for k, v in state.joint_positions.items()}

        # Otherwise, use the group methods that *do* exist
        if group_name is not None and hasattr(state, "get_joint_group_positions"):
            try:
                positions = state.get_joint_group_positions(group_name)
            except TypeError:
                # Some bindings require the actual JointModelGroup instance
                jmg = state.robot_model.get_joint_model_group(group_name)
                positions = state.get_joint_group_positions(jmg)

            # Try to get the joint names from the model
            names = None
            try:
                jmg = state.robot_model.get_joint_model_group(group_name)
                # One of these should exist in your binding – try them in order:
                for attr in ("get_active_joint_model_names", "get_variable_names", "get_joint_model_names"):
                    if hasattr(jmg, attr):
                        names = list(getattr(jmg, attr)())
                        break
            except AttributeError:
                pass

            if names is None:
                # last resort – print them indexed
                return {f"{group_name}[{i}]": round(v, precision) for i, v in enumerate(positions)}
            else:
                return {n: round(v, precision) for n, v in zip(names, positions)}

        # Really last resort: nothing better to do
        return {"<unavailable>": "cannot introspect this RobotState with current bindings"}
    
    start_state = ur5_manipulator.get_start_state()
    logger.info("Full state:" + str(dump_robot_state(start_state)))
    logger.info("UR5 group:" + str(dump_robot_state(start_state, "ur5_manipulator")))

    ur5_manipulator.set_start_state_to_current_state()
    # ur5_manipulator.set_start_state(configuration_name="up")
    logger.info("Setting goal state to 'home' position.")
    ur5_manipulator.set_goal_state(configuration_name="home")
    logger.info("Planning to 'home' position...")

    plan_result_home = ur5_manipulator.plan()
    logger.info("Plan result:")

    def pretty_print_plan(plan, group_name="ur5_manipulator"):
        from rosidl_runtime_py import message_to_yaml
        logger.info(f"success: {bool(plan)}")
        # error code
        ec = getattr(plan.error_code, "val", plan.error_code)
        logger.info(f"error_code: {ec}")
        # start state (this one *is* a ROS msg, so yaml works)
        try:
            logger.info("start_state:\n" + message_to_yaml(plan.start_state))
        except Exception as e:
            # logger.info("could not yaml-dump start_state:", e)
            logger.info(f"could not yaml-dump start_state: {e}")
        # trajectory: iterate waypoints if the API is exposed
        traj = plan.trajectory
        if traj is None:
            logger.info("trajectory: None")
            return
        # Common methods on RobotTrajectory in the py bindings
        for meth in ("get_waypoint_count", "getNumWaypoints"):
            if hasattr(traj, meth):
                get_count = getattr(traj, meth)
                break
        else:
            logger.info("Cannot introspect trajectory (no waypoint API exposed).")
            return
        n = get_count()
        logger.info(f"waypoints: {n}")
        get_wp = None
        for meth in ("get_waypoint", "getWaypoint"):
            if hasattr(traj, meth):
                get_wp = getattr(traj, meth)
                break
        if get_wp is None:
            logger.info("No get_waypoint method on trajectory – cannot dump waypoints.")
            return
        for i in range(n):
            st = get_wp(i)
            logger.info(f"wp {i}: {dump_robot_state(st, group_name)}")

    pretty_print_plan(plan_result_home, "ur5_manipulator")
    if plan_result_home:
        logger.info("Executing plan to 'home' position.")
        ur5_robot.execute("ur5_manipulator", plan_result_home.trajectory)#, controllers=[])
    else:
        logger.info("Planning to 'home' position failed.")
    time.sleep(3.0)

    rclpy.shutdown()


if __name__ == "__main__":
    main()

