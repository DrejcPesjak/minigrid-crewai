#!/usr/bin/env python3

import time
import rclpy
from rclpy.logging import get_logger
from moveit.planning import MoveItPy

def main():
    print("7 Starting UR5 simple demo node...")

    rclpy.init()
    logger = get_logger("moveit_py.ur5_launch_demo")

    robot = MoveItPy(
        node_name="ur5_simple_demo_node", 
        # launch_params_filepaths=[params_file] # would need a tmp node to load params
    )

    arm = robot.get_planning_component("ur5_manipulator")

    # Refresh the robot state
    tmp = rclpy.create_node("wait_for_state_tmp")
    psm = robot.get_planning_scene_monitor()
    psm.wait_for_current_robot_state(tmp.get_clock().now(), 2.0)
    tmp.destroy_node()

    logger.info("--- Planning to 'home' ---")
    # arm.set_start_state_to_current_state()
    arm.set_start_state(configuration_name="up")
    arm.set_goal_state(configuration_name="home")
    logger.info("Planning...")
    plan_result = arm.plan()
    logger.info(f"Plan result: {plan_result}")
    if plan_result:
        logger.info("Executing...")
        robot.execute("ur5_manipulator", plan_result.trajectory)
    else:
        logger.info("Planning failed.")

    time.sleep(3.0)
    rclpy.shutdown()

