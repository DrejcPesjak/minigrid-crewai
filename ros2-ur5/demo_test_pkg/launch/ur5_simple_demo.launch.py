# ros2_ws/src/demo_test_pkg/launch/ur5_simple_demo.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

from moveit_configs_utils import MoveItConfigsBuilder
from moveit.utils import create_params_file_from_dict


def generate_launch_description():
    ld = LaunchDescription()

    # Build full MoveIt config
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

    moveit_params_dict = moveit_config.to_dict()
    moveit_params_dict.update({'use_sim_time': True})
    moveit_params_dict.pop('sensors', None)
    moveit_params_dict.pop('default_sensor', None)

    params_file = create_params_file_from_dict(moveit_params_dict, "/**")


    # Launch the simple Python demo node (will pick up params via params-file)
    ur5_simple_demo_node = Node(
        package="demo_test_pkg",
        executable="ur5_simple_demo_node",
        name="ur5_simple_demo_node",
        output="screen",
        # parameters=[{'params_file': params_file}],
        parameters=[params_file],
        arguments=["--ros-args", "--log-level", "info"],
    )

    
    ld.add_action(ur5_simple_demo_node)
    return ld
