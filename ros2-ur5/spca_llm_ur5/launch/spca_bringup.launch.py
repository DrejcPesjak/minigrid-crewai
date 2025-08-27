from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from moveit.utils import create_params_file_from_dict
from launch.actions import SetEnvironmentVariable
import os


def generate_launch_description():
    # TODO: run ur_yt_sim / UR5 + gripper + Gazebo bringup before this

    pkg_share = get_package_share_directory('spca_llm_ur5')
    curriculum = os.path.join(pkg_share, 'config', 'curriculum.yaml')

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
    # moveit_params_dict.pop('sensors', None)
    # moveit_params_dict.pop('default_sensor', None)
    moveit_params_dict['octomap_frame'] = 'world'
    moveit_params_dict['octomap_resolution'] = 0.02 # 2 cm resolution

    params_file = create_params_file_from_dict(moveit_params_dict, "/**")


    return LaunchDescription([

        # # Set environment variables for logging - TF suppresses INFO/DEBUG/WARN logs
        # SetEnvironmentVariable(name='CONSOLE_BRIDGE_LOG_LEVEL', value='ERROR'),
        # SetEnvironmentVariable(name='RCUTILS_LOGGING_MIN_SEVERITY', value='WARN'),
    
        Node(
            package='spca_llm_ur5', 
            executable='referee_node', 
            name='referee',
            parameters=[{
                'level_yaml': '',  # Supervisor will set it via /referee/set_level
                'contacts_topic': '/gazebo/default/physics/contacts',
                'contact_stale_s': 0.3
            }]
        ),
        Node(
            package='spca_llm_ur5', 
            executable='executor_node', 
            name='executor',
            parameters=[params_file],
        ),
        Node(
            package='spca_llm_ur5', 
            executable='supervisor_node', 
            name='supervisor',
            parameters=[{'curriculum_yaml': curriculum}]
        ),
    ])
