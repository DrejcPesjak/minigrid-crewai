
# info
ros2 node list
ros2 node info /node_name
ros2 topic list
ros2 topic info /topic_name
ros2 topic echo --once /topic_name
ros2 service list
ros2 service type /srv_name
ros2 action list
ros2 interface show control_msgs/action/GripperCommand

ros2 service list
ros2 topic echo /joint_states --once

ros2 control list_controllers
ros2 control list_hardware_components


gz topic -l
gz topic -e /gazebo/default/physics/contacts


# diagnostics
colcon build --packages-select demo_test_pkg --symlink-install
colcon list --paths | grep demo_test_pkg        # should show just that pkg
ros2 pkg executables demo_test_pkg        

colcon list --base-paths ~/ros2_ws
colcon list --base-paths ~/humble_moveit_py_ws

echo $AMENT_PREFIX_PATH | tr ':' '\n'


# editing

gz model -m cone_green -x 0 -y 0 -z 3.05


ros2 service call /spawn_entity gazebo_msgs/srv/SpawnEntity "{  
  name: 'cone_green',
  xml: \"$(sed -e 's/\"/\\\\\"/g' /home/drew99/School/MastersT/ros-ur5/models/cone_green/model.sdf)\",
  robot_namespace: '',
  initial_pose:
    { position: { x: 0.5, y: 0.0, z: 0.2 },
      orientation: { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
    },
  reference_frame: 'world'
}"


# 2) open gripper (8 cm gap, 40 N effort limit)
ros2 action send_goal \
  /gripper_position_controller/gripper_cmd \
  control_msgs/action/GripperCommand \
  "{ command: { position: 0.08, max_effort: 40.0 } }"

# 3) close gripper fully (0.79 rad ≈ 0 cm gap)
ros2 action send_goal \
  /gripper_position_controller/gripper_cmd \
  control_msgs/action/GripperCommand \
  "{ command: { position: 0.79, max_effort: 40.0 } }"


ros2 topic pub /joint_trajectory_controller/joint_trajectory \
  trajectory_msgs/JointTrajectory "{
    joint_names: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
                  wrist_1_joint,    wrist_2_joint,        wrist_3_joint],
    points: [{
      positions: [0, -1.2, 1.8, -1.0, 1.5, 0.0],
      time_from_start: {sec: 3}
    }]
  }" --once


# very, slow, if no obstacles just run last traj_snippet (no for loop)
ros2 service call /compute_cartesian_path moveit_msgs/srv/GetCartesianPath "{  
  group_name: 'ur5_manipulator',  
  waypoints: [
    {  
      position:    { x: 0.0, y: 0.5, z: 1.0 },
      orientation: { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
    }
  ],  
  max_step: 0.01,  
  jump_threshold: 0.0  
}" > full_resp.yaml 2>&1

python3 extract_traj.py full_resp.yaml

#ros2 topic pub /joint_trajectory_controller/joint_trajectory \
#  trajectory_msgs/msg/JointTrajectory "$(cat traj_snippets/point_030.yaml)" --once

for f in traj_snippets/*.yaml; do
  ros2 topic pub /joint_trajectory_controller/joint_trajectory \
    trajectory_msgs/msg/JointTrajectory "$(cat "$f")" --once
done

