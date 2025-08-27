
# Make sure sim is running so switching works
ros2 service call /unpause_physics std_srvs/srv/Empty "{}"

# Deactivate motion controllers only (leave JSB active)
echo "--- Deactivating motion controllers ---"
ros2 service call /controller_manager/switch_controller controller_manager_msgs/srv/SwitchController \
"{deactivate_controllers: ['joint_trajectory_controller','gripper_position_controller'],
  strictness: 2, timeout: {sec: 5}}"

# Confirm both are 'inactive' (repeat until they are)
ros2 control list_controllers -v

# Now pause & reset time if you want a clean clock
echo "--- Pausing physics and resetting simulation ---"
ros2 service call /pause_physics std_srvs/srv/Empty "{}"
ros2 service call /reset_simulation std_srvs/srv/Empty "{}"

# Send a one-shot home trajectory (edit joint order/values for your UR5)
echo "--- Sending home trajectory to UR5 ---"
ros2 topic pub --once /joint_trajectory_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{
  joint_names: ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint'],
  points: [{positions: [0.0,-1.57,0.0,-1.57,0.0,0.0], time_from_start: {sec: 2}}]
}"

# Wait for the trajectory to finish
echo "--- Unpausing physics to resume simulation ---"
ros2 service call /unpause_physics std_srvs/srv/Empty "{}"

# Reactivate controllers
echo "--- Reactivating motion controllers ---"
ros2 service call /controller_manager/switch_controller controller_manager_msgs/srv/SwitchController \
"{activate_controllers: ['joint_trajectory_controller','gripper_position_controller'],
  strictness: 2, timeout: {sec: 5}}"


