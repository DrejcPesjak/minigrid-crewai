#!/usr/bin/env bash
set -euo pipefail

# 1. Pause physics so nothing fights you
ros2 service call /pause_physics std_srvs/srv/Empty "{}"

# 2. Stop every controller that is writing to the joints
ros2 service call /controller_manager/switch_controller \
  controller_manager_msgs/srv/SwitchController \
  "{deactivate_controllers: ['joint_trajectory_controller','gripper_position_controller'],
    strictness: 1}"



# 3. Kill the old model
gz model -m cobot -d     

# 4. Convert the xacro once
ros2 run xacro xacro \
  $(ros2 pkg prefix ur_yt_sim)/share/ur_yt_sim/urdf/ur5_camera_gripper.urdf.xacro \
  > /tmp/ur5_camera_gripper.urdf

# 5. Insert it back into the running server
gz model -f /tmp/ur5_camera_gripper.urdf -m cobot -x 0 -y 0 -z 0.8

sleep 1


# 6. UN-pause physics - the hardware interface now ticks
ros2 service call /unpause_physics std_srvs/srv/Empty "{}"

sleep 2

# 7. (re-)spawn the arm controller, already activated
ros2 run controller_manager spawner \
     joint_trajectory_controller \
     --controller-manager /controller_manager \
     --activate

# 8. same for the gripper
ros2 run controller_manager spawner \
     gripper_position_controller \
     --controller-manager /controller_manager \
     --activate

# 9. same for the joint state broadcaster
ros2 run controller_manager spawner joint_state_broadcaster \
     --controller-manager /controller_manager \
     --activate

#ros2 service call /controller_manager/switch_controller \
#  controller_manager_msgs/srv/SwitchController \
#  "{activate_controllers:['joint_trajectory_controller','gripper_position_controller', 'joint_state_broadcaster'],
#    strictness:2}"


# 10. Wait for the controllers to be active
sleep 2
ros2 control list_controllers
# should see 3 active controllers

# 11. Reset the simulation
#ros2 service call /reset_world std_srvs/srv/Empty "{}"
ros2 service call /reset_simulation std_srvs/srv/Empty "{}"

# 12. Wait for the joint states to be published
ros2 topic echo --once /joint_states > /dev/null 

sleep 5