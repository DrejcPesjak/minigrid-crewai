# 🛠️ UR5 + Robotiq 2F‑85 + RGB‑D camera

### ROS 2 Humble • Gazebo 11 • MoveIt 2

*last verified 20 July 2025*

---

## 0  Prerequisites

```bash
sudo apt update && sudo apt install -y \
    locales software-properties-common curl
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

Enable the **ROS 2 apt repo** (copy‑pasted from ros‑apt‑source):

```bash
ROS_APT_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
                   | grep -F tag_name | cut -d'"' -f4)
curl -L -o /tmp/ros2-apt.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_VERSION}/ros2-apt-source_${ROS_APT_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb"
sudo dpkg -i /tmp/ros2-apt.deb
```

---

## 1  Install ROS 2 Humble + simulation stack

```bash
sudo apt update
sudo apt upgrade -y

sudo apt install -y \
    ros-humble-desktop \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-controller-manager \
    python3-colcon-common-extensions \
    python3-rosdep2 python3-vcstool
```

```bash
sudo rosdep init        # first time only
rosdep update
```

---

## 2  Workspace layout

```bash
mkdir -p ~/ros2_ws/src
cd       ~/ros2_ws/src
```

### 2.1 Universal Robots packages

```bash
git clone https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver.git
cd Universal_Robots_ROS2_Driver && git checkout humble && cd ..
vcs import < Universal_Robots_ROS2_Driver/Universal_Robots_ROS2_Driver.humble.repos
git clone https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git
git clone -b humble https://github.com/UniversalRobots/Universal_Robots_ROS2_Gazebo_Simulation.git
```

### 2.2 Robotiq gripper (PickNik fork) — **trim unused dirs**

```bash
git clone -b humble https://github.com/PickNikRobotics/ros2_robotiq_gripper.git
# Delete driver & test packages that clash with upstream ros2_control
rm -rf ros2_robotiq_gripper/robotiq_driver/
rm -rf ros2_robotiq_gripper/robotiq_hardware_tests/
```

### 2.3 Combined demo package (UR + gripper + camera)

```bash
git clone https://github.com/LearnRoboticsWROS/ur_yt_sim.git
```

#### ✏️ Manual fixes in **ur\_yt\_sim**

| file                                | edit                                                                                                                                                                                                                                                                             |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `meshes/sensors/camera/kinetic.dae` | In `\<library_images>` change `init_from` to **`kinetic.png`** (instead of the old xtion image).                                                                                                                                                                                 |
| `package.xml`                       | Ensure export section looks like:<br>`xml<br><export>  <build_type>ament_cmake</build_type>  <gazebo_ros gazebo_model_path="/home/$USER/ros2_ws/install/ur_yt_sim/share/"/>  <gazebo_ros gazebo_model_path="/home/$USER/ros2_ws/install/robotiq_description/share/"/> </export>` |

*(Those two lines register both the camera and gripper meshes as Gazebo model paths.)*

---

## 3  Build

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
source install/setup.bash
```

---

## 4  Run‑time recipes

### 4.1 Plain UR5 sim (sanity check)

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch ur_simulation_gazebo ur_sim_control.launch.py \
        ur_type:=ur5 launch_rviz:=false
```

### 4.2 UR5 + 2F‑85 + camera

```bash
ros2 launch ur_yt_sim spawn_ur5_camera_gripper.launch.py
```

*Gazebo opens; arm, gripper and camera meshes are visible.*
The RViz window shows TFs and `/camera/points` point‑cloud.

### 4.3 MoveIt 2 interactive demo

```bash
LC_NUMERIC=en_US.UTF-8 ros2 launch ur_moveit_config ur_moveit.launch.py \
        ur_type:=ur5 sim:=true use_sim_time:=true launch_rviz:=true
```

* In RViz choose **Planning Group → `ur_manipulator`**.
* Drag the interactive marker, **Plan → Execute**.

---

## 5  Handy CLI snippets

```bash
# list controllers
ros2 control list_controllers

# echo RGB image stream
ros2 topic echo /camera/image_raw

# Gazebo model path check
ros2 pkg prefix ur_yt_sim
```

---

## 6  Clean rebuild (if things break)

```bash
cd ~/ros2_ws
rm -rf build install log
colcon build --symlink-install
source install/setup.bash
```
