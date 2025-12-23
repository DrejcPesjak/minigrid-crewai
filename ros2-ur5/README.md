# ROS2 UR5 SPCA: LLM-Powered Robotic Manipulation

This module extends the Sense-Plan-Code-Act (SPCA) paradigm from grid-world simulation to **physical robotic manipulation** using a UR5 arm with a Robotiq gripper in Gazebo simulation.

---

## Conceptual Overview

### From Grid-World to Robot Arm

The core SPCA loop remains the same, but the execution substrate changes fundamentally:

| MiniGrid SPCA | ROS2 UR5 SPCA |
|---------------|---------------|
| 2D grid navigation | 6-DOF arm motion planning |
| Discrete primitive actions (0-6) | MoveIt2 trajectory execution |
| Symbolic grid state | RGB-D camera perception |
| Instant state transitions | Real-time physics simulation |
| Grid cell collision | Contact-based success detection |


---

## Key Concepts

### 1. Vision-Language Perception (SENSE)

Unlike grid-world where state is symbolic, the robot perceives the world through an RGB camera. A **Vision-Language Model (VLM)** describes the scene in natural language:

- Camera captures workspace image
- VLM receives image + task description
- Outputs structured scene description (object positions, colors, gripper state)

This bridges the gap between raw pixels and symbolic planning.

### 2. Classical Planning with Physical Grounding (PLAN)

The planner generates PDDL that maps to physical robot capabilities:

- **Objects**: `cube_red`, `cube_blue`, `plate_green`, `gripper`
- **Predicates**: `holding(obj)`, `on_plate(cube, plate)`, `touching(a, b)`
- **Actions**: `move_to(target)`, `pick_up(cube)`, `place_on(cube, surface)`

The planner reasons about:
- Gripper state (open/closed, holding object)
- Spatial relationships (which cube is left/middle/right)
- Stacking constraints (must place on stable surface)

### 3. Code Generation for Robot Control (CODE)

The Coder LLM generates Python functions that use MoveIt2 for motion:

```python
def pick_up_cube(ctx, cube_name, done_callback=None):
    # 1. Get cube position from scene description
    # 2. Move gripper above cube
    # 3. Lower to grasp height
    # 4. Close gripper
    # 5. Lift object
```

Each action uses:
- **Inverse Kinematics (IK)**: Convert Cartesian pose to joint angles
- **Motion Planning**: Collision-free trajectories via MoveIt2
- **Async Callbacks**: Non-blocking execution with completion notification

### 4. Physics-Based Execution (ACT)

The Executor node:
- Dynamically loads generated Python actions from disk
- Parses the plan string into action calls
- Executes each action via MoveIt2
- Reports success/failure back to Supervisor

### 5. Contact-Based Success Validation

The **Referee node** monitors Gazebo physics contacts to determine task completion:

```yaml
# Level definition
success:
  collisions_true:
    - [cube_red, plate_red]    # cube must touch plate
  collisions_false:
    - [cube_red, table]        # cube must NOT touch table
fail:
  collisions_true:
    - [gripper, ground_plane]  # gripper touching ground = failure
```

This enables objective, physics-grounded evaluation without relying on pose estimation.

---

## Task Curriculum

The curriculum progresses through increasingly complex manipulation skills:

### Touch (Basic Motion Control)
Move gripper to touch specific cubes. Teaches spatial reasoning and motion planning.

### Pickup (Grasping)
Locate, approach, and grasp objects. Requires gripper coordination.

### Pick-and-Place (Transport)
Pick up objects and place them on designated surfaces. Combines grasping with placement.

### Stack (Precision Manipulation)
Build towers by stacking cubes. Requires precise placement and stability awareness.

Each level specifies:
- **Title/Description**: Natural language task
- **Success Conditions**: Required collision pairs
- **Failure Conditions**: Forbidden collision pairs
- **Time Limit**: Maximum execution time

---

## ROS2 Node Architecture

### Supervisor (`supervisor_node.py`)
- **Role**: Orchestrates the SPCA loop
- **Manages**: Curriculum progression, PDDL caching, retry logic
- **Calls**: PlannerLLM, CoderLLM, SenseLLM
- **Controls**: Gazebo reset, level dispatching

### Executor (`executor_node.py`)
- **Role**: Executes action plans
- **Loads**: Agent action Python module from disk (hot-reload)
- **Uses**: MoveIt2 via `Ctx` runtime context
- **Reports**: Execution status to Supervisor

### Referee (`referee_node.py`)
- **Role**: Judges task success/failure
- **Monitors**: Gazebo physics contact stream (`gz topic`)
- **Publishes**: Task status (running/success/fail/timeout)

### Context Runtime (`ctx_runtime.py`)
- **Role**: Provides robot control primitives
- **Wraps**: MoveIt2 planning, TF transforms, camera feeds
- **Exposes**: Arm/gripper planning components, action clients

---

## Directory Structure

```
ros2-ur5/
├── spca_llm_ur5/                    # Main ROS2 package
│   ├── config/
│   │   ├── curriculum.yaml          # Task group definitions
│   │   └── levels/                  # Individual level specs
│   │       ├── touch/
│   │       ├── pickup/
│   │       ├── pick_and_place/
│   │       └── stack/
│   ├── spca_llm_ur5/
│   │   ├── nodes/                   # ROS2 nodes
│   │   │   ├── supervisor_node.py
│   │   │   ├── executor_node.py
│   │   │   ├── referee_node.py
│   │   │   └── ctx_runtime.py       # MoveIt2 context
│   │   ├── llm/                     # LLM interfaces
│   │   │   ├── plannerLLM.py
│   │   │   ├── coderLLM.py
│   │   │   └── senseLLM.py
│   │   └── actions/                 # Generated action library
│   │       └── agent_actions.py
│   └── launch/
│       └── spca_bringup.launch.py
├── demo_test_pkg/                   # Standalone test nodes
├── ros-install_pretty2.md           # Full ROS2/Gazebo/MoveIt2 install guide
├── working_commands.sh              # Useful CLI commands reference
└── run_commds.txt                   # Quick-start commands
```

---

## Setup

### Prerequisites

Follow `ros-install_pretty2.md` for complete installation:
- ROS2 Humble
- Gazebo 11 (Classic)
- MoveIt2
- UR5 simulation packages
- Robotiq gripper packages

### Running the System

**Terminal 1: Launch Gazebo + UR5 + MoveIt2**
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch ur_yt_sim spawn_ur5_camera_gripper_moveit.launch.py \
    world:=$(ros2 pkg prefix ur_yt_sim)/share/ur_yt_sim/worlds/world_rgb_table_light.world
```

**Terminal 2: Launch SPCA System**
```bash
source /opt/ros/humble/setup.bash
source ~/humble_moveit_py_ws/install/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch spca_llm_ur5 spca_bringup.launch.py
```

---

## Differences from MiniGrid SPCA

| Aspect | MiniGrid | ROS2 UR5 |
|--------|----------|----------|
| State representation | Symbolic grid | Camera image → VLM description |
| Action primitives | Discrete codes (0-6) | MoveIt2 trajectories |
| Success detection | Grid state comparison | Physics contact monitoring |
| Environment reset | Instant | Gazebo service call |
| Retry mechanism | Checkpoint replay | Full Gazebo reset |
| Code execution | In-process generator | ROS2 node hot-reload |

