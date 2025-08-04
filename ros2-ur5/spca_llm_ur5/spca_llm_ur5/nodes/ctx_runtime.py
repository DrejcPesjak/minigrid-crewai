# ctx_runtime.py
import threading, numpy as np
import rclpy
from rclpy.node import Node
import tf2_ros

from moveit.planning import MoveItPy

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, JointState
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2 as pc2

from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient
from moveit_msgs.srv import GetCartesianPath

class Ctx:
    """Runtime context for actions. Static topics; no launch parameters."""
    # Static config (adapt here if your setup differs)
    WORLD_FRAME   = 'world'
    EEF_LINK      = 'tool0'
    ARM_GROUP     = 'ur5_manipulator'
    GRIPPER_GROUP = 'robotiq_gripper'

    RGB_TOPIC        = '/camera/image_raw'
    DEPTH_TOPIC      = '/camera/depth/image_raw'
    RGB_INFO_TOPIC   = '/camera/camera_info'
    DEPTH_INFO_TOPIC = '/camera/depth/camera_info'
    CLOUD_TOPIC      = '/camera/points'
    JOINT_STATES     = '/joint_states'

    FOLLOW_TRAJ_ACTION = '/joint_trajectory_controller/follow_joint_trajectory'
    GRIPPER_CMD_ACTION = '/gripper_position_controller/gripper_cmd'
    CARTESIAN_SRV      = '/compute_cartesian_path'

    def __init__(self, node: Node):
        self.node = node
        self._cancel = threading.Event()

        # MoveItPy (reads moveit params passed via launch to this node)
        self.robot   = MoveItPy(node_name='llm_actions_moveit')
        self.arm     = self.robot.get_planning_component(self.ARM_GROUP)
        self.gripper = self.robot.get_planning_component(self.GRIPPER_GROUP)


        self.psm = self.robot.get_planning_scene_monitor()
        try:
            self.tem = self.robot.get_trajectory_execution_manager()
        except Exception:
            self.tem = None

        self.log = node.get_logger()
        self.planning_frame = self.robot.get_robot_model().model_frame

        self.latest_rgb_header = None
        self.latest_depth_header = None


        # TF
        self.tfbuf = tf2_ros.Buffer()
        self.tfl   = tf2_ros.TransformListener(self.tfbuf, node, spin_thread=False)

        # Perception caches
        self.bridge = CvBridge()
        self.latest_rgb         = None   # np.uint8 HxWx3 (BGR)
        self.latest_depth       = None   # np.float32 HxW (meters)
        self.K_rgb              = None   # 3x3
        self.K_depth            = None   # 3x3
        self.rgb_frame          = None
        self.depth_frame        = None
        self.latest_cloud       = None   # list[(x,y,z)] or None

        node.create_subscription(Image,      self.RGB_TOPIC,        self._rgb_cb,   10)
        node.create_subscription(Image,      self.DEPTH_TOPIC,      self._depth_cb, 10)
        node.create_subscription(CameraInfo, self.RGB_INFO_TOPIC,   self._rgb_info, 10)
        node.create_subscription(CameraInfo, self.DEPTH_INFO_TOPIC, self._depth_info, 10)
        node.create_subscription(PointCloud2,self.CLOUD_TOPIC,      self._cloud_cb, 10)

        # Joint states cache
        self.joint_state = None
        node.create_subscription(JointState, self.JOINT_STATES, self._js_cb, 50)

        # Service / actions (exposed as attributes, no wrappers)
        self.cart_cli      = node.create_client(GetCartesianPath, self.CARTESIAN_SRV)
        self.follow_traj_ac= ActionClient(node, FollowJointTrajectory, self.FOLLOW_TRAJ_ACTION)
        self.gripper_ac    = ActionClient(node, GripperCommand,        self.GRIPPER_CMD_ACTION)

    # ---- cancel ----
    def cancel(self): self._cancel.set()
    def cancelled(self): return self._cancel.is_set()
    def clear_cancel(self): self._cancel.clear()

    # ---- callbacks ----
    def _rgb_info(self, msg: CameraInfo):
        self.K_rgb = np.array(msg.k, dtype=np.float32).reshape(3,3)
        self.rgb_frame = msg.header.frame_id

    def _depth_info(self, msg: CameraInfo):
        self.K_depth = np.array(msg.k, dtype=np.float32).reshape(3,3)
        self.depth_frame = msg.header.frame_id
    
    def _rgb_cb(self, msg: Image):
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.latest_rgb_header = msg.header

    def _depth_cb(self, msg: Image):
        d = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        arr = np.asarray(d)
        if arr.dtype != np.float32 and arr.max() > 1000:
            arr = arr.astype(np.float32) * 0.001
        self.latest_depth = arr
        self.latest_depth_header = msg.header

    def _cloud_cb(self, msg: PointCloud2):
        try:
            pts = [(x, y, z) for (x, y, z) in pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True)]
            self.latest_cloud = pts
        except Exception:
            self.latest_cloud = None

    def _js_cb(self, msg: JointState):
        self.joint_state = msg
    
    # ---- helpers ----
    def wait_for_current_state(self, timeout_s=2.0):
        now = self.node.get_clock().now()
        return self.psm.wait_for_current_robot_state(now, float(timeout_s))

    def wait_for_interfaces(self):
        self.cart_cli.wait_for_service()
        self.follow_traj_ac.wait_for_server()
        self.gripper_ac.wait_for_server()

    def stop_motion(self):
        if self.tem:
            self.tem.stop_execution() 
