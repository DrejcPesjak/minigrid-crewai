#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge, CvBridgeError
import numpy as np


class ClosestPoint3D(Node):
    def __init__(self):
        super().__init__('closest_point_3d')
        self.bridge = CvBridge()
        self.K = None  # camera intrinsics matrix

        # subs
        self.create_subscription(CameraInfo,
                                 '/camera/camera_info',
                                 self.info_cb,
                                 10)
        self.create_subscription(Image,
                                 '/camera/depth/image_raw',
                                 self.depth_cb,
                                 10)

        # publisher if you want to use the point elsewhere
        self.pub = self.create_publisher(PointStamped,
                                         'closest_point',
                                         10)

    def info_cb(self, msg: CameraInfo):
        # build 3x3 K
        self.K = np.array(msg.k).reshape(3, 3)
        # we only need it once
        self.get_logger().info(f'Got intrinsics K =\n{self.K}')
        self.destroy_subscription(self.info_cb)  # stop listening

    def depth_cb(self, msg: Image):
        if self.K is None:
            return  # no intrinsics yet

        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return

        depth = np.array(depth_image, dtype=np.float32)
        # if mm, convert to meters (optional)
        if depth.max() > 1000:
            depth *= 0.001

        # mask out zeros
        valid = depth > 0.001
        if not np.any(valid):
            self.get_logger().warn('No valid depth pixels!')
            return

        # find idx of the minimum depth
        idx = np.argmin(np.where(valid, depth, np.inf))
        v, u = np.unravel_index(idx, depth.shape)
        z = float(depth[v, u])

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        # back-project to camera frame
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        pt = PointStamped()
        pt.header = msg.header
        pt.point.x = x
        pt.point.y = y
        pt.point.z = z

        self.get_logger().info(f'Closest pt (u,v,z) = ({u},{v},{z:.3f} m) → (x,y,z) = ({x:.3f},{y:.3f},{z:.3f})')
        self.pub.publish(pt)


def main(args=None):
    rclpy.init(args=args)
    node = ClosestPoint3D()
    node.get_logger().info('Waiting for camera_info + depth frames...')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
