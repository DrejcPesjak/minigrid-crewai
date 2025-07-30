#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np


class RGBDProcessor(Node):
    def __init__(self):
        super().__init__('rgbd_processor')
        self.bridge = CvBridge()
        self.create_subscription(Image, '/camera/image_raw',    self.rgb_callback,   10)
        self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)

        # placeholders for latest frames
        self.latest_rgb = None
        self.latest_edges = None
        self.latest_depth_vis = None

    def rgb_callback(self, msg: Image):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'RGB CvBridge error: {e}')
            return

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        self.latest_rgb = rgb
        self.latest_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        self.display_if_ready()

    def depth_callback(self, msg: Image):
        # debug: show encoding
        self.get_logger().info(f'Depth msg encoding: {msg.encoding}')
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'Depth CvBridge error: {e}')
            return

        # # convert mm->m float32 if needed, here just normalize for display
        # depth = np.array(depth_raw, dtype=np.float32)
        # # clip to 0–4m, scale to 0–255
        # depth = np.clip(depth / 4000.0, 0.0, 1.0)
        depth = np.array(depth_raw, dtype=np.float32)
        # compute stats
        valid = depth > 0
        if np.any(valid):
            dmin, dmax = float(depth[valid].min()), float(depth[valid].max())
        else:
            dmin, dmax = 0.0, 0.0
        self.get_logger().info(f'Depth min/max (valid): {dmin:.3f} / {dmax:.3f}')

        # choose range based on observed data
        # if encoding is 16UC1: values are mm → convert to meters
        if msg.encoding == '16UC1' and dmax > 1000:
            depth = depth * 0.001  # mm→m
            dmin /= 1000; dmax /= 1000

        # normalize to 0–1 between dmin/dmax
        if dmax > dmin:
            norm = (depth - dmin) / (dmax - dmin)
        else:
            norm = np.zeros_like(depth)
        norm = np.clip(norm, 0.0, 1.0)
        depth_u8 = (depth * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

        self.latest_depth_vis = depth_colormap
        self.display_if_ready()

    def display_if_ready(self):
        if self.latest_rgb is None or self.latest_depth_vis is None:
            return

        # stack RGB+edges and depth side-by-side
        left = cv2.hconcat([self.latest_rgb, self.latest_edges])
        right = self.latest_depth_vis
        combined = cv2.hconcat([left, right])

        cv2.imshow('RGB | Edges || Depth (JET)', combined)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = RGBDProcessor()
    node.get_logger().info('RGBD processor started')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
