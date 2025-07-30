#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2


class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.bridge = CvBridge()

        # You might need a custom QoS profile for depth cameras; default should work for most
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def image_callback(self, msg: Image):
        try:
            # Convert ROS Image to OpenCV image (BGR8)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return

        # Do something: Canny edge detection
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Stack original and edges side by side
        combined = cv2.hconcat([cv_image, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)])

        # Show in window
        cv2.imshow('RGB | Edges', combined)
        cv2.waitKey(1)  # 1 ms delay to allow window events

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor()
    node.get_logger().info('Image processor node started, listening to /camera/image_raw')
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
