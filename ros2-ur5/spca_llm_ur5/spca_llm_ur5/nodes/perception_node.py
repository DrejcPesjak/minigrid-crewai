# #!/usr/bin/env python3
# import cv2
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
# from sensor_msgs.msg import Image, CompressedImage
# from cv_bridge import CvBridge


# class PerceptionNode(Node):
#     def __init__(self):
#         super().__init__('perception')
#         self.bridge = CvBridge()

#         self.sub = self.create_subscription(
#             Image, '/camera/image_raw', self._rgb_cb, 10
#         )

#         # Latched / transient local publisher for last JPEG frame
#         qos = QoSProfile(
#             depth=1,
#             reliability=ReliabilityPolicy.RELIABLE,
#             durability=DurabilityPolicy.TRANSIENT_LOCAL,
#             history=HistoryPolicy.KEEP_LAST,
#         )
#         self.pub = self.create_publisher(
#             CompressedImage, '/perception/last_image', qos
#         )

#     def _rgb_cb(self, msg: Image):
#         # Convert to BGR and JPEG-compress
#         bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#         ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
#         if not ok:
#             return
#         out = CompressedImage()
#         out.header = msg.header
#         out.format = 'jpeg'
#         out.data = np.asarray(buf).tobytes()
#         self.pub.publish(out)


# def main():
#     rclpy.init()
#     n = PerceptionNode()
#     rclpy.spin(n)
#     n.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()
