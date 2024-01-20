import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSubscriber(Node):
    def __init__(self, name):
        super().__init__(name)
        self.subscription = self.create_subscription(
            Image, 'image_raw', self.listener_callback, 10)
        self.subscription
        self.cv_bridge = CvBridge()
        self.publisher_modified = self.create_publisher(Image, 'image_modified', 10)

    def listener_callback(self, msg):
        image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        

        
        modified_msg = self.cv_bridge.cv2_to_imgmsg(gray_image, 'mono8')
        self.publisher_modified.publish(modified_msg)

        cv2.imshow('Image Subscriber', gray_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber("topic_webcam_sub")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
