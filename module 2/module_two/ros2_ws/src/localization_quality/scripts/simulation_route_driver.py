#!/usr/bin/env python3
"""Drive deterministic scene-specific routes using Gazebo ground truth feedback."""

import math

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node


ROUTES = {
    ("corridor", "route_a"): [(-16, 0), (16, 0), (-16, 0)],
    ("corridor", "route_b"): [(-16, -0.8), (0, 0.8), (16, -0.8), (-16, 0)],
    ("corridor", "route_c"): [(-15, 0.9), (15, -0.9), (-15, 0.9)],
    ("hall", "route_a"): [(-10, -10), (10, -10), (10, 10), (-10, 10)],
    ("hall", "route_b"): [(-10, 0), (0, 10), (10, 0), (0, -10), (-10, 0)],
    ("hall", "route_c"): [(-11, -7), (11, 7), (-11, 7), (11, -7)],
    ("outdoor", "route_a"): [(-12, -10), (12, -10), (12, 12), (-12, 12)],
    ("outdoor", "route_b"): [(-12, 0), (0, 12), (12, 0), (0, -12)],
    ("outdoor", "route_c"): [(-13, -8), (13, 8), (-13, 8), (13, -8)],
}


class RouteDriver(Node):
    def __init__(self):
        super().__init__("simulation_route_driver")
        self.declare_parameter("scene", "corridor")
        self.declare_parameter("route", "route_a")
        self.declare_parameter("speed_mps", 0.4)
        scene = self.get_parameter("scene").value
        route = self.get_parameter("route").value
        self.speed = float(self.get_parameter("speed_mps").value)
        self.waypoints = ROUTES[(scene, route)]
        self.index = 0
        self.pose = None
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(
            Odometry, "/ground_truth/odom", self.on_pose, 20)
        self.timer = self.create_timer(0.05, self.control)

    def on_pose(self, message):
        q = message.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            yaw,
        )

    @staticmethod
    def wrap(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def control(self):
        command = Twist()
        if self.pose is None:
            self.publisher.publish(command)
            return
        target = self.waypoints[self.index]
        dx = target[0] - self.pose[0]
        dy = target[1] - self.pose[1]
        distance = math.hypot(dx, dy)
        if distance < 0.45:
            self.index = (self.index + 1) % len(self.waypoints)
            target = self.waypoints[self.index]
            dx = target[0] - self.pose[0]
            dy = target[1] - self.pose[1]
            distance = math.hypot(dx, dy)
        heading_error = self.wrap(math.atan2(dy, dx) - self.pose[2])
        command.angular.z = max(-1.2, min(1.2, 2.0 * heading_error))
        command.linear.x = min(self.speed, distance) * max(
            0.0, 1.0 - abs(heading_error) / 1.4)
        self.publisher.publish(command)


def main():
    rclpy.init()
    node = RouteDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publisher.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
