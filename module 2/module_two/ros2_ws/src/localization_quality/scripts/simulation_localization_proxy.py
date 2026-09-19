#!/usr/bin/env python3
"""Publish deterministic visual/lidar localization estimates from Gazebo truth.

This is a controlled sensor-specific localization emulator for programmatic
algorithm development. Run metadata identifies every output as a
controlled_ground_truth_noise_emulator artifact.
"""

import copy
import json
import math
import random

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class LocalizationProxy(Node):
    def __init__(self):
        super().__init__("simulation_localization_proxy")
        for name, default in (
            ("scene", "corridor"),
            ("lighting", "normal"),
            ("geometry_complexity", "low"),
            ("fault", "normal"),
            ("route", "route_a"),
        ):
            self.declare_parameter(name, default)
        self.declare_parameter("seed", 0)
        self.declare_parameter("speed_mps", 0.4)
        self.parameters = {
            name: self.get_parameter(name).value
            for name in (
                "scene", "lighting", "geometry_complexity", "fault",
                "route", "seed", "speed_mps",
            )
        }
        self.random_visual = random.Random(10000 + int(self.parameters["seed"]))
        self.random_laser = random.Random(20000 + int(self.parameters["seed"]))
        self.visual_publisher = self.create_publisher(
            Odometry, "/visual_localization/odom", 20)
        self.laser_publisher = self.create_publisher(
            Odometry, "/laser_localization/odom", 20)
        metadata_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.metadata_publisher = self.create_publisher(
            String, "/localization_quality/run_metadata", metadata_qos)
        self.create_subscription(
            Odometry, "/ground_truth/odom", self.on_ground_truth, 50)
        self.metadata_timer = self.create_timer(1.0, self.publish_metadata)
        self.visual_bias = [0.0, 0.0]
        self.laser_bias = [0.0, 0.0]

    def publish_metadata(self):
        message = String()
        message.data = json.dumps({
            **self.parameters,
            "localization_source": "controlled_ground_truth_noise_emulator",
        }, sort_keys=True)
        self.metadata_publisher.publish(message)

    def noise_scales(self):
        visual = {"dark": 0.13, "normal": 0.025, "strong": 0.08}[
            self.parameters["lighting"]]
        laser = {
            ("corridor", "low"): 0.20,
            ("corridor", "medium"): 0.14,
            ("corridor", "high"): 0.09,
        }.get((
            self.parameters["scene"], self.parameters["geometry_complexity"]),
            {"low": 0.10, "medium": 0.06, "high": 0.035}[
                self.parameters["geometry_complexity"]],
        )
        if self.parameters["fault"] == "visual_degraded":
            visual *= 7.0
        if self.parameters["fault"] == "laser_degraded":
            laser *= 7.0
        return visual, laser

    @staticmethod
    def perturb(message, generator, bias, sigma, frame):
        output = copy.deepcopy(message)
        output.header.frame_id = "world"
        output.child_frame_id = frame
        # A slowly varying bias creates realistic local correlation from
        # current and historical truth samples.
        bias[0] = 0.985 * bias[0] + generator.gauss(0.0, sigma * 0.12)
        bias[1] = 0.985 * bias[1] + generator.gauss(0.0, sigma * 0.12)
        output.pose.pose.position.x += bias[0] + generator.gauss(0.0, sigma)
        output.pose.pose.position.y += bias[1] + generator.gauss(0.0, sigma)
        yaw_noise = generator.gauss(0.0, sigma * 0.6)
        half = yaw_noise / 2.0
        q = output.pose.pose.orientation
        z = math.sin(half)
        w = math.cos(half)
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        q.x = w * qx - z * qy
        q.y = w * qy + z * qx
        q.z = w * qz + z * qw
        q.w = w * qw - z * qz
        variance = sigma * sigma
        output.pose.covariance = [0.0] * 36
        for index in (0, 7, 14):
            output.pose.covariance[index] = variance
        for index in (21, 28, 35):
            output.pose.covariance[index] = variance * 0.36
        return output

    def on_ground_truth(self, message):
        visual_sigma, laser_sigma = self.noise_scales()
        visual = self.perturb(
            message, self.random_visual, self.visual_bias,
            visual_sigma, "visual_base")
        laser = self.perturb(
            message, self.random_laser, self.laser_bias,
            laser_sigma, "laser_base")
        self.visual_publisher.publish(visual)
        self.laser_publisher.publish(laser)


def main():
    rclpy.init()
    node = LocalizationProxy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
