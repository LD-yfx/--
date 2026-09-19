#!/usr/bin/env python3
"""Republish Gazebo sensors and inject configured visual/laser degradation."""

import copy
import math

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2, PointField


SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=20,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


class SensorPipeline(Node):
    def __init__(self):
        super().__init__("simulation_sensor_pipeline")
        self.declare_parameter("fault", "normal")
        self.declare_parameter("fault_severity", 1.0)
        self.fault = self.get_parameter("fault").value
        self.severity = float(self.get_parameter("fault_severity").value)
        if self.fault not in ("normal", "visual_degraded", "laser_degraded"):
            raise ValueError("invalid fault parameter")
        self.bridge = CvBridge()
        self.callback_group = ReentrantCallbackGroup()
        self.image_publisher = self.create_publisher(
            Image, "/camera/image_raw", SENSOR_QOS)
        self.info_publisher = self.create_publisher(
            CameraInfo, "/camera/camera_info", SENSOR_QOS)
        self.scan_publisher = self.create_publisher(
            LaserScan, "/scan", SENSOR_QOS)
        self.cloud_publisher = self.create_publisher(
            PointCloud2, "/points", SENSOR_QOS)
        self.create_subscription(
            Image, "/front_camera/image_raw", self.on_image,
            SENSOR_QOS, callback_group=self.callback_group)
        self.create_subscription(
            CameraInfo, "/front_camera/camera_info",
            self.info_publisher.publish,
            SENSOR_QOS, callback_group=self.callback_group)
        self.create_subscription(
            LaserScan, "/sim/scan", self.on_scan, SENSOR_QOS,
            callback_group=self.callback_group)
        self.create_subscription(
            PointCloud2, "/depth_camera/points", self.on_cloud,
            SENSOR_QOS, callback_group=self.callback_group)

    def on_image(self, message):
        if self.fault != "visual_degraded":
            self.image_publisher.publish(message)
            return
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            kernel = max(3, 1 + 2 * int(round(4 * self.severity)))
            degraded = cv2.GaussianBlur(image, (kernel, kernel), 0)
            degraded = cv2.convertScaleAbs(
                degraded, alpha=max(0.35, 1.0 - 0.45 * self.severity))
            output = self.bridge.cv2_to_imgmsg(degraded, encoding="bgr8")
            output.header = message.header
            self.image_publisher.publish(output)
        except Exception as error:  # cv_bridge can raise several exception types
            self.get_logger().error(f"image degradation failed: {error}")

    def on_scan(self, message):
        if self.fault != "laser_degraded":
            self.scan_publisher.publish(message)
            return
        output = LaserScan()
        output.header = message.header
        output.angle_min = message.angle_min
        output.angle_max = message.angle_max
        output.angle_increment = message.angle_increment
        output.time_increment = message.time_increment
        output.scan_time = message.scan_time
        output.range_min = message.range_min
        output.range_max = message.range_max
        output.intensities = list(message.intensities)
        close_range = max(message.range_min, 0.18)
        ranges = list(message.ranges)
        for index in range(len(ranges)):
            phase = math.fmod(index * 0.61803398875, 1.0)
            if phase < 0.85 * self.severity:
                ranges[index] = close_range
        output.ranges = ranges
        self.scan_publisher.publish(output)

    def on_cloud(self, message):
        if self.fault != "laser_degraded":
            self.cloud_publisher.publish(message)
            return
        fields = {field.name: field for field in message.fields}
        if not all(name in fields for name in ("x", "y", "z")):
            self.get_logger().error("PointCloud2 lacks x/y/z fields")
            return
        datatypes = {fields[name].datatype for name in ("x", "y", "z")}
        if len(datatypes) != 1 or next(iter(datatypes)) not in (
                PointField.FLOAT32, PointField.FLOAT64):
            self.get_logger().error(
                "PointCloud2 x/y/z must share FLOAT32 or FLOAT64 datatype")
            return
        output = copy.deepcopy(message)
        data = bytearray(output.data)
        datatype = next(iter(datatypes))
        scalar = np.dtype(
            (">" if output.is_bigendian else "<") +
            ("f4" if datatype == PointField.FLOAT32 else "f8"))
        point_dtype = np.dtype({
            "names": ["x", "y", "z"],
            "formats": [scalar, scalar, scalar],
            "offsets": [
                fields["x"].offset, fields["y"].offset, fields["z"].offset],
            "itemsize": output.point_step,
        })
        points = np.ndarray(
            shape=(output.height, output.width),
            dtype=point_dtype,
            buffer=data,
            strides=(output.row_step, output.point_step),
        )
        count = output.width * output.height
        phases = np.fmod(
            np.arange(count, dtype=np.float64) * 0.61803398875, 1.0)
        mask = (phases < 0.85 * self.severity).reshape(
            output.height, output.width)
        points["x"][mask] = 0.18
        points["y"][mask] = 0.0
        points["z"][mask] = 0.0
        output.data = bytes(data)
        self.cloud_publisher.publish(output)


def main():
    rclpy.init()
    node = SensorPipeline()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
