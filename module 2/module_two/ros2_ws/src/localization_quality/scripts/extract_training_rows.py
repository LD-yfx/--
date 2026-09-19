#!/usr/bin/env python3
"""Join offline quality features with causal visual/lidar/ground-truth poses."""

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
import sys

from nav_msgs.msg import Odometry
import rclpy
from rclpy.serialization import deserialize_message
import rosbag2_py


POSE_TOPICS = (
    "visual_localization",
    "laser_localization",
    "ground_truth",
)


def stamp(message, record_time):
    value = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
    return value if value > 0.0 else record_time / 1e9


def read_poses(bag_path, topics):
    requested = {topics[name]: name for name in POSE_TOPICS}
    output = {name: [] for name in POSE_TOPICS}
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"),
    )
    while reader.has_next():
        topic, data, record_time = reader.read_next()
        name = requested.get(topic)
        if name is None:
            continue
        message = deserialize_message(data, Odometry)
        output[name].append((stamp(message, record_time), message.pose.pose))
    for poses in output.values():
        poses.sort(key=lambda item: item[0])
    return output


def causal(poses, timestamp, tolerance):
    timestamps = [item[0] for item in poses]
    index = bisect.bisect_right(timestamps, timestamp) - 1
    if index < 0 or timestamp - poses[index][0] > tolerance:
        return None
    return poses[index][1]


def add_pose(row, prefix, pose):
    if pose is None:
        for axis in ("x", "y", "z"):
            row[f"{prefix}_{axis}"] = math.nan
        for axis in ("x", "y", "z", "w"):
            row[f"{prefix}_q{axis}"] = math.nan
        return
    row[f"{prefix}_x"] = pose.position.x
    row[f"{prefix}_y"] = pose.position.y
    row[f"{prefix}_z"] = pose.position.z
    row[f"{prefix}_qx"] = pose.orientation.x
    row[f"{prefix}_qy"] = pose.orientation.y
    row[f"{prefix}_qz"] = pose.orientation.z
    row[f"{prefix}_qw"] = pose.orientation.w


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_metadata")
    parser.add_argument("quality_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--max-features", type=float, default=500.0)
    args = parser.parse_args()
    try:
        metadata_path = Path(args.run_metadata).resolve()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        bag_path = Path(metadata["bag_path"])
        if not bag_path.is_absolute():
            bag_path = (metadata_path.parent / bag_path).resolve()
        poses = read_poses(bag_path, metadata["topics"])
        with Path(args.quality_csv).open(
                "r", encoding="utf-8", newline="") as stream:
            quality_rows = list(csv.DictReader(stream))
        if not quality_rows:
            raise ValueError("quality CSV is empty")
        output = []
        for quality in quality_rows:
            timestamp = float(quality["timestamp"])
            row = dict(quality)
            row.update({
                "run_id": metadata["run_id"],
                "scene": metadata["scene"],
                "lighting": metadata["lighting"],
                "geometry_label": metadata["geometry_complexity"],
                "fault": metadata["fault"],
                "seed": metadata["seed"],
                "trackable_features_norm": min(
                    1.0, float(quality["visual_trackable_features"]) /
                    args.max_features),
                "coverage_2d": quality["laser_coverage_rate"],
                "underexposed_ratio": quality[
                    "visual_underexposed_ratio"],
                "overexposed_ratio": quality[
                    "visual_overexposed_ratio"],
                "edge_density": quality["visual_edge_density"],
                "feature_match_rate": quality[
                    "visual_feature_match_rate"],
                "illumination_level": quality["illumination_level"],
                "illumination_quality": quality["illumination_quality"],
                "openness": quality["openness"],
                "geometry_complexity": quality["geometry_complexity"],
                "direction_entropy": quality["laser_direction_entropy"],
                "directional_anisotropy": quality[
                    "directional_anisotropy"],
                "linearity": quality["laser_linearity"],
                "planarity": quality["laser_planarity"],
                "visual_valid": quality["visual_valid"],
                "laser_valid": quality["laser_valid"],
            })
            for name, prefix in (
                ("visual_localization", "visual"),
                ("laser_localization", "laser"),
                ("ground_truth", "gt"),
            ):
                add_pose(row, prefix, causal(
                    poses[name], timestamp, args.tolerance))
            output.append(row)
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(output[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(output)
        print(json.dumps({
            "run_id": metadata["run_id"],
            "rows": len(output),
            "output": str(output_path.resolve()),
            "sync_direction": "causal_past",
        }, indent=2))
        return 0
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"extract_training_rows: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    rclpy.init()
    try:
        raise SystemExit(main())
    finally:
        rclpy.shutdown()
