#!/usr/bin/env python3
"""Launch and record the 27-run Gazebo dataset matrix with resumable metadata."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import time


TOPICS = {
    "rgb": "/camera/image_raw",
    "camera_info": "/camera/camera_info",
    "laser_scan": "/scan",
    "point_cloud": "/points",
    "odometry": "/odom",
    "visual_localization": "/visual_localization/odom",
    "laser_localization": "/laser_localization/odom",
    "ground_truth": "/ground_truth/odom",
    "tf": "/tf",
    "tf_static": "/tf_static",
    "clock": "/clock",
    "metadata": "/localization_quality/run_metadata",
}

TOPIC_TYPES = {
    "/camera/image_raw": "sensor_msgs/msg/Image",
    "/camera/camera_info": "sensor_msgs/msg/CameraInfo",
    "/scan": "sensor_msgs/msg/LaserScan",
    "/points": "sensor_msgs/msg/PointCloud2",
    "/odom": "nav_msgs/msg/Odometry",
    "/visual_localization/odom": "nav_msgs/msg/Odometry",
    "/laser_localization/odom": "nav_msgs/msg/Odometry",
    "/ground_truth/odom": "nav_msgs/msg/Odometry",
    "/tf": "tf2_msgs/msg/TFMessage",
    "/tf_static": "tf2_msgs/msg/TFMessage",
    "/clock": "rosgraph_msgs/msg/Clock",
    "/localization_quality/run_metadata": "std_msgs/msg/String",
}

EXPECTED_RATES_HZ = {
    "/camera/image_raw": 10.0,
    "/camera/camera_info": 10.0,
    "/scan": 15.0,
    "/points": 5.0,
    "/odom": 50.0,
    "/visual_localization/odom": 50.0,
    "/laser_localization/odom": 50.0,
    "/ground_truth/odom": 50.0,
    "/tf": 50.0,
    "/clock": 10.0,
    "/localization_quality/run_metadata": 1.0,
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stop_process(process, timeout=20):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def start(command, env, log):
    return subprocess.Popen(
        command,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def validate_bag_topics(bag_path, duration=None):
    """Require the contract types, non-empty data and 95% native output."""
    database_files = sorted(bag_path.glob("*.db3"))
    if not database_files:
        raise RuntimeError("recorded bag contains no sqlite3 database")
    observed = {}
    for database_path in database_files:
        with sqlite3.connect(str(database_path)) as database:
            rows = database.execute(
                """
                SELECT topics.name, topics.type, COUNT(messages.id)
                FROM topics
                LEFT JOIN messages ON messages.topic_id = topics.id
                GROUP BY topics.id, topics.name, topics.type
                """
            ).fetchall()
        for name, message_type, count in rows:
            previous = observed.get(name)
            if previous is not None and previous["type"] != message_type:
                raise RuntimeError(
                    f"topic {name} changes type across bag files")
            observed[name] = {
                "type": message_type,
                "message_count": count + (
                    previous["message_count"] if previous else 0),
            }
    errors = []
    for topic, expected_type in TOPIC_TYPES.items():
        entry = observed.get(topic)
        if entry is None:
            errors.append(f"{topic}: missing")
        elif entry["type"] != expected_type:
            errors.append(
                f"{topic}: type {entry['type']} != {expected_type}")
        elif entry["message_count"] <= 0:
            errors.append(f"{topic}: zero messages")
    minimum_counts = {}
    if duration is not None:
        # Recording begins just before entity spawn; allow three seconds for
        # plugin discovery and require at least 95% thereafter.
        active_duration = max(1.0, duration - 3.0)
        for topic, rate in EXPECTED_RATES_HZ.items():
            minimum = max(1, int(active_duration * rate * 0.95))
            minimum_counts[topic] = minimum
            entry = observed.get(topic)
            if entry is not None and entry["message_count"] < minimum:
                errors.append(
                    f"{topic}: {entry['message_count']} messages < "
                    f"95% native-rate minimum {minimum}")
    if errors:
        raise RuntimeError(
            "bag topic contract failed: " + "; ".join(errors))
    return {
        topic: {
            **observed[topic],
            "minimum_required": minimum_counts.get(topic, 1),
        }
        for topic in TOPIC_TYPES
    }


def wait_for_clock(env, launch_process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if launch_process.poll() is not None:
            return False
        result = subprocess.run(
            ["ros2", "topic", "echo", "--once", "/clock"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return True
    return False


def record_run(run, index, dataset_root, duration, startup_timeout):
    run_directory = dataset_root / run["run_id"]
    metadata_path = run_directory / "run_metadata.json"
    if metadata_path.exists():
        return "skipped"
    if run_directory.exists():
        archive_root = dataset_root / "_incomplete"
        archive_root.mkdir(parents=True, exist_ok=True)
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived = archive_root / f"{run['run_id']}_{suffix}"
        run_directory.replace(archived)
        print(
            f"archived incomplete run at {archived}",
            file=sys.stderr, flush=True)
    run_directory.mkdir(parents=True, exist_ok=True)
    bag_path = run_directory / "bag"
    log_path = run_directory / "simulation.log"
    environment = dict(os.environ)
    environment["ROS_DOMAIN_ID"] = str(30 + index % 50)
    environment["RCUTILS_COLORIZED_OUTPUT"] = "0"

    launch_command = [
        "ros2", "launch", "localization_quality", "simulation.launch.py",
        f"world:={run['world']}",
        f"scene:={run['scene']}",
        f"lighting:={run['lighting']}",
        f"geometry_complexity:={run['geometry_complexity']}",
        f"fault:={run['fault']}",
        f"route:={run['route']}",
        f"seed:={run['seed']}",
        f"speed_mps:={run['speed_mps']}",
    ]
    started = datetime.now(timezone.utc).isoformat()
    launch_process = recorder = None
    with log_path.open("wb") as log:
        try:
            launch_process = start(launch_command, environment, log)
            if not wait_for_clock(environment, launch_process, startup_timeout):
                raise RuntimeError("Gazebo did not publish /clock before timeout")
            record_command = [
                "ros2", "bag", "record", "-o", str(bag_path),
                *TOPICS.values(),
            ]
            recorder = start(record_command, environment, log)
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                if launch_process.poll() is not None:
                    raise RuntimeError("simulation exited during recording")
                if recorder.poll() is not None:
                    raise RuntimeError("ros2 bag record exited during recording")
                time.sleep(min(1.0, deadline - time.monotonic()))
        finally:
            stop_process(recorder)
            stop_process(launch_process)

    info = subprocess.run(
        ["ros2", "bag", "info", str(bag_path)],
        env=environment, text=True, capture_output=True, check=False)
    if info.returncode != 0:
        raise RuntimeError(f"recorded bag failed validation: {info.stderr}")
    topic_statistics = validate_bag_topics(bag_path, duration)
    bag_files = sorted(
        path for path in bag_path.iterdir() if path.is_file())
    if not bag_files:
        raise RuntimeError("recorded bag directory is empty")
    metadata = {
        **run,
        "world": os.path.relpath(
            Path(run["world"]).resolve(), start=run_directory.resolve()),
        "format_version": 1,
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_requested_sec": duration,
        "bag_path": "bag",
        "topics": TOPICS,
        "localization_source": "controlled_ground_truth_noise_emulator",
        "coordinate_truth_frame": "world",
        "bag_info": info.stdout,
        "topic_statistics": topic_statistics,
        "native_rates_hz": EXPECTED_RATES_HZ,
        "minimum_output_completeness": 0.95,
        "files": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in bag_files
        },
    }
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    temporary.replace(metadata_path)
    return "recorded"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix")
    parser.add_argument("dataset_root")
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--run-id", action="append", default=[],
        help="record only the selected run_id; may be repeated")
    args = parser.parse_args()
    if shutil.which("ros2") is None:
        print("collect_sim_dataset: ros2 command is unavailable", file=sys.stderr)
        return 2
    try:
        matrix_path = Path(args.matrix).resolve()
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        runs = matrix["runs"]
        if args.run_id:
            selected = set(args.run_id)
            runs = [run for run in runs if run["run_id"] in selected]
            missing = selected - {run["run_id"] for run in runs}
            if missing:
                raise ValueError(f"unknown run_id values: {sorted(missing)}")
        for run in runs:
            world = Path(run["world"])
            if not world.is_absolute():
                run["world"] = str((matrix_path.parent / world).resolve())
        if args.limit > 0:
            runs = runs[:args.limit]
        dataset_root = Path(args.dataset_root)
        dataset_root.mkdir(parents=True, exist_ok=True)
        journal_path = dataset_root / "collection_journal.json"
        journal_by_run = {}
        if journal_path.exists():
            existing_journal = json.loads(
                journal_path.read_text(encoding="utf-8"))
            journal_by_run = {
                item["run_id"]: item for item in existing_journal
                if "run_id" in item
            }
        for index, run in enumerate(runs):
            print(f"[{index + 1}/{len(runs)}] {run['run_id']}", flush=True)
            try:
                status = record_run(
                    run, index, dataset_root, args.duration,
                    args.startup_timeout)
                journal_by_run[run["run_id"]] = {
                    "run_id": run["run_id"], "status": status}
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                journal_by_run[run["run_id"]] = {
                    "run_id": run["run_id"], "status": "failed",
                    "error": str(error),
                }
                journal = [
                    journal_by_run[key] for key in sorted(journal_by_run)]
                journal_path.write_text(
                    json.dumps(journal, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
                raise
            journal = [
                journal_by_run[key] for key in sorted(journal_by_run)]
            journal_path.write_text(
                json.dumps(journal, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
        print(json.dumps(journal, ensure_ascii=False, indent=2))
        return 0
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        print(f"collect_sim_dataset: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
