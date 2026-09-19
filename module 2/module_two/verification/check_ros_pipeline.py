#!/usr/bin/env python3
"""Exercise real ROS2 nodes with synthetic inputs, never localization accuracy.

Run in a sourced ROS2 Humble + module_two colcon workspace. The independent
domain avoids interacting with a robot's normal ROS graph. No velocity command
publisher is created. Child process groups are always shut down in finally.
"""
import argparse
from collections import deque
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback


def seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def oracle_costs(snapshot, evaluation_time):
    """Scalar specification oracle, independent of the production model code."""
    output = [140] * (snapshot.info.width * snapshot.info.height)
    for offset, index in enumerate(snapshot.indices):
        mean = float(snapshot.mean[offset])
        variance = float(snapshot.variance[offset])
        count = int(snapshot.sample_count[offset])
        age = max(0.0, evaluation_time - seconds(snapshot.last_observed[offset]))
        quality = min(1.0, max(0.0, mean - math.sqrt(variance)))
        base = 1.0 - quality
        support = -math.expm1(-count / 5.0) * math.exp(-age / 30.0)
        risk = base + (1.0 - support) * max(0.0, 0.7 - base)
        output[index] = math.floor(200 * risk + 0.5)
    return output


def production_provenance():
    from ament_index_python.packages import get_package_prefix
    package = 'quality_aware_navigation'
    prefix = Path(get_package_prefix(package)).resolve()
    source_directory = Path(__file__).resolve().parents[1] / 'ros2_ws/src' / package / package
    rows = []
    for name in ('ros_node', 'model', 'grid'):
        module = importlib.import_module(package + '.' + name)
        loaded = Path(module.__file__).resolve()
        source = source_directory / (name + '.py')
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
        installed = []
        for candidate in prefix.rglob(name + '.py'):
            if candidate.parent.name == package:
                installed.append({'path': str(candidate.resolve()),
                                  'sha256': hashlib.sha256(candidate.read_bytes()).hexdigest()})
        loaded_hash = hashlib.sha256(loaded.read_bytes()).hexdigest()
        rows.append({
            'module': package + '.' + name, 'loaded_path': str(loaded), 'loaded_sha256': loaded_hash,
            'workspace_source_path': str(source), 'workspace_source_sha256': source_hash,
            'installed_copies': installed,
            'installed_source_loaded_match': bool(installed) and source_hash == loaded_hash and
            all(item['sha256'] == source_hash for item in installed),
        })
    return {
        'ros_distro': os.environ.get('ROS_DISTRO', 'unknown'),
        'python_executable': sys.executable, 'python_version': platform.python_version(),
        'platform': platform.platform(), 'package_prefix': str(prefix),
        'core_files': rows,
        'installed_source_loaded_match': all(item['installed_source_loaded_match'] for item in rows),
    }


class Verification:
    def __init__(self, args, report):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from builtin_interfaces.msg import Time
        from diagnostic_msgs.msg import DiagnosticArray
        from nav_msgs.msg import OccupancyGrid, Odometry
        from nav2_msgs.msg import Costmap
        from sensor_msgs.msg import Image, LaserScan
        from quality_navigation_msgs.msg import QualityGrid

        self.args, self.report = args, report
        self.rclpy, self.Time = rclpy, Time
        self.Image, self.LaserScan, self.Odometry = Image, LaserScan, Odometry
        self.QualityGrid = QualityGrid
        rclpy.init(args=[])
        self.node = Node('quality_navigation_pipeline_verification')
        self.processes = []
        self.subscriptions = []
        self.costs = {'pipeline': deque(maxlen=200), 'direct': deque(maxlen=200)}
        self.diagnostics = {'pipeline': None, 'direct': None}
        self.stats = deque(maxlen=100)
        self.mean_maps = deque(maxlen=100)
        self.qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.subscriptions.append(self.node.create_subscription(
            QualityGrid, '/localization_quality/quality_stats', self.stats.append, self.qos))
        self.subscriptions.append(self.node.create_subscription(
            OccupancyGrid, '/localization_quality/quality_map', self.mean_maps.append, self.qos))
        for key, topic, diagnostic_topic in (
                ('pipeline', '/localization_quality/navigation_cost', '/quality_navigation/diagnostics'),
                ('direct', '/verification/direct_navigation_cost', '/verification/direct_diagnostics')):
            self.subscriptions.append(self.node.create_subscription(
                Costmap, topic, lambda msg, key=key: self.costs[key].append(msg), self.qos))
            self.subscriptions.append(self.node.create_subscription(
                DiagnosticArray, diagnostic_topic,
                lambda msg, key=key: self.on_diagnostic(key, msg), 10))
        self.direct_publisher = self.node.create_publisher(
            QualityGrid, '/verification/direct_quality_stats', self.qos)
        self.image_publisher = self.node.create_publisher(Image, '/verification/image', 10)
        self.scan_publisher = self.node.create_publisher(LaserScan, '/verification/scan', 10)
        self.odom_publisher = self.node.create_publisher(Odometry, '/verification/odom', 10)

    def on_diagnostic(self, key, message):
        for status in message.status:
            if status.name == 'quality_navigation/cost_mapping':
                self.diagnostics[key] = {
                    'state': status.message,
                    'level': status.level[0] if isinstance(status.level, (bytes, bytearray)) else int(status.level),
                    **{entry.key: entry.value for entry in status.values}}

    def start(self, name, command):
        path = self.args.output.parent / (name + '.log')
        stream = path.open('w', encoding='utf-8')
        environment = dict(os.environ, ROS_DOMAIN_ID=str(self.args.domain_id), ROS_LOCALHOST_ONLY='1')
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        entry = {'name': name, 'process': process, 'stream': stream, 'log': str(path),
                 'command': command, 'stopped': False}
        self.processes.append(entry)
        return entry

    def stop(self, entry):
        if entry['stopped']:
            return
        process = entry['process']
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)
        entry['stream'].close()
        entry['stopped'] = True

    def spin(self, duration):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=max(0.0, min(0.025, deadline - time.monotonic())))
            for entry in self.processes:
                if not entry['stopped'] and entry['process'].poll() is not None:
                    raise RuntimeError(f"{entry['name']} exited early; inspect {entry['log']}")

    def wait(self, predicate, description, timeout=None):
        deadline = time.monotonic() + (self.args.timeout if timeout is None else timeout)
        next_progress = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            result = predicate()
            if result:
                return result
            self.spin(0.03)
            if time.monotonic() >= next_progress:
                print('WAIT ' + description + ' (child processes still live)', flush=True)
                next_progress = time.monotonic() + 10.0
        raise AssertionError(f'Timeout: {description}; diagnostics={self.diagnostics}')

    def stamp(self, timestamp=None):
        if timestamp is None:
            return self.node.get_clock().now().to_msg()
        whole = math.floor(timestamp)
        nano = int(math.floor((timestamp - whole) * 1e9 + 0.5))
        if nano >= 1_000_000_000:
            whole, nano = whole + 1, 0
        return self.Time(sec=int(whole), nanosec=nano)

    def record(self, name, **details):
        self.report['checks'].append({'name': name, 'passed': True, **details})
        print('PASS ' + name, flush=True)

    def cost_for(self, key, snapshot):
        stamp = seconds(snapshot.header.stamp)

        def matching():
            diag = self.diagnostics[key]
            if not diag or diag.get('state') not in ('cumulative', 'windowed'):
                return None
            if abs(float(diag.get('source_stamp', '-1')) - stamp) > 1e-6:
                return None
            for candidate in reversed(self.costs[key]):
                if seconds(candidate.header.stamp) < stamp:
                    continue
                if (candidate.metadata.size_x, candidate.metadata.size_y) != (
                        snapshot.info.width, snapshot.info.height):
                    continue
                expected = oracle_costs(snapshot, seconds(candidate.header.stamp))
                if list(candidate.data) == expected:
                    return candidate
            return None

        return self.wait(matching, key + ' exact model costs from the accepted snapshot')

    def launch_cost(self, direct=False):
        command = ['ros2', 'run', 'quality_aware_navigation', 'quality_cost_node', '--ros-args',
                   '-p', 'input_mode:=statistics', '-p', 'publish_period:=0.1',
                   '-p', 'source_timeout:=' + ('1.0' if direct else '2.0'),
                   '-p', 'mode:=full', '-p', 'beta:=1.0', '-p', 'gamma:=1.0',
                   '-p', 'sample_scale:=5.0', '-p', 'age_scale:=30.0',
                   '-p', 'unknown_risk:=0.7', '-p', 'max_cost:=200', '-p', 'stale_cost:=200']
        if direct:
            command += ['-p', 'statistics_topic:=/verification/direct_quality_stats',
                        '-p', 'output_topic:=/verification/direct_navigation_cost',
                        '-r', '/quality_navigation/diagnostics:=/verification/direct_diagnostics']
        return self.start('direct_cost' if direct else 'pipeline_cost', command)

    def module_one_pipeline(self):
        cost = self.launch_cost()
        evaluator = self.start('realtime_evaluator', [
            'ros2', 'run', 'localization_quality', 'realtime_evaluator', '--ros-args',
            '-p', 'frames.target:=map', '-p', 'fusion.mlp_enabled:=false',
            '-p', 'topics.scan:=/verification/scan', '-p', 'topics.image:=/verification/image',
            '-p', 'topics.odom:=/verification/odom', '-p', 'map.width:=8', '-p', 'map.height:=8',
            '-p', 'map.resolution:=0.5', '-p', 'map.origin_x:=-2.0', '-p', 'map.origin_y:=-2.0',
            '-p', 'map.publish_statistics:=true', '-p', 'map.statistics_publish_period_sec:=0.1',
            '-p', 'map.statistics_window_sec:=10.0', '-p', 'sync.tolerance_sec:=0.25'])
        self.wait(lambda: self.image_publisher.get_subscription_count() and
                  self.scan_publisher.get_subscription_count() and
                  self.odom_publisher.get_subscription_count(), 'module-one sensor subscriptions',
                  timeout=self.args.startup_timeout)
        self.wait(lambda: self.diagnostics['pipeline'] is not None,
                  'pipeline cost node ready before finite sensor sequence',
                  timeout=self.args.startup_timeout)
        self.spin(0.15)
        width, height = 160, 120
        pixels = bytes(value for y in range(height) for x in range(width)
                       for value in ([45, 160, 210] if (x // 8 + y // 8) % 2 else [180, 75, 50]))
        for _ in range(12):
            stamp = self.stamp()
            odom = self.Odometry()
            odom.header.stamp, odom.header.frame_id = stamp, 'map'
            odom.child_frame_id = 'base_link'
            odom.pose.pose.position.x = odom.pose.pose.position.y = 0.25
            odom.pose.pose.orientation.w = 1.0
            image = self.Image()
            image.header.stamp, image.header.frame_id = stamp, 'camera'
            image.width, image.height, image.encoding = width, height, 'bgr8'
            image.step, image.data = width * 3, pixels
            self.odom_publisher.publish(odom)
            self.image_publisher.publish(image)
            self.spin(0.06)  # let image/pose callbacks precede the causal laser pivot
            scan = self.LaserScan()
            scan.header.stamp, scan.header.frame_id = stamp, 'laser'
            scan.angle_min, scan.angle_max = -math.pi, math.pi
            scan.angle_increment = 2 * math.pi / 359
            scan.range_min, scan.range_max = 0.1, 10.0
            scan.ranges = [3.0 + 0.7 * math.sin(i * 0.071) + 0.25 * math.sin(i * 0.27)
                           for i in range(360)]
            self.scan_publisher.publish(scan)
            self.spin(0.08)
        self.wait(lambda: self.stats and max(self.stats[-1].sample_count, default=0) >= 3,
                  'module-one observed statistics')
        self.spin(0.15)
        snapshot = self.stats[-1]
        assert snapshot.statistics_mode == 'windowed'
        assert list(snapshot.indices) == [36], list(snapshot.indices)
        assert snapshot.header.frame_id == 'map'
        assert 0 <= snapshot.mean[0] <= 1
        assert seconds(snapshot.last_observed[0]) <= seconds(snapshot.header.stamp)
        output = self.cost_for('pipeline', snapshot)
        self.wait(lambda: self.mean_maps, 'compatible legacy quality map')
        legacy = self.mean_maps[-1]
        assert legacy.info.width == 8 and legacy.info.height == 8
        assert legacy.data[36] >= 0 and legacy.data[0] == -1
        self.record('synthetic_sensor_to_module_one_statistics_to_nav2_costmap',
                    known_cell=36, sample_count=int(snapshot.sample_count[0]),
                    mean=float(snapshot.mean[0]), variance=float(snapshot.variance[0]),
                    observed_at=seconds(snapshot.last_observed[0]),
                    source_stamp=seconds(snapshot.header.stamp),
                    mapped_cost=int(output.data[36]), unknown_cost=int(output.data[0]),
                    legacy_quality=int(legacy.data[36]), statistics_mode=snapshot.statistics_mode,
                    note='Synthetic images/scans/odometry test transport and score semantics; not localization accuracy.')
        self.stop(evaluator)
        self.stop(cost)

    def direct_snapshot(self, geometry=False):
        message = self.QualityGrid()
        message.header.stamp, message.header.frame_id = self.stamp(), 'map'
        message.info.resolution = 0.25 if geometry else 0.5
        message.info.width, message.info.height = (5, 3) if geometry else (4, 2)
        message.info.origin.position.x = -2.0 if geometry else -1.0
        message.info.origin.position.y = 3.0 if geometry else -1.0
        message.info.origin.orientation.z = math.sin(math.pi / 4) if geometry else 0.0
        message.info.origin.orientation.w = math.cos(math.pi / 4) if geometry else 1.0
        message.statistics_mode = 'windowed'
        message.indices = [0, 1, 2, 3]
        message.mean, message.variance = [1.0, 0.0, 1.0, 0.8], [0.0, 0.0, 0.0, 0.04]
        message.sample_count = [1000, 1000, 1000, 5]
        source = seconds(message.header.stamp)
        message.last_observed = [self.stamp(source - age) for age in (0, 0, 60, 0)]
        return message

    def publish_valid(self, geometry=False):
        message = self.direct_snapshot(geometry)
        self.direct_publisher.publish(message)
        return message, self.cost_for('direct', message)

    def conservative_fallback(self, previous_rejected=None):
        def matching():
            diag = self.diagnostics['direct']
            if not diag or diag.get('state') != 'source_stale_conservative_fallback':
                return None
            if previous_rejected is not None and int(diag['rejected']) <= previous_rejected:
                return None
            if not self.costs['direct']:
                return None
            message = self.costs['direct'][-1]
            return message if message.data and all(value >= 200 for value in message.data) else None
        return self.wait(matching, 'conservative fallback on invalid or missing source')

    def direct_contract(self):
        self.launch_cost(direct=True)
        self.wait(lambda: self.direct_publisher.get_subscription_count(), 'direct statistics subscriber',
                  timeout=self.args.startup_timeout)
        snapshot, output = self.publish_valid()
        assert output.data[1] == 200 and output.data[4] == 140
        assert output.data[0] < output.data[2] < 140
        self.record('exact_fresh_bad_stale_uncertain_and_unknown_costs',
                    source_stamp=seconds(snapshot.header.stamp), costs=list(output.data),
                    oracle=oracle_costs(snapshot, seconds(output.header.stamp)))

        for fault in ('array_length_mismatch', 'duplicate_indices', 'future_observation'):
            before = int(self.diagnostics['direct']['rejected'])
            malformed = self.direct_snapshot()
            if fault == 'array_length_mismatch':
                malformed.mean = [1.0]
            elif fault == 'duplicate_indices':
                malformed.indices = [0, 0, 2, 3]
            else:
                malformed.last_observed[0] = self.stamp(seconds(malformed.header.stamp) + 5.0)
            self.direct_publisher.publish(malformed)
            fallback = self.conservative_fallback(before)
            self.record('reject_' + fault, costs=list(fallback.data),
                        diagnostic=copy.deepcopy(self.diagnostics['direct']))
            self.publish_valid()

        # Replayed source time is not new evidence and must not refresh the source.
        snapshot, _ = self.publish_valid()
        before_rejected = int(self.diagnostics['direct']['rejected'])
        before_accepted = int(self.diagnostics['direct']['accepted'])
        self.direct_publisher.publish(snapshot)
        self.conservative_fallback(before_rejected)
        assert int(self.diagnostics['direct']['accepted']) == before_accepted
        self.record('duplicate_source_timestamp_does_not_refresh',
                    diagnostic=copy.deepcopy(self.diagnostics['direct']))

        snapshot, _ = self.publish_valid()
        self.spin(1.25)
        fallback = self.conservative_fallback()
        assert seconds(fallback.header.stamp) - seconds(snapshot.header.stamp) > 1.0
        self.record('source_timeout_conservative_fallback', costs=list(fallback.data),
                    elapsed_source_time=seconds(fallback.header.stamp) - seconds(snapshot.header.stamp),
                    diagnostic=copy.deepcopy(self.diagnostics['direct']))
        restored, output = self.publish_valid()
        assert min(output.data) < 200
        self.record('fresh_source_recovers_after_timeout', costs=list(output.data),
                    oracle=oracle_costs(restored, seconds(output.header.stamp)))

        changed, output = self.publish_valid(geometry=True)
        metadata = output.metadata
        assert metadata.size_x == 5 and metadata.size_y == 3 and len(output.data) == 15
        assert abs(metadata.resolution - 0.25) < 1e-6
        assert metadata.origin.position.x == -2.0 and metadata.origin.position.y == 3.0
        assert abs(metadata.origin.orientation.z - math.sin(math.pi / 4)) < 1e-6
        assert abs(metadata.origin.orientation.w - math.cos(math.pi / 4)) < 1e-6
        self.record('geometry_change_replaces_dimensions_origin_and_rotation',
                    width=5, height=3, resolution=metadata.resolution,
                    origin=[metadata.origin.position.x, metadata.origin.position.y, math.pi / 2],
                    costs=list(output.data))

        cleared = self.direct_snapshot(geometry=True)
        cleared.indices, cleared.mean, cleared.variance = [], [], []
        cleared.sample_count, cleared.last_observed = [], []
        self.direct_publisher.publish(cleared)
        output = self.cost_for('direct', cleared)
        assert list(output.data) == [140] * 15
        self.record('empty_snapshot_clears_previous_known_cells', costs=list(output.data))

    def close(self):
        for entry in reversed(self.processes):
            self.stop(entry)
        self.report['processes'] = []
        for entry in self.processes:
            self.report['processes'].append({
                'name': entry['name'], 'command': entry['command'], 'log': entry['log'],
                'returncode': entry['process'].returncode,
                'log_tail': Path(entry['log']).read_text(encoding='utf-8', errors='replace')[-4000:]})
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'results/ros_pipeline/report.json')
    parser.add_argument('--domain-id', type=int, default=73)
    parser.add_argument('--timeout', type=float, default=15.0)
    parser.add_argument('--startup-timeout', type=float, default=90.0,
                        help='Separate cold-start/discovery budget for ROS CLI and Python imports.')
    parser.add_argument('--skip-module-one', action='store_true',
                        help='Only test the installed cost node; report the omitted pipeline explicitly.')
    parser.add_argument('--require-installed-match', action='store_true',
                        help='Fail unless loaded, installed and workspace core Python files have identical hashes.')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ['ROS_DOMAIN_ID'] = str(args.domain_id)
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    report = {
        'artifact_type': 'real_ros2_pipeline_interface_verification',
        'scope': 'Synthetic sensor/statistics inputs passed through actual installed ROS2 nodes.',
        'does_not_measure': ['real localization accuracy', 'navigation success',
                             'Nav2 plugin loading or planner/controller behavior', 'robot collision avoidance'],
        'domain_id': args.domain_id, 'clock': 'shared ROS system/source clock; no simulation clock mixing',
        'module_one_pipeline_requested': not args.skip_module_one, 'checks': [], 'passed': False,
    }
    verification = None
    started = time.monotonic()
    try:
        report['provenance'] = production_provenance()
        verification = Verification(args, report)
        if not args.skip_module_one:
            verification.module_one_pipeline()
        verification.direct_contract()
        if args.require_installed_match and not report['provenance']['installed_source_loaded_match']:
            raise AssertionError('Production code differs between loaded, installed and workspace copies')
        report['passed'] = True
    except Exception as error:
        report['error'] = str(error)
        report['traceback'] = traceback.format_exc()
        print(report['traceback'], file=sys.stderr)
    finally:
        if verification is not None:
            try:
                verification.close()
            except Exception as error:
                report['cleanup_error'] = str(error)
                report['passed'] = False
        report['duration_seconds'] = time.monotonic() - started
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('Report: ' + str(args.output), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
