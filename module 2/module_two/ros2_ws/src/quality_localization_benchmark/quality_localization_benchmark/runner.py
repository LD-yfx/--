"""Run and record one independent Gazebo / AMCL / actual-module-one trial.

Truth is read only by measurement generation and this recorder. Fixed Nav2 goals
are never corrected using truth. Failed trials keep their raw data and status.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import time
import traceback

from .evaluation import evaluate_episode, summarize_quality_relation
from .quality_coverage import summarize_quality_coverage


METHODS = ('geometric', 'linear', 'full', 'no_variance', 'no_count', 'no_age')
PASSTHROUGH_COUNTERS = ('truth_missing_passthrough', 'queue_overflow_passthrough',
                        'measurement_error_passthrough', 'invalid_stamp_passthrough',
                        'clock_rewinds')
BAG_TOPICS = ['/clock', '/sim/scan', '/front_camera/image_raw',
              '/front_camera/camera_info', '/scan', '/camera/image_raw',
              '/camera/camera_info', '/odom', '/amcl_pose',
              '/localization/estimated_odom', '/laser_localization/odom',
              '/ground_truth/odom', '/tf', '/tf_static', '/cmd_vel', '/plan',
              '/localization_quality/quality_stats', '/quality_navigation/diagnostics',
              '/benchmark/sensor_bridge/diagnostics', '/localization/adapter_diagnostics',
              '/localization_quality/diagnostics', '/localization_quality/metrics',
              '/localization_quality/navigation_cost', '/global_costmap/costmap_raw',
              '/benchmark/tf_authorities']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def stamp_seconds(stamp):
    return stamp.sec+stamp.nanosec*1.e-9


def quaternion_angles(q):
    return (math.atan2(2*(q.w*q.x+q.y*q.z), 1-2*(q.x*q.x+q.y*q.y)),
            math.asin(max(-1., min(1., 2*(q.w*q.y-q.z*q.x)))),
            math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)))


def intersects(x, y, rectangles, radius):
    """Conservative circular footprint vs exact static obstacle rectangles."""
    return any((x-max(a, min(b, x)))**2+(y-max(c, min(d, y)))**2 <= radius**2
               for a, b, c, d in rectangles)


def counter_delta(before, after):
    result = {}
    for key in PASSTHROUGH_COUNTERS:
        try:
            start, finish = int(before[key]), int(after[key])
            result[key] = finish-start if finish >= start else None
        except (KeyError, TypeError, ValueError):
            result[key] = None
    return result


def action_completed_within_deadline(leg):
    # Cancellation and completion can race. A late STATUS_SUCCEEDED cannot undo
    # a deadline that the mission already exceeded.
    return leg.get('action_status') == 4 and not leg.get('timeout', False)


def truth_motion_audit(rows, start, end, rectangles, radius):
    """Use source-time boundaries, including truth delivered after mission end."""
    counts = dict(samples=0, collision_count=0, contact_samples=0, unstable_samples=0)
    if start is None or end is None:
        return counts
    previous_contact = False
    for row in rows:
        if not start <= row['t'] <= end:
            continue
        counts['samples'] += 1
        contact = intersects(row['x'], row['y'], rectangles, radius)
        counts['contact_samples'] += int(contact)
        counts['collision_count'] += int(contact and not previous_contact)
        previous_contact = contact
        counts['unstable_samples'] += int(abs(row.get('roll', 0.)) > .5 or
            abs(row.get('pitch', 0.)) > .5 or abs(row.get('z', 0.)) > .25)
    return counts


def cost_consumption_summary(rows, start, end):
    """Report actual consumer operation, separate from published quality coverage."""
    selected = [r for r in rows if r.get('name') == 'quality_navigation/cost_mapping']
    selected.sort(key=lambda r: max(r['t'], r.get('received_ros', r['t'])))
    durations = defaultdict(float)
    if start is not None and end is not None and end > start:
        for i, row in enumerate(selected):
            available = max(row['t'], row.get('received_ros', row['t']))
            next_t = max(selected[i+1]['t'], selected[i+1].get('received_ros', selected[i+1]['t'])) if i+1 < len(selected) else end
            duration = max(0., min(end, next_t, available+1.)-max(start, available))
            durations[row['message']] += duration
    denominator = end-start if start is not None and end is not None and end > start else None
    observed = sum(durations.values())
    before = [r for r in selected if start is not None and r['t'] <= start]
    after = [r for r in selected if end is not None and r['t'] >= end]
    first = before[-1] if before else (selected[0] if selected else {})
    last = after[0] if after else (selected[-1] if selected else {})
    deltas = {}
    for key in ('accepted', 'rejected', 'deferred', 'deferred_accepted', 'pending_expired', 'pending_rewind_dropped'):
        try:
            delta = int(last['values'][key])-int(first['values'][key])
            deltas[key] = delta if delta >= 0 else None
        except (KeyError, ValueError, TypeError):
            deltas[key] = None
    return dict(counter_envelope_delta=deltas, state_seconds=dict(durations),
                diagnostic_coverage=observed/denominator if denominator else None,
                fallback_fraction=durations.get('source_stale_conservative_fallback', 0.)/denominator if denominator else None,
                missing_diagnostic_seconds=max(0., denominator-observed) if denominator else None,
                counter_envelope=[first.get('t'), last.get('t')])


def read_jsonl(path):
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def stop_process(process):
    """Only stop a process group created by this runner; preserve other WSL jobs."""
    if process is None:
        return None
    for sig, timeout in ((signal.SIGINT, 20), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        if process.poll() is not None:
            break
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
    return process.poll()


def source_provenance(share):
    """Hash installed modules and their authoritative synchronized source files."""
    import importlib
    workspace = next((p for p in Path(share).parents if p.name == 'ros2_ws'), None)
    result = {}
    modules = {
        'runner': ('quality_localization_benchmark.runner', 'quality_localization_benchmark/quality_localization_benchmark/runner.py'),
        'evaluation': ('quality_localization_benchmark.evaluation', 'quality_localization_benchmark/quality_localization_benchmark/evaluation.py'),
        'quality_coverage': ('quality_localization_benchmark.quality_coverage', 'quality_localization_benchmark/quality_localization_benchmark/quality_coverage.py'),
        'adapter': ('quality_localization_benchmark.localization_adapter', 'quality_localization_benchmark/quality_localization_benchmark/localization_adapter.py'),
        'sensor_bridge': ('quality_localization_benchmark.sensor_bridge', 'quality_localization_benchmark/quality_localization_benchmark/sensor_bridge.py'),
        'cost_model': ('quality_aware_navigation.model', 'quality_aware_navigation/quality_aware_navigation/model.py'),
        'cost_node': ('quality_aware_navigation.ros_node', 'quality_aware_navigation/quality_aware_navigation/ros_node.py'),
    }
    for name, (module, relative) in modules.items():
        installed = Path(importlib.import_module(module).__file__).resolve()
        source = workspace/'src'/relative if workspace else None
        result[name] = dict(runtime_path=str(installed), runtime_sha256=sha256(installed),
                            source_path=str(source) if source else None,
                            source_sha256=sha256(source) if source and source.is_file() else None,
                            installed=bool(workspace and installed.is_relative_to(workspace/'install')))
        result[name]['source_matches_runtime'] = result[name]['source_sha256'] == result[name]['runtime_sha256']
    # Module-one package layouts differ from the new Python packages. Hash its
    # installed tree and the exact source tree instead of guessing import names.
    if workspace:
        for package in ('localization_quality', 'quality_nav2_layer', 'quality_localization_audit'):
            for category, base in (('source', workspace/'src'/package), ('installed', workspace/'install'/package)):
                result[package+'_'+category] = {
                    str(p.relative_to(base)): sha256(p) for p in sorted(base.rglob('*'))
                    if p.is_file() and p.suffix in ('.py', '.so', '.yaml', '.xml', '.hpp', '.cpp')
                    and '__pycache__' not in p.parts}
        binary = workspace/'install/localization_quality/lib/localization_quality/realtime_evaluator'
        result['module_one_executable'] = dict(path=str(binary), sha256=sha256(binary))
        audit_binary = workspace/'install/quality_localization_audit/lib/quality_localization_audit/tf_authority_audit'
        result['tf_audit_executable'] = dict(path=str(audit_binary), sha256=sha256(audit_binary))
    return result


def software_environment():
    versions = subprocess.run(['dpkg-query', '-W', '-f', '${Package}=${Version}\n',
        'ros-humble-nav2-amcl', 'gazebo', 'ros-humble-navigation2', 'ros-humble-cv-bridge'],
        text=True, capture_output=True, check=False)
    return dict(os=platform.platform(), python=platform.python_version(),
                cpu_count=os.cpu_count(), ros_distro=os.environ.get('ROS_DISTRO'),
                packages=versions.stdout.splitlines(), package_query_returncode=versions.returncode)


def create_recorder(output, metadata):
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
    from diagnostic_msgs.msg import DiagnosticArray
    from geometry_msgs.msg import Twist
    from lifecycle_msgs.srv import GetState
    from rcl_interfaces.srv import GetParameters
    from nav2_msgs.action import NavigateToPose
    from nav_msgs.msg import Odometry, Path as RosPath
    from sensor_msgs.msg import Image, LaserScan
    from std_msgs.msg import String
    from quality_navigation_msgs.msg import QualityGrid

    class Recorder(Node):
        def __init__(self):
            super().__init__('benchmark_runner', parameter_overrides=[Parameter('use_sim_time', value=True)])
            self.output = output
            self.files = {}
            self.poses = {name: [] for name in ('estimates', 'truth', 'encoder')}
            self.counts = Counter()
            self.diagnostics = {}
            self.last_diagnostic_stamp = {}
            self.frame_ids = defaultdict(set)
            self.tf_authorities = defaultdict(set)
            self.tf_audit_record = None
            self.tf_audit_received_ros = None
            self.stats_latest = None
            self.mission = False
            self.collision_count = 0
            self.in_collision = False
            self.physics_invalid_count = 0
            self.last_sensor_log = {}
            self.action = ActionClient(self, NavigateToPose, '/navigate_to_pose')
            pose_qos = QoSProfile(depth=4096, reliability=ReliabilityPolicy.RELIABLE)
            for topic, name in (('/localization/estimated_odom', 'estimates'),
                                ('/ground_truth/odom', 'truth'), ('/odom', 'encoder')):
                self.create_subscription(Odometry, topic, lambda msg, key=name: self.pose(msg, key), pose_qos)
            self.create_subscription(Twist, '/cmd_vel', self.command, 100)
            self.create_subscription(RosPath, '/plan', self.path, 10)
            self.create_subscription(LaserScan, '/scan', self.scan, qos_profile_sensor_data)
            self.create_subscription(Image, '/camera/image_raw', self.image, qos_profile_sensor_data)
            self.create_subscription(QualityGrid, '/localization_quality/quality_stats', self.quality, 10)
            for topic in ('/quality_navigation/diagnostics', '/benchmark/sensor_bridge/diagnostics',
                          '/localization/adapter_diagnostics', '/localization_quality/diagnostics'):
                self.create_subscription(DiagnosticArray, topic, self.diagnostic, 20)
            # Humble rclpy's executor discards publisher MessageInfo. The C++
            # helper obtains real publisher GIDs; never infer owners from frames.
            self.create_subscription(String, '/benchmark/tf_authorities', self.tf_audit, 10)
            self.ready_client = self.create_client(GetState, '/bt_navigator/get_state')

        def now_seconds(self):
            return self.get_clock().now().nanoseconds*1.e-9

        def log(self, name, value):
            if name not in self.files:
                self.files[name] = (self.output/(name+'.jsonl')).open('w', encoding='utf-8')
            self.files[name].write(json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n')

        def pose(self, msg, name):
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            roll, pitch, yaw = quaternion_angles(q)
            row = dict(t=stamp_seconds(msg.header.stamp), x=p.x, y=p.y, z=p.z, yaw=yaw,
                       roll=roll, pitch=pitch, frame=msg.header.frame_id, child=msg.child_frame_id,
                       received_ros=self.now_seconds(), received_wall=time.monotonic(),
                       covariance_x=msg.pose.covariance[0], covariance_y=msg.pose.covariance[7],
                       covariance_yaw=msg.pose.covariance[35], vx=msg.twist.twist.linear.x,
                       wz=msg.twist.twist.angular.z)
            self.poses[name].append(row)
            self.frame_ids[name].add(msg.header.frame_id)
            self.counts[name] += 1
            self.log(name, row)
            if name == 'truth' and self.mission:
                contact = intersects(p.x, p.y, metadata['rectangles'], metadata['robot_collision_radius'])
                if contact and not self.in_collision:
                    self.collision_count += 1
                    self.log('events', dict(t=row['t'], event='conservative_footprint_intersection', x=p.x, y=p.y))
                self.in_collision = contact
                if abs(roll) > .5 or abs(pitch) > .5 or abs(p.z) > .25:
                    self.physics_invalid_count += 1

        def command(self, msg):
            self.counts['commands'] += 1
            self.log('commands', dict(t=self.now_seconds(), vx=msg.linear.x, wz=msg.angular.z))

        def path(self, msg):
            self.counts['plans'] += 1
            self.log('plans', dict(t=stamp_seconds(msg.header.stamp), frame=msg.header.frame_id,
                                  received_ros=self.now_seconds(),
                                  points=[[p.pose.position.x, p.pose.position.y] for p in msg.poses]))

        def scan(self, msg):
            self.counts['scan'] += 1
            if self.now_seconds()-self.last_sensor_log.get('scan', -10.) < 1.:
                return
            self.last_sensor_log['scan'] = self.now_seconds()
            finite = [float(x) for x in msg.ranges if math.isfinite(x)]
            self.log('sensors', dict(kind='scan', t=stamp_seconds(msg.header.stamp),
                received_ros=self.now_seconds(), frame=msg.header.frame_id, n=len(msg.ranges),
                finite_fraction=len(finite)/len(msg.ranges) if msg.ranges else 0.,
                minimum=min(finite) if finite else None, maximum=max(finite) if finite else None))

        def image(self, msg):
            self.counts['image'] += 1
            if self.now_seconds()-self.last_sensor_log.get('image', -10.) < 1.:
                return
            self.last_sensor_log['image'] = self.now_seconds()
            self.log('sensors', dict(kind='image', t=stamp_seconds(msg.header.stamp),
                received_ros=self.now_seconds(), frame=msg.header.frame_id, width=msg.width,
                height=msg.height, encoding=msg.encoding, sha256=hashlib.sha256(bytes(msg.data)).hexdigest()))
            if not (self.output/'first_camera.png').exists():
                import cv2
                from cv_bridge import CvBridge
                cv2.imwrite(str(self.output/'first_camera.png'), CvBridge().imgmsg_to_cv2(msg, 'bgr8'))

        def quality(self, msg):
            self.counts['quality_stats'] += 1
            row = dict(t=stamp_seconds(msg.header.stamp), frame=msg.header.frame_id,
                received_ros=self.now_seconds(),
                mode=msg.statistics_mode, width=msg.info.width, height=msg.info.height,
                resolution=msg.info.resolution, indices=list(msg.indices), mean=list(msg.mean),
                variance=list(msg.variance), sample_count=list(msg.sample_count),
                last_observed=[stamp_seconds(s) for s in msg.last_observed],
                origin=[msg.info.origin.position.x, msg.info.origin.position.y,
                        quaternion_angles(msg.info.origin.orientation)[2]])
            self.stats_latest = row
            self.log('quality_stats', row)

        def diagnostic(self, msg):
            for status in msg.status:
                values = {v.key: v.value for v in status.values}
                level = status.level[0] if isinstance(status.level, (bytes, bytearray)) else int(status.level)
                row = dict(t=stamp_seconds(msg.header.stamp), name=status.name, level=level,
                           message=status.message, values=values, received_ros=self.now_seconds())
                self.diagnostics[status.name] = row
                self.last_diagnostic_stamp[status.name] = row['t']
                self.log('diagnostics', row)

        def tf_audit(self, msg):
            self.tf_audit_record = json.loads(msg.data)
            self.tf_audit_received_ros = self.now_seconds()
            self.tf_authorities = {key: set(value) for key, value in self.tf_audit_record['tf_authorities'].items()}
            self.log('tf_audit', dict(received_ros=self.tf_audit_received_ros, **self.tf_audit_record))

        def spin(self, seconds=.05):
            end = time.monotonic()+seconds
            while time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=min(.05, max(0., end-time.monotonic())))

        def future(self, future, timeout=15.):
            end = time.monotonic()+timeout
            while not future.done() and time.monotonic() < end:
                self.spin()
            if not future.done():
                raise TimeoutError('ROS request timeout')
            return future.result()

        def wait_ready(self, launch_process, timeout):
            end = time.monotonic()+timeout
            while time.monotonic() < end:
                self.spin(.2)
                if launch_process.poll() is not None:
                    raise RuntimeError('launch_exited_before_ready')
                fresh = all(self.poses[k] and abs(self.now_seconds()-self.poses[k][-1]['t']) < .5
                            for k in ('estimates', 'truth', 'encoder'))
                if (fresh and self.counts['scan'] > 10 and self.counts['image'] > 5
                        and self.stats_latest and self.stats_latest['indices']
                        and self.tf_audit_record is not None
                        and self.diagnostics.get('quality_navigation/cost_mapping', {}).get('message') in ('windowed', 'cumulative')
                        and self.action.server_is_ready() and self.ready_client.service_is_ready()):
                    state = self.future(self.ready_client.call_async(GetState.Request()), 5.)
                    if state.current_state.id == 3:
                        self.spin(2.)
                        return
            raise TimeoutError('startup_timeout_no_independent_navigation_chain')

        def graph(self):
            topics = {}
            for topic in ('/ground_truth/odom', '/odom', '/localization/estimated_odom',
                          '/scan', '/camera/image_raw', '/localization_quality/quality_stats', '/tf', '/tf_static'):
                def describe(endpoint):
                    return dict(node=endpoint.node_name, namespace=endpoint.node_namespace,
                                type=endpoint.topic_type, gid=bytes(endpoint.endpoint_gid).hex())
                topics[topic] = dict(publishers=[describe(e) for e in self.get_publishers_info_by_topic(topic)],
                                     subscribers=[describe(e) for e in self.get_subscriptions_info_by_topic(topic)])
            topics['/benchmark/tf_authorities'] = dict(
                publishers=[describe(e) for e in self.get_publishers_info_by_topic('/benchmark/tf_authorities')], subscribers=[])
            return dict(nodes=[dict(name=n, namespace=ns) for n, ns in self.get_node_names_and_namespaces()],
                        topics=topics, tf_authorities={k: sorted(v) for k, v in self.tf_authorities.items()},
                        tf_audit_record=self.tf_audit_record,
                        tf_audit_age=self.now_seconds()-self.tf_audit_received_ros if self.tf_audit_received_ros is not None else None)

        def navigation_parameters(self):
            client = self.create_client(GetParameters, '/global_costmap/global_costmap/get_parameters')
            try:
                if not client.wait_for_service(timeout_sec=5.):
                    raise TimeoutError('global costmap parameter service unavailable')
                request = GetParameters.Request()
                request.names = ['quality_layer.enabled', 'plugins', 'robot_radius']
                response = self.future(client.call_async(request), 5.)
                enabled, plugins, radius = response.values
                return dict(quality_layer_enabled=enabled.bool_value if enabled.type == 1 else None,
                            plugins=list(plugins.string_array_value), robot_radius=radius.double_value)
            finally:
                self.destroy_client(client)

        def navigate(self, goal_xy, index, sim_timeout, wall_timeout, launch_process):
            goal = NavigateToPose.Goal()
            goal.pose.header.frame_id = 'map'
            goal.pose.header.stamp = self.get_clock().now().to_msg()
            goal.pose.pose.position.x, goal.pose.pose.position.y = map(float, goal_xy)
            # A fixed orientation is part of the mission, never computed from truth.
            goal.pose.pose.orientation.w = 1.
            start, wall_start = self.now_seconds(), time.monotonic()
            collisions_before = self.collision_count
            record = dict(index=index, goal=list(goal_xy), goal_yaw=0., mission_start=start,
                          action_status=0, timeout=False, termination='goal_rejected')
            self.log('events', dict(t=start, event='nav2_goal_requested', goal=list(goal_xy), index=index))
            handle = self.future(self.action.send_goal_async(goal))
            if handle.accepted:
                result = handle.get_result_async()
                while not result.done():
                    self.spin()
                    if (self.now_seconds()-start > sim_timeout or time.monotonic()-wall_start > wall_timeout
                            or launch_process.poll() is not None):
                        record['timeout'] = True
                        record['termination'] = 'mission_timeout' if launch_process.poll() is None else 'launch_exited'
                        self.future(handle.cancel_goal_async(), 10.)
                        break
                response = self.future(result, 15.)
                record['action_status'] = int(response.status)
                if not record['timeout']:
                    record['termination'] = 'action_finished'
            record.update(mission_end=self.now_seconds(), wall_seconds=time.monotonic()-wall_start,
                          collision_count=self.collision_count-collisions_before)
            self.log('events', dict(t=record['mission_end'], event='nav2_goal_finished', **record))
            return record

        def close(self):
            for stream in self.files.values():
                stream.flush()
                stream.close()
            self.destroy_node()

    return Recorder()


def audit_graph(graph):
    failures = []
    topics = graph['topics']
    for endpoint in topics['/ground_truth/odom']['subscribers']:
        name = endpoint['node']
        if name not in ('sensor_bridge', 'benchmark_runner') and not name.startswith('rosbag2_recorder'):
            failures.append('truth_subscribed_by_'+name)
    nodes = {item['name'] for item in graph['nodes']}
    for forbidden in ('localization_proxy', 'route_driver', 'quality_simulator', 'quality_navigation_simulator'):
        if forbidden in nodes:
            failures.append('forbidden_node_'+forbidden)
    publishers = {e['gid']: e['node'] for topic in ('/tf', '/tf_static') for e in topics[topic]['publishers']}
    for pair, expected in (('map->odom', 'amcl'), ('odom->base_footprint', 'diff_drive')):
        gids = graph['tf_authorities'].get(pair, [])
        owners = [publishers.get(gid, 'unresolved') for gid in gids]
        if len(owners) != 1 or owners[0] != expected:
            failures.append(pair+'_authority_'+','.join(owners))
    for topic, expected in (
            ('/ground_truth/odom', 'ground_truth'), ('/odom', 'diff_drive'),
            ('/localization/estimated_odom', 'localization_adapter'),
            ('/localization_quality/quality_stats', 'realtime_evaluator'),
            ('/scan', 'sensor_bridge'), ('/camera/image_raw', 'sensor_bridge')):
        owners = topics.get(topic, {}).get('publishers', [])
        if len(owners) != 1 or owners[0]['node'] != expected or owners[0].get('namespace', '/') != '/':
            failures.append(topic+'_publisher_not_unique_'+expected)
    audit_owners = topics.get('/benchmark/tf_authorities', {}).get('publishers', [])
    if len(audit_owners) != 1 or audit_owners[0]['node'] != 'tf_authority_audit':
        failures.append('tf_audit_publisher_not_unique')
    audit_age = graph.get('tf_audit_age')
    if audit_age is None or not math.isfinite(audit_age) or not 0. <= audit_age <= 1.:
        failures.append('tf_authority_audit_missing_or_stale')
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map-id', required=True)
    parser.add_argument('--condition', choices=('normal', 'visual', 'laser', 'combined'), default='normal')
    parser.add_argument('--method', choices=METHODS, default='geometric')
    parser.add_argument('--phase', choices=('smoke', 'calibration', 'test', 'dynamic'), default='smoke')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run-id', help='Unique frozen campaign run identifier; default is the output directory name.')
    parser.add_argument('--ros-domain', type=int, default=78)
    parser.add_argument('--gazebo-port', type=int, default=11378)
    parser.add_argument('--startup-timeout', type=float, default=180.)
    parser.add_argument('--sim-timeout', type=float, default=None,
                        help='Total mission budget; default is the frozen map reference-path budget.')
    parser.add_argument('--wall-timeout', type=float, default=None,
                        help='Total mission wall limit; default max(600,4*source-time budget).')
    parser.add_argument('--one-way', action='store_true', help='Smoke only; formal missions revisit the start.')
    parser.add_argument('--no-bag', action='store_true', help='Smoke only; formal trials preserve raw sensor bags.')
    args = parser.parse_args(argv)
    if args.phase == 'dynamic':
        parser.error('Dynamic event orchestration is not implemented yet; a static run cannot count as dynamic evidence.')
    for key in ('startup_timeout', 'sim_timeout', 'wall_timeout'):
        value = getattr(args, key)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(key+' must be finite and positive')
    if not 0 <= args.ros_domain <= 232 or not 1024 <= args.gazebo_port <= 65535:
        parser.error('ROS domain or Gazebo port is outside the supported isolated range.')
    if args.phase != 'smoke' and (args.one_way or args.no_bag):
        parser.error('Formal trials require the complete mission and raw sensor bag.')
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be empty; failed trial evidence is never overwritten.')
    output.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or output.name
    os.environ.update(ROS_DOMAIN_ID=str(args.ros_domain), ROS_LOCALHOST_ONLY='1',
                      GAZEBO_MASTER_URI='http://127.0.0.1:'+str(args.gazebo_port),
                      GAZEBO_MODEL_DATABASE_URI='', ROS_LOG_DIR=str(output/'ros_logs'))
    from ament_index_python.packages import get_package_share_directory
    import rclpy
    share = Path(get_package_share_directory('quality_localization_benchmark'))
    world = share/'worlds'/args.map_id
    metadata = json.loads((world/'metadata.json').read_text())
    if args.phase == 'calibration' and metadata['split'] != 'development':
        parser.error('Calibration cannot use the held-out test maps.')
    if args.phase in ('test', 'dynamic') and metadata['split'] != 'test':
        parser.error('Test phases require held-out test maps.')
    start = metadata['start']
    planned_waypoints = (metadata['calibration_waypoints'] if args.phase == 'calibration'
                         else metadata['mission_waypoints'])
    if args.one_way:
        planned_waypoints = planned_waypoints[:1]
    budget_key = 'calibration_time_limit_sec' if args.phase == 'calibration' else 'mission_time_limit_sec'
    sim_budget = args.sim_timeout if args.sim_timeout is not None else metadata[budget_key]
    wall_budget = args.wall_timeout if args.wall_timeout is not None else max(600., 4*sim_budget)
    manifest = dict(schema_version=1, run_id=run_id, phase=args.phase, map_id=args.map_id,
        seed=args.seed, condition=args.condition, method=args.method, estimator='AMCL laser localization',
        seed_scope='Gazebo --seed and sensor timestamp-indexed noise; AMCL random seed not exposed by stock Humble',
        initial_pose=[start[0]+.2, start[1]+.1, start[2]+.05], initial_sigma_xy=.3,
        initial_sigma_yaw=.15, spawn_pose=start, frame_transform=dict(x=0., y=0., yaw=0.),
        planned_waypoints=planned_waypoints, mission_time_limit_sec=sim_budget,
        mission_wall_time_limit_sec=wall_budget,
        collision_definition='conservative circular footprint intersects exact SDF static rectangles',
        metadata=metadata, arguments=vars(args), created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        software_environment=software_environment(),
        provenance=source_provenance(share), inputs_sha256={str(p.relative_to(share)): sha256(p)
            for p in [world/'world.sdf', world/'map.pgm', world/'map.yaml', world/'metadata.json',
                      world/(args.condition+'.json'), share/'robot/research_robot.urdf.xacro',
                      share/'config/nav2_localization.yaml', share/'launch/benchmark.launch.py']+
                      sorted((share/'config').glob('*.yaml'))})
    write_json(output/'manifest.json', manifest)
    launch_args = dict(world=world/'world.sdf', map=world/'map.yaml',
        robot=share/'robot/research_robot.urdf.xacro', zones_json=world/(args.condition+'.json'),
        sensor_seed=args.seed, gazebo_seed=args.seed, cost_mode=args.method, module_one_csv=output/'module_one.csv',
        spawn_x=start[0], spawn_y=start[1], spawn_yaw=start[2],
        initial_x=start[0]+.2, initial_y=start[1]+.1, initial_yaw=start[2]+.05,
        quality_map_resolution=metadata['quality_resolution'], quality_map_width=metadata['quality_width'],
        quality_map_height=metadata['quality_height'], quality_map_origin_x=metadata['origin'][0],
        quality_map_origin_y=metadata['origin'][1])
    launch_command = ['ros2', 'launch', 'quality_localization_benchmark', 'benchmark.launch.py']+[
        key+':='+str(value) for key, value in launch_args.items()]
    write_json(output/'launch_command.json', launch_command)
    recorder = launch_process = bag_process = None
    logs, legs, invalid = [], [], []
    graph = None
    navigation_parameters = {}
    bridge_before = bridge_after = {}
    bridge_before_record = bridge_after_record = {}
    mission_start = mission_end = None
    mission_wall_start = mission_wall_end = None
    wall_start = time.monotonic()
    failure = None
    try:
        rclpy.init(args=[])
        recorder = create_recorder(output, metadata)
        launch_log = (output/'launch.log').open('w', encoding='utf-8'); logs.append(launch_log)
        launch_process = subprocess.Popen(launch_command, stdout=launch_log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
        if not args.no_bag:
            bag_log = (output/'bag.log').open('w', encoding='utf-8'); logs.append(bag_log)
            bag_process = subprocess.Popen(['ros2', 'bag', 'record', '-o', str(output/'bag'),
                '--compression-mode', 'file', '--compression-format', 'zstd',
                '--qos-profile-overrides-path', str(share/'config/recording_qos.yaml')]+BAG_TOPICS,
                stdout=bag_log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(event='starting', run_id=run_id)), flush=True)
        recorder.wait_ready(launch_process, args.startup_timeout)
        graph = recorder.graph(); write_json(output/'graph_ready.json', graph)
        invalid.extend(audit_graph(graph))
        navigation_parameters = recorder.navigation_parameters()
        write_json(output/'navigation_parameters.json', navigation_parameters)
        if navigation_parameters['quality_layer_enabled'] is not (args.method != 'geometric'):
            invalid.append('navigation_quality_layer_mode_mismatch')
        for name, entry in manifest['provenance'].items():
            if isinstance(entry, dict) and 'source_matches_runtime' in entry and not entry['source_matches_runtime']:
                invalid.append('source_runtime_mismatch_'+name)
        if not args.no_bag and bag_process.poll() is not None:
            invalid.append('bag_recorder_exited')
        bridge_name = 'benchmark/sensor_measurement_generation'
        bridge_before_record = recorder.diagnostics.get(bridge_name, {}).copy()
        bridge_before = bridge_before_record.get('values', {}).copy()
        print(json.dumps(dict(event='ready', run_id=run_id, clock=recorder.now_seconds(), invalid=invalid)), flush=True)
        if invalid:
            raise RuntimeError('architecture_or_recording_invalid')
        recorder.mission = True
        mission_start = recorder.now_seconds()
        mission_wall_start = time.monotonic()
        for index, goal in enumerate(planned_waypoints):
            remaining_sim = sim_budget-(recorder.now_seconds()-mission_start)
            remaining_wall = wall_budget-(time.monotonic()-mission_wall_start)
            if min(remaining_sim, remaining_wall) <= 0:
                break
            leg = recorder.navigate(goal, index, remaining_sim, remaining_wall, launch_process)
            legs.append(leg)
            print(json.dumps(dict(event='leg_finished', run_id=run_id, **leg)), flush=True)
            if not action_completed_within_deadline(leg):
                break
            recorder.spin(.3)
        mission_end = recorder.now_seconds()
        mission_wall_end = time.monotonic()
        recorder.mission = False
        # Drain delayed estimates/CSV/diagnostics. Evaluation still ends at the
        # fixed mission end; waiting does not extend a successful trajectory.
        recorder.spin(2.)
        drain_deadline = time.monotonic()+10.
        while time.monotonic() < drain_deadline and any(
                not recorder.poses[key] or recorder.poses[key][-1]['t'] < mission_end-.05
                for key in ('truth', 'encoder', 'estimates')):
            recorder.spin(.05)
        bridge_after_record = recorder.diagnostics.get(bridge_name, {}).copy()
        bridge_after = bridge_after_record.get('values', {}).copy()
        graph = recorder.graph(); write_json(output/'graph_finished.json', graph)
        invalid.extend(audit_graph(graph))
        if not args.no_bag and bag_process.poll() is not None:
            invalid.append('bag_recorder_exited_during_mission')
    except Exception as error:
        failure = type(error).__name__+': '+str(error)
        (output/'exception.txt').write_text(traceback.format_exc(), encoding='utf-8')
        invalid.append('infrastructure_exception')
        if recorder is not None and mission_start is not None:
            mission_end = recorder.now_seconds()
            mission_wall_end = time.monotonic()
            recorder.mission = False
    finally:
        bag_returncode = stop_process(bag_process)
        launch_returncode = stop_process(launch_process)
        for stream in logs:
            stream.close()
    execution_wall_seconds = time.monotonic()-wall_start
    wall_seconds = mission_wall_end-mission_wall_start if mission_wall_end is not None else None
    # Flush recorder sidecars before the offline readers below inspect them.
    if recorder:
        for stream in recorder.files.values():
            stream.flush()
    quality_rows = []
    if (output/'module_one.csv').is_file():
        with (output/'module_one.csv').open(newline='', encoding='utf-8') as stream:
            quality_rows = list(csv.DictReader(stream))
    if mission_start is None or mission_end is None:
        invalid.append('no_complete_mission_interval')
    if not quality_rows:
        invalid.append('actual_module_one_quality_missing')
    deltas = counter_delta(bridge_before, bridge_after)
    if any(value is None or value != 0 for value in deltas.values()):
        invalid.append('measurement_bridge_unplanned_passthrough_or_missing_diagnostics')
    def diagnostic_covers_mission():
        try:
            before_t, after_t = bridge_before_record['t'], bridge_after_record['t']
            return (mission_start is not None and mission_end is not None and
                -.05 <= mission_start-before_t <= 1. and
                0. <= after_t-mission_end <= 3. and after_t > before_t and
                int(bridge_after['scans']) > int(bridge_before['scans']) and
                int(bridge_after['images']) > int(bridge_before['images']))
        except (KeyError, TypeError, ValueError):
            return False
    if not diagnostic_covers_mission():
        invalid.append('measurement_bridge_diagnostics_do_not_cover_mission')
    if not args.no_bag and not (output/'bag/metadata.yaml').is_file():
        invalid.append('raw_bag_metadata_missing')
    estimates = recorder.poses['estimates'] if recorder else []
    truth = recorder.poses['truth'] if recorder else []
    motion_audit = truth_motion_audit(truth, mission_start, mission_end, metadata['rectangles'], metadata['robot_collision_radius'])
    if motion_audit['unstable_samples']:
        invalid.append('robot_unstable_or_out_of_plane')
    if recorder:
        for stream, frame in (('truth', 'world'), ('estimates', 'map'), ('encoder', 'odom')):
            if recorder.frame_ids[stream] != {frame}:
                invalid.append('unexpected_'+stream+'_coordinate_frame')
    for leg in legs:
        leg['collision_count'] = truth_motion_audit(truth, leg['mission_start'], leg['mission_end'],
            metadata['rectangles'], metadata['robot_collision_radius'])['collision_count']
    overall_manifest = dict(manifest, mission_start=mission_start, mission_end=mission_end,
        goal=planned_waypoints[-1],
        action_status=4 if len(legs) == len(planned_waypoints) and all(action_completed_within_deadline(l) for l in legs) else 0,
        collision_count=motion_audit['collision_count'], wall_seconds=wall_seconds)
    result = evaluate_episode(overall_manifest, estimates, truth, quality_rows)
    evaluated_legs = [evaluate_episode(dict(manifest, **dict(leg,
        action_status=leg['action_status'] if action_completed_within_deadline(leg) else 0)),
        estimates, truth, quality_rows) for leg in legs]
    # Missing AMCL estimates are algorithm failures, not a reason to remove a
    # trial from the success denominator. Missing independent truth is a recording
    # failure; it prevents evaluating the trial itself.
    if result['truth_coverage'] is None or result['truth_coverage'] < .95-1.e-9:
        invalid.append('independent_truth_coverage_insufficient')
    truth_input = result.get('input_statistics', {}).get('truth', {})
    if truth_input.get('time_reversals', 0) or truth_input.get('reset_events', 0):
        invalid.append('independent_truth_clock_discontinuity')
    result.update(phase=args.phase, completed=True, exception=failure, invalid_reasons=sorted(set(invalid)),
        valid_evidence=not invalid, pose_metrics_valid=result['evaluation_valid'], legs=evaluated_legs,
        leg_actions=legs, bag_recorded=not args.no_bag, bag_returncode=bag_returncode,
        launch_returncode=launch_returncode, bridge_counter_delta=deltas,
        navigation_parameters=navigation_parameters,
        execution_wall_seconds=execution_wall_seconds,
        truth_motion_audit=motion_audit,
        bridge_diagnostic_envelope=dict(before=bridge_before_record, after=bridge_after_record,
            semantics='conservative diagnostic envelope includes adjacent startup/drain time; faults invalidate, never omitted'),
        message_counts=dict(recorder.counts) if recorder else {},
        frame_ids={k: sorted(v) for k, v in recorder.frame_ids.items()} if recorder else {},
        diagnostics_last=recorder.diagnostics if recorder else {},
        collision_definition=manifest['collision_definition'])
    result['cost_consumption'] = cost_consumption_summary(read_jsonl(output/'diagnostics.jsonl'), mission_start, mission_end)
    if mission_start is not None and mission_end is not None and mission_end > mission_start:
        quality_coverage = summarize_quality_coverage(read_jsonl(output/'quality_stats.jsonl'),
            read_jsonl(output/'plans.jsonl'), estimates, truth, metadata, mission_start, mission_end)
        write_json(output/'quality_coverage.json', quality_coverage)
    result['sensor_fault_counter_delta'] = {}
    for key in ('scans', 'images', 'faulted_scans', 'faulted_images'):
        try:
            result['sensor_fault_counter_delta'][key] = int(bridge_after[key])-int(bridge_before[key])
        except (KeyError, ValueError, TypeError):
            result['sensor_fault_counter_delta'][key] = None
    mission_quality = []
    if mission_start is not None and mission_end is not None:
        for row in quality_rows:
            try:
                if mission_start <= float(row['timestamp']) <= mission_end:
                    mission_quality.append(row)
            except (KeyError, ValueError, TypeError):
                pass
    result['actual_module_one_mission_rows'] = len(mission_quality)
    if not mission_quality:
        result['warnings'].append('no_module_one_quality_during_mission; quality-guided benefit cannot be attributed')
    result['navigation_success'] = result['navigation_success'] and bool(evaluated_legs) and all(
        leg['navigation_success'] for leg in evaluated_legs)
    result['success'] = result['navigation_success'] and result['valid_evidence'] and result['pose_metrics_valid']
    if any(leg.get('timeout') for leg in legs):
        result['failure_reasons'].append('mission_deadline_exceeded')
    # Bulky per-frame data have separate files; the summary stays quick to read.
    for name in ('error_rows', 'quality_pairs'):
        rows = result.pop(name)
        with (output/(name+'.jsonl')).open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n')
        if name == 'quality_pairs':
            write_json(output/'quality_relation.json', summarize_quality_relation(rows))
    for leg in result['legs']:
        leg.pop('error_rows', None); leg.pop('quality_pairs', None)
    write_json(output/'result.json', result)
    if recorder:
        recorder.close()
    if rclpy.ok():
        rclpy.shutdown()
    print(json.dumps(dict(event='finished', output=str(output), success=result['success'],
        valid_evidence=result['valid_evidence'], failure_reasons=result['failure_reasons'],
        invalid_reasons=result['invalid_reasons'], position_error=result['position_error'])), flush=True)
    return 0 if result['valid_evidence'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
