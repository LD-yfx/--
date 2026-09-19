"""Gazebo measurement forwarding and preregistered sensor faults.

No quality score, localization error, region label or pose is published. Optional
ground truth is used ONLY inside measurement generation to select a physical
environment region, analogous to the simulator using pose to render a camera.
AMCL, module one and Nav2 must only receive the resulting sensor messages.

JSON schema (world coordinates, half-open spatial/time intervals):
{
  "version": 1, "seed": 42,
  "zones": [{"name": "corridor", "bounds": [-2, 2, -1, 1],
             "laser_sigma": 0.03, "dropout": 0.1,
             "visual_blur": 5, "exposure": 0.6}],
  "events": [{"zone": "corridor", "start": 20.0, "end": 40.0,
              "laser_sigma": 0.08, "dropout": 0.3}]
}
Events override the named zone's listed fields during ROS simulation seconds
[start,end) relative to event_epoch (default zero preserves absolute timestamps).
The optional Trigger /benchmark/sensor_bridge/start_events sets that epoch once;
repeat requests return the original epoch. With require_event_start=true, event
overrides stay disabled until triggered; baseline zone parameters still apply.
Zones must not overlap; events for the same zone must not
overlap. Unspecified values are sigma=0, dropout=0, blur=0, exposure=1. Blur is
zero or a positive odd Gaussian kernel size. Exposure multiplies pixel values.
No JSON file / no zones means transparent forwarding without a truth subscriber.

The seed parameter defaults to -1 (use JSON seed, or 42 without a file). Noise
and dropout use separate hashes of seed, sensor stamp and component name; there
is no mutable RNG sequence whose advancement could depend on the selected route.
"""
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


FAULT_FIELDS = frozenset(('laser_sigma', 'dropout', 'visual_blur', 'exposure'))


@dataclass(frozen=True)
class FaultParameters:
    laser_sigma: float = 0.0
    dropout: float = 0.0
    visual_blur: int = 0
    exposure: float = 1.0

    def __post_init__(self):
        if not all(math.isfinite(float(v)) for v in
                   (self.laser_sigma, self.dropout, self.exposure)):
            raise ValueError('fault parameters must be finite')
        if self.laser_sigma < 0 or not 0 <= self.dropout <= 1 or not 0 <= self.exposure <= 16:
            raise ValueError('require sigma>=0, dropout in [0,1], exposure in [0,16]')
        if isinstance(self.visual_blur, bool) or not isinstance(self.visual_blur, int):
            raise ValueError('visual_blur must be an integer Gaussian kernel size')
        if self.visual_blur < 0 or self.visual_blur > 101 or (
                self.visual_blur > 0 and self.visual_blur % 2 == 0):
            raise ValueError('visual_blur must be zero or an odd integer up to 101')

    @property
    def laser_passthrough(self):
        return self.laser_sigma == 0 and self.dropout == 0

    @property
    def image_passthrough(self):
        return self.visual_blur <= 1 and self.exposure == 1


def with_fault_fields(base, values):
    fields = {name: getattr(base, name) for name in FAULT_FIELDS}
    fields.update({name: value for name, value in values.items() if name in FAULT_FIELDS})
    return FaultParameters(**fields)


class EnvironmentModel:
    def __init__(self, configuration=None, seed_override=None):
        configuration = {} if configuration is None else configuration
        if not isinstance(configuration, dict):
            raise ValueError('sensor environment must be a JSON object')
        if set(configuration) - {'version', 'seed', 'zones', 'events'}:
            raise ValueError('unrecognized sensor environment field')
        if configuration.get('version', 1) != 1:
            raise ValueError('unsupported sensor environment version')
        self.seed = configuration.get('seed', 42) if seed_override is None else seed_override
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError('seed must be a nonnegative signed 64-bit integer')
        self.zones = []
        self.events = []
        names = set()
        for entry in configuration.get('zones', []):
            if not isinstance(entry, dict) or set(entry) - ({'name', 'bounds'} | FAULT_FIELDS):
                raise ValueError('invalid zone fields')
            name = entry.get('name')
            bounds = entry.get('bounds', [])
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError('zone names must be nonempty and unique')
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 4 or not all(math.isfinite(float(v)) for v in bounds):
                raise ValueError('zone bounds must contain four finite coordinates')
            bounds = tuple(float(v) for v in bounds)
            if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
                raise ValueError('zone bounds must be [xmin,xmax,ymin,ymax] with positive area')
            for previous in self.zones:
                other = previous['bounds']
                if max(bounds[0], other[0]) < min(bounds[1], other[1]) and \
                        max(bounds[2], other[2]) < min(bounds[3], other[3]):
                    raise ValueError('overlapping zones are ambiguous')
            names.add(name)
            self.zones.append({'name': name, 'bounds': bounds,
                               'fault': with_fault_fields(FaultParameters(), entry)})
        for entry in configuration.get('events', []):
            if not isinstance(entry, dict) or set(entry) - ({'zone', 'start', 'end'} | FAULT_FIELDS):
                raise ValueError('invalid event fields')
            zone, start, end = entry.get('zone'), entry.get('start'), entry.get('end')
            if zone not in names or start is None or end is None:
                raise ValueError('event requires an existing zone, start and end')
            start, end = float(start), float(end)
            if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
                raise ValueError('event requires finite 0 <= start < end')
            for previous in self.events:
                if previous['zone'] == zone and max(start, previous['start']) < min(end, previous['end']):
                    raise ValueError('overlapping events for the same zone are ambiguous')
            fields = {name: value for name, value in entry.items() if name in FAULT_FIELDS}
            with_fault_fields(FaultParameters(), fields)  # validate even before the event is active
            self.events.append({'zone': zone, 'start': start, 'end': end, 'fields': fields})

    @classmethod
    def load(cls, filename='', seed_override=None):
        data = json.loads(Path(filename).read_text(encoding='utf-8')) if filename else {}
        return cls(data, seed_override)

    @property
    def needs_truth(self):
        return bool(self.zones)

    def parameters_at(self, x, y, timestamp, *, event_epoch=0.0, events_enabled=True):
        if not all(math.isfinite(float(v)) for v in (x, y, timestamp)) or timestamp < 0:
            raise ValueError('environment lookup requires finite position and nonnegative sensor time')
        if not math.isfinite(event_epoch) or event_epoch < 0:
            raise ValueError('event epoch must be finite and nonnegative')
        event_time=timestamp-event_epoch
        for zone in self.zones:
            xmin, xmax, ymin, ymax = zone['bounds']
            if xmin <= x < xmax and ymin <= y < ymax:
                fault = zone['fault']
                for event in self.events:
                    if events_enabled and event['zone'] == zone['name'] and event['start'] <= event_time < event['end']:
                        fault = with_fault_fields(fault, event['fields'])
                return fault
        return FaultParameters()


class EventEpoch:
    """One-shot simulation schedule gate; repeated starts never refresh time."""
    def __init__(self, require_start=False):
        self.require_start=bool(require_start)
        self.epoch=None if self.require_start else 0.0
        self.started=False
        self.clock_invalid=False

    def start(self, timestamp):
        if not math.isfinite(timestamp) or timestamp<0:
            raise ValueError('event start needs a finite nonnegative source time')
        if self.clock_invalid:
            raise ValueError('event epoch invalidated by clock rewind')
        if not self.started:
            self.epoch=float(timestamp)
            self.started=True
        return self.epoch

    @property
    def enabled(self):
        return not self.clock_invalid and (not self.require_start or self.started)

    def clock_rewound(self):
        # Do not silently replay a dynamic event after an epoch discontinuity.
        if self.started or self.require_start:
            self.clock_invalid=True


def component_seed(seed, timestamp_ns, component):
    if not isinstance(timestamp_ns, int) or timestamp_ns < 0:
        raise ValueError('sensor timestamp must be a nonnegative integer nanosecond count')
    payload = f'{seed}:{timestamp_ns}:{component}'.encode('utf-8')
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'little')


def apply_laser_fault(ranges, range_min, range_max, fault, seed, timestamp_ns):
    if not math.isfinite(range_min) or not math.isfinite(range_max) or not 0 <= range_min < range_max:
        raise ValueError('laser range limits must be finite and ordered')
    output = np.asarray(ranges, dtype=np.float64).copy()
    valid = np.isfinite(output) & (output >= range_min) & (output <= range_max)
    if fault.laser_sigma > 0:
        generator = np.random.default_rng(component_seed(seed, timestamp_ns, 'laser_noise'))
        noise = generator.normal(0.0, fault.laser_sigma, size=output.shape)
        output[valid] = np.clip(output[valid] + noise[valid], range_min, range_max)
    if fault.dropout > 0:
        generator = np.random.default_rng(component_seed(seed, timestamp_ns, 'laser_dropout'))
        dropped = generator.random(output.shape) < fault.dropout
        output[valid & dropped] = np.inf
    return output


def apply_image_fault(image, fault):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError('configured image faults require an 8-bit image')
    output = image.copy()
    if fault.visual_blur > 1:
        import cv2
        output = cv2.GaussianBlur(output, (fault.visual_blur, fault.visual_blur), 0)
    if fault.exposure != 1:
        output = np.floor(np.clip(output.astype(np.float64) * fault.exposure, 0, 255) + 0.5).astype(np.uint8)
    return output


class TruthHistory:
    """Bounded causal lookup; an earlier epoch clears the measurement cache."""
    def __init__(self, max_age_sec=0.2, capacity=1000):
        if not math.isfinite(max_age_sec) or max_age_sec < 0 or capacity < 1:
            raise ValueError('invalid truth history limits')
        self.max_age_ns = int(max_age_sec * 1e9)
        self.entries = deque(maxlen=capacity)

    def clear(self):
        self.entries.clear()

    def push(self, timestamp_ns, x, y):
        if not isinstance(timestamp_ns, int) or timestamp_ns < 0 or not all(map(math.isfinite, (x, y))):
            raise ValueError('invalid truth observation')
        rewound = bool(self.entries and timestamp_ns < self.entries[-1][0])
        if rewound:
            self.clear()
        if self.entries and timestamp_ns == self.entries[-1][0]:
            self.entries.pop()
        self.entries.append((timestamp_ns, x, y))
        return rewound

    def covers(self, timestamp_ns):
        return bool(self.entries and self.entries[-1][0] >= timestamp_ns)

    def lookup(self, timestamp_ns):
        if not self.entries:
            return None
        entries = list(self.entries)
        index = bisect_right([entry[0] for entry in entries], timestamp_ns) - 1
        if index < 0 or timestamp_ns - entries[index][0] > self.max_age_ns:
            return None
        return entries[index][1], entries[index][2]


def create_node():
    import copy
    import rclpy
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from cv_bridge import CvBridge
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image, LaserScan, CameraInfo
    from std_srvs.srv import Trigger

    class SensorBridge(Node):
        def __init__(self):
            super().__init__('benchmark_sensor_bridge')
            defaults = {'zones_json': '', 'seed': -1, 'truth_topic': '/ground_truth/odom',
                        'truth_frame': 'world', 'truth_max_age_sec': 0.2,
                        'truth_wait_timeout_sec': 0.5, 'pending_capacity': 128,
                        'require_event_start': False}
            for key, value in defaults.items():
                self.declare_parameter(key, value)
            values = {key: self.get_parameter(key).value for key in defaults}
            if values['seed'] < -1 or values['pending_capacity'] < 1 or \
                    not math.isfinite(values['truth_wait_timeout_sec']) or values['truth_wait_timeout_sec'] <= 0:
                raise ValueError('invalid sensor bridge seed or queue limits')
            self.environment = EnvironmentModel.load(
                values['zones_json'], None if values['seed'] == -1 else values['seed'])
            self.event_epoch=EventEpoch(values['require_event_start'])
            self.event_start_service=self.create_service(
                Trigger,'/benchmark/sensor_bridge/start_events',self.start_events)
            self.history = TruthHistory(values['truth_max_age_sec'])
            self.truth_frame = values['truth_frame']
            self.wait_timeout = values['truth_wait_timeout_sec']
            self.capacity = values['pending_capacity']
            self.pending = deque()
            self.last_stamps = {}
            self.bridge = CvBridge()
            self.state = 'normal_passthrough'
            self.counters = {'images': 0, 'scans': 0, 'faulted_images': 0, 'faulted_scans': 0,
                             'truth_missing_passthrough': 0, 'queue_overflow_passthrough': 0,
                             'measurement_error_passthrough': 0, 'invalid_stamp_passthrough': 0,
                             'invalid_inputs': 0, 'clock_rewinds': 0}
            self.publishers_by_kind = {
                'scan': self.create_publisher(LaserScan, '/scan', qos_profile_sensor_data),
                'image': self.create_publisher(Image, '/camera/image_raw', qos_profile_sensor_data)}
            self.info_publisher = self.create_publisher(CameraInfo, '/camera/camera_info', qos_profile_sensor_data)
            self.diag_publisher = self.create_publisher(DiagnosticArray, '/benchmark/sensor_bridge/diagnostics', 10)
            self.subscriptions_owned = [
                self.create_subscription(LaserScan, '/sim/scan', self.on_scan, qos_profile_sensor_data),
                self.create_subscription(Image, '/front_camera/image_raw', self.on_image, qos_profile_sensor_data),
                self.create_subscription(CameraInfo, '/front_camera/camera_info', self.info_publisher.publish,
                                         qos_profile_sensor_data)]
            if self.environment.needs_truth:
                self.subscriptions_owned.append(self.create_subscription(
                    Odometry, values['truth_topic'], self.on_truth, qos_profile_sensor_data))
            self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
            self.drain_timer = self.create_timer(0.05, self.drain, clock=self.steady_clock)
            self.diag_timer = self.create_timer(0.5, self.diagnostics, clock=self.steady_clock)
            self.get_logger().info('Sensor bridge: measurement generation only; no truth/zone labels in output')

        def start_events(self, request, response):
            del request
            repeated=self.event_epoch.started
            try:
                now=self.get_clock().now().nanoseconds*1e-9
                if now<=0:
                    raise ValueError('simulation clock not ready')
                epoch=self.event_epoch.start(now)
                response.success=True
                response.message=json.dumps(dict(epoch=epoch,idempotent=repeated,
                                                 events_enabled=self.event_epoch.enabled))
            except ValueError as error:
                response.success=False
                response.message=json.dumps(dict(error=str(error),epoch=self.event_epoch.epoch))
            self.diagnostics()
            return response

        @staticmethod
        def stamp_ns(stamp):
            if stamp.sec < 0 or not 0 <= stamp.nanosec < 1_000_000_000:
                raise ValueError('invalid sensor timestamp')
            return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

        def on_truth(self, message):
            try:
                if message.header.frame_id != self.truth_frame:
                    raise ValueError('unexpected truth coordinate frame')
                position = message.pose.pose.position
                if self.history.push(self.stamp_ns(message.header.stamp), position.x, position.y):
                    self.pending.clear()
                    self.last_stamps.clear()
                    self.counters['clock_rewinds'] += 1
                    self.event_epoch.clock_rewound()
                self.drain()
            except (ValueError, TypeError):
                self.history.clear()
                self.counters['invalid_inputs'] += 1
                self.state = 'invalid_truth_passthrough'

        def on_scan(self, message):
            self.enqueue('scan', message)

        def on_image(self, message):
            self.enqueue('image', message)

        def enqueue(self, kind, message):
            self.counters['scans' if kind == 'scan' else 'images'] += 1
            if not self.environment.needs_truth:
                self.publishers_by_kind[kind].publish(message)
                self.state = 'normal_passthrough'
                return
            try:
                stamp = self.stamp_ns(message.header.stamp)
            except ValueError:
                self.publishers_by_kind[kind].publish(message)
                self.counters['invalid_inputs'] += 1
                self.counters['invalid_stamp_passthrough'] += 1
                self.state = 'invalid_sensor_stamp_passthrough'
                return
            if kind in self.last_stamps and stamp < self.last_stamps[kind]:
                self.history.clear()
                self.pending.clear()
                self.last_stamps.clear()
                self.counters['clock_rewinds'] += 1
                self.event_epoch.clock_rewound()
            self.last_stamps[kind] = stamp
            if len(self.pending) >= self.capacity:
                previous_kind, previous_message, _, _ = self.pending.popleft()
                self.publishers_by_kind[previous_kind].publish(previous_message)
                self.counters['queue_overflow_passthrough'] += 1
            self.pending.append((kind, message, stamp, time.monotonic()))
            self.drain()

        def drain(self):
            retained = deque()
            ready = []
            wall_time = time.monotonic()
            while self.pending:
                kind, message, stamp, arrival = self.pending.popleft()
                if self.history.covers(stamp) or wall_time - arrival >= self.wait_timeout:
                    ready.append((kind, message, stamp, arrival))
                else:
                    retained.append((kind, message, stamp, arrival))
            self.pending = retained
            for kind, message, stamp, _ in sorted(ready, key=lambda item: (item[2], item[0])):
                position = self.history.lookup(stamp) if self.history.covers(stamp) else None
                if position is None:
                    self.publishers_by_kind[kind].publish(message)
                    self.counters['truth_missing_passthrough'] += 1
                    self.state = 'truth_unavailable_passthrough'
                    continue
                fault = self.environment.parameters_at(position[0], position[1], stamp * 1e-9,
                    event_epoch=self.event_epoch.epoch or 0.0,events_enabled=self.event_epoch.enabled)
                try:
                    if kind == 'scan' and not fault.laser_passthrough:
                        output = copy.deepcopy(message)
                        output.ranges = apply_laser_fault(
                            message.ranges, message.range_min, message.range_max,
                            fault, self.environment.seed, stamp).astype(np.float32).tolist()
                        self.counters['faulted_scans'] += 1
                    elif kind == 'image' and not fault.image_passthrough:
                        pixels = self.bridge.imgmsg_to_cv2(message, desired_encoding='passthrough')
                        output = self.bridge.cv2_to_imgmsg(apply_image_fault(pixels, fault), encoding=message.encoding)
                        output.header = message.header
                        self.counters['faulted_images'] += 1
                    else:
                        output = message
                    self.publishers_by_kind[kind].publish(output)
                    self.state = 'normal_passthrough' if output is message else 'configured_measurement_fault_applied'
                except Exception as error:  # cv_bridge/OpenCV have separate extension exception types
                    self.publishers_by_kind[kind].publish(message)
                    self.counters['invalid_inputs'] += 1
                    self.counters['measurement_error_passthrough'] += 1
                    self.state = 'measurement_fault_error_passthrough'
                    self.get_logger().warning('Measurement fault could not be applied: ' + type(error).__name__)

        def diagnostics(self):
            status = DiagnosticStatus()
            status.name = 'benchmark/sensor_measurement_generation'
            status.level = DiagnosticStatus.WARN if ('unavailable' in self.state or 'invalid' in self.state or
                                                     'error' in self.state) else DiagnosticStatus.OK
            status.message = self.state
            # Deliberately omit true pose, current zone, zone labels and fault parameters.
            values = {'truth_dependency': 'measurement_generation_only' if self.environment.needs_truth else 'none',
                      'event_epoch': self.event_epoch.epoch,
                      'events_started': self.event_epoch.started,
                      'require_event_start': self.event_epoch.require_start,
                      'events_clock_invalid': self.event_epoch.clock_invalid,
                      'sensor_frames_total': self.counters['images'] + self.counters['scans'],
                      'pending_measurements': len(self.pending), **self.counters}
            status.values = [KeyValue(key=key, value=str(value)) for key, value in values.items()]
            output = DiagnosticArray()
            output.header.stamp = self.get_clock().now().to_msg()
            output.status = [status]
            self.diag_publisher.publish(output)

    return SensorBridge()


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = None
    try:
        node = create_node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
