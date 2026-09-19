"""Real DDS /clock ordering tests; run in a sourced ROS2 Humble workspace.

Uses an isolated domain and actual publishers/executor/steady timers. No robot
commands or Gazebo are needed. Clock pauses and rewinds are intentional inputs.
"""
import time
import pytest

rclpy = pytest.importorskip('rclpy')
pytest.importorskip('quality_navigation_msgs.msg')
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rosgraph_msgs.msg import Clock
from nav2_msgs.msg import Costmap
from builtin_interfaces.msg import Time
from quality_navigation_msgs.msg import QualityGrid
from quality_aware_navigation.ros_node import QualityCostNode


def ros_stamp(value):
    ns = round(value * 1e9)
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def snapshot(value, mean=.9):
    msg = QualityGrid()
    msg.header.frame_id = 'map'; msg.header.stamp = ros_stamp(value)
    msg.info.width = 1; msg.info.height = 1; msg.info.resolution = .2
    msg.info.origin.orientation.w = 1.
    msg.statistics_mode = 'windowed'
    msg.indices = [0]; msg.mean = [mean]; msg.variance = [.001]
    msg.sample_count = [20]; msg.last_observed = [ros_stamp(value)]
    return msg


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.setenv('ROS_DOMAIN_ID', '75')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path / 'ros_logs'))
    rclpy.init(args=['--ros-args', '--log-level', 'error', '-p', 'use_sim_time:=true',
                    '-p', 'publish_period:=0.05', '-p', 'pending_wall_timeout:=1.0'])
    node = QualityCostNode()
    driver = Node('clock_ordering_test_driver')
    executor = SingleThreadedExecutor()
    executor.add_node(node); executor.add_node(driver)
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
    clock = driver.create_publisher(Clock, '/clock', 10)
    source = driver.create_publisher(QualityGrid, '/localization_quality/quality_stats', qos)
    outputs = []
    driver.create_subscription(Costmap, '/localization_quality/navigation_cost', outputs.append, qos)

    def wait(predicate, timeout=3.):
        until = time.monotonic() + timeout
        while not predicate() and time.monotonic() < until:
            executor.spin_once(timeout_sec=.01)
        assert predicate(), 'Timed out waiting for a real ROS/DDS event'

    def spin(seconds):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            executor.spin_once(timeout_sec=.01)

    def set_clock(value):
        clock.publish(Clock(clock=ros_stamp(value)))
        wait(lambda: abs(node.get_clock().now().nanoseconds * 1e-9 - value) < 1e-6)
        spin(.035)

    try:
        wait(lambda: clock.get_subscription_count() >= 2 and source.get_subscription_count() >= 1)
        set_clock(100.)
        yield node, source, set_clock, wait, spin, outputs
    finally:
        executor.remove_node(node); executor.remove_node(driver)
        node.destroy_node(); driver.destroy_node(); executor.shutdown()
        rclpy.try_shutdown()


def test_dds_near_future_waits_for_clock_before_cost_is_eligible(runtime):
    node, source, clock, wait, spin, outputs = runtime
    source.publish(snapshot(99.9, .2)); wait(lambda: node.accepted == 1)
    received = node.last_receive
    source.publish(snapshot(100.05, .9)); wait(lambda: node.pending is not None)
    assert node.accepted == 1 and node.rejected == 0
    assert node.last_receive == received
    clock(100.02)
    assert node.source_stamp == pytest.approx(99.9)
    assert node.deferred_accepted == 0
    pending_received = node.pending[2]
    clock(100.05); wait(lambda: node.accepted == 2)
    assert node.last_receive == pending_received
    assert node.grid.last_observed[0, 0] == pytest.approx(100.05)
    clock(100.1); wait(lambda: any(m.header.stamp.sec == 100 and m.header.stamp.nanosec >= 50_000_000 for m in outputs))
    assert node.rejected == 0 and node.deferred_accepted == 1
    # A future input never changes the model/grid before /clock reaches it.
    for output in outputs:
        t = output.header.stamp.sec + output.header.stamp.nanosec * 1e-9
        if t < 100.05:
            assert output.data[0] > 140  # older .2-quality observation


def test_dds_far_future_is_rejected_immediately(runtime):
    node, source, clock, wait, spin, outputs = runtime
    source.publish(snapshot(100.25)); wait(lambda: node.rejected == 1)
    assert node.pending is None and node.accepted == 0
    clock(100.3); spin(.05)
    assert node.grid is None and not outputs


def test_dds_paused_clock_expires_pending_on_steady_time(runtime):
    node, source, clock, wait, spin, outputs = runtime
    source.publish(snapshot(99.9)); wait(lambda: node.accepted == 1)
    received = node.last_receive
    source.publish(snapshot(100.05)); wait(lambda: node.pending is not None)
    wait(lambda: node.pending_expired == 1, timeout=2.)
    assert node.get_clock().now().nanoseconds == 100_000_000_000
    assert node.pending is None and node.source_stamp == pytest.approx(99.9)
    assert node.last_receive == received and node.input_invalid
    clock(100.1); wait(lambda: outputs and min(outputs[-1].data) >= 200)


def test_dds_rewind_discards_pending_and_does_not_replay_after_jump(runtime):
    node, source, clock, wait, spin, outputs = runtime
    source.publish(snapshot(100.05)); wait(lambda: node.pending is not None)
    clock(1.); wait(lambda: node.pending_rewind_dropped == 1)
    clock(101.); spin(.1)
    assert node.grid is None and node.accepted == 0 and not outputs
    source.publish(snapshot(101.)); wait(lambda: node.accepted == 1)
    assert node.source_stamp == 101. and not node.input_invalid
