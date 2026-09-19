"""ROS message/adapter boundaries; ordinary Python skips optional ROS modules.

Run in a sourced Humble workspace with ROS_DOMAIN_ID=75. Tests use real ROS
messages and node initialization, but no spinning/robot commands or external peers.
"""
from types import SimpleNamespace
import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy', reason='ROS adapter tests require sourced ROS2 Humble')
pytest.importorskip('quality_navigation_msgs.msg', reason='Build/source quality_navigation_msgs first')
from builtin_interfaces.msg import Time as TimeMessage
from nav_msgs.msg import OccupancyGrid
from rclpy.time import Time
from quality_navigation_msgs.msg import QualityGrid as QualityGridMessage
from quality_aware_navigation import ros_node as adapter


def message(snapshot=100, observed=100):
    msg = QualityGridMessage()
    msg.header.frame_id = 'odom'
    msg.header.stamp.sec = snapshot
    msg.info.width = 2
    msg.info.height = 1
    msg.info.resolution = .5
    msg.info.origin.orientation.w = 1.
    msg.statistics_mode = 'windowed'
    msg.indices = [0]
    msg.mean = [.9]
    msg.variance = [.001]
    msg.sample_count = [20]
    msg.last_observed = [TimeMessage(sec=observed)]
    return msg


@pytest.mark.parametrize('z', [float('nan'), float('inf'), -float('inf'), .01, -.01])
def test_nonplanar_or_nonfinite_origin_rejected(z):
    msg = message()
    msg.info.origin.position.z = z
    with pytest.raises(ValueError, match='z=0'):
        adapter.parse_statistics(msg)


@pytest.mark.parametrize('nanoseconds', [1_000_000_000, 4_000_000_000])
def test_nanoseconds_must_be_normalized(nanoseconds):
    # Both values fit the message's uint32 field but violate ROS time semantics.
    with pytest.raises(ValueError, match='nanosec'):
        adapter.stamp_seconds(TimeMessage(sec=100, nanosec=nanoseconds))


@pytest.mark.parametrize('field', ['header', 'cell'])
def test_invalid_nanoseconds_rejected_from_complete_snapshot(field):
    msg = message()
    target = msg.header.stamp if field == 'header' else msg.last_observed[0]
    target.nanosec = 1_000_000_000
    with pytest.raises(ValueError, match='nanosec'):
        adapter.parse_statistics(msg)


def test_cell_observation_cannot_postdate_snapshot():
    msg = message()
    msg.last_observed[0].nanosec = 50_000
    with pytest.raises(ValueError, match='later'):
        adapter.parse_statistics(msg)


@pytest.mark.parametrize('corruption', ['length', 'duplicate', 'outside', 'negative_variance', 'no_samples'])
def test_malformed_sparse_statistics_are_rejected(corruption):
    msg = message()
    if corruption == 'length':
        msg.mean = []
    elif corruption == 'duplicate':
        msg.indices = [0, 0]
        msg.mean = [.9, .9]
        msg.variance = [.001, .001]
        msg.sample_count = [20, 20]
        msg.last_observed = [TimeMessage(sec=100), TimeMessage(sec=100)]
    elif corruption == 'outside':
        msg.indices = [2]
    elif corruption == 'negative_variance':
        msg.variance = [-.1]
    else:
        msg.sample_count = [0]
    with pytest.raises(ValueError):
        adapter.parse_statistics(msg)


def test_mean_only_map_preserves_unknown_and_advertises_missing_statistics():
    msg = OccupancyGrid()
    msg.header = message().header
    msg.info = message().info
    msg.data = [95, -1]
    grid = adapter.parse_mean_map(msg)
    assert grid.statistics_mode == 'mean_only'
    assert grid.known.tolist() == [[True, False]]
    assert grid.mean[0, 0] == pytest.approx(.95)


@pytest.fixture
def ros_runtime(request, monkeypatch, tmp_path):
    monkeypatch.setenv('ROS_DOMAIN_ID', '75')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path / 'ros_logs'))
    params = getattr(request, 'param', {})
    args = ['--ros-args', '--log-level', 'error']
    for key, value in params.items():
        if isinstance(value, float) and np.isnan(value):
            serialized = '.nan'
        elif value == float('inf'):
            serialized = '.inf'
        elif value == -float('inf'):
            serialized = '-.inf'
        else:
            serialized = str(value)
        args.extend(['-p', f'{key}:={serialized}'])
    rclpy.init(args=args)
    try:
        yield
    finally:
        if rclpy.ok():
            rclpy.shutdown()


@pytest.mark.parametrize('ros_runtime', [
    {'source_timeout': float('nan')}, {'source_timeout': float('inf')},
    {'publish_period': float('nan')}, {'publish_period': float('inf')},
    {'csv_as_of': float('nan')}, {'csv_as_of': float('inf')},
    {'source_timeout': 0.}, {'publish_period': -.5},
    {'future_max_lead': float('nan')}, {'future_max_lead': -.1},
    {'pending_wall_timeout': float('inf')}, {'pending_wall_timeout': 0.},
], indirect=True)
def test_nonfinite_or_invalid_runtime_configuration_fails_initialization(ros_runtime):
    # Keep a reference to the partially initialized ROS node for explicit cleanup.
    node = adapter.QualityCostNode.__new__(adapter.QualityCostNode)
    try:
        with pytest.raises(ValueError, match='invalid timeouts'):
            node.__init__()
    finally:
        node.destroy_node()


@pytest.mark.parametrize('ros_runtime', [
    {'input_mode': 'csv', 'csv_path': 'fixture.csv', 'csv_as_of': 99.},
], indirect=True)
def test_csv_reference_time_before_observation_fails_at_initialization(ros_runtime, monkeypatch):
    grid = adapter.parse_statistics(message())
    grid.statistics_mode = 'cumulative'
    monkeypatch.setattr(adapter, 'load_module_one', lambda _: grid)
    node = adapter.QualityCostNode.__new__(adapter.QualityCostNode)
    try:
        with pytest.raises(ValueError, match='future cell'):
            node.__init__()
    finally:
        node.destroy_node()


@pytest.mark.parametrize('ros_runtime', [
    {'input_mode': 'csv', 'csv_path': 'fixture.csv'},
], indirect=True)
def test_csv_default_reference_time_uses_source_clock(ros_runtime, monkeypatch):
    grid = adapter.parse_statistics(message())
    grid.statistics_mode = 'cumulative'
    monkeypatch.setattr(adapter, 'load_module_one', lambda _: grid)
    node = adapter.QualityCostNode()
    try:
        assert node.source_stamp == 100.
        assert node.last_error == 'offline_frozen_source_clock'
    finally:
        node.destroy_node()


@pytest.fixture
def node(ros_runtime, monkeypatch):
    value = adapter.QualityCostNode()
    # Fixed source clock makes future timestamps and stale handling deterministic.
    clock = SimpleNamespace(nanoseconds=100_000_000_000)
    monkeypatch.setattr(value, 'get_clock', lambda: SimpleNamespace(now=lambda: Time(nanoseconds=clock.nanoseconds)))
    output, diagnostics = [], []
    value.publisher = SimpleNamespace(publish=output.append)
    value.diagnostics = SimpleNamespace(publish=diagnostics.append)
    try:
        yield value, clock, output, diagnostics
    finally:
        value.destroy_node()


def test_future_snapshot_does_not_overwrite_last_valid_grid(node):
    value, _, output, _ = node
    valid = message(snapshot=99, observed=99)
    value.on_statistics(valid)
    original = value.grid
    future = message()
    future.header.stamp.nanosec = 250_000_000
    future.last_observed[0].nanosec = 250_000_000
    value.on_statistics(future)
    assert value.accepted == 1 and value.rejected == 1
    assert value.grid is original and value.source_stamp == 99
    assert value.input_invalid
    value.publish_cost()
    assert output and min(output[-1].data) >= value.settings['stale_cost']


def test_near_future_waits_without_publishing_or_refreshing_freshness(node, monkeypatch):
    value, clock, output, _ = node
    wall = SimpleNamespace(now=10.)
    monkeypatch.setattr(adapter.time, 'monotonic', lambda: wall.now)
    value.on_statistics(message(snapshot=99, observed=99))
    original = value.grid
    future = message()
    future.header.stamp.nanosec = 50_000_000
    future.last_observed[0].nanosec = 50_000_000
    wall.now = 10.25
    value.on_statistics(future)
    assert value.pending is not None and value.deferred == 1
    assert value.accepted == 1 and value.rejected == 0 and not value.input_invalid
    value.publish_cost()
    assert value.grid is original and value.source_stamp == 99
    assert value.last_receive == 10.
    clock.nanoseconds += 50_000_000
    wall.now = 10.5
    value.flush_pending()
    assert value.source_stamp == pytest.approx(100.05)
    assert value.grid.last_observed[0, 0] == pytest.approx(100.05)
    assert value.last_receive == 10.25  # original delivery, not deferred acceptance
    assert value.pending is None and value.deferred_accepted == 1


def test_pending_is_validated_before_waiting(node):
    value, _, _, _ = node
    future = message()
    future.header.stamp.nanosec = 50_000_000
    future.last_observed[0].nanosec = 60_000_000
    value.on_statistics(future)
    assert value.rejected == 1 and value.pending is None
    assert 'later than their snapshot' in value.last_error


def test_paused_clock_expires_pending_without_refreshing_old_source(node, monkeypatch):
    value, _, output, _ = node
    wall = SimpleNamespace(now=10.)
    monkeypatch.setattr(adapter.time, 'monotonic', lambda: wall.now)
    value.on_statistics(message())
    future = message()
    future.header.stamp.nanosec = 50_000_000
    value.on_statistics(future)
    wall.now = 11.01
    value.flush_pending()
    assert value.pending is None and value.pending_expired == 1 and value.rejected == 1
    assert value.source_stamp == 100 and value.last_receive == 10.
    value.publish_cost()
    assert min(output[-1].data) >= 200


def test_clock_rewind_discards_pending_without_replaying_it(node):
    value, clock, output, _ = node
    value.on_statistics(message())
    future = message()
    future.header.stamp.nanosec = 50_000_000
    value.on_statistics(future)
    clock.nanoseconds = 1_000_000_000
    value.flush_pending()
    assert value.pending is None and value.grid is None and value.pending_rewind_dropped == 1
    clock.nanoseconds = 101_000_000_000
    value.publish_cost()
    assert value.accepted == 1 and not output
    value.on_statistics(message(snapshot=101, observed=101))
    assert value.accepted == 2 and value.source_stamp == 101


def test_only_newest_complete_pending_snapshot_is_retained(node, monkeypatch):
    value, clock, _, _ = node
    wall = SimpleNamespace(now=10.)
    monkeypatch.setattr(adapter.time, 'monotonic', lambda: wall.now)
    first = message(); first.header.stamp.nanosec = 50_000_000
    value.on_statistics(first)
    newer = message(); newer.header.stamp.nanosec = 80_000_000
    newer.mean = [.2]
    wall.now = 10.1
    value.on_statistics(newer)
    wall.now = 10.2
    value.on_statistics(first)  # cannot reset the newer sample's wait deadline
    assert value.pending[1:] == pytest.approx((100.08, 10.1))
    assert value.pending_replaced == 1 and value.rejected == 1
    clock.nanoseconds += 80_000_000
    value.flush_pending()
    assert value.grid.mean[0, 0] == pytest.approx(.2) and value.accepted == 1


def test_bad_followup_retains_valid_geometry_but_publishes_conservative_fallback(node):
    value, clock, output, diagnostics = node
    value.on_statistics(message())
    original = value.grid
    clock.nanoseconds += 1_000_000_000
    invalid = message(snapshot=101, observed=101)
    invalid.info.origin.position.z = .5
    value.on_statistics(invalid)
    assert value.grid is original and value.input_invalid
    value.publish_cost()
    assert len(output) == 1 and min(output[0].data) >= 200
    assert diagnostics[-1].status[0].message == 'source_stale_conservative_fallback'


def test_rewind_clears_prior_clock_domain_and_new_snapshot_recovers(node):
    value, clock, output, diagnostics = node
    value.on_statistics(message())
    clock.nanoseconds = 1_000_000_000
    value.publish_cost()
    assert value.grid is None and not output
    assert diagnostics[-1].status[0].message == 'waiting_for_quality'
    value.on_statistics(message(snapshot=1, observed=1))
    value.publish_cost()
    assert value.grid is not None and len(output) == 1
    assert not value.input_invalid


def test_empty_sparse_snapshot_clears_previous_known_cells(node):
    value, clock, output, _ = node
    value.on_statistics(message())
    clock.nanoseconds += 1_000_000_000
    empty = message(snapshot=101, observed=101)
    empty.indices = []; empty.mean = []; empty.variance = []
    empty.sample_count = []; empty.last_observed = []
    value.on_statistics(empty)
    assert not value.grid.known.any()
    value.publish_cost()
    assert list(output[-1].data) == [140, 140]


def test_every_accepted_snapshot_is_evaluable_at_consumer_time(node):
    value, clock, _, _ = node
    msg = message()
    # Two individually tolerated offsets must not combine into an observation
    # farther in the future than the model can evaluate at the consumer clock.
    msg.header.stamp.nanosec = 800
    msg.last_observed[0].nanosec = 1600
    value.on_statistics(msg)
    if value.accepted:
        value.model.costs(value.grid, clock.nanoseconds * 1e-9)
    else:
        assert value.input_invalid
