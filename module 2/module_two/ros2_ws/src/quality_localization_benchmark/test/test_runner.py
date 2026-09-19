"""Research evidence must fail closed for truth leaks and interrupted counters."""
import copy

from quality_localization_benchmark.runner import (
    PASSTHROUGH_COUNTERS, action_completed_within_deadline, audit_graph, counter_delta, intersects,
)


def graph():
    return dict(nodes=[dict(name='amcl', namespace='/')],
        topics={
            '/ground_truth/odom': dict(publishers=[dict(node='ground_truth')], subscribers=[dict(node='benchmark_runner'),
                                                   dict(node='sensor_bridge'),
                                                   dict(node='rosbag2_recorder')]),
            '/odom': dict(publishers=[dict(node='diff_drive')]),
            '/localization/estimated_odom': dict(publishers=[dict(node='localization_adapter')]),
            '/localization_quality/quality_stats': dict(publishers=[dict(node='realtime_evaluator')]),
            '/scan': dict(publishers=[dict(node='sensor_bridge')]),
            '/camera/image_raw': dict(publishers=[dict(node='sensor_bridge')]),
            '/benchmark/tf_authorities': dict(publishers=[dict(node='tf_authority_audit')]),
            '/tf': dict(publishers=[dict(node='amcl', gid='amcl_gid'),
                                    dict(node='diff_drive', gid='odom_gid')]),
            '/tf_static': dict(publishers=[]),
        }, tf_audit_age=.1, tf_authorities={'map->odom': ['amcl_gid'], 'odom->base_footprint': ['odom_gid']})


def test_isolated_measurement_generation_and_scoring_are_allowed():
    assert audit_graph(graph()) == []


def test_localizer_truth_subscription_invalidates_evidence():
    data = graph()
    data['topics']['/ground_truth/odom']['subscribers'].append(dict(node='amcl'))
    assert 'truth_subscribed_by_amcl' in audit_graph(data)


def test_module_one_truth_subscription_invalidates_evidence():
    data = graph()
    data['topics']['/ground_truth/odom']['subscribers'].append(dict(node='realtime_evaluator'))
    assert 'truth_subscribed_by_realtime_evaluator' in audit_graph(data)


def test_duplicate_map_transform_cannot_be_valid():
    data = graph()
    data['tf_authorities']['map->odom'].append('ideal_gid')
    data['topics']['/tf']['publishers'].append(dict(node='ideal_simulator', gid='ideal_gid'))
    assert audit_graph(data)


def test_unresolved_transform_publisher_cannot_be_valid():
    data = graph()
    data['topics']['/tf']['publishers'] = []
    assert audit_graph(data)


def test_earlier_startup_faults_do_not_count_again_when_counters_stay_constant():
    before = {key: '7' for key in PASSTHROUGH_COUNTERS}
    assert all(value == 0 for value in counter_delta(before, before).values())


def test_missing_diagnostics_and_counter_reset_are_not_zero_faults():
    before = {key: '7' for key in PASSTHROUGH_COUNTERS}
    after = copy.deepcopy(before)
    after['clock_rewinds'] = '0'
    del after['truth_missing_passthrough']
    assert counter_delta(before, after)['clock_rewinds'] is None
    assert counter_delta(before, after)['truth_missing_passthrough'] is None


def test_conservative_circle_detects_corners_and_tangency():
    wall = [[0., 1., 0., 1.]]
    assert intersects(-.3, .5, wall, .3)
    assert intersects(-.2, -.2, wall, .3)
    assert not intersects(-.3, -.3, wall, .3)


def test_cancel_completion_race_cannot_turn_a_timeout_into_success():
    assert not action_completed_within_deadline(dict(action_status=4, timeout=True))
    assert action_completed_within_deadline(dict(action_status=4, timeout=False))
