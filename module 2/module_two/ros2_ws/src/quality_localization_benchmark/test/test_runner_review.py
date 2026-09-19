"""Independent evidence-audit regression cases; pure Python, no ROS processes.

These cases exercise adversarial graph/counter data rather than a duplicate of
the nominal launch. They prevent a single truth-derived odometry publisher from
passing merely because its TF edge happens to have exactly one owner.
"""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality_localization_benchmark.runner import audit_graph, counter_delta, PASSTHROUGH_COUNTERS


def endpoint(node, gid):
    return dict(node=node, namespace='/', gid=gid)


def valid_graph():
    def topic(publishers, subscribers=()):
        return dict(publishers=[endpoint(*item) for item in publishers],
                    subscribers=[endpoint(item, 'subscriber_'+item) for item in subscribers])
    return dict(nodes=[dict(name=n, namespace='/') for n in
                       ('amcl', 'diff_drive', 'realtime_evaluator', 'localization_adapter',
                        'sensor_bridge', 'benchmark_runner', 'rosbag2_recorder')],
        topics={
            '/ground_truth/odom': topic([('ground_truth', 'truth_odom_gid')],
                                       ['sensor_bridge', 'benchmark_runner', 'rosbag2_recorder']),
            '/odom': topic([('diff_drive', 'encoder_odom_gid')]),
            '/localization/estimated_odom': topic([('localization_adapter', 'adapter_odom_gid')]),
            '/localization_quality/quality_stats': topic([('realtime_evaluator', 'module_one_gid')]),
            '/tf': topic([('amcl', 'amcl_tf_gid'), ('diff_drive', 'encoder_tf_gid')]),
            '/tf_static': topic([('robot_state_publisher', 'fixed_tf_gid')]),
            '/scan': topic([('sensor_bridge', 'scan_gid')]),
            '/camera/image_raw': topic([('sensor_bridge', 'image_gid')]),
            '/benchmark/tf_authorities': topic([('tf_authority_audit', 'audit_gid')]),
        }, tf_audit_age=.1, tf_authorities={'map->odom': ['amcl_tf_gid'],
                           'odom->base_footprint': ['encoder_tf_gid']})


class RunnerReviewTest(unittest.TestCase):
    def test_complete_nominal_graph_is_valid(self):
        self.assertEqual(audit_graph(valid_graph()), [])

    def test_single_truth_derived_odom_transform_owner_is_rejected(self):
        graph = valid_graph()
        graph['topics']['/tf']['publishers'][1]['node'] = 'world_pose_proxy'
        self.assertTrue(audit_graph(graph), 'Uniqueness alone does not prove encoder odometry')

    def test_unresolved_single_odom_transform_owner_is_rejected(self):
        graph = valid_graph()
        graph['topics']['/tf']['publishers'].pop()
        self.assertTrue(audit_graph(graph), 'Every navigation transform needs a resolved owner')

    def test_module_one_stats_cannot_be_replaced_by_artificial_publisher(self):
        graph = valid_graph()
        graph['topics']['/localization_quality/quality_stats']['publishers'] = [
            endpoint('quality_fixture', 'fake_quality_gid')]
        self.assertTrue(audit_graph(graph), 'Receiving some QualityGrid is not module-one provenance')

    def test_additional_artificial_quality_publisher_is_rejected(self):
        graph = valid_graph()
        graph['topics']['/localization_quality/quality_stats']['publishers'].append(
            endpoint('quality_fixture', 'fake_quality_gid'))
        self.assertTrue(audit_graph(graph), 'Mixed quality publishers cannot support a causal comparison')

    def test_truth_derived_estimated_odometry_is_rejected(self):
        graph = valid_graph()
        graph['topics']['/localization/estimated_odom']['publishers'] = [
            endpoint('world_pose_proxy', 'fake_estimate_gid')]
        self.assertTrue(audit_graph(graph))

    def test_truth_derived_encoder_topic_is_rejected(self):
        graph = valid_graph()
        graph['topics']['/odom']['publishers'] = [endpoint('world_pose_proxy', 'fake_encoder_gid')]
        self.assertTrue(audit_graph(graph))

    def test_nonzero_unplanned_passthrough_is_retained(self):
        before = {key: '12' for key in PASSTHROUGH_COUNTERS}
        after = copy.deepcopy(before)
        after['truth_missing_passthrough'] = '15'
        result = counter_delta(before, after)
        self.assertEqual(result['truth_missing_passthrough'], 3)
        self.assertTrue(all(value == 0 for key, value in result.items()
                            if key != 'truth_missing_passthrough'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
