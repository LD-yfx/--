"""Real DDS test of the installed C++ audit, including an exited duplicate TF owner."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import pytest

rclpy = pytest.importorskip('rclpy')
from ament_index_python.packages import get_package_prefix, PackageNotFoundError
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


def transform(parent, child):
    t = TransformStamped()
    t.header.frame_id = parent; t.child_frame_id = child
    t.transform.rotation.w = 1.
    return TFMessage(transforms=[t])


def test_cpp_audit_matches_endpoint_gid_and_retains_exited_duplicate(monkeypatch, tmp_path):
    try:
        binary = Path(get_package_prefix('quality_localization_audit')) / 'lib/quality_localization_audit/tf_authority_audit'
    except PackageNotFoundError:
        pytest.skip('Build/source quality_localization_audit first')
    monkeypatch.setenv('ROS_DOMAIN_ID', '75')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('ROS_LOG_DIR', str(tmp_path / 'ros_logs'))
    rclpy.init()
    receiver, owner_a, owner_b = [Node(s) for s in ('audit_test_recorder', 'audit_owner_a', 'audit_owner_b')]
    executor = SingleThreadedExecutor()
    for n in (receiver, owner_a, owner_b): executor.add_node(n)
    latched = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.TRANSIENT_LOCAL)
    static = owner_a.create_publisher(TFMessage, '/tf_static', latched)
    static.publish(transform('/audit_base', '/audit_sensor'))
    pub_a = owner_a.create_publisher(TFMessage, '/tf', 100)
    pub_b = owner_b.create_publisher(TFMessage, '/tf', 100)
    reports = []
    receiver.create_subscription(String, '/benchmark/tf_authorities', lambda m: reports.append(json.loads(m.data)), latched)
    log = (tmp_path / 'audit.log').open('w')
    process = subprocess.Popen([str(binary)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def wait(predicate):
        until = time.monotonic() + 5.
        while not predicate() and time.monotonic() < until:
            executor.spin_once(timeout_sec=.02)
            assert process.poll() is None, (tmp_path / 'audit.log').read_text()
        assert predicate(), 'No matching C++ TF provenance event'

    try:
        wait(lambda: pub_a.get_subscription_count() >= 1 and pub_b.get_subscription_count() >= 1)
        expected = {info.node_name: bytes(info.endpoint_gid).hex()
                    for info in receiver.get_publishers_info_by_topic('/tf')}
        assert all(len(gid) == 48 for gid in expected.values())
        pub_a.publish(transform('/audit_map', '/audit_odom'))
        wait(lambda: reports and reports[-1]['tf_authorities'].get('audit_map->audit_odom') == [expected['audit_owner_a']])
        pub_b.publish(transform('/audit_map', '/audit_odom'))
        wait(lambda: len(reports[-1]['tf_authorities'].get('audit_map->audit_odom', [])) == 2)
        owner_b.destroy_publisher(pub_b)
        old_reports = len(reports)
        wait(lambda: len(reports) >= old_reports + 2)
        final = reports[-1]
        assert set(final['tf_authorities']['audit_map->audit_odom']) == set(expected.values())
        assert len(final['tf_authorities']['audit_base->audit_sensor']) == 1
        owners = {p['gid']: p['node'] for p in final['publishers']}
        assert owners[expected['audit_owner_a']] == 'audit_owner_a'
        assert owners[expected['audit_owner_b']] == 'audit_owner_b'
        assert Path(final['runtime_executable_path']).resolve() == binary.resolve()
        assert final['observed_messages'] >= 3
    finally:
        os.killpg(process.pid, signal.SIGINT)
        try: process.wait(timeout=5.)
        except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
        log.close()
        for n in (receiver, owner_a, owner_b): executor.remove_node(n); n.destroy_node()
        executor.shutdown(); rclpy.try_shutdown()
