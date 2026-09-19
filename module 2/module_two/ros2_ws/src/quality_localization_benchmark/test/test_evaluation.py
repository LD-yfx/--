"""Independent numeric examples; requires neither ROS nor a running simulator."""
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality_localization_benchmark.evaluation import evaluate_episode, summarize_quality_relation


def manifest(duration=2., **changes):
    result = dict(mission_start=0., mission_end=duration, goal=[duration, 0.],
                  action_status=4, collision_count=0, wall_seconds=duration,
                  run_id='test_run', map_id='test_map', seed=100,
                  condition='normal', method='full')
    result.update(changes)
    return result


def trajectory(duration=2., dt=.1, offset=0., yaw=0.):
    return [dict(t=round(i*dt, 9), x=round(i*dt, 9)+offset, y=0., yaw=yaw)
            for i in range(round(duration/dt)+1)]


def assert_serializable(value):
    json.dumps(value, allow_nan=False)


def test_constant_error_has_known_rmse_p95_and_true_executed_length():
    result = evaluate_episode(manifest(), trajectory(offset=.2), trajectory(), [])
    assert result['position_error']['rmse_m'] == pytest.approx(.2)
    assert result['position_error']['p95_m'] == pytest.approx(.2)
    assert result['position_error']['max_m'] == pytest.approx(.2)
    assert result['truth_path_length_m'] == pytest.approx(2.)
    assert result['goal_distance_m'] == pytest.approx(0.)
    assert result['estimate_coverage'] == pytest.approx(1.)
    assert result['estimate_missing_seconds'] == pytest.approx(0.)
    assert result['success'] and result['evaluation_valid']
    assert_serializable(result)


def test_fixed_map_to_world_transform_not_best_fit_alignment():
    est = trajectory()
    truth = [dict(t=p['t'], x=3., y=4.+p['x'], yaw=math.pi/2) for p in est]
    info = manifest(goal=[3., 6.], frame_transform={'x': 3., 'y': 4., 'yaw': math.pi/2})
    result = evaluate_episode(info, est, truth, [])
    assert result['position_error']['rmse_m'] == pytest.approx(0., abs=1e-12)
    assert result['yaw_error']['rmse_rad'] == pytest.approx(0.)
    # Removing the prescribed transform exposes the real mismatch; no fit hides it.
    wrong = evaluate_episode(manifest(goal=[3., 6.]), est, truth, [])
    assert wrong['position_error']['rmse_m'] > 4.


def test_shortest_angle_interpolation_across_pi():
    truth = [dict(t=0., x=0., y=0., yaw=math.radians(179)),
             dict(t=.1, x=.1, y=0., yaw=math.radians(-179))]
    est = [dict(t=.05, x=.05, y=0., yaw=math.pi)]
    result = evaluate_episode(manifest(.1), est, truth, [])
    assert result['error_rows'][0]['truth_pairing'] == 'interpolated'
    assert result['position_error']['rmse_m'] == pytest.approx(0.)
    assert result['yaw_error']['rmse_rad'] == pytest.approx(0., abs=1e-12)


def test_missing_difficult_interval_cannot_be_reported_as_success():
    truth = trajectory(10.)
    est = [row for row in truth if row['t'] <= 1. or row['t'] >= 9.]
    result = evaluate_episode(manifest(10.), est, truth, [])
    assert result['navigation_success']  # Robot physically arrived and action agreed.
    assert not result['success'] and not result['evaluation_valid']
    assert result['position_error']['rmse_m'] == 0.  # Observed fragments alone look perfect.
    assert result['estimate_coverage'] == pytest.approx(.21)
    assert result['estimate_missing_seconds'] == pytest.approx(7.9)
    assert result['estimate_dropout_count'] == 1
    assert result['estimate_dropout_intervals'] == [[1., 9.]]


def test_sparse_truth_is_not_interpolated_across_large_gap():
    truth = [trajectory()[0], trajectory()[-1]]
    result = evaluate_episode(manifest(), trajectory(), truth, [])
    assert all(row['t'] in (0., 2.) for row in result['error_rows'])
    assert result['truth_path_length_m'] is None
    assert not result['success']


@pytest.mark.parametrize('stream', ['truth', 'estimates'])
@pytest.mark.parametrize('discontinuity', ['backwards', 'reset_sequence', 'explicit_reset'])
def test_never_interpolate_across_reset_or_time_reversal(stream, discontinuity):
    truth, estimates = trajectory(), trajectory()
    target = truth if stream == 'truth' else estimates
    if discontinuity == 'backwards':
        target[10]['t'] = .2
    elif discontinuity == 'reset_sequence':
        for i, row in enumerate(target):
            row['reset_sequence'] = int(i >= 10)
    else:
        target[10]['reset'] = True
    result = evaluate_episode(manifest(), estimates, truth, [])
    assert result['data_status'] == 'clock_or_reset_discontinuity'
    assert not result['evaluation_valid'] and not result['success']
    assert result['position_error']['rmse_m'] is None
    assert not result['error_rows']
    assert_serializable(result)


def test_outside_mission_samples_do_not_change_results():
    outside = [dict(t=-1., x=1e6, y=0., yaw=0.)]
    result = evaluate_episode(manifest(), outside+trajectory(), outside+trajectory(), [])
    assert result['success']
    assert result['input_statistics']['truth']['outside_mission'] == 1
    assert result['truth_path_length_m'] == pytest.approx(2.)


@pytest.mark.parametrize('change', [
    {'action_status': 6}, {'collision_count': 1}, {'goal': [3., 0.]},
    {'collision_count': None}, {'action_status': True}, {'goal_yaw': math.pi},
])
def test_navigation_success_requires_status_true_goal_and_no_collision(change):
    result = evaluate_episode(manifest(**change), trajectory(), trajectory(), [])
    assert not result['navigation_success'] and not result['success']
    assert result['evaluation_valid']


def test_goal_is_checked_against_truth_not_estimated_pose():
    result = evaluate_episode(manifest(), trajectory(), trajectory(offset=1.), [])
    assert result['goal_distance_m'] == pytest.approx(1.)
    assert not result['success']


def test_missing_terminal_truth_is_explicit():
    result = evaluate_episode(manifest(), trajectory(), trajectory()[:-3], [])
    assert result['goal_distance_m'] is None
    assert not result['navigation_success']
    assert 'missing_terminal_truth' in result['failure_reasons']


def test_no_data_and_nonfinite_data_never_emit_nan_or_inf():
    for bad in ([], [dict(t=0., x=float('nan'), y=0., yaw=0.)],
                [dict(t=0., x=float('inf'), y=0., yaw=0.)]):
        result = evaluate_episode(manifest(), bad, [], [{'timestamp': 'nan', 'q_fused': 'inf'}])
        assert result['position_error']['rmse_m'] is None
        assert not result['success']
        assert_serializable(result)


@pytest.mark.parametrize('change', [
    {'mission_end': float('inf')}, {'mission_start': 3.},
    {'goal': [float('nan'), 0.]}, {'frame_transform': {'x': float('nan')}},
    {'goal_yaw': float('nan')},
])
def test_invalid_manifest_returns_explicit_invalid_state(change):
    result = evaluate_episode(manifest(**change), trajectory(), trajectory(), [])
    assert result['data_status'] == 'invalid_manifest'
    assert result['goal_distance_m'] is None
    assert_serializable(result)


def test_quality_csv_strings_pair_with_current_and_independent_future_error():
    truth = trajectory()
    est = trajectory(offset=.2)
    est[10]['x'] += .6  # Independent estimator error at t=1, not derived from Q.
    quality = [{'timestamp': '0.0', 'q_laser': '.9', 'q_visual': '.8', 'q_fused': '.85'},
               {'timestamp': '1.5', 'q_fused': '.1'}]
    result = evaluate_episode(manifest(), est, truth, quality)
    first, last = result['quality_pairs']
    assert first['q_fused'] == .85
    assert first['error_position_m'] == pytest.approx(.2)
    assert first['future_1s_max_error_m'] == pytest.approx(.8)
    assert first['future_1s_status'] == 'matched'
    assert last['future_1s_max_error_m'] is None
    assert last['future_1s_status'] == 'insufficient_horizon'
    assert_serializable(result)


def test_future_target_requires_entire_window_coverage():
    est = [r for r in trajectory() if not .3 < r['t'] < .9]
    result = evaluate_episode(manifest(), est, trajectory(), [{'timestamp': 0., 'q_fused': .8}])
    pair = result['quality_pairs'][0]
    assert pair['error_position_m'] == 0.
    assert pair['future_1s_max_error_m'] is None
    assert pair['future_1s_status'] == 'insufficient_coverage'


def test_quality_values_do_not_change_localization_error_measurement():
    low = evaluate_episode(manifest(), trajectory(offset=.2), trajectory(), [{'timestamp': 0., 'q_fused': 0.}])
    high = evaluate_episode(manifest(), trajectory(offset=.2), trajectory(), [{'timestamp': 0., 'q_fused': 1.}])
    assert low['position_error'] == high['position_error']
    assert low['quality_pairs'][0]['future_1s_max_error_m'] == high['quality_pairs'][0]['future_1s_max_error_m']


def test_quality_time_reversal_is_flagged_instead_of_reordered():
    result = evaluate_episode(manifest(), trajectory(), trajectory(),
                              [{'timestamp': 1., 'q_fused': .9}, {'timestamp': .5, 'q_fused': .1}])
    assert result['quality_status'] == 'clock_discontinuity'
    assert result['quality_pairs'] == []


def test_relation_spearman_ties_and_last_bin():
    rows = [dict(run_id='r1', q_fused=q, error_position_m=e,
                 future_1s_max_error_m=e) for q, e in [(0., 1.), (0., 1.), (1., 0.), (1., 0.)]]
    result = summarize_quality_relation(rows)
    relation = result['fields']['q_fused']['current']
    assert relation['spearman_rho'] == pytest.approx(-1.)
    assert relation['bins'][0]['n'] == 2
    assert relation['bins'][9]['n'] == 2
    assert sum(b['n'] for b in relation['bins']) == 4
    assert result['run_count'] == 1
    assert relation['bins'][1]['error_median_m'] is None
    assert_serializable(result)


def test_constant_quality_has_undefined_correlation_not_nan():
    rows = [{'q_visual': .5, 'error_position_m': .1*i} for i in range(5)]
    result = summarize_quality_relation(rows)
    assert result['fields']['q_visual']['current']['spearman_rho'] is None
    assert result['fields']['q_laser']['current']['n'] == 0
    assert_serializable(result)


def test_duplicate_pose_messages_do_not_inflate_sample_count():
    est = trajectory()
    est.insert(1, dict(est[0]))
    result = evaluate_episode(manifest(), est, trajectory(), [])
    assert result['position_error']['n'] == 21
    assert result['input_statistics']['estimates']['duplicate_timestamps'] == 1


def test_extreme_finite_coordinates_do_not_leak_infinity():
    est = trajectory()
    for row in est:
        row['x'] = 1e308
    result = evaluate_episode(manifest(), est, trajectory(), [])
    assert result['position_error']['rmse_m'] == pytest.approx(1e308)
    assert_serializable(result)


def test_exceedance_fraction_weights_source_time_not_publication_count():
    truth = trajectory(1.)
    est = trajectory(1.)
    for row in est:
        if row['t'] <= .5:
            row['x'] += .4
    result = evaluate_episode(manifest(1.), est, truth, [])
    assert result['position_exceedance_fraction_observed'] == pytest.approx(6/11)
    assert result['position_exceedance_seconds_observed'] == pytest.approx(.55)
    assert result['position_exceedance_fraction_observed_time'] == pytest.approx(.55)


def test_error_statistics_do_not_include_farther_than_50ms_truth():
    truth = [dict(t=0., x=0., y=0., yaw=0.), dict(t=1., x=1., y=0., yaw=0.)]
    est = [dict(t=.05, x=.05, y=0., yaw=0.), dict(t=.0501, x=.0501, y=0., yaw=0.)]
    result = evaluate_episode(manifest(1.), est, truth, [])
    assert len(result['error_rows']) == 1
    assert result['error_rows'][0]['t'] == .05


def test_extreme_finite_relation_values_remain_json_serializable():
    result = summarize_quality_relation([
        {'q_fused': .2, 'error_position_m': 1e308},
        {'q_fused': .2, 'error_position_m': 1e308},
    ])
    assert result['fields']['q_fused']['current']['bins'][2]['error_median_m'] == 1e308
    assert_serializable(result)
