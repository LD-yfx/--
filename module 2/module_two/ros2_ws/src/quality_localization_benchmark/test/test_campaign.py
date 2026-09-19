"""Campaign lifecycle and denominators using fake runners, never a ROS matrix."""
from collections import Counter
import json
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality_localization_benchmark.campaign import (
    freeze_campaign, load_campaign, matrix, execute_campaign, collect_results,
    summarize_campaign, validate_smoke, execute_one,
)
from quality_localization_benchmark.summarize import (
    paired_bootstrap, summarize_records, write_summary,
)


@pytest.fixture
def roots(tmp_path):
    result = {}
    for category in ('runtime', 'code', 'config', 'maps', 'protocol'):
        directory = tmp_path/category
        directory.mkdir()
        (directory/'artifact.txt').write_text(category, encoding='utf-8')
        result[category] = directory
    return result


@pytest.fixture
def smoke(tmp_path):
    directory = tmp_path/'smoke'
    directory.mkdir()
    report = directory/'result.json'
    report.write_text(json.dumps(dict(phase='smoke', completed=True, valid_evidence=True,
                                     pose_metrics_valid=True, navigation_success=True,
                                     success=True)), encoding='utf-8')
    (directory/'manifest.json').write_text('{"runtime": "fixture"}', encoding='utf-8')
    return report


def result_for(run, rmse=.2, success=True, valid=True, pose_valid=True):
    return dict(run, completed=True, valid_evidence=valid, pose_metrics_valid=pose_valid,
                evaluation_valid=pose_valid, success=success,
                navigation_success=success,
                position_error={'rmse_m': rmse, 'p95_m': rmse},
                yaw_error={'rmse_deg': 1.}, estimate_coverage=1. if pose_valid else .2,
                truth_path_length_m=10., wall_seconds=20.,
                failure_reasons=[] if success else ['algorithm_failure'],
                invalid_reasons=[] if valid else ['bad_evidence'],
                legs=[], quality_pairs=[])


def fake_runner(calls, first_failure=False):
    def run(command, log_path, timeout):
        arguments = {}
        for flag in ('map-id', 'condition', 'seed', 'method', 'phase', 'run-id', 'output', 'ros-domain', 'gazebo-port'):
            arguments[flag] = command[command.index('--'+flag)+1]
        output = Path(arguments['output'])
        assert output.is_absolute() and not output.exists()
        assert int(arguments['ros-domain']) >= 0 and int(arguments['gazebo-port']) >= 1024
        output.mkdir()
        run_info = {key: arguments[key.replace('_', '-')]
                    for key in ('map_id', 'condition', 'method', 'phase')}
        run_info['seed'] = int(arguments['seed'])
        run_info['run_id'] = arguments['run-id']
        if '--scenario' in command:
            run_info['scenario'] = command[command.index('--scenario')+1]
        fail = first_failure and not calls
        calls.append(arguments)
        result = result_for(run_info, success=not fail, valid=not fail)
        result.pop('quality_pairs')
        (output/'result.json').write_text(json.dumps(result), encoding='utf-8')
        (output/'manifest.json').write_text(json.dumps({'frozen': True, 'run_id': run_info['run_id']}), encoding='utf-8')
        pairs = [{'q_fused': .2, 'error_position_m': .4},
                 {'q_fused': .8, 'error_position_m': .1}]
        pairs = [dict(row, run_id=run_info['run_id']) for row in pairs]
        (output/'quality_pairs.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in pairs))
        (output/'quality_relation.json').write_text('{"rows": 2}')
        (output/'error_rows.jsonl').write_text('')
        (output/'quality_coverage.json').write_text('{"fixture": true}')
        Path(log_path).write_text('fake runner only\n', encoding='utf-8')
        return 2 if fail else 0
    return run


def test_complete_frozen_matrix_and_disjoint_splits():
    calibration = matrix('calibration')
    formal = matrix('test')
    assert len(calibration) == 24
    assert len(formal) == 144
    assert Counter(r['comparison_group'] for r in formal) == {'main': 72, 'ablation': 72}
    assert {r['map_id'] for r in calibration}.isdisjoint({r['map_id'] for r in formal})
    assert {r['seed'] for r in calibration} == {0, 1, 2}
    assert {r['seed'] for r in formal} == {100, 101, 102}
    assert len({r['run_id'] for r in formal}) == 144
    groups = Counter((r['map_id'], r['seed'], r['condition']) for r in formal)
    assert set(groups.values()) == {6}
    assert matrix('test', 73) == matrix('test', 73)
    assert matrix('test', 73) != matrix('test', 74)
    with pytest.raises(ValueError):
        matrix('undeclared_phase')


def test_dynamic_matrix_separates_scenario_from_condition_and_preserves_static_ids():
    runs = matrix('dynamic')
    assert len(runs) == len({r['run_id'] for r in runs}) == 27
    assert {r['map_id'] for r in runs} == {'test_ring'}
    assert {r['condition'] for r in runs} == {'normal'}
    assert {r['seed'] for r in runs} == {100, 101, 102}
    assert {r['method'] for r in runs} == {'geometric', 'linear', 'full'}
    assert Counter(r['scenario'] for r in runs) == {
        'partial_block': 9, 'full_block': 9, 'sensor_recovery': 9}
    assert all('scenario' not in r for phase in ('calibration', 'test') for r in matrix(phase))
    assert matrix('dynamic', 73) == matrix('dynamic', 73)
    assert matrix('dynamic', 73) != matrix('dynamic', 74)


def test_dynamic_dispatch_passes_unique_run_and_scenario_and_rejects_misidentification(tmp_path, roots, smoke):
    output = tmp_path/'campaign'
    plan = freeze_campaign(output, 'dynamic', roots)
    run = plan['runs'][0]
    calls = []
    result = execute_one(output, plan, run, 0, validate_smoke(smoke), fake_runner(calls))
    assert result['valid_evidence'] and result['scenario'] == run['scenario']
    request = json.loads((output/'runs'/run['run_id']/'request.json').read_text())
    argv = request['command']
    assert argv[argv.index('--scenario')+1] == run['scenario']
    assert argv[argv.index('--run-id')+1] == run['run_id']
    _, records, _ = collect_results(output)
    summary = summarize_records(plan, records, draws=100)
    assert len(summary['groups']) == 9
    assert len(summary['paired_comparisons']) == 6
    assert {c['n_planned_pairs'] for c in summary['paired_comparisons']} == {3}
    assert {c['scenario'] for c in summary['paired_comparisons']} == {
        'partial_block', 'full_block', 'sensor_recovery'}


@pytest.mark.parametrize('filename', ['manifest.json', 'result.json', 'quality_pairs.jsonl'])
def test_runner_identity_mismatch_is_invalid_completed_attempt(tmp_path, roots, smoke, filename):
    output = tmp_path/'campaign'
    plan = freeze_campaign(output, 'calibration', roots)
    fixture = fake_runner([])
    def wrong_id(command, log, timeout):
        status = fixture(command, log, timeout)
        path = Path(command[command.index('--output')+1])/filename
        if filename.endswith('.jsonl'):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['run_id'] = 'artifacts'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        else:
            value = json.loads(path.read_text())
            value['run_id'] = 'artifacts'
            path.write_text(json.dumps(value))
        return status
    result = execute_one(output, plan, plan['runs'][0], 0, validate_smoke(smoke), wrong_id)
    assert result['completed'] and not result['valid_evidence']
    assert 'identity mismatch' in result['error']


def test_freeze_does_not_execute_and_refuses_overwrite(tmp_path, roots):
    output = tmp_path/'campaign'
    plan = freeze_campaign(output, 'calibration', roots)
    assert not (output/'runs').exists()
    assert load_campaign(output)['frozen_sha256'] == plan['frozen_sha256']
    with tarfile.open(output/'snapshots'/'code.tar.gz') as archive:
        assert archive.extractfile('artifact.txt').read() == b'code'
    with pytest.raises(FileExistsError):
        freeze_campaign(output, 'calibration', roots)


def test_changed_artifact_and_tampered_manifest_are_rejected(tmp_path, roots):
    output = tmp_path/'campaign'
    freeze_campaign(output, 'calibration', roots)
    (roots['config']/'artifact.txt').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='artifact hashes changed'):
        load_campaign(output)
    # Historical reports can still be read, but execution cannot silently resume.
    assert load_campaign(output, verify_artifacts=False)['phase'] == 'calibration'
    with tarfile.open(output/'snapshots'/'config.tar.gz') as archive:
        assert archive.extractfile('artifact.txt').read() == b'config'
    document = json.loads((output/'manifest.json').read_text())
    document['runs'].pop()
    (output/'manifest.json').write_text(json.dumps(document))
    with pytest.raises(ValueError, match='manifest content hash'):
        load_campaign(output, verify_artifacts=False)


def test_archived_sources_are_verified_even_for_historical_reports(tmp_path, roots):
    output = tmp_path/'campaign'
    freeze_campaign(output, 'calibration', roots)
    (output/'snapshots'/'code.tar.gz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='snapshot archive'):
        load_campaign(output, verify_artifacts=False)


def test_output_inside_hashed_inputs_is_rejected(roots):
    with pytest.raises(ValueError, match='outside hashed'):
        freeze_campaign(roots['code']/'results', 'calibration', roots)


@pytest.mark.parametrize('change', [
    {'completed': False}, {'valid_evidence': False}, {'pose_metrics_valid': False},
    {'navigation_success': False}, {'success': 'true'}, {'phase': 'test'},
])
def test_smoke_gate_requires_complete_valid_qualified_smoke(smoke, change):
    data = json.loads(smoke.read_text())
    data.update(change)
    smoke.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='successful smoke'):
        validate_smoke(smoke)


def test_failed_attempts_are_completed_and_never_rerun_on_resume(tmp_path, roots, smoke):
    output = tmp_path/'campaign'
    freeze_campaign(output, 'calibration', roots)
    calls = []
    first = execute_campaign(output, smoke, run_process=fake_runner(calls, first_failure=True))
    assert len(calls) == 24 and first['n_invalid_evidence'] == 1
    with pytest.raises(FileExistsError, match='resume'):
        execute_campaign(output, smoke, run_process=fake_runner(calls))
    second = execute_campaign(output, smoke, resume=True, run_process=fake_runner(calls))
    assert len(calls) == 24
    assert first == second
    assert not (output/'execute.lock').exists()


def test_two_workers_have_fixed_disjoint_domain_port_assignments(tmp_path, roots, smoke):
    output = tmp_path/'campaign'
    plan = freeze_campaign(output, 'calibration', roots, workers=2, ros_domain=90, gazebo_port=11600)
    calls = []
    execute_campaign(output, smoke, run_process=fake_runner(calls))
    assert len(calls) == 24
    for run in plan['runs']:
        request = json.loads((output/'runs'/run['run_id']/'request.json').read_text())
        argv = request['command']
        assert request['worker_index'] == run['sequence'] % 2
        assert int(argv[argv.index('--ros-domain')+1]) == 90+run['sequence'] % 2
        assert int(argv[argv.index('--gazebo-port')+1]) == 11600+run['sequence'] % 2


def test_partial_output_preflight_prevents_any_new_dispatch(tmp_path, roots, smoke):
    output = tmp_path/'campaign'
    plan = freeze_campaign(output, 'calibration', roots)
    (output/'runs'/plan['runs'][-1]['run_id']).mkdir(parents=True)
    calls = []
    with pytest.raises(ValueError, match='partial run'):
        execute_campaign(output, smoke, resume=True, run_process=fake_runner(calls))
    assert not calls
    assert not (output/'execute.lock').exists()


def test_changed_result_cannot_be_skipped_as_if_identical(tmp_path, roots, smoke):
    output = tmp_path/'campaign'
    plan = freeze_campaign(output, 'calibration', roots)
    calls = []
    execute_campaign(output, smoke, run_process=fake_runner(calls))
    result = output/'runs'/plan['runs'][0]['run_id']/'artifacts'/'result.json'
    result.write_text('{}')
    with pytest.raises(ValueError, match='content changed'):
        execute_campaign(output, smoke, resume=True, run_process=fake_runner(calls))
    assert len(calls) == 24


def test_collector_loads_sealed_quality_sidecar_for_calibration(tmp_path, roots, smoke):
    output = tmp_path/'campaign'
    plan = freeze_campaign(output, 'calibration', roots)
    run = plan['runs'][0]
    execute_one(output, plan, run, 0, validate_smoke(smoke), fake_runner([]))
    _, records, partial = collect_results(output)
    assert not partial and len(records[run['run_id']]['quality_pairs']) == 2
    summary = summarize_records(plan, records, draws=200)
    relation = summary['quality_relations'][0]['relation']
    assert relation['fields']['q_fused']['current']['spearman_rho'] == -1.
    assert relation['run_count'] == 1


@pytest.mark.parametrize('filename', [
    'quality_pairs.jsonl', 'quality_relation.json', 'error_rows.jsonl', 'manifest.json',
    'quality_coverage.json',
])
def test_changed_sidecar_or_manifest_prevents_historical_summary(tmp_path, roots, smoke, filename):
    output = tmp_path/'campaign'
    plan = freeze_campaign(output, 'calibration', roots)
    run = plan['runs'][0]
    execute_one(output, plan, run, 0, validate_smoke(smoke), fake_runner([]))
    target = output/'runs'/run['run_id']/'artifacts'/filename
    target.write_text('changed')
    with pytest.raises(ValueError, match='content changed'):
        collect_results(output)


def test_missing_quality_sidecar_is_recorded_as_infrastructure_failure(tmp_path, roots, smoke):
    output = tmp_path/'campaign'
    plan = freeze_campaign(output, 'calibration', roots)
    run = plan['runs'][0]
    fixture = fake_runner([])
    def incomplete(command, log_path, timeout):
        code = fixture(command, log_path, timeout)
        target = Path(command[command.index('--output')+1])/'quality_pairs.jsonl'
        target.unlink()
        return code
    result = execute_one(output, plan, run, 0, validate_smoke(smoke), incomplete)
    assert result['completed'] and not result['valid_evidence']
    assert 'missing required sidecar' in result['error']
    _, records, _ = collect_results(output)
    assert len(records) == 1 and records[run['run_id']]['quality_pairs'] == []


def test_infrastructure_failure_is_preserved_without_overwriting_partial_result(tmp_path, roots, smoke):
    output = tmp_path/'campaign'
    plan = freeze_campaign(output, 'calibration', roots)
    def broken(command, log_path, timeout):
        target = Path(command[command.index('--output')+1])
        target.mkdir()
        (target/'result.json').write_text('{"partial":')
        return 7
    counts = execute_campaign(output, smoke, run_process=broken)
    assert counts['n_executed'] == counts['n_invalid_evidence'] == 24
    job = output/'runs'/plan['runs'][0]['run_id']
    assert (job/'artifacts'/'result.json').read_text() == '{"partial":'
    assert json.loads((job/'failure_result.json').read_text())['completed'] is True
    _, records, partial = collect_results(output)
    assert len(records) == 24 and not partial


def test_bootstrap_unit_is_run_and_is_deterministic():
    a = paired_bootstrap([-.1, -.3, -.2], seed=5, draws=500)
    assert a == paired_bootstrap([-.1, -.3, -.2], seed=5, draws=500)
    assert a['n_pairs'] == 3 and a['bootstrap_unit'] == 'complete_paired_run'
    assert a['mean_difference'] == pytest.approx(-.2)
    assert a['ci95'][0] <= -.2 <= a['ci95'][1]
    assert paired_bootstrap([.1])['ci95'] is None
    assert paired_bootstrap([])['mean_difference'] is None


def test_pose_dropout_is_valid_trial_failure_not_excluded_denominator():
    plan = {'phase': 'test', 'runs': matrix('test')}
    selected = [r for r in plan['runs'] if r['map_id'] == 'test_ring'
                and r['condition'] == 'normal' and r['seed'] in (100, 101)
                and r['method'] in ('full', 'linear')]
    records = {}
    for run in selected:
        dropout = run['method'] == 'full' and run['seed'] == 100
        records[run['run_id']] = result_for(run, rmse=.01 if dropout else .2,
                                           success=not dropout, pose_valid=not dropout)
    summary = summarize_records(plan, records, draws=200)
    assert summary['n_planned'] == 144 and summary['n_executed'] == 4
    assert summary['n_not_executed'] == 140 and summary['n_invalid_evidence'] == 0
    assert summary['n_pose_metrics_invalid_valid_evidence'] == 1
    assert summary['n_qualified_success'] == 3
    assert summary['n_failed_executed'] == 1 and summary['n_algorithm_failure_valid_evidence'] == 1
    comparison = next(c for c in summary['paired_comparisons'] if c['comparator'] == 'linear')
    assert comparison['n_planned_pairs'] == 24 and comparison['n_executed_pairs'] == 2
    assert comparison['qualified_success_difference_all_executed_pairs']['mean_difference'] == -.5
    rmse = comparison['metrics']['position_rmse_m']['both_valid_evaluation']
    assert rmse['n_pairs'] == 1
    row = next(r for r in summary['runs'] if r['method'] == 'full' and r['seed'] == 100
               and r['map_id'] == 'test_ring' and r['condition'] == 'normal')
    assert row['position_rmse_m'] is None and row['observed_position_rmse_m'] == .01
    assert row['executed'] and row['valid_evidence'] and not row['qualified_success']
    json.dumps(summary, allow_nan=False)


def test_invalid_evidence_and_unexecuted_trials_remain_separate():
    plan = {'phase': 'calibration', 'runs': matrix('calibration')}
    run = plan['runs'][0]
    summary = summarize_records(plan, {run['run_id']: result_for(run, valid=False)}, draws=200)
    assert summary['n_invalid_evidence'] == 1
    assert summary['n_failed_executed'] == 1 and summary['n_algorithm_failure_valid_evidence'] == 0
    assert summary['n_executed'] == 1 and summary['n_not_executed'] == 23
    assert summary['n_qualified_success'] == 0
    assert not summary['matrix_executed']


def test_quality_groups_count_whole_run_not_two_legs():
    plan = {'phase': 'calibration', 'runs': matrix('calibration')}
    run = plan['runs'][0]
    result = result_for(run)
    result.pop('quality_pairs')
    result['legs'] = [
        {'quality_pairs': [{'run_id': 'outbound', 'q_fused': .2, 'error_position_m': .4}]},
        {'quality_pairs': [{'run_id': 'return', 'q_fused': .8, 'error_position_m': .1}]},
    ]
    summary = summarize_records(plan, {run['run_id']: result}, draws=200)
    relationship = summary['quality_relations'][0]['relation']
    assert relationship['run_count'] == 1 and relationship['rows'] == 2
    assert relationship['fields']['q_fused']['current']['spearman_rho'] == -1.


def test_summary_writes_new_artifacts_and_does_not_overwrite(tmp_path, roots, smoke):
    campaign = tmp_path/'campaign'
    freeze_campaign(campaign, 'calibration', roots)
    calls = []
    execute_campaign(campaign, smoke, run_process=fake_runner(calls))
    destination = summarize_campaign(campaign, plots=False)
    assert (destination/'report.md').is_file()
    data = json.loads((destination/'summary.json').read_text())
    assert data['n_executed'] == 24
    with pytest.raises(FileExistsError):
        summarize_campaign(campaign, plots=False)


def test_unplanned_result_cannot_inflate_completion():
    plan = {'phase': 'calibration', 'runs': matrix('calibration')}
    with pytest.raises(ValueError, match='unplanned'):
        summarize_records(plan, {'invented': {}}, draws=200)


def test_quality_plot_can_be_rendered_without_ros(tmp_path):
    pytest.importorskip('matplotlib')
    plan = {'phase': 'calibration', 'runs': matrix('calibration')}
    run = plan['runs'][0]
    result = result_for(run)
    result['quality_pairs'] = [
        {'q_fused': .1, 'error_position_m': .4},
        {'q_fused': .9, 'error_position_m': .1},
    ]
    summary = summarize_records(plan, {run['run_id']: result}, draws=200)
    output = tmp_path/'plot'
    write_summary(summary, output, plots=True)
    assert (output/'quality_calibration_01.png').read_bytes().startswith(b'\x89PNG')
    assert (output/'quality_scatter_01.png').read_bytes().startswith(b'\x89PNG')


def test_scatter_subsamples_display_only_without_changing_full_statistics():
    plan = {'phase': 'calibration', 'runs': matrix('calibration')}
    run = plan['runs'][0]
    result = result_for(run)
    result['quality_pairs'] = [dict(q_fused=i/100., error_position_m=(100-i)/100.,
                                   future_1s_max_error_m=(100-i)/100.+.1) for i in range(101)]
    summary = summarize_records(plan, {run['run_id']: result}, draws=100)
    group = summary['quality_relations'][0]
    assert group['relation']['fields']['q_fused']['current']['n'] == 101
    assert group['relation']['fields']['q_fused']['current']['spearman_rho'] == -1.
    assert group['scatter_sample']['stride'] == 10
    assert group['scatter_sample']['n_display_rows'] == 11
    assert [r['q_fused'] for r in group['scatter_sample']['rows']] == [i/10. for i in range(11)]
