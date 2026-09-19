"""Freeze, execute and summarize a complete independent-localization matrix.

No ROS imports: execution dispatches the separate benchmark runner. Long matrices
are never started by freeze/summarize, and execute requires a successful smoke.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tarfile
import time

from .summarize import summarize_records, write_summary, pose_metrics_valid

DEVELOPMENT_MAPS = ('dev_asymmetric', 'dev_rooms')
TEST_MAPS = ('test_ring', 'test_corridors')
CONDITIONS = ('normal', 'visual', 'laser', 'combined')
MAIN_METHODS = ('geometric', 'linear', 'full')
ABLATIONS = ('no_variance', 'no_count', 'no_age')
DYNAMIC_SCENARIOS = ('partial_block', 'full_block', 'sensor_recovery')
ARTIFACT_CATEGORIES = ('runtime', 'code', 'config', 'maps', 'protocol')
RUNNER_SIDECARS = ('quality_pairs.jsonl', 'quality_relation.json', 'error_rows.jsonl')
OPTIONAL_SIDECARS = ('quality_coverage.json',)
IGNORED_DIRECTORIES = {'.git', '__pycache__', '.pytest_cache', 'build', 'log', 'logs'}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def matrix(phase, order_seed=20260919):
    if phase == 'dynamic':
        rng = random.Random(order_seed)
        groups = [(seed, scenario) for seed in (100, 101, 102) for scenario in DYNAMIC_SCENARIOS]
        rng.shuffle(groups)
        result = []
        for seed, scenario in groups:
            methods = list(MAIN_METHODS)
            rng.shuffle(methods)
            for method in methods:
                result.append({
                    'run_id': f'dynamic__test_ring__{scenario}__s{seed}__{method}',
                    'phase': phase, 'map_id': 'test_ring', 'seed': seed,
                    'condition': 'normal', 'scenario': scenario, 'method': method,
                    'comparison_group': 'dynamic', 'sequence': len(result),
                })
        return result
    if phase not in ('calibration', 'test'):
        raise ValueError('campaign phase must be calibration, test or dynamic')
    maps = DEVELOPMENT_MAPS if phase == 'calibration' else TEST_MAPS
    seeds = (0, 1, 2) if phase == 'calibration' else (100, 101, 102)
    methods = ('geometric',) if phase == 'calibration' else MAIN_METHODS+ABLATIONS
    groups = [(map_id, seed, condition) for map_id in maps for seed in seeds for condition in CONDITIONS]
    rng = random.Random(order_seed)
    rng.shuffle(groups)
    result = []
    for map_id, seed, condition in groups:
        group_methods = list(methods)
        rng.shuffle(group_methods)
        for method in group_methods:
            result.append({
                'run_id': f'{phase}__{map_id}__{condition}__s{seed}__{method}',
                'phase': phase, 'map_id': map_id, 'seed': seed,
                'condition': condition, 'method': method,
                'comparison_group': 'calibration' if phase == 'calibration' else
                                    'main' if method in MAIN_METHODS else 'ablation',
                'sequence': len(result),
            })
    return result


def snapshot_artifacts(roots):
    if set(roots) != set(ARTIFACT_CATEGORIES):
        raise ValueError('provide exactly runtime, code, config, maps and protocol artifact roots')
    snapshot = {}
    for category in ARTIFACT_CATEGORIES:
        root = Path(roots[category]).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f'{category} artifact root missing: {root}')
        files = {}
        if root.is_file():
            files[root.name] = {'sha256': _file_hash(root), 'bytes': root.stat().st_size}
        else:
            visited = set()
            for directory, subdirs, names in os.walk(root, followlinks=True):
                resolved = str(Path(directory).resolve())
                if resolved in visited:
                    subdirs[:] = []
                    continue
                visited.add(resolved)
                subdirs[:] = sorted(d for d in subdirs if d not in IGNORED_DIRECTORIES)
                for name in sorted(names):
                    path = Path(directory)/name
                    if name.endswith(('.pyc', '.pyo')) or not path.is_file():
                        continue
                    files[str(path.relative_to(root)).replace(os.sep, '/')] = {
                        'sha256': _file_hash(path), 'bytes': path.stat().st_size,
                    }
        if not files:
            raise ValueError(f'{category} artifact root contains no files')
        snapshot[category] = {'root': str(root), 'files': files,
                              'content_sha256': _digest(files)}
    return snapshot


def archive_artifacts(output, artifacts):
    """Save dereferenced finite input trees, not just hashes of disappearing code."""
    directory = output/'snapshots'
    directory.mkdir(exist_ok=False)
    archives = {}
    for category, snapshot in artifacts.items():
        root = Path(snapshot['root'])
        archive_path = directory/(category+'.tar.gz')
        with tarfile.open(archive_path, mode='x:gz') as archive:
            for relative, metadata in sorted(snapshot['files'].items()):
                path = root if root.is_file() else root/relative
                info = tarfile.TarInfo(relative)
                info.size = metadata['bytes']
                info.mode = path.stat().st_mode & 0o777
                info.mtime = 0
                with path.open('rb') as stream:
                    archive.addfile(info, stream)
        archives[category] = {
            'path': str(archive_path.relative_to(output)).replace(os.sep, '/'),
            'sha256': _file_hash(archive_path), 'bytes': archive_path.stat().st_size,
            'file_count': len(snapshot['files']),
            'contents': 'original file bytes; installed/source symlinks dereferenced',
        }
    return archives


def freeze_campaign(output, phase, roots, order_seed=20260919, workers=1,
                    ros_domain=85, gazebo_port=11585, runner_command=None,
                    process_timeout_seconds=900.):
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'campaign output already exists; never overwrite: {output}')
    if workers not in (1, 2):
        raise ValueError('only one or two isolated workers are supported')
    if not isinstance(ros_domain, int) or not 0 <= ros_domain <= 232-workers+1:
        raise ValueError('worker ROS domains must remain in [0,232]')
    if not isinstance(gazebo_port, int) or not 1024 <= gazebo_port <= 65535-workers+1:
        raise ValueError('worker Gazebo ports must remain in [1024,65535]')
    if not math.isfinite(process_timeout_seconds) or process_timeout_seconds <= 0:
        raise ValueError('positive finite process timeout required')
    for category, path in roots.items():
        root = Path(path).expanduser().resolve()
        if root.is_dir() and (output == root or root in output.parents):
            raise ValueError(f'campaign output must be outside hashed {category} root')
    artifacts = snapshot_artifacts(roots)
    command = runner_command or [sys.executable, '-m', 'quality_localization_benchmark.runner']
    if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
        raise ValueError('runner_command must be a nonempty JSON string array; no shell command strings')
    plan = {
        'schema_version': 1, 'phase': phase, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'order_seed': order_seed, 'runs': matrix(phase, order_seed), 'artifacts': artifacts,
        'runner_command': command,
        'execution': {'workers': workers, 'ros_domain_base': ros_domain,
                      'gazebo_port_base': gazebo_port, 'process_timeout_seconds': process_timeout_seconds},
        'scientific_scope': 'Independent estimator navigation; multimodal quality is not multimodal fused localization.',
        'stopping_rule': 'Preserve every planned failure/invalid result; never select or remove runs after results.',
        'dynamic_status': ('27_prespecified_runs; runner_must_implement_actual_events' if phase == 'dynamic'
                           else 'not_in_this_campaign; separate frozen dynamic matrix still required'),
    }
    output.mkdir(parents=True, exist_ok=False)
    plan['artifact_archives'] = archive_artifacts(output, artifacts)
    if snapshot_artifacts(roots) != artifacts:
        raise ValueError('inputs changed while freezing; partial snapshots retained, no valid manifest created')
    plan['frozen_sha256'] = _digest(plan)
    _write_new(output/'manifest.json', plan)
    return plan


def load_campaign(path, verify_artifacts=True):
    root = Path(path).expanduser().resolve()
    plan = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    expected = plan.get('frozen_sha256')
    unsigned = {k: v for k, v in plan.items() if k != 'frozen_sha256'}
    if expected != _digest(unsigned):
        raise ValueError('frozen manifest content hash mismatch')
    if plan['runs'] != matrix(plan['phase'], plan['order_seed']):
        raise ValueError('frozen campaign matrix is incomplete or was selectively changed')
    for snapshot in plan.get('artifact_archives', {}).values():
        archive_path = _safe_relative(root, snapshot['path'])
        if not archive_path.is_file() or _file_hash(archive_path) != snapshot['sha256']:
            raise ValueError('frozen artifact snapshot archive missing or changed')
    if set(plan.get('artifact_archives', {})) != set(ARTIFACT_CATEGORIES):
        raise ValueError('replayable source/runtime/config/map/protocol snapshots are required')
    if verify_artifacts:
        current = snapshot_artifacts({key: value['root'] for key, value in plan['artifacts'].items()})
        if current != plan['artifacts']:
            raise ValueError('runtime/code/config/map artifact hashes changed; preserve old campaign and freeze a new version')
    return plan


def validate_smoke(path):
    path = Path(path).expanduser().resolve()
    result = json.loads(path.read_text(encoding='utf-8'))
    required = ('completed', 'valid_evidence', 'navigation_success', 'success')
    if result.get('phase') != 'smoke' or any(result.get(key) is not True for key in required) or not pose_metrics_valid(result):
        raise ValueError('a completed, valid, coverage-qualified successful smoke result is required')
    manifest = path.with_name('manifest.json')
    if not manifest.is_file():
        raise ValueError('smoke result must have its frozen adjacent manifest.json')
    # Runner owns exact per-run runtime provenance; retain both files for audit.
    json.loads(manifest.read_text(encoding='utf-8'))
    return {'result_path': str(path), 'result_sha256': _file_hash(path),
            'manifest_path': str(manifest), 'manifest_sha256': _file_hash(manifest)}


def _run_hash(plan, run):
    return _digest({'frozen_sha256': plan['frozen_sha256'], 'run': run})


def _safe_relative(root, relative):
    path = (root/relative).resolve()
    if root != path and root not in path.parents:
        raise ValueError('recorded result path escapes its run directory')
    return path


def _read_quality_pairs(path):
    rows = []
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'quality pair must be an object: {path}:{number}')
            _json(row)
            rows.append(row)
    return rows


def _seal_sidecars(job, artifacts, run_id):
    sealed = {}
    for name in RUNNER_SIDECARS+OPTIONAL_SIDECARS:
        path = artifacts/name
        if not path.is_file():
            if name in OPTIONAL_SIDECARS:
                continue
            raise ValueError(f'runner output is missing required sidecar: {name}')
        sealed[name] = {'path': str(path.relative_to(job)).replace(os.sep, '/'),
                        'sha256': _file_hash(path), 'bytes': path.stat().st_size}
    pairs = _read_quality_pairs(artifacts/'quality_pairs.jsonl')
    if any(row.get('run_id') != run_id for row in pairs):
        raise ValueError('quality pair identity mismatch: run_id')
    relation = json.loads((artifacts/'quality_relation.json').read_text(encoding='utf-8'))
    if not isinstance(relation, dict):
        raise ValueError('quality_relation.json must contain an object')
    _json(relation)
    return sealed


def _completed_result(job, plan, run):
    record_path = job/'execution.json'
    if not record_path.is_file():
        raise ValueError(f'partial run directory has no completed execution record: {job}')
    record = json.loads(record_path.read_text(encoding='utf-8'))
    if record.get('completed') is not True or record.get('frozen_sha256') != plan['frozen_sha256'] or record.get('run_sha256') != _run_hash(plan, run):
        raise ValueError(f'run completion/hash mismatch: {job}')
    result_path = _safe_relative(job, record['result_path'])
    if not result_path.is_file() or _file_hash(result_path) != record.get('result_sha256'):
        raise ValueError(f'recorded result content changed: {result_path}')
    if record.get('runner_manifest_path'):
        manifest_path = _safe_relative(job, record['runner_manifest_path'])
        if not manifest_path.is_file() or _file_hash(manifest_path) != record.get('runner_manifest_sha256'):
            raise ValueError(f'runner manifest content changed: {manifest_path}')
    result = json.loads(result_path.read_text(encoding='utf-8'))
    if result.get('completed') is not True:
        raise ValueError(f'run is not completed: {result_path}')
    sidecars = record.get('runner_sidecars', {})
    if result_path.name == 'result.json' and not set(RUNNER_SIDECARS).issubset(sidecars):
        raise ValueError(f'completed runner result has missing sidecar seals: {job}')
    for name, metadata in sidecars.items():
        path = _safe_relative(job, metadata['path'])
        if not path.is_file() or _file_hash(path) != metadata.get('sha256'):
            raise ValueError(f'runner sidecar content changed: {path}')
    if 'quality_pairs.jsonl' in sidecars and result_path.name == 'result.json':
        rows = _read_quality_pairs(_safe_relative(job, sidecars['quality_pairs.jsonl']['path']))
        if 'quality_pairs' in result and result['quality_pairs'] != rows:
            raise ValueError(f'inline and sidecar quality pairs disagree: {job}')
        result['quality_pairs'] = rows
    return result


def _run_process(command, log_path, timeout):
    """Only terminate our own new process group on infrastructure timeout."""
    with Path(log_path).open('x', encoding='utf-8') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=(os.name == 'posix'))
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == 'posix':
                os.killpg(process.pid, signal.SIGINT)
            else:
                process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.kill()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == 'posix':
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
            raise TimeoutError('runner exceeded frozen infrastructure process timeout')


def _validate_result(result, run):
    if not isinstance(result, dict) or result.get('completed') is not True:
        raise ValueError('runner did not produce a completed result')
    if not isinstance(result.get('valid_evidence'), bool):
        raise ValueError('runner valid_evidence must be an explicit boolean')
    for key in ('run_id', 'phase', 'map_id', 'seed', 'condition', 'method')+ (('scenario',) if 'scenario' in run else ()):
        if result.get(key) != run[key]:
            raise ValueError(f'runner result identity mismatch: {key}')
    if type(result.get('seed')) is not int:
        raise ValueError('runner seed must be an integer, not a boolean or string')
    _json(result)  # Reject NaN/Inf and non-JSON data, including nested values.


def execute_one(root, plan, run, worker_index, smoke, run_process=_run_process):
    job = root/'runs'/run['run_id']
    job.mkdir(parents=True, exist_ok=False)
    artifacts = job/'artifacts'  # Intentionally absent before runner invocation.
    command = list(plan['runner_command'])
    for name in ('map_id', 'condition', 'seed', 'method', 'phase'):
        command.extend(['--'+name.replace('_', '-'), str(run[name])])
    if 'scenario' in run:
        command.extend(['--scenario', run['scenario']])
    command += ['--run-id', run['run_id'], '--output', str(artifacts),
                '--ros-domain', str(plan['execution']['ros_domain_base']+worker_index),
                '--gazebo-port', str(plan['execution']['gazebo_port_base']+worker_index)]
    request = {'frozen_sha256': plan['frozen_sha256'], 'run_sha256': _run_hash(plan, run),
               'run': run, 'worker_index': worker_index, 'command': command,
               'smoke_gate': smoke, 'started_utc': datetime.now(timezone.utc).isoformat()}
    _write_new(job/'request.json', request)
    began = time.monotonic()
    exit_code, error, result_path = None, None, artifacts/'result.json'
    runner_manifest = artifacts/'manifest.json'
    manifest_hash = None
    sidecars = {}
    try:
        exit_code = run_process(command, job/'stdout.log', plan['execution']['process_timeout_seconds'])
        load_campaign(root, verify_artifacts=True)
        if not result_path.is_file():
            raise ValueError('runner terminated without result.json')
        if not runner_manifest.is_file():
            raise ValueError('runner output is missing its frozen manifest.json')
        runner_metadata = json.loads(runner_manifest.read_text(encoding='utf-8'))
        if runner_metadata.get('run_id') != run['run_id']:
            raise ValueError('runner manifest identity mismatch: run_id')
        manifest_hash = _file_hash(runner_manifest)
        result = json.loads(result_path.read_text(encoding='utf-8'))
        _validate_result(result, run)
        sidecars = _seal_sidecars(job, artifacts, run['run_id'])
        if exit_code != 0 and result.get('success') is True:
            raise ValueError('nonzero runner exit contradicts reported success')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        # Preserve any malformed/partial original output; do not overwrite it.
        result_path = job/'failure_result.json'
        result = dict(run, completed=True, valid_evidence=False, evaluation_valid=False, pose_metrics_valid=False,
                      navigation_success=False, success=False,
                      invalid_reasons=['campaign_execution_failure', error],
                      failure_reasons=['infrastructure_or_incomplete_output'],
                      legs=[], quality_pairs=[], error=error)
        _write_new(result_path, result)
    record = {
        'completed': True, 'frozen_sha256': plan['frozen_sha256'],
        'run_sha256': _run_hash(plan, run), 'returncode': exit_code, 'error': error,
        'result_path': str(result_path.relative_to(job)).replace(os.sep, '/'),
        'result_sha256': _file_hash(result_path), 'elapsed_wall_seconds': time.monotonic()-began,
        'runner_manifest_path': str(runner_manifest.relative_to(job)).replace(os.sep, '/') if manifest_hash else None,
        'runner_manifest_sha256': manifest_hash,
        'runner_sidecars': sidecars,
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    }
    _write_new(job/'execution.json', record)
    return result


def execute_campaign(path, smoke_report, resume=False, run_process=_run_process):
    root = Path(path).expanduser().resolve()
    plan = load_campaign(root, verify_artifacts=True)
    smoke = validate_smoke(smoke_report)
    lock = root/'execute.lock'
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(_json({'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat()}))
    results, pending = {}, []
    try:
        # Preflight every directory before starting any new run.
        for run in plan['runs']:
            job = root/'runs'/run['run_id']
            if job.exists():
                if not resume:
                    raise FileExistsError(f'run output exists; explicit --resume required: {job}')
                results[run['run_id']] = _completed_result(job, plan, run)
            else:
                pending.append(run)
        workers = plan['execution']['workers']
        # Fixed allocation/order; faster methods do not choose subsequent cases.
        batches = [[run for run in pending if run['sequence'] % workers == index]
                   for index in range(workers)]
        def work(index):
            completed = {}
            for run in batches[index]:
                # Catch changes during long matrices before each subsequent run.
                load_campaign(root, verify_artifacts=True)
                completed[run['run_id']] = execute_one(root, plan, run, index, smoke, run_process)
            return completed
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(work, i) for i in range(workers)]
            for future in futures:
                results.update(future.result())
    finally:
        lock.unlink()
    return {'n_planned': len(plan['runs']), 'n_executed': len(results),
            'n_invalid_evidence': sum(r.get('valid_evidence') is not True for r in results.values()),
            'n_qualified_success': sum(r.get('success') is True and r.get('valid_evidence') is True
                                       and pose_metrics_valid(r) for r in results.values())}


def collect_results(path):
    root = Path(path).expanduser().resolve()
    # Historical reports remain readable after a later source revision. The
    # frozen manifest and each result seal must still validate.
    plan = load_campaign(root, verify_artifacts=False)
    records = {}
    partial = []
    for run in plan['runs']:
        job = root/'runs'/run['run_id']
        if not job.exists():
            continue
        if not (job/'execution.json').exists():
            partial.append(run['run_id'])
            continue
        records[run['run_id']] = _completed_result(job, plan, run)
    return plan, records, partial


def summarize_campaign(path, output=None, plots=True):
    root = Path(path).expanduser().resolve()
    plan, records, partial = collect_results(root)
    summary = summarize_records(plan, records)
    summary['partial_run_directories'] = partial
    summary['frozen_sha256'] = plan['frozen_sha256']
    content_digest = _digest(summary)
    destination = Path(output).expanduser().resolve() if output else root/'reports'/('summary-'+content_digest[:12])
    write_summary(summary, destination, plots=plots)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze = sub.add_parser('freeze', help='Freeze complete matrix; does not launch any run')
    freeze.add_argument('--output', required=True)
    freeze.add_argument('--phase', choices=('calibration', 'test', 'dynamic'), required=True)
    for category in ARTIFACT_CATEGORIES:
        freeze.add_argument('--'+category+'-root', required=True)
    freeze.add_argument('--order-seed', type=int, default=20260919)
    freeze.add_argument('--workers', type=int, choices=(1, 2), default=1)
    freeze.add_argument('--ros-domain', type=int, default=85)
    freeze.add_argument('--gazebo-port', type=int, default=11585)
    freeze.add_argument('--process-timeout-seconds', type=float, default=900.)
    freeze.add_argument('--runner-command-json', help='Optional argv JSON array; never interpreted by a shell')
    execute = sub.add_parser('execute', help='Requires successful smoke; dispatch all pending frozen runs')
    execute.add_argument('--campaign', required=True)
    execute.add_argument('--smoke-report', required=True)
    execute.add_argument('--resume', action='store_true')
    summary = sub.add_parser('summarize', help='Read-only inputs; write a new immutable report directory')
    summary.add_argument('--campaign', required=True)
    summary.add_argument('--output')
    summary.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'freeze':
            plan = freeze_campaign(args.output, args.phase,
                {category: getattr(args, category+'_root') for category in ARTIFACT_CATEGORIES},
                args.order_seed, args.workers, args.ros_domain, args.gazebo_port,
                json.loads(args.runner_command_json) if args.runner_command_json else None,
                args.process_timeout_seconds)
            print(_json({'manifest': str(Path(args.output).resolve()/'manifest.json'),
                         'runs': len(plan['runs']), 'frozen_sha256': plan['frozen_sha256']}))
        elif args.command == 'execute':
            result = execute_campaign(args.campaign, args.smoke_report, args.resume)
            print(_json(result))
            return 0 if result['n_invalid_evidence'] == 0 else 2
        else:
            output = summarize_campaign(args.campaign, args.output, not args.no_plots)
            print(_json({'report': str(output/'report.md')}))
    except (ValueError, FileExistsError, FileNotFoundError, json.JSONDecodeError) as exc:
        parser.exit(2, str(exc)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
