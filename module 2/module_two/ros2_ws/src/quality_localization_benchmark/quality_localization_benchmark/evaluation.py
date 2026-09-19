"""Pure offline evaluation against isolated truth, with explicit missing data.

Coordinates in estimates are transformed map -> world using the fixed manifest
transform. No fitted trajectory alignment, ROS dependency, control, or synthetic
quality-to-error rule is used here. Error summaries on observed samples never
substitute for time coverage: ``success`` additionally requires valid evaluation.
"""
from bisect import bisect_left, bisect_right
import math
from statistics import mean

MATCH_SECONDS = 0.05
MIN_COVERAGE = 0.95
POSITION_THRESHOLD_M = 0.30
YAW_THRESHOLD_RAD = math.radians(10.)
QUALITY_FIELDS = ('q_laser', 'q_visual', 'q_fused')


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def _quantile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    offset = (len(ordered)-1)*probability
    lo = math.floor(offset)
    hi = math.ceil(offset)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(offset-lo)


def _statistics(values, unit):
    scale = max(values) if values else 0.
    return {
        'n': len(values),
        'rmse_'+unit: scale*math.sqrt(mean((v/scale)**2 for v in values)) if scale else (0. if values else None),
        'median_'+unit: _quantile(values, .5),
        'p95_'+unit: _quantile(values, .95),
        'max_'+unit: max(values) if values else None,
    }


def _empty_result(manifest):
    # Preserve only finite, simple identifiers, not arbitrary NaN-bearing input.
    identity = {}
    for key in ('run_id', 'map_id', 'seed', 'condition', 'method'):
        value = manifest.get(key)
        identity[key] = value if isinstance(value, (str, int)) else _number(value)
    yaw = _statistics([], 'rad')
    yaw.update({key+'_deg': None for key in ('rmse', 'median', 'p95', 'max')})
    return dict(identity, schema_version=1, data_status='unavailable',
                evaluation_valid=False, navigation_success=False, success=False,
                failure_reasons=[], warnings=[], position_error=_statistics([], 'm'),
                yaw_error=yaw, error_rows=[], quality_pairs=[],
                estimate_coverage=None, raw_estimate_coverage=None, truth_coverage=None,
                estimate_missing_seconds=None, estimate_missing_intervals=[],
                estimate_dropout_count=None, estimate_dropout_seconds=None,
                estimate_dropout_intervals=[], truth_path_length_m=None,
                partial_truth_path_length_m=None, truth_path_complete=False, goal_distance_m=None,
                goal_yaw_error_rad=None, wall_seconds=_number(manifest.get('wall_seconds')),
                mission_seconds=None, collision_count=None, action_succeeded=False,
                quality_status='unavailable', input_statistics={},
                position_exceedance_fraction_observed=None,
                pose_exceedance_fraction_observed=None,
                position_exceedance_seconds_observed=None,
                pose_exceedance_seconds_observed=None,
                position_exceedance_fraction_observed_time=None,
                pose_exceedance_fraction_observed_time=None,
                alignment={'max_time_difference_s': MATCH_SECONDS,
                           'minimum_time_coverage': MIN_COVERAGE,
                           'method': 'bounded linear/shortest-angle interpolation, otherwise nearest within tolerance',
                           'frame_transform': None})


def _clean_poses(rows, start, end):
    clean = []
    stats = {'input_rows': 0, 'outside_mission': 0, 'invalid_rows': 0,
             'duplicate_timestamps': 0, 'time_reversals': 0, 'reset_events': 0}
    previous_time = None
    previous_marker = None
    for row in rows or []:
        stats['input_rows'] += 1
        if not isinstance(row, dict):
            stats['invalid_rows'] += 1
            continue
        t = _number(row.get('t'))
        if t is None:
            stats['invalid_rows'] += 1
            continue
        if t < start-1e-9 or t > end+1e-9:
            stats['outside_mission'] += 1
            continue
        marker = next((row[k] for k in ('reset_sequence', 'reset_id', 'segment_id') if k in row), None)
        if previous_time is not None and t < previous_time-1e-9:
            stats['time_reversals'] += 1
        if ((previous_marker is not None and marker is not None and marker != previous_marker)
                or (row.get('reset') is True and t > start+1e-9)):
            stats['reset_events'] += 1
        previous_time = t
        if marker is not None:
            previous_marker = marker
        pose = {key: _number(row.get(key)) for key in ('x', 'y', 'yaw')}
        if any(value is None for value in pose.values()):
            stats['invalid_rows'] += 1
            continue
        pose['t'] = t
        pose['yaw'] = _angle(pose['yaw'])
        if clean and abs(t-clean[-1]['t']) <= 1e-12:
            clean[-1] = pose  # A repeated publication cannot inflate sample size.
            stats['duplicate_timestamps'] += 1
        else:
            clean.append(pose)
    return clean, stats


def _pose_at(rows, times, t):
    """Never interpolate across a >100 ms observation gap; never extrapolate."""
    if not rows:
        return None
    index = bisect_left(times, t)
    if index < len(rows) and abs(times[index]-t) <= 1e-9:
        return dict(rows[index], pairing='exact', time_difference_s=0.)
    if 0 < index < len(rows):
        before, after = rows[index-1], rows[index]
        left, right = t-before['t'], after['t']-t
        if max(left, right) <= MATCH_SECONDS+1e-9:
            alpha = left/(after['t']-before['t'])
            return {'t': t, 'x': before['x']+alpha*(after['x']-before['x']),
                    'y': before['y']+alpha*(after['y']-before['y']),
                    'yaw': _angle(before['yaw']+alpha*_angle(after['yaw']-before['yaw'])),
                    'pairing': 'interpolated', 'time_difference_s': max(left, right)}
    candidates = [rows[i] for i in (index-1, index) if 0 <= i < len(rows)]
    nearest = min(candidates, key=lambda row: abs(row['t']-t))
    delta = abs(nearest['t']-t)
    if delta <= MATCH_SECONDS+1e-9:
        return dict(nearest, pairing='nearest', time_difference_s=delta)
    return None


def _coverage(times, start, end):
    """Measure covered wall/source-time intervals, not the fraction of rows."""
    intervals = sorted((max(start, t-MATCH_SECONDS), min(end, t+MATCH_SECONDS))
                       for t in times if start-MATCH_SECONDS <= t <= end+MATCH_SECONDS)
    merged = []
    for left, right in intervals:
        if right < left:
            continue
        if merged and left <= merged[-1][1]+1e-9:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    missing = []
    cursor = start
    for left, right in merged:
        if left > cursor+1e-9:
            missing.append([cursor, left])
        cursor = max(cursor, right)
    if cursor < end-1e-9:
        missing.append([cursor, end])
    uncovered = sum(b-a for a, b in missing)
    return max(0., min(1., 1-uncovered/(end-start))), uncovered, missing


def _dropouts(times, start, end):
    # A >0.5 s gap between publications is counted including endpoint gaps.
    boundaries = [start]+sorted(set(times))+[end]
    return [[a, b] for a, b in zip(boundaries, boundaries[1:]) if b-a > .5+1e-9]


def _exceedance_time(errors, start, end):
    covered = positional = pose = 0.
    for i, row in enumerate(errors):
        left = max(start, row['t']-MATCH_SECONDS)
        right = min(end, row['t']+MATCH_SECONDS)
        if i:
            left = max(left, (errors[i-1]['t']+row['t'])/2.)
        if i+1 < len(errors):
            right = min(right, (row['t']+errors[i+1]['t'])/2.)
        duration = max(0., right-left)
        covered += duration
        if row['error_position_m'] > POSITION_THRESHOLD_M:
            positional += duration
        if row['error_position_m'] > POSITION_THRESHOLD_M or row['error_yaw_rad'] > YAW_THRESHOLD_RAD:
            pose += duration
    return covered, positional, pose


def _transform(row, transform):
    c, s = math.cos(transform['yaw']), math.sin(transform['yaw'])
    result = {'t': row['t'], 'x': transform['x']+c*row['x']-s*row['y'],
              'y': transform['y']+s*row['x']+c*row['y'],
              'yaw': _angle(row['yaw']+transform['yaw'])}
    return result if all(math.isfinite(v) for v in result.values()) else None


def _action_succeeded(status):
    if isinstance(status, str):
        return status.strip().upper() in ('4', 'SUCCEEDED', 'STATUS_SUCCEEDED')
    return not isinstance(status, bool) and status == 4


def _quality_pairs(quality_rows, errors, start, end, identity):
    records, previous_time = [], None
    reversed_time = False
    by_time = {}
    for row in quality_rows or []:
        if not isinstance(row, dict):
            continue
        t = _number(row.get('timestamp', row.get('t')))
        if t is None or not start-1e-9 <= t <= end+1e-9:
            continue
        if previous_time is not None and t < previous_time-1e-9:
            reversed_time = True
        previous_time = t
        by_time[t] = row
    if reversed_time:
        return [], 'clock_discontinuity'
    error_times = [r['t'] for r in errors]
    for t, row in sorted(by_time.items()):
        record = dict(identity, t=t, pair_status='missing_estimate_or_truth',
                      matched_error_t=None, time_difference_s=None,
                      error_position_m=None, error_yaw_rad=None,
                      future_1s_max_error_m=None, future_1s_max_yaw_error_rad=None,
                      future_1s_status='insufficient_horizon', future_1s_coverage=None)
        for field in QUALITY_FIELDS:
            score = _number(row.get(field))
            record[field] = score if score is not None and 0 <= score <= 1 else None
        index = bisect_left(error_times, t)
        candidates = [errors[i] for i in (index-1, index) if 0 <= i < len(errors)]
        if candidates:
            nearest = min(candidates, key=lambda item: abs(item['t']-t))
            delta = abs(nearest['t']-t)
            if delta <= MATCH_SECONDS+1e-9:
                record.update(pair_status='matched', matched_error_t=nearest['t'],
                              time_difference_s=delta,
                              error_position_m=nearest['error_position_m'],
                              error_yaw_rad=nearest['error_yaw_rad'])
        if t+1. <= end+1e-9:
            future = errors[bisect_left(error_times, t-1e-9):bisect_right(error_times, t+1.+1e-9)]
            coverage, _, _ = _coverage([e['t'] for e in future], t, t+1.)
            record['future_1s_coverage'] = coverage
            record['future_1s_status'] = 'matched' if future and coverage >= MIN_COVERAGE-1e-9 else 'insufficient_coverage'
            if record['future_1s_status'] == 'matched':
                record['future_1s_max_error_m'] = max(e['error_position_m'] for e in future)
                record['future_1s_max_yaw_error_rad'] = max(e['error_yaw_rad'] for e in future)
        records.append(record)
    status = 'unavailable' if not records else ('available' if any(r[k] is not None for r in records for k in QUALITY_FIELDS) else 'no_valid_quality_scores')
    return records, status


def evaluate_episode(manifest, estimates, truth, quality_rows):
    """Evaluate one monotonic episode; return only JSON-serializable finite data.

    Required manifest: mission_start, mission_end, goal=[x,y], action_status,
    collision_count. Optional goal_yaw and frame_transform use radians.
    reset_sequence/reset_id/segment_id changes or ``reset: true`` inside the
    mission invalidate interpolation: split such recordings into separate runs.
    ``navigation_success`` and coverage-qualified ``success`` are distinct.
    """
    manifest = manifest if isinstance(manifest, dict) else {}
    result = _empty_result(manifest)
    start, end = _number(manifest.get('mission_start')), _number(manifest.get('mission_end'))
    goal = manifest.get('goal', [])
    transform_input = manifest.get('frame_transform') or {}
    transform = {k: _number(transform_input.get(k, 0.)) for k in ('x', 'y', 'yaw')} if isinstance(transform_input, dict) else {}
    valid_goal = isinstance(goal, (list, tuple)) and len(goal) >= 2 and all(_number(x) is not None for x in goal[:2])
    if (start is None or end is None or end <= start or not math.isfinite(end-start) or not valid_goal or
            ('goal_yaw' in manifest and _number(manifest['goal_yaw']) is None) or
            len(transform) != 3 or any(v is None for v in transform.values())):
        result['data_status'] = 'invalid_manifest'
        result['failure_reasons'] = ['finite positive mission interval, goal and fixed frame transform required']
        return result
    goal = [_number(x) for x in goal[:2]]
    transform['yaw'] = _angle(transform['yaw'])
    result['mission_seconds'] = end-start
    if result['wall_seconds'] is not None and result['wall_seconds'] < 0:
        result['wall_seconds'] = None
        result['warnings'].append('negative wall_seconds is invalid and was not used')
    result['alignment']['frame_transform'] = transform
    collision = _number(manifest.get('collision_count'))
    if collision is not None and collision >= 0 and collision.is_integer():
        result['collision_count'] = int(collision)
    else:
        result['failure_reasons'].append('collision_count_missing_or_invalid')
    result['action_succeeded'] = _action_succeeded(manifest.get('action_status'))
    est, est_stats = _clean_poses(estimates, start, end)
    gt, gt_stats = _clean_poses(truth, start, end)
    result['input_statistics'] = {'estimates': est_stats, 'truth': gt_stats}
    if any(stats['time_reversals'] or stats['reset_events'] for stats in (est_stats, gt_stats)):
        result['data_status'] = 'clock_or_reset_discontinuity'
        result['failure_reasons'].append('split_episode_at_time_reversal_or_reset')
        return result
    for name, stats in (('estimates', est_stats), ('truth', gt_stats)):
        if stats['invalid_rows']:
            result['warnings'].append(f'{name}: {stats["invalid_rows"]} invalid rows excluded and counted')
        if stats['duplicate_timestamps']:
            result['warnings'].append(f'{name}: duplicate publications deduplicated')
    gt_times = [r['t'] for r in gt]
    errors = []
    for original in est:
        transformed = _transform(original, transform)
        paired = _pose_at(gt, gt_times, original['t'])
        if paired is None or transformed is None:
            continue
        position_error = math.hypot(transformed['x']-paired['x'], transformed['y']-paired['y'])
        if not math.isfinite(position_error):
            result['warnings'].append('nonfinite coordinate arithmetic excluded from paired coverage')
            continue
        errors.append({'t': original['t'],
                       'error_position_m': position_error,
                       'error_yaw_rad': abs(_angle(transformed['yaw']-paired['yaw'])),
                       'truth_pairing': paired['pairing'],
                       'truth_time_difference_s': paired['time_difference_s']})
    result['error_rows'] = errors
    result['position_error'] = _statistics([r['error_position_m'] for r in errors], 'm')
    result['yaw_error'] = _statistics([r['error_yaw_rad'] for r in errors], 'rad')
    for key in ('rmse', 'median', 'p95', 'max'):
        value = result['yaw_error'][key+'_rad']
        result['yaw_error'][key+'_deg'] = math.degrees(value) if value is not None else None
    paired_times = [r['t'] for r in errors]
    coverage, missing, intervals = _coverage(paired_times, start, end)
    truth_coverage, _, _ = _coverage(gt_times, start, end)
    raw_est_coverage, _, _ = _coverage([r['t'] for r in est], start, end)
    dropouts = _dropouts([r['t'] for r in est], start, end)
    result.update(estimate_coverage=coverage, raw_estimate_coverage=raw_est_coverage,
                  truth_coverage=truth_coverage, estimate_missing_seconds=missing,
                  estimate_missing_intervals=intervals, estimate_dropout_count=len(dropouts),
                  estimate_dropout_seconds=sum(b-a for a, b in dropouts),
                  estimate_dropout_intervals=dropouts)
    first, last = _pose_at(gt, gt_times, start), _pose_at(gt, gt_times, end)
    partial_length = sum(math.hypot(b['x']-a['x'], b['y']-a['y'])
                         for a, b in zip(gt, gt[1:]) if b['t']-a['t'] <= 2*MATCH_SECONDS+1e-9)
    result['partial_truth_path_length_m'] = partial_length if gt and math.isfinite(partial_length) else None
    result['truth_path_complete'] = bool(gt and first is not None and last is not None and
        math.isfinite(partial_length) and all(b['t']-a['t'] <= 2*MATCH_SECONDS+1e-9 for a, b in zip(gt, gt[1:])))
    if result['truth_path_complete']:
        result['truth_path_length_m'] = partial_length
    if last is not None:
        distance = math.hypot(last['x']-goal[0], last['y']-goal[1])
        result['goal_distance_m'] = distance if math.isfinite(distance) else None
        goal_yaw = _number(manifest.get('goal_yaw'))
        if goal_yaw is not None:
            result['goal_yaw_error_rad'] = abs(_angle(last['yaw']-goal_yaw))
    result['navigation_success'] = bool(result['action_succeeded'] and result['collision_count'] == 0
        and result['goal_distance_m'] is not None and result['goal_distance_m'] <= POSITION_THRESHOLD_M+1e-9
        and (result['goal_yaw_error_rad'] is None or result['goal_yaw_error_rad'] <= math.radians(15.)+1e-9))
    result['evaluation_valid'] = bool(errors and coverage >= MIN_COVERAGE-1e-9
        and truth_coverage >= MIN_COVERAGE-1e-9 and first is not None and last is not None)
    result['success'] = result['navigation_success'] and result['evaluation_valid']
    result['data_status'] = 'valid' if result['evaluation_valid'] else 'insufficient_coverage'
    if not result['action_succeeded']:
        result['failure_reasons'].append('navigation_action_not_succeeded')
    if result['collision_count'] and result['collision_count'] > 0:
        result['failure_reasons'].append('collision')
    if result['goal_distance_m'] is None:
        result['failure_reasons'].append('missing_terminal_truth')
    elif result['goal_distance_m'] > POSITION_THRESHOLD_M+1e-9:
        result['failure_reasons'].append('true_goal_not_reached')
    if result['goal_yaw_error_rad'] is not None and result['goal_yaw_error_rad'] > math.radians(15.)+1e-9:
        result['failure_reasons'].append('true_goal_heading_not_reached')
    if coverage < MIN_COVERAGE-1e-9:
        result['failure_reasons'].append('insufficient_estimate_truth_pair_coverage')
    if truth_coverage < MIN_COVERAGE-1e-9:
        result['failure_reasons'].append('insufficient_truth_coverage')
    result['position_exceedance_fraction_observed'] = (sum(r['error_position_m'] > POSITION_THRESHOLD_M for r in errors)/len(errors)) if errors else None
    result['pose_exceedance_fraction_observed'] = (sum(r['error_position_m'] > POSITION_THRESHOLD_M or r['error_yaw_rad'] > YAW_THRESHOLD_RAD for r in errors)/len(errors)) if errors else None
    covered_seconds, positional_seconds, pose_seconds = _exceedance_time(errors, start, end)
    result['position_exceedance_seconds_observed'] = positional_seconds if errors else None
    result['pose_exceedance_seconds_observed'] = pose_seconds if errors else None
    result['position_exceedance_fraction_observed_time'] = positional_seconds/covered_seconds if covered_seconds else None
    result['pose_exceedance_fraction_observed_time'] = pose_seconds/covered_seconds if covered_seconds else None
    result['quality_pairs'], result['quality_status'] = _quality_pairs(quality_rows, errors, start, end,
        dict({k: result[k] for k in ('run_id', 'map_id', 'seed', 'condition', 'method')},
             episode_evaluation_valid=result['evaluation_valid']))
    return result


def _ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.] * len(values)
    i = 0
    while i < len(order):
        j = i+1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        average = (i+1+j)/2.
        for index in order[i:j]:
            ranks[index] = average
        i = j
    return ranks


def _spearman(xs, ys):
    if len(xs) < 2:
        return None
    x, y = _ranks(xs), _ranks(ys)
    mx, my = mean(x), mean(y)
    numerator = sum((a-mx)*(b-my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
    return max(-1., min(1., numerator/denominator)) if denominator else None


def _relation(rows, quality_key, error_key):
    valid = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        q, e = _number(row.get(quality_key)), _number(row.get(error_key))
        if q is not None and 0 <= q <= 1 and e is not None and e >= 0:
            valid.append((q, e))
    rho = _spearman([q for q, _ in valid], [e for _, e in valid])
    bins = []
    for i in range(10):
        values = [e for q, e in valid if min(9, math.floor(q*10)) == i]
        bins.append({'lower': i/10., 'upper': (i+1)/10., 'upper_inclusive': i == 9,
                     'n': len(values), 'low_sample_count': len(values) < 10,
                     'error_median_m': _quantile(values, .5),
                     'error_p95_m': _quantile(values, .95),
                     'position_exceedance_fraction': sum(e > POSITION_THRESHOLD_M for e in values)/len(values) if values else None})
    return {'n': len(valid), 'spearman_rho': rho,
            'status': 'available' if rho is not None else 'insufficient_or_constant_data', 'bins': bins}


def summarize_quality_relation(rows):
    """Descriptive frame-level correlations only; no IID-frame significance claim.

    Input is the concatenation of evaluate_episode(...)["quality_pairs"]. Call
    separately per estimator/map/condition for subgroup reports. Bootstrap is
    intentionally absent until implemented at the complete-run level.
    """
    rows = list(rows or [])
    run_ids = sorted({str(row['run_id']) for row in rows if isinstance(row, dict) and row.get('run_id') is not None})
    return {'schema_version': 1, 'rows': len(rows), 'run_count': len(run_ids),
            'run_ids': run_ids, 'position_threshold_m': POSITION_THRESHOLD_M,
            'uncertainty_status': 'descriptive_only_no_frame_iid_inference',
            'fields': {field: {'current': _relation(rows, field, 'error_position_m'),
                              'future_1s': _relation(rows, field, 'future_1s_max_error_m')}
                       for field in QUALITY_FIELDS}}
