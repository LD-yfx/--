"""Offline trial plots from recorded evidence; no ROS or fitted alignment."""
import argparse
import hashlib
import json
import math
from pathlib import Path

QUALITY_FIELDS = ('q_visual', 'q_laser', 'q_fused')
INPUT_FILES = ('manifest.json', 'result.json', 'estimates.jsonl', 'truth.jsonl',
               'quality_pairs.jsonl', 'error_rows.jsonl', 'plans.jsonl')


def finite(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def read_jsonl(path):
    """Missing file is explicit in the input manifest; malformed data is fatal."""
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'expected object at {path}:{number}')
            rows.append(row)
    return rows


def split_segments(rows, fields=('x', 'y'), max_gap=.1, max_step=None, window=None):
    """Keep acquisition order and break at invalid data, resets and time reversal."""
    if finite(max_gap) is None or max_gap <= 0 or (
            max_step is not None and (finite(max_step) is None or max_step <= 0)):
        raise ValueError('finite positive gap/step limits required')
    segments, segment = [], []
    previous_marker = None
    for raw in rows:
        t = finite(raw.get('t', raw.get('timestamp')))
        values = [finite(raw.get(field)) for field in fields]
        marker = next((raw[key] for key in ('reset_sequence', 'reset_id', 'segment_id') if key in raw), None)
        valid = t is not None and all(value is not None for value in values)
        if window and t is not None and not window[0] <= t <= window[1]:
            valid = False
        discontinuity = raw.get('reset') is True or (
            marker is not None and previous_marker is not None and marker != previous_marker)
        if segment and valid:
            dt = t-segment[-1]['t']
            discontinuity |= dt <= 0 or dt > max_gap+1e-9
            if max_step is not None:
                discontinuity |= math.dist(values, [segment[-1][field] for field in fields]) > max_step
        if not valid or discontinuity:
            if segment:
                segments.append(segment)
            segment = []
        if valid:
            segment.append(dict(t=t, **dict(zip(fields, values))))
        if marker is not None:
            previous_marker = marker
    if segment:
        segments.append(segment)
    return segments


def missing_intervals(rows, fields, window, tolerance=.05):
    """Complement of measured timestamp support, with no interpolation through gaps."""
    observed = []
    for row in rows:
        t = finite(row.get('t', row.get('timestamp')))
        if t is not None and all(finite(row.get(field)) is not None for field in fields):
            left, right = max(window[0], t-tolerance), min(window[1], t+tolerance)
            if right > left:
                observed.append((left, right))
    missing, cursor = [], window[0]
    for left, right in sorted(observed):
        if left > cursor+1e-9:
            missing.append([cursor, left])
        cursor = max(cursor, right)
    if cursor < window[1]-1e-9:
        missing.append([cursor, window[1]])
    return missing


def fixed_transform(manifest):
    transform = manifest.get('frame_transform', {'x': 0., 'y': 0., 'yaw': 0.})
    if not isinstance(transform, dict):
        raise ValueError('fixed map-to-world transform must be an object')
    values = {key: finite(transform.get(key, 0.)) for key in ('x', 'y', 'yaw')}
    if any(value is None for value in values.values()):
        raise ValueError('fixed map-to-world transform must be finite')
    return values


def transform_xy(x, y, transform):
    c, s = math.cos(transform['yaw']), math.sin(transform['yaw'])
    return transform['x']+c*x-s*y, transform['y']+s*x+c*y


def transform_rows(rows, transform):
    result = []
    for raw in rows:
        row = dict(raw)
        x, y = finite(raw.get('x')), finite(raw.get('y'))
        if x is not None and y is not None:
            row['x'], row['y'] = transform_xy(x, y, transform)
        result.append(row)
    return result


def mission_window(manifest, result, streams):
    """Prefer declared mission/action boundaries, never silently call recording a mission."""
    for source in (result, manifest):
        start, end = finite(source.get('mission_start')), finite(source.get('mission_end'))
        if start is not None and end is not None and end > start:
            return (start, end), 'declared_mission'
    actions = result.get('leg_actions', [])
    starts = [finite(row.get('mission_start')) for row in actions if isinstance(row, dict)]
    ends = [finite(row.get('mission_end')) for row in actions if isinstance(row, dict)]
    starts, ends = [x for x in starts if x is not None], [x for x in ends if x is not None]
    if starts and ends and max(ends) > min(starts):
        return (min(starts), max(ends)), 'recorded_action_envelope'
    times = [finite(row.get('t', row.get('timestamp'))) for rows in streams for row in rows]
    times = [t for t in times if t is not None]
    if times and max(times) > min(times):
        return (min(times), max(times)), 'recording_window_only_mission_boundaries_missing'
    return (0., 1.), 'no_valid_time_interval'


def render_trial(trial, output=None, max_gap=.1, max_step=.5, quality_gap=.5, plan_stride=20):
    if (finite(max_gap) is None or max_gap <= 0 or finite(max_step) is None or max_step <= 0 or
            finite(quality_gap) is None or quality_gap <= 0 or type(plan_stride) is not int or plan_stride < 1):
        raise ValueError('finite positive gap/step and positive integer plan stride required')
    trial = Path(trial).expanduser().resolve()
    manifest = json.loads((trial/'manifest.json').read_text(encoding='utf-8'))
    result = json.loads((trial/'result.json').read_text(encoding='utf-8'))
    transform = fixed_transform(manifest)
    inputs = {name: {'exists': (trial/name).is_file(),
                    'sha256': hashlib.sha256((trial/name).read_bytes()).hexdigest() if (trial/name).is_file() else None}
              for name in INPUT_FILES}
    settings = dict(max_gap_seconds=max_gap, max_step_m=max_step,
                    quality_gap_seconds=quality_gap, plan_stride=plan_stride)
    renderer_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(dict(inputs=inputs, settings=settings, renderer_sha256=renderer_sha256), sort_keys=True).encode()).hexdigest()[:12]
    output = Path(output).expanduser().resolve() if output else trial/('plots-'+digest)
    if output.exists():
        raise FileExistsError(f'plot output already exists; use a new destination: {output}')
    streams = {name: read_jsonl(trial/(name+'.jsonl'))
               for name in ('estimates', 'truth', 'quality_pairs', 'error_rows', 'plans')}
    window, window_source = mission_window(manifest, result, streams.values())
    estimates = transform_rows(streams['estimates'], transform)
    truth = streams['truth']
    quality = []
    for raw in streams['quality_pairs']:
        row = dict(raw)
        for field in QUALITY_FIELDS:
            value = finite(row.get(field))
            row[field] = value if value is not None and 0 <= value <= 1 else None
        quality.append(row)
    errors = streams['error_rows'] if inputs['error_rows.jsonl']['exists'] else quality
    error_source = 'error_rows.jsonl' if inputs['error_rows.jsonl']['exists'] else 'quality_pairs.jsonl (lower-rate matched samples)'
    paths = {'truth': split_segments(truth, max_gap=max_gap, max_step=max_step, window=window),
             'estimates': split_segments(estimates, max_gap=max_gap, max_step=max_step, window=window)}
    audit = dict(schema_version=1, run_id=manifest.get('run_id'), source_trial=str(trial),
                 input_files=inputs, settings=settings, renderer_sha256=renderer_sha256, frame_transform=transform,
                 window=list(window), window_source=window_source, error_source=error_source,
                 recorded_result={k: result.get(k) for k in ('completed', 'valid_evidence', 'pose_metrics_valid',
                                                             'navigation_success', 'success')},
                 row_counts={key: len(value) for key, value in streams.items()},
                 trajectory_segments={name: len(segments) for name, segments in paths.items()},
                 notes=['No trajectory fitting or estimated truth. Coordinates use the fixed manifest transform.',
                        'No line crosses a time reversal/reset, invalid row, temporal gap or excessive adjacent spatial step.',
                        'Availability marks timestamp support, not localization correctness. Quality support uses its declared display tolerance.'],
                 missing_intervals={})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    output.mkdir(parents=True, exist_ok=False)
    title = ' / '.join(str(manifest.get(key, '')) for key in ('map_id', 'condition', 'scenario', 'method') if manifest.get(key))
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    metadata = manifest.get('metadata', {})
    rectangles = metadata.get('rectangles', [])
    for number, rect in enumerate(rectangles):
        if len(rect) != 4 or any(finite(value) is None for value in rect):
            raise ValueError('manifest rectangle must have four finite world coordinates')
        x0, x1, y0, y1 = map(float, rect)
        if x1 <= x0 or y1 <= y0:
            raise ValueError('manifest rectangle bounds are invalid')
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, color='#666666',
                              label='SDF static collision rectangles' if number == 0 else None))
    if not rectangles:
        audit['notes'].append('SDF rectangle metadata missing: geometry cannot be verified from this plot.')
        ax.text(.01, .99, 'Static geometry unavailable', transform=ax.transAxes, va='top', color='red')
    visible_plans = []
    for record in streams['plans'][::plan_stride]:
        t = finite(record.get('received_ros', record.get('t')))
        if t is None or not window[0] <= t <= window[1]:
            continue
        frame = str(record.get('frame', ''))
        if frame not in ('map', 'world'):
            continue
        # A plan is one spatial polyline, never joined to a preceding plan.
        points, segment = [], []
        for point in record.get('points', []):
            if not isinstance(point, (list, tuple)) or len(point) < 2 or any(finite(x) is None for x in point[:2]):
                if segment:
                    points.append(segment)
                segment = []
                continue
            xy = tuple(map(float, point[:2]))
            segment.append(transform_xy(*xy, transform) if frame == 'map' else xy)
        if segment:
            points.append(segment)
        for segment in points:
            ax.plot([p[0] for p in segment], [p[1] for p in segment], color='#9b59b6', alpha=.2, lw=.7,
                    label='Recorded plans (display sample)' if not visible_plans else None)
            visible_plans.append(segment)
    for name, color, label in (('truth', '#202020', 'Independent truth'), ('estimates', '#e67e22', 'Estimated pose')):
        for index, segment in enumerate(paths[name]):
            ax.plot([p['x'] for p in segment], [p['y'] for p in segment], color=color, lw=1.4,
                    marker='.' if len(segment) == 1 else None, label=label if index == 0 else None)
            ax.scatter([segment[0]['x'], segment[-1]['x']], [segment[0]['y'], segment[-1]['y']],
                       color=color, marker='x', s=16)
    for index, point in enumerate(manifest.get('planned_waypoints', []), 1):
        if not isinstance(point, (list, tuple)) or len(point) < 2 or any(finite(x) is None for x in point[:2]):
            continue
        x, y = transform_xy(*map(float, point[:2]), transform)
        ax.scatter([x], [y], marker='*', color='#2166ac', s=100, label='Fixed mission waypoint' if index == 1 else None)
        ax.annotate(str(index), (x, y), xytext=(5, 5), textcoords='offset points')
    bounds = metadata.get('bounds')
    if isinstance(bounds, list) and len(bounds) == 4 and all(finite(x) is not None for x in bounds):
        ax.set_xlim(bounds[0]-.3, bounds[1]+.3)
        ax.set_ylim(bounds[2]-.3, bounds[3]+.3)
    ax.set(aspect='equal', xlabel='World x (m)', ylabel='World y (m)',
           title=title+f"\n{window_source}; success={result.get('success')}, valid_evidence={result.get('valid_evidence')}")
    ax.grid(alpha=.2)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc='upper left', fontsize=8)
    fig.supxlabel(f"Crosses mark segment endpoints. Breaks: gap > {max_gap:g} s or step > {max_step:g} m. "
                  f"Plans: every {plan_stride}th record.\n"
                  "Static rectangles only; dynamic obstacle occupancy is not inferred.", fontsize=8)
    fig.savefig(output/'trajectory.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True,
                             gridspec_kw={'height_ratios': [2, 2, 1.5]})
    for rows, field, color, label, gap in ((errors, 'error_position_m', '#202020', 'Current position error', max_gap),
            (quality, 'future_1s_max_error_m', '#2166ac', 'Next 1 s maximum (complete horizon only)', quality_gap)):
        for index, segment in enumerate(split_segments(rows, (field,), max_gap=gap, window=window)):
            axes[0].plot([r['t'] for r in segment], [r[field] for r in segment],
                         color=color, lw=.9, marker='.' if len(segment) == 1 else None,
                         label=label if index == 0 else None)
    for field, color in zip(QUALITY_FIELDS, ('#1b9e77', '#7570b3', '#d95f02')):
        for index, segment in enumerate(split_segments(quality, (field,), max_gap=quality_gap, window=window)):
            axes[1].plot([r['t'] for r in segment], [r[field] for r in segment],
                         color=color, lw=.9, marker='.' if len(segment) == 1 else None,
                         label=field if index == 0 else None)
    lanes = [('truth', truth, ('x', 'y'), .05), ('estimate', estimates, ('x', 'y'), .05),
             ('paired error', errors, ('error_position_m',), .05)]+[
             (field, quality, (field,), quality_gap/2.) for field in QUALITY_FIELDS]
    for index, (name, rows, fields, tolerance) in enumerate(lanes):
        gaps = missing_intervals(rows, fields, window, tolerance=tolerance)
        audit['missing_intervals'][name] = gaps
        axes[2].broken_barh([(window[0], window[1]-window[0])], (index-.35, .7), facecolors='#c7e9c0')
        axes[2].broken_barh([(a, b-a) for a, b in gaps], (index-.35, .7), facecolors='#ef8a62')
    for a, b in audit['missing_intervals']['paired error']:
        axes[0].axvspan(a, b, color='#bbbbbb', alpha=.3)
    axes[0].axhline(.3, color='#d73027', ls='--', lw=.7, label='0.30 m threshold')
    axes[0].set(ylabel='Position error (m)', title=title+
                f"\nvalid_evidence={result.get('valid_evidence')}, success={result.get('success')} | Recorded error and quality; gaps remain visible")
    axes[1].set(ylabel='Quality score', ylim=(-.03, 1.03))
    axes[2].set(yticks=list(range(len(lanes))), yticklabels=[lane[0] for lane in lanes],
                xlabel='Source clock time (s)', xlim=window,
                title='Timestamp support: green = available, orange = missing (not an accuracy score)')
    for ax in axes[:2]:
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc='upper right')
    fig.savefig(output/'error_quality_timeline.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    audit['plans_displayed_segments'] = len(visible_plans)
    (output/'render_manifest.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--output')
    parser.add_argument('--max-gap-seconds', type=float, default=.1)
    parser.add_argument('--max-step-m', type=float, default=.5)
    parser.add_argument('--quality-gap-seconds', type=float, default=.5)
    parser.add_argument('--plan-stride', type=int, default=20)
    args = parser.parse_args(argv)
    try:
        output = render_trial(args.trial, args.output, args.max_gap_seconds, args.max_step_m,
                              args.quality_gap_seconds, args.plan_stride)
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc)+'\n')
    print(json.dumps({'plots': str(output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
