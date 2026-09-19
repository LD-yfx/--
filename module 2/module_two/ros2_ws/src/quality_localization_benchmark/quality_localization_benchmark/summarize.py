"""Run-level campaign summaries: all denominators, paired effects, no causal claim."""
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean

from .evaluation import summarize_quality_relation

QUALITY_FIELDS = ('q_visual', 'q_laser', 'q_fused')
SCATTER_STRIDE = 10
GROUP_FIELDS = ('phase', 'map_id', 'condition', 'method', 'scenario')

METRICS = {
    'position_rmse_m': ('position_error', 'rmse_m'),
    'position_p95_m': ('position_error', 'p95_m'),
    'yaw_rmse_deg': ('yaw_error', 'rmse_deg'),
    'estimate_coverage': ('estimate_coverage',),
    'truth_path_length_m': ('truth_path_length_m',),
    'mission_seconds': ('mission_seconds',),
    'collision_count': ('collision_count',),
    'wall_seconds': ('wall_seconds',),
}


def _finite(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def metric(result, name):
    value = result
    for key in METRICS[name]:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _finite(value)


def _percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    offset = (len(values)-1)*p
    i, j = math.floor(offset), math.ceil(offset)
    return values[i]+(values[j]-values[i])*(offset-i)


def paired_bootstrap(differences, seed=20260919, draws=2000):
    """Resample complete paired runs; one frame/leg is never a bootstrap unit."""
    values = [_finite(v) for v in differences]
    if any(v is None for v in values):
        raise ValueError('paired differences must be finite')
    if draws < 100:
        raise ValueError('at least 100 bootstrap draws required')
    if not values:
        return {'n_pairs': 0, 'mean_difference': None, 'ci95': None,
                'status': 'no_pairs', 'bootstrap_unit': 'complete_paired_run'}
    if len(values) < 2:
        return {'n_pairs': 1, 'mean_difference': values[0], 'ci95': None,
                'status': 'too_few_runs_for_interval', 'bootstrap_unit': 'complete_paired_run'}
    rng = random.Random(seed)
    samples = [mean(rng.choices(values, k=len(values))) for _ in range(draws)]
    return {'n_pairs': len(values), 'mean_difference': mean(values),
            'ci95': [_percentile(samples, .025), _percentile(samples, .975)],
            'status': 'descriptive_paired_bootstrap', 'bootstrap_unit': 'complete_paired_run',
            'seed': seed, 'draws': draws}


def _qualified(result):
    return bool(result.get('completed') is True and result.get('valid_evidence') is True
                and pose_metrics_valid(result) and result.get('success') is True)


def pose_metrics_valid(result):
    # Estimator dropout is an algorithm outcome, not invalid architecture evidence.
    return result.get('pose_metrics_valid', result.get('evaluation_valid')) is True


def _numbers(values):
    numbers = [v for v in values if v is not None]
    return {'n': len(numbers), 'mean': mean(numbers) if numbers else None,
            'median': _percentile(numbers, .5), 'p95': _percentile(numbers, .95)}


def _group_key(run):
    return tuple(run.get(field, '') for field in GROUP_FIELDS)


def _comparison(by_pair, comparator, bootstrap_seed, draws, scenario):
    comparison = {'candidate': 'full', 'comparator': comparator, 'scenario': scenario,
                  'difference_direction': 'full_minus_comparator',
                  'n_planned_pairs': len(by_pair), 'n_executed_pairs': 0,
                  'n_pairs_with_invalid_evidence': 0, 'metrics': {}}
    usable = []
    for key, methods in sorted(by_pair.items()):
        a, b = methods.get('full'), methods.get(comparator)
        if a is None or b is None:
            continue
        comparison['n_executed_pairs'] += 1
        comparison['n_pairs_with_invalid_evidence'] += int(
            a.get('valid_evidence') is not True or b.get('valid_evidence') is not True)
        usable.append((key, a, b))
    comparison['qualified_success_difference_all_executed_pairs'] = paired_bootstrap(
        [float(_qualified(a))-float(_qualified(b)) for _, a, b in usable], bootstrap_seed, draws)
    for name in METRICS:
        scopes = {}
        for scope in ('both_valid_evaluation', 'both_qualified_success'):
            differences, identifiers = [], []
            for key, a, b in usable:
                eligible = (_qualified(a) and _qualified(b)) if scope == 'both_qualified_success' else (
                    a.get('valid_evidence') is True and b.get('valid_evidence') is True and
                    pose_metrics_valid(a) and pose_metrics_valid(b))
                av, bv = metric(a, name), metric(b, name)
                if eligible and av is not None and bv is not None:
                    differences.append(av-bv)
                    identifiers.append({'map_id': key[0], 'seed': key[1],
                                        'condition': key[2], 'scenario': key[3]})
            scopes[scope] = dict(paired_bootstrap(differences, bootstrap_seed, draws),
                                 pair_identifiers=identifiers)
        comparison['metrics'][name] = scopes
    return comparison


def _scatter_sample(rows):
    fields = QUALITY_FIELDS+('error_position_m', 'future_1s_max_error_m')
    selected = [{key: _finite(row.get(key)) for key in fields} for row in rows[::SCATTER_STRIDE]]
    return {'stride': SCATTER_STRIDE, 'selection': 'every_nth_source_row_before_filtering_any_score_or_error',
            'n_source_rows': len(rows), 'n_display_rows': len(selected),
            'display_fraction': len(selected)/len(rows) if rows else 0.,
            'statistics_use': 'all_source_rows; sampling_is_display_only', 'rows': selected}


def summarize_records(plan, records, bootstrap_seed=20260919, draws=2000):
    """Records map planned run_id -> executed result; missing runs remain planned."""
    groups, quality_groups, audit_rows = {}, {}, []
    planned = plan['runs']
    if len({r['run_id'] for r in planned}) != len(planned) or set(records)-{r['run_id'] for r in planned}:
        raise ValueError('duplicate planned ids or unplanned result records')
    for run in planned:
        result = records.get(run['run_id'])
        group_key = _group_key(run)
        group = groups.setdefault(group_key, {'planned': [], 'executed': []})
        group['planned'].append(run)
        if result is not None:
            group['executed'].append(result)
        audit = dict(run_id=run['run_id'], phase=run['phase'], map_id=run['map_id'],
                     seed=run['seed'], condition=run['condition'], method=run['method'], scenario=run.get('scenario', ''),
                     executed=result is not None, valid_evidence=False,
                     navigation_success=False, qualified_success=False,
                     pose_metrics_valid=False, failure_reasons='not_executed')
        if result is not None:
            audit.update(valid_evidence=result.get('valid_evidence') is True,
                         navigation_success=result.get('navigation_success') is True,
                         qualified_success=_qualified(result),
                         pose_metrics_valid=pose_metrics_valid(result),
                         failure_reasons='; '.join(str(x) for x in
                             list(result.get('failure_reasons') or [])+
                             list(result.get('invalid_reasons') or [])))
        for name in METRICS:
            observed = metric(result, name) if result is not None else None
            if name.startswith(('position_', 'yaw_')):
                audit['observed_'+name] = observed
                audit[name] = observed if result is not None and result.get('valid_evidence') is True and pose_metrics_valid(result) else None
            else:
                audit[name] = observed
        audit_rows.append(audit)
        if result is None or result.get('valid_evidence') is not True:
            continue
        pairs = result.get('quality_pairs')
        if not isinstance(pairs, list):
            pairs = [p for leg in result.get('legs', []) if isinstance(leg, dict)
                     for p in leg.get('quality_pairs', [])]
        key = _group_key(run)
        for pair in pairs:
            if isinstance(pair, dict):
                # An outbound/return leg is not an independent run.
                normalized = dict(pair, run_id=run['run_id'], map_id=run['map_id'],
                                  condition=run['condition'], method=run['method'], scenario=run.get('scenario', ''))
                quality_groups.setdefault(key, []).append(normalized)
    group_reports = []
    for key, group in sorted(groups.items()):
        executed = group['executed']
        valid = [r for r in executed if r.get('valid_evidence') is True]
        successful = [r for r in executed if _qualified(r)]
        evaluated = [r for r in valid if pose_metrics_valid(r)]
        report = dict(zip(GROUP_FIELDS, key))
        report.update(n_planned=len(group['planned']), n_executed=len(executed),
                      n_not_executed=len(group['planned'])-len(executed),
                      n_invalid_evidence=len(executed)-len(valid),
                      n_failed_executed=len(executed)-len(successful),
                      n_algorithm_failure_valid_evidence=len(valid)-len(successful),
                      n_pose_metrics_invalid_valid_evidence=len(valid)-len(evaluated),
                      n_valid_evaluation=len(evaluated),
                      n_navigation_success=sum(r.get('navigation_success') is True for r in executed),
                      n_qualified_success=len(successful),
                      qualified_success_fraction_of_planned=len(successful)/len(group['planned']),
                      qualified_success_fraction_of_executed=len(successful)/len(executed) if executed else None)
        report['all_observed_valid_evidence_metrics'] = {
            name: _numbers([metric(r, name) for r in valid]) for name in METRICS}
        report['valid_evaluation_metrics'] = {
            name: _numbers([metric(r, name) for r in evaluated]) for name in METRICS}
        report['qualified_success_only_metrics'] = {
            name: _numbers([metric(r, name) for r in successful]) for name in METRICS}
        group_reports.append(report)
    comparisons = []
    test_runs = [r for r in planned if r['phase'] in ('test', 'dynamic')]
    scenarios = sorted({r.get('scenario', '') for r in test_runs})
    for scenario in scenarios:
        by_pair = {}
        selected = [r for r in test_runs if r.get('scenario', '') == scenario]
        for run in selected:
            key = (run['map_id'], run['seed'], run['condition'], scenario)
            by_pair.setdefault(key, {})[run['method']] = records.get(run['run_id'])
        comparators = ('geometric', 'linear') if scenario else ('geometric', 'linear', 'no_variance', 'no_count', 'no_age')
        for comparator in comparators:
            comparisons.append(_comparison(by_pair, comparator, bootstrap_seed, draws, scenario))
    quality = [dict(zip(GROUP_FIELDS, key), relation=summarize_quality_relation(rows),
                    scatter_sample=_scatter_sample(rows))
               for key, rows in sorted(quality_groups.items())]
    return {'schema_version': 1, 'phase': plan['phase'],
            'n_planned': len(planned), 'n_executed': len(records),
            'n_not_executed': len(planned)-len(records),
            'matrix_executed': len(records) == len(planned),
            'n_invalid_evidence': sum(r.get('valid_evidence') is not True for r in records.values()),
            'n_failed_executed': sum(not _qualified(r) for r in records.values()),
            'n_algorithm_failure_valid_evidence': sum(r.get('valid_evidence') is True and not _qualified(r) for r in records.values()),
            'n_pose_metrics_invalid_valid_evidence': sum(r.get('valid_evidence') is True and not pose_metrics_valid(r) for r in records.values()),
            'n_qualified_success': sum(_qualified(r) for r in records.values()),
            'groups': group_reports, 'paired_comparisons': comparisons,
            'quality_relations': quality, 'runs': audit_rows,
            'limitations': [
                'AMCL measures laser localization navigation; multimodal quality scoring is not multimodal fused localization.',
                'Quality correlations are observational, not evidence of causation.',
                'Paired bootstrap resamples complete runs, not frames or mission legs.',
                'Few maps and seeds limit generalization; intervals do not establish broad significance.',
                'All planned/executed/invalid denominators are retained. Successful-only metrics are conditional.',
                'All-observed metrics may cover failed runs only before termination and must not be treated as mission-wide accuracy.',
                'mission_seconds uses the simulation source clock; wall_seconds reports compute-time performance separately.',
            ]}


def write_summary(summary, output, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (output/'runs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary['runs'][0]))
        writer.writeheader()
        writer.writerows(summary['runs'])
    lines = [
        '# 独立定位研究实验汇总', '',
        '**这是 AMCL/实际配置定位器下的导航评估；多模态质量评分不等于多模态融合定位。相关性不证明因果。**', '',
        f"计划 {summary['n_planned']} 次；执行 {summary['n_executed']} 次；未执行 {summary['n_not_executed']} 次；证据无效 {summary['n_invalid_evidence']} 次；有效试验中定位评价不完整 {summary['n_pose_metrics_invalid_valid_evidence']} 次；合格成功 {summary['n_qualified_success']} 次。", '',
        '| 地图 | 条件 / 动态场景 | 方法 | 已执行/计划 | 无效 | 合格成功/已执行 | 定位覆盖合格RMSE均值 m | 成功子集RMSE均值 m |',
        '|---|---|---|---:|---:|---:|---:|---:|',
    ]
    for g in summary['groups']:
        rmse = g['qualified_success_only_metrics']['position_rmse_m']['mean']
        formatted = '—' if rmse is None else f'{rmse:.4f}'
        primary = g['valid_evaluation_metrics']['position_rmse_m']['mean']
        primary_text = '—' if primary is None else f'{primary:.4f}'
        context = g['condition']+(' / '+g['scenario'] if g.get('scenario') else '')
        lines.append(f"| {g['map_id']} | {context} | {g['method']} | {g['n_executed']}/{g['n_planned']} | {g['n_invalid_evidence']} | {g['n_qualified_success']}/{g['n_executed']} | {primary_text} | {formatted} |")
    lines += ['', '成功子集指标具有条件性。全部失败/无效运行仍在分母和 runs.csv 中；summary.json 另保留全部可观测片段、有效评价和成功子集三种口径。', '',
              '## 配对比较', '',
              '差值方向固定为 full 减对照。误差/长度/时间差为负表示该指标较低，但较短失败路线不能被解释成收益。', '',
              '| 场景 | 对照 | 已执行配对/计划 | 含无效证据配对 | 合格成功率差 | 95%区间 |',
              '|---|---|---:|---:|---:|---|']
    for comparison in summary['paired_comparisons']:
        effect = comparison['qualified_success_difference_all_executed_pairs']
        delta = effect['mean_difference']
        lines.append(f"| {comparison.get('scenario') or 'static'} | {comparison['comparator']} | {comparison['n_executed_pairs']}/{comparison['n_planned_pairs']} | {comparison['n_pairs_with_invalid_evidence']} | {'—' if delta is None else f'{delta:.4f}'} | {effect['ci95']} |")
    lines += ['', '连续指标配对结果和逐对标识见 summary.json。重采样单位是完整运行；去程与回程不作两个独立样本。质量关系按地图、条件、方法分组，十个质量区间保留空格与低样本标记。', '',
              f'散点图在筛选质量值或误差之前，固定每 {SCATTER_STRIDE} 行取 1 行；各图注明实际显示比例。相关系数与十箱统计始终使用全量数据，抽样仅用于显示。动态三种场景分别配对汇总。', '',
              '## 失败与证据无效运行', '']
    unsuccessful = [r for r in summary['runs'] if r['executed'] and not r['qualified_success']]
    if not unsuccessful:
        lines.append('当前已执行运行中无此类记录；仍需结合未执行数量判断矩阵是否完成。')
    for row in unsuccessful:
        lines.append(f"- {row['run_id']}: {row['failure_reasons'] or 'not_qualified_success'}")
    lines += ['', '## 解释范围', '']+['- '+x for x in summary['limitations']]
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if plots:
        make_plots(summary, output)


def make_plots(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    relationships = summary['quality_relations']
    for page in range(0, len(relationships), 8):
        selected = relationships[page:page+8]
        fig, axes = plt.subplots(math.ceil(len(selected)/2), 2,
                                 figsize=(13, max(3, 2.8*math.ceil(len(selected)/2))), squeeze=False,
                                 constrained_layout=True)
        for ax in axes.flat:
            ax.set_visible(False)
        for ax, group in zip(axes.flat, selected):
            ax.set_visible(True)
            for field, details in group['relation']['fields'].items():
                bins = details['current']['bins']
                if any(b['n'] for b in bins):
                    ax.plot([(b['lower']+b['upper'])/2 for b in bins],
                            [b['error_median_m'] if b['n'] else None for b in bins], marker='o', label=field)
            ax.set(title=_plot_title(group),
                   xlabel='Module-one quality score', ylabel='Median position error (m)', xlim=(0, 1))
            if ax.lines:
                ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.suptitle('Descriptive calibration only; no causal or multimodal fusion claim')
        fig.savefig(Path(output)/f'quality_calibration_{page//8+1:02d}.png', dpi=140)
        plt.close(fig)
    for number, group in enumerate(relationships, 1):
        sample = group.get('scatter_sample')
        if sample is None:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for column, field in enumerate(QUALITY_FIELDS):
            for index, (error_key, scope) in enumerate((('error_position_m', 'current'),
                    ('future_1s_max_error_m', 'future_1s'))):
                ax = axes[index, column]
                pairs = [(row.get(field), row.get(error_key)) for row in sample['rows']]
                pairs = [(q, error) for q, error in pairs if q is not None and 0 <= q <= 1 and error is not None and error >= 0]
                if pairs:
                    ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], s=7, alpha=.35)
                detail = group['relation']['fields'][field][scope]
                rho = detail['spearman_rho']
                ax.set(title=f"{scope}: all valid n={detail['n']}, rho={rho if rho is None else round(rho, 3)}",
                       xlabel=field, ylabel='Position error (m)' if index == 0 else 'Next 1 s maximum error (m)',
                       xlim=(0, 1))
                ax.grid(alpha=.2)
        fig.suptitle(_plot_title(group)+'\n'+
            f"Display stride={sample['stride']}: {sample['n_display_rows']}/{sample['n_source_rows']} source rows "
            f"({sample['display_fraction']:.1%}); statistics use all rows. Observational only.")
        fig.savefig(Path(output)/f'quality_scatter_{number:02d}.png', dpi=140)
        plt.close(fig)


def _plot_title(group):
    return ' / '.join(str(group.get(key, '')) for key in ('map_id', 'condition', 'method', 'scenario') if group.get(key))
