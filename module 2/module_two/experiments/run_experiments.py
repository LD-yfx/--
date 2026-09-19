#!/usr/bin/env python3
"""Reproducible mechanism experiments; never a claim of localization accuracy.

All modes see identical geometry, quality observations, start/goal, footprint and
events. A hidden, manually assigned quality field is an independent diagnostic,
NOT a measured localization error or a learned probability of failure.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE / 'ros2_ws' / 'src' / 'quality_aware_navigation'))
from quality_aware_navigation.grid import GridSpec, QualityGrid, load_module_one
from quality_aware_navigation.model import CostModel, CostParameters
from quality_aware_navigation.planning import astar, inflate_obstacles, path_is_valid, Replanner

MODES = CostModel.MODES
SCENARIOS = ('dual_route', 'sparse_misleading_score', 'stale_high_score',
             'high_variance', 'unobserved_direct_route', 'no_path')
# Fixed engineering choices, committed before test seeds. No fitting on results.
PARAMETERS = CostParameters(beta=1., gamma=2., sample_scale=5., age_scale=30.,
                            unknown_risk=.7, max_cost=200, mean_only_floor=.7)
WEIGHT = 3.
RADIUS = .35
AS_OF = 400.  # Simulated/source seconds, not the host wall clock.


@dataclass
class Scene:
    name: str
    seed: int
    occupied: np.ndarray
    quality: QualityGrid
    latent_quality: np.ndarray
    start: tuple = (4, 10)
    goal: tuple = (42, 10)
    as_of: float = AS_OF


def make_scene(name: str, seed: int) -> Scene:
    if name not in SCENARIOS and name != 'dynamic':
        raise ValueError(name)
    rng = np.random.default_rng(seed)
    spec = GridSpec(47, 28, 1.)
    occupied = np.ones(spec.shape, dtype=bool)
    occupied[8:13, 2:45] = False  # Direct route.
    occupied[20:25, 2:45] = False  # Longer alternative.
    occupied[8:25, 2:8] = False
    occupied[8:25, 39:45] = False
    g = QualityGrid.empty(spec)
    g.source = 'manually_constructed_mechanism_scene'
    g.known[:] = ~occupied
    g.mean[:] = np.clip(.9 + rng.normal(0, .008, spec.shape), 0, 1)
    g.variance[:] = .0004
    g.count[:] = rng.integers(45, 65, spec.shape)
    g.last_observed[:] = AS_OF
    latent = g.mean.copy()
    direct = np.zeros(spec.shape, dtype=bool)
    direct[8:13, 10:37] = True
    if name != 'dynamic':
        latent[direct] = np.clip(.2 + rng.normal(0, .025, int(direct.sum())), 0, 1)
        if name in ('dual_route', 'no_path'):
            g.mean[direct] = latent[direct]
        elif name == 'sparse_misleading_score':
            g.mean[direct] = .97
            g.count[direct] = 1
            g.variance[direct] = 0.
        elif name == 'stale_high_score':
            g.mean[direct] = .97
            g.last_observed[direct] = AS_OF - 300
        elif name == 'high_variance':
            # Feasible moments of bounded quality samples, not an arbitrary
            # variance inconsistent with a near-one mean.
            samples = np.array([1.] * 40 + [.25] * 10)
            g.mean[direct] = float(samples.mean())
            g.variance[direct] = float(samples.var(ddof=1))
            g.count[direct] = len(samples)
        elif name == 'unobserved_direct_route':
            g.known[direct] = False
            g.count[direct] = 0
            g.mean[direct] = 0
    if name == 'no_path':
        occupied[:, 23] = True
    return Scene(name, seed, occupied, g, latent)


def integrate(path, values, resolution):
    return sum(math.dist(a, b) * resolution *
               (float(values[a[1], a[0]]) + float(values[b[1], b[0]])) / 2
               for a, b in zip(path, path[1:]))


def path_metrics(path, scene, blocked, costs):
    if not path:
        return dict(success=False, length_m=None, latent_quality_mean=None,
                    synthetic_low_quality_exposure_m=None, normalized_cost_integral=None,
                    unknown_exposure_m=None, collision_cells=0, valid_path=False,
                    route='no_path')
    length = sum(math.dist(a, b) * scene.quality.spec.resolution for a, b in zip(path, path[1:]))
    latent_mean = (integrate(path, scene.latent_quality, scene.quality.spec.resolution) / length
                   if length else float(scene.latent_quality[path[0][1], path[0][0]]))
    return dict(success=True, length_m=length, latent_quality_mean=latent_mean,
                synthetic_low_quality_exposure_m=integrate(path, scene.latent_quality < .4, scene.quality.spec.resolution),
                normalized_cost_integral=integrate(path, costs.astype(float) / 252., scene.quality.spec.resolution),
                unknown_exposure_m=integrate(path, ~scene.quality.known, scene.quality.spec.resolution),
                collision_cells=sum(bool(blocked[y, x]) for x, y in path),
                valid_path=path_is_valid(path, blocked),
                route='detour' if max(y for _, y in path) >= 20 else 'direct')


def run_static(seeds, parameters=PARAMETERS):
    rows, paths = [], []
    for name in SCENARIOS:
        for seed in seeds:
            scene = make_scene(name, seed)
            blocked = inflate_obstacles(scene.occupied, scene.quality.spec.resolution, RADIUS)
            for mode in MODES:
                model = CostModel(parameters, mode)
                costs = model.costs(scene.quality, scene.as_of)
                result = astar(blocked, costs, scene.start, scene.goal, scene.quality.spec.resolution, WEIGHT)
                row = dict(scenario=name, seed=seed, mode=mode,
                           objective=result.objective if result.success else None,
                           planning_ms=result.elapsed_ms, expanded=result.expanded,
                           **path_metrics(result.path, scene, blocked, costs))
                rows.append(row)
                paths.append(dict(scenario=name, seed=seed, mode=mode, path=result.path))
    return rows, paths


def run_events(seeds):
    """Snapshot replanning only: these are not moving-robot control experiments."""
    rows, paths = [], []
    for series in ('dynamic_obstacles', 'quality_change'):
        for seed in seeds:
            for mode in MODES:
                replanner = Replanner(resolution=1., weight=WEIGHT, min_interval=1., improvement_fraction=.03)
                previous = None
                changes = 0
                if series == 'dynamic_obstacles':
                    events = ((0., 'initial'), (.1, 'direct_blocked'), (2., 'obstacle_removed'),
                              (2.1, 'all_routes_blocked'), (4., 'routes_reopened'))
                else:
                    events = ((0., 'initial'), (2., 'quality_deteriorates'), (4., 'quality_recovers'))
                for now, event in events:
                    scene = make_scene('dynamic', seed)
                    scene.as_of = AS_OF + now
                    scene.quality.last_observed[scene.quality.known] = scene.as_of
                    if event == 'direct_blocked':
                        scene.occupied[7:14, 23] = True
                    elif event == 'all_routes_blocked':
                        scene.occupied[:, 23] = True
                    elif event == 'quality_deteriorates':
                        scene.quality.mean[8:13, 10:37] = .05
                        scene.latent_quality[8:13, 10:37] = .05
                    blocked = inflate_obstacles(scene.occupied, 1., RADIUS)
                    costs = CostModel(PARAMETERS, mode).costs(scene.quality, scene.as_of)
                    began = time.perf_counter()
                    path, reason = replanner.update(blocked, costs, scene.start, scene.goal, now)
                    elapsed = (time.perf_counter()-began)*1000
                    changed = previous is not None and path != previous
                    changes += int(changed)
                    previous = list(path)
                    rows.append(dict(series=series, seed=seed, mode=mode, event=event,
                                     time_s=now, source_as_of=scene.as_of, reason=reason,
                                     route_changed=changed, cumulative_route_changes=changes,
                                     update_ms=elapsed, **path_metrics(path, scene, blocked, costs)))
                    paths.append(dict(series=series, seed=seed, mode=mode, event=event, path=path))
    return rows, paths


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_module_one(root, expected=27):
    root = Path(root)
    if not root.is_dir():
        return dict(status='not_available', root=str(root), expected=expected, files=[])
    files = []
    for path in sorted(root.glob('*/quality_grid.csv')):
        entry = dict(path=str(path), csv_sha256=sha256(path))
        try:
            grid = load_module_one(path)
            as_of = float(np.max(grid.last_observed[grid.known])) if grid.known.any() else 0.
            costs = CostModel(PARAMETERS).costs(grid, as_of)
            entry.update(status='passed', frame=grid.spec.frame, shape=list(grid.spec.shape),
                         resolution=grid.spec.resolution, known_cells=int(grid.known.sum()),
                         known_fraction=float(grid.known.mean()), as_of_source_seconds=as_of,
                         oldest_source_seconds=float(np.min(grid.last_observed[grid.known])) if grid.known.any() else None,
                         cost_min=int(costs.min()), cost_max=int(costs.max()),
                         metadata_sha256=sha256(path.with_name('quality_grid_metadata.json')))
        except Exception as exc:
            entry.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        files.append(entry)
    passed = len(files) == expected and all(f['status'] == 'passed' for f in files)
    return dict(status='passed' if passed else 'failed', root=str(root), expected=expected,
                found=len(files), passed=sum(f['status'] == 'passed' for f in files), files=files,
                limitation='Import/cost compatibility only. These trajectory quality grids contain no obstacle map; no navigation or localization improvement is inferred.')


def aggregate(rows):
    summary = []
    for scenario in SCENARIOS:
        for mode in MODES:
            group = [r for r in rows if r['scenario'] == scenario and r['mode'] == mode]
            successful = [r for r in group if r['success']]
            item = dict(scenario=scenario, mode=mode, runs=len(group), successes=len(successful),
                        success_rate=len(successful)/len(group),
                        collision_cells=sum(r['collision_cells'] for r in group),
                        invalid_successes=sum(r['success'] and not r['valid_path'] for r in group),
                        detour_fraction=sum(r['route'] == 'detour' for r in successful)/len(successful) if successful else None)
            for metric in ('length_m', 'latent_quality_mean', 'synthetic_low_quality_exposure_m',
                           'unknown_exposure_m', 'planning_ms'):
                values = [r[metric] for r in (group if metric == 'planning_ms' else successful)]
                item[metric+'_mean'] = float(np.mean(values)) if values else None
                item[metric+'_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else (0. if values else None)
            summary.append(item)
    return summary


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def make_figures(output, summary, paths, seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'geometric': '#666666', 'linear': '#2878b5', 'nonlinear': '#70a6ce',
              'full': '#d1495b', 'no_variance': '#edae49', 'no_count': '#66a182', 'no_age': '#8f77b5'}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, name in zip(axes.flat, SCENARIOS):
        scene = make_scene(name, seed)
        image = np.ma.array(scene.latent_quality, mask=scene.occupied)
        cmap = plt.get_cmap('RdYlGn').copy()
        cmap.set_bad('#202833')
        im = ax.imshow(image, origin='lower', vmin=0, vmax=1, cmap=cmap)
        for mode in ('geometric', 'linear', 'full'):
            item = next(r for r in paths if r['scenario'] == name and r['seed'] == seed and r['mode'] == mode)
            if item['path']:
                xs, ys = zip(*item['path'])
                ax.plot(xs, ys, color=colors[mode], label=mode, lw=2,
                        linestyle={'geometric': ':', 'linear': '--', 'full': '-'}[mode])
        ax.scatter(*scene.start, marker='o', color='white', edgecolors='black', s=30)
        ax.scatter(*scene.goal, marker='*', color='white', edgecolors='black', s=60)
        ax.set_title(name.replace('_', ' '), fontsize=10)
        ax.set_xlabel('x cell'); ax.set_ylabel('y cell')
        if name == 'dual_route':
            ax.legend(fontsize=8)
    fig.suptitle(f'Constructed quality field and planned paths (seed {seed}); not localization error', fontsize=13)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.7, pad=.015,
                 label='Manually assigned diagnostic quality (0 low, 1 high)')
    fig.savefig(output / 'routes.png', dpi=160)
    plt.close(fig)
    names = [x for x in SCENARIOS if x != 'no_path']
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    x = np.arange(len(names))
    width = .11
    for i, mode in enumerate(MODES):
        chosen = [next(r for r in summary if r['scenario'] == name and r['mode'] == mode) for name in names]
        for ax, metric, label in zip(axes,
                                    ('synthetic_low_quality_exposure_m', 'length_m'),
                                    ('Synthetic low-quality exposure (m)', 'Path length (m)')):
            ax.bar(x+(i-3)*width, [r[metric+'_mean'] for r in chosen], width,
                   yerr=[r[metric+'_std'] for r in chosen], label=mode, color=colors[mode], capsize=2)
            ax.set_ylabel(label)
            ax.set_xticks(x, [n.replace('_', '\n') for n in names], fontsize=9)
            ax.grid(axis='y', alpha=.2)
    axes[0].set_ylim(0, 33)
    axes[0].legend(ncol=7, fontsize=8, loc='upper center')
    fig.suptitle('All seeds retained; error bars are sample SD; manual stress scenarios only')
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)


def make_report(output, summary, rows, events, compatibility, seeds, gamma1_summary):
    lines = [
        '# 模块二离线机制实验', '',
        '**这些实验使用人为构造的质量场，验证代价映射、路线选择和重规划机制；没有运行独立定位器，不能据此报告真实定位误差改善。**', '',
        f'- 固定测试种子：`{seeds}`。每个场景、种子下，全部方法共享障碍、质量观测、起终点、机器人半径和事件。',
        f'- 参数：`{json.dumps(asdict(PARAMETERS))}`；规划权重 `{WEIGHT}`；栅格分辨率 `1 m`；机器人半径 `{RADIUS} m`，使用保守障碍膨胀。',
        '- 参数为事先固定的工程默认值；未根据测试结果调参，未执行自动参数标定。后续标定应使用独立开发集。',
        '- 主实验 gamma=2；同时预先指定 gamma=1 静态敏感性对照，所有种子和场景保持一致。gamma=1 时 linear 与 nonlinear 按定义一致；不从测试结果中选择指数。',
        '- 几何基线忽略质量；linear 使用 1-mean；nonlinear 使用 (1-mean)^gamma；full 加入方差、样本量和时效；其余模式逐项移除。',
        '- “潜在质量”是实验设计者手工指定的另一个诊断场，不由待测代价模型计算，也不是真实定位误差。低质量暴露是路径上潜在质量 < 0.4 的距离积分。',
        '- 稀疏、陈旧和高方差场景被有意构造成高分误导场景，属于针对性压力测试，不代表真实场景分布或普遍性能优势。',
        '- 高方差观测来自 40 个 1.0 和 10 个 0.25 的有界质量样本，其均值与样本方差为实际样本矩；单样本场景方差记为 0，由样本量项体现证据不足。',
        '- 时间使用场景或 CSV 的源时钟；CSV 参考时间取该图最新观测时间，不使用本机当前时间。',
        '- 各模式的 objective 定义不同，原始值仅供各自规划诊断，不作为跨模式定位性能提升证据。', '',
        '## 静态结果（均值 ± 样本标准差）', '',
        '| 场景 | 方法 | 成功/次数 | 路径长度 m | 人为低质量暴露 m | 潜在质量均值 | 规划耗时 ms |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    def fmt(row, metric):
        value = row[metric+'_mean']
        return '—' if value is None else f"{value:.3f} ± {row[metric+'_std']:.3f}"
    for row in summary:
        lines.append(f"| {row['scenario']} | {row['mode']} | {row['successes']}/{row['runs']} | "
                     f"{fmt(row, 'length_m')} | {fmt(row, 'synthetic_low_quality_exposure_m')} | "
                     f"{fmt(row, 'latent_quality_mean')} | {fmt(row, 'planning_ms')} |")
    lines += ['', '### 保留的负结果与代价', '']
    for name in SCENARIOS:
        if name == 'no_path':
            continue
        full = next(r for r in summary if r['scenario'] == name and r['mode'] == 'full')
        base = next(r for r in summary if r['scenario'] == name and r['mode'] == 'geometric')
        delta = full['synthetic_low_quality_exposure_m_mean'] - base['synthetic_low_quality_exposure_m_mean']
        if abs(delta) < 1e-9:
            lines.append(f"- `{name}`：full 与几何基线的人为低质量暴露相同，为 {full['synthetic_low_quality_exposure_m_mean']:.2f} m；当前配置没有改善该指标。")
        else:
            lines.append(f"- `{name}`：full 人为低质量暴露变化 {delta:+.2f} m；路径长度从 {base['length_m_mean']:.2f} m 变为 {full['length_m_mean']:.2f} m。此变化仅描述所构造场景的路线权衡。")
    collisions = sum(r['collision_cells'] for r in rows + events)
    invalid = sum(r['success'] and not r['valid_path'] for r in rows + events)
    lines += ['', f'所有静态与事件快照中的障碍格穿越数：**{collisions}**；成功却不满足邻接/禁止穿角约束的路径数：**{invalid}**。', '',
              '## 预先指定的指数敏感性对照', '',
              '仅比较同一 full 模型的 gamma=1 与 gamma=2，其他参数与输入保持不变。全部七种模式的 gamma=1 原始结果保存在 `gamma1_static_runs.csv`。', '',
              '| 场景 | gamma | 路径长度 m | 人为低质量暴露 m | 绕行比例 |',
              '|---|---:|---:|---:|---:|']
    for name in SCENARIOS:
        for gamma, group in ((1, gamma1_summary), (2, summary)):
            row = next(r for r in group if r['scenario'] == name and r['mode'] == 'full')
            detour = '—' if row['detour_fraction'] is None else f"{row['detour_fraction']:.2f}"
            lines.append(f"| {name} | {gamma} | {fmt(row, 'length_m')} | {fmt(row, 'synthetic_low_quality_exposure_m')} | {detour} |")
    lines += ['',
              '## 动态事件', '',
              '动态测试为固定起终点的地图快照重规划，未执行连续运动或传感器驱动控制，不构成完整局部避障/闭环导航验收。', '',
              '| 事件序列 | 事件 | 方法 | 有路次数/总数 | 路线变化次数 |',
              '|---|---|---|---:|---:|']
    keys = list(dict.fromkeys((r['series'], r['event'], r['mode']) for r in events))
    for series, event, mode in keys:
        group = [r for r in events if (r['series'], r['event'], r['mode']) == (series, event, mode)]
        lines.append(f"| {series} | {event} | {mode} | {sum(r['success'] for r in group)}/{len(group)} | {sum(r['route_changed'] for r in group)} |")
    lines += ['', '“路线变化次数”统计相邻快照输出路径的改变，包含停止与恢复，不等于控制器执行次数。无路场景的 0 成功率是预期安全停止，应与可通行场景分别阅读。', '',
              '## 模块一数据兼容性', '',
              f"状态：**{compatibility['status']}**；发现 `{compatibility.get('found', 0)}` 份，成功 `{compatibility.get('passed', 0)}` 份，预期 `{compatibility['expected']}` 份。", '',
              '仅校验实际 CSV/元数据读取、质量语义、统计量、源时钟和代价输出。模块一轨迹质量图不是障碍物地图，不在此虚构其导航起终点或定位真值。每份输入的 SHA256、覆盖率与时间范围见 `module_one_compatibility.json`。', '',
              '归档 CSV 使用 cumulative 累计统计；新增 ROS 稀疏统计接口默认使用 10 秒 / 256 条样本窗口。离线文件导入不把累计统计伪装成窗口统计；质量恢复的快照演示直接替换观测场，不能证明累计图具有相同响应速度。', '',
              '## 可复现文件与局限', '',
              '- `static_runs.csv` / `static_paths.json`：所有静态测试的原始结果与路径，包括失败。',
              '- `event_runs.csv` / `event_paths.json`：动态障碍、完全封路、移除障碍、质量恶化及恢复的全部快照。',
              '- `summary.csv`、`manifest.json`：汇总、种子、配置、代码哈希和运行环境。',
              '- `routes.png`、`comparison.png`：示例路径与全种子比较。',
              '- 样本标准差反映本测试集的种子变化，不是置信区间；微小耗时受解释器/系统负载影响。',
              '- 保留无差异与不利结果；不声明完整模型在所有场景更优。需要独立真实定位器、真值、闭环运动和独立测试数据才能验证定位收益。',
              '- 本报告不代替 C++ 插件、ROS 话题链路、局部控制器或真实机器人验证。', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=MODULE / 'results' / 'offline')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(100, 110)))
    parser.add_argument('--module-one-root', type=Path,
                        default=MODULE.parent / 'module_one_reference' / 'localization_quality_final' / 'results' / 'formal_training' / 'quality_maps')
    parser.add_argument('--expected-module-one-maps', type=int, default=27)
    parser.add_argument('--no-plots', action='store_true', help='Diagnostic runs only; final report requires plots.')
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('duplicate seeds would falsely inflate the sample count')
    args.output.mkdir(parents=True, exist_ok=True)
    rows, paths = run_static(args.seeds)
    gamma1_rows, gamma1_paths = run_static(args.seeds, replace(PARAMETERS, gamma=1.))
    events, event_paths = run_events(args.seeds)
    summary = aggregate(rows)
    gamma1_summary = aggregate(gamma1_rows)
    compatibility = check_module_one(args.module_one_root, args.expected_module_one_maps)
    write_csv(args.output / 'static_runs.csv', rows)
    write_csv(args.output / 'event_runs.csv', events)
    write_csv(args.output / 'summary.csv', summary)
    write_csv(args.output / 'gamma1_static_runs.csv', gamma1_rows)
    write_csv(args.output / 'gamma1_summary.csv', gamma1_summary)
    write_json(args.output / 'gamma1_static_paths.json', gamma1_paths)
    write_json(args.output / 'static_paths.json', paths)
    write_json(args.output / 'event_paths.json', event_paths)
    write_json(args.output / 'module_one_compatibility.json', compatibility)
    code = [Path(__file__), *(MODULE / 'ros2_ws' / 'src' / 'quality_aware_navigation' / 'quality_aware_navigation').glob('*.py')]
    write_json(args.output / 'manifest.json', dict(seeds=args.seeds, scenarios=SCENARIOS, modes=MODES,
               parameters=asdict(PARAMETERS), weight=WEIGHT, radius_m=RADIUS, resolution_m=1.,
               clock='source_seconds', synthetic_as_of=AS_OF, parameters_fitted=False,
               test_rows=len(rows), gamma_sensitivity=[1, 2], gamma1_test_rows=len(gamma1_rows),
               event_rows=len(events), python=sys.version, numpy=np.__version__,
               platform=platform.platform(), source_sha256={str(p.relative_to(MODULE)): sha256(p) for p in code},
               evidence_scope='manually constructed mechanism experiments; no real localization error measured'))
    make_report(args.output, summary, rows, events, compatibility, args.seeds, gamma1_summary)
    if not args.no_plots:
        make_figures(args.output, summary, paths, args.seeds[0])
    unsafe = any(r['success'] and (not r['valid_path'] or r['collision_cells']) for r in rows + gamma1_rows + events)
    print(json.dumps(dict(output=str(args.output), static_runs=len(rows), gamma1_static_runs=len(gamma1_rows), event_snapshots=len(events),
                          compatibility=compatibility['status'], unsafe_paths=unsafe), ensure_ascii=False))
    return 1 if unsafe or compatibility['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
