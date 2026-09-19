# 模块二离线机制实验

**这些实验使用人为构造的质量场，验证代价映射、路线选择和重规划机制；没有运行独立定位器，不能据此报告真实定位误差改善。**

- 固定测试种子：`[100, 101, 102, 103, 104, 105, 106, 107, 108, 109]`。每个场景、种子下，全部方法共享障碍、质量观测、起终点、机器人半径和事件。
- 参数：`{"beta": 1.0, "gamma": 2.0, "sample_scale": 5.0, "age_scale": 30.0, "unknown_risk": 0.7, "max_cost": 200, "mean_only_floor": 0.7}`；规划权重 `3.0`；栅格分辨率 `1 m`；机器人半径 `0.35 m`，使用保守障碍膨胀。
- 参数为事先固定的工程默认值；未根据测试结果调参，未执行自动参数标定。后续标定应使用独立开发集。
- 主实验 gamma=2；同时预先指定 gamma=1 静态敏感性对照，所有种子和场景保持一致。gamma=1 时 linear 与 nonlinear 按定义一致；不从测试结果中选择指数。
- 几何基线忽略质量；linear 使用 1-mean；nonlinear 使用 (1-mean)^gamma；full 加入方差、样本量和时效；其余模式逐项移除。
- “潜在质量”是实验设计者手工指定的另一个诊断场，不由待测代价模型计算，也不是真实定位误差。低质量暴露是路径上潜在质量 < 0.4 的距离积分。
- 稀疏、陈旧和高方差场景被有意构造成高分误导场景，属于针对性压力测试，不代表真实场景分布或普遍性能优势。
- 高方差观测来自 40 个 1.0 和 10 个 0.25 的有界质量样本，其均值与样本方差为实际样本矩；单样本场景方差记为 0，由样本量项体现证据不足。
- 时间使用场景或 CSV 的源时钟；CSV 参考时间取该图最新观测时间，不使用本机当前时间。
- 各模式的 objective 定义不同，原始值仅供各自规划诊断，不作为跨模式定位性能提升证据。

## 静态结果（均值 ± 样本标准差）

| 场景 | 方法 | 成功/次数 | 路径长度 m | 人为低质量暴露 m | 潜在质量均值 | 规划耗时 ms |
|---|---|---:|---:|---:|---:|---:|
| dual_route | geometric | 10/10 | 38.000 ± 0.000 | 27.000 ± 0.000 | 0.403 ± 0.004 | 0.510 ± 0.201 |
| dual_route | linear | 10/10 | 56.485 ± 0.000 | 0.000 ± 0.000 | 0.901 ± 0.001 | 5.287 ± 4.375 |
| dual_route | nonlinear | 10/10 | 56.485 ± 0.000 | 0.000 ± 0.000 | 0.900 ± 0.001 | 6.490 ± 5.067 |
| dual_route | full | 10/10 | 56.485 ± 0.000 | 0.000 ± 0.000 | 0.901 ± 0.001 | 3.649 ± 3.928 |
| dual_route | no_variance | 10/10 | 56.485 ± 0.000 | 0.000 ± 0.000 | 0.900 ± 0.001 | 2.735 ± 3.073 |
| dual_route | no_count | 10/10 | 56.485 ± 0.000 | 0.000 ± 0.000 | 0.901 ± 0.001 | 4.702 ± 4.611 |
| dual_route | no_age | 10/10 | 56.485 ± 0.000 | 0.000 ± 0.000 | 0.901 ± 0.001 | 4.722 ± 4.987 |
| sparse_misleading_score | geometric | 10/10 | 38.000 ± 0.000 | 27.000 ± 0.000 | 0.403 ± 0.004 | 1.303 ± 2.992 |
| sparse_misleading_score | linear | 10/10 | 38.000 ± 0.000 | 27.000 ± 0.000 | 0.403 ± 0.004 | 0.857 ± 0.230 |
| sparse_misleading_score | nonlinear | 10/10 | 38.000 ± 0.000 | 27.000 ± 0.000 | 0.403 ± 0.004 | 0.387 ± 0.091 |
| sparse_misleading_score | full | 10/10 | 56.485 ± 0.000 | 0.000 ± 0.000 | 0.901 ± 0.001 | 4.014 ± 4.054 |
| sparse_misleading_score | no_variance | 10/10 | 56.485 ± 0.000 | 0.000 ± 0.000 | 0.900 ± 0.001 | 6.556 ± 5.117 |
| sparse_misleading_score | no_count | 10/10 | 38.000 ± 0.000 | 27.000 ± 0.000 | 0.403 ± 0.004 | 1.299 ± 2.829 |
| sparse_misleading_score | no_age | 10/10 | 56.485 ± 0.000 | 0.000 ± 0.000 | 0.901 ± 0.001 | 3.814 ± 4.135 |
| stale_high_score | geometric | 10/10 | 38.000 ± 0.000 | 27.000 ± 0.000 | 0.403 ± 0.004 | 0.461 ± 0.129 |
| stale_high_score | linear | 10/10 | 38.000 ± 0.000 | 27.000 ± 0.000 | 0.403 ± 0.004 | 1.981 ± 3.135 |
| stale_high_score | nonlinear | 10/10 | 38.000 ± 0.000 | 27.000 ± 0.000 | 0.403 ± 0.004 | 1.114 ± 2.083 |
| stale_high_score | full | 10/10 | 56.485 ± 0.000 | 0.000 ± 0.000 | 0.901 ± 0.001 | 3.956 ± 3.636 |
| stale_high_score | no_variance | 10/10 | 56.485 ± 0.000 | 0.000 ± 0.000 | 0.900 ± 0.001 | 4.395 ± 4.214 |
| stale_high_score | no_count | 10/10 | 56.485 ± 0.000 | 0.000 ± 0.000 | 0.901 ± 0.001 | 2.706 ± 2.823 |
| stale_high_score | no_age | 10/10 | 38.000 ± 0.000 | 27.000 ± 0.000 | 0.403 ± 0.004 | 0.637 ± 0.135 |
| high_variance | geometric | 10/10 | 38.000 ± 0.000 | 27.000 ± 0.000 | 0.403 ± 0.004 | 0.411 ± 0.040 |
| high_variance | linear | 10/10 | 38.000 ± 0.000 | 27.000 ± 0.000 | 0.403 ± 0.004 | 2.241 ± 3.299 |
| high_variance | nonlinear | 10/10 | 38.000 ± 0.000 | 27.000 ± 0.000 | 0.403 ± 0.004 | 1.863 ± 3.288 |
| high_variance | full | 10/10 | 38.000 ± 0.000 | 27.000 ± 0.000 | 0.403 ± 0.004 | 3.599 ± 4.585 |
| high_variance | no_variance | 10/10 | 38.000 ± 0.000 | 27.000 ± 0.000 | 0.403 ± 0.004 | 0.785 ± 0.180 |
| high_variance | no_count | 10/10 | 38.000 ± 0.000 | 27.000 ± 0.000 | 0.403 ± 0.004 | 3.115 ± 3.312 |
| high_variance | no_age | 10/10 | 38.000 ± 0.000 | 27.000 ± 0.000 | 0.403 ± 0.004 | 2.415 ± 3.176 |
| unobserved_direct_route | geometric | 10/10 | 38.000 ± 0.000 | 27.000 ± 0.000 | 0.403 ± 0.004 | 0.941 ± 1.729 |
| unobserved_direct_route | linear | 10/10 | 56.485 ± 0.000 | 0.000 ± 0.000 | 0.901 ± 0.001 | 4.924 ± 4.837 |
| unobserved_direct_route | nonlinear | 10/10 | 56.485 ± 0.000 | 0.000 ± 0.000 | 0.900 ± 0.001 | 1.731 ± 0.291 |
| unobserved_direct_route | full | 10/10 | 56.485 ± 0.000 | 0.000 ± 0.000 | 0.901 ± 0.001 | 5.900 ± 5.089 |
| unobserved_direct_route | no_variance | 10/10 | 56.485 ± 0.000 | 0.000 ± 0.000 | 0.900 ± 0.001 | 2.973 ± 3.036 |
| unobserved_direct_route | no_count | 10/10 | 56.485 ± 0.000 | 0.000 ± 0.000 | 0.901 ± 0.001 | 3.770 ± 4.153 |
| unobserved_direct_route | no_age | 10/10 | 56.485 ± 0.000 | 0.000 ± 0.000 | 0.901 ± 0.001 | 5.046 ± 4.775 |
| no_path | geometric | 0/10 | — | — | — | 1.390 ± 0.287 |
| no_path | linear | 0/10 | — | — | — | 4.373 ± 4.827 |
| no_path | nonlinear | 0/10 | — | — | — | 2.383 ± 2.251 |
| no_path | full | 0/10 | — | — | — | 2.342 ± 1.593 |
| no_path | no_variance | 0/10 | — | — | — | 1.615 ± 1.329 |
| no_path | no_count | 0/10 | — | — | — | 2.141 ± 2.933 |
| no_path | no_age | 0/10 | — | — | — | 2.105 ± 2.440 |

### 保留的负结果与代价

- `dual_route`：full 人为低质量暴露变化 -27.00 m；路径长度从 38.00 m 变为 56.49 m。此变化仅描述所构造场景的路线权衡。
- `sparse_misleading_score`：full 人为低质量暴露变化 -27.00 m；路径长度从 38.00 m 变为 56.49 m。此变化仅描述所构造场景的路线权衡。
- `stale_high_score`：full 人为低质量暴露变化 -27.00 m；路径长度从 38.00 m 变为 56.49 m。此变化仅描述所构造场景的路线权衡。
- `high_variance`：full 与几何基线的人为低质量暴露相同，为 27.00 m；当前配置没有改善该指标。
- `unobserved_direct_route`：full 人为低质量暴露变化 -27.00 m；路径长度从 38.00 m 变为 56.49 m。此变化仅描述所构造场景的路线权衡。

所有静态与事件快照中的障碍格穿越数：**0**；成功却不满足邻接/禁止穿角约束的路径数：**0**。

## 预先指定的指数敏感性对照

仅比较同一 full 模型的 gamma=1 与 gamma=2，其他参数与输入保持不变。全部七种模式的 gamma=1 原始结果保存在 `gamma1_static_runs.csv`。

| 场景 | gamma | 路径长度 m | 人为低质量暴露 m | 绕行比例 |
|---|---:|---:|---:|---:|
| dual_route | 1 | 56.485 ± 0.000 | 0.000 ± 0.000 | 1.00 |
| dual_route | 2 | 56.485 ± 0.000 | 0.000 ± 0.000 | 1.00 |
| sparse_misleading_score | 1 | 56.485 ± 0.000 | 0.000 ± 0.000 | 1.00 |
| sparse_misleading_score | 2 | 56.485 ± 0.000 | 0.000 ± 0.000 | 1.00 |
| stale_high_score | 1 | 56.485 ± 0.000 | 0.000 ± 0.000 | 1.00 |
| stale_high_score | 2 | 56.485 ± 0.000 | 0.000 ± 0.000 | 1.00 |
| high_variance | 1 | 38.000 ± 0.000 | 27.000 ± 0.000 | 0.00 |
| high_variance | 2 | 38.000 ± 0.000 | 27.000 ± 0.000 | 0.00 |
| unobserved_direct_route | 1 | 56.485 ± 0.000 | 0.000 ± 0.000 | 1.00 |
| unobserved_direct_route | 2 | 56.485 ± 0.000 | 0.000 ± 0.000 | 1.00 |
| no_path | 1 | — | — | — |
| no_path | 2 | — | — | — |

## 动态事件

动态测试为固定起终点的地图快照重规划，未执行连续运动或传感器驱动控制，不构成完整局部避障/闭环导航验收。

| 事件序列 | 事件 | 方法 | 有路次数/总数 | 路线变化次数 |
|---|---|---|---:|---:|
| dynamic_obstacles | initial | geometric | 10/10 | 0 |
| dynamic_obstacles | direct_blocked | geometric | 10/10 | 10 |
| dynamic_obstacles | obstacle_removed | geometric | 10/10 | 10 |
| dynamic_obstacles | all_routes_blocked | geometric | 0/10 | 10 |
| dynamic_obstacles | routes_reopened | geometric | 10/10 | 10 |
| dynamic_obstacles | initial | linear | 10/10 | 0 |
| dynamic_obstacles | direct_blocked | linear | 10/10 | 10 |
| dynamic_obstacles | obstacle_removed | linear | 10/10 | 10 |
| dynamic_obstacles | all_routes_blocked | linear | 0/10 | 10 |
| dynamic_obstacles | routes_reopened | linear | 10/10 | 10 |
| dynamic_obstacles | initial | nonlinear | 10/10 | 0 |
| dynamic_obstacles | direct_blocked | nonlinear | 10/10 | 10 |
| dynamic_obstacles | obstacle_removed | nonlinear | 10/10 | 10 |
| dynamic_obstacles | all_routes_blocked | nonlinear | 0/10 | 10 |
| dynamic_obstacles | routes_reopened | nonlinear | 10/10 | 10 |
| dynamic_obstacles | initial | full | 10/10 | 0 |
| dynamic_obstacles | direct_blocked | full | 10/10 | 10 |
| dynamic_obstacles | obstacle_removed | full | 10/10 | 10 |
| dynamic_obstacles | all_routes_blocked | full | 0/10 | 10 |
| dynamic_obstacles | routes_reopened | full | 10/10 | 10 |
| dynamic_obstacles | initial | no_variance | 10/10 | 0 |
| dynamic_obstacles | direct_blocked | no_variance | 10/10 | 10 |
| dynamic_obstacles | obstacle_removed | no_variance | 10/10 | 10 |
| dynamic_obstacles | all_routes_blocked | no_variance | 0/10 | 10 |
| dynamic_obstacles | routes_reopened | no_variance | 10/10 | 10 |
| dynamic_obstacles | initial | no_count | 10/10 | 0 |
| dynamic_obstacles | direct_blocked | no_count | 10/10 | 10 |
| dynamic_obstacles | obstacle_removed | no_count | 10/10 | 10 |
| dynamic_obstacles | all_routes_blocked | no_count | 0/10 | 10 |
| dynamic_obstacles | routes_reopened | no_count | 10/10 | 10 |
| dynamic_obstacles | initial | no_age | 10/10 | 0 |
| dynamic_obstacles | direct_blocked | no_age | 10/10 | 10 |
| dynamic_obstacles | obstacle_removed | no_age | 10/10 | 10 |
| dynamic_obstacles | all_routes_blocked | no_age | 0/10 | 10 |
| dynamic_obstacles | routes_reopened | no_age | 10/10 | 10 |
| quality_change | initial | geometric | 10/10 | 0 |
| quality_change | quality_deteriorates | geometric | 10/10 | 0 |
| quality_change | quality_recovers | geometric | 10/10 | 0 |
| quality_change | initial | linear | 10/10 | 0 |
| quality_change | quality_deteriorates | linear | 10/10 | 10 |
| quality_change | quality_recovers | linear | 10/10 | 10 |
| quality_change | initial | nonlinear | 10/10 | 0 |
| quality_change | quality_deteriorates | nonlinear | 10/10 | 10 |
| quality_change | quality_recovers | nonlinear | 10/10 | 10 |
| quality_change | initial | full | 10/10 | 0 |
| quality_change | quality_deteriorates | full | 10/10 | 10 |
| quality_change | quality_recovers | full | 10/10 | 10 |
| quality_change | initial | no_variance | 10/10 | 0 |
| quality_change | quality_deteriorates | no_variance | 10/10 | 10 |
| quality_change | quality_recovers | no_variance | 10/10 | 10 |
| quality_change | initial | no_count | 10/10 | 0 |
| quality_change | quality_deteriorates | no_count | 10/10 | 10 |
| quality_change | quality_recovers | no_count | 10/10 | 10 |
| quality_change | initial | no_age | 10/10 | 0 |
| quality_change | quality_deteriorates | no_age | 10/10 | 10 |
| quality_change | quality_recovers | no_age | 10/10 | 10 |

“路线变化次数”统计相邻快照输出路径的改变，包含停止与恢复，不等于控制器执行次数。无路场景的 0 成功率是预期安全停止，应与可通行场景分别阅读。

## 模块一数据兼容性

状态：**passed**；发现 `27` 份，成功 `27` 份，预期 `27` 份。

仅校验实际 CSV/元数据读取、质量语义、统计量、源时钟和代价输出。模块一轨迹质量图不是障碍物地图，不在此虚构其导航起终点或定位真值。每份输入的 SHA256、覆盖率与时间范围见 `module_one_compatibility.json`。

归档 CSV 使用 cumulative 累计统计；新增 ROS 稀疏统计接口默认使用 10 秒 / 256 条样本窗口。离线文件导入不把累计统计伪装成窗口统计；质量恢复的快照演示直接替换观测场，不能证明累计图具有相同响应速度。

## 可复现文件与局限

- `static_runs.csv` / `static_paths.json`：所有静态测试的原始结果与路径，包括失败。
- `event_runs.csv` / `event_paths.json`：动态障碍、完全封路、移除障碍、质量恶化及恢复的全部快照。
- `summary.csv`、`manifest.json`：汇总、种子、配置、代码哈希和运行环境。
- `routes.png`、`comparison.png`：示例路径与全种子比较。
- 样本标准差反映本测试集的种子变化，不是置信区间；微小耗时受解释器/系统负载影响。
- 保留无差异与不利结果；不声明完整模型在所有场景更优。需要独立真实定位器、真值、闭环运动和独立测试数据才能验证定位收益。
- 本报告不代替 C++ 插件、ROS 话题链路、局部控制器或真实机器人验证。
