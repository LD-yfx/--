# 模块一源码副本与模块二统计接口

此目录从用户提供的 `sitp.multimodal-autonomous-mobile-robot-main.zip` 解压源码
`module_one_reference/localization_quality_final` 复制而来，原始归档保持不变。
原始说明保存在 [UPSTREAM_README.md](UPSTREAM_README.md)。本副本保留许可证、
核心融合算法、模型参数、脚本与测试，没有复制原始 rosbag、data 或 results。
原始说明里的历史实验数值来自模块一归档，不是模块二重测结果；相关外部数据、
结果路径需另行提供。本次新增统计接口没有重新训练或修改融合模型。

## 新增输出与参数

实时节点在已有 `/localization_quality/quality_map` 之外，可选发布
`quality_navigation_msgs/msg/QualityGrid`：

| 参数 | 默认值 | 用途 |
|---|---|---|
| `topics.quality_stats` | `/localization_quality/quality_stats` | 统计快照话题 |
| `map.publish_statistics` | `true` | 是否创建并发布统计话题 |
| `map.statistics_publish_period_sec` | `0.5` | 两次快照的最小传感器时间间隔，必须大于零 |
| `map.statistics_window_sec` | `10.0` | 大于零时采用近期时间窗口；零时采用累计统计 |
| `map.statistics_max_samples_per_cell` | `256` | 窗口模式每格最多保留的最新样本数，必须大于零 |

QoS 为 reliable / transient-local / keep-last-1。节点仍在每个有匹配位姿的传感器
评估回调中发布原 OccupancyGrid（100 表示质量最好，-1 表示未知），该图仍保持原
累计统计行为。新增统计维护独立状态，从首个有匹配位姿的回调开始，按传感器时间
限频发布；没有新的匹配位姿回调时不会定时重发。消费者应独立处理输入中断和时效。

## 消息契约

统计消息携带完整地图几何信息，以及相同长度的 `indices`、`mean`、`variance`、
`sample_count`、`last_observed` 数组。`indices = y * width + x` 按行升序排列。
每条消息是**完整稀疏快照**，只包含样本数大于零的格子；消费者整体替换旧快照，
未列出的格子均为未知。空快照清除已有统计，不能解释为“没有更新”。

- `header.stamp` 来自激光/点云质量评估的传感器时间，沿用上游零时间戳回退到
  节点 ROS 时钟的行为。仿真、回放和导航各节点必须一致使用 `use_sim_time`。
- `last_observed[i]` 来自该格实际最后一次采样的时间，不能用消息头替代。
- 方差为无偏样本方差；单样本方差为零，不代表高置信度。采样数不代表独立样本数。
- 新增统计接口遇到时间倒退时清空自身状态，开始新的统计周期；原质量图和融合
  状态不受影响。输入流应保持时间顺序；延迟到达的旧帧也会触发这一保守清空策略。
- 原 OccupancyGrid 与窗口统计均值可能不同；两者来自同一融合分数流，但时间范围
  不同。代价节点应选择一种输入模式，不能把两个消息的均值、方差交叉拼接。

`statistics_mode="windowed"` 表示仅使用
`[snapshot_time - statistics_window_sec, snapshot_time]` 内保留的样本。发布前淘汰
过期样本，完全过期的格子从快照中消失。每格达到容量后先删除最旧样本，因此高频
采样时有效时间范围可能短于配置窗口。它支持旧的低质量样本过期后恢复，但不会预测
从未观测过的空间质量；窗口长度和支持度参数需要通过实验选择。

窗口会遗忘好的证据，也会遗忘坏的证据。完全过期的格子转为未知，因此原本风险
高于未知风险的坏格子，也可能在过期后降低到未知代价。这是有限记忆的接口语义，
并非时间单调性的保证。代价模型“风险随 age 非减”的性质只在均值、方差和样本数
保持固定时成立，不能扩展为窗口更新后的整个系统风险永远随时间增加。未知区域
策略和物理障碍约束仍须独立处理。

`statistics_mode="cumulative"` 表示自统计状态重置以来的累计均值/方差，与模块一
原累计算法一致；每格容量仅约束窗口模式。累计模式里一条新观测会刷新最后观测
时间，而均值仍包含旧样本，因此不应据此声称所有历史数据仍反映当前环境。离线
CSV 本来就是累计统计，不能仅通过改字段名称把它当成窗口数据。

## 构建与运行

在 ROS2 Humble 工作区中同时构建接口包和本包：

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-up-to localization_quality
source install/setup.bash
colcon test --packages-select localization_quality
colcon test-result --verbose
ros2 run localization_quality realtime_evaluator --ros-args \
  --params-file src/localization_quality/config/quality_params.yaml
ros2 topic echo /localization_quality/quality_stats --once \
  --qos-reliability reliable --qos-durability transient_local
```

新增 `test_quality_statistics` 检查稀疏快照、每格时间、样本方差、空快照、非法时间、
限频、窗口过期与恢复、容量限制、累计模式兼容和时钟回退。新增头文件
`quality_statistics.hpp` 只负责新增统计状态、消息转换和限频判断；核心质量计算
源文件、融合模型与原始归档相同。

原有输入话题、传感器 QoS、模型配置和工具用法见
[UPSTREAM_README.md](UPSTREAM_README.md) 及 [docs](docs)。
