# 模块使用说明

完整命令见根目录 `README.md`。本文列出关键ROS2参数和接口契约。

## 输入参数

| 参数 | 值 |
|---|---|
| `input.image` | `raw` / `compressed` |
| `input.laser` | `scan` / `pointcloud2` |
| `input.odom_required` | 空间位姿要求开关 |
| `sync.tolerance_sec` | 因果匹配最大时间差 |
| `sync.sensor_timeout_sec` | 传感器/定位结果超时 |
| `sync.queue_depth` | 每路有界队列容量 |
| `qos.<input>.reliability` | `best_effort` / `reliable` |
| `qos.<input>.depth` | DDS历史深度 |
| `qos.<input>.durability` | `volatile` / `transient_local` |
| `frames.target` | `odom`或`map`等目标坐标系 |
| `fusion.strategy` | `mlp` / `rule` |
| `fusion.model_path` | 外部OpenCV YAML模型 |
| `context.scene_classifier_enabled` | 场景分类开关 |

所有话题位于 `topics.*`，可完全重映射。参数默认值见
`config/quality_params.yaml`。`<input>` 可分别为 `image`、`laser`、
`odom`、`localization`，允许适配真机异构DDS配置。

## 实时输出

`/localization_quality/metrics` 为兼容数组。前五项固定为：

```text
q_laser, q_visual, q_fused, w_laser, w_visual
```

后续依次追加场景概率、光照、几何、有效标志、同步误差和延迟。稳定程序
应优先读取 `/localization_quality/diagnostics` 的具名KeyValue。

`/localization_quality/quality_map` 是 `nav_msgs/msg/OccupancyGrid`：

- `-1`：待采样；
- `0..100`：低质量到高质量；
- header frame为 `frames.target`。

odom位姿通过TF转换到目标坐标系。质量和诊断持续发布，成功转换的位姿
进入栅格聚合。

## 离线输出

离线CSV保留v0.1列，同时追加55项以上的上下文、原始特征、策略、同步方向
和坐标系。旧bag模式基于传感器原始特征，正式bag模式进一步把定位状态和
协方差纳入质量。

离线和实时均使用 `causal_past`，匹配范围覆盖当前时刻及历史帧。

## 模型路径

默认参数使用随包安装的正式模型：

```bash
fusion.model_path:=package://localization_quality/models/context_mlp_formal.yaml
```

`package://` 路径通过ament索引解析，可从任意工作目录启动。也可在
命令行指定实机重训模型的绝对路径：

```bash
--ros-args -p fusion.model_path:=/absolute/path/context_mlp_real.yaml
```

模型加载异常时节点自动切换并明确输出 `rule_fallback`。bootstrap模型
用于接口测试。

## 真机启动

```bash
ros2 launch localization_quality real_machine.launch.py \
  scan_topic:=/scan \
  image_topic:=/camera/image_raw \
  odom_topic:=/odom
```

该入口使用真机时钟并默认启动规则回退。基于真实定位输出与参考真值完成
重训和验收后，可设置
`mlp_enabled:=true model_path:=/path/to/real_model.yaml`。
