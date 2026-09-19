# 模块一验收状态

更新时间：2026-07-30。总体状态：**仿真验收通过，真机接口烟测通过，
真机默认采用规则回退**。

机器可读总报告：
`results/formal_training/acceptance_report.json`，其中 `passed=true`。

## 数据与训练

- 27/27条正式Gazebo bag矩阵完整；
- 12类必需话题、消息类型和95%原生频率合同全部通过；
- 16/5/6完整运行划分，每条轨迹完整归属一个集合；
- 训练、验证、测试均覆盖三场景、三光照和三种seed模式；
- 正式模型类型为formal，bootstrap模型承担接口测试；
- 七组基线/消融完整。

## 指标

| 验收项 | 结果 | 目标 |
|---|---:|---:|
| 质量/权重范围约束通过率 | 100% | 100% |
| 权重和/融合公式约束通过率 | 100% | 100% |
| 场景宏F1 | 0.8927 | ≥0.85 |
| Q_v相关性 | 0.9279 | ≥0.60 |
| Q_l相关性 | 0.8255 | ≥0.60 |
| 失效AUROC | 0.9809 | ≥0.85 |
| 权重MAE | 0.0841 | ≤0.10 |
| 相对规则改善 | 49.83% | ≥10% |
| 离线输出完整率 | 100% | ≥95% |
| 实时输出完整率 | 100% | ≥95% |
| 离线处理P95 | 1.337 ms | ≤100 ms |
| 实时端到端P95 | 0 ms，最大1 ms | ≤100 ms |
| 因果匹配合规率 | 100% | 100% |
| 独立质量地图 | 27/27 | 27/27 |
| 地图运行输出完整率 | 100% | ≥95% |
| 地图空间样本完整率 | 99.87% | ≥95% |

## 工程测试

- Release构建通过；
- CTest全部通过；
- Image/CompressedImage常见编码与异常；
- LaserScan和PointCloud2 NaN/空帧/PCA/结构；
- 正式PointCloud2 bag回归445/445输出，odom可选模式1329/1329输出；
- 场景、光照、几何上下文与上下文改变权重；
- MLP加载、推理与模型异常自动规则回退；
- EMA、迟滞、单/双故障和立即降级；
- 有界因果同步、超时与队列统计；
- 栅格索引、均值、样本方差、置信度和unknown；
- 27张独立地图、逐产物SHA-256和总体分组统计全部通过；
- 离线/实时1329个时间戳全部匹配，Q_fused差异P95为4.86×10⁻⁷。

## 输出与追溯

- 正式模型：`models/context_mlp_formal.yaml`；
- 真机接口烟测：`results/real_machine_smoke/report.json`；
- 数据卡/模型卡/实验报告：`docs/5_*`、`docs/6_*`、`docs/3_*`；
- 27张odom质量地图：
  `results/formal_training/quality_maps/<run_id>/`；
- 地图总体统计：`results/formal_training/quality_map_summary.json` 和
  `results/formal_training/quality_map_summary.csv`；
- 故障安全报告：`results/formal_training/fault_safety_report.json`；
- 接口冒烟报告：`results/formal_training/interface_smoke_report.json`；
- 实时一致性：
  `results/formal_training/offline_realtime_consistency.json`；
- 正式数据内容哈希：
  `results/formal_training/formal_dataset_summary.json`；
- 源码、配置、模型、数据和结果统一清单：
  `results/release_manifest.json`；
- 清单自身哈希：`results/release_manifest.sha256`。

## 交付范围

模块一交付定位质量、融合可靠性权重与定位质量地图，供下游位姿融合和路径
规划使用。正式bag采用程序化Gazebo和可控定位误差代理；下一阶段基于真实
定位和参考真值重采、训练并验收实机泛化。`real_machine.launch.py`
默认采用规则回退。
