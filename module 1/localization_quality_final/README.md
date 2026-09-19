# 多模态定位质量评估与上下文融合（模块一）

本包实现模块一的工程链路：

```text
RGB → Q_v
LaserScan / PointCloud2 → Q_l
场景概率 + 光照 + 几何复杂度 → 环境上下文
Q_v + Q_l + 环境上下文 → NumPy训练的MLP → Q_fused
Q_fused + map/odom位姿 → Q(x,y)
```

正式输出使用
`mlp`、`mlp_safety_override`、`rule_fallback`、`both_degraded`
逐帧记录实际执行策略。

## 当前完成状态

源码、三场景仿真、27条正式bag、真值标签、纯NumPy训练、C++推理、
七组消融、实时/离线节点、27张六格式质量栅格、自动验收和SHA-256追溯均已
完成。正式矩阵校验为27/27；最终测试集所有指标通过
`results/formal_training/acceptance_report.json`。

仓库内的 `models/bootstrap_context_mlp.yaml` 由程序化合成行训练，
用于验证NumPy导出与C++推理接口。27条正式Gazebo数据训练结果为
`models/context_mlp_formal.yaml`；默认配置用可移植的
`package://localization_quality/models/context_mlp_formal.yaml` 加载。
`my_dataset` 为真机bag，用于接口与时延烟测；正式训练采用27条Gazebo数据。真机入口默认采用规则回退策略。即使显式启用 Gazebo 模型，融合层也会检测“健康原始视觉证据被仿真校准器压低”的分布外矛盾，并逐帧退回原始传感器评分和规则融合；诊断与 CSV 的 `calibration_domain_guard` 会记录该保护是否生效。明确的视觉跟踪丢失或低质量图像不会触发此回退。

正式测试集结果：

| 指标 | 结果 | 目标 |
|---|---:|---:|
| 场景宏平均F1 | 0.8927 | ≥0.85 |
| Q_v与负误差相关性 | 0.9279 | ≥0.60 |
| Q_l与负误差相关性 | 0.8255 | ≥0.60 |
| 失效识别AUROC | 0.9809 | ≥0.85 |
| MLP权重MAE | 0.0841 | ≤0.10 |
| 相对规则融合改善 | 49.83% | ≥10% |
| 实时输出完整率 | 100% | ≥95% |
| 实时端到端延迟P95 | 0 ms（最大1 ms） | ≤100 ms |

## 主要能力

- 视觉：模糊、亮度、对比度、欠曝/过曝、边缘密度、特征数量、
  帧间ORB匹配率、运动模糊、定位跟踪状态和协方差；
- 二维激光：有效回波、扇区覆盖、遮挡、距离统计、方向熵、各向异性、
  走廊退化、距离突变和结构复杂度；
- 三维点云：有效点、空间覆盖、近距遮挡、密度、PCA特征值、
  线/面/体结构比例和条件数；
- 上下文：走廊/大厅/室外概率、暗/正常/强光、几何复杂度、开放度、
  方向各向异性、退化度和2D/3D覆盖；
- MLP：视觉/激光监督校准、`11→16→8→3` 场景分类器和
  `10→32→16→2` Softmax融合网络；
- 安全：EMA、进入/恢复迟滞、失效输入立即降级、单双传感器保护和
  模型异常规则回退；
- ROS2：原始/压缩图像、LaserScan/PointCloud2、可配QoS、odom可选、
  TF目标坐标系、有界因果同步、diagnostics和实时OccupancyGrid；
- 地图：均值、样本方差、数量、置信度、更新时间和unknown。

## 构建与测试

```bash
source /opt/ros/humble/setup.bash
cmake -S . -B build/release \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/release --parallel 2
ctest --test-dir build/release --output-on-failure
```

推荐的colcon方式：

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select localization_quality \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
colcon test --packages-select localization_quality
colcon test-result --verbose
```

## 仿真与27条bag

安装依赖：

```bash
sudo apt-get install -y \
  ros-humble-gazebo-ros-pkgs ros-humble-xacro
```

生成矩阵（仓库已经包含一次确定性生成结果）：

```bash
python3 scripts/generate_sim_worlds.py \
  --output simulation/generated
```

安装/构建本包后，录制3场景×3光照×3独立种子：

```bash
python3 scripts/collect_sim_dataset.py \
  simulation/generated/matrix.json \
  data/formal_gazebo \
  --duration 90
python3 scripts/prepare_dataset.py validate-matrix data/formal_gazebo
```

种子0/1/2分别覆盖低/中/高几何复杂度，以及正常/视觉退化/激光退化；
路线和速度也随种子变化。每条bag记录RGB、CameraInfo、LaserScan、
PointCloud2、Odometry、视觉/激光定位、Gazebo真值、TF、Clock和元数据。

仿真中的两路定位结果来自明确标注的
`controlled_ground_truth_noise_emulator`，用于第一阶段可控监督开发，
其产物类型为可控真值噪声定位代理。

## 正式训练

```bash
python3 scripts/build_training_dataset.py \
  data/formal_gazebo results/formal_training \
  --params-file config/quality_params.yaml

python3 scripts/train_context_mlp.py \
  results/formal_training/training_rows.csv \
  --split results/formal_training/dataset_split.json \
  --dataset-kind formal \
  --output models/context_mlp_formal.yaml \
  --report results/formal_training/training_report.json

python3 scripts/evaluate_ablations.py \
  results/formal_training/training_rows.csv \
  --split results/formal_training/dataset_split.json \
  --model-dir /tmp/localization_quality_ablations \
  --output results/formal_training/ablation_report.json
```

划分以完整 `run_id` 为单位，目标为16/5/6条运行（约60/20/20），同一
轨迹完整归属一个集合。

默认配置已经指向随包安装的正式模型。模型加载或推理异常时，节点自动
切换并明确输出 `rule_fallback`。

完整归档复跑可使用：

```bash
./scripts/reproduce_archive.sh /tmp/localization_quality_reproduced
```

## 实时节点

```bash
ros2 run localization_quality realtime_evaluator \
  --ros-args --params-file "$(pwd)/config/quality_params.yaml"
```

真机使用独立入口并以 `rule_fallback` 启动；取得真实定位真值并完成重训
验收后，可显式传入真机模型：

```bash
ros2 launch localization_quality real_machine.launch.py \
  scan_topic:=/scan image_topic:=/camera/image_raw odom_topic:=/odom
```

`/localization_quality/metrics` 的前五项保持v0.1兼容：

```text
[q_laser, q_visual, q_fused, w_laser, w_visual, ...新增字段]
```

具名特征、场景概率、策略、同步误差和延迟通过
`/localization_quality/diagnostics` 发布。质量地图发布到
`/localization_quality/quality_map`，语义为 `-1=unknown`、
`0..100=低质量..高质量`，产物类型为定位质量地图。

## 离线与栅格

```bash
ros2 run localization_quality offline_evaluator \
  BAG_PATH normal quality_features.csv \
  --ros-args --params-file "$(pwd)/config/quality_params.yaml"

python3 scripts/generate_quality_grid.py \
  quality_features.csv results/quality_grid
```

导出 `quality_grid.csv/yaml/pgm/png/svg` 和
`quality_grid_metadata.json`。输入位姿是 `odom` 时，元数据明确写成
`odometry_coordinate_quality_grid`。

正式27运行地图集合使用：

```bash
python3 scripts/generate_formal_quality_maps.py \
  data/formal_gazebo results/formal_training/quality_maps \
  --split results/formal_training/dataset_split.json \
  --params-file config/quality_params.yaml \
  --model-path models/context_mlp_formal.yaml \
  --summary-json results/formal_training/quality_map_summary.json \
  --summary-csv results/formal_training/quality_map_summary.csv \
  --jobs 2

python3 scripts/validate_quality_map_collection.py \
  results/formal_training/quality_maps \
  results/formal_training/quality_map_summary.json \
  results/formal_training/quality_map_summary.csv
```

当前集合包含27张独立地图、35,827个空间质量样本和3,537个累计已知
栅格。总体融合质量均值为0.7022，P50为0.7000，P95为0.9002；JSON报告
按数据划分、场景、光照、故障和几何复杂度提供分组统计。

## 验收与追溯

```bash
python3 scripts/validate_module_one.py \
  --training-report results/formal_training/training_report.json \
  --runtime-csv results/formal_training/runtime_normal.csv \
  --ablation-report results/formal_training/ablation_report.json \
  --expected-sensor-frames 1329 \
  --output results/formal_training/acceptance_report.json

python3 scripts/compare_offline_realtime.py \
  results/formal_training/runtime_normal.csv \
  results/formal_training/realtime_playback.csv \
  --expected-sensor-frames 1329 \
  --output results/formal_training/offline_realtime_consistency.json

python3 scripts/create_hash_manifest.py . results/release_manifest.json \
  --include-real-bag
```

详细设计、数据边界和模型限制见：

- `docs/1_algorithm_principle.md`
- `docs/5_data_card.md`
- `docs/6_model_card.md`
- `docs/7_simulation_and_collection.md`
- `docs/8_acceptance_status.md`
- `docs/9_archive_layout.md`

许可证：Apache-2.0。
