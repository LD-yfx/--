# Gazebo仿真与正式采集

## 平台与场景

目标平台为Ubuntu 22.04、ROS2 Humble和Gazebo Classic 11：

```bash
sudo apt-get install -y ros-humble-gazebo-ros-pkgs ros-humble-xacro
```

- corridor：狭长平行墙面，突出方向退化；
- hall：开阔大厅，用立柱数量改变结构复杂度；
- outdoor：开放地面、建筑立面和街景结构；
- dark/normal/strong：改变环境光和定向光。

世界、路线和运行矩阵由固定seed生成，位于 `simulation/generated/`。

## 机器人与传感器

- RGB/CameraInfo：320×240，10 Hz；
- LaserScan：720线，15 Hz；
- 深度PointCloud2：80×60，5 Hz；
- 里程计、TF和Gazebo真值：50 Hz。

`simulation_sensor_pipeline.py` 在评分器之前注入图像模糊/亮度退化或激光
遮挡；`simulation_route_driver.py` 执行确定性路线；
`simulation_localization_proxy.py` 发布可控偏置、噪声和协方差的两路定位。

## 正式采集命令

```bash
cd <archive>/localization_quality_final
source /opt/ros/humble/setup.bash
source install/runtime/share/localization_quality/local_setup.bash

python3 scripts/collect_sim_dataset.py \
  simulation/generated/matrix.json \
  data/formal_gazebo \
  --duration 90 --startup-timeout 90
```

采集器会等待 `/clock`、使用独立ROS_DOMAIN_ID、正常关闭bag、验证12类
话题类型及95%原生频率下限，并写入每个bag文件SHA-256。已完成运行可跳过；
开发中的运行目录统一移动到 `_incomplete`，通过合同验证的运行标记为完成。

正式矩阵验证：

```bash
python3 scripts/prepare_dataset.py validate-matrix data/formal_gazebo
python3 scripts/summarize_formal_dataset.py \
  data/formal_gazebo \
  results/formal_training/formal_dataset_summary.json
```

当前结果为27/27通过，RGB每条868–898帧、LaserScan 1295–1340帧、
PointCloud2 434–449帧，所有运行达到原生频率合同。

## 可复现重采

可用多个 `--run-id` 精确重采指定运行。采集器会合并现有journal并保留
其他完成项。例如：

```bash
python3 scripts/collect_sim_dataset.py \
  simulation/generated/matrix.json data/formal_gazebo \
  --duration 90 --startup-timeout 90 \
  --run-id corridor_dark_seed1
```

正式目录内容哈希覆盖27条已接收运行，其聚合SHA-256和逐文件清单位于
`results/formal_training/formal_dataset_summary.json`；早期开发运行存入
冷归档。
