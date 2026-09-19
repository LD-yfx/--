# 模块二定位质量感知导航

本工作区把模块一的定位质量转换成导航软代价，并接入 ROS2 Humble / Nav2。实现方案和研究边界见 [实施方案](docs/implementation_plan.md)。质量代价永远不能清除障碍物或把未知占据空间变成已知可通区域。

**2026-09-19 整合修正版：解压后先读 [本次调整与运行说明](docs/release_notes.md)。** 已加入 Gazebo 实际图像/激光、编码器里程计、独立 AMCL 定位与数据完整性审计，完成 WSL 往返复测。最新结果见 [独立定位测试报告](results/research/runner_smoke_02/result.json)。

[早期 WSL 验收报告](docs/acceptance_report.md) 保留原型机制测试记录；本次通过的是独立定位闭环测试，尚未完成质量收益的大规模对照实验。新增动态实验辅助算法已单测，动态 ROS 编排尚未集成，runner 会明确拒绝该阶段。

## 实现的研究方法

基线是线性质量惩罚；完整模型联合质量均值、波动、观测数量和数据年龄进行单侧保守映射。新增窗口统计接口使地图可以遗忘旧样本。评分是可解释的经验风险指标，既不是定位失败概率，也不是统计置信区间。

```
模块一 Image + LaserScan/PointCloud2 + 位姿
       ↓ 质量评估与融合（沿用）
quality_navigation_msgs/QualityGrid（新增窗口统计）
       ↓ quality_cost_node（同一 Python 数学模型供离线与实时使用）
nav2_msgs/Costmap，0..252 软代价
       ↓ quality_nav2_layer::QualityCostLayer
Nav2 全局代价地图 → 全局路径 → 局部控制与避障
```

原解压源码在工作区外层 `module_one_reference/`；这里的 `ros2_ws/src/localization_quality` 是可构建副本，新增统计接口与窗口累加器，保留原融合与质量图语义。

## 环境和构建

目标环境是 Ubuntu 22.04 + ROS2 Humble，本项目已在用户 WSL 中进行验证。安装依赖：

```bash
sudo apt-get update
sudo apt-get install -y ros-humble-navigation2 ros-humble-nav2-bringup \
  ros-humble-gazebo-ros-pkgs ros-humble-xacro ros-humble-cv-bridge \
  ros-humble-rosbag2 ros-humble-rosbag2-compression-zstd \
  ros-humble-tf2-geometry-msgs python3-pytest python3-matplotlib \
  python3-colcon-common-extensions python3-rosdep rsync
source /opt/ros/humble/setup.bash
cd module_two/ros2_ws
colcon build --executor sequential --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
source install/setup.bash
colcon test --event-handlers console_direct+
colcon test-result --verbose
```

ZIP 不包含 build/install/log；解压后需要在自己的 ROS2 环境构建。推荐使用 [本次运行说明](docs/release_notes.md) 中的脚本，自动把源码复制到 Linux 原生目录后构建。

本次构建观察到 `/mnt/c` 的 DrvFS 元数据访问造成明显等待。频繁开发建议把 `ros2_ws/src` 复制到 WSL 原生 Linux 目录中的工作区，并在那里重新 `colcon build`，使源码、build、install、log 都位于 ext4。不要将已有 build/install 目录迁移后直接复用；生成文件可能包含原路径。

## 启动质量映射

先启动模块一对应的传感器、位姿与 TF，再启动本工作区的 `localization_quality` 实时评估器。`map.publish_statistics=true` 是默认值。地图几何需要覆盖实际路线；模块一的默认 `[-10,10)` 范围不覆盖部分归档走廊路线。

```bash
ros2 run localization_quality realtime_evaluator --ros-args \
  --params-file src/localization_quality/config/quality_params.yaml \
  -p frames.target:=map -p map.origin_x:=-20.0 -p map.origin_y:=-20.0 \
  -p map.width:=800 -p map.height:=800
ros2 launch quality_aware_navigation quality_mapping.launch.py use_sim_time:=false
```

`frames.target:=map` 要求位姿源本就在 map 或提供正确 TF，不能只改 frame 字符串。Gazebo 等仿真应为所有相关节点设置 `use_sim_time:=true`。真机入口默认禁用仿真训练模型，沿用模块一的域偏移保护。

Nav2 global costmap 加入最后一层：

```yaml
plugins: [static_layer, obstacle_layer, inflation_layer, quality_layer]
quality_layer:
  plugin: quality_nav2_layer::QualityCostLayer
  topic: /localization_quality/navigation_cost
  unknown_cost: 140
  stale_cost: 200
  stale_timeout: 5.0
  combination_method: saturating_add
```

需要同时提供正常的障碍物地图、激光、机器人 footprint、定位与 Nav2 配置。关闭模块一 `simulation_route_driver.py`，使导航控制器成为唯一运动命令来源。详见 [插件说明](ros2_ws/src/quality_nav2_layer/README.md)。

## 离线质量图转换

在已 source 工作区的终端运行：

```bash
ros2 run quality_aware_navigation quality_to_cost \
  ../../module_one_reference/localization_quality_final/results/formal_training/quality_grid/quality_grid.csv \
  ../results/example_cost
```

上述命令以 `module_two/ros2_ws` 为当前目录，需要另行提供原模块一归档数据目录（不在本 ZIP 中），也可将输入改成自己的质量 CSV。默认以输入最后观测时间作为评价时刻；`--as-of` 必须使用数据的同一时钟域。输出 `costs.csv` 与描述坐标、参数和来源的 `metadata.json`。输出仅有质量软代价，不包含障碍物。

原模块一 PGM 中质量灰度与未知值存在歧义，且 YAML 并非标准占据地图合同，离线应读取有明确 `known` 字段的 CSV 和 JSON 元数据。

## 输入模式与故障处理

- `statistics`：完整统计模式，使用每格真实时间；默认窗口统计。旧消息、未来时刻、数组不一致和非法质量拒绝。
- `mean_only`：兼容原 OccupancyGrid。只使用均值与保守下限，不编造方差、次数或每格时间，诊断明确标记降级。
- `csv`：冻结的离线源时钟模式，只用于已登记地图实验；不能把历史 bag 秒当作当前墙钟。
- 源消息超时或损坏：输出保守代价。上游代价节点失联时，Nav2 插件还有独立超时保护。
- 未知质量和未知占据分别处理；前者用软惩罚，后者沿用物理地图与规划器的通行策略。
- 插件每周期重算全图以保证旧代价消除；大图需要测量耗时。

窗口完全过期会把好、坏观测都变成未知；旧坏格的惩罚可能因此降低。映射函数关于年龄的单调性质只针对固定统计量，不能理解为整个窗口系统的时间单调保证。

## 运行验证与导航演示

在已完成构建的 WSL 终端中，从项目根目录运行完整验证：

```bash
cd /mnt/c/Users/flx66/Documents/ChatGPT/sitp
bash module_two/verification/run_all.sh
```

脚本依次执行 Python 回归测试、C++/ROS 包测试、离线消融、实际模块一节点到代价节点的 ROS 管线和 Nav2 闭环。三个 ROS 测试使用独立 domain，并将进程限制在本机；测试结束会停止其启动的导航节点。报告保存在 `module_two/results/`。

单独运行导航闭环：

```bash
source /opt/ros/humble/setup.bash
source module_two/ros2_ws/install/setup.bash
ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=74 ros2 run quality_navigation_demo integration_runner \
  --output "$PWD/module_two/results/ros_navigation" --require-installed
```

演示运行真实 Nav2 NavFn、DWB 和 BT Navigator，传感器由二维运动学模拟器提供，不需要图形桌面。测试比较路线选择，并检查动态封路、移除障碍和完全无路后的停止。启动参数与手动操作见 [导航演示说明](ros2_ws/src/quality_navigation_demo/README.md)，离线实验见 [结果报告](results/offline/report.md)。

## 验证范围

离线算法测试验证数学性质、碰撞几何与重规划；实际模块一 CSV 验证格式兼容。ROS 管线验证新增统计到代价消息；Nav2 闭环演示验证实际规划器、控制器与动态障碍响应。人工质量场、理想里程计和二维运动学仿真不能证明真实 SLAM 精度提高。

原始模块一 rosbag 未包含在用户提供的 ZIP 中。真实定位收益需要另行取得独立定位输出与参考真值并完成实验；本实现不会把模块一代理误差或本算法目标函数的下降描述为真实定位精度改善。验证结果以 `results/` 中实际生成的报告为准。
