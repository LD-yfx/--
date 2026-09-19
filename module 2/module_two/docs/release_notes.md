# 模块二整合修正版：调整与运行说明

交付日期：2026-09-19。目标环境：Ubuntu 22.04 + ROS2 Humble；已在本机 WSL2、Ubuntu 22.04.5、Gazebo 11.10.2、Nav2 1.1.20 中运行。

## 本次修复

- 将独立实验接到 Gazebo 实际相机和激光测量。模块一根据测量计算质量；AMCL 根据激光、地图和编码器里程计定位。Gazebo 真值只用于评价。
- 修正差速驱动里程计来源，明确使用编码器模式；独立实验关闭旧真值加噪声代理、路线驱动器和旧训练模型，运动由 Nav2 控制。
- 增加 C++ TF 发布者审计，记录真实 DDS 发布者标识及运行文件哈希，检查 map→odom 的唯一来源。
- 修复仿真时钟与质量消息到达顺序：仅对很小的时钟超前进行有上限的等待，保留原始观测时间；非法、远未来、超时消息仍拒绝。
- 几何基线实际关闭质量代价层，其他方法启用，避免基线受到质量超时回退惩罚污染。
- 修复位姿记录丢帧：增大可靠订阅队列、配置 rosbag QoS、结束时排空队列；保留覆盖率门槛和独立真值终点检查。
- 增加场景、评估、可视化与实验计划工具；启动脚本改为可配置的 Linux 工作区路径。

## 最新实际测试

运行标识：`independent_closed_loop_smoke_02`。场景 `dev_asymmetric`，正常传感器条件，`geometric` 方法，往返两段。

| 检查 | 结果 |
|---|---:|
| 两段 Nav2 任务 | 均成功，无超时 |
| 独立真值终点距离 | 去程 0.0988 m；回程 0.2171 m |
| 碰撞次数 | 0 |
| 真值 / 定位估计覆盖率 | 100% / 100% |
| 位置误差 RMSE | 0.1056 m |
| 位置误差 P95 | 0.1343 m |
| success / valid_evidence / pose_metrics_valid | 均为 true |
| 实际 quality_layer_enabled | false（几何基线） |

证据：[结果 JSON](../results/research/runner_smoke_02/result.json)、[运行清单](../results/research/runner_smoke_02/manifest.json)、[实际导航参数](../results/research/runner_smoke_02/navigation_parameters.json)、[轨迹](../results/research/runner_smoke_02/figures/trajectory.png)、[误差与质量时序](../results/research/runner_smoke_02/figures/error_quality_timeline.png)。

这些结果证明独立定位、导航和记录链路能够运行；单次几何基线测试不能证明质量感知方法降低定位误差。当前定位器为 AMCL，尚未验证多模态融合定位器的收益。研究协议中的 195 次正式实验尚未执行。新增动态事件及离线评价辅助代码已有单测，但动态 ROS 编排尚未集成，`benchmark_runner --phase dynamic` 会拒绝运行；早期演示包的动态障碍测试属于另一套机制验证。

ZIP 保留源码、测试、配置、场景和小型结果报告，不含编译产物、大型 rosbag 或逐帧原始数据。包内图表可直接查看；重新生成图表需用下述命令自行运行获得完整记录。历史报告里的本机绝对路径仅用于溯源。

## 解压后构建

先在已经安装 ROS2 Humble 的 Ubuntu / WSL 终端中操作。WSL 需要可用的 WSLg / 图形显示环境来渲染 Gazebo 相机；本机测试使用 WSLg。首次准备系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y python3-colcon-common-extensions python3-rosdep \
  python3-pytest python3-matplotlib rsync \
  ros-humble-navigation2 ros-humble-nav2-bringup \
  ros-humble-gazebo-ros-pkgs ros-humble-xacro ros-humble-cv-bridge \
  ros-humble-rosbag2 ros-humble-rosbag2-compression-zstd \
  ros-humble-tf2-geometry-msgs
source /opt/ros/humble/setup.bash
```

进入解压出来的 `module_two` 目录。以下命令自动复制源码到 `$HOME/sitp_quality_research/ros2_ws` 并构建，以避免 Windows 挂载盘造成的构建等待。

```bash
cd /你的解压目录/module_two
# 只在尚未初始化 rosdep 的系统执行一次：sudo rosdep init
rosdep update
rosdep install --from-paths ros2_ws/src --ignore-src -r -y --rosdistro humble
bash integration/build_research_native.sh
```

自定义位置时，在构建和运行两个终端中使用同一个设置：

```bash
export SITP_RESEARCH_WS="$HOME/自定义目录/ros2_ws"
```

本次已验证的本机安装位于 `/home/flx666/sitp_quality_research_01a0b893/ros2_ws`，本机想复用它时把上述变量设为该路径；其他电脑直接用默认路径重新构建。

## 运行一次独立闭环

仍在解压后的 `module_two` 目录中运行：

```bash
RUN_DIR="$HOME/sitp_results/geometric_$(date +%Y%m%d_%H%M%S)"
bash integration/run_research_native.sh benchmark_runner \
  --map-id dev_asymmetric --condition normal --method geometric \
  --phase smoke --seed 0 --output "$RUN_DIR"
```

脚本自动启动 Gazebo、AMCL、模块一、质量映射和 Nav2，并执行往返导航、保存结果后关闭它启动的进程。输出目录必须为空；每次使用新目录。默认同时保存 rosbag，需预留磁盘空间。同一时刻只运行一个实验；并行实验需显式分配不同 `--ros-domain` 和 `--gazebo-port`。

将 `--method geometric` 改为 `--method full` 并使用新的输出目录，即可运行完整质量模型。这是运行入口，收益仍需同条件、多次对照评价。结束后检查 `result.json` 的 `success`、`valid_evidence` 和 `pose_metrics_valid`；进程退出码只表示证据有效性，不能单独当作导航成功。

## 包内目录

- `ros2_ws/src/`：七个 ROS2 包，含模块一必要源码及统计扩展、模块二算法和 Nav2 插件、旧演示、独立实验和 TF 审计。
- `integration/`：构建、运行及集成辅助脚本。
- `tests/`、各包的 `test/`：回归测试。
- `docs/`：实现方案、历史验收报告、后续研究协议和本说明。
- `results/`：已生成的小型报告及图表；最新独立往返验收在 `research/runner_smoke_02/`。
- ZIP 根目录 `MANIFEST_SHA256.json`：各文件长度与 SHA-256；ZIP 旁另有整体校验值。

模块一原始归档数据和未修改的参考目录不在 ZIP 中；独立 Gazebo 入口使用包内场景和源码，不依赖该参考目录。旧离线复现实验若指定 `module_one_reference/`，需要用户原始归档数据。
