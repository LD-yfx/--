# quality_nav2_layer (ROS 2 Humble)

`quality_nav2_layer::QualityCostLayer` 将 `nav2_msgs/msg/Costmap` 中的定位质量软代价加入 Nav2 master costmap。输入为**已经映射过的导航代价**，不是模块一的 0–100 质量分数。插件不发布速度指令。

## 接入

在 global costmap 的 `plugins` 最后添加 `quality_layer`，放在 static/obstacle/inflation 层之后。local costmap 通常保留实时障碍与膨胀层；仅在希望局部控制也考虑质量时，才同样加入该插件。

```yaml
plugins: ["static_layer", "obstacle_layer", "inflation_layer", "quality_layer"]
quality_layer:
  plugin: "quality_nav2_layer::QualityCostLayer"
  enabled: true
  topic: /localization_quality/navigation_cost
  stale_timeout: 5.0
  stale_cost: 200
  unknown_cost: 160
  combination_method: saturating_add
```

参数在初始化时读取；修改后需重新配置 costmap lifecycle。`stale_timeout` 为有限正数，两个 fallback cost 必须为整数 0–252；`combination_method` 仅支持 `saturating_add` 和 `max`。不合法参数导致初始化失败。发布者 QoS 必须与 `reliable + transient_local + depth 1` 匹配。

默认合成：`min(252, master + quality)`。master 的 253（内切膨胀）、254（致命障碍）、255（未知）保持原值。质量层不能把未知空间变成可通行空间；Nav2 的 `allow_unknown` 仍由导航配置决定。可选 `max` 合成便于与标准代价层策略对照。软代价上限 252 不代表障碍，且默认不会清除任何障碍。

## 地图、时间和故障语义

- 输入允许 0–252 和 255（未知）；拒绝 253/254、长度不匹配、空 frame、非有限几何、非正分辨率及非平面/非单位四元数。origin 的 yaw 支持任意角度；数据为 row-major。
- 对每个 master cell 的中心，使用最新 `source_frame <- global_frame` TF，再应用输入 origin 的逆旋转并采样。master 可滚动，输入尺寸、原点、分辨率可变化。只有 2D 平面 TF 受支持；本实现不对质量插值，也不将地图框架的运动误认为每格观测更新。
- 无观测格、地图范围外取 `unknown_cost`。尚未收到地图时取 `max(unknown_cost, stale_cost)`。
- 超时同时检查 ROS header 时间和上次**有效且更晚时间戳消息**的 steady-clock 接收时间。旧 latched 数据、重复/乱序消息不会刷新接收时间。超过当前 ROS 时间 0.5 秒的未来消息被拒绝。仿真暂停超过超时也会保守降级；ROS 时间倒退后应重启质量发布链及 costmap，重新建立时间序列。
- 数据陈旧或新消息无效时使用 `max(last_quality_cost, stale_cost)`。TF 失败/过旧时复用上一次成功投影的地图和变换并提高到该下限；未成功投影过则全图 fallback。收到更新的有效消息并恢复 TF 后可恢复正常。日志以 5 秒节流报告降级。静态零时间戳 TF 不做超时判定。
- header 时间只用于消息新鲜度，逐格观测时效由上游代价节点处理；发布最新 header 不能替代每格观测时间。
- 降级仍输出完整的保守软代价并将 `current_` 设为 true，以允许 Nav2 继续规划。`current_` 不是质量源健康指示器。若系统必须在失去质量源时停止，应在更上层制定停止策略。

## 旧代价清除及性能

每次 `updateBounds` 请求整个 master 的边界。Nav2 重置此范围并按顺序重放所有代价层，确保质量降低、输入地图缩小、输入 origin 变化和 master 滚动时旧质量代价消失，也避免饱和相加跨周期累积。不能直接反复调用 `updateCosts` 而不执行 Nav2 的 reset/replay 流程。`reset()` 不删除环境质量证据，障碍清理恢复不应抹去质量信息。

这是优先保证正确性的首版策略，代价为每周期 O(master cell count) 采样，并迫使其他层重算全范围；大地图上应测量实际频率。将来可对几何变化前后范围、变化格及超时状态做精确脏区跟踪。该优化不是当前实现的一部分。

## 构建和测试

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select quality_nav2_layer --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select quality_nav2_layer --event-handlers console_direct+
colcon test-result --verbose
```

依赖由 `package.xml` 声明。gtest 覆盖全部软代价值的单调性/饱和性、三个保留主地图编码、未知和边界、旋转原点和替换地图几何、非法消息及双时间源超时。插件加载、ROS QoS/TF 链路、Nav2 重建过程和闭环导航需由工作区集成测试验证；单元测试不能替代这些验收。

接口核对来源：[Nav2 Humble Layer](https://github.com/ros-navigation/navigation2/blob/humble/nav2_costmap_2d/include/nav2_costmap_2d/layer.hpp)、[Costmap](https://github.com/ros-navigation/navigation2/blob/humble/nav2_msgs/msg/Costmap.msg)。
