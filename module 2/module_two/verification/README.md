# ROS2 管线验收

`check_ros_pipeline.py` 启动真实 ROS2 可执行程序，通过 DDS 发送和接收消息，使用独立
标量公式检查输出。合成输入用于接口、时效和异常处理验收，不是定位精度实验。

先构建工作区的 `quality_navigation_msgs`、`localization_quality` 和
`quality_aware_navigation`，再运行：

```bash
bash module_two/verification/run_ros_pipeline.sh
```

包装脚本先清空子进程继承的 `PYTHONPATH`，再加载 ROS2 Humble 和同目录工作区的
`install/setup.bash`，因此实际执行已安装的生产代码，不使用工作区 Python 源码
覆盖。使用脚本文件也避免 PowerShell→WSL 的嵌套变量展开问题。源码改动后应先
重新构建，再进行正式验收；若需调试源码覆盖，可自行 source 环境、设置路径并
直接调用 Python 脚本，但来源报告会显示实际加载位置。

默认域为 `ROS_DOMAIN_ID=73`，仅本机通信；脚本不创建速度指令发布者。三个子进程
（模块一评估器、链路代价节点、独立输入代价节点）按阶段运行，日志写入报告目录，
无论成功或失败都在 `finally` 中结束各自进程组。

完整验收包括：

1. 合成图像、激光和里程计经过实际模块一节点，产生窗口统计；实际模块二节点
   接收统计并输出逐格匹配公式的 `nav2_msgs/Costmap`，同时检查旧质量图兼容性。
2. 使用确定输入精确检查好/差/陈旧/波动/未知格代价。
3. 数组长度不一致、重复索引和未来观测时间被拒绝并触发保守代价。
4. 重复源时间戳不会作为新证据接收；源超时触发保守代价，新输入到来后恢复。
5. 地图尺寸、分辨率、原点、旋转变化及空快照完整替换语义。

默认结果为 `module_two/results/ros_pipeline/report.json`，包含每项结果、实际成本、
独立公式结果、诊断状态、子进程命令、退出状态及日志末尾。默认业务等待 15 秒，
首次导入与 DDS 发现另有 90 秒冷启动上限。

报告还记录 ROS 发行版、Python 环境、实际加载的生产代码路径与 SHA-256，并比对
`ros_node.py`、`model.py`、`grid.py` 的工作区和安装副本。交付前重新构建工作区，
再使用 `--require-installed-match` 运行完整验收；副本不一致会令最终报告失败。

```bash
# 模块一尚未构建时，只运行模块二边界测试；报告明确记录省略完整管线。
bash module_two/verification/run_ros_pipeline.sh --skip-module-one \
  --output module_two/results/ros_pipeline/direct_report.json
```

此验收不证明真实定位误差下降，也不证明 Nav2 插件被加载、规划控制效果或机器人
避障安全。上述项目需要各自的插件测试、导航闭环实验和独立定位真值。
