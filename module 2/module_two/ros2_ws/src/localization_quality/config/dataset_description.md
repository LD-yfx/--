# 真机bag说明与校验清单

> 项目负责人已确认 `my_dataset` 为真机采集bag。正式27运行仿真数据规范见
> `docs/5_data_card.md`；该真机bag用于接口/时延烟测，当前上下文MLP训练
> 采用27运行仿真数据。

以下信息来自项目内 `my_dataset/metadata.yaml`、`ros2 bag info`、实际
文件大小和SHA-256，构成当前数据版本的权威记录。

## 1. 文件清单

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `metadata.yaml` | 1,937 | `36c20abe36e31fd48fa3ba4de5388346984203ce61ad4b2470ed4adaeed040fd` |
| `my_dataset_0.db3` | 3,403,112,448 | `186ec266cb63874c2f827cc6ade86d126f3e962afb8632cb3e07cd03e10d29c4` |

存储格式为ROS2 bag sqlite3原始存储；db3约3.2 GiB。归档容量以解压后
bag的实际大小计量。

## 2. 时间与消息

- 时长：91.843587842 秒；
- 起始记录时间：2026-06-09 19:38:05.861861565（bag 工具显示时间）；
- 总消息数：2,858。

| 话题 | 类型 | 数量 | 按总时长估算频率 |
|---|---|---:|---:|
| `/scan` | `sensor_msgs/msg/LaserScan` | 336 | 3.66 Hz |
| `/camera/image_raw` | `sensor_msgs/msg/Image` | 546 | 5.94 Hz |
| `/odom` | `nav_msgs/msg/Odometry` | 1,976 | 21.51 Hz |

频率按总数量除以bag时长计算，表示整段采集的平均消息速率。

## 3. 可用数据与验证范围

该bag提供 `/scan`、`/camera/image_raw` 和 `/odom`。后续硬件归档阶段将
补录采集机器人、传感器型号、标定、场景、天气、路线、控制速度、定位
输出、TF、地图和参考真值。

当前验证覆盖消息兼容性、输出完整率和本机处理时延。质量分数—真实定位
误差评估与监督重训将在含参考真值的真机数据集上完成。

正式Gazebo模型在该bag上发生显著域偏移：原始视觉质量均值约0.716，经
仿真校准器后约0.0027。因此真机启动默认使用规则回退，Gazebo模型只保留
为域偏移对照。机器报告位于
`results/real_machine_smoke/report.json`。

## 4. 核验命令

```bash
ros2 bag info ./my_dataset
sha256sum ./my_dataset/metadata.yaml ./my_dataset/my_dataset_0.db3
```

文件哈希共同标识数据集版本；新版本按相同流程重新生成全部实验结果和报告。
