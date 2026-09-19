# 正式数据卡

## 数据概览

目录：`data/formal_gazebo`。

```text
3场景（corridor/hall/outdoor）
× 3光照（dark/normal/strong）
× 3独立运行（seed 0/1/2）
= 27条bag
```

27/27运行通过矩阵与话题合同校验。共722,970条消息，哈希覆盖
8,074,564,489字节。数据集内容SHA-256为
`1487b9b47b04d46fdf42922dfc9eaa4c24a85e5196bf5ba2faa9f45016512f33`；
完整文件清单见 `results/formal_training/formal_dataset_summary.json`。

| seed | 几何复杂度 | 故障 | 路线 | 速度 |
|---:|---|---|---|---:|
| 0 | low | normal | route_a | 0.40 m/s |
| 1 | medium | visual_degraded | route_b | 0.70 m/s |
| 2 | high | laser_degraded | route_c | 1.00 m/s |

这种设计满足27条正式矩阵。seed同时绑定几何、故障、路线和速度，构成
实验结论中的混杂因素。

## 传感器与消息规模

每条运行约90秒，记录：

- RGB与CameraInfo：868–898帧，10 Hz目标；
- LaserScan：1295–1340帧，15 Hz目标；
- PointCloud2：434–449帧，5 Hz目标；
- odom、视觉/激光定位、真值和TF：约4345–4490帧，50 Hz目标；
- TF_STATIC、Clock和运行元数据。

每条 `run_metadata.json` 保存场景、光照、几何、故障、路线、速度、seed、
话题映射、定位来源、原生频率下限、消息计数以及db3/metadata文件SHA-256。
其中 `world` 和 `bag_path` 均相对元数据文件解析，可随归档目录整体迁移。

## 标签

以LaserScan时间戳为主，使用当前时刻或历史范围内最新的定位和真值。
平移误差为三维欧氏距离，旋转误差为单位四元数夹角。10帧因果窗口内计算：

```text
Q* = exp(-sqrt((translation/0.50)^2 + (rotation/0.35)^2))
r_i = 1 / (loss_i + 1e-6)
w_i* = r_i / (r_visual + r_laser)
```

有限且完整的真值行进入监督数据集，其余行统一计数归档。

## 划分

固定种子20260727生成16/5/6条训练/验证/测试运行，共35,825帧。划分单位
是完整 `run_id`，每条轨迹完整归属一个集合。三个集合各自覆盖三场景、三光照和
seed0/1/2三种故障—几何模式。具体运行列表见
`results/formal_training/dataset_split.json`。

## 空间质量产物

27条运行分别生成独立 `odom` 坐标质量地图，共汇集35,827个空间质量样本
和3,537个累计已知栅格。每条地图保留逐帧运行CSV、六格式栅格、元数据及
运行摘要。总体报告按数据划分、场景、光照、故障和几何复杂度组织统计，
位于 `results/formal_training/quality_map_summary.json` 和
`results/formal_training/quality_map_summary.csv`。

## 来源与适用范围

仿真定位结果由明确标注的 `controlled_ground_truth_noise_emulator`
产生，产物类型为可控真值噪声定位代理，可用于程序化算法开发、监督标签、
消融和安全回归。实机HDR、眩光、雨雾、玻璃多径及真实定位器泛化由下一
阶段真机数据矩阵验收。

`my_dataset` 是项目负责人确认的真机bag，用于接口与时延烟测。正式训练
采用27条Gazebo bag。早期开发运行存入工作区同级冷归档；正式矩阵与数据
内容哈希覆盖27条已接收运行。
