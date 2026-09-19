# 正式模型卡

## 标识与用途

模型：`models/context_mlp_formal.yaml`  
SHA-256：`44b2df270c9e19aa4ea2045eb5fb41b7c5bc1fafb9cffc9fa27ecaab485603be`

模型根据单模态质量与环境上下文输出视觉/激光可靠性权重和融合质量，供
下游定位融合与规划模块使用。

## 结构

1. 视觉监督校准：9维标准化逻辑回归，包括跟踪状态和协方差评分；
2. 激光监督校准：9维标准化逻辑回归；
3. 场景分类：`11→16(ReLU)→8(ReLU)→3(Softmax)`；
4. 融合网络：`10→32(ReLU)→16(ReLU)→2(Softmax)`。

融合输入：

```text
Q_v, Q_l,
P(corridor), P(hall), P(outdoor),
illumination_quality,
geometry_complexity,
openness,
visual_valid, laser_valid
```

训练只依赖NumPy，Adam学习率0.0003、批量128、固定随机种子20260727，
并使用标准化特征噪声增强。验证集权重MAE用于早停和最佳轮次恢复。C++
用OpenCV矩阵完成相同前向计算，训练与部署运行时分别为NumPy和OpenCV。

## 正式测试结果

| 指标 | 结果 |
|---|---:|
| 场景宏F1 | 0.8927 |
| 视觉质量—负误差相关性 | 0.9279 |
| 激光质量—负误差相关性 | 0.8255 |
| 失效AUROC | 0.9809 |
| 权重MAE | 0.0841 |
| 相对规则融合改善 | 49.83% |

完整机器报告为 `results/formal_training/training_report.json`。

## 安全行为

- 正常推理：`mlp`；
- 单传感器退化或失效：`mlp_safety_override`；
- 两路均退化：`both_degraded`；
- 文件为空、缺失、格式/维度错误或推理异常：`rule_fallback`。

EMA和进入/恢复迟滞作用于监督校准质量；失效输入立即置零。规则质量比例
定位为可解释基线和故障安全回退。

## 部署与更新

默认配置使用
`package://localization_quality/models/context_mlp_formal.yaml`，安装后
通过ament索引解析。也可把 `fusion.model_path` 设置为任意外部绝对路径，
因此后续实机数据可直接重训替换并复用C++推理接口。

`models/bootstrap_context_mlp.yaml` 用于格式与单元测试；正式指标和部署
使用 `models/context_mlp_formal.yaml`。

## 适用条件

- Softmax权重表达两路传感器的相对可靠性，统计协方差由定位模块独立估计；
- 当前场景分布覆盖走廊、大厅和室外仿真环境，真机场景采用配套数据重训；
- 当前数据seed同时绑定多个因素，部分上下文有冗余；
- 场景+几何消融略优于完整模型，当前证据支持上下文整体增益，光照分量的
  独立贡献将在解耦变量矩阵中评估；
- `my_dataset` 真机烟测已观察到视觉校准域偏移，真机入口采用规则回退；
- 实机部署流程使用真实定位输出和参考真值重采、重训并复验QoS与延迟。
