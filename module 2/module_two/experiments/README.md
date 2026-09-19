# 离线机制验证

在模块二目录运行（无需 ROS）：

```bash
python3 -m pytest tests -q
python3 experiments/run_experiments.py --output results/offline
```

依赖：Python 3.10+、numpy、matplotlib；测试另需 pytest。默认固定种子 100–109，比较七个模型，覆盖六类静态场景及动态障碍、质量变化两组事件。

`--module-one-root` 指向包含 27 个场景子目录的实际 `quality_maps` 目录。不存在时明确记录未检查；存在但数量错误或任意输入解析失败时程序返回非零。每份 CSV 使用其最新观测时间作为参考时间，不混用墙上时钟。原始参考数据只读。

所有数值来自程序实际运行，包含无路、无差异和不利结果。地图、质量场和质量失真均为人为构造，低质量暴露不是定位误差；动态快照也不是控制器闭环仿真。参数事先固定，尚未用独立开发集标定。报告不得被表述为真实 SLAM 精度改善或完整机器人导航验收。
