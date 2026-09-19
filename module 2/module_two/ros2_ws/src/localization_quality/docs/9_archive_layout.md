# 实验归档结构与复跑

## 主归档

主目录包含：

- 源码、配置、launch、测试和仿真世界生成器；
- `data/formal_gazebo/` 中27组已接受bag及相对路径元数据；
- `my_dataset/` 真机接口烟测bag；
- `results/formal_training/training_rows.csv`、固定划分、最终报告和运行证据；
- `results/formal_training/quality_maps/` 中27张独立地图及逐运行摘要；
- `quality_map_summary.json/csv` 总体与分组统计；
- 正式模型、数据摘要与 `results/release_manifest.json`。

构建目录、安装目录、Python缓存、ROS日志、逐运行中间CSV、消融临时模型和
早期结果集中存放在冷归档。

## 冷归档

早期运行、过程产物和历史实现移动到工作区同级
`localization_quality_cold_archive/`，用于开发追溯。正式数据内容哈希和
模块一验收聚焦主归档中的已接收产物。

## 一键复跑

```bash
./scripts/reproduce_archive.sh /tmp/localization_quality_reproduced
```

脚本在临时目录中完成干净构建、27组特征提取、训练、消融、故障回归、
27张独立地图、总体地图统计和模块一验收。离线提取固定OpenCV线程数与
随机种子；规则基线固定使用 `q_visual_raw/q_laser_raw`。输出报告同时
记录训练表、划分、模型和地图产物SHA-256。

真机bag用于接口、输出完整率和时延烟测，机器结论见
`results/real_machine_smoke/report.json`。真机入口采用规则回退，仿真模型
作为域偏移对照。

最终清单必须包含真机大文件：

```bash
python3 scripts/create_hash_manifest.py . results/release_manifest.json \
  --include-real-bag
```

命令同时生成 `release_manifest.sha256`。
