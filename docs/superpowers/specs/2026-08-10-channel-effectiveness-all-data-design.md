# 全量通道有效度消融设计

## 目标

在不改动原始 JSON、清洗数据或严格三折基线的前提下，对当前 `0810_test_time_grid_v2_clean` 的全部合格样本执行全量拟合通道消融。输出用于识别明确冗余通道和当前数据上的拟合差异，不作为跨 session 或跨设备泛化结论。

## 范围

- 输入数据固定为 `data/training/0810_test_time_grid_v2_clean` 的 2,582 条 eligible 样本。
- 每个合同使用全部样本拟合 scaler 与模型，不读取 `strict_group_folds.csv`，不创建验证集，不早停。
- 训练超参数沿用严格基线配置，固定 80 epochs。
- 运行 9 个通道合同，最多 7 个独立 worker process 并行。
- 输出根目录固定为 `models/通道有效度消融测试`。

## 合同

1. `imu11`
2. `touch23`
3. `touch16_deduplicated`
4. `touch15_geometry`
5. `imu14_geometry`
6. `imu12_touch_valid`
7. `imu8_no_gravity`
8. `imu6_motion`
9. `imu9_no_position`

`touch16_deduplicated` 保留一个 `touch_valid`，删除当前数据中与它逐值完全相同的七个通道。其余合同用于分别检测几何、形状比例、绝对姿态、动态信号和位置的拟合贡献。

## 输出与边界

每个合同输出权重、scaler、模型合同、按 epoch 的训练历史和训练指标。汇总文件按训练集平衡准确率、Macro-F1、最差类别召回率和最终损失排序。所有输出必须标记 `evaluationScope=all_data_fit_only` 与 `deploymentAllowed=false`。

全量训练没有验证信号，不能将排名解释为真实泛化有效度，不能部署到 Android。
