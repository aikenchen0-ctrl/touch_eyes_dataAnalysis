# 历史训练路径

本目录用于存放旧 schema v5 训练与消融说明。

历史路径可能读取以下非因果字段：

- 根级 `contactPatch`；
- 根级 `touchNormalization`；
- `frames[].timeSinceAnchor`；
- 按整次触摸聚合的高斯图、协方差、半径、面积和置信度。

这些路径只允许用于历史模型复现和差异审计，不得用于 schema v6 按下期模型训练，也不得作为 Android 默认模型输入。

当前因果训练入口是：

```text
android/analysis/train_android_finger_four_class_cnn.py
```

它只能接受 `causal_*` feature contract，并从逐帧 `touchFrames[]` 按双时间门 ZOH 重建接触特征。
