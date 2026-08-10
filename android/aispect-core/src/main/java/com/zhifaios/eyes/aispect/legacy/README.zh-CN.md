# Android 遗留特征路径

本包只用于历史模型复现和差异审计。

- 允许读取旧模型使用的 `contactPatch.*`、`touchNormalization.*`、`gaussian.*`、`multi.*`、`covariance.*` 与 `time_since_anchor`。
- `causal_imu12_v1` 和 `causal_imu11_touch9_v1` 不会进入本包。
- 新模型不得依赖本包；因果模型必须通过 `AispectCausalPressFeatureBuilder` 从 IMU 与逐帧 `touchFrames[]` 构建输入。
- 本包中的默认值和缺帧补零策略只代表旧模型兼容语义，不代表新训练数据标准。
