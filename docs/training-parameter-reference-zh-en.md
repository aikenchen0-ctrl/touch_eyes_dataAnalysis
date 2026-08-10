# 四分类触摸 CNN 训练参数中英对照

> Source / 来源：`android/analysis/train_android_finger_four_class_cnn.py`
> Scope / 范围：拇指/食指 x 轻触/重触四分类 CNN，以及 schema v5 按下锚点数据准备。
> Updated / 更新日期：2026-08-07

## 1. 推荐配置 / Recommended Configuration

当前 schema v5 数据以 `ACTION_DOWN` 为 `frameIndex = 0`。部署候选必须使用按下锚点，且只读取固定窗口中的真实帧；抬起锚点仅用于离线对照。

| 项目 / Item | 当前严格验证配置 / Current strict-validation configuration | 含义 / Meaning |
|---|---|---|
| 锚点 / Anchor | `--anchor-mode down` | 以按下帧为 0 帧 / Use press-down as frame 0 |
| 主窗口 / Main window | `--frame-indices=-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9` | 按下前 5 帧、按下帧、按下后 9 帧，共 15 帧 / 5 pre-down + down + 9 post-down frames, 15 total |
| 缺帧策略 / Missing-frame policy | `--missing-frame-policy error` | schema v5 完整窗口不允许补零 / Reject incomplete windows instead of zero filling |
| 切分 / Split | `--split-strategy predefined` | 使用 manifest 中的 `split` 列 / Use manifest-defined train/valid split |
| 最大轮数 / Maximum epochs | `--epochs 120` | 严格场景留出阶段的上限 / Cap for strict holdout training |
| 早停 / Early stopping | `--early-stopping-patience 20` | 验证 balanced accuracy 连续 20 轮不提升则停止 / Stop after 20 non-improving validation epochs |
| 批大小 / Batch size | `--batch-size 128` | 每次送入优化器的样本数 / Samples per optimizer step |
| 学习率 / Learning rate | `--learning-rate 0.001` | AdamW 更新步长 / AdamW step size |
| 权重衰减 / Weight decay | `--weight-decay 0.0001` | AdamW 的 L2 正则项 / AdamW L2 regularization |
| 随机种子 / Seed | `--seed 22` | 训练顺序和初始化可复现 / Reproducible ordering and initialization |

严格验证胜出的诊断候选为 `old13_raw_radius + -5...+9`。该结果不等于可直接部署：跨运动场景最弱类别召回仍不足，必须保留为诊断模型。

## 2. 参数总表 / Complete CLI Parameter Reference

### 2.1 数据准备模式 / Dataset Preparation Mode

| 参数 / Parameter | 中文说明 | English description | 默认值 / Default | 推荐配置 / Recommended use |
|---|---|---|---|---|
| `--prepare-planned-manifests` | 启用 schema v5 原始 JSON 扫描和 manifest 生成模式；此模式不训练 CNN。 | Enables schema v5 raw JSON scanning and manifest generation; does not train a CNN. | `false` | 生成新数据集训练视图时开启 / Enable when preparing a new raw dataset. |
| `--dataset-dir` | 原始 JSON 根目录。脚本递归扫描其中的 `.json`。 | Root directory of raw JSON files; scanned recursively. | `""` | 准备模式必填 / Required in preparation mode. |
| `--temporal-validation-ratio` | 每个原始 session 尾部划为验证集的比例。 | Fraction of each raw session tail assigned to validation. | `0.20` | `0.20` 仅用于快速时间漂移筛选 / Use `0.20` only for quick temporal screening. |
| `--output-dir` | manifest、模型、scaler、报告的输出目录。 | Output directory for manifests, models, scalers, and reports. | Required | 每次实验使用独立目录 / Use a separate directory per experiment. |

准备模式会校验：`schemaVersion = 5`、`recordStatus = accepted`、`windowComplete = true`、`downReferenceFrameIndex = 0`、按下前至少 15 帧、抬起后至少 15 帧，以及请求的固定窗口全部存在。

### 2.2 输入和标签 / Inputs and Labels

| 参数 / Parameter | 中文说明 | English description | 默认值 / Default | 注意事项 / Notes |
|---|---|---|---|---|
| `--manifest` | 训练 CSV 清单，至少需要 `id`、`path`、`class_label`；预定义切分还需要 `split`。 | Training CSV manifest. It needs at least `id`, `path`, and `class_label`; predefined splits also need `split`. | `""` | 训练模式必填 / Required for training mode. |
| `--feature-preset` | 选择预定义输入通道集合。 | Selects a predefined input-feature set. | `enhanced_patch22` | 与 `--features` 二选一，后者优先 / `--features` overrides this option. |
| `--features` | 以逗号分隔的自定义通道名。 | Comma-separated custom feature names. | `""` | 仅用于受控实验 / Use only for controlled experiments. |
| `--frame-indices` | 相对锚点的帧索引列表。 | Frame indices relative to the selected anchor. | `-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9` | 每个数字对应一个 IMU 时间步 / Each value is one IMU timestep. |
| `--anchor-mode` | 选择时间轴锚点。 | Selects the time-axis anchor. | `down` | `down` 可部署；`lift` 仅离线诊断 / `down` is deployable; `lift` is diagnostic only. |
| `--missing-frame-policy` | 请求帧缺失时的处理方式。 | Handling policy when a requested frame is absent. | `zero` | schema v5 固定窗口使用 `error` / Use `error` for schema v5 fixed windows. |

`--missing-frame-policy` 的取值：

| 值 / Value | 中文 | English |
|---|---|---|
| `zero` | 用全零特征行补缺帧，主要为旧数据兼容。 | Zero-fill missing frames, mainly for legacy compatibility. |
| `skip` | 丢弃存在缺帧的样本。 | Drop samples with missing frames. |
| `error` | 立即报错，阻止不完整窗口进入训练。 | Fail immediately and prevent incomplete windows from entering training. |

### 2.3 模型标识和输出 / Model Identity and Output

| 参数 / Parameter | 中文说明 | English description | 默认值 / Default |
|---|---|---|---|
| `--model-id` | 模型唯一标识，用于权重、scaler、catalog 文件名。 | Unique model identifier used in weights, scaler, and catalog filenames. | `android_finger_four_class_patch22_pre5_post9` |
| `--display-name` | 面向人类的模型名称，写入训练报告和 catalog 条目。 | Human-readable model name written to the training report and catalog entry. | `android finger four class patch22 -5...+9` |
| `--output-dir` | 输出 `AispectImpactCNNWeights_*.json`、`AispectImpactCNNScaler_*.json`、`training_report.json` 和 `catalog_entry.json`。 | Writes weights, scaler, training report, and catalog entry. | Required |

四分类标签顺序固定为：

| 类别 / Class label | 中文 |
|---|---|
| `thumb_light` | 拇指轻触 / Thumb light touch |
| `thumb_heavy` | 拇指重触 / Thumb heavy touch |
| `index_light` | 食指轻触 / Index-finger light touch |
| `index_heavy` | 食指重触 / Index-finger heavy touch |

### 2.4 优化器和训练轮次 / Optimizer and Training Schedule

| 参数 / Parameter | 中文说明 | English description | 默认值 / Default | 本轮建议 / Current recommendation |
|---|---|---|---|---|
| `--epochs` | 最大训练轮数，不保证一定跑满。 | Maximum epochs; training may stop earlier. | `120` | 严格折 `120`；筛选阶段 `60`；全量最终训练使用严格折中位最佳轮数。 |
| `--batch-size` | 单个 mini-batch 的样本数。 | Number of samples in one mini-batch. | `128` | `128` |
| `--learning-rate` | AdamW 的学习率，控制每次参数更新幅度。 | AdamW learning rate; controls update magnitude. | `0.001` | `0.001` |
| `--weight-decay` | AdamW 权重衰减，抑制过拟合。 | AdamW weight decay; regularizes model weights. | `0.0001` | `0.0001` |
| `--early-stopping-patience` | 相对最佳验证 balanced accuracy 可容忍的不提升轮数。`0` 表示关闭。 | Number of non-improving validation epochs tolerated after the best balanced accuracy. `0` disables early stopping. | `0` | 严格折 `20`；筛选 `12`；`all` 全量训练保持 `0`。 |
| `--seed` | Python、PyTorch 和数据顺序随机种子。 | Random seed for Python, PyTorch, and data ordering. | `22` | `22`，多 seed 消融时单独记录。 |

训练使用 AdamW、类别逆频率 `CrossEntropyLoss` 和 manifest 的 `sample_weight`。验证的模型选择指标是 `balancedAccuracy`，而不是普通 accuracy。

### 2.5 数据切分和加权 / Splits and Weighting

| 参数 / Parameter | 中文说明 | English description | 默认值 / Default | 何时使用 / When to use |
|---|---|---|---|---|
| `--split-strategy random` | 每个类别随机分层切分。 | Random stratified split within each class. | Default | 仅作历史对照，不用于最终选择。 |
| `--split-strategy temporal` | 每个类别按 `createdAt` 排序，最新尾段为验证集。 | Per-class temporal tail split by `createdAt`. | Optional | 检查时间漂移，但可能仍共享 session。 |
| `--split-strategy predefined` | 直接读取 manifest 的 `split=train/valid`。 | Reads `split=train/valid` from the manifest. | Recommended | 场景留出、设备留出、人工指定折。 |
| `--split-strategy all` | 所有样本进入训练，不保留验证集。 | Uses every sample for training and keeps no validation set. | Optional | 仅在严格验证确定 epoch 后训练最终产物。 |
| `--validation-ratio` | `random` 或 `temporal` 时的验证比例。 | Validation ratio for `random` or `temporal` splits. | `0.25` | 对 `predefined` 和 `all` 无效。 |
| `--group-balance` | 对训练折中的 `collectionName + classLabel` 做逆频率加权。 | Applies inverse-frequency weighting by `collectionName + classLabel` in the training split. | `false` | 仅在 collection 不均衡且分组定义可信时启用。 |
| `--sample-cache` | 特征矩阵缓存 `.pkl` 路径。 | Path to a feature-matrix cache `.pkl`. | `""` | 同一 preset、窗口、锚点跨折复用，减少 JSON 重读。 |

缓存签名包含特征名、帧索引、缺帧策略和锚点模式。任一项变化时不得复用旧缓存。

### 2.6 输入变换 / Input Matrix Transform

| `--matrix-transform` 值 | 中文说明 | English description | 默认值 |
|---|---|---|---|
| `none` | 不变换，直接使用 preset 特征。 | No transform; use preset features directly. | Default |
| `imu11_no_time` | 只保留坐标和 9 个 IMU 通道。 | Keeps coordinates plus nine IMU channels. | Optional |
| `invariant_shape` | 加入无量纲的轴比、协方差相关性和偏心率。 | Adds dimensionless axis ratio, covariance correlation, and eccentricity. | Optional |
| `relative_touch` | 使用同组历史接触形态的因果相对比值。 | Uses causal relative contact-shape ratios from prior same-group samples. | Optional |
| `baseline_imu` | 用按下前帧的稳健中心和尺度归一化 IMU。 | Normalizes IMU using robust pre-touch centers and scales. | Optional |

本轮 `bbsiak` 主消融保持 `none`，避免同时改变特征语义、窗口和验证切分。

## 3. Feature Preset / 特征预设

| Preset | 中文说明 | English description |
|---|---|---|
| `old13` | 坐标、加速度、陀螺仪、重力和相对时间 13 通道。 | 13 channels of coordinates, acceleration, gyroscope, gravity, and relative time. |
| `old13_raw_radius` | `old13` 加原始长短轴和有效标记。 | `old13` plus raw major/minor radii and validity marker. |
| `old13_covariance` | `old13` 加接触协方差特征。 | `old13` plus contact covariance features. |
| `old13_multichannel` | `old13` 加多通道接触形态特征。 | `old13` plus multi-channel contact-shape features. |
| `old13_gaussian_summary` | `old13` 加高斯接触图摘要。 | `old13` plus Gaussian contact-map summaries. |
| `old13_gaussian_map9x9` | `old13` 加完整 9x9 高斯接触图。 | `old13` plus the full 9x9 Gaussian contact map. |
| `gaussian_map9x9_only` | 仅完整高斯接触图。 | Full Gaussian contact map only. |
| `multichannel_only` | 仅多通道接触特征。 | Multi-channel contact features only. |
| `covariance_only` | 仅接触协方差特征。 | Contact covariance features only. |
| `patch9` | 九个基础接触 patch 特征。 | Nine basic contact-patch features. |
| `enhanced_patch21_no_confidence` | `old13` 加八个 patch 特征，不含质量置信度。 | `old13` plus eight patch features, excluding capture-quality confidence. |
| `enhanced_patch22` | `old13` 加九个 patch 特征，包含质量置信度。 | `old13` plus nine patch features, including capture-quality confidence. |

## 4. 当前执行命令 / Current Command Examples

### 4.1 生成 schema v5 manifest / Prepare schema v5 manifests

```powershell
python android\analysis\train_android_finger_four_class_cnn.py `
  --prepare-planned-manifests `
  --dataset-dir "C:\Users\血饮\Desktop\bbsiak" `
  --output-dir "android\analysis\datasets\bbsiak_schema5_down_anchor_20260806" `
  --frame-indices=-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9 `
  --temporal-validation-ratio 0.20
```

### 4.2 严格场景留出训练 / Strict motion-holdout training

```powershell
python android\analysis\train_android_finger_four_class_cnn.py `
  --manifest "android\analysis\datasets\bbsiak_schema5_down_anchor_20260806\folds\motion_walking.csv" `
  --output-dir "models\bbsiak_schema5_down_anchor_20260806\motion_holdout\old13_raw_radius__pre5_post9\walking" `
  --model-id "bbsiak_motion_old13_raw_radius_pre5_post9_walking" `
  --feature-preset old13_raw_radius `
  --frame-indices=-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9 `
  --anchor-mode down `
  --missing-frame-policy error `
  --split-strategy predefined `
  --epochs 120 `
  --early-stopping-patience 20 `
  --batch-size 128 `
  --learning-rate 0.001 `
  --weight-decay 0.0001 `
  --seed 22
```

### 4.3 全量最终训练 / Final all-data training

```powershell
python android\analysis\train_android_finger_four_class_cnn.py `
  --manifest "android\analysis\datasets\bbsiak_schema5_down_anchor_20260806\master_manifest.csv" `
  --output-dir "models\bbsiak_schema5_down_anchor_20260806\final_all\old13_raw_radius_pre5_post9" `
  --model-id "bbsiak_schema5_down_pre5_post9_old13_raw_radius_all_e36" `
  --feature-preset old13_raw_radius `
  --frame-indices=-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9 `
  --anchor-mode down `
  --missing-frame-policy error `
  --split-strategy all `
  --epochs 36 `
  --batch-size 128 `
  --learning-rate 0.001 `
  --weight-decay 0.0001 `
  --seed 22
```

`all` 模式不产生验证指标。`36` 是严格三场景留出中该候选的中位最佳 epoch，不应根据全量训练 accuracy 再次挑选模型。

## 5. 固定模型结构 / Fixed Model Architecture

模型结构不是 CLI 参数，当前脚本中固定如下：

```text
Input [batch, channels, frames]
  -> Conv1D channels -> 32, kernel=3
  -> BatchNorm + ReLU + Dropout(0.10)
  -> Conv1D 32 -> 64, kernel=3
  -> BatchNorm + ReLU + MaxPool1D(2) + Dropout(0.10)
  -> Conv1D 64 -> 128, kernel=3
  -> BatchNorm + ReLU + MaxPool1D(2)
  -> Linear -> 64 -> ReLU -> Dropout(0.10)
  -> Linear -> 4 classes
```

输入帧数由 `--frame-indices` 决定。过短窗口会降低时间信息，也会改变池化后的特征长度，因此窗口消融必须保持模型结构、学习率、batch size 和切分不变。

## 6. 报告字段 / Training Report Fields

每次训练会输出 `training_report.json`，其中重要字段如下：

| 字段 / Field | 中文 | English |
|---|---|---|
| `args` | 本次完整 CLI 配置。 | Complete CLI configuration for this run. |
| `bestEpoch` | 验证 balanced accuracy 最佳的轮数；`all` 模式为最后一轮。 | Epoch with best validation balanced accuracy; final epoch for `all`. |
| `anchorMode` | `down` 或 `lift`。 | `down` or `lift`. |
| `trainMetrics` / `validMetrics` | 总体 accuracy、balanced accuracy、最小召回和混淆矩阵。 | Overall accuracy, balanced accuracy, minimum recall, and confusion matrix. |
| `trainSubgroupMetrics` / `validSubgroupMetrics` | 按运动场景和按压时长桶分组的指标。 | Metrics grouped by motion state and press-duration bucket. |
| `trainClassCounts` / `validClassCounts` | 训练和验证中的四类样本数。 | Four-class sample counts in training and validation. |

## 7. 结果解读规则 / Result Interpretation Rules

1. 优先比较严格 `predefined` 场景留出的 mean/worst balanced accuracy，而不是随机切分。
   Prefer strict predefined motion-holdout mean/worst balanced accuracy over random-split accuracy.
2. 同时检查 `validMinRecall`；某一类接近零时，较高平均分不代表模型可用。
   Check `validMinRecall`; a near-zero class recall invalidates an otherwise high average score.
3. `lift` 锚点结果不能作为按下期部署模型的性能声明。
   A `lift`-anchor result cannot be claimed as press-phase deployment performance.
4. `all` 全量训练 accuracy 只是拟合状态，不是泛化结论。
   All-data training accuracy is fit status, not a generalization result.
5. schema v5 的长按压样本应按 `duration_bucket` 单独审计，不能让抬起时间隐式成为模型输入。
   Audit long-press schema v5 samples by `duration_bucket`; do not let lift duration become an implicit input.
