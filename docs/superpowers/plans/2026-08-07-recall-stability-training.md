# 按下期模型召回率与稳定性训练计划 / Recall Stability Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标 / Goal:** 提升四分类按下期模型中最弱类别的召回率，同时不把运动留出集当作训练期选择集，并加入有界梯度训练诊断。 / Improve the four-class press-phase model's weakest-class recall without treating a motion holdout as a training-time selection set, and add bounded-gradient training diagnostics.

**架构 / Architecture:** 固定已选因果窗口（`pre1_post2`，帧 `-1,0,1,2`）和特征预设（`old13_raw_radius`）。先将数据契约从训练/验证改为训练/开发/测试，再为现有 CNN 训练器增加可配置的正则化、损失函数、学习率调度和梯度控制选项。新的可恢复运行器分两阶段筛选参数，只有在开发集选定配置后才公开测试集指标。 / Keep the selected causal window (`pre1_post2`, frames `-1,0,1,2`) and feature preset (`old13_raw_radius`) fixed. First change the data contract from train/validation to train/development/test, then add configurable regularization, loss, scheduler, and gradient-control options to the existing CNN trainer. A new resumable runner performs two small, staged parameter screens and exposes test metrics only after a development-selected configuration is locked.

**技术栈 / Tech Stack:** Python 标准库、PyTorch、现有 manifest CSV、现有 JSON 模型/报告格式、`unittest`。 / Python standard library, PyTorch, existing manifest CSV files, existing JSON model/report format, `unittest`.

---

## 中文执行摘要 / Chinese Execution Summary

本阶段不再改时间窗口或特征，而是只验证训练策略是否能提升 `thumb_heavy / 重压拇指` 的最弱类别召回率。当前三折中它分别为 `57.64%`、`50.24%`、`47.96%`，是共同短板。 / This phase does not change the time window or features. It tests whether the training strategy can improve `thumb_heavy`, the shared weakest class across the three folds at `57.64%`, `50.24%`, and `47.96%` recall.

当前没有梯度范数日志，因此不能断言已发生梯度爆炸。梯度裁剪用于限制偶发尖峰并留下诊断证据；真正控制过拟合的是开发集与测试集隔离、权重衰减、适度 dropout、学习率衰减和按最弱类别召回率选择检查点。 / There is no gradient-norm log yet, so gradient explosion is not established. Gradient clipping bounds rare spikes and creates evidence; overfitting is primarily controlled through dev/test isolation, weight decay, moderate dropout, learning-rate decay, and recall-aware checkpoint selection.

### 核心术语 / Core Terminology

| 中文 | English | 本计划中的含义 / Meaning in this plan |
|---|---|---|
| 训练集 | Training set | 用于反向传播和更新参数。 / Used for backpropagation and parameter updates. |
| 开发集 | Development set / Dev set | 用于早停和参数选择，不是最终测试。 / Used for early stopping and parameter selection, not final testing. |
| 测试集 | Test set | 配置锁定后才评估，不参与任何训练决策。 / Evaluated only after configuration lock; never used for training decisions. |
| 最弱类别召回率 | Minimum class recall | 四个类别中最低的召回率。 / The lowest recall among the four classes. |
| 平衡准确率 | Balanced accuracy | 四个类别召回率的平均值。 / Mean recall across the four classes. |
| 权重衰减 | Weight decay | 约束参数规模以减少记忆训练样本。 / Constrains parameter magnitude to reduce memorization. |
| 丢弃率 | Dropout rate | 训练时随机失活部分特征连接。 / Randomly deactivates feature connections during training. |
| 学习率预热 | Learning-rate warmup | 前几轮从较低学习率平滑升高。 / Smoothly increases from a lower learning rate in initial epochs. |
| 余弦退火 | Cosine annealing | 按预定曲线逐步降低学习率。 / Gradually lowers learning rate on a predefined curve. |
| 梯度裁剪 | Gradient clipping | 将全局梯度 L2 范数限制在阈值内。 / Limits global gradient L2 norm to a threshold. |
| 焦点损失 | Focal loss | 对难分类样本赋予更多损失权重。 / Places more loss emphasis on hard examples. |
| 标签平滑 | Label smoothing | 降低过度自信输出的轻度正则化。 / Mild regularization that reduces overconfident outputs. |

### 推荐起点 / Recommended Starting Profile

| 中文参数 | English parameter | 值 / Value | 目的 / Purpose |
|---|---|---:|---|
| 优化器 | Optimizer | `AdamW` | 保留解耦权重衰减。 / Keep decoupled weight decay. |
| 学习率 | Learning rate | `5e-4` | 降低训练初期波动。 / Reduce early optimization volatility. |
| 预热与调度 | Warmup and scheduler | `3` epochs + cosine to `1e-5` | 不读取开发/测试指标的平滑衰减。 / Smooth decay without reading dev/test metrics. |
| 权重衰减 | Weight decay | `5e-4` | 抑制记忆。 / Reduce memorization. |
| 丢弃率 | Dropout rate | `0.15` | 提供适度正则化。 / Apply moderate regularization. |
| 梯度裁剪阈值 | Gradient clip norm | `1.0` | 限制偶发梯度尖峰。 / Bound rare gradient spikes. |
| 最大轮数 | Maximum epochs | `80` | 保留足够收敛空间而非无限拟合。 / Allow convergence without open-ended fitting. |
| 早停耐心值 | Early-stopping patience | `15` | 等待有意义的开发集改善。 / Wait for meaningful dev improvement. |
| 最小改善量 | Selection minimum delta | `0.003` | 防止由微小波动替换最佳检查点。 / Avoid replacing a checkpoint on tiny fluctuations. |

### 三阶段实验 / Three-Stage Experiment

| 阶段 | English stage | 训练次数 | 目标 / Objective |
|---|---|---:|---|
| 阶段 A：优化器与正则化 | Stage A: Optimizer and regularization | `45` | 确认学习率、调度、裁剪、权重衰减和 dropout 的稳定组合。 / Identify a stable combination of learning rate, schedule, clipping, weight decay, and dropout. |
| 阶段 B：召回率损失函数 | Stage B: Recall-loss screen | `36` | 在锁定的稳定参数上比较加权 CE、标签平滑和 Focal Loss。 / Compare weighted CE, label smoothing, and focal loss with stable parameters locked. |
| 阶段 C：锁定确认 | Stage C: Locked confirmation | `15` | 参数锁定后才评价外部测试集并报告方差。 / Evaluate outer test sets and report variance only after parameter lock. |

## 证据与约束 / Evidence And Constraints

- The current offline best is `pre1_post2`: mean balanced accuracy `69.63%`, worst-fold balanced accuracy `68.04%`, and worst-fold weakest-class recall `47.96%`.
- `thumb_heavy` is the weakest class in all three motion folds (`57.64%`, `50.24%`, and `47.96%`). The class counts are not severely imbalanced, so indiscriminate oversampling is not justified.
- The current trainer already uses AdamW, inverse-frequency class weights, and `0.10` dropout. It has no learning-rate scheduler, gradient clipping, gradient telemetry, label smoothing, or focal-loss option.
- In the `handheld_static` fold, the best checkpoint is epoch `4`; continued training reaches epoch `24` before early stopping, while validation balanced accuracy falls from `68.96%` to `52.65%`. The delivered checkpoint is restored to epoch `4`, but the behavior warrants stronger checkpoint selection and telemetry.
- The current motion-holdout `valid` split is used every epoch for checkpoint selection. It is therefore a development split, not an untouched test set.
- The schema-v5 dataset has `12,172` rows from `19` sessions and one collector. Most large sessions correspond to a single class and motion state. Optimizer changes cannot remove this collection confound; new interleaved-label sessions remain necessary for a production claim.

## 范围与非目标 / Scope And Non-Goals

- Keep `--anchor-mode down`, `--require-press-active`, `--missing-frame-policy error`, `pre1_post2`, and `old13_raw_radius` fixed for this phase.
- Do not change Android runtime code, model catalog assets, raw data, labels, or the time-window definition.
- Do not use mixup, arbitrary frame shifts, or per-channel random noise in the first cycle. With a four-frame causal window, those augmentations can destroy the event-time meaning that the ablation selected.
- Do not evaluate or rank candidate configurations on the outer test split during parameter screening.

## 评估契约 / Evaluation Contract

For each outer motion fold, transform the existing manifest as follows:

| 原始划分 / Existing split | 新划分 / New split | 规则 / Rule |
|---|---|---|
| `valid` target-motion rows | `test` | Never used for loss, optimizer, scheduler, early stopping, threshold calibration, or configuration ranking. |
| `train` rows | `train` and `dev` | Within every `raw_session_id` and class, order by `created_at`; for groups with at least three rows, reserve the newest `15%` for `dev` while keeping at least two rows in `train`. Smaller groups remain wholly in `train`; the builder then verifies every class is represented in `dev`. |

The training checkpoint metric is calculated from `dev` only:

```text
selectionScore = 0.70 * devMinRecall + 0.30 * devBalancedAccuracy
```

This gives the lowest class recall primary influence while preventing a configuration from improving one class by collapsing all others. A checkpoint replaces the previous best only when `selectionScore > bestSelectionScore + 0.003`; early stopping uses patience `15`. The report must expose raw `devMinRecall`, `devBalancedAccuracy`, and `selectionScore` for every epoch.

The outer test is evaluated only by the final confirmation command after the selected configuration has been written to `selected_profile.json`. A separately collected, label-interleaved schema-v5 dataset remains the only meaningful final blind test for future sessions/devices.

## 参数契约 / Parameter Contract

| 参数 / Parameter | 当前 / Current | 稳定起点 / Stable starting profile | 目的 / Purpose |
|---|---:|---:|---|
| Optimizer | AdamW | AdamW | Retain decoupled weight decay. |
| Learning rate | `1e-3` | `5e-4` | Reduce early optimization volatility. |
| Scheduler | none | 3-epoch linear warmup then cosine decay to `1e-5` over 80 epochs | Decay without reading the dev/test score. |
| Epoch limit | `120` | `80` | The current best epochs are 4, 31, and 55; 80 leaves convergence room without open-ended fitting. |
| Early stopping | BA, patience 20 | recall-balanced score, patience 15, min delta `0.003` | Align checkpoint selection to the stated recall objective. |
| Weight decay | `1e-4` | `5e-4` | Mildly penalize memorization; test separately from stronger regularization. |
| Dropout | `0.10` | `0.15` | Moderate regularization for the small four-frame model. |
| Class loss | inverse-frequency weighted CE | same initially | Preserve the current class-balancing baseline. |
| Gradient clipping | none | global L2 norm `1.0` | Bound rare gradient spikes; not a substitute for regularization. |
| Batch size | `128` | `128` | Keep the BatchNorm operating regime fixed. |

Gradient clipping is applied immediately after `loss.backward()` and before `optimizer.step()`. The trainer must abort on a non-finite pre-clip norm rather than silently writing a corrupted model. Each report must record `meanPreClipGradNorm`, `p95PreClipGradNorm`, `maxPreClipGradNorm`, `clippedStepCount`, `totalStepCount`, `nonFiniteGradientCount`, and `finalLearningRate`.

Interpretation rule: clipping on fewer than `5%` of steps is a guardrail; clipping on more than `10%` of steps is a stability failure, so reduce the learning rate rather than raising the clip threshold.

## 分阶段实验矩阵 / Staged Experiment Matrix

All screening runs use outer folds `handheld_static`, `placed_static`, and `walking`, and seeds `22`, `41`, and `73`. Each stage therefore uses the same nine train/dev jobs per configuration. Do not change more than one conceptual family at a time.

### 阶段 A：优化器与正则化筛选 / Stage A: Optimizer And Regularization Screen

| 标识 / ID | 学习率 / 调度 / Learning rate / schedule | 权重衰减 / Weight decay | 丢弃率 / Dropout | 裁剪阈值 / Clip norm | 损失 / Loss |
|---|---|---:|---:|---:|---|
| `A0_current` | `1e-3`, none | `1e-4` | `0.10` | disabled | weighted CE |
| `A1_clip_only` | `1e-3`, none | `1e-4` | `0.10` | `1.0` | weighted CE |
| `A2_stable_optimizer` | `5e-4`, warmup+cosine | `1e-4` | `0.10` | `1.0` | weighted CE |
| `A3_moderate_regularization` | `5e-4`, warmup+cosine | `5e-4` | `0.15` | `1.0` | weighted CE |
| `A4_strong_regularization` | `5e-4`, warmup+cosine | `1e-3` | `0.20` | `1.0` | weighted CE |

Run count: `5 x 3 folds x 3 seeds = 45` development-screen runs.

Rank profiles by: (1) worst dev minimum recall across all nine runs, (2) mean dev selection score, (3) mean dev balanced accuracy, (4) lower standard deviation, and (5) lower clip fraction. A profile is eligible only if it improves or matches `A0_current` worst dev minimum recall, does not lower mean dev balanced accuracy by more than `1` percentage point, and has zero non-finite gradients.

### 阶段 B：召回率损失函数筛选 / Stage B: Recall-Loss Screen

Lock the Stage-A winner's optimizer and regularization parameters. Compare exactly these loss modes:

| 标识 / ID | 损失模式 / Loss mode | Focal gamma | 标签平滑 / Label smoothing |
|---|---|---:|---:|
| `B0_weighted_ce` | inverse-frequency weighted cross entropy | `0` | `0` |
| `B1_smoothed_ce` | inverse-frequency weighted cross entropy | `0` | `0.02` |
| `B2_focal_05` | inverse-frequency weighted focal cross entropy | `0.5` | `0` |
| `B3_focal_10` | inverse-frequency weighted focal cross entropy | `1.0` | `0` |

Focal loss and label smoothing are deliberately not combined. Run count: `4 x 3 folds x 3 seeds = 36` development-screen runs. Use the same eligibility and ranking rules as Stage A. Do not add oversampling unless a separate data audit demonstrates a real class-count imbalance after the inner split.

### 阶段 C：锁定确认 / Stage C: Locked Confirmation

1. Write the selected Stage-B profile and development ranking to `selected_profile.json` before generating any outer test metrics.
2. Run the selected profile for the three motion folds with seeds `22`, `41`, `73`, `101`, and `131`: `3 x 5 = 15` confirmation trainings.
3. Evaluate each restored dev-selected checkpoint exactly once on its corresponding outer `test` split.
4. Report mean, standard deviation, and worst values for test minimum recall, balanced accuracy, and all four class recalls. Do not deploy from this result.
5. Evaluate the locked profile once on a newly collected, interleaved-label, unseen-session dataset. This is the release decision input; it must not be used to alter any parameter.

## 文件结构 / File Structure

- Modify: `android/analysis/train_android_finger_four_class_cnn.py`
  - Add explicit train/dev/test handling, stable optimizer options, focal/label-smoothed loss, gradient telemetry, and dev-only checkpoint selection.
- Create: `android/analysis/build_schema5_recall_stability_folds.py`
  - Convert current motion manifests to deterministic train/dev/test manifests without changing raw data.
- Create: `android/analysis/run_android_recall_stability_ablation.py`
  - Enumerate Stage A, Stage B, and locked Stage C; resume completed jobs; never rank on test metrics.
- Create: `android/analysis/tests/test_build_schema5_recall_stability_folds.py`
  - Verify split exclusivity, class coverage, deterministic temporal allocation, and unchanged outer test membership.
- Create: `android/analysis/tests/test_run_android_recall_stability_ablation.py`
  - Verify exact job counts, argument forwarding, resume rules, and dev-only ranking.
- Modify: `android/analysis/tests/test_train_android_finger_four_class_cnn.py`
  - Verify new CLI defaults/validation, score selection, focal-loss identity at gamma zero, clipping telemetry, and no test-set checkpoint selection.
- Modify: `FunctionAndDataStructureDescription.md`
  - Document the new train/dev/test contract, CLI parameters, telemetry fields, staged runner, and offline-only restriction.

## 实施任务 / Implementation Tasks

### 任务 1：构建确定性训练/开发/测试 Manifest / Task 1: Build Deterministic Train/Dev/Test Manifests

**Files:**
- Create: `android/analysis/build_schema5_recall_stability_folds.py`
- Create: `android/analysis/tests/test_build_schema5_recall_stability_folds.py`

- [ ] **Step 1: Write failing split-contract tests.**

```python
def test_inner_dev_split_preserves_outer_test_and_class_coverage(self):
    rows = make_motion_fold_rows()
    result = builder.build_train_dev_test_rows(rows, dev_ratio=0.15)
    assigned = {row["id"]: row["split"] for row in result}
    test_ids = {row["id"] for row in result if row["split"] == "test"}
    non_test_ids = {row["id"] for row in result if row["split"] != "test"}

    self.assertEqual(
        {row["id"] for row in rows if row["split"] == "valid"},
        test_ids,
    )
    self.assertEqual(len(rows), len(assigned))
    self.assertEqual({row["id"] for row in rows}, set(assigned))
    self.assertFalse(test_ids & non_test_ids)
    for split in ("train", "dev", "test"):
        self.assertEqual(set(trainer.CLASS_ORDER), {
            row["class_label"] for row in result if row["split"] == split
        })
```

- [ ] **Step 2: Run the focused test and confirm it fails because the builder does not exist.**

```powershell
python -m unittest android.analysis.tests.test_build_schema5_recall_stability_folds
```

- [ ] **Step 3: Implement deterministic per-session temporal allocation.**

```python
def build_train_dev_test_rows(rows, dev_ratio):
    test_rows = [dict(row, split="test") for row in rows if row["split"] == "valid"]
    train_sources = [row for row in rows if row["split"] == "train"]
    grouped = group_by_session_and_class(train_sources)
    output = []
    for key in sorted(grouped):
        ordered = sorted(grouped[key], key=lambda row: (float(row["created_at"]), row["id"]))
        if len(ordered) < 3:
            output.extend(dict(row, split="train") for row in ordered)
            continue
        dev_count = max(1, round(len(ordered) * dev_ratio))
        dev_count = min(dev_count, len(ordered) - 2)
        output.extend(dict(row, split="train") for row in ordered[:-dev_count])
        output.extend(dict(row, split="dev") for row in ordered[-dev_count:])
    validate_three_way_class_coverage(output + test_rows)
    return output + test_rows
```

- [ ] **Step 4: Re-run the focused test and require deterministic passing output.**

```powershell
python -m unittest android.analysis.tests.test_build_schema5_recall_stability_folds
```

### 任务 2：加入仅开发集选择和梯度稳定控制 / Task 2: Add Dev-Only Selection And Gradient-Stability Controls

**Files:**
- Modify: `android/analysis/train_android_finger_four_class_cnn.py`
- Modify: `android/analysis/tests/test_train_android_finger_four_class_cnn.py`

- [ ] **Step 1: Add failing tests for CLI arguments, selection scoring, and loss identity.**

```python
def test_recall_balanced_selection_prioritizes_minimum_recall(self):
    stronger_recall = trainer.selection_score({"minRecall": 0.60, "balancedAccuracy": 0.70}, 0.70)
    stronger_accuracy = trainer.selection_score({"minRecall": 0.55, "balancedAccuracy": 0.90}, 0.70)
    self.assertGreater(stronger_recall, stronger_accuracy)

def test_focal_gamma_zero_matches_weighted_cross_entropy(self):
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    weights = torch.tensor([1.2, 0.8])
    self.assertTrue(torch.allclose(
        trainer.focal_cross_entropy(logits, labels, weights, gamma=0.0, torch=torch),
        torch.nn.functional.cross_entropy(logits, labels, weight=weights, reduction="none"),
    ))
```

- [ ] **Step 2: Run the focused trainer tests and confirm failure.**

```powershell
python -m unittest android.analysis.tests.test_train_android_finger_four_class_cnn
```

- [ ] **Step 3: Add the exact CLI contract while preserving existing defaults.**

```python
parser.add_argument("--dropout", type=float, default=0.10)
parser.add_argument("--loss-mode", choices=("weighted_ce", "weighted_focal"), default="weighted_ce")
parser.add_argument("--label-smoothing", type=float, default=0.0)
parser.add_argument("--focal-gamma", type=float, default=1.0)
parser.add_argument("--max-grad-norm", type=float, default=0.0)
parser.add_argument("--scheduler", choices=("none", "warmup_cosine"), default="none")
parser.add_argument("--warmup-epochs", type=int, default=0)
parser.add_argument("--min-learning-rate", type=float, default=0.0)
parser.add_argument("--selection-recall-weight", type=float, default=0.70)
parser.add_argument("--selection-min-delta", type=float, default=0.003)
parser.add_argument("--evaluate-test", action="store_true")
```

Validate dropout in `[0, 0.5)`, non-negative clipping/smoothing/gamma, valid warmup bounds, and prohibit `--evaluate-test` from influencing the checkpoint selection path.

- [ ] **Step 4: Add explicit train/dev/test tensors and dev-only model selection.**

```python
dev_metrics = evaluate(model, x_dev, y_dev, torch)
score = selection_score(dev_metrics, args.selection_recall_weight)
if score > best_score + args.selection_min_delta:
    best_score = score
    best_epoch = epoch
    best_state = clone_state(model, torch)

if args.evaluate_test:
    test_metrics = evaluate(model, x_test, y_test, torch)
else:
    test_metrics = None
```

The report must name the three splits exactly `trainMetrics`, `devMetrics`, and optional `testMetrics`; retain legacy `validMetrics` only as a read-compatible alias if existing tools require it.

- [ ] **Step 5: Add bounded-gradient training telemetry.**

```python
loss.backward()
if args.max_grad_norm > 0:
    pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    if not torch.isfinite(pre_clip_norm):
        raise RuntimeError("non_finite_gradient_norm")
    gradient_stats.add(float(pre_clip_norm), float(pre_clip_norm) > args.max_grad_norm)
optimizer.step()
```

Use a no-op norm measurement when clipping is disabled so every run has comparable telemetry. Add `mean`, `p95`, `max`, clipped-step count, total-step count, and non-finite count to `training_report.json`.

- [ ] **Step 6: Add loss factory and scheduler.**

```python
if args.loss_mode == "weighted_ce":
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing, reduction="none"
    )
else:
    loss_fn = lambda logits, labels: focal_cross_entropy(
        logits, labels, class_weights, args.focal_gamma, torch
    )

scheduler = make_scheduler(
    optimizer, args.scheduler, args.warmup_epochs, args.epochs, args.min_learning_rate, torch
)
```

For `warmup_cosine`, use three warmup epochs with a start factor of `0.1`, then cosine decay to `min_learning_rate`. Step it once per epoch after the optimizer updates; do not use a dev-driven plateau scheduler.

- [ ] **Step 7: Re-run the trainer test module.**

```powershell
python -m unittest android.analysis.tests.test_train_android_finger_four_class_cnn
```

### 任务 3：加入可恢复的仅开发集实验运行器 / Task 3: Add a Resumable, Dev-Only Experiment Runner

**Files:**
- Create: `android/analysis/run_android_recall_stability_ablation.py`
- Create: `android/analysis/tests/test_run_android_recall_stability_ablation.py`

- [ ] **Step 1: Write matrix and command tests.**

```python
def test_stage_a_has_45_development_jobs(self):
    jobs = runner.stage_a_jobs(Path("folds"), Path("models"))
    self.assertEqual(45, len(jobs))
    self.assertEqual({"A0_current", "A1_clip_only", "A2_stable_optimizer", "A3_moderate_regularization", "A4_strong_regularization"}, {job["profile"] for job in jobs})
    self.assertFalse(any("--evaluate-test" in runner.build_command(job, "python") for job in jobs))
```

- [ ] **Step 2: Run the focused runner test and confirm failure.**

```powershell
python -m unittest android.analysis.tests.test_run_android_recall_stability_ablation
```

- [ ] **Step 3: Implement Stage A and Stage B command builders.**

Each command must pass `--split-strategy predefined`, `--anchor-mode down`, `--require-press-active`, `--deterministic`, `--frame-indices=-1,0,1,2`, the profile arguments from the two tables above, and no `--evaluate-test`. Cache keys must include window, feature preset, and the three-way manifest signature, but not seed or profile.

- [ ] **Step 4: Implement durable development aggregation.**

The CSV schema must include profile, stage, fold, seed, best epoch, dev minimum recall, dev balanced accuracy, selection score, gradient telemetry, and artifact completion. Rank only this schema. A run is complete only if its report, weights, scaler, and catalog entry exist.

- [ ] **Step 5: Implement locked Stage C confirmation.**

Require `--selected-profile selected_profile.json`; verify that its selected stage is complete and that its hash matches the source aggregate. Only then append `--evaluate-test` and generate 15 confirmation jobs. Test metrics must be stored separately from development ranking output.

- [ ] **Step 6: Re-run focused runner tests.**

```powershell
python -m unittest android.analysis.tests.test_run_android_recall_stability_ablation
```

### 任务 4：文档化、验证并运行各阶段 / Task 4: Document, Verify, And Run The Stages

**Files:**
- Modify: `FunctionAndDataStructureDescription.md`
- Output: `models/bbsiak_schema5_recall_stability_<date>/`

- [ ] **Step 1: Update the technical description.**

Document all CLI controls, the train/dev/test boundary, telemetry fields, the Stage A/B/C matrices, and the restriction that no model is automatically copied into Android assets.

- [ ] **Step 2: Run all analysis tests and static compilation.**

```powershell
python -m unittest discover -s android/analysis/tests -p "test_*.py"
python -m py_compile android/analysis/train_android_finger_four_class_cnn.py android/analysis/build_schema5_recall_stability_folds.py android/analysis/run_android_recall_stability_ablation.py
git diff --check -- android/analysis FunctionAndDataStructureDescription.md
```

- [ ] **Step 3: Validate Stage A without training.**

```powershell
python android/analysis/run_android_recall_stability_ablation.py --stage a --dry-run
```

Expected: exactly 45 commands and no `--evaluate-test` flag.

- [ ] **Step 4: Run Stage A, select a profile only from development metrics, then run Stage B.**

```powershell
python android/analysis/run_android_recall_stability_ablation.py --stage a
python android/analysis/run_android_recall_stability_ablation.py --stage b --selected-profile models/bbsiak_schema5_recall_stability_<date>/selected_stage_a_profile.json
```

Expected: 45 Stage-A and 36 Stage-B development runs, each resumable; test metrics remain absent.

- [ ] **Step 5: Lock and run Stage C only after Stage-B selection.**

```powershell
python android/analysis/run_android_recall_stability_ablation.py --stage c --selected-profile models/bbsiak_schema5_recall_stability_<date>/selected_profile.json
```

Expected: 15 test-evaluation confirmations, separate test summary, no Android asset import.

## 最终验收标准 / Final Acceptance Criteria

1. No training-time component observes outer test rows before the selected profile is locked.
2. Every report identifies exact parameters, seed, split counts, best dev epoch, dev metrics, optional test metrics, and gradient telemetry.
3. No non-finite gradient is silently accepted; clip rates above the stated gate are reported as failures.
4. Profile selection prioritizes worst-class recall and cannot win by sacrificing mean balanced accuracy by more than one percentage point.
5. The full test suite, static compilation, and diff whitespace check pass before any experiment starts.
6. No model is deployed until an unseen-session, label-interleaved blind set has been evaluated once.
