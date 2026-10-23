#!/usr/bin/env python3
"""Compare compact classifiers on the fixed causal time-grid dataset."""

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch


CLASS_LABELS = ("thumb_light", "thumb_heavy", "index_light", "index_heavy")
FEATURE_NAMES = (
    "x_norm",
    "y_norm",
    "userAcceleration.x",
    "userAcceleration.y",
    "userAcceleration.z",
    "rotationRate.x",
    "rotationRate.y",
    "rotationRate.z",
    "gravity.x",
    "gravity.y",
    "gravity.z",
    "pressure",
    "size",
    "major_norm",
    "minor_norm",
    "log_axis_ratio",
    "orientation_sin",
    "orientation_cos",
    "pointer_count",
    "touch_valid",
    "pressure_available",
    "size_available",
    "major_available",
    "minor_available",
    "orientation_available",
    "pointer_count_available",
)
PHYSICAL_NAMES = (
    "x_norm",
    "y_norm",
    "userAcceleration.x",
    "userAcceleration.y",
    "userAcceleration.z",
    "rotationRate.x",
    "rotationRate.y",
    "rotationRate.z",
    "size",
    "log_axis_ratio",
)
SEQUENCE_SOURCE_NAMES = (
    "x_norm",
    "y_norm",
    "userAcceleration.x",
    "userAcceleration.y",
    "userAcceleration.z",
    "rotationRate.x",
    "rotationRate.y",
    "rotationRate.z",
    "gravity.x",
    "gravity.y",
    "gravity.z",
    "size",
    "log_axis_ratio",
)
PHYSICAL_INDICES = tuple(FEATURE_NAMES.index(name) for name in PHYSICAL_NAMES)


def configure_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def read_rows(dataset_dir):
    path = Path(dataset_dir) / "training_touch23.jsonl"
    if not path.is_file():
        path = Path(dataset_dir) / "training_touch26.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_folds(dataset_dir, rows):
    by_hash = {row["sourceSha256"]: row for row in rows}
    if len(by_hash) != len(rows):
        raise RuntimeError("duplicate_training_source_sha256")
    folds = {}
    path = Path(dataset_dir) / "strict_group_folds.csv"
    with path.open("r", encoding="utf-8", newline="") as source:
        for fold_row in csv.DictReader(source):
            fold = int(fold_row["fold"])
            row = by_hash[fold_row["sourceSha256"]]
            sample = {
                "sourceSha256": row["sourceSha256"],
                "sampleId": row["sampleId"],
                "classLabel": row["classLabel"],
                "label": CLASS_LABELS.index(row["classLabel"]),
                "groupId": row["groupId"],
                "matrix": np.asarray(row["matrix"], dtype=np.float32),
                "split": fold_row["split"],
            }
            folds.setdefault(fold, []).append(sample)
    result = {}
    for fold, samples in folds.items():
        train = [sample for sample in samples if sample["split"] == "train"]
        valid = [sample for sample in samples if sample["split"] == "valid"]
        train_groups = {sample["groupId"] for sample in train}
        valid_groups = {sample["groupId"] for sample in valid}
        if train_groups.intersection(valid_groups):
            raise RuntimeError("cross_split_group_leakage")
        if {sample["classLabel"] for sample in train} != set(CLASS_LABELS):
            raise RuntimeError("train_missing_class")
        if {sample["classLabel"] for sample in valid} != set(CLASS_LABELS):
            raise RuntimeError("valid_missing_class")
        result[fold] = (train, valid)
    return result


def select_matrix(samples, indices):
    return np.stack(
        [sample["matrix"][:, indices] for sample in samples], axis=0
    )


def fit_scaler(values):
    flat = values.reshape(-1, values.shape[-1])
    center = np.median(flat, axis=0)
    low = np.percentile(flat, 25, axis=0)
    high = np.percentile(flat, 75, axis=0)
    scale = high - low
    scale[scale < 1e-6] = 1.0
    return center.astype(np.float32), scale.astype(np.float32)


def transform(values, center, scale):
    return np.clip((values - center) / scale, -8.0, 8.0).astype(np.float32)


def sample_weights(samples):
    group_counts = Counter(sample["groupId"] for sample in samples)
    raw = np.asarray(
        [1.0 / math.sqrt(group_counts[sample["groupId"]]) for sample in samples],
        dtype=np.float32,
    )
    labels = np.asarray([sample["label"] for sample in samples])
    target = len(samples) / len(CLASS_LABELS)
    for label in range(len(CLASS_LABELS)):
        raw[labels == label] *= target / raw[labels == label].sum()
    return np.clip(raw, 0.25, 4.0) / raw.mean()


def stats_features(values):
    """Convert a short sequence into physically meaningful summaries."""
    mean = values.mean(axis=1)
    std = values.std(axis=1)
    minimum = values.min(axis=1)
    maximum = values.max(axis=1)
    last_delta = values[:, -1] - values[:, 0]
    acceleration = values[:, :, 2:5]
    rotation = values[:, :, 5:8]
    accel_magnitude = np.linalg.norm(acceleration, axis=2)
    rotation_magnitude = np.linalg.norm(rotation, axis=2)
    accel_jerk = np.linalg.norm(np.diff(acceleration, axis=1), axis=2)
    rotation_jerk = np.linalg.norm(np.diff(rotation, axis=1), axis=2)
    derived = np.stack(
        (
            accel_magnitude.mean(axis=1),
            accel_magnitude.std(axis=1),
            accel_magnitude.max(axis=1),
            accel_jerk.mean(axis=1),
            accel_jerk.max(axis=1),
            rotation_magnitude.mean(axis=1),
            rotation_magnitude.std(axis=1),
            rotation_magnitude.max(axis=1),
            rotation_jerk.mean(axis=1),
            rotation_jerk.max(axis=1),
        ),
        axis=1,
    )
    # 中文注释：统计特征合同按“通道内五项统计”交错排列，
    # 与 stats_mlp_contract 及 Android 运行时保持相同列序。
    channel_stats = np.stack(
        (mean, std, minimum, maximum, last_delta),
        axis=2,
    ).reshape(values.shape[0], -1)
    return np.concatenate((channel_stats, derived), axis=1).astype(np.float32)


class StatsMlp(torch.nn.Module):
    def __init__(
        self,
        input_count,
        hidden_count,
        multitask,
        context_class_counts=None,
    ):
        super().__init__()
        self.multitask = multitask
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(input_count, hidden_count),
            torch.nn.GELU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(hidden_count, hidden_count),
            torch.nn.GELU(),
        )
        if multitask:
            self.finger = torch.nn.Linear(hidden_count, 2)
            self.strength = torch.nn.Linear(hidden_count, 2)
            self.context = torch.nn.ModuleList(
                torch.nn.Linear(hidden_count, count)
                for count in (context_class_counts or ())
            )
        else:
            self.output = torch.nn.Linear(hidden_count, 4)

    def forward(self, values):
        encoded = self.shared(values)
        if self.multitask:
            finger = self.finger(encoded)
            strength = self.strength(encoded)
            if self.context:
                return finger, strength, tuple(
                    head(encoded) for head in self.context
                )
            return finger, strength
        return self.output(encoded)


def unpack_multitask_output(output):
    """Keep legacy two-head callers compatible with optional context heads."""
    if not isinstance(output, tuple) or len(output) < 2:
        raise RuntimeError("invalid_multitask_output")
    if len(output) == 2:
        return output[0], output[1], ()
    if len(output) != 3 or not isinstance(output[2], tuple):
        raise RuntimeError("invalid_multitask_output")
    return output[0], output[1], output[2]


class TinyDepthwiseCnn(torch.nn.Module):
    def __init__(self, input_count, hidden_count, multitask):
        super().__init__()
        self.multitask = multitask
        self.projection = torch.nn.Conv1d(
            input_count, hidden_count, kernel_size=1
        )
        self.depthwise = torch.nn.Conv1d(
            hidden_count,
            hidden_count,
            kernel_size=3,
            padding=1,
            groups=hidden_count,
        )
        self.pointwise = torch.nn.Conv1d(
            hidden_count, hidden_count, kernel_size=1
        )
        self.norm = torch.nn.GroupNorm(4, hidden_count)
        self.dropout = torch.nn.Dropout(0.15)
        if multitask:
            self.finger = torch.nn.Linear(hidden_count * 2, 2)
            self.strength = torch.nn.Linear(hidden_count * 2, 2)
        else:
            self.output = torch.nn.Linear(hidden_count * 2, 4)

    def forward(self, values):
        values = values.transpose(1, 2)
        values = torch.nn.functional.gelu(self.projection(values))
        values = self.depthwise(values)
        values = self.pointwise(values)
        values = torch.nn.functional.gelu(self.norm(values))
        pooled = torch.cat(
            (values.mean(dim=2), values.amax(dim=2)), dim=1
        )
        encoded = self.dropout(pooled)
        if self.multitask:
            return self.finger(encoded), self.strength(encoded)
        return self.output(encoded)


class TinyTcn(torch.nn.Module):
    def __init__(self, input_count, hidden_count, multitask):
        super().__init__()
        self.multitask = multitask
        self.projection = torch.nn.Conv1d(
            input_count, hidden_count, kernel_size=1
        )
        self.block1 = torch.nn.Conv1d(
            hidden_count,
            hidden_count,
            kernel_size=3,
            padding=1,
            groups=hidden_count,
        )
        self.block2 = torch.nn.Conv1d(
            hidden_count,
            hidden_count,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=hidden_count,
        )
        self.mix1 = torch.nn.Conv1d(hidden_count, hidden_count, kernel_size=1)
        self.mix2 = torch.nn.Conv1d(hidden_count, hidden_count, kernel_size=1)
        self.norm = torch.nn.GroupNorm(4, hidden_count)
        if multitask:
            self.finger = torch.nn.Linear(hidden_count * 2, 2)
            self.strength = torch.nn.Linear(hidden_count * 2, 2)
        else:
            self.output = torch.nn.Linear(hidden_count * 2, 4)

    def forward(self, values):
        values = values.transpose(1, 2)
        values = self.projection(values)
        residual = values
        values = self.mix1(torch.nn.functional.gelu(self.block1(values)))
        values = self.mix2(torch.nn.functional.gelu(self.block2(values)))
        values = torch.nn.functional.gelu(self.norm(values + residual))
        pooled = torch.cat(
            (values.mean(dim=2), values.amax(dim=2)), dim=1
        )
        if self.multitask:
            return self.finger(pooled), self.strength(pooled)
        return self.output(pooled)


def logits_for_four_class(model, values):
    output = model(values)
    if not model.multitask:
        return output
    finger, strength, _ = unpack_multitask_output(output)
    return (
        torch.nn.functional.log_softmax(finger, dim=1)[:, :, None]
        + torch.nn.functional.log_softmax(strength, dim=1)[:, None, :]
    ).reshape(-1, 4)


def four_class_losses(model, values, targets, reduction="none"):
    """Return the correct joint loss and auxiliary logits in one forward."""
    output = model(values)
    if not model.multitask:
        return (
            torch.nn.functional.cross_entropy(
                output,
                targets,
                reduction=reduction,
            ),
            None,
            None,
        )
    finger_logits, strength_logits, _ = unpack_multitask_output(output)
    joint_log_probs = (
        torch.nn.functional.log_softmax(finger_logits, dim=1)[:, :, None]
        + torch.nn.functional.log_softmax(strength_logits, dim=1)[:, None, :]
    ).reshape(-1, 4)
    return (
        torch.nn.functional.nll_loss(
            joint_log_probs,
            targets,
            reduction=reduction,
        ),
        finger_logits,
        strength_logits,
    )


def predict(model, values, batch_size=1024):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            logits = logits_for_four_class(model, values[start : start + batch_size])
            predictions.extend(logits.argmax(dim=1).tolist())
    return predictions


def metrics(truths, predictions):
    confusion = [[0 for _ in CLASS_LABELS] for _ in CLASS_LABELS]
    for truth, prediction in zip(truths, predictions):
        confusion[int(truth)][int(prediction)] += 1
    recalls = []
    for index in range(len(CLASS_LABELS)):
        total = sum(confusion[index])
        recalls.append(confusion[index][index] / total if total else 0.0)
    correct = sum(confusion[index][index] for index in range(4))
    return {
        "accuracy": correct / len(truths),
        "balancedAccuracy": sum(recalls) / 4.0,
        "recallByClass": dict(zip(CLASS_LABELS, recalls)),
        "confusionMatrix": confusion,
    }


def train_fold(
    train_samples,
    valid_samples,
    model_kind,
    feature_kind,
    hidden_count,
    multitask,
    seed,
):
    configure_determinism(seed)
    if feature_kind == "physical":
        indices = PHYSICAL_INDICES
        source_names = PHYSICAL_NAMES
    else:
        raise RuntimeError("unsupported_feature_kind")
    train_raw = select_matrix(train_samples, indices)
    valid_raw = select_matrix(valid_samples, indices)
    center, scale = fit_scaler(train_raw)
    train_normalized = transform(train_raw, center, scale)
    valid_normalized = transform(valid_raw, center, scale)
    if model_kind == "stats":
        train_values = stats_features(train_normalized)
        valid_values = stats_features(valid_normalized)
    else:
        train_values = train_normalized
        valid_values = valid_normalized
    stats_center = train_values.mean(axis=0)
    stats_scale = train_values.std(axis=0)
    stats_scale[stats_scale < 1e-6] = 1.0
    train_values = np.clip(
        (train_values - stats_center) / stats_scale, -8.0, 8.0
    ).astype(np.float32)
    valid_values = np.clip(
        (valid_values - stats_center) / stats_scale, -8.0, 8.0
    ).astype(np.float32)
    x_train = torch.from_numpy(train_values)
    x_valid = torch.from_numpy(valid_values)
    y_train = torch.tensor(
        [sample["label"] for sample in train_samples], dtype=torch.long
    )
    y_valid = torch.tensor(
        [sample["label"] for sample in valid_samples], dtype=torch.long
    )
    weights = torch.from_numpy(sample_weights(train_samples))
    if model_kind == "stats":
        model = StatsMlp(train_values.shape[1], hidden_count, multitask)
    elif model_kind == "depthwise":
        model = TinyDepthwiseCnn(train_values.shape[2], hidden_count, multitask)
    elif model_kind == "tcn":
        model = TinyTcn(train_values.shape[2], hidden_count, multitask)
    else:
        raise RuntimeError("unsupported_model_kind")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.003)
    epochs = 55
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.00005
    )
    best_state = None
    best_score = -1.0
    best_epoch = 0
    patience = 10
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), 128):
            batch = order[start : start + 128]
            losses, finger_logits, strength_logits = four_class_losses(
                model,
                x_train[batch],
                y_train[batch],
            )
            loss = (losses * weights[batch]).sum() / weights[batch].sum()
            if multitask:
                loss += 0.25 * (
                    torch.nn.functional.cross_entropy(
                        finger_logits, y_train[batch] // 2
                    )
                    + torch.nn.functional.cross_entropy(
                        strength_logits, y_train[batch] % 2
                    )
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        valid_metrics = metrics(y_valid.tolist(), predict(model, x_valid))
        if valid_metrics["balancedAccuracy"] > best_score + 1e-12:
            best_score = valid_metrics["balancedAccuracy"]
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    train_metrics = metrics(y_train.tolist(), predict(model, x_train))
    valid_metrics = metrics(y_valid.tolist(), predict(model, x_valid))
    return {
        "modelKind": model_kind,
        "featureKind": feature_kind,
        "hiddenCount": hidden_count,
        "multitask": multitask,
        "parameterCount": sum(parameter.numel() for parameter in model.parameters()),
        "inputCount": train_values.shape[1],
        "sequenceChannelCount": (
            train_values.shape[2] if model_kind != "stats" else 0
        ),
        "sequenceChannelNames": list(source_names),
        "bestEpoch": best_epoch,
        "epochsCompleted": epoch,
        "trainMetrics": train_metrics,
        "validMetrics": valid_metrics,
        "generalizationGap": (
            train_metrics["balancedAccuracy"] - valid_metrics["balancedAccuracy"]
        ),
    }


def aggregate(reports):
    valid = [report["validMetrics"] for report in reports]
    recalls = [
        value for metrics in valid for value in metrics["recallByClass"].values()
    ]
    return {
        "meanBalancedAccuracy": float(
            np.mean([metrics["balancedAccuracy"] for metrics in valid])
        ),
        "worstFoldBalancedAccuracy": float(
            np.min([metrics["balancedAccuracy"] for metrics in valid])
        ),
        "meanAccuracy": float(np.mean([metrics["accuracy"] for metrics in valid])),
        "worstClassRecall": float(np.min(recalls)),
        "meanGeneralizationGap": float(
            np.mean([report["generalizationGap"] for report in reports])
        ),
        "foldBalancedAccuracies": [
            metrics["balancedAccuracy"] for metrics in valid
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_rows(args.dataset_dir)
    folds = load_folds(args.dataset_dir, rows)
    specs = (
        ("stats", "physical", 16, False),
        ("stats", "physical", 24, True),
        ("depthwise", "physical", 16, False),
        ("depthwise", "physical", 20, True),
        ("tcn", "physical", 12, False),
        ("tcn", "physical", 16, True),
    )
    output = {"scope": "single_device_cross_session", "models": {}}
    for kind, feature_kind, hidden, multitask in specs:
        name = f"{kind}_{hidden}_{'multitask' if multitask else 'single'}"
        reports = []
        for fold in sorted(folds):
            train_samples, valid_samples = folds[fold]
            print(json.dumps({"model": name, "fold": fold}), flush=True)
            reports.append(
                train_fold(
                    train_samples,
                    valid_samples,
                    kind,
                    feature_kind,
                    hidden,
                    multitask,
                    20261022 + fold,
                )
            )
        output["models"][name] = {
            "aggregate": aggregate(reports),
            "folds": reports,
        }
        print(
            json.dumps(
                {"model": name, **output["models"][name]["aggregate"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
    output["deploymentAllowed"] = False
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
