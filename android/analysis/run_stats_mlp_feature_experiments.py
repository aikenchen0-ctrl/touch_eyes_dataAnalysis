#!/usr/bin/env python3
"""Compare touch-channel validity windows for the statistics MLP candidate."""

import argparse
import collections
import math
import json
from pathlib import Path

import numpy as np
import torch

try:
    from . import run_lightweight_time_grid_models as lightweight
    from . import stats_mlp_contract as contract
except ImportError:
    import run_lightweight_time_grid_models as lightweight
    import stats_mlp_contract as contract


IMU_CHANNEL_COUNT = 8


def fit_scaler_windowed(values, touch_start):
    """Fit touch channels only on frames where touch data can be valid."""
    center = np.zeros(values.shape[2], dtype=np.float32)
    scale = np.ones(values.shape[2], dtype=np.float32)
    for channel in range(values.shape[2]):
        frames = values[:, :, channel]
        if channel >= IMU_CHANNEL_COUNT:
            frames = frames[:, touch_start:]
        flat = frames.reshape(-1)
        center[channel] = np.median(flat)
        low, high = np.percentile(flat, (25, 75))
        delta = high - low
        scale[channel] = delta if delta >= 1e-6 else 1.0
    return center, scale


def stats_features_windowed(values, touch_start):
    """Use complete IMU history and only valid touch frames for touch shape."""
    channel_rows = []
    for channel in range(values.shape[2]):
        frames = values[:, :, channel]
        if channel >= IMU_CHANNEL_COUNT:
            frames = frames[:, touch_start:]
        channel_rows.append(np.stack((
            frames.mean(axis=1),
            frames.std(axis=1),
            frames.min(axis=1),
            frames.max(axis=1),
            frames[:, -1] - frames[:, 0],
        ), axis=1))
    channel_stats = np.concatenate(channel_rows, axis=1)
    acceleration = values[:, :, 2:5]
    rotation = values[:, :, 5:8]
    accel_magnitude = np.linalg.norm(acceleration, axis=2)
    rotation_magnitude = np.linalg.norm(rotation, axis=2)
    accel_jerk = np.linalg.norm(np.diff(acceleration, axis=1), axis=2)
    rotation_jerk = np.linalg.norm(np.diff(rotation, axis=1), axis=2)
    derived = np.stack((
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
    ), axis=1)
    result = np.concatenate((channel_stats, derived), axis=1).astype(
        np.float32
    )
    if result.shape[1] != contract.STATS_FEATURE_COUNT:
        raise RuntimeError("unexpected_stats_feature_count")
    return result


def sample_weights(samples, mode):
    """Build session-aware weights without letting large sessions dominate."""
    group_counts = collections.Counter(
        sample["groupId"] for sample in samples
    )
    labels = np.asarray([sample["label"] for sample in samples])
    if mode == "current":
        raw = np.asarray([
            1.0 / math.sqrt(group_counts[sample["groupId"]])
            for sample in samples
        ], dtype=np.float32)
    elif mode == "group_equal":
        raw = np.asarray([
            1.0 / group_counts[sample["groupId"]]
            for sample in samples
        ], dtype=np.float32)
    elif mode == "class_equal":
        raw = np.ones(len(samples), dtype=np.float32)
    elif mode == "session_class_equal":
        class_group_counts = collections.Counter(
            sample["classLabel"] for sample in samples
        )
        raw = np.asarray([
            1.0 / (
                group_counts[sample["groupId"]]
                * class_group_counts[sample["classLabel"]]
            )
            for sample in samples
        ], dtype=np.float32)
    else:
        raise RuntimeError("unsupported_weight_mode")
    target = len(samples) / len(contract.CLASS_LABELS)
    for label in range(len(contract.CLASS_LABELS)):
        raw[labels == label] *= target / raw[labels == label].sum()
    return np.clip(raw, 0.25, 4.0) / raw.mean()


def apply_feature_mode(values, mode, touch_start):
    """Remove channels that encode page position rather than touch dynamics."""
    result = values.copy()
    x_start = 0
    y_start = 5
    if mode == "no_absolute_xy":
        result[:, x_start] = 0.0
        result[:, y_start] = 0.0
    elif mode == "delta_xy":
        for start in (x_start, y_start):
            original = result[:, start:start + 5].copy()
            result[:, start] = 0.0
            if touch_start < 8:
                result[:, start + 1:start + 5] = np.stack((
                    original[:, 1] - original[:, 0],
                    original[:, 2] - original[:, 1],
                    original[:, 3] - original[:, 2],
                    original[:, 4] - original[:, 3],
                ), axis=1)
    elif mode == "relative_xy":
        for start in (x_start, y_start):
            anchor = result[:, start + touch_start]
            result[:, start:start + 5] -= anchor[:, None]
    elif mode != "all":
        raise RuntimeError("unsupported_feature_mode")
    return result


def prepare_fold(train_samples, valid_samples, touch_start):
    train_raw = lightweight.select_matrix(
        train_samples,
        lightweight.PHYSICAL_INDICES,
    )
    valid_raw = lightweight.select_matrix(
        valid_samples,
        lightweight.PHYSICAL_INDICES,
    )
    center, scale = fit_scaler_windowed(train_raw, touch_start)
    train_normalized = lightweight.transform(train_raw, center, scale)
    valid_normalized = lightweight.transform(valid_raw, center, scale)
    train_stats = stats_features_windowed(train_normalized, touch_start)
    valid_stats = stats_features_windowed(valid_normalized, touch_start)
    stats_center = train_stats.mean(axis=0)
    stats_scale = train_stats.std(axis=0)
    stats_scale[stats_scale < 1e-6] = 1.0
    train_stats = np.clip(
        (train_stats - stats_center) / stats_scale,
        contract.STATS_CLIP_MIN,
        contract.STATS_CLIP_MAX,
    ).astype(np.float32)
    valid_stats = np.clip(
        (valid_stats - stats_center) / stats_scale,
        contract.STATS_CLIP_MIN,
        contract.STATS_CLIP_MAX,
    ).astype(np.float32)
    return train_stats, valid_stats


def train_variant(
    train_samples,
    valid_samples,
    touch_start,
    weight_mode,
    feature_mode,
    seed,
):
    lightweight.configure_determinism(seed)
    train_values, valid_values = prepare_fold(
        train_samples,
        valid_samples,
        touch_start,
    )
    train_values = apply_feature_mode(train_values, feature_mode, touch_start)
    valid_values = apply_feature_mode(valid_values, feature_mode, touch_start)
    x_train = torch.from_numpy(train_values)
    x_valid = torch.from_numpy(valid_values)
    y_train = torch.tensor(
        [sample["label"] for sample in train_samples],
        dtype=torch.long,
    )
    y_valid = torch.tensor(
        [sample["label"] for sample in valid_samples],
        dtype=torch.long,
    )
    weights = torch.from_numpy(sample_weights(train_samples, weight_mode))
    model = lightweight.StatsMlp(
        contract.STATS_FEATURE_COUNT,
        24,
        True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=0.003,
    )
    epochs = 55
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=0.00005,
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
            batch = order[start:start + 128]
            losses, finger_logits, strength_logits = (
                lightweight.four_class_losses(
                    model,
                    x_train[batch],
                    y_train[batch],
                )
            )
            loss = (losses * weights[batch]).sum() / weights[batch].sum()
            loss += 0.25 * (
                torch.nn.functional.cross_entropy(
                    finger_logits,
                    y_train[batch] // 2,
                )
                + torch.nn.functional.cross_entropy(
                    strength_logits,
                    y_train[batch] % 2,
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        valid_metrics = lightweight.metrics(
            y_valid.tolist(),
            lightweight.predict(model, x_valid),
        )
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
    train_metrics = lightweight.metrics(
        y_train.tolist(),
        lightweight.predict(model, x_train),
    )
    valid_metrics = lightweight.metrics(
        y_valid.tolist(),
        lightweight.predict(model, x_valid),
    )
    return {
        "touchStartFrame": touch_start,
        "weightMode": weight_mode,
        "featureMode": feature_mode,
        "bestEpoch": best_epoch,
        "epochsCompleted": epoch,
        "parameterCount": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainMetrics": train_metrics,
        "validMetrics": valid_metrics,
        "generalizationGap": (
            train_metrics["balancedAccuracy"]
            - valid_metrics["balancedAccuracy"]
        ),
    }


def aggregate(reports):
    valid = [report["validMetrics"] for report in reports]
    recalls = [
        value for metrics in valid
        for value in metrics["recallByClass"].values()
    ]
    return {
        "meanBalancedAccuracy": float(np.mean([
            metrics["balancedAccuracy"] for metrics in valid
        ])),
        "worstFoldBalancedAccuracy": float(np.min([
            metrics["balancedAccuracy"] for metrics in valid
        ])),
        "meanAccuracy": float(np.mean([
            metrics["accuracy"] for metrics in valid
        ])),
        "worstClassRecall": float(np.min(recalls)),
        "meanGeneralizationGap": float(np.mean([
            report["generalizationGap"] for report in reports
        ])),
        "foldBalancedAccuracies": [
            metrics["balancedAccuracy"] for metrics in valid
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = lightweight.read_rows(args.dataset_dir)
    folds = lightweight.load_folds(args.dataset_dir, rows)
    variants = {
        "window7_all": (7, "current", "all"),
        "window7_no_absolute_xy": (7, "current", "no_absolute_xy"),
        "window7_delta_xy": (7, "current", "delta_xy"),
        "window7_relative_xy": (7, "current", "relative_xy"),
    }
    output = {
        "scope": "single_device_cross_session",
        "datasetDir": str(Path(args.dataset_dir).resolve()),
        "selectionMetric": "worst_fold_balanced_accuracy",
        "experiments": {},
    }
    for name, (touch_start, weight_mode, feature_mode) in variants.items():
        reports = []
        for fold, (train_samples, valid_samples) in sorted(folds.items()):
            print(json.dumps({
                "variant": name,
                "fold": fold,
            }), flush=True)
            report = train_variant(
                train_samples,
                valid_samples,
                touch_start,
                weight_mode,
                feature_mode,
                20261022 + fold,
            )
            reports.append(report)
        output["experiments"][name] = {
            "aggregate": aggregate(reports),
            "folds": reports,
        }
    output_path = Path(args.output_dir) / "report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
