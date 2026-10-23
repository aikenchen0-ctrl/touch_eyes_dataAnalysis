#!/usr/bin/env python3
"""Measure which physical channel groups support lightweight classification."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from . import run_lightweight_time_grid_models as lightweight
except ImportError:
    import run_lightweight_time_grid_models as lightweight


FEATURE_VARIANTS = {
    "imu8": (
        "userAcceleration.x",
        "userAcceleration.y",
        "userAcceleration.z",
        "rotationRate.x",
        "rotationRate.y",
        "rotationRate.z",
    ),
    "touch2": ("size", "log_axis_ratio"),
    "touch_size": ("size",),
    "touch_ratio": ("log_axis_ratio",),
    "xy": ("x_norm", "y_norm"),
    "xy_touch2": ("x_norm", "y_norm", "size", "log_axis_ratio"),
    "physical10": (
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
    ),
    "physical10_gravity": (
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
    ),
    "imu8_touch_size": (
        "userAcceleration.x",
        "userAcceleration.y",
        "userAcceleration.z",
        "rotationRate.x",
        "rotationRate.y",
        "rotationRate.z",
        "size",
    ),
    "imu8_touch_ratio": (
        "userAcceleration.x",
        "userAcceleration.y",
        "userAcceleration.z",
        "rotationRate.x",
        "rotationRate.y",
        "rotationRate.z",
        "log_axis_ratio",
    ),
}


def subset_values(values, source_names, target_names):
    indices = [source_names.index(name) for name in target_names]
    return values[:, :, indices]


def normalized_stats(train_raw, valid_raw, seed):
    center, scale = lightweight.fit_scaler(train_raw)
    train_normalized = lightweight.transform(train_raw, center, scale)
    valid_normalized = lightweight.transform(valid_raw, center, scale)
    train_stats = lightweight.stats_features(train_normalized)
    valid_stats = lightweight.stats_features(valid_normalized)
    stats_center = train_stats.mean(axis=0)
    stats_scale = train_stats.std(axis=0)
    stats_scale[stats_scale < 1e-6] = 1.0
    train_stats = np.clip(
        (train_stats - stats_center) / stats_scale, -8.0, 8.0
    ).astype(np.float32)
    valid_stats = np.clip(
        (valid_stats - stats_center) / stats_scale, -8.0, 8.0
    ).astype(np.float32)
    return train_stats, valid_stats


def train_variant(
    train_samples,
    valid_samples,
    source_names,
    target_names,
    hidden_count,
    multitask,
    seed,
):
    lightweight.configure_determinism(seed)
    train_indices = [source_names.index(name) for name in target_names]
    valid_indices = train_indices
    train_raw = lightweight.select_matrix(train_samples, train_indices)
    valid_raw = lightweight.select_matrix(valid_samples, valid_indices)
    train_stats, valid_stats = normalized_stats(train_raw, valid_raw, seed)
    x_train = torch.from_numpy(train_stats)
    x_valid = torch.from_numpy(valid_stats)
    y_train = torch.tensor(
        [sample["label"] for sample in train_samples], dtype=torch.long
    )
    y_valid = torch.tensor(
        [sample["label"] for sample in valid_samples], dtype=torch.long
    )
    weights = torch.from_numpy(lightweight.sample_weights(train_samples))
    model = lightweight.StatsMlp(
        train_stats.shape[1], hidden_count, multitask
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.002, weight_decay=0.003
    )
    epochs = 55
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0.00005
    )
    best_state = None
    best_score = -1.0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), 128):
            batch = order[start : start + 128]
            losses, finger_logits, strength_logits = (
                lightweight.four_class_losses(
                    model,
                    x_train[batch],
                    y_train[batch],
                )
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
        valid_metrics = lightweight.metrics(
            y_valid.tolist(), lightweight.predict(model, x_valid)
        )
        if valid_metrics["balancedAccuracy"] > best_score + 1e-12:
            best_score = valid_metrics["balancedAccuracy"]
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
        if epoch - best_epoch >= 10:
            break
    model.load_state_dict(best_state)
    train_metrics = lightweight.metrics(
        y_train.tolist(), lightweight.predict(model, x_train)
    )
    valid_metrics = lightweight.metrics(
        y_valid.tolist(), lightweight.predict(model, x_valid)
    )
    return {
        "featureNames": list(target_names),
        "inputCount": train_stats.shape[1],
        "parameterCount": sum(p.numel() for p in model.parameters()),
        "bestEpoch": best_epoch,
        "trainMetrics": train_metrics,
        "validMetrics": valid_metrics,
        "generalizationGap": (
            train_metrics["balancedAccuracy"]
            - valid_metrics["balancedAccuracy"]
        ),
    }


def aggregate(reports):
    scores = [
        report["validMetrics"]["balancedAccuracy"] for report in reports
    ]
    recalls = [
        value
        for report in reports
        for value in report["validMetrics"]["recallByClass"].values()
    ]
    return {
        "meanBalancedAccuracy": float(np.mean(scores)),
        "worstFoldBalancedAccuracy": float(np.min(scores)),
        "worstClassRecall": float(np.min(recalls)),
        "meanGeneralizationGap": float(
            np.mean([report["generalizationGap"] for report in reports])
        ),
        "foldBalancedAccuracies": scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = lightweight.read_rows(args.dataset_dir)
    folds = lightweight.load_folds(args.dataset_dir, rows)
    output = {
        "scope": "single_device_cross_session",
        "model": "stats_mlp16_single",
        "variants": {},
    }
    for variant, names in FEATURE_VARIANTS.items():
        reports = []
        for fold in sorted(folds):
            train_samples, valid_samples = folds[fold]
            reports.append(
                train_variant(
                    train_samples,
                    valid_samples,
                    lightweight.SEQUENCE_SOURCE_NAMES,
                    names,
                    16,
                    False,
                    20261022 + fold,
                )
            )
        output["variants"][variant] = {
            "aggregate": aggregate(reports),
            "folds": reports,
        }
        print(
            json.dumps(
                {
                    "variant": variant,
                    **output["variants"][variant]["aggregate"],
                }
            ),
            flush=True,
        )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
