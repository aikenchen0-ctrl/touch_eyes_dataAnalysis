#!/usr/bin/env python3
"""Compare compact, runtime-friendly feature families for the stats MLP."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

try:
    from . import run_lightweight_time_grid_models as lightweight
    from . import stats_mlp_contract as contract
except ImportError:
    import run_lightweight_time_grid_models as lightweight
    import stats_mlp_contract as contract


TOUCH_START = 7
BASE_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 12, 15)
CONTACT_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15)
MOTION_STATES = ("handheld_static", "placed_static", "walking")
TOUCH_SOURCE_INDICES = (12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25)


def load_samples(dataset_dir):
    root = Path(dataset_dir)
    rows = lightweight.read_rows(root)
    by_hash = {row["sourceSha256"]: row for row in rows}
    result = {}
    with (root / "strict_group_folds.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as source:
        for item in csv.DictReader(source):
            fold = int(item["fold"])
            row = by_hash[item["sourceSha256"]]
            sample = {
                "sourceSha256": row["sourceSha256"],
                "sampleId": row["sampleId"],
                "classLabel": row["classLabel"],
                "label": lightweight.CLASS_LABELS.index(row["classLabel"]),
                "groupId": row["groupId"],
                "motionState": row.get("motionState", ""),
                "matrix": np.asarray(row["matrix"], dtype=np.float32),
                "split": item["split"],
            }
            result.setdefault(fold, []).append(sample)
    folds = {}
    for fold, samples in result.items():
        train = [sample for sample in samples if sample["split"] == "train"]
        valid = [sample for sample in samples if sample["split"] == "valid"]
        train_groups = {sample["groupId"] for sample in train}
        valid_groups = {sample["groupId"] for sample in valid}
        if train_groups.intersection(valid_groups):
            raise RuntimeError("cross_split_group_leakage")
        if {sample["classLabel"] for sample in train} != set(
            contract.CLASS_LABELS
        ):
            raise RuntimeError("train_missing_class")
        if {sample["classLabel"] for sample in valid} != set(
            contract.CLASS_LABELS
        ):
            raise RuntimeError("valid_missing_class")
        folds[fold] = (train, valid)
    return folds


def fit_scaler(sequences):
    center = np.zeros(sequences.shape[2], dtype=np.float32)
    scale = np.ones(sequences.shape[2], dtype=np.float32)
    for channel in range(sequences.shape[2]):
        values = sequences[:, :, channel]
        if channel >= 8:
            values = values[:, TOUCH_START:]
        flat = values.reshape(-1)
        center[channel] = np.median(flat)
        low, high = np.percentile(flat, (25, 75))
        delta = high - low
        scale[channel] = delta if delta >= 1e-6 else 1.0
    return center, scale


def stats_for_sequence(values, mask=None, with_derived=True):
    rows = []
    for channel in range(values.shape[2]):
        current = values[:, :, channel]
        if mask is not None and channel >= 8:
            current = current * mask
            count = mask.sum(axis=1)
            safe_count = np.maximum(count, 1)
            mean = current.sum(axis=1) / safe_count
            variance = ((current - mean[:, None]) ** 2 * mask).sum(axis=1)
            std = np.sqrt(variance / safe_count)
            minimum = np.where(mask > 0, current, np.inf).min(axis=1)
            maximum = np.where(mask > 0, current, -np.inf).max(axis=1)
            minimum = np.where(count > 0, minimum, 0.0)
            maximum = np.where(count > 0, maximum, 0.0)
            first = np.argmax(mask, axis=1)
            last = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
            first_value = current[np.arange(len(current)), first]
            last_value = current[np.arange(len(current)), last]
            last_delta = np.where(count > 0, last_value - first_value, 0.0)
        else:
            mean = current.mean(axis=1)
            std = current.std(axis=1)
            minimum = current.min(axis=1)
            maximum = current.max(axis=1)
            last_delta = current[:, -1] - current[:, 0]
        rows.append(np.stack((mean, std, minimum, maximum, last_delta), axis=1))
    output = np.concatenate(rows, axis=1)
    if with_derived:
        acceleration = values[:, :, 2:5]
        rotation = values[:, :, 5:8]
        acceleration_magnitude = np.linalg.norm(acceleration, axis=2)
        rotation_magnitude = np.linalg.norm(rotation, axis=2)
        acceleration_jerk = np.linalg.norm(
            np.diff(acceleration, axis=1),
            axis=2,
        )
        rotation_jerk = np.linalg.norm(
            np.diff(rotation, axis=1),
            axis=2,
        )
        output = np.concatenate((output, np.stack((
            acceleration_magnitude.mean(axis=1),
            acceleration_magnitude.std(axis=1),
            acceleration_magnitude.max(axis=1),
            acceleration_jerk.mean(axis=1),
            acceleration_jerk.max(axis=1),
            rotation_magnitude.mean(axis=1),
            rotation_magnitude.std(axis=1),
            rotation_magnitude.max(axis=1),
            rotation_jerk.mean(axis=1),
            rotation_jerk.max(axis=1),
        ), axis=1)), axis=1)
    return output.astype(np.float32)


def gravity_features(values, gravity):
    gravity_norm = np.linalg.norm(gravity, axis=2, keepdims=True)
    gravity_norm[gravity_norm < 1e-6] = 1.0
    unit_gravity = gravity / gravity_norm
    acceleration = values[:, :, 2:5]
    rotation = values[:, :, 5:8]
    acceleration_parallel = np.sum(acceleration * unit_gravity, axis=2)
    acceleration_perpendicular = np.linalg.norm(
        acceleration - acceleration_parallel[:, :, None] * unit_gravity,
        axis=2,
    )
    rotation_parallel = np.sum(rotation * unit_gravity, axis=2)
    rotation_perpendicular = np.linalg.norm(
        rotation - rotation_parallel[:, :, None] * unit_gravity,
        axis=2,
    )
    features = []
    for current in (
        acceleration_parallel,
        acceleration_perpendicular,
        rotation_parallel,
        rotation_perpendicular,
    ):
        features.extend((
            current.mean(axis=1),
            current.std(axis=1),
            current.max(axis=1),
        ))
    return np.stack(features, axis=1).astype(np.float32)


def prepare(samples, feature_spec):
    indices = CONTACT_INDICES if "contact" in feature_spec else BASE_INDICES
    raw = np.stack([sample["matrix"][:, indices] for sample in samples])
    # 中文注释：重力只用于坐标对齐，不直接作为姿态身份特征进入分类器。
    gravity_raw = np.stack([sample["matrix"][:, 8:11] for sample in samples])
    return raw, gravity_raw


def build_features(
    raw,
    gravity_raw,
    mask,
    center,
    scale,
    feature_spec,
    motion_values,
):
    normalized = lightweight.transform(raw, center, scale)
    gravity = gravity_raw.copy()
    gravity = gravity / np.maximum(
        np.median(np.abs(gravity), axis=(0, 1), keepdims=True),
        1e-6,
    )
    if "imu_delta" in feature_spec:
        baseline = normalized[:, :4, 2:8].mean(axis=1, keepdims=True)
        normalized[:, :, 2:8] -= baseline
    if "gravity" in feature_spec:
        normalized[:, :, 2:8] = np.where(
            mask[:, :, None] > 0,
            normalized[:, :, 2:8],
            normalized[:, :, 2:8],
        )
    stats = stats_for_sequence(
        normalized,
        mask=mask if "validity" in feature_spec else None,
        with_derived=True,
    )
    if "gravity" in feature_spec:
        stats = np.concatenate(
            (stats, gravity_features(normalized, gravity)),
            axis=1,
        )
    if "validity" in feature_spec:
        stats = np.concatenate((
            stats,
            mask.sum(axis=1)[:, None],
            np.argmax(mask, axis=1)[:, None],
        ), axis=1)
    if "motion" in feature_spec:
        motion = np.zeros((len(raw), len(MOTION_STATES)), dtype=np.float32)
        for index, state in enumerate(motion_values):
            if state in MOTION_STATES:
                motion[index, MOTION_STATES.index(state)] = 1.0
        stats = np.concatenate((stats, motion), axis=1)
    return stats


def weights(samples):
    return lightweight.sample_weights(samples)


def train_variant(train_samples, valid_samples, feature_spec, seed):
    lightweight.configure_determinism(seed)
    train_raw, train_gravity = prepare(train_samples, feature_spec)
    valid_raw, valid_gravity = prepare(valid_samples, feature_spec)
    center, scale = fit_scaler(train_raw)
    train_mask = np.stack([
        sample["matrix"][:, 19] > 0.5 for sample in train_samples
    ])
    valid_mask = np.stack([
        sample["matrix"][:, 19] > 0.5 for sample in valid_samples
    ])
    train_features = build_features(
        train_raw,
        train_gravity,
        train_mask,
        center,
        scale,
        feature_spec,
        [sample["motionState"] for sample in train_samples],
    )
    valid_features = build_features(
        valid_raw,
        valid_gravity,
        valid_mask,
        center,
        scale,
        feature_spec,
        [sample["motionState"] for sample in valid_samples],
    )
    feature_center = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0)
    feature_scale[feature_scale < 1e-6] = 1.0
    train_features = np.clip(
        (train_features - feature_center) / feature_scale,
        -8.0,
        8.0,
    ).astype(np.float32)
    valid_features = np.clip(
        (valid_features - feature_center) / feature_scale,
        -8.0,
        8.0,
    ).astype(np.float32)
    x_train = torch.from_numpy(train_features)
    x_valid = torch.from_numpy(valid_features)
    y_train = torch.tensor(
        [sample["label"] for sample in train_samples],
        dtype=torch.long,
    )
    y_valid = torch.tensor(
        [sample["label"] for sample in valid_samples],
        dtype=torch.long,
    )
    train_weights = torch.from_numpy(weights(train_samples))
    model = lightweight.StatsMlp(train_features.shape[1], 24, True)
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
            loss = (losses * train_weights[batch]).sum()
            loss = loss / train_weights[batch].sum()
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
        "featureSpec": sorted(feature_spec),
        "featureCount": train_features.shape[1],
        "parameterCount": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "bestEpoch": best_epoch,
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
    folds = load_samples(args.dataset_dir)
    specs = {
        "base": frozenset(),
        "contact": frozenset(("contact",)),
        "validity": frozenset(("validity",)),
        "imu_delta": frozenset(("imu_delta",)),
        "gravity": frozenset(("gravity",)),
        "motion": frozenset(("motion",)),
        "contact_validity_motion": frozenset((
            "contact",
            "validity",
            "motion",
        )),
    }
    output = {
        "scope": "single_device_cross_session",
        "touchStartFrame": TOUCH_START,
        "selectionMetric": "worst_fold_balanced_accuracy",
        "experiments": {},
    }
    for name, feature_spec in specs.items():
        reports = []
        for fold, (train_samples, valid_samples) in sorted(folds.items()):
            print(json.dumps({
                "variant": name,
                "fold": fold,
            }), flush=True)
            reports.append(train_variant(
                train_samples,
                valid_samples,
                feature_spec,
                20261022 + fold,
            ))
        output["experiments"][name] = {
            "aggregate": aggregate(reports),
            "folds": reports,
        }
    path = Path(args.output_dir) / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
