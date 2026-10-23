#!/usr/bin/env python3
"""Measure whether the classifier generalizes across samples in one session."""

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

try:
    from . import run_lightweight_time_grid_models as lightweight
    from . import run_stats_mlp_feature_experiments as feature_experiments
except ImportError:
    import run_lightweight_time_grid_models as lightweight
    import run_stats_mlp_feature_experiments as feature_experiments


TOUCH_START = 7


def prepare_rows(dataset_dir):
    rows = lightweight.read_rows(dataset_dir)
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["groupId"]].append({
            "sourceSha256": row["sourceSha256"],
            "sampleId": row["sampleId"],
            "classLabel": row["classLabel"],
            "label": lightweight.CLASS_LABELS.index(row["classLabel"]),
            "groupId": row["groupId"],
            "matrix": np.asarray(row["matrix"], dtype=np.float32),
        })
    return groups


def split_group(samples):
    """Hold out the tail of each session to approximate future touches."""
    ordered = sorted(
        samples,
        key=lambda sample: (
            sample.get("sourceFile", ""),
            sample["sourceSha256"],
        ),
    )
    valid_count = max(2, int(round(len(ordered) * 0.2)))
    split_index = len(ordered) - valid_count
    train = ordered[:split_index]
    valid = ordered[split_index:]
    return train, valid

def summarize_metrics(reports):
    """Report one-class sessions with the metric users can interpret."""
    valid_count = sum(item["validSampleCount"] for item in reports)
    correct_count = sum(
        round(
            item["metrics"]["accuracy"]
            * item["validSampleCount"]
        )
        for item in reports
    )
    target_recalls = [
        item["targetClassRecall"] for item in reports
    ]
    class_summaries = {}
    for label in lightweight.CLASS_LABELS:
        class_reports = [
            item for item in reports if item["classLabel"] == label
        ]
        class_valid = sum(
            item["validSampleCount"] for item in class_reports
        )
        class_correct = sum(
            round(
                item["metrics"]["accuracy"]
                * item["validSampleCount"]
            )
            for item in class_reports
        )
        class_summaries[label] = {
            "groupCount": len(class_reports),
            "validSampleCount": class_valid,
            "accuracy": (
                class_correct / class_valid if class_valid else 0.0
            ),
            "meanGroupAccuracy": float(np.mean([
                item["metrics"]["accuracy"]
                for item in class_reports
            ])) if class_reports else 0.0,
            "worstGroupAccuracy": float(np.min([
                item["metrics"]["accuracy"]
                for item in class_reports
            ])) if class_reports else 0.0,
        }
    return {
        "overallAccuracy": (
            correct_count / valid_count if valid_count else 0.0
        ),
        "meanGroupAccuracy": float(np.mean([
            item["metrics"]["accuracy"] for item in reports
        ])) if reports else 0.0,
        "worstGroupAccuracy": float(np.min([
            item["metrics"]["accuracy"] for item in reports
        ])) if reports else 0.0,
        "meanTargetRecall": float(np.mean(target_recalls)) if reports else 0.0,
        "worstTargetRecall": float(np.min(target_recalls)) if reports else 0.0,
        "classSummaries": class_summaries,
    }


def prepare_values(train_samples, valid_samples):
    train_raw = lightweight.select_matrix(
        train_samples,
        lightweight.PHYSICAL_INDICES,
    )
    valid_raw = lightweight.select_matrix(
        valid_samples,
        lightweight.PHYSICAL_INDICES,
    )
    center, scale = feature_experiments.fit_scaler_windowed(
        train_raw,
        TOUCH_START,
    )
    train_stats = feature_experiments.stats_features_windowed(
        lightweight.transform(train_raw, center, scale),
        TOUCH_START,
    )
    valid_stats = feature_experiments.stats_features_windowed(
        lightweight.transform(valid_raw, center, scale),
        TOUCH_START,
    )
    stats_center = train_stats.mean(axis=0)
    stats_scale = train_stats.std(axis=0)
    stats_scale[stats_scale < 1e-6] = 1.0
    train_stats = np.clip(
        (train_stats - stats_center) / stats_scale,
        -8.0,
        8.0,
    ).astype(np.float32)
    valid_stats = np.clip(
        (valid_stats - stats_center) / stats_scale,
        -8.0,
        8.0,
    ).astype(np.float32)
    return train_stats, valid_stats


def train_once(model, values, labels, seed, epochs=20):
    x = torch.from_numpy(values)
    y = torch.tensor(labels, dtype=torch.long)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=0.003,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=0.00005,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(order), 128):
            batch = order[start:start + 128]
            losses, finger_logits, strength_logits = (
                lightweight.four_class_losses(
                    model,
                    x[batch],
                    y[batch],
                )
            )
            loss = losses.mean()
            loss += 0.25 * (
                torch.nn.functional.cross_entropy(
                    finger_logits,
                    y[batch] // 2,
                )
                + torch.nn.functional.cross_entropy(
                    strength_logits,
                    y[batch] % 2,
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    groups = prepare_rows(args.dataset_dir)
    folds = lightweight.load_folds(
        args.dataset_dir,
        lightweight.read_rows(args.dataset_dir),
    )
    reports = []
    for fold, (train_samples, _) in sorted(folds.items()):
        train_groups = {sample["groupId"] for sample in train_samples}
        fold_group_reports = []
        for group_id in sorted(train_groups):
            samples = groups[group_id]
            if len(samples) < 20:
                continue
            split_train, split_valid = split_group(samples)
            if not split_train or not split_valid:
                continue
            train_values, valid_values = prepare_values(
                split_train,
                split_valid,
            )
            model = lightweight.StatsMlp(
                train_values.shape[1],
                24,
                True,
            )
            train_once(
                model,
                train_values,
                [sample["label"] for sample in split_train],
                20261022 + fold,
            )
            metrics = lightweight.metrics(
                [sample["label"] for sample in split_valid],
                lightweight.predict(
                    model,
                    torch.from_numpy(valid_values),
                ),
            )
            target_class = samples[0]["classLabel"]
            target_index = lightweight.CLASS_LABELS.index(target_class)
            target_total = sum(metrics["confusionMatrix"][target_index])
            target_correct = metrics["confusionMatrix"][target_index][
                target_index
            ]
            fold_group_reports.append({
                "groupId": group_id,
                "sampleCount": len(samples),
                "classLabel": target_class,
                "validSampleCount": len(split_valid),
                "targetClassRecall": (
                    target_correct / target_total
                    if target_total else 0.0
                ),
                "metrics": metrics,
            })
        reports.extend(fold_group_reports)
        print(json.dumps({
            "fold": fold,
            "groupCount": len(fold_group_reports),
            "meanBa": float(np.mean([
                item["metrics"]["balancedAccuracy"]
                for item in fold_group_reports
            ])) if fold_group_reports else 0.0,
        }), flush=True)
    summary = summarize_metrics(reports)
    output = {
        "scope": "within_session_temporal_holdout",
        "touchStartFrame": TOUCH_START,
        "warning": (
            "Each session contains one class. Use overallAccuracy, "
            "meanTargetRecall, and classSummaries; four-class "
            "balancedAccuracy is not comparable here."
        ),
        "reports": reports,
        **summary,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(path),
        "groupCount": len(reports),
        "overallAccuracy": output["overallAccuracy"],
        "meanGroupAccuracy": output["meanGroupAccuracy"],
        "worstGroupAccuracy": output["worstGroupAccuracy"],
        "worstTargetRecall": output["worstTargetRecall"],
    }))


if __name__ == "__main__":
    main()
