#!/usr/bin/env python3
"""Validate the touch-valid statistics window without outer-fold leakage."""

import argparse
import json
import random
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


def group_split(samples, seed, validation_ratio=0.22):
    rng = random.Random(seed)
    groups_by_class = {
        label: [] for label in lightweight.CLASS_LABELS
    }
    group_labels = {
        sample["groupId"]: sample["classLabel"] for sample in samples
    }
    for group_id, label in group_labels.items():
        groups_by_class[label].append(group_id)
    inner_valid_groups = set()
    for label, group_ids in groups_by_class.items():
        ordered = sorted(group_ids)
        rng.shuffle(ordered)
        count = max(1, int(round(len(ordered) * validation_ratio)))
        inner_valid_groups.update(ordered[:count])
    fit = [
        sample for sample in samples
        if sample["groupId"] not in inner_valid_groups
    ]
    inner_valid = [
        sample for sample in samples
        if sample["groupId"] in inner_valid_groups
    ]
    if {sample["groupId"] for sample in fit}.intersection(
        {sample["groupId"] for sample in inner_valid}
    ):
        raise RuntimeError("inner_group_leakage")
    for split in (fit, inner_valid):
        if {sample["classLabel"] for sample in split} != set(
            lightweight.CLASS_LABELS
        ):
            raise RuntimeError("inner_split_missing_class")
    return fit, inner_valid


def prepare_values(fit_samples, inner_samples, outer_samples):
    fit_raw = lightweight.select_matrix(
        fit_samples,
        lightweight.PHYSICAL_INDICES,
    )
    inner_raw = lightweight.select_matrix(
        inner_samples,
        lightweight.PHYSICAL_INDICES,
    )
    outer_raw = lightweight.select_matrix(
        outer_samples,
        lightweight.PHYSICAL_INDICES,
    )
    center, scale = feature_experiments.fit_scaler_windowed(
        fit_raw,
        TOUCH_START,
    )
    fit_stats = feature_experiments.stats_features_windowed(
        lightweight.transform(fit_raw, center, scale),
        TOUCH_START,
    )
    inner_stats = feature_experiments.stats_features_windowed(
        lightweight.transform(inner_raw, center, scale),
        TOUCH_START,
    )
    outer_stats = feature_experiments.stats_features_windowed(
        lightweight.transform(outer_raw, center, scale),
        TOUCH_START,
    )
    stats_center = fit_stats.mean(axis=0)
    stats_scale = fit_stats.std(axis=0)
    stats_scale[stats_scale < 1e-6] = 1.0
    fit_stats = np.clip(
        (fit_stats - stats_center) / stats_scale,
        -8.0,
        8.0,
    ).astype(np.float32)
    inner_stats = np.clip(
        (inner_stats - stats_center) / stats_scale,
        -8.0,
        8.0,
    ).astype(np.float32)
    outer_stats = np.clip(
        (outer_stats - stats_center) / stats_scale,
        -8.0,
        8.0,
    ).astype(np.float32)
    return fit_stats, inner_stats, outer_stats


def train_once(
    model,
    x_train,
    y_train,
    train_weights,
    seed,
    epochs,
    early_stop_x=None,
    early_stop_y=None,
):
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
    best_state = None
    best_score = -1.0
    best_epoch = 0
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
        if early_stop_x is not None:
            inner_metrics = lightweight.metrics(
                early_stop_y.tolist(),
                lightweight.predict(model, early_stop_x),
            )
            score = inner_metrics["balancedAccuracy"]
            if score > best_score + 1e-12:
                best_score = score
                best_epoch = epoch
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }
            if epoch - best_epoch >= 8:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_score


def run_seed(fit_samples, inner_samples, outer_samples, seed):
    lightweight.configure_determinism(seed)
    fit_stats, inner_stats, outer_stats = prepare_values(
        fit_samples,
        inner_samples,
        outer_samples,
    )
    fit_x = torch.from_numpy(fit_stats)
    inner_x = torch.from_numpy(inner_stats)
    outer_x = torch.from_numpy(outer_stats)
    fit_y = torch.tensor(
        [sample["label"] for sample in fit_samples],
        dtype=torch.long,
    )
    inner_y = torch.tensor(
        [sample["label"] for sample in inner_samples],
        dtype=torch.long,
    )
    outer_y = torch.tensor(
        [sample["label"] for sample in outer_samples],
        dtype=torch.long,
    )
    fit_weights = torch.from_numpy(
        feature_experiments.sample_weights(fit_samples, "current")
    )
    selection_model = lightweight.StatsMlp(
        fit_stats.shape[1],
        24,
        True,
    )
    _, best_epoch, inner_score = train_once(
        selection_model,
        fit_x,
        fit_y,
        fit_weights,
        seed,
        55,
        early_stop_x=inner_x,
        early_stop_y=inner_y,
    )
    all_train = sorted(
        fit_samples + inner_samples,
        key=lambda sample: sample["sourceSha256"],
    )
    all_raw = lightweight.select_matrix(
        all_train,
        lightweight.PHYSICAL_INDICES,
    )
    outer_raw = lightweight.select_matrix(
        outer_samples,
        lightweight.PHYSICAL_INDICES,
    )
    center, scale = feature_experiments.fit_scaler_windowed(
        all_raw,
        TOUCH_START,
    )
    all_stats = feature_experiments.stats_features_windowed(
        lightweight.transform(all_raw, center, scale),
        TOUCH_START,
    )
    outer_stats = feature_experiments.stats_features_windowed(
        lightweight.transform(outer_raw, center, scale),
        TOUCH_START,
    )
    stats_center = all_stats.mean(axis=0)
    stats_scale = all_stats.std(axis=0)
    stats_scale[stats_scale < 1e-6] = 1.0
    all_stats = np.clip(
        (all_stats - stats_center) / stats_scale,
        -8.0,
        8.0,
    ).astype(np.float32)
    outer_stats = np.clip(
        (outer_stats - stats_center) / stats_scale,
        -8.0,
        8.0,
    ).astype(np.float32)
    final_x = torch.from_numpy(all_stats)
    final_y = torch.tensor(
        [sample["label"] for sample in all_train],
        dtype=torch.long,
    )
    final_weights = torch.from_numpy(
        feature_experiments.sample_weights(all_train, "current")
    )
    final_model = lightweight.StatsMlp(all_stats.shape[1], 24, True)
    fixed_epochs = max(8, best_epoch)
    train_once(
        final_model,
        final_x,
        final_y,
        final_weights,
        seed + 10000,
        fixed_epochs,
    )
    outer_metrics = lightweight.metrics(
        outer_y.tolist(),
        lightweight.predict(final_model, torch.from_numpy(outer_stats)),
    )
    return {
        "seed": seed,
        "innerBestEpoch": best_epoch,
        "innerBestBalancedAccuracy": inner_score,
        "parameterCount": sum(
            parameter.numel() for parameter in final_model.parameters()
        ),
        "outerMetrics": outer_metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = lightweight.read_rows(args.dataset_dir)
    folds = lightweight.load_folds(args.dataset_dir, rows)
    model_folds = []
    for fold, (train_samples, outer_samples) in sorted(folds.items()):
        fit_samples, inner_samples = group_split(
            train_samples,
            81000 + fold,
        )
        seed_reports = []
        for seed_offset in range(3):
            seed_reports.append(run_seed(
                fit_samples,
                inner_samples,
                outer_samples,
                20261022 + fold * 100 + seed_offset,
            ))
        scores = [
            item["outerMetrics"]["balancedAccuracy"]
            for item in seed_reports
        ]
        recalls = [
            value
            for item in seed_reports
            for value in item["outerMetrics"]["recallByClass"].values()
        ]
        model_folds.append({
            "fold": fold,
            "seedReports": seed_reports,
            "meanBalancedAccuracy": float(np.mean(scores)),
            "worstSeedBalancedAccuracy": float(np.min(scores)),
            "worstClassRecall": float(np.min(recalls)),
        })
        print(json.dumps({
            "fold": fold,
            "meanBa": model_folds[-1]["meanBalancedAccuracy"],
            "worstSeedBa": model_folds[-1][
                "worstSeedBalancedAccuracy"
            ],
            "worstRecall": model_folds[-1]["worstClassRecall"],
        }), flush=True)
    report = {
        "scope": "single_device_cross_session",
        "touchStartFrame": TOUCH_START,
        "earlyStopping": "inner_group_holdout",
        "outerFoldUsedForModelSelection": False,
        "folds": model_folds,
        "meanBalancedAccuracy": float(np.mean([
            item["meanBalancedAccuracy"] for item in model_folds
        ])),
        "worstFoldBalancedAccuracy": float(np.min([
            item["worstSeedBalancedAccuracy"] for item in model_folds
        ])),
        "worstClassRecall": float(np.min([
            item["worstClassRecall"] for item in model_folds
        ])),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(path),
        "meanBa": report["meanBalancedAccuracy"],
        "worstBa": report["worstFoldBalancedAccuracy"],
        "worstRecall": report["worstClassRecall"],
    }))


if __name__ == "__main__":
    main()
