#!/usr/bin/env python3
"""Nested session validation for the lightweight touch classifiers."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

try:
    from . import run_lightweight_time_grid_models as lightweight
except ImportError:
    import run_lightweight_time_grid_models as lightweight


def group_split(samples, seed, validation_ratio=0.22):
    """Reserve whole sessions of every class for early stopping."""
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
    if {sample["classLabel"] for sample in fit} != set(
        lightweight.CLASS_LABELS
    ):
        raise RuntimeError("inner_fit_missing_class")
    if {sample["classLabel"] for sample in inner_valid} != set(
        lightweight.CLASS_LABELS
    ):
        raise RuntimeError("inner_valid_missing_class")
    if {sample["groupId"] for sample in fit}.intersection(
        {sample["groupId"] for sample in inner_valid}
    ):
        raise RuntimeError("inner_group_leakage")
    return fit, inner_valid


def prepare_values(fit_samples, inner_samples, outer_samples):
    indices = lightweight.PHYSICAL_INDICES
    fit_raw = lightweight.select_matrix(fit_samples, indices)
    inner_raw = lightweight.select_matrix(inner_samples, indices)
    outer_raw = lightweight.select_matrix(outer_samples, indices)
    center, scale = lightweight.fit_scaler(fit_raw)
    fit_stats = lightweight.stats_features(
        lightweight.transform(fit_raw, center, scale)
    )
    inner_stats = lightweight.stats_features(
        lightweight.transform(inner_raw, center, scale)
    )
    outer_stats = lightweight.stats_features(
        lightweight.transform(outer_raw, center, scale)
    )
    stats_center = fit_stats.mean(axis=0)
    stats_scale = fit_stats.std(axis=0)
    stats_scale[stats_scale < 1e-6] = 1.0
    fit_stats = np.clip(
        (fit_stats - stats_center) / stats_scale, -8.0, 8.0
    ).astype(np.float32)
    inner_stats = np.clip(
        (inner_stats - stats_center) / stats_scale, -8.0, 8.0
    ).astype(np.float32)
    outer_stats = np.clip(
        (outer_stats - stats_center) / stats_scale, -8.0, 8.0
    ).astype(np.float32)
    return fit_stats, inner_stats, outer_stats


def train_once(
    model,
    x_train,
    y_train,
    weights,
    seed,
    epochs,
    learning_rate=0.002,
    early_stop_x=None,
    early_stop_y=None,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.003
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate / 30
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
            if model.multitask:
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


def run_seed(
    fit_samples,
    inner_samples,
    outer_samples,
    hidden_count,
    multitask,
    seed,
):
    lightweight.configure_determinism(seed)
    fit_stats, inner_stats, outer_stats = prepare_values(
        fit_samples, inner_samples, outer_samples
    )
    fit_x = torch.from_numpy(fit_stats)
    inner_x = torch.from_numpy(inner_stats)
    outer_x = torch.from_numpy(outer_stats)
    fit_y = torch.tensor(
        [sample["label"] for sample in fit_samples], dtype=torch.long
    )
    inner_y = torch.tensor(
        [sample["label"] for sample in inner_samples], dtype=torch.long
    )
    outer_y = torch.tensor(
        [sample["label"] for sample in outer_samples], dtype=torch.long
    )
    fit_weights = torch.from_numpy(lightweight.sample_weights(fit_samples))
    selection_model = lightweight.StatsMlp(
        fit_stats.shape[1], hidden_count, multitask
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
    # Refit on every session of the outer training fold with a fixed budget.
    all_train = fit_samples + inner_samples
    all_train.sort(key=lambda sample: sample["sourceSha256"])
    center, scale = lightweight.fit_scaler(
        lightweight.select_matrix(
            all_train, lightweight.PHYSICAL_INDICES
        )
    )
    all_stats = lightweight.stats_features(
        lightweight.transform(
            lightweight.select_matrix(
                all_train, lightweight.PHYSICAL_INDICES
            ),
            center,
            scale,
        )
    )
    outer_stats_final = lightweight.stats_features(
        lightweight.transform(
            lightweight.select_matrix(
                outer_samples, lightweight.PHYSICAL_INDICES
            ),
            center,
            scale,
        )
    )
    stats_center = all_stats.mean(axis=0)
    stats_scale = all_stats.std(axis=0)
    stats_scale[stats_scale < 1e-6] = 1.0
    all_stats = np.clip(
        (all_stats - stats_center) / stats_scale, -8.0, 8.0
    ).astype(np.float32)
    outer_stats_final = np.clip(
        (outer_stats_final - stats_center) / stats_scale, -8.0, 8.0
    ).astype(np.float32)
    final_x = torch.from_numpy(all_stats)
    final_y = torch.tensor(
        [sample["label"] for sample in all_train], dtype=torch.long
    )
    final_weights = torch.from_numpy(lightweight.sample_weights(all_train))
    final_model = lightweight.StatsMlp(
        all_stats.shape[1], hidden_count, multitask
    )
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
        lightweight.predict(final_model, torch.from_numpy(outer_stats_final)),
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
    specs = (
        ("stats_16_single", 16, False),
        ("stats_24_multitask", 24, True),
    )
    report = {
        "scope": "single_device_cross_session",
        "earlyStopping": "inner_group_holdout",
        "outerFoldUsedForModelSelection": False,
        "models": {},
    }
    for name, hidden_count, multitask in specs:
        model_folds = []
        for fold, (train_samples, outer_samples) in sorted(folds.items()):
            fit_samples, inner_samples = group_split(
                train_samples, 81000 + fold
            )
            seed_reports = []
            for seed_offset in range(3):
                seed_reports.append(
                    run_seed(
                        fit_samples,
                        inner_samples,
                        outer_samples,
                        hidden_count,
                        multitask,
                        20261022 + fold * 100 + seed_offset,
                    )
                )
            scores = [
                item["outerMetrics"]["balancedAccuracy"]
                for item in seed_reports
            ]
            recalls = [
                value
                for item in seed_reports
                for value in item["outerMetrics"]["recallByClass"].values()
            ]
            model_folds.append(
                {
                    "fold": fold,
                    "seedReports": seed_reports,
                    "meanBalancedAccuracy": float(np.mean(scores)),
                    "worstSeedBalancedAccuracy": float(np.min(scores)),
                    "worstClassRecall": float(np.min(recalls)),
                }
            )
            print(
                json.dumps(
                    {
                        "model": name,
                        "fold": fold,
                        "meanBa": model_folds[-1]["meanBalancedAccuracy"],
                        "worstSeedBa": model_folds[-1][
                            "worstSeedBalancedAccuracy"
                        ],
                        "worstRecall": model_folds[-1]["worstClassRecall"],
                    }
                ),
                flush=True,
            )
        report["models"][name] = {
            "folds": model_folds,
            "meanBalancedAccuracy": float(
                np.mean(
                    [
                        fold_report["meanBalancedAccuracy"]
                        for fold_report in model_folds
                    ]
                )
            ),
            "worstFoldBalancedAccuracy": float(
                np.min(
                    [
                        fold_report["worstSeedBalancedAccuracy"]
                        for fold_report in model_folds
                    ]
                )
            ),
            "worstClassRecall": float(
                np.min(
                    [
                        fold_report["worstClassRecall"]
                        for fold_report in model_folds
                    ]
                )
            ),
        }
        print(
            json.dumps({"model": name, **report["models"][name]}),
            flush=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
