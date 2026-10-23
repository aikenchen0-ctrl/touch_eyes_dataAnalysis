#!/usr/bin/env python3
"""Test label-free session adaptation for the compact statistics MLP."""

import argparse
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
ADAPTATION_SIZES = (0, 4, 8, 16, 32, 64)

def prepare_raw(train_samples, valid_samples):
    """Create fold-level values once and reuse them across adaptation modes."""
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
    return (
        lightweight.transform(train_raw, center, scale),
        lightweight.transform(valid_raw, center, scale),
        center,
        scale,
    )

def normalized_statistics(values, stats_center, stats_scale):
    """Apply fixed train statistics to an already sequence-normalized array."""
    stats = feature_experiments.stats_features_windowed(
        values,
        TOUCH_START,
    )
    return np.clip(
        (stats - stats_center) / stats_scale,
        -8.0,
        8.0,
    ).astype(np.float32)

def chronological_session_order(valid_samples):
    """Sort by source filename so adaptation sees only earlier touches."""
    return sorted(
        range(len(valid_samples)),
        key=lambda index: (
            valid_samples[index].get("sourceFile", ""),
            valid_samples[index]["sourceSha256"],
        ),
    )

def standardized_quantile_adaptation(
    base_values,
    adaptation_indices,
    center,
    scale,
    strength,
):
    """Estimate robust location and spread from unlabeled touches."""
    if strength <= 0.0 or not adaptation_indices:
        return base_values.copy(), None
    sample = base_values[np.asarray(adaptation_indices, dtype=np.int64)]
    target_center = np.zeros(base_values.shape[2], dtype=np.float32)
    target_scale = np.ones(base_values.shape[2], dtype=np.float32)
    for channel in range(base_values.shape[2]):
        frames = sample[:, :, channel]
        if channel >= 8:
            frames = frames[:, TOUCH_START:]
        flat = frames.reshape(-1)
        target_center[channel] = np.median(flat)
        low, high = np.percentile(flat, (25, 75))
        delta = high - low
        target_scale[channel] = delta if delta >= 1e-6 else 1.0
    adjusted = base_values.copy()
    for channel in range(adjusted.shape[2]):
        source = adjusted[:, :, channel]
        if channel >= 8:
            source = source[:, TOUCH_START:]
        transformed = (
            (source - target_center[channel])
            / target_scale[channel]
        )
        transformed = np.clip(transformed, -8.0, 8.0)
        if channel >= 8:
            adjusted[:, TOUCH_START:, channel] = (
                (1.0 - strength) * source
                + strength * transformed
            )
        else:
            adjusted[:, :, channel] = (
                (1.0 - strength) * source
                + strength * transformed
            )
    return adjusted, {
        "center": target_center.tolist(),
        "scale": target_scale.tolist(),
    }

def standardized_statistic_adaptation(
    values,
    adaptation_indices,
    stats_center,
    stats_scale,
    strength,
):
    """Recenter final statistics from unlabeled touches without labels."""
    if strength <= 0.0 or not adaptation_indices:
        return values.copy(), None
    sample = values[np.asarray(adaptation_indices, dtype=np.int64)]
    target_center = np.median(sample, axis=0)
    low, high = np.percentile(sample, (25, 75), axis=0)
    target_scale = high - low
    target_scale[target_scale < 1e-6] = 1.0
    target = np.clip(
        (values - target_center) / target_scale,
        -8.0,
        8.0,
    ).astype(np.float32)
    return (
        ((1.0 - strength) * values + strength * target).astype(
            np.float32
        ),
        {
            "center": target_center.tolist(),
            "scale": target_scale.tolist(),
        },
    )

def train_model(
    train_values,
    train_samples,
    seed,
    epochs=20,
):
    """Train one fold model with the production feature contract."""
    lightweight.configure_determinism(seed)
    x = torch.from_numpy(train_values)
    y = torch.tensor(
        [sample["label"] for sample in train_samples],
        dtype=torch.long,
    )
    weights = torch.from_numpy(
        feature_experiments.sample_weights(train_samples, "current")
    )
    model = lightweight.StatsMlp(train_values.shape[1], 24, True)
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
            loss = (losses * weights[batch]).sum()
            loss = loss / weights[batch].sum()
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

def predict_probabilities(model, values):
    """Return probabilities for the joint finger and strength classes."""
    model.eval()
    with torch.no_grad():
        logits = lightweight.logits_for_four_class(
            model,
            torch.from_numpy(values),
        )
        return torch.softmax(logits, dim=1).numpy()

def prior_calibrated_probabilities(
    probabilities,
    adaptation_indices,
    strength,
):
    """Correct the expected class prior using unlabeled probabilities only."""
    if strength <= 0.0 or not adaptation_indices:
        return probabilities.copy(), None
    sample = probabilities[
        np.asarray(adaptation_indices, dtype=np.int64)
    ]
    observed = np.clip(sample.mean(axis=0), 0.02, 0.98)
    observed = observed / observed.sum()
    target = np.full(
        len(lightweight.CLASS_LABELS),
        1.0 / len(lightweight.CLASS_LABELS),
        dtype=np.float64,
    )
    log_correction = np.log(target) - np.log(observed)
    corrected = probabilities * np.exp(
        strength * log_correction[None, :]
    )
    corrected = corrected / corrected.sum(axis=1, keepdims=True)
    return corrected.astype(np.float32), {
        "estimatedPrior": observed.tolist(),
        "logCorrection": log_correction.tolist(),
    }

def score(probabilities, samples):
    predictions = probabilities.argmax(axis=1).tolist()
    return lightweight.metrics(
        [sample["label"] for sample in samples],
        predictions,
    )

def evaluate_variant(
    model,
    base_values,
    valid_samples,
    chronological,
    mode,
    adaptation_count,
    strength,
    center,
    scale,
    stats_center,
    stats_scale,
):
    """Adapt from the first N valid touches and score the untouched tail."""
    adaptation_indices = chronological[:adaptation_count]
    score_indices = chronological[adaptation_count:]
    if not score_indices:
        return None
    values = base_values
    metadata = None
    if mode in ("sequence", "sequence_statistics"):
        values, metadata = standardized_quantile_adaptation(
            values,
            adaptation_indices,
            center,
            scale,
            strength,
        )
    stats = normalized_statistics(values, stats_center, stats_scale)
    if mode in ("statistics", "sequence_statistics"):
        stats, statistic_metadata = standardized_statistic_adaptation(
            stats,
            adaptation_indices,
            stats_center,
            stats_scale,
            strength,
        )
        if statistic_metadata is not None:
            metadata = {
                "sequence": metadata,
                "statistics": statistic_metadata,
            }
    probabilities = predict_probabilities(model, stats)
    if mode in ("prior", "all"):
        probabilities, prior_metadata = prior_calibrated_probabilities(
            probabilities,
            adaptation_indices,
            strength,
        )
        if prior_metadata is not None:
            metadata = prior_metadata
    scored_samples = [valid_samples[index] for index in score_indices]
    return {
        "mode": mode,
        "adaptationCount": adaptation_count,
        "strength": strength,
        "scoredCount": len(score_indices),
        "metrics": score(
            probabilities[np.asarray(score_indices, dtype=np.int64)],
            scored_samples,
        ),
        "adaptationMetadata": metadata,
    }

def aggregate(reports):
    """Summarize folds while exposing the weakest class recall."""
    recalls = [
        value for report in reports
        for value in report["metrics"]["recallByClass"].values()
    ]
    return {
        "meanBalancedAccuracy": float(np.mean([
            report["metrics"]["balancedAccuracy"]
            for report in reports
        ])),
        "worstFoldBalancedAccuracy": float(np.min([
            report["metrics"]["balancedAccuracy"]
            for report in reports
        ])),
        "meanAccuracy": float(np.mean([
            report["metrics"]["accuracy"] for report in reports
        ])),
        "worstClassRecall": float(np.min(recalls)),
        "foldBalancedAccuracies": [
            report["metrics"]["balancedAccuracy"]
            for report in reports
        ],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    rows = lightweight.read_rows(args.dataset_dir)
    folds = lightweight.load_folds(args.dataset_dir, rows)
    variants = [
        (mode, count, strength)
        for mode in ("none", "sequence", "statistics", "prior", "all")
        for count in ADAPTATION_SIZES
        for strength in (0.25, 0.5, 1.0)
        if mode != "none" or (count == 0 and strength == 1.0)
    ]
    reports = {name: [] for name in (
        "{}_{}_{}".format(mode, count, strength)
        for mode, count, strength in variants
    )}
    fold_reports = []
    for fold, (train_samples, valid_samples) in sorted(folds.items()):
        train_raw, valid_raw, center, scale = prepare_raw(
            train_samples,
            valid_samples,
        )
        train_stats_raw = feature_experiments.stats_features_windowed(
            train_raw,
            TOUCH_START,
        )
        stats_center = train_stats_raw.mean(axis=0)
        stats_scale = train_stats_raw.std(axis=0)
        stats_scale[stats_scale < 1e-6] = 1.0
        train_values = normalized_statistics(
            train_raw,
            stats_center,
            stats_scale,
        )
        model = train_model(
            train_values,
            train_samples,
            20261022 + fold,
            args.epochs,
        )
        chronological = chronological_session_order(valid_samples)
        fold_variant_reports = []
        for mode, count, strength in variants:
            key = "{}_{}_{}".format(mode, count, strength)
            report = evaluate_variant(
                model,
                valid_raw,
                valid_samples,
                chronological,
                mode,
                count,
                strength,
                center,
                scale,
                stats_center,
                stats_scale,
            )
            if report is None:
                continue
            report["fold"] = fold
            report["key"] = key
            reports[key].append(report)
            fold_variant_reports.append(report)
        fold_reports.append({
            "fold": fold,
            "variants": fold_variant_reports,
        })
        baseline = next(
            report for report in fold_variant_reports
            if report["key"] == "none_0_1.0"
        )
        print(json.dumps({
            "fold": fold,
            "baselineBa": baseline["metrics"]["balancedAccuracy"],
            "bestBa": max(
                report["metrics"]["balancedAccuracy"]
                for report in fold_variant_reports
            ),
        }), flush=True)
    aggregate_reports = {
        key: aggregate(values)
        for key, values in reports.items()
        if len(values) == len(folds)
    }
    output = {
        "scope": "single_device_cross_session_unlabeled_warmup",
        "touchStartFrame": TOUCH_START,
        "adaptationSizes": list(ADAPTATION_SIZES),
        "adaptationProtocol": (
            "Use the earliest N outer-validation touches without labels; "
            "score only later touches."
        ),
        "selectionMetric": "worst_fold_balanced_accuracy_then_worst_class_recall",
        "folds": fold_reports,
        "aggregates": aggregate_reports,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ranked = sorted(
        aggregate_reports.items(),
        key=lambda item: (
            item[1]["worstFoldBalancedAccuracy"],
            item[1]["worstClassRecall"],
            item[1]["meanBalancedAccuracy"],
        ),
        reverse=True,
    )
    print(json.dumps({
        "output": str(path),
        "top": ranked[:10],
    }))

if __name__ == "__main__":
    main()
