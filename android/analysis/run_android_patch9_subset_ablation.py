#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from train_android_touch_cnn import DEFAULT_FEATURES
from train_android_touch_cnn import PATCH_FEATURES
from train_android_touch_cnn import feature_value


FEATURE_GROUPS = {
    "covariance": {"contactPatch.cov_11", "contactPatch.cov_12", "contactPatch.cov_22"},
    "sigma": {"contactPatch.sigmaX", "contactPatch.sigmaY"},
    "raw_axis": {"touchNormalization.rawMajorPx", "touchNormalization.rawMinorPx"},
    "area": {"touchNormalization.areaValue"},
    "confidence": {"touchNormalization.confidence"},
}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_indices = parse_int_list(args.frame_indices)
    dataset = load_dataset(Path(args.manifest), frame_indices)
    seeds = parse_int_list(args.seeds)
    run_rows = []
    summary_rows = []
    for subset_indices in patch_subsets(args.min_subset_size, args.max_subset_size):
        subset = [PATCH_FEATURES[index] for index in subset_indices]
        subset_key = subset_name(subset)
        x = matrix_for_subset(dataset, subset_indices)
        subset_runs = []
        for seed in seeds:
            train_index, valid_index = stratified_indices(dataset["labels"], args.validation_ratio, seed)
            model = train_model(x[train_index], dataset["labels"][train_index], args.variance_floor)
            train_metrics = evaluate_model(model, x[train_index], dataset["labels"][train_index])
            valid_metrics = evaluate_model(model, x[valid_index], dataset["labels"][valid_index])
            row = make_run_row(subset_key, subset, seed, x, train_index, valid_index, train_metrics, valid_metrics, args)
            run_rows.append(row)
            subset_runs.append(row)
        summary_rows.append(make_summary_row(subset_key, subset, subset_runs))
    summary_rows.sort(key=rank_key, reverse=True)
    for index, row in enumerate(summary_rows, start=1):
        row["rank"] = index
    write_csv(output_dir / "patch9_subset_runs.csv", run_rows)
    write_csv(output_dir / "patch9_subset_summary.csv", summary_rows)
    write_json(output_dir / "patch9_subset_report.json", {
        "manifest": args.manifest,
        "sampleCount": int(dataset["labels"].shape[0]),
        "labelCounts": dict(Counter(dataset["strengths"])),
        "frameIndices": frame_indices,
        "baseFeatureCount": len(DEFAULT_FEATURES),
        "patchFeatureCount": len(PATCH_FEATURES),
        "subsetCount": len(summary_rows),
        "seeds": seeds,
        "validationRatio": args.validation_ratio,
        "modelKind": "diagonal_lda_probe_numpy",
        "rankingNote": "Sorted by balanced accuracy, min recall, information score, compactness, then AUC.",
        "top20": summary_rows[:20],
    })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-indices", default="-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--validation-ratio", type=float, default=0.25)
    parser.add_argument("--seeds", default="7,11,19")
    parser.add_argument("--variance-floor", type=float, default=0.05)
    parser.add_argument("--min-subset-size", type=int, default=1)
    parser.add_argument("--max-subset-size", type=int, default=9)
    return parser.parse_args()


def load_dataset(manifest_path, frame_indices):
    base_rows = []
    patch_rows = []
    labels = []
    strengths = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            strength = row.get("touch_strength", "")
            if strength not in ("heavy", "light"):
                continue
            path = resolve_path(row, manifest_path)
            if path is None:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            by_index = {}
            for frame in data.get("frames") or []:
                by_index[int_value(frame.get("frameIndex"), 0)] = frame
            base = []
            for frame_index in frame_indices:
                frame = by_index.get(frame_index, {})
                for name in DEFAULT_FEATURES:
                    base.append(finite_float(feature_value(data, frame, name), 0.0))
            patch = [finite_float(feature_value(data, {}, name), 0.0) for name in PATCH_FEATURES]
            base_rows.append(base)
            patch_rows.append(patch)
            labels.append(1 if strength == "heavy" else 0)
            strengths.append(strength)
    return {
        "base": np.asarray(base_rows, dtype=np.float64),
        "patch": np.asarray(patch_rows, dtype=np.float64),
        "labels": np.asarray(labels, dtype=np.int64),
        "strengths": strengths,
        "frameCount": len(frame_indices),
    }


def matrix_for_subset(dataset, subset_indices):
    patch = dataset["patch"][:, subset_indices]
    repeated_patch = np.tile(patch, (1, dataset["frameCount"]))
    return np.concatenate([dataset["base"], repeated_patch], axis=1)


def patch_subsets(min_size, max_size):
    lower = max(1, min_size)
    upper = min(len(PATCH_FEATURES), max_size)
    for size in range(lower, upper + 1):
        for subset in itertools.combinations(range(len(PATCH_FEATURES)), size):
            yield list(subset)


def stratified_indices(labels, validation_ratio, seed):
    rng = np.random.default_rng(seed)
    train_parts = []
    valid_parts = []
    for label in sorted(set(labels.tolist())):
        indices = np.where(labels == label)[0]
        rng.shuffle(indices)
        valid_count = max(1, int(round(indices.shape[0] * validation_ratio)))
        valid_parts.append(indices[:valid_count])
        train_parts.append(indices[valid_count:])
    train = np.concatenate(train_parts)
    valid = np.concatenate(valid_parts)
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def train_model(x_train, y_train, variance_floor):
    center = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    normalized = (x_train - center) / scale
    heavy = normalized[y_train == 1]
    light = normalized[y_train == 0]
    heavy_mean = heavy.mean(axis=0) if heavy.shape[0] else np.zeros(normalized.shape[1])
    light_mean = light.mean(axis=0) if light.shape[0] else np.zeros(normalized.shape[1])
    heavy_var = heavy.var(axis=0, ddof=1) if heavy.shape[0] > 1 else np.ones(normalized.shape[1])
    light_var = light.var(axis=0, ddof=1) if light.shape[0] > 1 else np.ones(normalized.shape[1])
    pooled = (
        heavy_var * max(1, heavy.shape[0] - 1)
        + light_var * max(1, light.shape[0] - 1)
    ) / max(1, heavy.shape[0] + light.shape[0] - 2)
    pooled = np.maximum(pooled, variance_floor)
    weights = (heavy_mean - light_mean) / pooled
    prior_heavy = max(1e-9, heavy.shape[0] / max(1, normalized.shape[0]))
    prior_light = max(1e-9, light.shape[0] / max(1, normalized.shape[0]))
    bias = math.log(prior_heavy / prior_light)
    bias += float(-0.5 * np.sum((heavy_mean * heavy_mean - light_mean * light_mean) / pooled))
    return {
        "center": center,
        "scale": scale,
        "weights": weights,
        "bias": bias,
    }


def evaluate_model(model, x, y):
    normalized = (x - model["center"]) / model["scale"]
    scores = normalized @ model["weights"] + model["bias"]
    prediction = (scores >= 0).astype(np.int64)
    heavy_mask = y == 1
    light_mask = y == 0
    heavy_heavy = int(np.sum((prediction == 1) & heavy_mask))
    heavy_light = int(np.sum((prediction == 0) & heavy_mask))
    light_heavy = int(np.sum((prediction == 1) & light_mask))
    light_light = int(np.sum((prediction == 0) & light_mask))
    heavy_total = heavy_heavy + heavy_light
    light_total = light_light + light_heavy
    heavy_recall = heavy_heavy / heavy_total if heavy_total else 0.0
    light_recall = light_light / light_total if light_total else 0.0
    total = max(1, y.shape[0])
    return {
        "accuracy": (heavy_heavy + light_light) / total,
        "balancedAccuracy": (heavy_recall + light_recall) * 0.5,
        "heavyRecall": heavy_recall,
        "lightRecall": light_recall,
        "minRecall": min(heavy_recall, light_recall),
        "auc": auc_score(scores, y),
        "confusion": {
            "heavyHeavy": heavy_heavy,
            "heavyLight": heavy_light,
            "lightHeavy": light_heavy,
            "lightLight": light_light,
        },
    }


def auc_score(scores, labels):
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(scores)
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    rank_sum = 0.0
    index = 0
    while index < sorted_scores.shape[0]:
        next_index = index + 1
        while next_index < sorted_scores.shape[0] and sorted_scores[next_index] == sorted_scores[index]:
            next_index += 1
        rank = (index + 1 + next_index) * 0.5
        rank_sum += rank * int(np.sum(sorted_labels[index:next_index] == 1))
        index = next_index
    return (rank_sum - positives * (positives + 1) * 0.5) / (positives * negatives)


def make_run_row(subset_key, subset, seed, x, train_index, valid_index, train_metrics, valid_metrics, args):
    return {
        "subset": subset_key,
        "seed": seed,
        "patch_channel_count": len(subset),
        "total_channel_count": len(DEFAULT_FEATURES) + len(subset),
        "feature_count": int(x.shape[1]),
        "information_score": information_score(subset),
        "groups": "|".join(active_groups(subset)),
        "features": "|".join(subset),
        "sample_count": int(x.shape[0]),
        "train_count": int(train_index.shape[0]),
        "valid_count": int(valid_index.shape[0]),
        "validation_ratio": format_number(args.validation_ratio),
        "train_balanced_accuracy": format_number(train_metrics["balancedAccuracy"]),
        "train_auc": format_number(train_metrics["auc"]),
        "valid_accuracy": format_number(valid_metrics["accuracy"]),
        "valid_balanced_accuracy": format_number(valid_metrics["balancedAccuracy"]),
        "valid_auc": format_number(valid_metrics["auc"]),
        "valid_heavy_recall": format_number(valid_metrics["heavyRecall"]),
        "valid_light_recall": format_number(valid_metrics["lightRecall"]),
        "valid_min_recall": format_number(valid_metrics["minRecall"]),
        "valid_heavy_as_heavy": valid_metrics["confusion"]["heavyHeavy"],
        "valid_heavy_as_light": valid_metrics["confusion"]["heavyLight"],
        "valid_light_as_heavy": valid_metrics["confusion"]["lightHeavy"],
        "valid_light_as_light": valid_metrics["confusion"]["lightLight"],
    }


def make_summary_row(subset_key, subset, rows):
    output = {
        "rank": 0,
        "subset": subset_key,
        "patch_channel_count": len(subset),
        "total_channel_count": len(DEFAULT_FEATURES) + len(subset),
        "information_score": information_score(subset),
        "groups": "|".join(active_groups(subset)),
        "features": "|".join(subset),
        "runs": len(rows),
    }
    for key in (
            "valid_accuracy",
            "valid_balanced_accuracy",
            "valid_auc",
            "valid_heavy_recall",
            "valid_light_recall",
            "valid_min_recall",
            "train_balanced_accuracy"):
        values = [finite_float(row[key], 0.0) for row in rows]
        output[key + "_mean"] = format_number(mean(values))
        output[key + "_std"] = format_number(stddev(values))
    return output


def rank_key(row):
    return (
        finite_float(row["valid_balanced_accuracy_mean"], 0.0),
        finite_float(row["valid_min_recall_mean"], 0.0),
        finite_float(row["information_score"], 0.0),
        -int(row["patch_channel_count"]),
        finite_float(row["valid_auc_mean"], 0.0),
    )


def information_score(subset):
    groups = set(active_groups(subset))
    score = len(groups) * 10
    if {"covariance", "raw_axis"}.issubset(groups):
        score += 4
    if {"sigma", "raw_axis"}.issubset(groups):
        score += 3
    if {"area", "raw_axis"}.issubset(groups):
        score += 2
    if "confidence" in groups:
        score -= 2
    return score


def active_groups(subset):
    values = set(subset)
    return [name for name, features in FEATURE_GROUPS.items() if values & features]


def subset_name(subset):
    return "+".join(short_name(name) for name in subset)


def short_name(name):
    return {
        "contactPatch.cov_11": "cov11",
        "contactPatch.cov_12": "cov12",
        "contactPatch.cov_22": "cov22",
        "contactPatch.sigmaX": "sigmaX",
        "contactPatch.sigmaY": "sigmaY",
        "touchNormalization.rawMajorPx": "rawMajor",
        "touchNormalization.rawMinorPx": "rawMinor",
        "touchNormalization.areaValue": "area",
        "touchNormalization.confidence": "confidence",
    }.get(name, name)


def resolve_path(row, manifest_path):
    for key in ("path", "raw_original_path", "raw_path", "chosen_path", "all_flat_path"):
        value = row.get(key)
        if not value:
            continue
        candidate = Path(value)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if candidate.exists():
            return candidate
        for root in [Path.cwd()] + list(manifest_path.resolve().parents):
            resolved = root / candidate
            if resolved.exists():
                return resolved
    return None


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def int_value(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def finite_float(value, fallback):
    try:
        output = float(value)
    except (TypeError, ValueError):
        return fallback
    return output if math.isfinite(output) else fallback


def format_number(value):
    return f"{float(value):.10g}"


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stddev(values):
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


if __name__ == "__main__":
    main()
