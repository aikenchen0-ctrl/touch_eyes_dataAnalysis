#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from export_android_contact_encoding_report import MAP_SIZE
from export_android_contact_encoding_report import collect_contact_stats
from export_android_contact_encoding_report import covariance_values
from export_android_contact_encoding_report import gaussian_map
from export_android_contact_encoding_report import weighted_spread
from train_android_touch_cnn import DEFAULT_FEATURES
from train_android_touch_cnn import DEFAULT_FRAME_INDICES
from train_android_touch_cnn import PATCH_FEATURES
from train_android_touch_cnn import feature_value


GAUSSIAN_MAP_FEATURES = [f"gaussian.map.{index:02d}" for index in range(MAP_SIZE * MAP_SIZE)]

GAUSSIAN_SUMMARY_FEATURES = [
    "gaussian.center",
    "gaussian.sum",
    "gaussian.energy",
    "gaussian.spread",
    "gaussian.sigma_x_px",
    "gaussian.sigma_y_px",
    "gaussian.valid_mask",
]

MULTI_CHANNEL_FEATURES = [
    "multi.center_x_norm",
    "multi.center_y_norm",
    "multi.major_px",
    "multi.minor_px",
    "multi.orientation_rad",
    "multi.valid_mask",
]

COVARIANCE_FEATURES = [
    "covariance.x_norm",
    "covariance.y_norm",
    "covariance.cov_11",
    "covariance.cov_12",
    "covariance.cov_22",
    "covariance.valid_mask",
]

PATCH_FEATURES_WITHOUT_CONFIDENCE = [
    name for name in PATCH_FEATURES
    if name != "touchNormalization.confidence"
]

PRESETS = {
    "old13": {
        "frameFeatures": DEFAULT_FEATURES,
        "staticFeatures": [],
        "description": "legacy 13 channel time-series input",
    },
    "old13_raw_radius": {
        "frameFeatures": DEFAULT_FEATURES,
        "staticFeatures": [
            "multi.major_px",
            "multi.minor_px",
            "multi.valid_mask",
        ],
        "description": "old13 plus raw major and minor contact radius",
    },
    "old13_covariance": {
        "frameFeatures": DEFAULT_FEATURES,
        "staticFeatures": COVARIANCE_FEATURES,
        "description": "old13 plus covariance matrix contact patch",
    },
    "old13_multichannel": {
        "frameFeatures": DEFAULT_FEATURES,
        "staticFeatures": MULTI_CHANNEL_FEATURES,
        "description": "old13 plus center, axis, orientation and validity channels",
    },
    "old13_gaussian_summary": {
        "frameFeatures": DEFAULT_FEATURES,
        "staticFeatures": GAUSSIAN_SUMMARY_FEATURES,
        "description": "old13 plus gaussian contact map summary values",
    },
    "old13_gaussian_map9x9": {
        "frameFeatures": DEFAULT_FEATURES,
        "staticFeatures": GAUSSIAN_MAP_FEATURES + ["gaussian.valid_mask"],
        "description": "old13 plus flattened 9x9 gaussian contact map",
    },
    "patch9_only": {
        "frameFeatures": PATCH_FEATURES,
        "staticFeatures": [],
        "description": "current 9 contact patch channels without old13",
    },
    "enhanced_patch21_no_confidence": {
        "frameFeatures": DEFAULT_FEATURES + PATCH_FEATURES_WITHOUT_CONFIDENCE,
        "staticFeatures": [],
        "description": "enhanced patch input without confidence diagnostic",
    },
    "enhanced_patch22": {
        "frameFeatures": DEFAULT_FEATURES + PATCH_FEATURES,
        "staticFeatures": [],
        "description": "old13 plus current 9 contact patch channels",
    },
}


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    frame_indices = parse_int_list(args.frame_indices) if args.frame_indices else list(DEFAULT_FRAME_INDICES)
    preset_names = parse_csv_list(args.presets) if args.presets else list(PRESETS.keys())
    samples = load_samples(manifest_path)
    run_rows = []
    prediction_rows = []
    preset_summaries = []

    for preset_name in preset_names:
        preset = PRESETS[preset_name]
        records, skipped = build_records(samples, preset, frame_indices)
        for seed in seeds:
            train_records, valid_records = stratified_split(records, args.validation_ratio, seed)
            normalizer = fit_normalizer([record["vector"] for record in train_records])
            train_vectors = normalize_records(train_records, normalizer)
            valid_vectors = normalize_records(valid_records, normalizer)
            model = train_diagonal_lda(train_vectors, args.variance_floor)
            train_metrics = evaluate(model, train_vectors)
            valid_metrics, valid_predictions = evaluate(model, valid_vectors, include_predictions=True)
            run_rows.append(run_row(
                preset_name,
                preset,
                seed,
                records,
                skipped,
                train_vectors,
                valid_vectors,
                train_metrics,
                valid_metrics,
                args,
            ))
            prediction_rows.extend(prediction_row(preset_name, seed, item) for item in valid_predictions)
        preset_summaries.append(summary_row(preset_name, preset, run_rows))

    preset_summaries.sort(key=lambda row: (
        float(row["valid_balanced_accuracy_mean"]),
        float(row["valid_auc_mean"]),
        float(row["valid_min_recall_mean"]),
    ), reverse=True)
    for rank_index, row in enumerate(preset_summaries, start=1):
        row["rank"] = rank_index

    write_csv(output_dir / "ablation_runs.csv", run_rows)
    write_csv(output_dir / "ablation_summary.csv", preset_summaries)
    write_csv(output_dir / "ablation_predictions.csv", prediction_rows)
    write_json(output_dir / "ablation_report.json", {
        "manifest": str(manifest_path),
        "outputDir": str(output_dir),
        "sampleCount": len(samples),
        "labelCounts": dict(Counter(sample["strength"] for sample in samples)),
        "modelKind": "diagonal_lda_probe",
        "modelNote": "Dependency-free training-level probe. It is not a CNN replacement.",
        "cnnTrainingStatus": detect_cnn_training_status(),
        "validationRatio": args.validation_ratio,
        "seeds": seeds,
        "frameIndices": frame_indices,
        "varianceFloor": args.variance_floor,
        "presets": preset_report(preset_names),
        "summary": preset_summaries,
    })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--presets", default="")
    parser.add_argument("--frame-indices", default="")
    parser.add_argument("--validation-ratio", type=float, default=0.25)
    parser.add_argument("--seeds", default="7,11,19,23,29")
    parser.add_argument("--variance-floor", type=float, default=0.05)
    return parser.parse_args()


def load_samples(manifest_path):
    samples = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            strength = row.get("touch_strength", "")
            if strength not in ("heavy", "light"):
                continue
            data_path = resolve_path(row, manifest_path)
            if data_path is None:
                continue
            with data_path.open("r", encoding="utf-8") as data_file:
                data = json.load(data_file)
            samples.append({
                "id": str(data.get("id") or row.get("id") or ""),
                "path": str(data_path),
                "fileName": data_path.name,
                "strength": strength,
                "label": 1 if strength == "heavy" else 0,
                "data": data,
            })
    return samples


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


def build_records(samples, preset, frame_indices):
    records = []
    skipped = []
    for sample in samples:
        vector = build_vector(sample["data"], preset, frame_indices)
        if vector is None:
            skipped.append(sample["id"])
            continue
        records.append({
            "id": sample["id"],
            "path": sample["path"],
            "fileName": sample["fileName"],
            "strength": sample["strength"],
            "label": sample["label"],
            "vector": vector,
        })
    return records, skipped


def build_vector(data, preset, frame_indices):
    by_index = {}
    for frame in data.get("frames") or []:
        by_index[int_value(frame.get("frameIndex"), 0)] = frame
    values = []
    for frame_index in frame_indices:
        frame = by_index.get(frame_index)
        if frame is None:
            return None
        for feature_name in preset["frameFeatures"]:
            values.append(finite_float(feature_value(data, frame, feature_name), 0.0))
    if preset["staticFeatures"]:
        contact_values = contact_feature_values(data)
        for feature_name in preset["staticFeatures"]:
            values.append(contact_values.get(feature_name, 0.0))
    return values


def contact_feature_values(data):
    stats = collect_contact_stats(data)
    gaussian = gaussian_map(stats)
    covariance = covariance_values(stats)
    values = {
        "gaussian.center": gaussian[(MAP_SIZE * MAP_SIZE) // 2],
        "gaussian.sum": sum(gaussian),
        "gaussian.energy": sum(value * value for value in gaussian),
        "gaussian.spread": weighted_spread(gaussian),
        "gaussian.sigma_x_px": stats["sigmaXPx"],
        "gaussian.sigma_y_px": stats["sigmaYPx"],
        "gaussian.valid_mask": stats["validMask"],
        "multi.center_x_norm": stats["centerXNorm"],
        "multi.center_y_norm": stats["centerYNorm"],
        "multi.major_px": stats["rawMajorPx"],
        "multi.minor_px": stats["rawMinorPx"],
        "multi.orientation_rad": stats["orientationRad"],
        "multi.valid_mask": stats["validMask"],
        "covariance.x_norm": stats["centerXNorm"],
        "covariance.y_norm": stats["centerYNorm"],
        "covariance.cov_11": covariance[0],
        "covariance.cov_12": covariance[1],
        "covariance.cov_22": covariance[2],
        "covariance.valid_mask": stats["validMask"],
    }
    for index, value in enumerate(gaussian):
        values[f"gaussian.map.{index:02d}"] = value
    return values


def stratified_split(records, validation_ratio, seed):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["label"]].append(record)
    rng = random.Random(seed)
    train = []
    valid = []
    for label in sorted(grouped):
        items = list(grouped[label])
        rng.shuffle(items)
        valid_count = max(1, int(round(len(items) * validation_ratio)))
        valid.extend(items[:valid_count])
        train.extend(items[valid_count:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def fit_normalizer(vectors):
    if not vectors:
        return {"center": [], "scale": []}
    size = len(vectors[0])
    center = [0.0] * size
    for vector in vectors:
        for index, value in enumerate(vector):
            center[index] += value
    count = len(vectors)
    center = [value / count for value in center]
    variance = [0.0] * size
    for vector in vectors:
        for index, value in enumerate(vector):
            diff = value - center[index]
            variance[index] += diff * diff
    scale = [math.sqrt(value / count) if value > 0 else 1.0 for value in variance]
    scale = [value if value >= 1e-9 else 1.0 for value in scale]
    return {"center": center, "scale": scale}


def normalize_records(records, normalizer):
    output = []
    center = normalizer["center"]
    scale = normalizer["scale"]
    for record in records:
        vector = [
            (value - center[index]) / scale[index]
            for index, value in enumerate(record["vector"])
        ]
        output.append({
            "id": record["id"],
            "path": record["path"],
            "fileName": record["fileName"],
            "strength": record["strength"],
            "label": record["label"],
            "vector": vector,
        })
    return output


def train_diagonal_lda(records, variance_floor):
    labels = [0, 1]
    size = len(records[0]["vector"]) if records else 0
    grouped = {label: [] for label in labels}
    for record in records:
        grouped[record["label"]].append(record["vector"])
    means = {label: mean_vector(grouped[label], size) for label in labels}
    variances = {label: variance_vector(grouped[label], means[label], size) for label in labels}
    pooled = []
    for index in range(size):
        value = 0.0
        total = 0
        for label in labels:
            count = len(grouped[label])
            value += variances[label][index] * max(1, count - 1)
            total += max(1, count - 1)
        pooled.append(max(variance_floor, value / max(1, total)))
    priors = {
        label: len(grouped[label]) / max(1, len(records))
        for label in labels
    }
    weights = []
    bias = math.log(max(1e-9, priors[1]) / max(1e-9, priors[0]))
    for index in range(size):
        heavy_mean = means[1][index]
        light_mean = means[0][index]
        variance = pooled[index]
        weights.append((heavy_mean - light_mean) / variance)
        bias += -0.5 * ((heavy_mean * heavy_mean - light_mean * light_mean) / variance)
    return {
        "weights": weights,
        "bias": bias,
    }


def mean_vector(vectors, size):
    output = [0.0] * size
    if not vectors:
        return output
    for vector in vectors:
        for index, value in enumerate(vector):
            output[index] += value
    return [value / len(vectors) for value in output]


def variance_vector(vectors, mean_values, size):
    output = [0.0] * size
    if not vectors:
        return [1.0] * size
    for vector in vectors:
        for index, value in enumerate(vector):
            diff = value - mean_values[index]
            output[index] += diff * diff
    return [value / max(1, len(vectors) - 1) for value in output]


def evaluate(model, records, include_predictions=False):
    predictions = []
    scores = []
    labels = []
    confusion = {
        "heavyHeavy": 0,
        "heavyLight": 0,
        "lightHeavy": 0,
        "lightLight": 0,
    }
    for record in records:
        score = score_record(model, record["vector"])
        probability = sigmoid(score)
        predicted_label = 1 if score >= 0.0 else 0
        labels.append(record["label"])
        scores.append(score)
        if record["label"] == 1 and predicted_label == 1:
            confusion["heavyHeavy"] += 1
        elif record["label"] == 1 and predicted_label == 0:
            confusion["heavyLight"] += 1
        elif record["label"] == 0 and predicted_label == 1:
            confusion["lightHeavy"] += 1
        else:
            confusion["lightLight"] += 1
        if include_predictions:
            predictions.append({
                "id": record["id"],
                "file_name": record["fileName"],
                "path": record["path"],
                "truth": record["strength"],
                "prediction": "heavy" if predicted_label == 1 else "light",
                "heavy_probability": probability,
                "score": score,
            })
    heavy_total = confusion["heavyHeavy"] + confusion["heavyLight"]
    light_total = confusion["lightLight"] + confusion["lightHeavy"]
    heavy_recall = confusion["heavyHeavy"] / heavy_total if heavy_total else 0.0
    light_recall = confusion["lightLight"] / light_total if light_total else 0.0
    total = max(1, heavy_total + light_total)
    metrics = {
        "accuracy": (confusion["heavyHeavy"] + confusion["lightLight"]) / total,
        "balancedAccuracy": (heavy_recall + light_recall) * 0.5,
        "heavyRecall": heavy_recall,
        "lightRecall": light_recall,
        "minRecall": min(heavy_recall, light_recall),
        "auc": auc_score(scores, labels),
        "confusion": confusion,
    }
    if include_predictions:
        return metrics, predictions
    return metrics


def score_record(model, vector):
    total = model["bias"]
    for weight, value in zip(model["weights"], vector):
        total += weight * value
    return total


def auc_score(scores, labels):
    positives = sum(1 for label in labels if label == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        next_index = index + 1
        while next_index < len(pairs) and pairs[next_index][0] == pairs[index][0]:
            next_index += 1
        rank = (index + 1 + next_index) * 0.5
        for pair_index in range(index, next_index):
            if pairs[pair_index][1] == 1:
                rank_sum += rank
        index = next_index
    return (rank_sum - positives * (positives + 1) * 0.5) / (positives * negatives)


def run_row(preset_name, preset, seed, records, skipped, train_records, valid_records, train_metrics, valid_metrics, args):
    return {
        "preset": preset_name,
        "model_kind": "diagonal_lda_probe",
        "seed": seed,
        "sample_count": len(records),
        "skipped_count": len(skipped),
        "train_count": len(train_records),
        "valid_count": len(valid_records),
        "feature_count": len(records[0]["vector"]) if records else 0,
        "frame_feature_count": len(preset["frameFeatures"]),
        "static_feature_count": len(preset["staticFeatures"]),
        "validation_ratio": format_number(args.validation_ratio),
        "variance_floor": format_number(args.variance_floor),
        "train_accuracy": format_number(train_metrics["accuracy"]),
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
        "description": preset["description"],
    }


def prediction_row(preset_name, seed, item):
    return {
        "preset": preset_name,
        "seed": seed,
        "id": item["id"],
        "file_name": item["file_name"],
        "truth": item["truth"],
        "prediction": item["prediction"],
        "heavy_probability": format_number(item["heavy_probability"]),
        "score": format_number(item["score"]),
        "path": item["path"],
    }


def summary_row(preset_name, preset, run_rows):
    rows = [row for row in run_rows if row["preset"] == preset_name]
    keys = [
        "valid_accuracy",
        "valid_balanced_accuracy",
        "valid_auc",
        "valid_heavy_recall",
        "valid_light_recall",
        "valid_min_recall",
        "train_balanced_accuracy",
    ]
    output = {
        "rank": 0,
        "preset": preset_name,
        "model_kind": "diagonal_lda_probe",
        "runs": len(rows),
        "feature_count": rows[0]["feature_count"] if rows else 0,
        "frame_feature_count": len(preset["frameFeatures"]),
        "static_feature_count": len(preset["staticFeatures"]),
        "description": preset["description"],
    }
    for key in keys:
        values = [finite_float(row[key], 0.0) for row in rows]
        output[key + "_mean"] = format_number(mean(values))
        output[key + "_std"] = format_number(stddev(values))
    return output


def preset_report(preset_names):
    output = {}
    for name in preset_names:
        preset = PRESETS[name]
        output[name] = {
            "description": preset["description"],
            "frameFeatures": preset["frameFeatures"],
            "staticFeatures": preset["staticFeatures"],
        }
    return output


def detect_cnn_training_status():
    try:
        import torch
    except ModuleNotFoundError:
        return {
            "available": False,
            "reason": "pytorch_not_installed",
        }
    return {
        "available": True,
        "torchVersion": str(torch.__version__),
    }


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


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
    if not math.isfinite(output):
        return fallback
    return output


def sigmoid(value):
    if value >= 0:
        factor = math.exp(-min(60.0, value))
        return 1.0 / (1.0 + factor)
    factor = math.exp(max(-60.0, value))
    return factor / (1.0 + factor)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stddev(values):
    if not values:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def format_number(value):
    numeric = finite_float(value, None)
    if numeric is None:
        return ""
    return f"{numeric:.10g}"


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


if __name__ == "__main__":
    main()
