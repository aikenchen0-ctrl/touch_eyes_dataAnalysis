#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from train_android_touch_cnn import (
    DEFAULT_FEATURES,
    FEATURE_PRESETS,
    PATCH_FEATURES,
    feature_value,
)


NORMALIZATION_SCHEME_FEATURES = [
    "touchNormalization.screenAreaValue",
    "touchNormalization.covarianceAreaValue",
    "touchNormalization.baselineAreaValue",
]

REPORT_FEATURES = PATCH_FEATURES + NORMALIZATION_SCHEME_FEATURES


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(manifest_path)
    feature_rows = build_feature_rows(samples, REPORT_FEATURES)
    summary_rows = build_summary_rows(feature_rows, REPORT_FEATURES)
    threshold_rows = build_threshold_rows(feature_rows, REPORT_FEATURES)
    preview_rows = feature_rows[:args.preview_count]

    write_csv(output_dir / "enhanced_patch_feature_summary.csv", summary_rows)
    write_csv(output_dir / "enhanced_patch_threshold_scan.csv", threshold_rows)
    write_csv(output_dir / "enhanced_patch_samples_preview.csv", preview_rows)

    report = {
        "manifest": str(manifest_path),
        "sampleCount": len(samples),
        "labelCounts": dict(Counter(sample["strength"] for sample in samples)),
        "deviceLevelCounts": dict(Counter(sample["deviceLevel"] for sample in samples)),
        "normalizationSchemeCounts": dict(Counter(sample["selectedScheme"] for sample in samples)),
        "featurePresets": feature_presets_report(),
        "analyzedFeatures": REPORT_FEATURES,
        "topThresholdFeatures": threshold_rows[:8],
    }
    write_json(output_dir / "enhanced_patch_feature_report.json", report)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preview-count", type=int, default=24)
    return parser.parse_args()


def load_samples(manifest_path):
    samples = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            strength = row.get("touch_strength", "")
            if strength not in ("heavy", "light"):
                continue
            data_path = resolve_data_path(row.get("path") or "", manifest_path)
            if data_path is None:
                continue
            with data_path.open("r", encoding="utf-8") as data_file:
                data = json.load(data_file)
            samples.append({
                "path": str(data_path),
                "id": str(data.get("id") or row.get("id") or ""),
                "strength": strength,
                "data": data,
                "deviceLevel": str((data.get("deviceCapability") or {}).get("level") or row.get("device_level") or ""),
                "selectedScheme": str((data.get("touchNormalization") or {}).get("selectedScheme") or ""),
            })
    return samples


def resolve_data_path(value, manifest_path):
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate
    search_roots = [Path.cwd()] + list(manifest_path.resolve().parents)
    for root in search_roots:
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return None


def build_feature_rows(samples, features):
    rows = []
    for sample in samples:
        data = sample["data"]
        frame = first_frame(data)
        row = {
            "id": sample["id"],
            "touch_strength": sample["strength"],
            "file_name": Path(sample["path"]).name,
            "device_level": sample["deviceLevel"],
            "selected_scheme": sample["selectedScheme"],
        }
        for name in features:
            row[name] = format_number(feature_value(data, frame, name))
        rows.append(row)
    return rows


def first_frame(data):
    frames = data.get("frames") or []
    if frames:
        return frames[0]
    return {}


def build_summary_rows(feature_rows, features):
    rows = []
    grouped = defaultdict(list)
    for row in feature_rows:
        grouped[row["touch_strength"]].append(row)
    for feature in features:
        heavy_values = numeric_values(grouped["heavy"], feature)
        light_values = numeric_values(grouped["light"], feature)
        effect_size = cohen_effect_size(heavy_values, light_values)
        for strength in ("heavy", "light"):
            values = numeric_values(grouped[strength], feature)
            stats = describe(values)
            rows.append({
                "feature": feature,
                "feature_group": feature_group(feature),
                "touch_strength": strength,
                "count": stats["count"],
                "mean": format_number(stats["mean"]),
                "median": format_number(stats["median"]),
                "std": format_number(stats["std"]),
                "p10": format_number(stats["p10"]),
                "p25": format_number(stats["p25"]),
                "p75": format_number(stats["p75"]),
                "p90": format_number(stats["p90"]),
                "min": format_number(stats["min"]),
                "max": format_number(stats["max"]),
                "zero_count": stats["zeroCount"],
                "abs_mean_effect_size": format_number(effect_size),
            })
    return rows


def build_threshold_rows(feature_rows, features):
    rows = []
    for feature in features:
        values = []
        for row in feature_rows:
            strength = row["touch_strength"]
            value = finite_float(row.get(feature), None)
            if strength in ("heavy", "light") and value is not None:
                values.append((value, strength))
        best = best_threshold(values)
        best["feature"] = feature
        best["feature_group"] = feature_group(feature)
        rows.append(best)
    rows.sort(key=lambda item: (item["balanced_accuracy"], item["min_recall"], item["effect_size"]), reverse=True)
    for row in rows:
        for key in ("threshold", "balanced_accuracy", "heavy_recall", "light_recall", "min_recall", "effect_size"):
            row[key] = format_number(row[key])
    return rows


def best_threshold(values):
    if not values:
        return empty_threshold()
    unique_values = sorted(set(value for value, _ in values))
    thresholds = threshold_candidates(unique_values)
    heavy_values = [value for value, strength in values if strength == "heavy"]
    light_values = [value for value, strength in values if strength == "light"]
    effect_size = cohen_effect_size(heavy_values, light_values)
    best = None
    for direction in ("higher_is_heavy", "lower_is_heavy"):
        for threshold in thresholds:
            metrics = threshold_metrics(values, threshold, direction)
            metrics["direction"] = direction
            metrics["threshold"] = threshold
            metrics["effect_size"] = effect_size
            if best is None or threshold_rank(metrics) > threshold_rank(best):
                best = metrics
    return best if best is not None else empty_threshold()


def threshold_candidates(unique_values):
    if len(unique_values) == 1:
        value = unique_values[0]
        step = max(abs(value) * 0.001, 1e-9)
        return [value - step, value, value + step]
    thresholds = [unique_values[0]]
    for index in range(1, len(unique_values)):
        thresholds.append((unique_values[index - 1] + unique_values[index]) * 0.5)
    thresholds.append(unique_values[-1])
    return thresholds


def threshold_metrics(values, threshold, direction):
    heavy_total = 0
    light_total = 0
    heavy_hit = 0
    light_hit = 0
    for value, strength in values:
        if direction == "higher_is_heavy":
            predicted = "heavy" if value >= threshold else "light"
        else:
            predicted = "heavy" if value <= threshold else "light"
        if strength == "heavy":
            heavy_total += 1
            if predicted == "heavy":
                heavy_hit += 1
        elif strength == "light":
            light_total += 1
            if predicted == "light":
                light_hit += 1
    heavy_recall = heavy_hit / heavy_total if heavy_total else 0
    light_recall = light_hit / light_total if light_total else 0
    return {
        "balanced_accuracy": (heavy_recall + light_recall) * 0.5,
        "heavy_recall": heavy_recall,
        "light_recall": light_recall,
        "min_recall": min(heavy_recall, light_recall),
        "heavy_count": heavy_total,
        "light_count": light_total,
    }


def threshold_rank(metrics):
    return (
        metrics["balanced_accuracy"],
        metrics["min_recall"],
        metrics["heavy_recall"],
        -abs(metrics["threshold"]),
    )


def empty_threshold():
    return {
        "threshold": 0,
        "direction": "none",
        "balanced_accuracy": 0,
        "heavy_recall": 0,
        "light_recall": 0,
        "min_recall": 0,
        "heavy_count": 0,
        "light_count": 0,
        "effect_size": 0,
    }


def describe(values):
    values = sorted(values)
    count = len(values)
    if count == 0:
        return {
            "count": 0,
            "mean": 0,
            "median": 0,
            "std": 0,
            "p10": 0,
            "p25": 0,
            "p75": 0,
            "p90": 0,
            "min": 0,
            "max": 0,
            "zeroCount": 0,
        }
    mean_value = sum(values) / count
    return {
        "count": count,
        "mean": mean_value,
        "median": percentile(values, 0.50),
        "std": stddev(values, mean_value),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "min": values[0],
        "max": values[-1],
        "zeroCount": sum(1 for value in values if value == 0),
    }


def numeric_values(rows, key):
    values = []
    for row in rows:
        value = finite_float(row.get(key), None)
        if value is not None:
            values.append(value)
    return values


def percentile(values, ratio):
    if not values:
        return 0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * ratio
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def stddev(values, mean_value):
    if not values:
        return 0
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(max(0, variance))


def cohen_effect_size(heavy_values, light_values):
    if not heavy_values or not light_values:
        return 0
    heavy_mean = sum(heavy_values) / len(heavy_values)
    light_mean = sum(light_values) / len(light_values)
    heavy_std = stddev(heavy_values, heavy_mean)
    light_std = stddev(light_values, light_mean)
    pooled = math.sqrt((heavy_std ** 2 + light_std ** 2) * 0.5)
    if pooled <= 1e-12:
        return 0
    return abs(heavy_mean - light_mean) / pooled


def feature_group(feature):
    if feature in PATCH_FEATURES:
        return "enhanced_patch22_input"
    return "normalization_diagnostic"


def feature_presets_report():
    return {
        "old13": {
            "channelCount": len(DEFAULT_FEATURES),
            "features": list(DEFAULT_FEATURES),
        },
        "patch9": {
            "channelCount": len(PATCH_FEATURES),
            "features": list(PATCH_FEATURES),
        },
        "enhanced_patch22": {
            "channelCount": len(FEATURE_PRESETS["enhanced_patch22"]),
            "features": list(FEATURE_PRESETS["enhanced_patch22"]),
        },
    }


def finite_float(value, fallback):
    try:
        output = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(output):
        return fallback
    return output


def format_number(value):
    numeric = finite_float(value, None)
    if numeric is None:
        return ""
    return f"{numeric:.10g}"


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
