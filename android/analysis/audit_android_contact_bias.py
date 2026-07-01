#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter
from collections import defaultdict
from pathlib import Path


NUMERIC_FIELDS = [
    "raw_major_px",
    "raw_minor_px",
    "max_touch_major",
    "max_touch_minor",
    "mean_touch_major",
    "mean_touch_minor",
    "norm_area",
    "baseline_area",
    "max_abs_delta",
    "impact_mean_abs_delta",
    "touch_frame_count",
    "duration_ms",
    "path_travel_px",
]

SAMPLE_FIELDS = [
    "created_at",
    "age_from_latest_sec",
    "quality_bucket",
    "quality_score",
    "quality_reasons",
    "label_id",
    "finger_type",
    "touch_strength",
    "cnn_decision",
    "cnn_heavy_probability",
    "cnn_margin",
    "raw_major_px",
    "raw_minor_px",
    "max_touch_major",
    "max_touch_minor",
    "norm_area",
    "max_abs_delta",
    "impact_mean_abs_delta",
    "duration_ms",
    "touch_frame_count",
    "path_travel_px",
    "file_name",
    "path",
    "id",
]


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(manifest_path)
    latest = max((row["created_at"] for row in rows), default=0.0)
    cutoff = latest - args.recent_seconds
    for row in rows:
        row["age_from_latest_sec"] = latest - row["created_at"]

    high_contact_thresholds = build_high_contact_thresholds(rows, args.high_contact_quantile)
    for row in rows:
        row["high_contact_light"] = is_high_contact_light(row, high_contact_thresholds)

    recent_rows = [row for row in rows if row["created_at"] >= cutoff]
    selected_rows = [row for row in rows if row.get("quality_bucket") == "high_quality"]
    recent_selected_rows = [row for row in recent_rows if row.get("quality_bucket") == "high_quality"]
    report = {
        "manifest": str(manifest_path),
        "outputDir": str(output_dir),
        "latestCreatedAt": latest,
        "recentCutoffCreatedAt": cutoff,
        "recentSeconds": args.recent_seconds,
        "highContactQuantile": args.high_contact_quantile,
        "highContactThresholds": high_contact_thresholds,
        "all": dataset_summary(rows),
        "highQuality": dataset_summary(selected_rows),
        "recent": dataset_summary(recent_rows),
        "recentHighQuality": dataset_summary(recent_selected_rows),
        "correlations": {
            "all": correlation_summary(rows),
            "highQuality": correlation_summary(selected_rows),
            "recent": correlation_summary(recent_rows),
            "recentHighQuality": correlation_summary(recent_selected_rows),
        },
    }

    write_json(output_dir / "contact_bias_audit_report.json", report)
    write_csv(output_dir / "recent_samples.csv", sorted(recent_rows, key=lambda row: row["created_at"]))
    write_csv(output_dir / "recent_high_quality_samples.csv", sorted(recent_selected_rows, key=lambda row: row["created_at"]))
    write_csv(
        output_dir / "light_large_contact_false_heavy.csv",
        sorted(
            [
                row for row in rows
                if row.get("touch_strength") == "light"
                and prediction_strength(row) == "heavy"
                and row.get("high_contact_light")
            ],
            key=lambda row: (-finite_float(row.get("cnn_heavy_probability"), 0.0), -finite_float(row.get("raw_major_px"), 0.0)),
        ),
    )
    write_csv(
        output_dir / "recent_light_large_contact_false_heavy.csv",
        sorted(
            [
                row for row in recent_rows
                if row.get("touch_strength") == "light"
                and prediction_strength(row) == "heavy"
                and row.get("high_contact_light")
            ],
            key=lambda row: (-finite_float(row.get("cnn_heavy_probability"), 0.0), -finite_float(row.get("raw_major_px"), 0.0)),
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--recent-seconds", type=float, default=300.0)
    parser.add_argument("--high-contact-quantile", type=float, default=0.75)
    return parser.parse_args()


def load_rows(path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            clean = dict(row)
            for key in ["created_at", "quality_score", "cnn_heavy_probability", "cnn_light_probability", "cnn_margin"] + NUMERIC_FIELDS:
                clean[key] = finite_float(clean.get(key), 0.0)
            rows.append(clean)
    return rows


def build_high_contact_thresholds(rows, fraction):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("touch_strength") != "light":
            continue
        finger = row.get("finger_type") or "unknown"
        for key in ("raw_major_px", "raw_minor_px", "norm_area", "max_touch_major", "max_touch_minor"):
            value = finite_float(row.get(key), None)
            if value is not None and value > 0:
                grouped[finger][key].append(value)
    thresholds = {}
    for finger, fields in grouped.items():
        thresholds[finger] = {}
        for key, values in fields.items():
            thresholds[finger][key] = percentile(values, fraction)
    return thresholds


def is_high_contact_light(row, thresholds):
    if row.get("touch_strength") != "light":
        return False
    finger = row.get("finger_type") or "unknown"
    fields = thresholds.get(finger, {})
    hits = 0
    for key in ("raw_major_px", "raw_minor_px", "norm_area", "max_touch_major", "max_touch_minor"):
        threshold = finite_float(fields.get(key), 0.0)
        value = finite_float(row.get(key), 0.0)
        if threshold > 0 and value >= threshold:
            hits += 1
    return hits >= 2


def dataset_summary(rows):
    label_counts = Counter(row.get("label_id") or "unknown" for row in rows)
    strength_counts = Counter(row.get("touch_strength") or "unknown" for row in rows)
    quality_counts = Counter(row.get("quality_bucket") or "unknown" for row in rows)
    prediction_counts = Counter(row.get("cnn_decision") or "unknown" for row in rows)
    false_heavy = [
        row for row in rows
        if row.get("touch_strength") == "light" and prediction_strength(row) == "heavy"
    ]
    false_light = [
        row for row in rows
        if row.get("touch_strength") == "heavy" and prediction_strength(row) == "light"
    ]
    light_rows = [row for row in rows if row.get("touch_strength") == "light"]
    heavy_rows = [row for row in rows if row.get("touch_strength") == "heavy"]
    high_contact_light_rows = [row for row in light_rows if row.get("high_contact_light")]
    high_contact_false_heavy = [row for row in false_heavy if row.get("high_contact_light")]
    normal_contact_light_rows = [row for row in light_rows if not row.get("high_contact_light")]
    normal_contact_false_heavy = [row for row in false_heavy if not row.get("high_contact_light")]
    return {
        "count": len(rows),
        "labelCounts": dict(sorted(label_counts.items())),
        "strengthCounts": dict(sorted(strength_counts.items())),
        "qualityCounts": dict(sorted(quality_counts.items())),
        "predictionCounts": dict(sorted(prediction_counts.items())),
        "confusion": confusion(rows),
        "falseHeavyCount": len(false_heavy),
        "falseHeavyRateAmongLight": rate(len(false_heavy), len(light_rows)),
        "falseLightCount": len(false_light),
        "falseLightRateAmongHeavy": rate(len(false_light), len(heavy_rows)),
        "highContactLightCount": len(high_contact_light_rows),
        "highContactLightRate": rate(len(high_contact_light_rows), len(light_rows)),
        "highContactLightFalseHeavyCount": len(high_contact_false_heavy),
        "highContactLightFalseHeavyRate": rate(len(high_contact_false_heavy), len(high_contact_light_rows)),
        "normalContactLightFalseHeavyCount": len(normal_contact_false_heavy),
        "normalContactLightFalseHeavyRate": rate(len(normal_contact_false_heavy), len(normal_contact_light_rows)),
        "numeric": numeric_summary(rows),
        "lightNumeric": numeric_summary(light_rows),
        "heavyNumeric": numeric_summary(heavy_rows),
        "falseHeavyNumeric": numeric_summary(false_heavy),
        "highContactFalseHeavyTop": top_sample_summary(high_contact_false_heavy, 12),
    }


def confusion(rows):
    output = {
        "heavy_as_heavy": 0,
        "heavy_as_light": 0,
        "light_as_heavy": 0,
        "light_as_light": 0,
        "other": 0,
    }
    for row in rows:
        strength = row.get("touch_strength")
        decision = prediction_strength(row)
        if strength == "heavy" and decision == "heavy":
            output["heavy_as_heavy"] += 1
        elif strength == "heavy" and decision == "light":
            output["heavy_as_light"] += 1
        elif strength == "light" and decision == "heavy":
            output["light_as_heavy"] += 1
        elif strength == "light" and decision == "light":
            output["light_as_light"] += 1
        else:
            output["other"] += 1
    return output


def prediction_strength(row):
    decision = str(row.get("cnn_decision") or "")
    if decision in ("heavy", "light"):
        return decision
    if decision.endswith("_heavy"):
        return "heavy"
    if decision.endswith("_light"):
        return "light"
    return ""


def numeric_summary(rows):
    summary = {}
    for key in NUMERIC_FIELDS + ["cnn_heavy_probability", "cnn_margin"]:
        values = [finite_float(row.get(key), None) for row in rows]
        values = [value for value in values if value is not None]
        summary[key] = {
            "count": len(values),
            "mean": mean(values),
            "p50": percentile(values, 0.50),
            "p75": percentile(values, 0.75),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
        }
    return summary


def correlation_summary(rows):
    output = {}
    y = [finite_float(row.get("cnn_heavy_probability"), None) for row in rows]
    for key in NUMERIC_FIELDS:
        x = [finite_float(row.get(key), None) for row in rows]
        output[key] = pearson(x, y)
    return dict(sorted(output.items(), key=lambda item: abs(item[1]), reverse=True))


def top_sample_summary(rows, limit):
    output = []
    ordered = sorted(rows, key=lambda row: (-finite_float(row.get("cnn_heavy_probability"), 0.0), -finite_float(row.get("raw_major_px"), 0.0)))
    for row in ordered[:limit]:
        output.append({
            "created_at": row.get("created_at", 0.0),
            "age_from_latest_sec": row.get("age_from_latest_sec", 0.0),
            "label_id": row.get("label_id", ""),
            "cnn_decision": row.get("cnn_decision", ""),
            "cnn_heavy_probability": row.get("cnn_heavy_probability", 0.0),
            "raw_major_px": row.get("raw_major_px", 0.0),
            "raw_minor_px": row.get("raw_minor_px", 0.0),
            "norm_area": row.get("norm_area", 0.0),
            "max_abs_delta": row.get("max_abs_delta", 0.0),
            "file_name": row.get("file_name", ""),
            "path": row.get("path", ""),
        })
    return output


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SAMPLE_FIELDS})


def write_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def percentile(values, fraction):
    clean = sorted(value for value in values if value is not None and math.isfinite(value))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def mean(values):
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return sum(clean) / len(clean) if clean else 0.0


def pearson(x_values, y_values):
    pairs = [
        (x, y) for x, y in zip(x_values, y_values)
        if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)
    ]
    if len(pairs) < 2:
        return 0.0
    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x, _ in pairs))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for _, y in pairs))
    if x_den <= 0 or y_den <= 0:
        return 0.0
    return numerator / (x_den * y_den)


def rate(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def finite_float(value, fallback):
    try:
        output = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(output):
        return fallback
    return output


if __name__ == "__main__":
    main()
