#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


MAP_SIZE = 9
GRID_RADIUS_PX = 48.0
DEFAULT_SIGMA_PX = 12.0

SCHEME_FEATURES = {
    "gaussian_map": [
        "gaussian.center",
        "gaussian.sum",
        "gaussian.energy",
        "gaussian.spread",
        "gaussian.sigma_x_px",
        "gaussian.sigma_y_px",
        "gaussian.valid_mask",
    ],
    "multi_channel": [
        "multi.center_x_norm",
        "multi.center_y_norm",
        "multi.major_px",
        "multi.minor_px",
        "multi.orientation_rad",
        "multi.valid_mask",
    ],
    "covariance_matrix": [
        "covariance.x_norm",
        "covariance.y_norm",
        "covariance.cov_11",
        "covariance.cov_12",
        "covariance.cov_22",
        "covariance.valid_mask",
    ],
}


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(manifest_path)
    rows = [encode_sample(sample) for sample in samples]
    summary_rows = build_summary_rows(rows)
    threshold_rows = build_threshold_rows(rows)

    write_csv(output_dir / "contact_encoding_samples.csv", rows)
    write_csv(output_dir / "contact_encoding_summary.csv", summary_rows)
    write_csv(output_dir / "contact_encoding_threshold_scan.csv", threshold_rows)
    report = {
        "manifest": str(manifest_path),
        "sampleCount": len(rows),
        "labelCounts": dict(Counter(row["touch_strength"] for row in rows)),
        "schemeFeatures": SCHEME_FEATURES,
        "encodingRules": {
            "gaussian_map": "touch contact map with 9x9 elliptical gaussian and valid mask",
            "multi_channel": "center heatmap plus major/minor/orientation/valid mask scalar channels",
            "covariance_matrix": "x, y, cov_11, cov_12, cov_22, valid mask",
            "unknownRadius": "sigma defaults to 12px and valid_mask becomes 0",
        },
        "topThresholdFeatures": threshold_rows[:12],
    }
    write_json(output_dir / "contact_encoding_report.json", report)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_samples(manifest_path):
    samples = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            strength = row.get("touch_strength", "")
            if strength not in ("heavy", "light"):
                continue
            data_path = resolve_path(row.get("path") or "", manifest_path)
            if data_path is None:
                continue
            with data_path.open("r", encoding="utf-8") as data_file:
                data = json.load(data_file)
            samples.append({
                "path": str(data_path),
                "fileName": data_path.name,
                "id": str(data.get("id") or row.get("id") or ""),
                "strength": strength,
                "data": data,
            })
    return samples


def resolve_path(value, manifest_path):
    if not value:
        return None
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


def encode_sample(sample):
    stats = collect_contact_stats(sample["data"])
    gaussian = gaussian_map(stats)
    covariance = covariance_values(stats)
    row = {
        "id": sample["id"],
        "file_name": sample["fileName"],
        "touch_strength": sample["strength"],
        "source": stats["source"],
        "valid": str(bool(stats["valid"])).lower(),
        "used_fallback": str(bool(stats["usedFallback"])).lower(),
        "raw_major_px": format_number(stats["rawMajorPx"]),
        "raw_minor_px": format_number(stats["rawMinorPx"]),
        "sigma_x_px": format_number(stats["sigmaXPx"]),
        "sigma_y_px": format_number(stats["sigmaYPx"]),
        "orientation_rad": format_number(stats["orientationRad"]),
    }
    row.update({
        "gaussian.center": format_number(gaussian[(MAP_SIZE * MAP_SIZE) // 2]),
        "gaussian.sum": format_number(sum(gaussian)),
        "gaussian.energy": format_number(sum(value * value for value in gaussian)),
        "gaussian.spread": format_number(weighted_spread(gaussian)),
        "gaussian.sigma_x_px": format_number(stats["sigmaXPx"]),
        "gaussian.sigma_y_px": format_number(stats["sigmaYPx"]),
        "gaussian.valid_mask": format_number(stats["validMask"]),
        "multi.center_x_norm": format_number(stats["centerXNorm"]),
        "multi.center_y_norm": format_number(stats["centerYNorm"]),
        "multi.major_px": format_number(stats["rawMajorPx"]),
        "multi.minor_px": format_number(stats["rawMinorPx"]),
        "multi.orientation_rad": format_number(stats["orientationRad"]),
        "multi.valid_mask": format_number(stats["validMask"]),
        "covariance.x_norm": format_number(stats["centerXNorm"]),
        "covariance.y_norm": format_number(stats["centerYNorm"]),
        "covariance.cov_11": format_number(covariance[0]),
        "covariance.cov_12": format_number(covariance[1]),
        "covariance.cov_22": format_number(covariance[2]),
        "covariance.valid_mask": format_number(stats["validMask"]),
    })
    return row


def collect_contact_stats(data):
    frames = data.get("touchFrames") or []
    x_norm_sum = 0.0
    y_norm_sum = 0.0
    count = 0
    raw_major = 0.0
    raw_minor = 0.0
    orientation = 0.0
    width = 0.0
    height = 0.0
    source = "unavailable"
    valid = False
    has_minor = False
    for frame in frames:
        count += 1
        x_norm = finite_float(frame.get("x_norm"), 0.0)
        y_norm = finite_float(frame.get("y_norm"), 0.0)
        x_norm_sum += x_norm
        y_norm_sum += y_norm
        width = max(width, infer_axis_size(frame.get("x"), x_norm))
        height = max(height, infer_axis_size(frame.get("y"), y_norm))
        major = first_positive(frame.get("touchMajor"), 0.0)
        minor = first_positive(frame.get("touchMinor"), 0.0)
        if major > 0:
            raw_major = max(raw_major, major)
            source = "ellipse_major_minor" if finite_float(frame.get("touchMajor"), 0.0) > 0 else "missing_touch_major"
            valid = True
        if minor > 0:
            raw_minor = max(raw_minor, minor)
            has_minor = True
        frame_orientation = finite_float(frame.get("orientation"), 0.0)
        if abs(frame_orientation) > 0.0001:
            orientation = frame_orientation

    patch = data.get("contactPatch") or {}
    patch_raw = patch.get("raw") or {}
    if raw_major <= 0 and finite_float(patch_raw.get("majorPx"), 0.0) > 0:
        raw_major = finite_float(patch_raw.get("majorPx"), 0.0)
        raw_minor = finite_float(patch_raw.get("minorPx"), 0.0)
        orientation = finite_float(patch_raw.get("orientationRad"), 0.0)
        source = str(patch.get("source") or "contact_patch")
        valid = bool(patch.get("valid"))
        has_minor = raw_minor > 0

    center_x = x_norm_sum / count if count else finite_float(patch.get("x_norm"), 0.0)
    center_y = y_norm_sum / count if count else finite_float(patch.get("y_norm"), 0.0)
    used_fallback = False
    if raw_major <= 0:
        raw_major = DEFAULT_SIGMA_PX * 2.0
        raw_minor = DEFAULT_SIGMA_PX * 2.0
        source = "unknown_default"
        used_fallback = True
        valid = False
    elif raw_minor <= 0:
        raw_minor = raw_major
        source = "radius_or_major_circle"
    if valid and has_minor:
        sigma_x = max(1.0, raw_major * 0.5)
        sigma_y = max(1.0, raw_minor * 0.5)
    elif valid:
        sigma_x = max(1.0, raw_major * 0.5)
        sigma_y = sigma_x
    else:
        sigma_x = DEFAULT_SIGMA_PX
        sigma_y = DEFAULT_SIGMA_PX
    return {
        "centerXNorm": clamp01(center_x),
        "centerYNorm": clamp01(center_y),
        "rawMajorPx": raw_major,
        "rawMinorPx": raw_minor,
        "sigmaXPx": sigma_x,
        "sigmaYPx": sigma_y,
        "orientationRad": orientation,
        "width": max(1.0, width),
        "height": max(1.0, height),
        "valid": valid,
        "validMask": 1.0 if valid else 0.0,
        "usedFallback": used_fallback,
        "source": source,
    }


def gaussian_map(stats):
    values = []
    center = MAP_SIZE // 2
    cos_value = math.cos(stats["orientationRad"])
    sin_value = math.sin(stats["orientationRad"])
    for y_index in range(MAP_SIZE):
        for x_index in range(MAP_SIZE):
            dx = ((x_index - center) / center) * GRID_RADIUS_PX
            dy = ((y_index - center) / center) * GRID_RADIUS_PX
            rotated_x = cos_value * dx + sin_value * dy
            rotated_y = -sin_value * dx + cos_value * dy
            exponent = -0.5 * (
                (rotated_x * rotated_x) / max(1.0, stats["sigmaXPx"] * stats["sigmaXPx"])
                + (rotated_y * rotated_y) / max(1.0, stats["sigmaYPx"] * stats["sigmaYPx"])
            )
            values.append(math.exp(exponent))
    return values


def covariance_values(stats):
    sigma_x_norm = stats["sigmaXPx"] / max(1.0, stats["width"])
    sigma_y_norm = stats["sigmaYPx"] / max(1.0, stats["height"])
    cos_value = math.cos(stats["orientationRad"])
    sin_value = math.sin(stats["orientationRad"])
    variance_x = sigma_x_norm * sigma_x_norm
    variance_y = sigma_y_norm * sigma_y_norm
    cov_11 = cos_value * cos_value * variance_x + sin_value * sin_value * variance_y
    cov_22 = sin_value * sin_value * variance_x + cos_value * cos_value * variance_y
    cov_12 = sin_value * cos_value * (variance_x - variance_y)
    return cov_11, cov_12, cov_22


def weighted_spread(values):
    total = sum(values)
    if total <= 0:
        return 0.0
    center = MAP_SIZE // 2
    spread = 0.0
    index = 0
    for y_index in range(MAP_SIZE):
        for x_index in range(MAP_SIZE):
            dx = x_index - center
            dy = y_index - center
            spread += values[index] * math.sqrt(dx * dx + dy * dy)
            index += 1
    return spread / total


def build_summary_rows(rows):
    output = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["touch_strength"]].append(row)
    for scheme, features in SCHEME_FEATURES.items():
        for feature in features:
            heavy = numeric_values(grouped["heavy"], feature)
            light = numeric_values(grouped["light"], feature)
            effect = effect_size(heavy, light)
            for strength in ("heavy", "light"):
                values = numeric_values(grouped[strength], feature)
                stats = describe(values)
                output.append({
                    "scheme": scheme,
                    "feature": feature,
                    "touch_strength": strength,
                    "count": stats["count"],
                    "mean": format_number(stats["mean"]),
                    "median": format_number(stats["median"]),
                    "std": format_number(stats["std"]),
                    "p10": format_number(stats["p10"]),
                    "p90": format_number(stats["p90"]),
                    "effect_size": format_number(effect),
                })
    return output


def build_threshold_rows(rows):
    output = []
    for scheme, features in SCHEME_FEATURES.items():
        for feature in features:
            values = []
            for row in rows:
                value = finite_float(row.get(feature), None)
                strength = row.get("touch_strength")
                if value is not None and strength in ("heavy", "light"):
                    values.append((value, strength))
            best = best_threshold(values)
            best["scheme"] = scheme
            best["feature"] = feature
            output.append(best)
    output.sort(key=lambda item: (item["balanced_accuracy"], item["min_recall"], item["effect_size"]), reverse=True)
    for row in output:
        for key in ("threshold", "balanced_accuracy", "heavy_recall", "light_recall", "min_recall", "effect_size"):
            row[key] = format_number(row[key])
    return output


def best_threshold(values):
    if not values:
        return empty_threshold()
    unique = sorted(set(value for value, _ in values))
    thresholds = threshold_candidates(unique)
    heavy = [value for value, strength in values if strength == "heavy"]
    light = [value for value, strength in values if strength == "light"]
    effect = effect_size(heavy, light)
    best = None
    for direction in ("higher_is_heavy", "lower_is_heavy"):
        for threshold in thresholds:
            metrics = threshold_metrics(values, threshold, direction)
            metrics["threshold"] = threshold
            metrics["direction"] = direction
            metrics["effect_size"] = effect
            if best is None or rank(metrics) > rank(best):
                best = metrics
    return best if best is not None else empty_threshold()


def threshold_candidates(values):
    if len(values) == 1:
        step = max(abs(values[0]) * 0.001, 1e-9)
        return [values[0] - step, values[0], values[0] + step]
    output = [values[0]]
    for index in range(1, len(values)):
        output.append((values[index - 1] + values[index]) * 0.5)
    output.append(values[-1])
    return output


def threshold_metrics(values, threshold, direction):
    heavy_total = 0
    light_total = 0
    heavy_hit = 0
    light_hit = 0
    for value, strength in values:
        if direction == "higher_is_heavy":
            prediction = "heavy" if value >= threshold else "light"
        else:
            prediction = "heavy" if value <= threshold else "light"
        if strength == "heavy":
            heavy_total += 1
            heavy_hit += 1 if prediction == "heavy" else 0
        elif strength == "light":
            light_total += 1
            light_hit += 1 if prediction == "light" else 0
    heavy_recall = heavy_hit / heavy_total if heavy_total else 0.0
    light_recall = light_hit / light_total if light_total else 0.0
    return {
        "heavy_count": heavy_total,
        "light_count": light_total,
        "heavy_recall": heavy_recall,
        "light_recall": light_recall,
        "balanced_accuracy": (heavy_recall + light_recall) * 0.5,
        "min_recall": min(heavy_recall, light_recall),
    }


def empty_threshold():
    return {
        "heavy_count": 0,
        "light_count": 0,
        "heavy_recall": 0,
        "light_recall": 0,
        "balanced_accuracy": 0,
        "min_recall": 0,
        "threshold": 0,
        "direction": "none",
        "effect_size": 0,
    }


def rank(metrics):
    return metrics["balanced_accuracy"], metrics["min_recall"], metrics["heavy_recall"]


def describe(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "mean": 0, "median": 0, "std": 0, "p10": 0, "p90": 0}
    mean_value = sum(values) / len(values)
    return {
        "count": len(values),
        "mean": mean_value,
        "median": percentile(values, 0.5),
        "std": stddev(values, mean_value),
        "p10": percentile(values, 0.1),
        "p90": percentile(values, 0.9),
    }


def numeric_values(rows, key):
    output = []
    for row in rows:
        value = finite_float(row.get(key), None)
        if value is not None:
            output.append(value)
    return output


def percentile(values, ratio):
    if not values:
        return 0.0
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
        return 0.0
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))


def effect_size(heavy, light):
    if not heavy or not light:
        return 0.0
    heavy_mean = sum(heavy) / len(heavy)
    light_mean = sum(light) / len(light)
    heavy_std = stddev(heavy, heavy_mean)
    light_std = stddev(light, light_mean)
    pooled = math.sqrt((heavy_std * heavy_std + light_std * light_std) * 0.5)
    if pooled <= 1e-12:
        return 0.0
    return abs(heavy_mean - light_mean) / pooled


def infer_axis_size(position, normalized):
    position_value = finite_float(position, 0.0)
    normalized_value = finite_float(normalized, 0.0)
    if position_value > 0 and normalized_value > 0.000001:
        return position_value / normalized_value
    return 0.0


def first_positive(first, second):
    first_value = finite_float(first, 0.0)
    if first_value > 0:
        return first_value
    second_value = finite_float(second, 0.0)
    if second_value > 0:
        return second_value
    return 0.0


def finite_float(value, fallback):
    try:
        output = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(output):
        return fallback
    return output


def clamp01(value):
    return max(0.0, min(1.0, value))


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
