#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


GRID_SIZE = 32


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path, args.bucket)
    samples = []
    for row in rows:
        path = Path(row["path"])
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        samples.append(make_sample(row, data))

    defaults = compute_defaults(samples)
    for sample in samples:
        encode_sample(sample, defaults, args.grid_size)

    write_sample_features(output_dir / "contact_patch_features.csv", samples)
    write_metrics(output_dir / "encoding_metrics.csv", output_dir / "encoding_summary.json", samples, defaults)
    if args.emit_arrays:
        write_arrays(output_dir / "contact_patch_arrays.jsonl", samples)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bucket", default="high_quality")
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--emit-arrays", action="store_true")
    return parser.parse_args()


def load_manifest(path, bucket):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if bucket and row.get("quality_bucket") != bucket:
                continue
            strength = row.get("touch_strength")
            if strength not in ("light", "heavy"):
                continue
            rows.append(row)
    return rows


def make_sample(row, data):
    frames = data.get("touchFrames") or []
    if not frames:
        frames = []
    latest = frames[-1] if frames else {}
    x = finite_float(latest.get("x"), finite_float(row.get("x"), 0.0))
    y = finite_float(latest.get("y"), finite_float(row.get("y"), 0.0))
    stored_x_norm = finite_float(latest.get("x_norm"), 0.0)
    stored_y_norm = finite_float(latest.get("y_norm"), 0.0)
    width = infer_axis_size(x, stored_x_norm, finite_float(latest.get("width"), 0.0))
    height = infer_axis_size(y, stored_y_norm, finite_float(latest.get("height"), 0.0))
    x_norm = clamp01(x / max(1.0, width))
    y_norm = clamp01(y / max(1.0, height))
    major = 0.0
    minor = 0.0
    orientation = 0.0
    for frame in frames:
        frame_major = max(
            0.0,
            finite_float(frame.get("touchMajor"), 0.0),
        )
        frame_minor = max(
            0.0,
            finite_float(frame.get("touchMinor"), 0.0),
        )
        if frame_major > major:
            major = frame_major
            minor = frame_minor
            orientation = finite_float(frame.get("orientation"), 0.0)
    return {
        "id": row.get("id", ""),
        "file_name": row.get("file_name", ""),
        "path": row.get("path", ""),
        "label_id": row.get("label_id", ""),
        "touch_strength": row.get("touch_strength", ""),
        "label": 1 if row.get("touch_strength") == "heavy" else 0,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "width": width,
        "height": height,
        "major_px": major,
        "minor_px": minor,
        "orientation_rad": orientation,
        "source": contact_source(major, minor),
    }


def compute_defaults(samples):
    sigma_x_values = []
    sigma_y_values = []
    major_values = []
    minor_values = []
    for sample in samples:
        major = sample["major_px"]
        minor = sample["minor_px"]
        if major > 0 and minor > 0:
            sigma_x_values.append(max(1.0, major * 0.5))
            sigma_y_values.append(max(1.0, minor * 0.5))
            major_values.append(major)
            minor_values.append(minor)
    default_sigma = median(sigma_x_values + sigma_y_values) if sigma_x_values or sigma_y_values else 12.0
    default_major = median(major_values) if major_values else default_sigma * 2.0
    default_minor = median(minor_values) if minor_values else default_sigma * 2.0
    return {
        "sigma_x_px": default_sigma,
        "sigma_y_px": default_sigma,
        "major_px": default_major,
        "minor_px": default_minor,
    }


def encode_sample(sample, defaults, grid_size):
    width = max(1.0, sample["width"])
    height = max(1.0, sample["height"])
    major = sample["major_px"]
    minor = sample["minor_px"]
    orientation = sample["orientation_rad"]
    valid = 1.0 if major > 0 else 0.0
    if major > 0 and minor > 0:
        sigma_x_px = max(1.0, major * 0.5)
        sigma_y_px = max(1.0, minor * 0.5)
        axis_major = major
        axis_minor = minor
    elif major > 0:
        sigma_x_px = max(1.0, major * 0.5)
        sigma_y_px = sigma_x_px
        axis_major = major
        axis_minor = major
        orientation = 0.0
    else:
        sigma_x_px = defaults["sigma_x_px"]
        sigma_y_px = defaults["sigma_y_px"]
        axis_major = defaults["major_px"]
        axis_minor = defaults["minor_px"]
        orientation = 0.0

    sigma_x_norm = sigma_x_px / width
    sigma_y_norm = sigma_y_px / height
    cov11, cov12, cov22 = covariance(sigma_x_norm, sigma_y_norm, orientation)
    gaussian_map = gaussian_grid(sample["x_norm"], sample["y_norm"], cov11, cov12, cov22, grid_size)
    gaussian_sum = sum(gaussian_map)
    covariance_area = math.pi * math.sqrt(max(0.0, cov11 * cov22 - cov12 * cov12))
    multichannel_area = (axis_major / width) * (axis_minor / height)
    sample["gaussian"] = {
        "grid_size": grid_size,
        "valid": valid,
        "sigma_x_px": sigma_x_px,
        "sigma_y_px": sigma_y_px,
        "sigma_x_norm": sigma_x_norm,
        "sigma_y_norm": sigma_y_norm,
        "sum": gaussian_sum,
        "peak": max(gaussian_map) if gaussian_map else 0.0,
        "map": gaussian_map,
    }
    sample["multichannel"] = {
        "center_x": sample["x_norm"],
        "center_y": sample["y_norm"],
        "major_norm": axis_major / width,
        "minor_norm": axis_minor / height,
        "orientation": orientation,
        "valid": valid,
        "area": multichannel_area,
    }
    sample["covariance"] = {
        "x": sample["x_norm"],
        "y": sample["y_norm"],
        "cov11": cov11,
        "cov12": cov12,
        "cov22": cov22,
        "valid": valid,
        "area": covariance_area,
    }


def covariance(sigma_x, sigma_y, theta):
    cos_v = math.cos(theta)
    sin_v = math.sin(theta)
    var_x = sigma_x * sigma_x
    var_y = sigma_y * sigma_y
    cov11 = cos_v * cos_v * var_x + sin_v * sin_v * var_y
    cov22 = sin_v * sin_v * var_x + cos_v * cos_v * var_y
    cov12 = sin_v * cos_v * (var_x - var_y)
    return cov11, cov12, cov22


def gaussian_grid(x_center, y_center, cov11, cov12, cov22, grid_size):
    det = cov11 * cov22 - cov12 * cov12
    if det <= 1e-18:
        return [0.0] * (grid_size * grid_size)
    inv11 = cov22 / det
    inv12 = -cov12 / det
    inv22 = cov11 / det
    values = []
    for row in range(grid_size):
        y = (row + 0.5) / grid_size
        dy = y - y_center
        for col in range(grid_size):
            x = (col + 0.5) / grid_size
            dx = x - x_center
            exponent = -0.5 * (inv11 * dx * dx + 2.0 * inv12 * dx * dy + inv22 * dy * dy)
            values.append(math.exp(max(-60.0, exponent)))
    return values


def write_sample_features(path, samples):
    fields = [
        "id",
        "file_name",
        "label_id",
        "touch_strength",
        "source",
        "x_norm",
        "y_norm",
        "major_px",
        "minor_px",
        "orientation_rad",
        "gaussian_valid",
        "gaussian_sigma_x_px",
        "gaussian_sigma_y_px",
        "gaussian_sum",
        "multichannel_valid",
        "multichannel_major_norm",
        "multichannel_minor_norm",
        "multichannel_orientation",
        "multichannel_area",
        "covariance_valid",
        "covariance_11",
        "covariance_12",
        "covariance_22",
        "covariance_area",
        "path",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "id": sample["id"],
                "file_name": sample["file_name"],
                "label_id": sample["label_id"],
                "touch_strength": sample["touch_strength"],
                "source": sample["source"],
                "x_norm": sample["x_norm"],
                "y_norm": sample["y_norm"],
                "major_px": sample["major_px"],
                "minor_px": sample["minor_px"],
                "orientation_rad": sample["orientation_rad"],
                "gaussian_valid": sample["gaussian"]["valid"],
                "gaussian_sigma_x_px": sample["gaussian"]["sigma_x_px"],
                "gaussian_sigma_y_px": sample["gaussian"]["sigma_y_px"],
                "gaussian_sum": sample["gaussian"]["sum"],
                "multichannel_valid": sample["multichannel"]["valid"],
                "multichannel_major_norm": sample["multichannel"]["major_norm"],
                "multichannel_minor_norm": sample["multichannel"]["minor_norm"],
                "multichannel_orientation": sample["multichannel"]["orientation"],
                "multichannel_area": sample["multichannel"]["area"],
                "covariance_valid": sample["covariance"]["valid"],
                "covariance_11": sample["covariance"]["cov11"],
                "covariance_12": sample["covariance"]["cov12"],
                "covariance_22": sample["covariance"]["cov22"],
                "covariance_area": sample["covariance"]["area"],
                "path": sample["path"],
            })


def write_metrics(csv_path, json_path, samples, defaults):
    metric_defs = [
        ("gaussian_map", "gaussian_sum", lambda sample: sample["gaussian"]["sum"]),
        ("multichannel", "axis_area", lambda sample: sample["multichannel"]["area"]),
        ("multichannel", "major_norm", lambda sample: sample["multichannel"]["major_norm"]),
        ("multichannel", "minor_norm", lambda sample: sample["multichannel"]["minor_norm"]),
        ("covariance_matrix", "covariance_area", lambda sample: sample["covariance"]["area"]),
        ("covariance_matrix", "cov11", lambda sample: sample["covariance"]["cov11"]),
        ("covariance_matrix", "cov22", lambda sample: sample["covariance"]["cov22"]),
    ]
    rows = []
    for scheme, feature, getter in metric_defs:
        values = [(getter(sample), sample["label"]) for sample in samples]
        heavy = [value for value, label in values if label == 1]
        light = [value for value, label in values if label == 0]
        best = best_balanced_accuracy(values)
        rows.append({
            "scheme": scheme,
            "feature": feature,
            "count": len(values),
            "heavy_count": len(heavy),
            "light_count": len(light),
            "valid_count": sum(1 for sample in samples if sample[scheme_key(scheme)]["valid"] > 0),
            "heavy_median": median(heavy),
            "light_median": median(light),
            "median_ratio": safe_divide(median(heavy), median(light)),
            "auc": auc(values),
            "best_balanced_accuracy": best["balanced_accuracy"],
            "threshold": best["threshold"],
            "heavy_recall": best["heavy_recall"],
            "light_recall": best["light_recall"],
        })
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    summary = {
        "sampleCount": len(samples),
        "defaultPatch": defaults,
        "metrics": rows,
        "encodingDefinitions": {
            "gaussian_map": "centered_elliptical_gaussian_grid",
            "multichannel": "center_major_minor_orientation_valid",
            "covariance_matrix": "x_y_cov11_cov12_cov22_valid",
        },
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)


def write_arrays(path, samples):
    with path.open("w", encoding="utf-8") as file:
        for sample in samples:
            row = {
                "id": sample["id"],
                "label": sample["touch_strength"],
                "gaussianMap": sample["gaussian"]["map"],
                "multichannel": sample["multichannel"],
                "covariance": sample["covariance"],
            }
            file.write(json.dumps(row, separators=(",", ":")) + "\n")


def best_balanced_accuracy(values):
    scores = sorted(set(score for score, _ in values))
    if not scores:
        return empty_metric()
    thresholds = [scores[0] - 1e-12]
    thresholds += [(left + right) * 0.5 for left, right in zip(scores, scores[1:])]
    thresholds.append(scores[-1] + 1e-12)
    best = empty_metric()
    for threshold in thresholds:
        tp = fn = tn = fp = 0
        for score, label in values:
            pred = 1 if score >= threshold else 0
            if label == 1 and pred == 1:
                tp += 1
            elif label == 1:
                fn += 1
            elif pred == 0:
                tn += 1
            else:
                fp += 1
        heavy_recall = tp / (tp + fn) if tp + fn else 0.0
        light_recall = tn / (tn + fp) if tn + fp else 0.0
        balanced = (heavy_recall + light_recall) * 0.5
        if balanced > best["balanced_accuracy"]:
            best = {
                "balanced_accuracy": balanced,
                "threshold": threshold,
                "heavy_recall": heavy_recall,
                "light_recall": light_recall,
            }
    return best


def empty_metric():
    return {
        "balanced_accuracy": 0.0,
        "threshold": 0.0,
        "heavy_recall": 0.0,
        "light_recall": 0.0,
    }


def auc(values):
    heavy = [score for score, label in values if label == 1]
    light = [score for score, label in values if label == 0]
    if not heavy or not light:
        return 0.0
    wins = 0.0
    for h_value in heavy:
        for l_value in light:
            if h_value > l_value:
                wins += 1.0
            elif h_value == l_value:
                wins += 0.5
    return wins / (len(heavy) * len(light))


def scheme_key(scheme):
    if scheme == "gaussian_map":
        return "gaussian"
    if scheme == "covariance_matrix":
        return "covariance"
    return "multichannel"


def contact_source(major, minor):
    if major > 0 and minor > 0:
        return "ellipse"
    if major > 0:
        return "diameter_only"
    return "unknown_default"


def median(values):
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return 0.0
    middle = len(clean) // 2
    if len(clean) % 2:
        return clean[middle]
    return (clean[middle - 1] + clean[middle]) * 0.5


def safe_divide(left, right):
    if not right:
        return 0.0
    return left / right


def finite_float(value, fallback):
    try:
        output = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(output):
        return fallback
    return output


def infer_axis_size(value, normalized, fallback):
    if fallback and fallback > 1:
        return fallback
    if normalized and normalized > 0:
        inferred = value / normalized
        if inferred > 1 and math.isfinite(inferred):
            return inferred
    return 1.0


def clamp01(value):
    return max(0.0, min(1.0, value))


if __name__ == "__main__":
    main()
