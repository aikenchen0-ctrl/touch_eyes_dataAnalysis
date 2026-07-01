#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from run_android_touch_ablation import evaluate
from run_android_touch_ablation import fit_normalizer
from run_android_touch_ablation import format_number
from run_android_touch_ablation import normalize_records
from run_android_touch_ablation import parse_int_list
from run_android_touch_ablation import stratified_split
from run_android_touch_ablation import train_diagonal_lda
from train_android_touch_cnn import DEFAULT_FEATURES
from train_android_touch_cnn import PATCH_FEATURES
from train_android_touch_cnn import feature_value


FEATURE_PRESETS = {
    "old13": DEFAULT_FEATURES,
    "enhanced_patch22": DEFAULT_FEATURES + PATCH_FEATURES,
}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = FEATURE_PRESETS[args.feature_preset]
    pre_values = parse_int_list(args.pre_values)
    post_values = parse_int_list(args.post_values)
    seeds = parse_int_list(args.seeds)
    frame_indices = sorted(set(index for pre in pre_values for index in range(-pre, max(post_values) + 1)))
    samples = load_samples(Path(args.manifest), feature_names, frame_indices)
    run_rows = []
    summary_rows = []

    for pre in pre_values:
        for post in post_values:
            indices = list(range(-pre, post + 1))
            if len(indices) < args.min_frame_count:
                continue
            records = build_records(samples, indices)
            for seed in seeds:
                train_records, valid_records = stratified_split(records, args.validation_ratio, seed)
                normalizer = fit_normalizer([record["vector"] for record in train_records])
                train_vectors = normalize_records(train_records, normalizer)
                valid_vectors = normalize_records(valid_records, normalizer)
                model = train_diagonal_lda(train_vectors, args.variance_floor)
                train_metrics = evaluate(model, train_vectors)
                valid_metrics = evaluate(model, valid_vectors)
                run_rows.append(make_run_row(
                    args.feature_preset,
                    pre,
                    post,
                    seed,
                    records,
                    train_records,
                    valid_records,
                    train_metrics,
                    valid_metrics,
                    len(feature_names),
                    len(indices),
                    args,
                ))
            summary_rows.append(make_summary_row(args.feature_preset, pre, post, run_rows))

    summary_rows.sort(key=lambda row: (
        float(row["valid_balanced_accuracy_mean"]),
        float(row["valid_auc_mean"]),
        float(row["valid_min_recall_mean"]),
    ), reverse=True)
    for index, row in enumerate(summary_rows, start=1):
        row["rank"] = index

    write_csv(output_dir / "window_ablation_runs.csv", run_rows)
    write_csv(output_dir / "window_ablation_summary.csv", summary_rows)
    write_json(output_dir / "window_ablation_report.json", {
        "manifest": args.manifest,
        "featurePreset": args.feature_preset,
        "sampleCount": len(samples),
        "labelCounts": dict(Counter(sample["strength"] for sample in samples)),
        "modelKind": "diagonal_lda_probe",
        "modelNote": "No filtering; missing frames are zero-filled so every raw sample can participate.",
        "seeds": seeds,
        "preValues": pre_values,
        "postValues": post_values,
        "minFrameCount": args.min_frame_count,
        "validationRatio": args.validation_ratio,
        "varianceFloor": args.variance_floor,
        "topWindows": summary_rows[:10],
    })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-preset", default="enhanced_patch22", choices=sorted(FEATURE_PRESETS.keys()))
    parser.add_argument("--pre-values", default="0,1,2,3,4,5")
    parser.add_argument("--post-values", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
    parser.add_argument("--min-frame-count", type=int, default=6)
    parser.add_argument("--validation-ratio", type=float, default=0.25)
    parser.add_argument("--seeds", default="7,11,19")
    parser.add_argument("--variance-floor", type=float, default=0.05)
    return parser.parse_args()


def load_samples(manifest_path, feature_names, frame_indices):
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
            frames_by_index = {}
            for frame in data.get("frames") or []:
                frames_by_index[int_value(frame.get("frameIndex"), 0)] = frame
            frame_values = {}
            for frame_index in frame_indices:
                frame = frames_by_index.get(frame_index, {})
                frame_values[frame_index] = [
                    finite_float(feature_value(data, frame, feature_name), 0.0)
                    for feature_name in feature_names
                ]
            samples.append({
                "id": str(data.get("id") or row.get("id") or ""),
                "fileName": data_path.name,
                "path": str(data_path),
                "strength": strength,
                "label": 1 if strength == "heavy" else 0,
                "frameValues": frame_values,
            })
    return samples


def resolve_path(row, manifest_path):
    for key in ("path", "raw_original_path", "raw_path", "chosen_path"):
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


def build_records(samples, frame_indices):
    records = []
    for sample in samples:
        vector = []
        for frame_index in frame_indices:
            vector.extend(sample["frameValues"].get(frame_index, []))
        records.append({
            "id": sample["id"],
            "path": sample["path"],
            "fileName": sample["fileName"],
            "strength": sample["strength"],
            "label": sample["label"],
            "vector": vector,
        })
    return records


def make_run_row(
        preset,
        pre,
        post,
        seed,
        records,
        train_records,
        valid_records,
        train_metrics,
        valid_metrics,
        channel_count,
        frame_count,
        args):
    return {
        "window": window_name(pre, post),
        "feature_preset": preset,
        "pre_frames": pre,
        "post_frames": post,
        "first_frame": -pre,
        "last_frame": post,
        "frame_count": frame_count,
        "channel_count": channel_count,
        "feature_count": channel_count * frame_count,
        "seed": seed,
        "sample_count": len(records),
        "train_count": len(train_records),
        "valid_count": len(valid_records),
        "validation_ratio": format_number(args.validation_ratio),
        "train_balanced_accuracy": format_number(train_metrics["balancedAccuracy"]),
        "train_auc": format_number(train_metrics["auc"]),
        "valid_accuracy": format_number(valid_metrics["accuracy"]),
        "valid_balanced_accuracy": format_number(valid_metrics["balancedAccuracy"]),
        "valid_auc": format_number(valid_metrics["auc"]),
        "valid_heavy_recall": format_number(valid_metrics["heavyRecall"]),
        "valid_light_recall": format_number(valid_metrics["lightRecall"]),
        "valid_min_recall": format_number(valid_metrics["minRecall"]),
    }


def make_summary_row(preset, pre, post, run_rows):
    rows = [
        row for row in run_rows
        if row["feature_preset"] == preset and int(row["pre_frames"]) == pre and int(row["post_frames"]) == post
    ]
    output = {
        "rank": 0,
        "window": window_name(pre, post),
        "feature_preset": preset,
        "pre_frames": pre,
        "post_frames": post,
        "first_frame": -pre,
        "last_frame": post,
        "frame_count": rows[0]["frame_count"] if rows else 0,
        "channel_count": rows[0]["channel_count"] if rows else 0,
        "feature_count": rows[0]["feature_count"] if rows else 0,
        "runs": len(rows),
        "sample_count": rows[0]["sample_count"] if rows else 0,
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


def window_name(pre, post):
    return "pre" + str(pre) + "_post" + str(post)


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


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stddev(values):
    if not values:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


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
