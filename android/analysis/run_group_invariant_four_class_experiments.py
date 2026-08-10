#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


TRANSFORMS = (
    "imu11_no_time",
    "invariant_shape",
    "relative_touch",
)

FOLDS = (
    "le2110_tao_20260722",
    "phk110_zhuo_20260722",
    "le2110_zhuo_20260723",
    "phk110_tao_20260723",
)


def main():
    args = parse_args()
    specs = experiment_specs(Path(args.manifest_root), Path(args.output_root), Path(args.sample_cache))
    for spec in specs:
        report = spec["outputDir"] / "training_report.json"
        if report.exists() and not args.force:
            continue
        command = build_command(spec, args.python)
        if args.dry_run:
            print(" ".join(command))
            continue
        subprocess.run(command, check=True)
    if not args.dry_run:
        rows, candidates = aggregate_reports(Path(args.output_root))
        write_comparison(Path(args.output_root) / "model_comparison.csv", rows)
        write_analysis(Path(args.output_root) / "training_analysis.json", rows, candidates)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-root",
        default="android/analysis/datasets/all_collections_grouped_clean_v2_20260724",
    )
    parser.add_argument(
        "--output-root",
        default="models/all_collections_invariant_v3_20260724",
    )
    parser.add_argument(
        "--sample-cache",
        default="models/all_collections_grouped_clean_v2_20260724/samples_patch21.pkl",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def experiment_specs(manifest_root, output_root, sample_cache):
    specs = []
    for transform in TRANSFORMS:
        for fold in FOLDS:
            model_id = "all_collections_invariant_v3_" + transform + "_" + fold
            specs.append({
                "transform": transform,
                "fold": fold,
                "manifest": manifest_root / fold / "manifest.csv",
                "outputDir": output_root / transform / fold,
                "sampleCache": sample_cache,
                "modelId": model_id,
            })
    return specs


def build_command(spec, python_executable):
    return [
        python_executable,
        "android/analysis/train_android_finger_four_class_cnn.py",
        "--manifest", str(spec["manifest"]),
        "--output-dir", str(spec["outputDir"]),
        "--model-id", spec["modelId"],
        "--display-name", spec["modelId"],
        "--feature-preset", "enhanced_patch21_no_confidence",
        "--frame-indices=-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9",
        "--epochs", "120",
        "--batch-size", "128",
        "--learning-rate", "0.001",
        "--weight-decay", "0.0001",
        "--validation-ratio", "0.25",
        "--seed", "22",
        "--missing-frame-policy", "zero",
        "--split-strategy", "predefined",
        "--early-stopping-patience", "20",
        "--sample-cache", str(spec["sampleCache"]),
        "--matrix-transform", spec["transform"],
        "--group-balance",
    ]


def aggregate_reports(output_root):
    rows = []
    for path in sorted(output_root.glob("*/*/training_report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        history = report.get("history", [])
        best = max(history, key=lambda item: item.get("validBalancedAccuracy", 0.0))
        valid = report["validMetrics"]
        recalls = valid["classRecalls"]
        rows.append({
            "matrixTransform": path.parent.parent.name,
            "fold": path.parent.name,
            "epochsRun": len(history),
            "bestEpoch": best.get("epoch", 0),
            "trainCount": report["trainMetrics"]["count"],
            "validCount": valid["count"],
            "trainBalancedAccuracy": report["trainMetrics"]["balancedAccuracy"],
            "validAccuracy": valid["accuracy"],
            "validBalancedAccuracy": valid["balancedAccuracy"],
            "validMinRecall": min(recalls.values()),
            "thumbLightRecall": recalls["thumb_light"],
            "thumbHeavyRecall": recalls["thumb_heavy"],
            "indexLightRecall": recalls["index_light"],
            "indexHeavyRecall": recalls["index_heavy"],
            "confusionMatrix": json.dumps(valid["confusionMatrix"], separators=(",", ":")),
        })

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["matrixTransform"]].append(row)
    candidates = []
    for transform, items in sorted(grouped.items()):
        balanced = [item["validBalancedAccuracy"] for item in items]
        candidates.append({
            "matrixTransform": transform,
            "completedFolds": len(items),
            "meanValidBalancedAccuracy": sum(balanced) / len(balanced),
            "worstFoldBalancedAccuracy": min(balanced),
            "bestFoldBalancedAccuracy": max(balanced),
            "worstClassRecall": min(item["validMinRecall"] for item in items),
        })
    candidates.sort(key=lambda item: item["meanValidBalancedAccuracy"], reverse=True)
    return rows, candidates


def write_comparison(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("missing_training_reports")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_analysis(path, rows, candidates):
    payload = {
        "policy": "existing_data_only_device_invariance",
        "target": {
            "meanHoldoutBalancedAccuracy": 0.70,
            "worstFoldBalancedAccuracy": 0.60,
        },
        "candidateSummary": candidates,
        "foldResults": rows,
        "selectedCandidate": candidates[0]["matrixTransform"] if candidates else None,
        "targetMet": bool(candidates) and candidates[0]["meanValidBalancedAccuracy"] >= 0.70 and candidates[0]["worstFoldBalancedAccuracy"] >= 0.60,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
