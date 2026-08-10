#!/usr/bin/env python3

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


CANDIDATES = {
    "old13": {"featurePreset": "old13", "matrixTransform": "none"},
    "patch21": {
        "featurePreset": "enhanced_patch21_no_confidence",
        "matrixTransform": "none",
    },
    "relative_touch": {
        "featurePreset": "enhanced_patch21_no_confidence",
        "matrixTransform": "relative_touch",
    },
    "baseline_imu": {
        "featurePreset": "enhanced_patch21_no_confidence",
        "matrixTransform": "baseline_imu",
    },
}


def experiment_specs(fold_root, output_root, cache_root):
    fold_root = Path(fold_root)
    output_root = Path(output_root)
    cache_root = Path(cache_root)
    manifests = sorted(fold_root.glob("domain_*.csv"))
    if not manifests:
        raise RuntimeError("missing_domain_folds")
    specs = []
    for candidate, config in CANDIDATES.items():
        preset = config["featurePreset"]
        for manifest in manifests:
            fold = manifest.stem
            model_id = "deduplicated_clean_v3_probe_" + candidate + "_" + fold
            specs.append(
                {
                    "candidate": candidate,
                    "candidateConfig": config,
                    "fold": fold,
                    "manifest": manifest,
                    "outputDir": output_root / candidate / fold,
                    "sampleCache": cache_root / ("samples_" + preset + ".pkl"),
                    "modelId": model_id,
                }
            )
    return specs


def build_command(spec, python_executable, epochs, patience, batch_size, seed):
    config = spec["candidateConfig"]
    return [
        python_executable,
        "android/analysis/train_android_finger_four_class_cnn.py",
        "--manifest",
        str(spec["manifest"]),
        "--output-dir",
        str(spec["outputDir"]),
        "--model-id",
        spec["modelId"],
        "--display-name",
        spec["modelId"],
        "--feature-preset",
        config["featurePreset"],
        "--frame-indices=-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        "0.001",
        "--weight-decay",
        "0.0001",
        "--validation-ratio",
        "0.25",
        "--seed",
        str(seed),
        "--missing-frame-policy",
        "zero",
        "--split-strategy",
        "predefined",
        "--early-stopping-patience",
        str(patience),
        "--sample-cache",
        str(spec["sampleCache"]),
        "--matrix-transform",
        config["matrixTransform"],
        "--group-balance",
    ]


def pending_specs(specs, force=False):
    if force:
        return list(specs)
    return [
        spec
        for spec in specs
        if not (spec["outputDir"] / "training_report.json").exists()
    ]


def aggregate_reports(output_root):
    rows = []
    for path in sorted(Path(output_root).glob("*/*/training_report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        history = report.get("history", [])
        if not history:
            continue
        best = max(history, key=lambda item: item.get("validBalancedAccuracy", 0.0))
        valid = report.get("validMetrics", {})
        recalls = valid.get("classRecalls", {})
        if any(class_label not in recalls for class_label in ("thumb_light", "thumb_heavy", "index_light", "index_heavy")):
            continue
        channel_count = len(report.get("featureNames", []))
        if channel_count == 0:
            scaler_paths = sorted(path.parent.glob("AispectImpactCNNScaler_*.json"))
            if scaler_paths:
                scaler = json.loads(scaler_paths[0].read_text(encoding="utf-8"))
                channel_count = len(scaler.get("featureNames", []))
        rows.append(
            {
                "candidate": path.parent.parent.name,
                "fold": path.parent.name,
                "channelCount": channel_count,
                "epochsRun": len(history),
                "bestEpoch": best.get("epoch", 0),
                "trainCount": report.get("trainMetrics", {}).get("count", 0),
                "validCount": valid.get("count", 0),
                "trainBalancedAccuracy": report.get("trainMetrics", {}).get("balancedAccuracy", 0.0),
                "validAccuracy": valid.get("accuracy", 0.0),
                "validBalancedAccuracy": valid.get("balancedAccuracy", 0.0),
                "validMinRecall": min(recalls.values()),
                "thumbLightRecall": recalls["thumb_light"],
                "thumbHeavyRecall": recalls["thumb_heavy"],
                "indexLightRecall": recalls["index_light"],
                "indexHeavyRecall": recalls["index_heavy"],
                "confusionMatrix": json.dumps(valid.get("confusionMatrix", []), separators=(",", ":")),
            }
        )

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["candidate"]].append(row)
    candidates = []
    for candidate, items in grouped.items():
        balanced = [item["validBalancedAccuracy"] for item in items]
        candidates.append(
            {
                "candidate": candidate,
                "completedFolds": len(items),
                "channelCount": max(item["channelCount"] for item in items),
                "meanValidBalancedAccuracy": sum(balanced) / len(balanced),
                "worstFoldBalancedAccuracy": min(balanced),
                "bestFoldBalancedAccuracy": max(balanced),
                "worstClassRecall": min(item["validMinRecall"] for item in items),
            }
        )
    candidates.sort(
        key=lambda item: (
            item["meanValidBalancedAccuracy"],
            item["worstFoldBalancedAccuracy"],
            item["worstClassRecall"],
            -item["channelCount"],
        ),
        reverse=True,
    )
    return rows, candidates


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("missing_training_reports")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_reports(output_root, rows, candidates):
    output_root = Path(output_root)
    write_csv(output_root / "probe_results.csv", rows)
    write_csv(output_root / "candidate_comparison.csv", candidates)
    target_met = bool(candidates) and candidates[0]["meanValidBalancedAccuracy"] >= 0.70 and candidates[0][
        "worstFoldBalancedAccuracy"
    ] >= 0.55 and candidates[0]["worstClassRecall"] >= 0.45
    payload = {
        "policy": "deduplicated_clean_v3_strict_domain_probe",
        "target": {
            "meanHoldoutBalancedAccuracy": 0.70,
            "worstFoldBalancedAccuracy": 0.55,
            "worstClassRecall": 0.45,
        },
        "candidateSummary": candidates,
        "rankedCandidates": [item["candidate"] for item in candidates],
        "selectedCandidates": [item["candidate"] for item in candidates[:2]] if target_met else [],
        "fullTrainingRecommended": target_met,
        "targetMet": target_met,
    }
    (output_root / "probe_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run Clean V3 strict-domain probes.")
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=Path("android/analysis/datasets/deduplicated_clean_v3_20260728/folds"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("models/deduplicated_clean_v3_20260728"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("models/deduplicated_clean_v3_20260728/cache"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_args()
    specs = experiment_specs(arguments.fold_root, arguments.output_root, arguments.cache_root)
    for spec in pending_specs(specs, force=arguments.force):
        command = build_command(
            spec,
            arguments.python,
            epochs=arguments.epochs,
            patience=arguments.patience,
            batch_size=arguments.batch_size,
            seed=arguments.seed,
        )
        if arguments.dry_run:
            print(" ".join(command))
            continue
        subprocess.run(command, check=True)
    if not arguments.dry_run:
        rows, candidates = aggregate_reports(arguments.output_root)
        write_reports(arguments.output_root, rows, candidates)


if __name__ == "__main__":
    main()
