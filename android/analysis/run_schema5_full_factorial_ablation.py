#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


FEATURE_PRESET_NAMES = (
    "old13",
    "old13_raw_radius",
    "old13_covariance",
    "old13_multichannel",
    "old13_gaussian_summary",
    "old13_gaussian_map9x9",
    "gaussian_map9x9_only",
    "multichannel_only",
    "covariance_only",
    "patch9",
    "enhanced_patch21_no_confidence",
    "enhanced_patch22",
)

WINDOWS = {
    "pre5_post0": "-5,-4,-3,-2,-1,0",
    "pre5_post3": "-5,-4,-3,-2,-1,0,1,2,3",
    "pre5_post6": "-5,-4,-3,-2,-1,0,1,2,3,4,5,6",
    "pre5_post9": "-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9",
    "pre5_post15": "-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
    "pre3_post9": "-3,-2,-1,0,1,2,3,4,5,6,7,8,9",
    "pre0_post9": "0,1,2,3,4,5,6,7,8,9",
}

MOTION_FOLDS = (
    "handheld_static",
    "placed_static",
    "walking",
)

STAGE_SETTINGS = {
    "temporal": {
        "epochs": 60,
        "earlyStoppingPatience": 12,
    },
    "strict": {
        "epochs": 120,
        "earlyStoppingPatience": 20,
    },
}


def temporal_specs(manifest_root, output_root, cache_root):
    specs = []
    for preset in FEATURE_PRESET_NAMES:
        for window, frame_indices in WINDOWS.items():
            if window == "pre5_post9":
                output_dir = output_root / "feature_temporal" / preset
                model_id = "bbsiak_feature_temporal_" + preset
            else:
                output_dir = output_root / "window_temporal" / (preset + "__" + window)
                model_id = "bbsiak_window_temporal_" + preset + "_" + window
            specs.append({
                "stage": "temporal",
                "preset": preset,
                "window": window,
                "frameIndices": frame_indices,
                "manifest": manifest_root / "folds" / "temporal_tail.csv",
                "outputDir": output_dir,
                "cachePath": cache_root / (preset + "_" + window + ".pkl"),
                "modelId": model_id,
            })
    return specs


def strict_specs(manifest_root, output_root, cache_root):
    specs = []
    for temporal in temporal_specs(manifest_root, output_root, cache_root):
        for motion in MOTION_FOLDS:
            specs.append({
                "stage": "strict",
                "preset": temporal["preset"],
                "window": temporal["window"],
                "frameIndices": temporal["frameIndices"],
                "motion": motion,
                "manifest": manifest_root / "folds" / ("motion_" + motion + ".csv"),
                "outputDir": output_root / "motion_holdout" / (
                    temporal["preset"] + "__" + temporal["window"]
                ) / motion,
                "cachePath": temporal["cachePath"],
                "modelId": "bbsiak_motion_" + temporal["preset"] + "__" + temporal["window"] + "_" + motion,
            })
    return specs


def completed_spec(spec):
    output_dir = spec["outputDir"]
    model_id = spec["modelId"]
    return all((
        (output_dir / "training_report.json").exists(),
        (output_dir / ("AispectImpactCNNWeights_" + model_id + ".json")).exists(),
        (output_dir / ("AispectImpactCNNScaler_" + model_id + ".json")).exists(),
    ))


def build_command(spec, python_executable=sys.executable):
    settings = STAGE_SETTINGS[spec["stage"]]
    return [
        python_executable,
        str(Path(__file__).with_name("train_android_finger_four_class_cnn.py")),
        "--manifest", str(spec["manifest"]),
        "--output-dir", str(spec["outputDir"]),
        "--model-id", spec["modelId"],
        "--display-name", spec["modelId"],
        "--feature-preset", spec["preset"],
        "--frame-indices=" + spec["frameIndices"],
        "--epochs", str(settings["epochs"]),
        "--batch-size", "128",
        "--learning-rate", "0.001",
        "--weight-decay", "0.0001",
        "--validation-ratio", "0.25",
        "--seed", "22",
        "--missing-frame-policy", "error",
        "--anchor-mode", "down",
        "--require-press-active",
        "--split-strategy", "predefined",
        "--early-stopping-patience", str(settings["earlyStoppingPatience"]),
        "--sample-cache", str(spec["cachePath"]),
        "--matrix-transform", "none",
    ]


def report_row(spec):
    report_path = spec["outputDir"] / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    valid = report.get("validMetrics", {})
    recalls = valid.get("classRecalls", {})
    return {
        "preset": spec["preset"],
        "window": spec["window"],
        "motion": spec.get("motion", ""),
        "modelId": spec["modelId"],
        "modelDir": str(spec["outputDir"]),
        "bestEpoch": int(report.get("bestEpoch", 0)),
        "validCount": int(valid.get("count", 0)),
        "validAccuracy": float(valid.get("accuracy", 0.0)),
        "validBalancedAccuracy": float(valid.get("balancedAccuracy", 0.0)),
        "validMinRecall": min(recalls.values()) if recalls else 0.0,
        "validClassRecalls": recalls,
    }


def aggregate_strict_specs(specs):
    completed_rows = [report_row(spec) for spec in specs if completed_spec(spec)]
    rows_by_pair = defaultdict(list)
    specs_by_pair = defaultdict(list)
    for spec in specs:
        specs_by_pair[(spec["preset"], spec["window"])].append(spec)
    for row in completed_rows:
        rows_by_pair[(row["preset"], row["window"])].append(row)

    aggregates = []
    for pair, pair_specs in sorted(specs_by_pair.items()):
        pair_rows = rows_by_pair[pair]
        balanced = [row["validBalancedAccuracy"] for row in pair_rows]
        min_recalls = [row["validMinRecall"] for row in pair_rows]
        aggregates.append({
            "preset": pair[0],
            "window": pair[1],
            "completedFolds": len(pair_rows),
            "pendingFolds": len(pair_specs) - len(pair_rows),
            "meanValidBalancedAccuracy": sum(balanced) / len(balanced) if balanced else 0.0,
            "worstValidBalancedAccuracy": min(balanced) if balanced else 0.0,
            "worstValidMinRecall": min(min_recalls) if min_recalls else 0.0,
            "foldRows": sorted(pair_rows, key=lambda row: row["motion"]),
        })
    aggregates.sort(
        key=lambda item: (
            item["completedFolds"] == len(MOTION_FOLDS),
            item["meanValidBalancedAccuracy"],
            item["worstValidBalancedAccuracy"],
            item["worstValidMinRecall"],
        ),
        reverse=True,
    )
    for index, aggregate in enumerate(aggregates, start=1):
        aggregate["rank"] = index
    return completed_rows, aggregates


def aggregate_temporal_specs(specs):
    rows = []
    for spec in specs:
        if not completed_spec(spec):
            continue
        row = report_row(spec)
        row["frameIndices"] = spec["frameIndices"]
        row["frameCount"] = len(spec["frameIndices"].split(","))
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["validBalancedAccuracy"],
            row["validMinRecall"],
            row["validAccuracy"],
        ),
        reverse=True,
    )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def status_for_specs(specs):
    completed = sum(1 for spec in specs if completed_spec(spec))
    return {
        "planned": len(specs),
        "completed": completed,
        "pending": len(specs) - completed,
    }


def csv_row(row):
    output = dict(row)
    for key, value in list(output.items()):
        if isinstance(value, (dict, list)):
            output[key] = json.dumps(value, separators=(",", ":"), sort_keys=True)
    return output


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(csv_row(row))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_outputs(output_root, temporal, strict, requested_stage):
    temporal_rows = aggregate_temporal_specs(temporal)
    strict_rows, strict_aggregates = aggregate_strict_specs(strict)
    temporal_fields = (
        "rank",
        "preset",
        "window",
        "frameIndices",
        "frameCount",
        "modelId",
        "modelDir",
        "bestEpoch",
        "validCount",
        "validAccuracy",
        "validBalancedAccuracy",
        "validMinRecall",
        "validClassRecalls",
    )
    strict_fields = (
        "rank",
        "preset",
        "window",
        "completedFolds",
        "pendingFolds",
        "meanValidBalancedAccuracy",
        "worstValidBalancedAccuracy",
        "worstValidMinRecall",
        "foldRows",
    )
    write_csv(output_root / "full_factorial_temporal_summary.csv", temporal_rows, temporal_fields)
    write_csv(output_root / "full_factorial_strict_summary.csv", strict_aggregates, strict_fields)
    write_json(output_root / "full_factorial_temporal_summary.json", {
        "plannedPairs": len(temporal),
        "completedPairs": len(temporal_rows),
        "pendingPairs": len(temporal) - len(temporal_rows),
        "rows": temporal_rows,
    })
    write_json(output_root / "full_factorial_strict_summary.json", {
        "plannedRuns": len(strict),
        "completedRuns": len(strict_rows),
        "pendingRuns": len(strict) - len(strict_rows),
        "plannedPairs": len(FEATURE_PRESET_NAMES) * len(WINDOWS),
        "completePairs": sum(
            1 for aggregate in strict_aggregates
            if aggregate["completedFolds"] == len(MOTION_FOLDS)
        ),
        "offlineOnly": True,
        "aggregates": strict_aggregates,
    })
    write_json(output_root / "full_factorial_status.json", {
        "requestedStage": requested_stage,
        "matrix": {
            "featurePresetCount": len(FEATURE_PRESET_NAMES),
            "windowCount": len(WINDOWS),
            "motionFoldCount": len(MOTION_FOLDS),
        },
        "trainingContract": {
            "anchorMode": "down",
            "missingFramePolicy": "error",
            "splitStrategy": "predefined",
            "seed": 22,
            "batchSize": 128,
            "learningRate": 0.001,
            "weightDecay": 0.0001,
            "temporalEpochs": STAGE_SETTINGS["temporal"]["epochs"],
            "temporalEarlyStoppingPatience": STAGE_SETTINGS["temporal"]["earlyStoppingPatience"],
            "strictEpochs": STAGE_SETTINGS["strict"]["epochs"],
            "strictEarlyStoppingPatience": STAGE_SETTINGS["strict"]["earlyStoppingPatience"],
        },
        "temporal": status_for_specs(temporal),
        "strict": status_for_specs(strict),
        "automaticAppDeployment": False,
    })


def run_specs(specs, args, write_progress):
    for spec in specs:
        if completed_spec(spec) and not args.force:
            continue
        command = build_command(spec, args.python)
        if args.dry_run:
            print(subprocess.list2cmdline(command))
            continue
        subprocess.run(command, check=True)
        write_progress()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-root",
        default="android/analysis/datasets/bbsiak_schema5_down_anchor_20260806",
    )
    parser.add_argument(
        "--output-root",
        default="models/bbsiak_schema5_down_anchor_20260806",
    )
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--stage", choices=("temporal", "strict", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest_root = Path(args.manifest_root)
    output_root = Path(args.output_root)
    cache_root = Path(args.cache_root) if args.cache_root else output_root / "cache"
    temporal = temporal_specs(manifest_root, output_root, cache_root)
    strict = strict_specs(manifest_root, output_root, cache_root)

    def write_progress():
        write_outputs(output_root, temporal, strict, args.stage)

    write_progress()
    if args.stage in ("temporal", "all"):
        run_specs(temporal, args, write_progress)
    if args.stage in ("strict", "all"):
        run_specs(strict, args, write_progress)
    write_progress()


if __name__ == "__main__":
    main()
