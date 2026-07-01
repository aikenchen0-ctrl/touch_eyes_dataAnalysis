#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from train_android_touch_cnn import FEATURE_PRESETS
from train_android_touch_cnn import parse_csv_list


DEFAULT_PRESETS = [
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
]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    presets = parse_csv_list(args.presets) if args.presets else list(DEFAULT_PRESETS)
    rows = []
    for preset in presets:
        if preset not in FEATURE_PRESETS:
            raise RuntimeError("unknown_feature_preset:" + preset)
        model_id = args.model_prefix + "_" + safe_id(preset)
        model_dir = output_dir / model_id
        command = [
            sys.executable,
            str(Path(__file__).with_name("train_android_touch_cnn.py")),
            "--manifest",
            args.manifest,
            "--output-dir",
            str(model_dir),
            "--model-id",
            model_id,
            "--display-name",
            args.display_prefix + " " + preset,
            "--feature-preset",
            preset,
            "--frame-indices=" + args.frame_indices,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--validation-ratio",
            str(args.validation_ratio),
            "--seed",
            str(args.seed),
            "--missing-frame-policy",
            args.missing_frame_policy,
        ]
        if not completed_model(model_dir, model_id):
            run_command(command, output_dir / (model_id + ".log"))
        rows.append(report_row(preset, model_id, model_dir, args))
        write_outputs(output_dir, rows, args, presets)
    write_outputs(output_dir, rows, args, presets)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--presets", default="")
    parser.add_argument("--frame-indices", default="-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--validation-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--missing-frame-policy", default="zero", choices=("zero", "skip", "error"))
    parser.add_argument("--model-prefix", default="android_cnn_ablation")
    parser.add_argument("--display-prefix", default="Android CNN ablation")
    return parser.parse_args()


def run_command(command, log_path):
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError("cnn_training_failed:" + str(log_path))


def completed_model(model_dir, model_id):
    report_path = model_dir / "training_report.json"
    scaler_path = model_dir / ("AispectImpactCNNScaler_" + model_id + ".json")
    weights_path = model_dir / ("AispectImpactCNNWeights_" + model_id + ".json")
    return report_path.exists() and scaler_path.exists() and weights_path.exists()


def report_row(preset, model_id, model_dir, args):
    report_path = model_dir / "training_report.json"
    scaler_path = model_dir / ("AispectImpactCNNScaler_" + model_id + ".json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    train = report["trainMetrics"]
    valid = report["validMetrics"]
    feature_names = scaler.get("featureNames", [])
    frame_indices = scaler.get("frameIndices", [])
    return {
        "preset": preset,
        "model_id": model_id,
        "channel_count": len(feature_names),
        "frame_count": len(frame_indices),
        "first_frame": frame_indices[0] if frame_indices else "",
        "last_frame": frame_indices[-1] if frame_indices else "",
        "feature_names": "|".join(feature_names),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "train_count": train.get("count", 0),
        "valid_count": valid.get("count", 0),
        "train_accuracy": train.get("accuracy", 0),
        "train_balanced_accuracy": train.get("balancedAccuracy", 0),
        "valid_accuracy": valid.get("accuracy", 0),
        "valid_balanced_accuracy": valid.get("balancedAccuracy", 0),
        "valid_heavy_recall": valid.get("heavyRecall", 0),
        "valid_light_recall": valid.get("lightRecall", 0),
        "valid_confusion_heavy_heavy": valid.get("confusionMatrixHeavyLight", [[0, 0], [0, 0]])[0][0],
        "valid_confusion_heavy_light": valid.get("confusionMatrixHeavyLight", [[0, 0], [0, 0]])[0][1],
        "valid_confusion_light_heavy": valid.get("confusionMatrixHeavyLight", [[0, 0], [0, 0]])[1][0],
        "valid_confusion_light_light": valid.get("confusionMatrixHeavyLight", [[0, 0], [0, 0]])[1][1],
        "model_dir": str(model_dir),
    }


def write_outputs(output_dir, rows, args, presets):
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["valid_balanced_accuracy"]),
            min(float(row["valid_heavy_recall"]), float(row["valid_light_recall"])),
            float(row["valid_accuracy"]),
        ),
        reverse=True,
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    write_csv(output_dir / "cnn_ablation_summary.csv", ranked)
    write_json(output_dir / "cnn_ablation_report.json", {
        "manifest": args.manifest,
        "outputDir": str(output_dir),
        "presets": presets,
        "frameIndices": args.frame_indices,
        "epochs": args.epochs,
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "weightDecay": args.weight_decay,
        "validationRatio": args.validation_ratio,
        "seed": args.seed,
        "missingFramePolicy": args.missing_frame_policy,
        "summary": ranked,
    })


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


def safe_id(value):
    return value.replace("-", "_").replace(".", "_")


if __name__ == "__main__":
    main()
