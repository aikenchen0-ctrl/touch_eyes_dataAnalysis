#!/usr/bin/env python3
"""Train all-data pose-invariant time-grid comparison models."""

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

try:
    from . import train_time_grid_all_data_ablation as all_data_trainer
    from . import train_time_grid_strict_cnn as base_trainer
except ImportError:
    import train_time_grid_all_data_ablation as all_data_trainer
    import train_time_grid_strict_cnn as base_trainer


SOURCE_FEATURE_COUNT = 26
TOUCH23_SOURCE_INDICES = tuple(range(16)) + (18, 19, 20, 21, 22, 23, 25)
TOUCH23_FEATURE_NAMES = (
    "x_norm",
    "y_norm",
    "userAcceleration.x",
    "userAcceleration.y",
    "userAcceleration.z",
    "rotationRate.x",
    "rotationRate.y",
    "rotationRate.z",
    "gravity.x",
    "gravity.y",
    "gravity.z",
    "pressure",
    "size",
    "major_norm",
    "minor_norm",
    "log_axis_ratio",
    "pointer_count",
    "touch_valid",
    "pressure_available",
    "size_available",
    "major_available",
    "minor_available",
    "pointer_count_available",
)
NO_GRAVITY_SOURCE_INDICES = tuple(
    index for index in TOUCH23_SOURCE_INDICES if index not in (8, 9, 10)
)
NO_GRAVITY_FEATURE_NAMES = tuple(
    name for index, name in zip(TOUCH23_SOURCE_INDICES, TOUCH23_FEATURE_NAMES)
    if index not in (8, 9, 10)
)
DELTA_GRAVITY_FEATURE_NAMES = tuple(
    {
        "gravity.x": "deltaGravity.x",
        "gravity.y": "deltaGravity.y",
        "gravity.z": "deltaGravity.z",
    }.get(name, name)
    for name in TOUCH23_FEATURE_NAMES
)

VARIANTS = {
    "no_gravity": {
        "featureNames": NO_GRAVITY_FEATURE_NAMES,
        "sourceIndices": NO_GRAVITY_SOURCE_INDICES,
        "featureContract": "causal_time_grid_touch7_mask5_no_gravity_v1",
    },
    "delta_gravity": {
        "featureNames": DELTA_GRAVITY_FEATURE_NAMES,
        "sourceIndices": TOUCH23_SOURCE_INDICES,
        "featureContract": "causal_time_grid_imu11_touch7_mask5_delta_gravity_v1",
    },
}


def build_variant_matrix(matrix, variant):
    if variant not in VARIANTS:
        raise RuntimeError("unknown_pose_invariant_variant")
    if not isinstance(matrix, list) or len(matrix) != 9:
        raise RuntimeError("invalid_touch26_matrix_shape")
    selected = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != SOURCE_FEATURE_COUNT:
            raise RuntimeError("invalid_touch26_matrix_shape")
        selected.append([float(row[index]) for index in VARIANTS[variant]["sourceIndices"]])
    if variant != "delta_gravity":
        return selected
    previous = None
    for row in selected:
        current = row[8:11]
        row[8:11] = (
            [0.0, 0.0, 0.0]
            if previous is None
            else [current[index] - previous[index] for index in range(3)]
        )
        previous = current
    return selected


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_all_data(dataset_dir, variant):
    root = Path(dataset_dir)
    report = json.loads((root / "training_cleaning_report.json").read_text(encoding="utf-8"))
    if int(report.get("eligibleCount", 0)) <= 0:
        raise RuntimeError("no_eligible_training_samples")
    rows = read_jsonl(root / "training_touch26.jsonl")
    hashes = [str(row.get("sourceSha256") or "") for row in rows]
    if not all(hashes) or len(set(hashes)) != len(hashes):
        raise RuntimeError("duplicate_or_missing_training_source_sha256")
    samples = []
    for row in rows:
        matrix = row.get("matrix")
        raw_touch_available = []
        for frame in matrix or []:
            if not isinstance(frame, list) or len(frame) != SOURCE_FEATURE_COUNT:
                raise RuntimeError("invalid_touch26_matrix_shape")
            raw_touch_available.append(float(frame[19]))
        if len(raw_touch_available) != 9:
            raise RuntimeError("invalid_touch_availability_shape")
        label = str(row.get("classLabel") or "")
        if label not in base_trainer.CLASS_LABELS:
            raise RuntimeError("unknown_class_label")
        samples.append({
            "sourceSha256": str(row["sourceSha256"]),
            "sampleId": str(row.get("sampleId") or ""),
            "classLabel": label,
            "label": base_trainer.CLASS_LABELS.index(label),
            "groupId": str(row.get("groupId") or ""),
            "motionState": str(row.get("motionState") or ""),
            "trainingWeight": float(row.get("trainingWeight", 1.0)),
            "matrix": build_variant_matrix(matrix, variant),
            "touchAvailable": raw_touch_available,
        })
    if len(samples) != int(report["eligibleCount"]):
        raise RuntimeError("eligible_count_mismatch")
    if not all(sample["groupId"] for sample in samples):
        raise RuntimeError("missing_group_identifier")
    return samples, VARIANTS[variant]["featureNames"]


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_variant(dataset_dir, output_dir, variant, config=None):
    config = dict(all_data_trainer.default_all_data_config() if config is None else config)
    if config.get("earlyStoppingPatience") is not None:
        raise RuntimeError("all_data_training_requires_disabled_early_stopping")
    base_trainer.configure_determinism(int(config["seed"]))
    samples, feature_names = load_all_data(dataset_dir, variant)
    scaler = all_data_trainer.fit_all_data_scaler(samples, feature_names)
    x_train, y_train = all_data_trainer.tensors(samples, scaler)
    weights = torch.tensor(base_trainer.compute_sample_weights(samples), dtype=torch.float32)
    model = base_trainer.StrictTimeGridCnn(torch, len(feature_names))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learningRate"]),
        weight_decay=float(config["weightDecay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["epochs"]),
        eta_min=float(config["minimumLearningRate"]),
    )
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        loss = base_trainer.train_epoch(
            model,
            x_train,
            y_train,
            weights,
            optimizer,
            int(config["batchSize"]),
            int(config["seed"]) + epoch,
        )
        scheduler.step()
        metrics = base_trainer.evaluate(model, x_train, y_train)
        history.append({
            "epoch": epoch,
            "learningRate": optimizer.param_groups[0]["lr"],
            "trainLoss": loss,
            "trainBalancedAccuracy": metrics["balancedAccuracy"],
            "trainMacroF1": metrics["macroF1"],
        })
    train_metrics = base_trainer.evaluate(model, x_train, y_train)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "weights.json", base_trainer.state_payload(model))
    weights_sha256 = hashlib.sha256((output_root / "weights.json").read_bytes()).hexdigest()
    write_json(output_root / "scaler.json", {
        **scaler,
        "featureContract": VARIANTS[variant]["featureContract"],
        "evaluationScope": "all_data_fit_only",
        "deploymentAllowed": False,
    })
    write_json(output_root / "model_contract.json", {
        "featureContract": VARIANTS[variant]["featureContract"],
        "featureNames": list(feature_names),
        "matrixShape": [9, len(feature_names)],
        "gridOffsetsMs": [-15, -10, -5, 0, 5, 10, 15, 20, 25],
        "variant": variant,
        "deltaDefinition": (
            "current_grid_gravity_minus_previous_grid_gravity_first_slot_zero"
            if variant == "delta_gravity" else None
        ),
        "evaluationScope": "all_data_fit_only",
        "deploymentAllowed": False,
    })
    with (output_root / "history.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (output_root / "source_manifest.csv").open("w", encoding="utf-8", newline="") as output:
        fields = ("sourceSha256", "sampleId", "groupId", "classLabel", "trainingWeight")
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for sample, weight in zip(samples, base_trainer.compute_sample_weights(samples)):
            writer.writerow({
                "sourceSha256": sample["sourceSha256"],
                "sampleId": sample["sampleId"],
                "groupId": sample["groupId"],
                "classLabel": sample["classLabel"],
                "trainingWeight": weight,
            })
    report = {
        "variant": variant,
        "contract": VARIANTS[variant]["featureContract"],
        "evaluationScope": "all_data_fit_only",
        "deploymentAllowed": False,
        "seed": int(config["seed"]),
        "featureNames": list(feature_names),
        "featureCount": len(feature_names),
        "parameterCount": sum(parameter.numel() for parameter in model.parameters()),
        "trainCount": len(samples),
        "groupCount": len({sample["groupId"] for sample in samples}),
        "epochsCompleted": len(history),
        "finalTrainLoss": history[-1]["trainLoss"],
        "trainMetrics": train_metrics,
        "weightsSha256": weights_sha256,
        "constantFeatures": [channel["name"] for channel in scaler["channels"] if channel["constant"]],
        "config": config,
    }
    write_json(output_root / "training_report.json", report)
    return report


def run_all(dataset_dir, output_dir, max_workers=2):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    config = all_data_trainer.default_all_data_config()
    specs = list(VARIANTS)
    write_json(output_root / "experiment_contract.json", {
        "evaluationScope": "all_data_fit_only",
        "deploymentAllowed": False,
        "datasetDir": str(Path(dataset_dir).resolve()),
        "taskCount": len(specs),
        "maxWorkers": min(int(max_workers), len(specs)),
        "config": config,
        "variants": specs,
    })
    reports = {}
    with ProcessPoolExecutor(max_workers=min(int(max_workers), len(specs))) as executor:
        futures = {
            executor.submit(
                train_variant,
                str(Path(dataset_dir).resolve()),
                str(output_root / variant),
                variant,
                config,
            ): variant
            for variant in specs
        }
        for future in as_completed(futures):
            reports[futures[future]] = future.result()
    aggregate = {
        "evaluationScope": "all_data_fit_only",
        "deploymentAllowed": False,
        "warning": "These are in-sample fits and are not deployment or generalization results.",
        "variants": [reports[variant] for variant in specs],
    }
    write_json(output_root / "aggregate_report.json", aggregate)
    with (output_root / "aggregate_results.csv").open("w", encoding="utf-8", newline="") as output:
        fields = ("variant", "featureCount", "trainCount", "trainBalancedAccuracy", "trainMacroF1", "worstClassRecall")
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for variant in specs:
            report = reports[variant]
            writer.writerow({
                "variant": variant,
                "featureCount": report["featureCount"],
                "trainCount": report["trainCount"],
                "trainBalancedAccuracy": report["trainMetrics"]["balancedAccuracy"],
                "trainMacroF1": report["trainMetrics"]["macroF1"],
                "worstClassRecall": min(report["trainMetrics"]["recallByClass"].values()),
            })
    return aggregate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    return parser.parse_args()


def main():
    arguments = parse_args()
    print(json.dumps(run_all(arguments.dataset_dir, arguments.output_dir, arguments.max_workers), sort_keys=True))


if __name__ == "__main__":
    main()
