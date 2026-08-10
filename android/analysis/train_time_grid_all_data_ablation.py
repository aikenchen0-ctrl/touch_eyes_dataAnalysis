#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

try:
    from . import train_time_grid_strict_cnn as base
except ImportError:
    import train_time_grid_strict_cnn as base


SOURCE_FEATURE_COUNT = 26
TOUCH_VALID_SOURCE_INDEX = 19
IMU11_SOURCE_INDICES = tuple(range(11))
TOUCH23_SOURCE_INDICES = tuple(range(16)) + (18, 19, 20, 21, 22, 23, 25)
GEOMETRY3_SOURCE_INDICES = (12, 13, 14)
GEOMETRY4_SOURCE_INDICES = GEOMETRY3_SOURCE_INDICES + (15,)
TOUCH16_SOURCE_INDICES = IMU11_SOURCE_INDICES + GEOMETRY4_SOURCE_INDICES + (19,)
TOUCH15_SOURCE_INDICES = IMU11_SOURCE_INDICES + GEOMETRY4_SOURCE_INDICES
IMU8_NO_GRAVITY_SOURCE_INDICES = tuple(range(8))
IMU6_MOTION_SOURCE_INDICES = tuple(range(2, 8))
IMU9_NO_POSITION_SOURCE_INDICES = tuple(range(2, 11))
IMU12_TOUCH_VALID_SOURCE_INDICES = IMU11_SOURCE_INDICES + (19,)
TOUCH_MEASUREMENT_FEATURES = {
    "pressure",
    "size",
    "major_norm",
    "minor_norm",
    "log_axis_ratio",
    "pointer_count",
}
IDENTITY_FEATURES = {
    "x_norm",
    "y_norm",
    "touch_valid",
    "pressure_available",
    "size_available",
    "major_available",
    "minor_available",
    "pointer_count_available",
}
CHANNEL_CONTRACTS = {
    "imu11": {"sourceIndices": IMU11_SOURCE_INDICES},
    "touch23": {"sourceIndices": TOUCH23_SOURCE_INDICES},
    "touch16_deduplicated": {"sourceIndices": TOUCH16_SOURCE_INDICES},
    "touch15_geometry": {"sourceIndices": TOUCH15_SOURCE_INDICES},
    "imu14_geometry": {"sourceIndices": IMU11_SOURCE_INDICES + GEOMETRY3_SOURCE_INDICES},
    "imu12_touch_valid": {"sourceIndices": IMU12_TOUCH_VALID_SOURCE_INDICES},
    "imu8_no_gravity": {"sourceIndices": IMU8_NO_GRAVITY_SOURCE_INDICES},
    "imu6_motion": {"sourceIndices": IMU6_MOTION_SOURCE_INDICES},
    "imu9_no_position": {"sourceIndices": IMU9_NO_POSITION_SOURCE_INDICES},
}


def default_all_data_config():
    config = dict(base.default_config())
    config["earlyStoppingPatience"] = None
    config["evaluationScope"] = "all_data_fit_only"
    return config


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_matrix(matrix, source_indices):
    if not isinstance(matrix, list) or len(matrix) != 9:
        raise RuntimeError("invalid_time_grid_row_count")
    selected = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != SOURCE_FEATURE_COUNT:
            raise RuntimeError("invalid_touch26_matrix_shape")
        selected.append([float(row[index]) for index in source_indices])
    return selected


def load_all_data(dataset_dir, contract):
    if contract not in CHANNEL_CONTRACTS:
        raise RuntimeError("unknown_channel_contract")
    root = Path(dataset_dir)
    report = json.loads((root / "training_cleaning_report.json").read_text(encoding="utf-8"))
    if int(report.get("eligibleCount", 0)) <= 0:
        raise RuntimeError("no_eligible_training_samples")
    rows = read_jsonl(root / "training_touch26.jsonl")
    source_hashes = [str(row.get("sourceSha256") or "") for row in rows]
    if not all(source_hashes) or len(set(source_hashes)) != len(source_hashes):
        raise RuntimeError("duplicate_or_missing_training_source_sha256")
    source_indices = tuple(CHANNEL_CONTRACTS[contract]["sourceIndices"])
    source_feature_names = tuple(rows[0].get("featureNames") or ())
    if len(source_feature_names) != SOURCE_FEATURE_COUNT:
        raise RuntimeError("invalid_touch26_feature_contract")
    feature_names = tuple(source_feature_names[index] for index in source_indices)
    samples = []
    for row in rows:
        matrix = row.get("matrix")
        touch_available = []
        for frame in matrix or []:
            if not isinstance(frame, list) or len(frame) != SOURCE_FEATURE_COUNT:
                raise RuntimeError("invalid_touch26_matrix_shape")
            touch_available.append(float(frame[TOUCH_VALID_SOURCE_INDEX]))
        if len(touch_available) != 9:
            raise RuntimeError("invalid_touch_availability_shape")
        class_label = str(row.get("classLabel") or "")
        if class_label not in base.CLASS_LABELS:
            raise RuntimeError("unknown_class_label")
        samples.append({
            "sourceSha256": str(row["sourceSha256"]),
            "sampleId": str(row.get("sampleId") or ""),
            "classLabel": class_label,
            "label": base.CLASS_LABELS.index(class_label),
            "groupId": str(row.get("groupId") or ""),
            "motionState": str(row.get("motionState") or ""),
            "trainingWeight": float(row.get("trainingWeight", 1.0)),
            "matrix": select_matrix(matrix, source_indices),
            "touchAvailable": touch_available,
        })
    if len(samples) != int(report["eligibleCount"]):
        raise RuntimeError("eligible_count_mismatch")
    if not all(sample["groupId"] for sample in samples):
        raise RuntimeError("missing_group_identifier")
    return samples, feature_names


def fit_all_data_scaler(samples, feature_names):
    channels = []
    for feature_index, name in enumerate(feature_names):
        if name in IDENTITY_FEATURES:
            channels.append({
                "name": name,
                "strategy": "identity",
                "center": 0.0,
                "scale": 1.0,
                "clipMin": None,
                "clipMax": None,
                "requiresTouchAvailability": False,
                "constant": False,
                "fitCount": len(samples) * 9,
            })
            continue
        requires_touch = name in TOUCH_MEASUREMENT_FEATURES
        values = []
        for sample in samples:
            for row_index, row in enumerate(sample["matrix"]):
                if requires_touch and sample["touchAvailable"][row_index] < 0.5:
                    continue
                value = float(row[feature_index])
                if not math.isfinite(value):
                    raise RuntimeError("non_finite_training_feature")
                values.append(value)
        if not values:
            center = 0.0
            scale = 1.0
            constant = True
        else:
            center = base.median_value(values)
            scale = base.percentile_nearest(values, 0.75) - base.percentile_nearest(values, 0.25)
            constant = abs(scale) < 1e-6
            if constant:
                scale = 1.0
        channels.append({
            "name": name,
            "strategy": "robust_median_iqr",
            "center": center,
            "scale": scale,
            "clipMin": -8.0,
            "clipMax": 8.0,
            "requiresTouchAvailability": requires_touch,
            "constant": constant,
            "fitCount": len(values),
        })
    return {
        "schemaVersion": 1,
        "featureNames": list(feature_names),
        "fitPartition": "all_eligible_samples",
        "evaluationScope": "all_data_fit_only",
        "channels": channels,
    }


def transform_all_data_matrix(sample, scaler):
    matrix = sample["matrix"]
    channels = scaler["channels"]
    if len(matrix) != 9:
        raise RuntimeError("invalid_time_grid_row_count")
    output = []
    for row_index, row in enumerate(matrix):
        if len(row) != len(channels):
            raise RuntimeError("scaler_feature_count_mismatch")
        transformed = []
        for feature_index, channel in enumerate(channels):
            value = float(row[feature_index])
            if channel["requiresTouchAvailability"] and sample["touchAvailable"][row_index] < 0.5:
                value = 0.0
            elif channel["strategy"] == "robust_median_iqr":
                value = (value - channel["center"]) / channel["scale"]
                value = max(channel["clipMin"], min(channel["clipMax"], value))
            transformed.append(value)
        output.append(transformed)
    return output


def tensors(samples, scaler):
    matrices = [transform_all_data_matrix(sample, scaler) for sample in samples]
    x = torch.tensor(matrices, dtype=torch.float32)
    y = torch.tensor([sample["label"] for sample in samples], dtype=torch.long)
    return x, y


def state_payload(model):
    return {
        name: {"shape": list(value.shape), "values": value.detach().reshape(-1).tolist()}
        for name, value in sorted(model.state_dict().items())
    }


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_all_data_contract(dataset_dir, output_dir, contract, config=None):
    config = dict(default_all_data_config() if config is None else config)
    if config.get("earlyStoppingPatience") is not None:
        raise RuntimeError("all_data_training_requires_disabled_early_stopping")
    if int(config["epochs"]) <= 0:
        raise RuntimeError("epochs_must_be_positive")
    base.configure_determinism(int(config["seed"]))
    samples, feature_names = load_all_data(dataset_dir, contract)
    scaler = fit_all_data_scaler(samples, feature_names)
    x_train, y_train = tensors(samples, scaler)
    weights = torch.tensor(base.compute_sample_weights(samples), dtype=torch.float32)
    model = base.StrictTimeGridCnn(torch, len(feature_names))
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
        loss = base.train_epoch(
            model,
            x_train,
            y_train,
            weights,
            optimizer,
            int(config["batchSize"]),
            int(config["seed"]) + epoch,
        )
        scheduler.step()
        metrics = base.evaluate(model, x_train, y_train)
        history.append({
            "epoch": epoch,
            "learningRate": optimizer.param_groups[0]["lr"],
            "trainLoss": loss,
            "trainBalancedAccuracy": metrics["balancedAccuracy"],
            "trainMacroF1": metrics["macroF1"],
        })
    train_metrics = base.evaluate(model, x_train, y_train)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "weights.json", state_payload(model))
    weights_sha256 = hashlib.sha256((output_root / "weights.json").read_bytes()).hexdigest()
    write_json(output_root / "scaler.json", scaler)
    write_json(output_root / "model_contract.json", {
        "featureContract": "all_data_channel_ablation_" + contract + "_v1",
        "featureNames": list(feature_names),
        "matrixShape": [9, len(feature_names)],
        "gridOffsetsMs": [-15, -10, -5, 0, 5, 10, 15, 20, 25],
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
        for sample, weight in zip(samples, base.compute_sample_weights(samples)):
            writer.writerow({
                "sourceSha256": sample["sourceSha256"],
                "sampleId": sample["sampleId"],
                "groupId": sample["groupId"],
                "classLabel": sample["classLabel"],
                "trainingWeight": weight,
            })
    report = {
        "contract": contract,
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--contract", choices=tuple(CHANNEL_CONTRACTS))
    return parser.parse_args()


def main():
    args = parse_args()
    report = train_all_data_contract(args.dataset_dir, args.output_dir, args.contract)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
