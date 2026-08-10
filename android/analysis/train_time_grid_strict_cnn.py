#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch


CLASS_LABELS = ("thumb_light", "thumb_heavy", "index_light", "index_heavy")
IMU11_FEATURE_NAMES = (
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
)
TOUCH23_SOURCE_INDICES = tuple(range(16)) + (18, 19, 20, 21, 22, 23, 25)
TOUCH23_FEATURE_NAMES = IMU11_FEATURE_NAMES + (
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
MASK_BY_FEATURE = {
    "pressure": "pressure_available",
    "size": "size_available",
    "major_norm": "major_available",
    "minor_norm": "minor_available",
    "log_axis_ratio": ("major_available", "minor_available"),
    "pointer_count": "pointer_count_available",
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


class StrictTimeGridCnn(torch.nn.Module):
    def __init__(self, torch_module, feature_count):
        super().__init__()
        self.conv1 = torch_module.nn.Conv1d(feature_count, 32, kernel_size=3, padding=1)
        self.norm1 = torch_module.nn.GroupNorm(4, 32)
        self.conv2 = torch_module.nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.norm2 = torch_module.nn.GroupNorm(4, 32)
        self.dropout = torch_module.nn.Dropout(0.15)
        self.activation = torch_module.nn.GELU()
        self.output = torch_module.nn.Linear(32, 4)

    def forward(self, values):
        values = values.transpose(1, 2)
        values = self.dropout(self.activation(self.norm1(self.conv1(values))))
        values = self.activation(self.norm2(self.conv2(values)))
        values = values.mean(dim=2)
        return self.output(values)


def select_touch23_matrix(matrix):
    if not isinstance(matrix, list) or len(matrix) != 9:
        raise RuntimeError("invalid_touch26_matrix_shape")
    selected = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != 26:
            raise RuntimeError("invalid_touch26_matrix_shape")
        selected.append([float(row[index]) for index in TOUCH23_SOURCE_INDICES])
    return selected


def percentile_nearest(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise RuntimeError("percentile_requires_values")
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def median_value(values):
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def fit_masked_scaler(samples, feature_names):
    feature_names = tuple(feature_names)
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
                "maskIndices": [],
                "constant": False,
                "fitCount": len(samples) * 9,
            })
            continue
        mask_names = MASK_BY_FEATURE.get(name, ())
        if isinstance(mask_names, str):
            mask_names = (mask_names,)
        mask_indices = [feature_names.index(mask_name) for mask_name in mask_names]
        values = []
        for sample in samples:
            for row in sample["matrix"]:
                if mask_indices and not all(float(row[index]) >= 0.5 for index in mask_indices):
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
            center = median_value(values)
            scale = percentile_nearest(values, 0.75) - percentile_nearest(values, 0.25)
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
            "maskIndices": mask_indices,
            "constant": constant,
            "fitCount": len(values),
        })
    return {
        "schemaVersion": 1,
        "featureNames": list(feature_names),
        "fitPartition": "train_only",
        "channels": channels,
    }


def transform_matrix(matrix, scaler):
    output = []
    channels = scaler["channels"]
    for row in matrix:
        if len(row) != len(channels):
            raise RuntimeError("scaler_feature_count_mismatch")
        transformed = []
        for index, channel in enumerate(channels):
            value = float(row[index])
            mask_indices = channel["maskIndices"]
            if mask_indices and not all(float(row[mask_index]) >= 0.5 for mask_index in mask_indices):
                value = 0.0
            elif channel["strategy"] == "robust_median_iqr":
                value = (value - channel["center"]) / channel["scale"]
                value = max(channel["clipMin"], min(channel["clipMax"], value))
            transformed.append(value)
        output.append(transformed)
    return output


def compute_sample_weights(samples):
    if not samples:
        return []
    group_counts = Counter(sample["groupId"] for sample in samples)
    raw = []
    for sample in samples:
        source_weight = float(sample.get("trainingWeight", 1.0))
        raw.append(source_weight / math.sqrt(group_counts[sample["groupId"]]))
    target_per_class = len(samples) / len(CLASS_LABELS)
    class_totals = {
        label: sum(weight for weight, sample in zip(raw, samples) if sample["classLabel"] == label)
        for label in CLASS_LABELS
    }
    weighted = [
        weight * target_per_class / class_totals[sample["classLabel"]]
        for weight, sample in zip(raw, samples)
    ]
    weighted = [max(0.25, min(4.0, weight)) for weight in weighted]
    mean_weight = sum(weighted) / len(weighted)
    return [weight / mean_weight for weight in weighted]


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_fold(dataset_dir, fold, contract):
    root = Path(dataset_dir)
    report = json.loads((root / "training_cleaning_report.json").read_text(encoding="utf-8"))
    if report.get("strictGroupValidationStatus") != "ready" or int(report.get("strictFoldCount", 0)) != 3:
        raise RuntimeError("training_cleaning_report_not_ready")
    source_name = "training_imu11.jsonl" if contract == "imu11" else "training_touch26.jsonl"
    rows = read_jsonl(root / source_name)
    by_hash = {row["sourceSha256"]: row for row in rows}
    if len(by_hash) != len(rows):
        raise RuntimeError("duplicate_training_source_sha256")
    with (root / "strict_group_folds.csv").open("r", encoding="utf-8", newline="") as source:
        fold_rows = [row for row in csv.DictReader(source) if int(row["fold"]) == int(fold)]
    if set(row["sourceSha256"] for row in fold_rows) != set(by_hash):
        raise RuntimeError("fold_matrix_source_mismatch")
    samples = []
    for fold_row in fold_rows:
        row = by_hash[fold_row["sourceSha256"]]
        matrix = row["matrix"] if contract == "imu11" else select_touch23_matrix(row["matrix"])
        samples.append({
            "sourceSha256": row["sourceSha256"],
            "sampleId": row["sampleId"],
            "classLabel": row["classLabel"],
            "label": CLASS_LABELS.index(row["classLabel"]),
            "groupId": row["groupId"],
            "motionState": row.get("motionState", ""),
            "trainingWeight": float(row.get("trainingWeight", 1.0)),
            "split": fold_row["split"],
            "matrix": matrix,
        })
    train = [sample for sample in samples if sample["split"] == "train"]
    valid = [sample for sample in samples if sample["split"] == "valid"]
    validate_fold(train, valid)
    return train, valid


def validate_fold(train, valid):
    train_groups = {sample["groupId"] for sample in train}
    valid_groups = {sample["groupId"] for sample in valid}
    if train_groups.intersection(valid_groups):
        raise RuntimeError("cross_split_group_leakage")
    if {sample["classLabel"] for sample in train} != set(CLASS_LABELS):
        raise RuntimeError("train_missing_class")
    if {sample["classLabel"] for sample in valid} != set(CLASS_LABELS):
        raise RuntimeError("valid_missing_class")


def configure_determinism(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def tensors(samples, scaler):
    matrices = [transform_matrix(sample["matrix"], scaler) for sample in samples]
    x = torch.tensor(matrices, dtype=torch.float32)
    y = torch.tensor([sample["label"] for sample in samples], dtype=torch.long)
    return x, y


def metrics_from_predictions(truths, predictions):
    confusion = [[0 for _ in CLASS_LABELS] for _ in CLASS_LABELS]
    for truth, prediction in zip(truths, predictions):
        confusion[int(truth)][int(prediction)] += 1
    recalls = []
    precisions = []
    for index in range(len(CLASS_LABELS)):
        true_positive = confusion[index][index]
        truth_total = sum(confusion[index])
        predicted_total = sum(row[index] for row in confusion)
        recalls.append(true_positive / truth_total if truth_total else 0.0)
        precisions.append(true_positive / predicted_total if predicted_total else 0.0)
    f1_values = [
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
        for precision, recall in zip(precisions, recalls)
    ]
    accuracy = sum(confusion[index][index] for index in range(4)) / len(truths) if truths else 0.0
    return {
        "count": len(truths),
        "accuracy": accuracy,
        "balancedAccuracy": sum(recalls) / len(recalls),
        "macroF1": sum(f1_values) / len(f1_values),
        "recallByClass": dict(zip(CLASS_LABELS, recalls)),
        "confusionMatrix": confusion,
    }


def predict(model, x, batch_size=256):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            predictions.extend(model(x[start:start + batch_size]).argmax(dim=1).tolist())
    return predictions


def evaluate(model, x, y):
    return metrics_from_predictions(y.tolist(), predict(model, x))


def train_epoch(model, x, y, weights, optimizer, batch_size, seed):
    model.train()
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(x), generator=generator)
    total_loss = 0.0
    total_weight = 0.0
    for start in range(0, len(x), batch_size):
        indices = order[start:start + batch_size]
        logits = model(x[indices])
        losses = torch.nn.functional.cross_entropy(logits, y[indices], reduction="none")
        batch_weights = weights[indices]
        loss = (losses * batch_weights).sum() / batch_weights.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float((losses.detach() * batch_weights).sum())
        total_weight += float(batch_weights.sum())
    return total_loss / total_weight


def clone_state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def state_payload(model):
    return {
        name: {"shape": list(value.shape), "values": value.detach().reshape(-1).tolist()}
        for name, value in sorted(model.state_dict().items())
    }


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_fold(dataset_dir, output_dir, contract, fold, config):
    seed = int(config["seed"])
    configure_determinism(seed)
    train_samples, valid_samples = load_fold(dataset_dir, fold, contract)
    feature_names = IMU11_FEATURE_NAMES if contract == "imu11" else TOUCH23_FEATURE_NAMES
    scaler = fit_masked_scaler(train_samples, feature_names)
    x_train, y_train = tensors(train_samples, scaler)
    x_valid, y_valid = tensors(valid_samples, scaler)
    weights = torch.tensor(compute_sample_weights(train_samples), dtype=torch.float32)
    model = StrictTimeGridCnn(torch, len(feature_names))
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
    best_state = None
    best_score = -1.0
    best_loss = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        loss = train_epoch(
            model,
            x_train,
            y_train,
            weights,
            optimizer,
            int(config["batchSize"]),
            seed + epoch,
        )
        scheduler.step()
        train_metrics = evaluate(model, x_train, y_train)
        valid_metrics = evaluate(model, x_valid, y_valid)
        score = valid_metrics["balancedAccuracy"]
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and loss < best_loss):
            best_state = clone_state(model)
            best_score = score
            best_loss = loss
            best_epoch = epoch
        history.append({
            "epoch": epoch,
            "learningRate": optimizer.param_groups[0]["lr"],
            "trainLoss": loss,
            "trainBalancedAccuracy": train_metrics["balancedAccuracy"],
            "validBalancedAccuracy": valid_metrics["balancedAccuracy"],
        })
        if epoch - best_epoch >= int(config["earlyStoppingPatience"]):
            break
    model.load_state_dict(best_state)
    train_metrics = evaluate(model, x_train, y_train)
    valid_predictions = predict(model, x_valid)
    valid_metrics = metrics_from_predictions(y_valid.tolist(), valid_predictions)
    generalization_gap = train_metrics["balancedAccuracy"] - valid_metrics["balancedAccuracy"]

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    weights_payload = state_payload(model)
    write_json(output_root / "weights.json", weights_payload)
    weights_sha256 = hashlib.sha256((output_root / "weights.json").read_bytes()).hexdigest()
    write_json(output_root / "scaler.json", scaler)
    contract_payload = {
        "featureContract": "causal_time_grid_imu11_v2" if contract == "imu11" else "causal_time_grid_imu11_touch7_mask5_v3",
        "featureNames": list(feature_names),
        "matrixShape": [9, len(feature_names)],
        "gridOffsetsMs": [-15, -10, -5, 0, 5, 10, 15, 20, 25],
        "removedSourceIndices": [] if contract == "imu11" else [16, 17, 24],
    }
    write_json(output_root / "model_contract.json", contract_payload)
    with (output_root / "history.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (output_root / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as output:
        fields = ("sourceSha256", "sampleId", "groupId", "truth", "prediction", "correct")
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for sample, prediction in zip(valid_samples, valid_predictions):
            writer.writerow({
                "sourceSha256": sample["sourceSha256"],
                "sampleId": sample["sampleId"],
                "groupId": sample["groupId"],
                "truth": sample["classLabel"],
                "prediction": CLASS_LABELS[prediction],
                "correct": int(sample["label"] == prediction),
            })
    with (output_root / "source_split_manifest.csv").open("w", encoding="utf-8", newline="") as output:
        fields = ("sourceSha256", "sampleId", "groupId", "classLabel", "split", "trainingWeight")
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for sample, weight in zip(train_samples, compute_sample_weights(train_samples)):
            writer.writerow({**{field: sample.get(field, "") for field in fields}, "split": "train", "trainingWeight": weight})
        for sample in valid_samples:
            writer.writerow({**{field: sample.get(field, "") for field in fields}, "split": "valid", "trainingWeight": ""})
    report = {
        "contract": contract,
        "fold": fold,
        "evaluationScope": "single_device_cross_session_baseline",
        "seed": seed,
        "featureCount": len(feature_names),
        "parameterCount": sum(parameter.numel() for parameter in model.parameters()),
        "bestEpoch": best_epoch,
        "epochsCompleted": len(history),
        "trainCount": len(train_samples),
        "validCount": len(valid_samples),
        "trainGroupCount": len({sample["groupId"] for sample in train_samples}),
        "validGroupCount": len({sample["groupId"] for sample in valid_samples}),
        "trainMetrics": train_metrics,
        "validMetrics": valid_metrics,
        "generalizationGap": generalization_gap,
        "weightsSha256": weights_sha256,
        "config": config,
        "constantFeatures": [channel["name"] for channel in scaler["channels"] if channel["constant"]],
    }
    write_json(output_root / "training_report.json", report)
    return report


def default_config():
    return {
        "learningRate": 0.001,
        "minimumLearningRate": 0.00001,
        "weightDecay": 0.001,
        "batchSize": 64,
        "epochs": 80,
        "earlyStoppingPatience": 12,
        "gradientClipNorm": 1.0,
        "dropout": 0.15,
        "seed": 20260810,
        "device": "cpu",
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--contract", choices=("imu11", "touch23"), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    report = train_fold(args.dataset_dir, args.output_dir, args.contract, args.fold, default_config())
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
