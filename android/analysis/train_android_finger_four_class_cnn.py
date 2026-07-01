#!/usr/bin/env python3
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from train_android_touch_cnn import FEATURE_PRESETS
from train_android_touch_cnn import clone_state
from train_android_touch_cnn import import_torch
from train_android_touch_cnn import normalize
from train_android_touch_cnn import parse_csv_list
from train_android_touch_cnn import parse_int_list
from train_android_touch_cnn import preset_features
from train_android_touch_cnn import sample_matrix
from train_android_touch_cnn import select_device
from train_android_touch_cnn import seed_everything
from train_android_touch_cnn import tensor_to_json
from train_android_touch_cnn import tensor_to_list
from train_android_touch_cnn import train_one_epoch
from train_android_touch_cnn import write_json


CLASS_ORDER = [
    "thumb_light",
    "thumb_heavy",
    "index_light",
    "index_heavy",
]


def main():
    args = parse_args()
    torch = import_torch()
    seed_everything(args.seed, torch)
    device = select_device(torch)

    feature_names = parse_csv_list(args.features) if args.features else preset_features(args.feature_preset)
    frame_indices = parse_int_list(args.frame_indices)
    samples = load_samples(Path(args.manifest), feature_names, frame_indices, args.missing_frame_policy)
    if len(samples) < len(CLASS_ORDER) * 2:
        raise RuntimeError("not_enough_samples")

    train_samples, valid_samples = split_samples(samples, args.validation_ratio, args.seed, args.split_strategy)
    x_train, y_train, w_train = make_tensors(train_samples, torch)
    x_valid, y_valid, _ = make_tensors(valid_samples, torch)

    center = x_train.mean(dim=(0, 2))
    scale = x_train.std(dim=(0, 2), unbiased=False).clamp_min(1e-6)
    x_train = normalize(x_train, center, scale)
    x_valid = normalize(x_valid, center, scale)

    model = FingerFourClassCnn(torch, len(feature_names), len(frame_indices), len(CLASS_ORDER)).to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    w_train = w_train.to(device)
    x_valid = x_valid.to(device)
    y_valid = y_valid.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_multi(y_train, len(CLASS_ORDER), torch).to(device), reduction="none")

    best_state = None
    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, x_train, y_train, w_train, optimizer, loss_fn, args.batch_size, torch)
        train_metrics = evaluate(model, x_train, y_train, torch)
        valid_metrics = evaluate(model, x_valid, y_valid, torch)
        row = {
            "epoch": epoch,
            "trainLoss": train_loss,
            "trainAccuracy": train_metrics["accuracy"],
            "validAccuracy": valid_metrics["accuracy"],
            "validBalancedAccuracy": valid_metrics["balancedAccuracy"],
        }
        history.append(row)
        score = valid_metrics["balancedAccuracy"]
        if score >= best_score:
            best_score = score
            best_state = clone_state(model, torch)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_model(
        output_dir,
        args.model_id,
        args.display_name,
        model,
        center,
        scale,
        feature_names,
        frame_indices,
        evaluate(model, x_train, y_train, torch),
        evaluate(model, x_valid, y_valid, torch),
        history,
        args,
        train_samples,
        valid_samples,
        torch,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="android_finger_four_class_patch22_pre5_post9")
    parser.add_argument("--display-name", default="android finger four class patch22 -5...+9")
    parser.add_argument("--feature-preset", default="enhanced_patch22", choices=sorted(FEATURE_PRESETS.keys()))
    parser.add_argument("--features", default="")
    parser.add_argument("--frame-indices", default="-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--validation-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--missing-frame-policy", default="zero", choices=("zero", "skip", "error"))
    parser.add_argument("--split-strategy", default="random", choices=("random", "temporal"))
    return parser.parse_args()


def load_samples(manifest_path, feature_names, frame_indices, missing_frame_policy):
    samples = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            class_label = row.get("class_label", "")
            if class_label not in CLASS_ORDER:
                continue
            path = resolve_path(row, manifest_path)
            if path is None:
                continue
            with path.open("r", encoding="utf-8") as data_file:
                data = json.load(data_file)
            matrix = sample_matrix(data, feature_names, frame_indices, missing_frame_policy)
            if matrix is None:
                continue
            samples.append({
                "path": str(path),
                "label": CLASS_ORDER.index(class_label),
                "classLabel": class_label,
                "matrix": matrix,
                "createdAt": sample_created_at(row, data),
                "sampleWeight": sample_weight_for_row(row),
            })
    return samples


def resolve_path(row, manifest_path):
    value = row.get("path")
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


def stratified_split(samples, validation_ratio, seed):
    grouped = {index: [] for index in range(len(CLASS_ORDER))}
    for sample in samples:
        grouped[sample["label"]].append(sample)
    rng = random.Random(seed)
    train = []
    valid = []
    for label, items in grouped.items():
        rng.shuffle(items)
        valid_count = max(1, int(round(len(items) * validation_ratio)))
        valid.extend(items[:valid_count])
        train.extend(items[valid_count:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def temporal_split(samples, validation_ratio):
    grouped = {index: [] for index in range(len(CLASS_ORDER))}
    for sample in samples:
        grouped[sample["label"]].append(sample)
    train = []
    valid = []
    for label, items in grouped.items():
        ordered = sorted(items, key=lambda sample: (sample.get("createdAt", 0.0), sample.get("path", "")))
        valid_count = max(1, int(round(len(ordered) * validation_ratio)))
        train.extend(ordered[:-valid_count])
        valid.extend(ordered[-valid_count:])
    return train, valid


def split_samples(samples, validation_ratio, seed, strategy):
    if strategy == "temporal":
        return temporal_split(samples, validation_ratio)
    return stratified_split(samples, validation_ratio, seed)


def sample_created_at(row, data):
    value = row.get("created_at", "")
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        return float(data.get("createdAt", 0))
    except (TypeError, ValueError):
        return 0.0


def sample_weight_for_row(row):
    try:
        value = float(row.get("sample_weight", 1.0))
    except (TypeError, ValueError):
        value = 1.0
    if value <= 0:
        return 1.0
    return value


def make_tensors(samples, torch):
    x_rows = []
    y_rows = []
    w_rows = []
    for sample in samples:
        channels = list(zip(*sample["matrix"]))
        x_rows.append(channels)
        y_rows.append(sample["label"])
        w_rows.append(sample.get("sampleWeight", 1.0))
    return torch.tensor(x_rows, dtype=torch.float32), torch.tensor(y_rows, dtype=torch.long), torch.tensor(w_rows, dtype=torch.float32)


def class_weights_multi(y, class_count, torch):
    counts = torch.bincount(y, minlength=class_count).float().clamp_min(1.0)
    return counts.sum() / (counts * class_count)


class FingerFourClassCnn:
    def __new__(cls, torch, input_channels, length, class_count):
        # 中文注释：沿时间窗口做一维卷积，输出四个手指和轻重组合类别。
        nn = torch.nn
        model = nn.Module()
        model.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.10),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        model.classifier = nn.Sequential(
            nn.Linear(128 * (length // 4), 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, class_count),
        )

        def forward(x):
            out = model.features(x)
            out = out.flatten(1)
            return model.classifier(out)

        model.forward = forward
        return model


def evaluate(model, x, y, torch):
    model.eval()
    class_count = len(CLASS_ORDER)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1)
        accuracy = float((pred == y).float().mean()) if y.numel() else 0.0
        confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
        for truth, guess in zip(y.tolist(), pred.tolist()):
            confusion[truth][guess] += 1
        recalls = []
        for label in range(class_count):
            total = sum(confusion[label])
            recalls.append(confusion[label][label] / total if total else 0.0)
        return {
            "accuracy": accuracy,
            "balancedAccuracy": sum(recalls) / len(recalls),
            "classRecalls": dict(zip(CLASS_ORDER, recalls)),
            "confusionMatrix": confusion,
            "count": int(y.numel()),
        }


def export_model(
        output_dir,
        model_id,
        display_name,
        model,
        center,
        scale,
        feature_names,
        frame_indices,
        train_metrics,
        valid_metrics,
        history,
        args,
        train_samples,
        valid_samples,
        torch):
    weights_name = "AispectImpactCNNWeights_" + model_id + ".json"
    scaler_name = "AispectImpactCNNScaler_" + model_id + ".json"
    state = model.state_dict()
    weights = {}
    for key in state:
        weights[key] = tensor_to_json(state[key])
    scaler = {
        "center": tensor_to_list(center),
        "scale": tensor_to_list(scale),
        "featureNames": feature_names,
        "frameIndices": frame_indices,
        "frameCount": len(frame_indices),
        "labelOrder": CLASS_ORDER,
        "classCount": len(CLASS_ORDER),
    }
    report = {
        "modelId": model_id,
        "displayName": display_name,
        "labelOrder": CLASS_ORDER,
        "trainMetrics": train_metrics,
        "validMetrics": valid_metrics,
        "trainClassCounts": dict(Counter(sample["classLabel"] for sample in train_samples)),
        "validClassCounts": dict(Counter(sample["classLabel"] for sample in valid_samples)),
        "history": history,
        "args": vars(args),
        "mobileRuntimeNote": "requires_four_class_output_support",
    }
    catalog_entry = {
        "id": model_id,
        "displayName": display_name,
        "weightsResourceName": weights_name[:-5],
        "scalerResourceName": scaler_name[:-5],
    }
    write_json(output_dir / weights_name, weights)
    write_json(output_dir / scaler_name, scaler)
    write_json(output_dir / "training_report.json", report)
    write_json(output_dir / "catalog_entry.json", catalog_entry)


if __name__ == "__main__":
    main()
