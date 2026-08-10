#!/usr/bin/env python3

import argparse
import csv
import json
import math
import pickle
import random
from collections import Counter
from pathlib import Path


CLASS_ORDER = ("thumb_light", "thumb_heavy", "index_light", "index_heavy")
FEATURE_NAMES = (
    "x_norm",
    "y_norm",
    "pressure",
    "size",
    "patch_sigma_x",
    "patch_sigma_y",
    "patch_log_ratio",
    "patch_log_area",
    "orientation_sin",
    "orientation_cos",
)
DEFAULT_PATCH_SIGMA = 0.012


def finite_float(value, fallback=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def clipped_log_ratio(numerator, denominator):
    if numerator <= 0 or denominator <= 0:
        return 0.0
    return max(-6.0, min(6.0, math.log(numerator / denominator)))


def first_touch_features(data):
    frames = data.get("touchFrames") or []
    if not frames:
        return None
    frame = frames[0] or {}
    patch = frame.get("contactPatch") or {}
    normalized = patch.get("normalized") or {}
    sigma_x = max(1e-9, finite_float(normalized.get("sigmaX"), DEFAULT_PATCH_SIGMA))
    sigma_y = max(1e-9, finite_float(normalized.get("sigmaY"), DEFAULT_PATCH_SIGMA))
    orientation = finite_float(normalized.get("orientationRad"), finite_float(frame.get("orientation"), 0.0))
    return [
        max(0.0, min(1.0, finite_float(frame.get("x_norm"), 0.0))),
        max(0.0, min(1.0, finite_float(frame.get("y_norm"), 0.0))),
        max(0.0, finite_float(frame.get("pressure"), 0.0)),
        max(0.0, finite_float(frame.get("size"), 0.0)),
        sigma_x,
        sigma_y,
        clipped_log_ratio(sigma_x, sigma_y),
        max(-20.0, min(0.0, math.log(max(1e-12, sigma_x * sigma_y)))),
        math.sin(orientation),
        math.cos(orientation),
    ]


def stratified_indices(labels, validation_ratio, seed):
    grouped = {index: [] for index in range(len(CLASS_ORDER))}
    for index, label in enumerate(labels):
        grouped[int(label)].append(index)
    rng = random.Random(seed)
    train = []
    valid = []
    for label in range(len(CLASS_ORDER)):
        items = list(grouped[label])
        rng.shuffle(items)
        valid_count = max(1, int(round(len(items) * validation_ratio)))
        valid.extend(items[:valid_count])
        train.extend(items[valid_count:])
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def build_export_payload(
        model_id,
        center,
        scale,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias):
    hidden_rows = len(hidden_weight)
    hidden_columns = len(hidden_weight[0]) if hidden_rows else 0
    output_rows = len(output_weight)
    output_columns = len(output_weight[0]) if output_rows else 0
    return {
        "schemaVersion": 1,
        "modelId": model_id,
        "displayName": "First touch early experimental v1",
        "featureNames": list(FEATURE_NAMES),
        "labelOrder": list(CLASS_ORDER),
        "center": list(center),
        "scale": list(scale),
        "hiddenWeightShape": [hidden_rows, hidden_columns],
        "hiddenWeight": hidden_weight,
        "hiddenBias": list(hidden_bias),
        "outputWeightShape": [output_rows, output_columns],
        "outputWeight": output_weight,
        "outputBias": list(output_bias),
    }


def resolve_path(row, manifest_path):
    value = row.get("path", "")
    if not value:
        return None
    candidate = Path(value)
    if candidate.exists():
        return candidate
    for root in [Path.cwd()] + list(manifest_path.resolve().parents):
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return None


def sample_weight(row):
    value = finite_float(row.get("sample_weight"), 1.0)
    return value if value > 0 else 1.0


def load_samples(manifest_path):
    samples = []
    with manifest_path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            label = row.get("class_label", "")
            if label not in CLASS_ORDER:
                continue
            path = resolve_path(row, manifest_path)
            if path is None:
                continue
            with path.open("r", encoding="utf-8") as data_file:
                data = json.load(data_file)
            features = first_touch_features(data)
            if features is None:
                continue
            samples.append({
                "id": row.get("id", ""),
                "path": str(path),
                "label": CLASS_ORDER.index(label),
                "classLabel": label,
                "features": features,
                "sampleWeight": sample_weight(row),
            })
    return samples


def cache_signature(manifest_path):
    stat = manifest_path.stat()
    return {
        "manifest": str(manifest_path.resolve()),
        "manifestSize": stat.st_size,
        "manifestMtimeNs": stat.st_mtime_ns,
        "featureNames": list(FEATURE_NAMES),
    }


def load_or_cache_samples(manifest_path, cache_path):
    signature = cache_signature(manifest_path)
    if cache_path and cache_path.exists():
        with cache_path.open("rb") as source:
            payload = pickle.load(source)
        if payload.get("signature") == signature:
            return payload.get("samples", [])
    samples = load_samples(manifest_path)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as output:
            pickle.dump({"signature": signature, "samples": samples}, output)
    return samples


def scaler_for_indices(samples, indices):
    columns = [[] for _ in FEATURE_NAMES]
    for index in indices:
        for feature_index, value in enumerate(samples[index]["features"]):
            columns[feature_index].append(value)
    center = []
    scale = []
    for values in columns:
        mean = sum(values) / max(1, len(values))
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values))
        center.append(mean)
        scale.append(max(math.sqrt(variance), 1e-6))
    return center, scale


def class_weights(labels, indices):
    counts = Counter(labels[index] for index in indices)
    total = max(1, len(indices))
    return [total / max(1, len(CLASS_ORDER) * counts[label]) for label in range(len(CLASS_ORDER))]


def should_update_best_state(has_validation, score, best_score):
    return not has_validation or score >= best_score


def metrics_for_predictions(labels, predictions):
    matrix = [[0 for _ in CLASS_ORDER] for _ in CLASS_ORDER]
    for truth, predicted in zip(labels, predictions):
        matrix[int(truth)][int(predicted)] += 1
    recalls = {}
    correct = 0
    total = 0
    for index, label in enumerate(CLASS_ORDER):
        row_total = sum(matrix[index])
        correct += matrix[index][index]
        total += row_total
        recalls[label] = matrix[index][index] / row_total if row_total else 0.0
    return {
        "count": total,
        "accuracy": correct / total if total else 0.0,
        "balancedAccuracy": sum(recalls.values()) / len(CLASS_ORDER),
        "classRecalls": recalls,
        "confusionMatrix": matrix,
    }


def tensor_rows(samples, indices, center, scale, torch):
    features = []
    labels = []
    weights = []
    for index in indices:
        sample = samples[index]
        features.append([
            (value - center[position]) / scale[position]
            for position, value in enumerate(sample["features"])
        ])
        labels.append(sample["label"])
        weights.append(sample["sampleWeight"])
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(weights, dtype=torch.float32),
    )


def evaluate(model, x, y, batch_size, torch, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            logits = model(x[start:start + batch_size].to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return metrics_for_predictions(y.tolist(), predictions)


def train_model(samples, args, torch):
    labels = [sample["label"] for sample in samples]
    if args.split_strategy == "all":
        train_indices = list(range(len(samples)))
        valid_indices = []
    else:
        train_indices, valid_indices = stratified_indices(labels, args.validation_ratio, args.seed)
    center, scale = scaler_for_indices(samples, train_indices)
    x_train, y_train, w_train = tensor_rows(samples, train_indices, center, scale, torch)
    if valid_indices:
        x_valid, y_valid, _ = tensor_rows(samples, valid_indices, center, scale, torch)
    else:
        x_valid = torch.empty((0, len(FEATURE_NAMES)), dtype=torch.float32)
        y_valid = torch.empty((0,), dtype=torch.long)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Sequential(
        torch.nn.Linear(len(FEATURE_NAMES), args.hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden_size, len(CLASS_ORDER)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights(labels, train_indices), dtype=torch.float32, device=device),
        reduction="none",
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    history = []
    best_score = -1.0
    best_epoch = 0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(y_train), generator=generator)
        total_loss = 0.0
        total_weight = 0.0
        for start in range(0, len(order), args.batch_size):
            indexes = order[start:start + args.batch_size]
            xb = x_train[indexes].to(device)
            yb = y_train[indexes].to(device)
            wb = w_train[indexes].to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = loss_fn(model(xb), yb)
            loss = (losses * wb).sum() / wb.sum().clamp_min(1e-6)
            loss.backward()
            optimizer.step()
            total_loss += float((losses * wb).sum().detach().cpu())
            total_weight += float(wb.sum().detach().cpu())
        train_metrics = evaluate(model, x_train, y_train, args.batch_size, torch, device)
        valid_metrics = evaluate(model, x_valid, y_valid, args.batch_size, torch, device) if valid_indices else None
        score = valid_metrics["balancedAccuracy"] if valid_metrics else train_metrics["balancedAccuracy"]
        history.append({
            "epoch": epoch,
            "trainLoss": total_loss / max(total_weight, 1e-6),
            "trainBalancedAccuracy": train_metrics["balancedAccuracy"],
            "validBalancedAccuracy": valid_metrics["balancedAccuracy"] if valid_metrics else 0.0,
        })
        if should_update_best_state(bool(valid_indices), score, best_score):
            best_score = score
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    train_metrics = evaluate(model, x_train, y_train, args.batch_size, torch, device)
    valid_metrics = evaluate(model, x_valid, y_valid, args.batch_size, torch, device) if valid_indices else None
    return model.cpu(), center, scale, train_metrics, valid_metrics, history, best_epoch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default="first_touch_early_v1")
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--validation-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--split-strategy", choices=("random", "all"), default="random")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch

    samples = load_or_cache_samples(args.manifest, args.feature_cache)
    if not samples:
        raise RuntimeError("empty_training_samples")
    model, center, scale, train_metrics, valid_metrics, history, best_epoch = train_model(samples, args, torch)
    state = model.state_dict()
    payload = build_export_payload(
        args.model_id,
        center,
        scale,
        state["0.weight"].tolist(),
        state["0.bias"].tolist(),
        state["2.weight"].tolist(),
        state["2.bias"].tolist(),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / ("AispectFirstTouchModel_" + args.model_id + ".json")
    model_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report = {
        "modelId": args.model_id,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "sampleCount": len(samples),
        "classCounts": dict(Counter(sample["classLabel"] for sample in samples)),
        "bestEpoch": best_epoch,
        "trainMetrics": train_metrics,
        "validMetrics": valid_metrics,
        "history": history,
        "modelPath": str(model_path),
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "modelId": args.model_id,
        "sampleCount": len(samples),
        "bestEpoch": best_epoch,
        "trainBalancedAccuracy": train_metrics["balancedAccuracy"],
        "validBalancedAccuracy": valid_metrics["balancedAccuracy"] if valid_metrics else None,
    }, indent=2))


if __name__ == "__main__":
    main()
