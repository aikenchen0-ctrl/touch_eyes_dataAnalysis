#!/usr/bin/env python3
"""历史 schema v5 训练器，仅用于复现旧模型，不得处理 schema v6 因果数据。"""

LEGACY_ONLY = True

import argparse
import csv
import json
import math
import random
from pathlib import Path

try:
    from export_android_contact_encoding_report import MAP_SIZE
    from export_android_contact_encoding_report import collect_contact_stats
    from export_android_contact_encoding_report import covariance_values
    from export_android_contact_encoding_report import gaussian_map
    from export_android_contact_encoding_report import weighted_spread
except ModuleNotFoundError:
    from android.analysis.export_android_contact_encoding_report import MAP_SIZE
    from android.analysis.export_android_contact_encoding_report import collect_contact_stats
    from android.analysis.export_android_contact_encoding_report import covariance_values
    from android.analysis.export_android_contact_encoding_report import gaussian_map
    from android.analysis.export_android_contact_encoding_report import weighted_spread


DEFAULT_FEATURES = [
    # 旧 13 通道保留坐标、IMU 和时间信息，是所有接触区域方案的动态基线。
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
    "time_since_touch_down",
    "time_since_anchor",
]

DEFAULT_FRAME_INDICES = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

PATCH_FEATURES = [
    # patch9 是探索型接触特征集合，包含协方差、轴长、面积和字段质量，不代表最终最小输入。
    "contactPatch.cov_11",
    "contactPatch.cov_12",
    "contactPatch.cov_22",
    "contactPatch.sigmaX",
    "contactPatch.sigmaY",
    "touchNormalization.rawMajorPx",
    "touchNormalization.rawMinorPx",
    "touchNormalization.areaValue",
    "touchNormalization.confidence",
]

PATCH_FEATURES_WITHOUT_CONFIDENCE = [
    # 去掉 confidence 的版本用于检查模型是否真正学习接触形状，而不是依赖采集质量信号。
    name for name in PATCH_FEATURES
    if name != "touchNormalization.confidence"
]

GAUSSIAN_MAP_FEATURES = [f"gaussian.map.{index:02d}" for index in range(MAP_SIZE * MAP_SIZE)]

GAUSSIAN_SUMMARY_FEATURES = [
    # 高斯摘要用少量统计值近似 9x9 热图，减少通道数并保留接触扩散程度。
    "gaussian.center",
    "gaussian.sum",
    "gaussian.energy",
    "gaussian.spread",
    "gaussian.sigma_x_px",
    "gaussian.sigma_y_px",
    "gaussian.valid_mask",
]

MULTI_CHANNEL_FEATURES = [
    # 多通道方案直接暴露中心、长短轴、方向和有效性，便于模型学习可解释接触几何。
    "multi.center_x_norm",
    "multi.center_y_norm",
    "multi.major_px",
    "multi.minor_px",
    "multi.orientation_rad",
    "multi.valid_mask",
]

COVARIANCE_FEATURES = [
    # 协方差方案用三个矩阵项表达接触区域大小、方向和椭圆形状，是最紧凑的物理表达。
    "covariance.x_norm",
    "covariance.y_norm",
    "covariance.cov_11",
    "covariance.cov_12",
    "covariance.cov_22",
    "covariance.valid_mask",
]

FEATURE_PRESETS = {
    "old13": DEFAULT_FEATURES,
    "old13_raw_radius": DEFAULT_FEATURES + [
        "multi.major_px",
        "multi.minor_px",
        "multi.valid_mask",
    ],
    "old13_covariance": DEFAULT_FEATURES + COVARIANCE_FEATURES,
    "old13_multichannel": DEFAULT_FEATURES + MULTI_CHANNEL_FEATURES,
    "old13_gaussian_summary": DEFAULT_FEATURES + GAUSSIAN_SUMMARY_FEATURES,
    "old13_gaussian_map9x9": DEFAULT_FEATURES + GAUSSIAN_MAP_FEATURES + ["gaussian.valid_mask"],
    "gaussian_map9x9_only": GAUSSIAN_MAP_FEATURES + ["gaussian.valid_mask"],
    "multichannel_only": MULTI_CHANNEL_FEATURES,
    "covariance_only": COVARIANCE_FEATURES,
    "enhanced_patch21_no_confidence": DEFAULT_FEATURES + PATCH_FEATURES_WITHOUT_CONFIDENCE,
    "enhanced_patch22": DEFAULT_FEATURES + PATCH_FEATURES,
    "patch9": PATCH_FEATURES,
}


def main():
    args = parse_args()
    torch = import_torch()
    seed_everything(args.seed, torch)
    device = select_device(torch)

    feature_names = parse_csv_list(args.features) if args.features else preset_features(args.feature_preset)
    frame_indices = parse_int_list(args.frame_indices) if args.frame_indices else DEFAULT_FRAME_INDICES
    samples = load_samples(Path(args.manifest), feature_names, frame_indices, args.missing_frame_policy)
    if len(samples) < 4:
        raise RuntimeError("not_enough_samples")

    train_samples, valid_samples = stratified_split(samples, args.validation_ratio, args.seed)
    x_train, y_train, w_train = make_tensors(train_samples, torch)
    x_valid, y_valid, _ = make_tensors(valid_samples, torch)

    center = x_train.mean(dim=(0, 2))
    scale = x_train.std(dim=(0, 2), unbiased=False).clamp_min(1e-6)
    x_train = normalize(x_train, center, scale)
    x_valid = normalize(x_valid, center, scale)

    model = ImpactCnn(torch, len(feature_names), len(frame_indices)).to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    w_train = w_train.to(device)
    x_valid = x_valid.to(device)
    y_valid = y_valid.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(y_train, torch).to(device), reduction="none")

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

    train_metrics = evaluate(model, x_train, y_train, torch)
    valid_metrics = evaluate(model, x_valid, y_valid, torch)
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
        train_metrics,
        valid_metrics,
        history,
        args,
        torch,
    )
    if args.asset_dir:
        install_assets(Path(args.asset_dir), output_dir, args.model_id, args.display_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="android_patch21_no_confidence_pre5_post9")
    parser.add_argument("--display-name", default="android patch21 no confidence #1 -5...+9")
    parser.add_argument("--feature-preset", default="enhanced_patch21_no_confidence", choices=sorted(FEATURE_PRESETS.keys()))
    parser.add_argument("--features", default="")
    parser.add_argument("--frame-indices", default="")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--validation-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--asset-dir", default="")
    parser.add_argument("--missing-frame-policy", default="zero", choices=("zero", "skip", "error"))
    return parser.parse_args()


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("pytorch_not_installed") from error
    return torch


def select_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed, torch):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_samples(manifest_path, feature_names, frame_indices, missing_frame_policy):
    samples = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            strength = row.get("touch_strength", "")
            if strength not in ("heavy", "light"):
                continue
            path = resolve_path(row, manifest_path)
            if path is None:
                continue
            with path.open("r", encoding="utf-8") as data_file:
                data = json.load(data_file)
            matrix = sample_matrix(data, feature_names, frame_indices, missing_frame_policy)
            if matrix is None:
                continue
            sample_weight = finite_float(row.get("sample_weight"), 1.0)
            if sample_weight <= 0:
                sample_weight = 1.0
            label = 0 if strength == "heavy" else 1
            samples.append({
                "path": str(path),
                "label": label,
                "strength": strength,
                "matrix": matrix,
                "sampleWeight": sample_weight,
            })
    return samples


def resolve_path(row, manifest_path):
    for key in ("path", "raw_original_path", "raw_path", "chosen_path", "all_flat_path"):
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


def sample_matrix(data, feature_names, frame_indices, missing_frame_policy):
    by_index = {}
    for frame in data.get("frames") or []:
        by_index[int_value(frame.get("frameIndex"), 0)] = frame
    reference_frame_index = int_value(data.get("referenceFrameIndex"), 0)
    contact_values = contact_feature_values(data)
    rows = []
    for index in frame_indices:
        frame = by_index.get(reference_frame_index + index)
        if frame is None:
            if missing_frame_policy == "skip":
                return None
            if missing_frame_policy == "error":
                raise RuntimeError("missing_frame_index")
            # 缺帧默认补零，与 Android 端推理一致，避免短窗口样本被训练端额外筛掉。
            rows.append([0.0 for _ in feature_names])
            continue
        rows.append([feature_value(data, frame, name, contact_values) for name in feature_names])
    return rows


def frame_index_for_reference(data, relative_index):
    return int_value(data.get("referenceFrameIndex"), 0) + int(relative_index)


def feature_value(data, frame, name, contact_values=None):
    if name == "delta":
        return finite_float(frame.get("delta"), 0.0)
    if name in ("x", "userAcceleration.x"):
        return finite_float(frame.get("x"), 0.0)
    if name in ("y", "userAcceleration.y"):
        return finite_float(frame.get("y"), 0.0)
    if name in ("z", "userAcceleration.z"):
        return finite_float(frame.get("z"), 0.0)
    if name == "rotationRate.x":
        return finite_float(frame.get("rotationRateX"), 0.0)
    if name == "rotationRate.y":
        return finite_float(frame.get("rotationRateY"), 0.0)
    if name == "rotationRate.z":
        return finite_float(frame.get("rotationRateZ"), 0.0)
    if name == "gravity.x":
        return finite_float(frame.get("gravityX"), 0.0)
    if name == "gravity.y":
        return finite_float(frame.get("gravityY"), 0.0)
    if name == "gravity.z":
        return finite_float(frame.get("gravityZ"), 0.0)
    if name == "x_norm":
        return finite_float(frame.get("x_norm"), 0.0)
    if name == "y_norm":
        return finite_float(frame.get("y_norm"), 0.0)
    if name == "time_since_touch_down":
        return finite_float(frame.get("timeSinceTouchDown"), 0.0)
    if name == "time_since_anchor":
        return finite_float(frame.get("timeSinceAnchor"), 0.0)
    if name.startswith("contactPatch."):
        return contact_patch_value(data, name)
    if name.startswith("touchNormalization."):
        return touch_normalization_value(data, name)
    if contact_values is None:
        contact_values = contact_feature_values(data)
    if name in contact_values:
        return contact_values[name]
    return 0.0


def contact_feature_values(data):
    # 历史 JSON 没有 contactEncoding 时，从原始触摸字段重建三种 Contact Patch 表达，不丢原始样本。
    stats = collect_contact_stats(data)
    gaussian = gaussian_map(stats)
    covariance = covariance_values(stats)
    values = {
        "gaussian.center": gaussian[(MAP_SIZE * MAP_SIZE) // 2],
        "gaussian.sum": sum(gaussian),
        "gaussian.energy": sum(value * value for value in gaussian),
        "gaussian.spread": weighted_spread(gaussian),
        "gaussian.sigma_x_px": stats["sigmaXPx"],
        "gaussian.sigma_y_px": stats["sigmaYPx"],
        "gaussian.valid_mask": stats["validMask"],
        "multi.center_x_norm": stats["centerXNorm"],
        "multi.center_y_norm": stats["centerYNorm"],
        "multi.major_px": stats["rawMajorPx"],
        "multi.minor_px": stats["rawMinorPx"],
        "multi.orientation_rad": stats["orientationRad"],
        "multi.valid_mask": stats["validMask"],
        "covariance.x_norm": stats["centerXNorm"],
        "covariance.y_norm": stats["centerYNorm"],
        "covariance.cov_11": covariance[0],
        "covariance.cov_12": covariance[1],
        "covariance.cov_22": covariance[2],
        "covariance.valid_mask": stats["validMask"],
    }
    for index, value in enumerate(gaussian):
        values[f"gaussian.map.{index:02d}"] = value
    return values


def contact_patch_value(data, name):
    patch = dict_value(data.get("contactPatch"))
    covariance = dict_value(patch.get("covariance"))
    normalized = dict_value(patch.get("normalized"))
    raw = dict_value(patch.get("raw"))
    if name == "contactPatch.cov_11":
        return finite_float(covariance.get("cov_11"), 0.0)
    if name == "contactPatch.cov_12":
        return finite_float(covariance.get("cov_12"), 0.0)
    if name == "contactPatch.cov_22":
        return finite_float(covariance.get("cov_22"), 0.0)
    if name == "contactPatch.sigmaX":
        return finite_float(normalized.get("sigmaX"), 0.0)
    if name == "contactPatch.sigmaY":
        return finite_float(normalized.get("sigmaY"), 0.0)
    if name == "contactPatch.rawMajorPx":
        return finite_float(raw.get("majorPx"), 0.0)
    if name == "contactPatch.rawMinorPx":
        return finite_float(raw.get("minorPx"), 0.0)
    return 0.0


def touch_normalization_value(data, name):
    # confidence 只表示接触字段质量和兜底状态，不是轻重触真值。
    profile = dict_value(data.get("touchNormalization"))
    selected = dict_value(profile.get("selectedResult"))
    screen = dict_value(profile.get("screenScale"))
    covariance = dict_value(profile.get("covarianceMatrix"))
    baseline = dict_value(profile.get("sessionBaseline"))
    if name == "touchNormalization.majorValue":
        return finite_float(selected.get("majorValue"), 0.0)
    if name == "touchNormalization.minorValue":
        return finite_float(selected.get("minorValue"), 0.0)
    if name == "touchNormalization.areaValue":
        return finite_float(selected.get("areaValue"), 0.0)
    if name == "touchNormalization.confidence":
        return finite_float(selected.get("confidence"), 0.0)
    if name == "touchNormalization.rawMajorPx":
        return finite_float(selected.get("rawMajorPx"), 0.0)
    if name == "touchNormalization.rawMinorPx":
        return finite_float(selected.get("rawMinorPx"), 0.0)
    if name == "touchNormalization.screenAreaValue":
        return finite_float(screen.get("areaValue"), 0.0)
    if name == "touchNormalization.covarianceAreaValue":
        return finite_float(covariance.get("areaValue"), 0.0)
    if name == "touchNormalization.baselineAreaValue":
        return finite_float(baseline.get("areaValue"), 0.0)
    return 0.0


def stratified_split(samples, validation_ratio, seed):
    grouped = {0: [], 1: []}
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


def make_tensors(samples, torch):
    x_rows = []
    y_rows = []
    w_rows = []
    for sample in samples:
        matrix = sample["matrix"]
        channels = list(zip(*matrix))
        x_rows.append(channels)
        y_rows.append(sample["label"])
        w_rows.append(sample.get("sampleWeight", 1.0))
    x = torch.tensor(x_rows, dtype=torch.float32)
    y = torch.tensor(y_rows, dtype=torch.long)
    w = torch.tensor(w_rows, dtype=torch.float32)
    return x, y, w


def normalize(x, center, scale):
    return (x - center.view(1, -1, 1)) / scale.view(1, -1, 1)


def class_weights(y, torch):
    counts = torch.bincount(y, minlength=2).float().clamp_min(1.0)
    weights = counts.sum() / (counts * 2.0)
    return weights


class ImpactCnn:
    def __new__(cls, torch, input_channels, length):
        # 1D CNN 沿时间窗口卷积，通道维承载坐标、IMU、时间和 Contact Patch 特征。
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
        pooled = length // 4
        model.classifier = nn.Sequential(
            nn.Linear(128 * pooled, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 2),
        )

        def forward(x):
            out = model.features(x)
            out = out.flatten(1)
            return model.classifier(out)

        model.forward = forward
        return model


def train_one_epoch(model, x, y, sample_weight, optimizer, loss_fn, batch_size, torch):
    model.train()
    order = torch.randperm(x.shape[0], device=x.device)
    total_loss = 0.0
    total = 0
    for start in range(0, x.shape[0], batch_size):
        index = order[start:start + batch_size]
        xb = x[index]
        yb = y[index]
        wb = sample_weight[index]
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        if loss.dim() > 0:
            loss = loss * wb
            loss = loss.sum() / wb.sum().clamp_min(1e-6)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * xb.shape[0]
        total += xb.shape[0]
    return total_loss / max(1, total)


def evaluate(model, x, y, torch):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1)
        y_cpu = y.detach().cpu()
        pred_cpu = pred.detach().cpu()
        accuracy = float((pred_cpu == y_cpu).float().mean()) if y_cpu.numel() else 0.0
        confusion = [[0, 0], [0, 0]]
        for truth, guess in zip(y_cpu.tolist(), pred_cpu.tolist()):
            confusion[truth][guess] += 1
        recalls = []
        for label in (0, 1):
            total = sum(confusion[label])
            recalls.append(confusion[label][label] / total if total else 0.0)
        return {
            "accuracy": accuracy,
            "balancedAccuracy": sum(recalls) / len(recalls),
            "confusionMatrixHeavyLight": confusion,
            "heavyRecall": recalls[0],
            "lightRecall": recalls[1],
            "count": int(y.numel()),
        }


def evaluate_to_cpu(model, x, y, torch):
    return evaluate(model, x, y, torch)


def clone_state(model, torch):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


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
        torch):
    # 导出权重、scaler 和 featureNames，手机端严格按这些元数据复现训练时的输入张量。
    weights_name = "AispectImpactCNNWeights_" + model_id + ".json"
    scaler_name = "AispectImpactCNNScaler_" + model_id + ".json"
    weights = {}
    state = model.state_dict()
    for key in (
            "features.0.weight",
            "features.0.bias",
            "features.1.weight",
            "features.1.bias",
            "features.1.running_mean",
            "features.1.running_var",
            "features.4.weight",
            "features.4.bias",
            "features.5.weight",
            "features.5.bias",
            "features.5.running_mean",
            "features.5.running_var",
            "features.9.weight",
            "features.9.bias",
            "features.10.weight",
            "features.10.bias",
            "features.10.running_mean",
            "features.10.running_var",
            "classifier.0.weight",
            "classifier.0.bias",
            "classifier.3.weight",
            "classifier.3.bias"):
        weights[key] = tensor_to_json(state[key])

    scaler = {
        "center": tensor_to_list(center),
        "scale": tensor_to_list(scale),
        "featureNames": feature_names,
        "frameIndices": frame_indices,
        "frameCount": len(frame_indices),
    }
    report = {
        "modelId": model_id,
        "displayName": display_name,
        "labelOrder": ["heavy", "light"],
        "trainMetrics": train_metrics,
        "validMetrics": valid_metrics,
        "history": history,
        "args": vars(args),
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


def install_assets(asset_dir, output_dir, model_id, display_name):
    asset_dir.mkdir(parents=True, exist_ok=True)
    weights_name = "AispectImpactCNNWeights_" + model_id + ".json"
    scaler_name = "AispectImpactCNNScaler_" + model_id + ".json"
    (asset_dir / weights_name).write_text((output_dir / weights_name).read_text(encoding="utf-8"), encoding="utf-8")
    (asset_dir / scaler_name).write_text((output_dir / scaler_name).read_text(encoding="utf-8"), encoding="utf-8")
    catalog_path = asset_dir / "AispectImpactCNNModelCatalog.json"
    catalog = []
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog = [item for item in catalog if item.get("id") != model_id]
    catalog.insert(0, {
        "id": model_id,
        "displayName": display_name,
        "weightsResourceName": weights_name[:-5],
        "scalerResourceName": scaler_name[:-5],
    })
    write_json(catalog_path, catalog)


def tensor_to_json(tensor):
    return {
        "shape": list(tensor.shape),
        "values": tensor_to_list(tensor),
    }


def tensor_to_list(tensor):
    return [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def write_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def preset_features(name):
    return list(FEATURE_PRESETS.get(name, FEATURE_PRESETS["enhanced_patch21_no_confidence"]))


def dict_value(value):
    return value if isinstance(value, dict) else {}


def finite_float(value, fallback):
    try:
        output = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(output):
        return fallback
    return output


def int_value(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


if __name__ == "__main__":
    main()
