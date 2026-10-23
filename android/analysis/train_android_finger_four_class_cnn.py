#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import median
from collection_annotations import is_raw_sample_path, require_trainable, read_annotation, exclusion_reason

from causal_press_features import CAUSAL_FEATURE_PRESETS
from causal_press_features import TIME_GRID_FEATURE_CONTRACTS
from cross_device_causal_features import TIME_GRID_OFFSETS_MS
from cross_device_causal_features import build_time_grid_matrix
from causal_press_features import build_matrix as build_causal_matrix
from causal_press_features import feature_names as causal_feature_names
from training_common import clone_state
from training_common import finite_float
from training_common import import_torch
from training_common import normalize
from training_common import parse_csv_list
from training_common import parse_int_list
from training_common import select_device
from training_common import seed_everything
from training_common import tensor_to_json
from training_common import tensor_to_list
from training_common import write_json


CLASS_ORDER = [
    "thumb_light",
    "thumb_heavy",
    "index_light",
    "index_heavy",
]

PLANNED_MOTION_STATES = [
    "handheld_static",
    "placed_static",
    "walking",
]

LEGACY_FEATURE_PRESET_NAMES = (
    "old13",
    "old13_raw_radius",
    "old13_covariance",
    "old13_multichannel",
    "old13_gaussian_summary",
    "old13_gaussian_map9x9",
    "gaussian_map9x9_only",
    "multichannel_only",
    "covariance_only",
    "enhanced_patch21_no_confidence",
    "enhanced_patch22",
    "patch9",
)

ALL_FEATURE_PRESETS = tuple(sorted(set(CAUSAL_FEATURE_PRESETS) | set(LEGACY_FEATURE_PRESET_NAMES)))


def main():
    args = parse_args()
    validate_cli_args(args)
    frame_indices = parse_int_list(args.frame_indices)
    if args.prepare_planned_manifests:
        summary = prepare_planned_manifests(
            Path(args.dataset_dir),
            Path(args.output_dir),
            frame_indices,
            args.temporal_validation_ratio,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    torch = import_torch()
    configure_determinism(torch, args.deterministic)
    seed_everything(args.seed, torch)
    device = select_device(torch)

    feature_names = causal_feature_names(args.feature_preset)
    samples = load_or_cache_samples(
        Path(args.manifest),
        feature_names,
        frame_indices,
        args.missing_frame_policy,
        Path(args.sample_cache) if args.sample_cache else None,
        args.anchor_mode,
        args.require_press_active,
        args.feature_preset,
        Path(args.matrix_jsonl) if args.matrix_jsonl else None,
    )
    if len(samples) < len(CLASS_ORDER) * 2:
        raise RuntimeError("not_enough_samples")

    feature_names = transform_sample_matrices(samples, feature_names, args.matrix_transform)

    train_samples, valid_samples = split_samples(samples, args.validation_ratio, args.seed, args.split_strategy)
    validate_session_split_integrity(train_samples, valid_samples)
    if args.group_balance:
        apply_collection_class_balance(train_samples)
    x_train, y_train, w_train = make_tensors(train_samples, torch)
    if valid_samples:
        x_valid, y_valid, _ = make_tensors(valid_samples, torch)
    else:
        x_valid = x_train[:0].clone()
        y_valid = y_train[:0].clone()

    center = x_train.mean(dim=(0, 2))
    scale = x_train.std(dim=(0, 2), unbiased=False).clamp_min(1e-6)
    x_train = normalize(x_train, center, scale)
    x_valid = normalize(x_valid, center, scale)

    model = FingerFourClassCnn(torch, len(feature_names), len(frame_indices), len(CLASS_ORDER)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_multi(y_train, len(CLASS_ORDER), torch).to(device), reduction="none")

    best_state = None
    best_score = (-1.0, -1.0, -1.0)
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch_on_device(
            model,
            x_train,
            y_train,
            w_train,
            optimizer,
            loss_fn,
            args.batch_size,
            torch,
            device,
        )
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
        # 中文注释：最差类召回单独决定部署价值，不能被多数类拉高的平衡准确率掩盖。
        score = checkpoint_score(valid_metrics)
        if should_update_best_state(bool(valid_samples), score, best_score):
            best_score = score
            best_epoch = epoch
            best_state = clone_state(model, torch)
        if valid_samples and should_stop_training(epoch, best_epoch, args.early_stopping_patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_metrics = evaluate(model, x_train, y_train, torch)
    valid_metrics = evaluate(model, x_valid, y_valid, torch)
    train_subgroup_metrics = evaluate_subgroups(model, x_train, y_train, train_samples, torch)
    valid_subgroup_metrics = evaluate_subgroups(model, x_valid, y_valid, valid_samples, torch)
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
        train_subgroup_metrics,
        valid_subgroup_metrics,
        history,
        best_epoch,
        args,
        train_samples,
        valid_samples,
        torch,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="")
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--prepare-planned-manifests", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="android_finger_four_class_causal_imu11_touch9_v1_pre5_post9")
    parser.add_argument("--display-name", default="android finger four class causal imu11 touch9 v1 -5...+9")
    parser.add_argument("--feature-preset", default="causal_imu11_touch9_v1", choices=ALL_FEATURE_PRESETS)
    parser.add_argument("--features", default="")
    parser.add_argument("--frame-indices", default="-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--validation-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--missing-frame-policy", default="zero", choices=("zero", "skip", "error"))
    parser.add_argument("--anchor-mode", default="down", choices=("down", "lift"))
    parser.add_argument("--split-strategy", default="random", choices=("random", "temporal", "predefined", "all"))
    parser.add_argument("--temporal-validation-ratio", type=float, default=0.20)
    parser.add_argument("--sample-cache", default="")
    parser.add_argument(
        "--matrix-jsonl",
        default="",
        help="使用预生成矩阵 JSONL，必须与清单的 sourceSha256 和契约完全一致",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument(
        "--matrix-transform",
        default="none",
        choices=("none", "imu11_no_time", "invariant_shape", "relative_touch", "baseline_imu"),
    )
    parser.add_argument("--group-balance", action="store_true")
    parser.add_argument("--require-press-active", action="store_true", default=True)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args(argv)


def validate_cli_args(args):
    if args.prepare_planned_manifests:
        if not args.dataset_dir:
            raise RuntimeError("dataset_dir_required_for_prepare")
        if not 0.0 < args.temporal_validation_ratio < 1.0:
            raise RuntimeError("invalid_temporal_validation_ratio")
        return
    if not args.manifest:
        raise RuntimeError("manifest_required_for_training")
    if args.anchor_mode != "down":
        raise RuntimeError("press_phase_requires_down_anchor")
    if not args.require_press_active:
        raise RuntimeError("press_active_filter_required")
    if args.feature_preset not in CAUSAL_FEATURE_PRESETS or args.features:
        raise RuntimeError("causal_press_feature_preset_required")
    if args.feature_preset in TIME_GRID_FEATURE_CONTRACTS and parse_int_list(args.frame_indices) != list(range(-3, 6)):
        raise RuntimeError("time_grid_frame_indices_required")
    if args.matrix_transform != "none":
        raise RuntimeError("causal_press_matrix_transform_required")


def load_or_cache_samples(
        manifest_path,
        feature_names,
        frame_indices,
        missing_frame_policy,
        cache_path,
        anchor_mode="down",
        require_press_active=True,
        feature_contract="",
        matrix_jsonl=None):
    if anchor_mode == "down" and not require_press_active:
        raise RuntimeError("press_active_filter_required")
    signature = sample_cache_signature(
        feature_names,
        frame_indices,
        missing_frame_policy,
        anchor_mode,
        require_press_active,
        feature_contract,
    )
    if matrix_jsonl is not None:
        matrix_path = Path(matrix_jsonl)
        if not matrix_path.is_file():
            raise RuntimeError("matrix_jsonl_missing")
        signature["matrixJsonl"] = str(matrix_path.resolve())
        signature["matrixJsonlSha256"] = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
        signature["manifestSha256"] = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    else:
        signature["collectionReviewDigest"] = manifest_review_digest(manifest_path)
    if cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as file:
            payload = pickle.load(file)
        if payload.get("signature") != signature:
            raise RuntimeError("sample_cache_signature_mismatch")
        samples = payload.get("samples", [])
        if matrix_jsonl is not None:
            return samples
        # 复核可在缓存生成后发生；已失效的缓存必须重建，不能悄悄训练旧标签。
        for sample in samples:
            source = Path(sample["path"])
            data = json.loads(source.read_text(encoding="utf-8"))
            annotation = require_trainable(source, data)
            if sample.get("collectionAnnotation") != annotation:
                raise RuntimeError("sample_cache_review_changed_rebuild_required")
        apply_manifest_splits(samples, manifest_path, allow_manifest_subset=anchor_mode == "down")
        return samples

    if matrix_jsonl is None:
        samples = load_samples(
            manifest_path,
            feature_names,
            frame_indices,
            missing_frame_policy,
            anchor_mode,
            require_press_active,
            feature_contract,
        )
    else:
        samples = load_matrix_samples(
            manifest_path,
            matrix_jsonl,
            feature_names,
            frame_indices,
            feature_contract,
        )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with temporary.open("wb") as file:
            pickle.dump({"signature": signature, "samples": samples}, file, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(cache_path)
    return samples


def manifest_review_digest(manifest_path):
    # 将被排除样本的复核也纳入缓存身份，恢复标签后必须重新构建，不能永久漏采。
    entries = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            path = resolve_path(row, manifest_path)
            if path is None:
                entries.append((row.get("path", ""), "missing", ""))
                continue
            raw = path.read_bytes()
            annotation = read_annotation(path, json.loads(raw))
            entries.append((str(path.resolve()), hashlib.sha256(raw).hexdigest(),
                            json.dumps(annotation, sort_keys=True)))
    return hashlib.sha256(json.dumps(sorted(entries)).encode("utf-8")).hexdigest()


def sample_cache_signature(
        feature_names,
        frame_indices,
        missing_frame_policy,
        anchor_mode,
        require_press_active,
        feature_contract,
):
    selected_contract = feature_contract or feature_contract_for_names(feature_names)
    signature = {
        "featureNames": list(feature_names),
        "frameIndices": list(frame_indices),
        "missingFramePolicy": missing_frame_policy,
        "anchorMode": anchor_mode,
        "requirePressActive": bool(require_press_active),
        "causalPressActiveV1": anchor_mode == "down",
        "causalPressFeaturesV1": "" if selected_contract in TIME_GRID_FEATURE_CONTRACTS else selected_contract,
        "featureContract": selected_contract,
        "platformAdapterVersion": "",
        "coordinateTransformVersion": "",
        "gridRateHz": None,
        "gridOffsetsMs": [],
        "interpolation": "",
        "maxGapRule": "",
    }
    if selected_contract in TIME_GRID_FEATURE_CONTRACTS:
        signature.update({
            "platformAdapterVersion": "cross_device_v1",
            "coordinateTransformVersion": "cross_device_v1",
            "gridRateHz": 200,
            "gridOffsetsMs": list(TIME_GRID_OFFSETS_MS),
            "interpolation": "linear_sensor_time",
            "maxGapRule": "max(2*5ms,4*median_dt)",
        })
    return signature


def checkpoint_score(metrics):
    """优先优化最差类召回，再以平衡准确率和准确率打破并列。"""
    return (
        float(metrics.get("minRecall", 0.0)),
        float(metrics.get("balancedAccuracy", 0.0)),
        float(metrics.get("accuracy", 0.0)),
    )


def apply_manifest_splits(samples, manifest_path, allow_manifest_subset=False):
    rows_by_id = {}
    rows_by_path = {}
    manifest_count = 0
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            manifest_count += 1
            sample_id = row.get("id", "")
            path = row.get("path", "")
            if sample_id:
                rows_by_id[sample_id] = row
            if path:
                rows_by_path[path] = row
    missing = 0
    for sample in samples:
        row = rows_by_id.get(sample.get("id", ""))
        if row is None:
            row = rows_by_path.get(sample.get("path", ""))
        if row is None:
            missing += 1
            continue
        sample["split"] = row.get("split", "")
        sample["collectionName"] = row.get("collection_name", sample.get("collectionName", "unknown"))
        sample["baselineGroup"] = baseline_group_for_row(row)
        sample["motionState"] = row.get("motion_state", sample.get("motionState", "unknown"))
        sample["durationBucket"] = row.get("duration_bucket", sample.get("durationBucket", "unknown"))
        sample["rawSessionId"] = row.get("raw_session_id", sample.get("rawSessionId", ""))
        sample["sampleWeight"] = sample_weight_for_row(row)
    if missing or (not allow_manifest_subset and manifest_count != len(samples)):
        raise RuntimeError("sample_cache_manifest_mismatch")


def should_stop_training(epoch, best_epoch, patience):
    return patience > 0 and epoch - best_epoch >= patience


def should_update_best_state(has_validation, score, best_score):
    return not has_validation or score >= best_score


def sample_matrix_for_anchor(
        data,
        feature_names,
        frame_indices,
        missing_frame_policy,
        anchor_mode,
        feature_contract="",
        return_quality=False):
    if anchor_mode == "down":
        selected_contract = feature_contract or feature_contract_for_names(feature_names)
        if selected_contract in TIME_GRID_FEATURE_CONTRACTS:
            try:
                payload = build_time_grid_matrix(data, data, selected_contract)
            except RuntimeError:
                if missing_frame_policy == "skip":
                    return None
                raise
            return payload if return_quality else payload["matrix"]
        if selected_contract:
            return build_causal_matrix(data, selected_contract, frame_indices, missing_frame_policy)
    if anchor_mode not in ("down", "lift"):
        raise RuntimeError("unknown_anchor_mode:" + str(anchor_mode))
    # 中文注释：该分支只供历史 API 兼容，主入口在 validate_cli_args 中拒绝 legacy preset。
    from legacy.train_android_touch_cnn_legacy import sample_matrix as legacy_sample_matrix
    reference_index = int(data.get(
        "downReferenceFrameIndex" if anchor_mode == "down" else "liftReferenceFrameIndex",
        data.get("referenceFrameIndex", 0),
    ))
    aligned = dict(data)
    aligned["referenceFrameIndex"] = reference_index
    return legacy_sample_matrix(aligned, feature_names, frame_indices, missing_frame_policy)


def feature_contract_for_names(feature_names):
    names = tuple(feature_names)
    for preset, expected_names in CAUSAL_FEATURE_PRESETS.items():
        if names == tuple(expected_names):
            return preset
    return ""


def preset_features(name):
    """历史测试/复现兼容接口；主因果训练不会调用。"""
    from legacy.train_android_touch_cnn_legacy import preset_features as legacy_preset_features
    return legacy_preset_features(name)


def sample_matrix(data, feature_names, frame_indices, missing_frame_policy):
    """历史测试/复现兼容接口；主因果训练不会调用。"""
    from legacy.train_android_touch_cnn_legacy import sample_matrix as legacy_sample_matrix
    return legacy_sample_matrix(data, feature_names, frame_indices, missing_frame_policy)


def is_press_active_window(data, frame_indices):
    last_post_down_index = max((index for index in frame_indices if index >= 0), default=0)
    lift_index = int(data.get("liftReferenceFrameIndex", -1))
    return lift_index >= last_post_down_index


def is_time_grid_press_active(data):
    alignment = data.get("timeAlignment") or {}
    down_ns = _positive_int(alignment.get("downEventElapsedRealtimeNanos"))
    lift_ns = _positive_int(alignment.get("liftEventElapsedRealtimeNanos"))
    if down_ns <= 0 or lift_ns <= 0:
        return False
    return lift_ns >= down_ns + max(TIME_GRID_OFFSETS_MS) * 1_000_000


def _positive_int(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def duration_bucket(lift_reference_frame_index):
    index = int(lift_reference_frame_index)
    if index <= 14:
        return "q1"
    if index <= 18:
        return "q2"
    if index <= 23:
        return "q3"
    if index <= 60:
        return "q4"
    return "extreme"


def schema5_manifest_row(path, data, required_frame_indices, sha256):
    if int(data.get("schemaVersion", 0)) != 5:
        return None, "unsupported_schema"
    if data.get("recordStatus") != "accepted":
        return None, "record_not_accepted"
    if not data.get("windowComplete"):
        return None, "incomplete_window"
    if int(data.get("downReferenceFrameIndex", -1)) != 0:
        return None, "invalid_down_reference"
    if int(data.get("preDownFrameCount", -1)) < 15:
        return None, "insufficient_pre_down_frames"
    if int(data.get("postLiftFrameCount", -1)) < 15:
        return None, "insufficient_post_lift_frames"

    label = data.get("label") or {}
    finger_type = str(label.get("fingerType") or "")
    touch_strength = str(label.get("touchStrength") or "")
    motion_state = str(label.get("motionState") or "")
    class_label = finger_type + "_" + touch_strength
    if class_label not in CLASS_ORDER:
        return None, "unsupported_class"
    if motion_state not in PLANNED_MOTION_STATES:
        return None, "unsupported_motion_state"

    sample_id = str(data.get("id") or "")
    session_id = str(data.get("collectionSessionID") or "")
    if not sample_id:
        return None, "missing_id"
    if not session_id:
        return None, "missing_session_id"

    observed_indices = {
        int(frame.get("frameIndex"))
        for frame in data.get("frames") or []
        if isinstance(frame.get("frameIndex"), int)
    }
    if any(index not in observed_indices for index in required_frame_indices):
        return None, "missing_required_frame"

    lift_index = int(data.get("liftReferenceFrameIndex", 0))
    collector = data.get("collector") or {}
    return {
        "id": sample_id,
        "path": str(path.resolve()),
        "file_name": path.name,
        "sha256": sha256,
        "schema_version": 5,
        "class_label": class_label,
        "finger_type": finger_type,
        "touch_strength": touch_strength,
        "motion_state": motion_state,
        "created_at": float(data.get("createdAt", 0.0)),
        "raw_session_id": session_id,
        "collection_name": "bbsiak:" + motion_state + ":" + session_id,
        "collector_user_name": str(collector.get("userName") or "unknown"),
        "lift_reference_frame_index": lift_index,
        "duration_bucket": duration_bucket(lift_index),
        "sample_weight": 1.0,
        "split": "",
    }, ""


def build_schema5_manifest_rows(dataset_dir, required_frame_indices):
    rows = []
    reasons = Counter()
    seen_ids = {}
    paths = sorted(path for path in Path(dataset_dir).rglob("*.json") if is_raw_sample_path(path))
    for path in paths:
        try:
            payload = path.read_bytes()
            data = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            reasons["json_load_failed"] += 1
            continue
        sha256 = hashlib.sha256(payload).hexdigest()
        try:
            require_trainable(path, data)
        except RuntimeError as error:
            reasons[str(error)] += 1
            continue
        row, reason = schema5_manifest_row(path, data, required_frame_indices, sha256)
        if row is None:
            reasons[reason] += 1
            continue
        sample_id = row["id"]
        if sample_id in seen_ids:
            duplicate_reason = "duplicate_id" if seen_ids[sample_id] == sha256 else "conflicting_id"
            reasons[duplicate_reason] += 1
            continue
        seen_ids[sample_id] = sha256
        rows.append(row)
    rows.sort(key=lambda row: (row["created_at"], row["id"]))
    summary = {
        "inputFiles": len(paths),
        "acceptedRows": len(rows),
        "rejectedRows": len(paths) - len(rows),
        "rejectionReasons": dict(sorted(reasons.items())),
        "classCounts": dict(sorted(Counter(row["class_label"] for row in rows).items())),
        "motionCounts": dict(sorted(Counter(row["motion_state"] for row in rows).items())),
        "durationBucketCounts": dict(sorted(Counter(row["duration_bucket"] for row in rows).items())),
        "sessionCount": len({row["raw_session_id"] for row in rows}),
    }
    return rows, summary


def validate_no_session_split_leakage(rows):
    splits_by_session = defaultdict(set)
    for row in rows:
        splits_by_session[row.get("raw_session_id", "")].add(row.get("split", ""))
    leaking = [session for session, splits in splits_by_session.items() if len(splits) > 1]
    if leaking:
        raise RuntimeError("cross_split_session:" + leaking[0])


def build_motion_holdout_folds(rows):
    folds = {}
    required_classes = set(CLASS_ORDER)
    for motion_state in PLANNED_MOTION_STATES:
        fold_rows = []
        for source in rows:
            row = dict(source)
            row["split"] = "valid" if row.get("motion_state") == motion_state else "train"
            fold_rows.append(row)
        train_classes = {row["class_label"] for row in fold_rows if row["split"] == "train"}
        valid_classes = {row["class_label"] for row in fold_rows if row["split"] == "valid"}
        if train_classes != required_classes or valid_classes != required_classes:
            raise RuntimeError("motion_holdout_missing_class:" + motion_state)
        validate_no_session_split_leakage(fold_rows)
        folds[motion_state] = fold_rows
    return folds


def build_temporal_tail_split(rows, validation_ratio):
    grouped = defaultdict(list)
    for source in rows:
        grouped[source.get("raw_session_id", "")].append(source)
    output = []
    for items in grouped.values():
        ordered = sorted(items, key=lambda row: (float(row.get("created_at", 0.0)), row.get("id", "")))
        if len(ordered) < 2:
            valid_count = 0
        else:
            valid_count = max(1, int(math.ceil(len(ordered) * validation_ratio)))
            valid_count = min(valid_count, len(ordered) - 1)
        boundary = len(ordered) - valid_count
        for index, source in enumerate(ordered):
            row = dict(source)
            row["split"] = "valid" if index >= boundary else "train"
            output.append(row)
    required_classes = set(CLASS_ORDER)
    train_classes = {row["class_label"] for row in output if row["split"] == "train"}
    valid_classes = {row["class_label"] for row in output if row["split"] == "valid"}
    if train_classes != required_classes or valid_classes != required_classes:
        raise RuntimeError("temporal_tail_missing_class")
    return sorted(output, key=lambda row: (row["split"], float(row.get("created_at", 0.0)), row["id"]))


def write_manifest_rows(path, rows):
    if not rows:
        raise RuntimeError("empty_manifest")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def planned_fold_summary(rows):
    train_rows = [row for row in rows if row.get("split") == "train"]
    valid_rows = [row for row in rows if row.get("split") == "valid"]
    return {
        "rowCount": len(rows),
        "trainCount": len(train_rows),
        "validCount": len(valid_rows),
        "trainClassCount": len({row["class_label"] for row in train_rows}),
        "validClassCount": len({row["class_label"] for row in valid_rows}),
        "splitClassCounts": dict(sorted(Counter(
            row.get("split", "") + ":" + row.get("class_label", "")
            for row in rows
        ).items())),
    }


def prepare_planned_manifests(
        dataset_dir,
        output_dir,
        required_frame_indices,
        temporal_validation_ratio):
    output_dir = Path(output_dir)
    rows, dataset_summary = build_schema5_manifest_rows(dataset_dir, required_frame_indices)
    if not rows:
        raise RuntimeError("empty_schema5_dataset")
    write_manifest_rows(output_dir / "master_manifest.csv", rows)

    temporal_rows = build_temporal_tail_split(rows, temporal_validation_ratio)
    write_manifest_rows(output_dir / "folds" / "temporal_tail.csv", temporal_rows)
    folds = {"temporal_tail": planned_fold_summary(temporal_rows)}

    motion_folds = build_motion_holdout_folds(rows)
    for motion_state, fold_rows in motion_folds.items():
        name = "motion_" + motion_state
        write_manifest_rows(output_dir / "folds" / (name + ".csv"), fold_rows)
        folds[name] = planned_fold_summary(fold_rows)

    summary = {
        "dataset": dataset_summary,
        "requiredFrameIndices": list(required_frame_indices),
        "temporalValidationRatio": temporal_validation_ratio,
        "motionHoldoutFolds": list(PLANNED_MOTION_STATES),
        "folds": folds,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "manifest_summary.json", summary)
    return summary


def metrics_from_predictions(truths, guesses):
    class_count = len(CLASS_ORDER)
    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for truth, guess in zip(truths, guesses):
        confusion[int(truth)][int(guess)] += 1
    correct = sum(confusion[index][index] for index in range(class_count))
    recalls = {}
    observed_recalls = []
    for index, class_label in enumerate(CLASS_ORDER):
        total = sum(confusion[index])
        if total:
            recall = confusion[index][index] / total
            observed_recalls.append(recall)
        else:
            recall = 0.0
        recalls[class_label] = recall
    count = len(truths)
    return {
        "accuracy": correct / count if count else 0.0,
        "balancedAccuracy": sum(observed_recalls) / len(observed_recalls) if observed_recalls else 0.0,
        "minRecall": min(observed_recalls) if observed_recalls else 0.0,
        "classRecalls": recalls,
        "confusionMatrix": confusion,
        "count": count,
    }


def grouped_metrics_from_predictions(truths, guesses, samples):
    output = {}
    for field, output_name in (
            ("motionState", "byMotionState"),
            ("durationBucket", "byDurationBucket")):
        indices_by_group = defaultdict(list)
        for index, sample in enumerate(samples):
            indices_by_group[str(sample.get(field) or "unknown")].append(index)
        output[output_name] = {
            group: metrics_from_predictions(
                [truths[index] for index in indices],
                [guesses[index] for index in indices],
            )
            for group, indices in sorted(indices_by_group.items())
        }
    return output


def transform_sample_matrices(
        samples,
        source_feature_names,
        transform_name,
        baseline_window=256,
        baseline_warmup=32):
    if transform_name == "none":
        return list(source_feature_names)

    index = {name: position for position, name in enumerate(source_feature_names)}
    imu_names = [
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
    ]
    shape_names = [
        "invariant.log_axis_ratio",
        "invariant.covariance_correlation",
        "invariant.eccentricity",
    ]
    relative_names = [
        "relative.log_major_ratio",
        "relative.log_minor_ratio",
        "relative.log_area_ratio",
        "relative.baseline_readiness",
    ]
    required = list(imu_names)
    if transform_name in ("invariant_shape", "relative_touch"):
        required.extend([
            "contactPatch.cov_11",
            "contactPatch.cov_12",
            "contactPatch.cov_22",
            "contactPatch.sigmaX",
            "contactPatch.sigmaY",
            "touchNormalization.rawMajorPx",
            "touchNormalization.rawMinorPx",
        ])
    missing = [name for name in required if name not in index]
    if missing:
        raise RuntimeError("matrix_transform_missing_features")

    if transform_name == "baseline_imu":
        for sample in samples:
            sample["matrix"] = baseline_imu_matrix(sample["matrix"], index, imu_names)
        return list(imu_names[:2]) + ["baseline." + name for name in imu_names[2:]]

    relative_values = {}
    if transform_name == "relative_touch":
        relative_values = causal_relative_values(
            samples,
            index,
            baseline_window,
            baseline_warmup,
        )

    for sample in samples:
        transformed = []
        relative = relative_values.get(sample_identity(sample), [0.0, 0.0, 0.0, 0.0])
        for row in sample["matrix"]:
            output = [row[index[name]] for name in imu_names]
            if transform_name in ("invariant_shape", "relative_touch"):
                output.extend(dimensionless_shape_values(row, index))
            if transform_name == "relative_touch":
                output.extend(relative)
            transformed.append(output)
        sample["matrix"] = transformed

    names = list(imu_names)
    if transform_name in ("invariant_shape", "relative_touch"):
        names.extend(shape_names)
    if transform_name == "relative_touch":
        names.extend(relative_names)
    return names


def baseline_imu_matrix(matrix, index, imu_names, pre_touch_count=5):
    if not matrix:
        return []
    pre_rows = matrix[:max(1, min(pre_touch_count, len(matrix)))]
    centers = {}
    scales = {}
    for name in imu_names[2:]:
        values = [finite_float(row[index[name]], 0.0) for row in pre_rows]
        center = median(values)
        deviation = median([abs(value - center) for value in values])
        if name.startswith("userAcceleration."):
            floor = 0.25
        else:
            floor = 0.05
        centers[name] = center
        scales[name] = max(floor, deviation * 1.4826)

    output = []
    for row in matrix:
        transformed = [row[index[imu_names[0]]], row[index[imu_names[1]]]]
        for name in imu_names[2:]:
            value = (finite_float(row[index[name]], 0.0) - centers[name]) / scales[name]
            transformed.append(max(-20.0, min(20.0, value)))
        output.append(transformed)
    return output


def dimensionless_shape_values(row, index):
    major = finite_float(row[index["touchNormalization.rawMajorPx"]], 0.0)
    minor = finite_float(row[index["touchNormalization.rawMinorPx"]], 0.0)
    covariance11 = max(0.0, finite_float(row[index["contactPatch.cov_11"]], 0.0))
    covariance12 = finite_float(row[index["contactPatch.cov_12"]], 0.0)
    covariance22 = max(0.0, finite_float(row[index["contactPatch.cov_22"]], 0.0))
    sigma_x = max(0.0, finite_float(row[index["contactPatch.sigmaX"]], 0.0))
    sigma_y = max(0.0, finite_float(row[index["contactPatch.sigmaY"]], 0.0))
    denominator = math.sqrt(covariance11 * covariance22)
    correlation = covariance12 / denominator if denominator > 0 else 0.0
    eccentricity_denominator = sigma_x + sigma_y
    eccentricity = abs(sigma_x - sigma_y) / eccentricity_denominator if eccentricity_denominator > 0 else 0.0
    return [
        safe_log_ratio(major, minor),
        max(-1.0, min(1.0, correlation)),
        max(0.0, min(1.0, eccentricity)),
    ]


def causal_relative_values(samples, index, baseline_window, baseline_warmup):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample.get("baselineGroup", "unknown")].append(sample)
    result = {}
    for items in grouped.values():
        major_history = deque(maxlen=max(1, baseline_window))
        minor_history = deque(maxlen=max(1, baseline_window))
        area_history = deque(maxlen=max(1, baseline_window))
        ordered = sorted(items, key=lambda item: (item.get("createdAt", 0.0), item.get("path", "")))
        for sample in ordered:
            major, minor, area = sample_contact_values(sample, index)
            count = len(major_history)
            readiness = min(1.0, count / max(1, baseline_warmup))
            if count >= max(1, baseline_warmup):
                values = [
                    safe_log_ratio(major, median(major_history)),
                    safe_log_ratio(minor, median(minor_history)),
                    safe_log_ratio(area, median(area_history)),
                    readiness,
                ]
            else:
                values = [0.0, 0.0, 0.0, readiness]
            result[sample_identity(sample)] = values
            if major > 0 and minor > 0 and area > 0:
                major_history.append(major)
                minor_history.append(minor)
                area_history.append(area)
    return result


def sample_contact_values(sample, index):
    major_values = []
    minor_values = []
    for row in sample["matrix"]:
        major_values.append(max(0.0, finite_float(row[index["touchNormalization.rawMajorPx"]], 0.0)))
        minor_values.append(max(0.0, finite_float(row[index["touchNormalization.rawMinorPx"]], 0.0)))
    major = max(major_values, default=0.0)
    minor = max(minor_values, default=0.0)
    return major, minor, major * minor


def sample_identity(sample):
    return sample.get("id", "") or sample.get("path", "")


def safe_log_ratio(value, baseline):
    if value <= 0 or baseline <= 0:
        return 0.0
    return max(-4.0, min(4.0, math.log(value / baseline)))


def apply_collection_class_balance(samples):
    counts = Counter((sample.get("collectionName", "unknown"), sample.get("classLabel", "")) for sample in samples)
    raw_factors = [1.0 / counts[(sample.get("collectionName", "unknown"), sample.get("classLabel", ""))] for sample in samples]
    mean_factor = sum(raw_factors) / max(1, len(raw_factors))
    for sample, raw_factor in zip(samples, raw_factors):
        factor = raw_factor / max(mean_factor, 1e-12)
        factor = max(0.25, min(4.0, factor))
        sample["sampleWeight"] = round(sample.get("sampleWeight", 1.0) * factor, 6)


def load_samples(
        manifest_path,
        feature_names,
        frame_indices,
        missing_frame_policy,
        anchor_mode="down",
        require_press_active=True,
        feature_contract=""):
    if anchor_mode == "down" and not require_press_active:
        raise RuntimeError("press_active_filter_required")
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
            annotation = read_annotation(path, data)
            if exclusion_reason(annotation):
                continue
            if anchor_mode == "down":
                if feature_contract in TIME_GRID_FEATURE_CONTRACTS:
                    if not is_time_grid_press_active(data):
                        continue
                elif not is_press_active_window(data, frame_indices):
                    continue
            matrix_payload = sample_matrix_for_anchor(
                data,
                feature_names,
                frame_indices,
                missing_frame_policy,
                anchor_mode,
                feature_contract,
                return_quality=feature_contract in TIME_GRID_FEATURE_CONTRACTS,
            )
            if matrix_payload is None:
                continue
            grid_quality = matrix_payload.get("gridQuality") if isinstance(matrix_payload, dict) else None
            matrix = matrix_payload["matrix"] if isinstance(matrix_payload, dict) else matrix_payload
            samples.append({
                "collectionAnnotation": annotation,
                "id": row.get("id", ""),
                "path": str(path),
                "label": CLASS_ORDER.index(class_label),
                "classLabel": class_label,
                "matrix": matrix,
                "createdAt": sample_created_at(row, data),
                "sampleWeight": sample_weight_for_row(row),
                "split": row.get("split", ""),
                "collectionName": row.get("collection_name", "unknown"),
                "baselineGroup": baseline_group_for_row(row),
                "motionState": row.get("motion_state", (data.get("label") or {}).get("motionState", "unknown")),
                "durationBucket": row.get(
                    "duration_bucket",
                    duration_bucket(data.get("liftReferenceFrameIndex", 0)),
                ),
                "rawSessionId": row.get("raw_session_id", data.get("collectionSessionID", "")),
                "liftReferenceFrameIndex": int(data.get("liftReferenceFrameIndex", -1)),
                "gridQuality": grid_quality,
            })
    return samples


def load_matrix_samples(
        manifest_path,
        matrix_jsonl,
        feature_names,
        frame_indices,
        feature_contract):
    """读取预生成矩阵，并逐项校验来源身份、标签、契约和权重。"""
    expected_contract = feature_contract or feature_contract_for_names(feature_names)
    if not expected_contract:
        raise RuntimeError("matrix_feature_contract_required")
    expected_shape = [len(frame_indices), len(feature_names)]
    matrices = {}
    with Path(matrix_jsonl).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    "invalid_matrix_jsonl:" + str(line_number)
                ) from error
            source_sha256 = str(row.get("sourceSha256") or "")
            if not source_sha256:
                raise RuntimeError("matrix_missing_source_sha256")
            if source_sha256 in matrices:
                raise RuntimeError("duplicate_matrix_source_sha256")
            matrices[source_sha256] = row

    samples = []
    seen_manifest_sources = set()
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as source:
        for manifest_row in csv.DictReader(source):
            class_label = manifest_row.get("class_label", "")
            if class_label not in CLASS_ORDER:
                continue
            source_sha256 = (
                manifest_row.get("sourceSha256")
                or manifest_row.get("source_sha256")
                or ""
            )
            if not source_sha256:
                raise RuntimeError("manifest_missing_source_sha256")
            if source_sha256 in seen_manifest_sources:
                raise RuntimeError("duplicate_manifest_source_sha256")
            seen_manifest_sources.add(source_sha256)
            row = matrices.get(source_sha256)
            if row is None:
                raise RuntimeError("matrix_source_missing")
            source_path = resolve_path(manifest_row, manifest_path)
            if source_path is None:
                raise RuntimeError("matrix_source_file_missing")
            source_raw = source_path.read_bytes()
            if hashlib.sha256(source_raw).hexdigest() != source_sha256:
                raise RuntimeError("matrix_source_sha256_mismatch")
            annotation = read_annotation(source_path, json.loads(source_raw))
            if exclusion_reason(annotation):
                raise RuntimeError("matrix_reviewed_excluded")
            sample_id = str(row.get("sampleId") or "")
            manifest_sample_id = str(manifest_row.get("sampleId") or manifest_row.get("id") or "")
            if not sample_id or (manifest_sample_id and sample_id != manifest_sample_id):
                raise RuntimeError("matrix_sample_identity_mismatch")
            if str(row.get("classLabel") or "") != class_label:
                raise RuntimeError("matrix_class_label_mismatch")
            if str(row.get("featureContract") or "") != expected_contract:
                raise RuntimeError("matrix_feature_contract_mismatch")
            if list(row.get("featureNames") or []) != list(feature_names):
                raise RuntimeError("matrix_feature_names_mismatch")
            matrix = row.get("matrix")
            shape = row.get("matrixShape")
            if shape != expected_shape:
                raise RuntimeError(
                    "matrix_shape_mismatch:expected="
                    + str(expected_shape)
                    + ",actual="
                    + str(shape)
                    + ",contract="
                    + expected_contract
                )
            if not isinstance(matrix, list) or len(matrix) != expected_shape[0]:
                raise RuntimeError("matrix_shape_mismatch")
            for values in matrix:
                if not isinstance(values, list) or len(values) != expected_shape[1]:
                    raise RuntimeError("matrix_shape_mismatch")
                if not all(finite_float(value, None) is not None for value in values):
                    raise RuntimeError("matrix_non_finite_value")
            manifest_weight = finite_float(manifest_row.get("trainingWeight"), 1.0)
            matrix_weight = finite_float(row.get("trainingWeight"), manifest_weight)
            if not math.isclose(manifest_weight, matrix_weight, rel_tol=0.0, abs_tol=1e-9):
                raise RuntimeError("matrix_training_weight_mismatch")
            label_status = str(row.get("labelStatus") or "")
            if label_status in {"mistake", "uncertain"}:
                raise RuntimeError("matrix_reviewed_excluded")
            if label_status and label_status != annotation["labelStatus"]:
                raise RuntimeError("matrix_label_status_mismatch")
            review_sha256 = str(row.get("reviewSha256") or "")
            if review_sha256 != annotation["reviewSha256"]:
                raise RuntimeError("matrix_review_digest_mismatch")
            samples.append({
                "collectionAnnotation": {
                    "labelStatus": annotation["labelStatus"],
                    "reviewSha256": annotation["reviewSha256"],
                },
                "id": sample_id,
                "path": "",
                "label": CLASS_ORDER.index(class_label),
                "classLabel": class_label,
                "matrix": matrix,
                "createdAt": finite_float(
                    manifest_row.get("sourceTimestampMs")
                    or manifest_row.get("created_at")
                    or row.get("sourceTimestampMs"),
                    0.0,
                ),
                "sampleWeight": manifest_weight,
                "split": manifest_row.get("split", ""),
                "collectionName": (
                    manifest_row.get("collection_name")
                    or row.get("collectorUserName")
                    or "unknown"
                ),
                "baselineGroup": (
                    manifest_row.get("groupId")
                    or manifest_row.get("group_id")
                    or row.get("groupId")
                    or ""
                ),
                "motionState": (
                    manifest_row.get("motion_state")
                    or row.get("motionState")
                    or "unknown"
                ),
                "durationBucket": manifest_row.get("duration_bucket", "unknown"),
                "rawSessionId": (
                    manifest_row.get("raw_session_id")
                    or row.get("sessionId")
                    or ""
                ),
                "sourceSha256": source_sha256,
                "gridQuality": row.get("gridQuality"),
            })
    if not samples:
        raise RuntimeError("empty_matrix_manifest")
    return samples


def baseline_group_for_row(row):
    collection = row.get("collection_name", "unknown")
    source = row.get("source_person", "") or row.get("collector_user_name", "")
    return collection + (":" + source if source else "")


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
    if strategy == "all":
        return list(samples), []
    if strategy == "predefined":
        train = [sample for sample in samples if sample.get("split") == "train"]
        valid = [sample for sample in samples if sample.get("split") == "valid"]
        if not train or not valid:
            raise RuntimeError("missing_predefined_split")
        return train, valid
    if strategy == "temporal":
        return temporal_split(samples, validation_ratio)
    return stratified_split(samples, validation_ratio, seed)


def validate_session_split_integrity(train_samples, valid_samples):
    if not valid_samples:
        return
    train_sessions = set()
    valid_sessions = set()
    for sample in train_samples:
        session_id = str(sample.get("rawSessionId", "")).strip()
        if not session_id:
            raise RuntimeError("missing_raw_session_id")
        train_sessions.add(session_id)
    for sample in valid_samples:
        session_id = str(sample.get("rawSessionId", "")).strip()
        if not session_id:
            raise RuntimeError("missing_raw_session_id")
        valid_sessions.add(session_id)
    overlap = sorted(train_sessions.intersection(valid_sessions))
    if overlap:
        raise RuntimeError("cross_split_session:" + overlap[0])


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


def configure_determinism(torch, enabled):
    if not enabled:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


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


def train_one_epoch_on_device(model, x, y, sample_weight, optimizer, loss_fn, batch_size, torch, device):
    model.train()
    order = torch.randperm(x.shape[0], device=device)
    total_loss = 0.0
    total = 0
    for start in range(0, x.shape[0], batch_size):
        device_index = order[start:start + batch_size]
        cpu_index = device_index.cpu() if x.device.type == "cpu" else device_index
        xb = x[cpu_index].to(device)
        yb = y[cpu_index].to(device)
        wb = sample_weight[cpu_index].to(device)
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


def prediction_lists(model, x, y, torch, batch_size=2048):
    model.eval()
    device = next(model.parameters()).device
    truths = []
    guesses = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start:start + batch_size].to(device)
            pred = model(xb).argmax(dim=1).cpu()
            truths.extend(y[start:start + batch_size].cpu().tolist())
            guesses.extend(pred.tolist())
    return truths, guesses


def evaluate(model, x, y, torch, batch_size=2048):
    truths, guesses = prediction_lists(model, x, y, torch, batch_size=batch_size)
    return metrics_from_predictions(truths, guesses)


def evaluate_subgroups(model, x, y, samples, torch, batch_size=2048):
    truths, guesses = prediction_lists(model, x, y, torch, batch_size=batch_size)
    return grouped_metrics_from_predictions(truths, guesses, samples)


def scaler_metadata(center, scale, feature_names, frame_indices, feature_contract=""):
    metadata = {
        "center": list(center),
        "scale": list(scale),
        "featureNames": list(feature_names),
        "frameIndices": list(frame_indices),
        "frameCount": len(frame_indices),
        "labelOrder": CLASS_ORDER,
        "classCount": len(CLASS_ORDER),
        "featureContract": feature_contract or feature_contract_for_names(feature_names),
    }
    selected_contract = feature_contract or feature_contract_for_names(feature_names)
    if selected_contract in TIME_GRID_FEATURE_CONTRACTS:
        metadata.update({
            "gridRateHz": 200,
            "gridOffsetsMs": list(TIME_GRID_OFFSETS_MS),
            "interpolation": "linear_sensor_time",
            "maxGapRule": "max(2*5ms,4*median_dt)",
            "platformAdapterVersion": "cross_device_v1",
            "coordinateTransformVersion": "cross_device_v1",
            "missingMaskPolicy": "quality_metadata_and_zero_value",
        })
    return metadata


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
        train_subgroup_metrics,
        valid_subgroup_metrics,
        history,
        best_epoch,
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
    scaler = scaler_metadata(
        tensor_to_list(center),
        tensor_to_list(scale),
        feature_names,
        frame_indices,
        args.feature_preset,
    )
    report = {
        "modelId": model_id,
        "displayName": display_name,
        "labelOrder": CLASS_ORDER,
        "trainMetrics": train_metrics,
        "validMetrics": valid_metrics,
        "trainSubgroupMetrics": train_subgroup_metrics,
        "validSubgroupMetrics": valid_subgroup_metrics,
        "trainClassCounts": dict(Counter(sample["classLabel"] for sample in train_samples)),
        "validClassCounts": dict(Counter(sample["classLabel"] for sample in valid_samples)),
        "bestEpoch": best_epoch,
        "anchorMode": args.anchor_mode,
        "featureContract": args.feature_preset,
        "gridRateHz": 200 if args.feature_preset in TIME_GRID_FEATURE_CONTRACTS else None,
        "gridOffsetsMs": list(TIME_GRID_OFFSETS_MS) if args.feature_preset in TIME_GRID_FEATURE_CONTRACTS else [],
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
