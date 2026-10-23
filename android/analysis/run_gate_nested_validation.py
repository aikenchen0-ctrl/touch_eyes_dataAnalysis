#!/usr/bin/env python3
"""Nested validation for the light/heavy gate with fold-external calibration."""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    from . import gate_threshold_metrics
    from . import gate_evidence_features as evidence_features
    from . import early_press_evidence
    from . import device_normalization
    from . import run_lightweight_time_grid_models as lightweight
    from . import stats_mlp_contract as contract
    from . import train_stats_mlp_android_candidate as trainer
except ImportError:
    import gate_threshold_metrics
    import gate_evidence_features as evidence_features
    import early_press_evidence
    import device_normalization
    import run_lightweight_time_grid_models as lightweight
    import stats_mlp_contract as contract
    import train_stats_mlp_android_candidate as trainer


TARGET_FPRS = (0.01, 0.05)
DEFAULT_THRESHOLD_CALIBRATION_FPR = 0.005
DEFAULT_FALSE_POSITIVE_LIMIT = 0.01
MODEL_SPECS = (
    ("gate_no_context", False),
    ("gate_context", True),
)
MODEL_SELECTION_MODES = (
    "balanced_accuracy",
    "fixed_fpr_recall",
)
GATE_LOSS_MODES = (
    "cross_entropy",
    "pairwise_ranking",
    "ce_pairwise",
    "asymmetric_focal",
)
ORDINARY_CLASS_LABEL = "ordinary_touch"
ORDINARY_COLLECTION_PROTOCOL = "negative_tap_v1"
MIN_GROUP_POSITIVES = 20
MIN_GROUP_NEGATIVES = 20
MIN_PROOF_SESSION_COUNT = 5
EXPERIMENT_BLOCK_MAX_GAP_MS = 90_000
EXPERIMENT_BLOCK_FIELDS = (
    "sourceTimestampMs",
    "collectorUserName",
    "collectorDeviceId",
    "deviceName",
    "taskId",
    "bodyPosture",
    "motionState",
)
DEPLOYMENT_RECALL_TARGET = 0.90
DEPLOYMENT_PRIMARY_FPR = "0.01"
DEPLOYMENT_MIN_SEEDS = 3
DEPLOYMENT_MIN_DEVICES = 3
DEPLOYMENT_MIN_POSTURES = 2
DEPLOYMENT_MIN_MOTION_STATES = 2
UNVERIFIED_GROUP_VALUES = frozenset((
    "",
    "unknown",
    "unspecified",
    "unconstrained",
    "none",
    "null",
))
FEATURE_MODES = (
    "full",
    "compact",
    "no_absolute_xy",
    "no_shape",
    "motion_only",
    "shape_only",
    "size_only",
    "axis_ratio_only",
    "motion_size",
    "motion_axis_ratio",
    "magnitude_only",
    "magnitude_temporal",
    "physical_magnitude",
    "physical_magnitude_temporal",
    "physical_magnitude_shape",
    "physical_magnitude_temporal_shape",
    "device_relative_magnitude",
    "contact_relative_magnitude",
    "contact_relative_magnitude_shape",
    "magnitude_contact_relative",
    "historical_contact_relative",
    "historical_contact_relative_no_shape",
    "historical_contact_relative_no_xy",
    "historical_contact_magnitude_no_xy",
    "magnitude_historical_contact",
    "historical_contact_local_shape",
    "historical_contact_motion_no_shape",
    "historical_contact_motion_shape",
    "historical_contact_causal_motion",
    "historical_contact_causal_motion_shape",
    "raw_contact_no_xy",
    "raw_contact_magnitude_no_xy",
    "device_causal_contact",
    "device_causal_contact_no_xy",
    "device_causal_contact_magnitude_no_xy",
    "touch_local_shape",
    "touch_local_motion",
    "touch_local_motion_shape",
)
WEIGHTING_MODES = (
    "session_balanced",
    "block_balanced",
    "device_balanced",
    "device_motion_balanced",
)
FEATURE_FAMILIES = ("stats", "evidence")
EVIDENCE_CONTRACTS = tuple(evidence_features.EVIDENCE_CONTRACTS)
IMU_MODES = (
    "raw",
    "pretouch_centered",
    "guarded_pretouch_centered",
    "earliest_pretouch_centered",
    "gravity_aligned",
    "gravity_aligned_pretouch_centered",
    "gravity_aligned_guarded_pretouch_centered",
    "gravity_aligned_earliest_pretouch_centered",
)
ADAPTIVE_FEATURE_MODES = (
    "none",
    "imu_delta",
    "imu_delta_shape",
)
AUGMENTATION_MODES = (
    "none",
    "channel_scale",
)
EARLY_EVIDENCE_MODES = ("none",) + early_press_evidence.EVIDENCE_MODES
NORMALIZATION_MODES = device_normalization.NORMALIZATION_MODES
# 中文注释：时间网格的 0 ms 位于索引 3；索引 7 是 +20 ms，
# 只可用于旧证据窗口对照，不能作为重力参考或按下偏置的真实边界。
TOUCH_START_FRAME = 3
TOUCH_GRID_MS = (-15, -10, -5, 0, 5, 10, 15, 20, 25)
TOUCH_LOCAL_MODES = (
    "touch_local_shape",
    "touch_local_motion",
    "touch_local_motion_shape",
    "historical_contact_local_shape",
    "historical_contact_motion_shape",
)
TOUCH_LOCAL_FEATURE_NAMES = tuple(
    name
    for channel in ("size", "major", "minor")
    for name in (
        channel + "_available_fraction",
        channel + "_range_relative",
        channel + "_peak_relative",
        channel + "_rise_time_relative",
        channel + "_decay_time_relative",
        channel + "_normalized_area",
    )
)
CAUSAL_MOTION_FEATURE_NAMES = (
    "imu_pre_coverage",
    "imu_touch_coverage",
    "imu_ready",
    "acc_touch_std_log",
    "acc_jerk_max_log",
    "acc_baseline_to_touch_log",
    "acc_std_ratio_log",
    "acc_jerk_ratio_log",
    "rot_touch_std_log",
    "rot_jerk_max_log",
    "rot_baseline_to_touch_log",
    "rot_std_ratio_log",
    "rot_jerk_ratio_log",
)
HISTORICAL_CONTACT_MODES = (
    "historical_contact_relative",
    "historical_contact_relative_no_shape",
    "magnitude_historical_contact",
    "historical_contact_local_shape",
    "historical_contact_motion_no_shape",
    "historical_contact_motion_shape",
    "historical_contact_causal_motion",
    "historical_contact_causal_motion_shape",
)
CAUSAL_MOTION_MODES = (
    "historical_contact_causal_motion",
    "historical_contact_causal_motion_shape",
)
CAUSAL_MOTION_SHAPE_MODES = (
    "historical_contact_causal_motion_shape",
)
HISTORICAL_CONTACT_FEATURE_NAMES = (
    "current_size_coverage",
    "history_size_coverage",
    "history_ratio_coverage",
    "history_ready",
    "log_history_touch_count",
    "log_history_age_seconds",
    "log_size_median_ratio",
    "log_size_peak_ratio",
    "history_size_mad_relative",
    "axis_ratio_median_offset",
    "axis_ratio_peak_offset",
)
HISTORICAL_CONTACT_MAX_TOUCHES = 32
HISTORICAL_CONTACT_MIN_TOUCHES = 4
DEVICE_CAUSAL_CONTACT_MAX_TOUCHES = 64
DEVICE_CAUSAL_CONTACT_MIN_TOUCHES = 8
DEVICE_CAUSAL_CONTACT_MODES = (
    "device_causal_contact",
    "device_causal_contact_no_xy",
    "device_causal_contact_magnitude_no_xy",
)


def dataset_id(path):
    """Keep fold identities unique when several datasets are merged."""
    resolved = Path(path).resolve()
    return resolved.parent.name + "/" + resolved.name


def load_manifest_metadata(root):
    """按 source hash 读取训练清单中的时间和采集者元数据。"""
    root = Path(root)
    path = (
        root
        if root.is_file() and root.name == "training_candidate_manifest.csv"
        else root / "training_candidate_manifest.csv"
    )
    if not path.is_file():
        return {}
    result = {}
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            source_sha256 = str(row.get("sourceSha256") or "").strip()
            if source_sha256:
                result[source_sha256] = row
    return result


def attach_manifest_metadata(rows, root):
    """只补充矩阵行缺失的元数据，避免覆盖已验证的原始字段。"""
    metadata_by_hash = load_manifest_metadata(root)
    for row in rows:
        metadata = metadata_by_hash.get(str(row.get("sourceSha256") or ""))
        if not metadata:
            continue
        for field in EXPERIMENT_BLOCK_FIELDS:
            value = metadata.get(field)
            if value not in (None, "") and row.get(field) in (None, ""):
                row[field] = value


def timestamp_ms(row):
    """返回可比较的毫秒时间戳，缺失时返回 None。"""
    value = row.get("sourceTimestampMs")
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _source_timestamp_ms(row):
    """Read the collection timestamp from the immutable source file name."""
    source_file = str(row.get("sourceFile") or "").strip()
    prefix = source_file.split("_", 1)[0]
    try:
        value = int(prefix)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def experiment_block_key(row):
    """返回用于完整实验块留出的稳定键。"""
    return str(row.get("_experimentBlock") or "")


def attach_experiment_blocks(rows):
    """把同设备、用户、场景且间隔不超过 90 秒的 session 归入实验块。"""
    sessions = defaultdict(list)
    for row in rows:
        sessions[str(row.get("groupId") or "")].append(row)
    session_info = {}
    for session_id, session_rows in sessions.items():
        timestamps = [
            value
            for value in (timestamp_ms(row) for row in session_rows)
            if value is not None
        ]
        first = session_rows[0]
        session_info[session_id] = {
            "sessionId": session_id,
            "timestampMs": min(timestamps) if timestamps else None,
            "endTimestampMs": max(timestamps) if timestamps else None,
            "identity": (
                physical_device_key(first),
                str(first.get("collectorUserName") or "").strip(),
                posture_domain(first),
            ),
            "classes": {
                str(row.get("classLabel") or "") for row in session_rows
            },
            "rowCount": len(session_rows),
        }
    ordered = sorted(
        session_info.values(),
        key=lambda item: (
            item["identity"],
            item["timestampMs"] is None,
            item["timestampMs"] or 0,
            item["sessionId"],
        ),
    )
    blocks = []
    current = []
    previous = None
    for info in ordered:
        starts_new = (
            previous is None
            or info["identity"] != previous["identity"]
            or info["timestampMs"] is None
            or previous["endTimestampMs"] is None
            or info["timestampMs"] - previous["endTimestampMs"]
            > EXPERIMENT_BLOCK_MAX_GAP_MS
        )
        if starts_new and current:
            blocks.append(current)
            current = []
        current.append(info)
        previous = info
    if current:
        blocks.append(current)
    for block_index, block in enumerate(blocks, start=1):
        identity = block[0]["identity"]
        block_id = "block:{:03d}:{}:{}:{}".format(
            block_index,
            identity[0],
            identity[1] or "unknown_user",
            identity[2],
        )
        session_ids = {item["sessionId"] for item in block}
        for session_id in session_ids:
            for row in sessions[session_id]:
                row["_experimentBlock"] = block_id
    return {
        "blockId": block_id
        for block_id in sorted({row.get("_experimentBlock") for row in rows})
    }


def load_rows(
        dataset_dirs,
        early_evidence_files=None,
        metadata_manifest=None,
):
    """Merge training rows by source hash and reject conflicting duplicates."""
    rows_by_hash = {}
    for path in dataset_dirs:
        root = Path(path).resolve()
        identifier = dataset_id(root)
        dataset_rows = lightweight.read_rows(root)
        attach_manifest_metadata(dataset_rows, root)
        if not any(timestamp_ms(row) is not None for row in dataset_rows):
            external_metadata = (
                load_manifest_metadata(Path(metadata_manifest))
                if metadata_manifest
                else {}
            )
            for row in dataset_rows:
                metadata = external_metadata.get(
                    str(row.get("sourceSha256") or "")
                )
                if not metadata:
                    continue
                # 中文注释：派生目录可能没有清单，允许显式提供可信清单，
                # 仅补时间与采集者，不改变已冻结的矩阵。
                for field in EXPERIMENT_BLOCK_FIELDS:
                    value = metadata.get(field)
                    if value not in (None, ""):
                        row[field] = value
        for row in dataset_rows:
            source_sha256 = str(row.get("sourceSha256") or "")
            if not source_sha256:
                raise RuntimeError("training_row_missing_source_sha256")
            copied = dict(row)
            copied["_datasetDir"] = str(root)
            copied["_datasetId"] = identifier
            copied["collectionBatch"] = collection_batch_key(copied)
            # 中文注释：保留原始姿势字段，同时生成统一姿态域。
            # 部分采集记录用 taskId 表示倾斜手持，原始 bodyPosture
            # 为 unspecified；统一域可避免该组被错误排除。
            copied["postureDomain"] = posture_domain(copied)
            existing = rows_by_hash.get(source_sha256)
            if existing is None:
                rows_by_hash[source_sha256] = copied
                continue
            if (
                existing.get("classLabel") != copied.get("classLabel")
                or existing.get("matrix") != copied.get("matrix")
            ):
                raise RuntimeError("conflicting_duplicate_training_source")
    if not rows_by_hash:
        raise RuntimeError("empty_training_dataset")
    rows = sorted(rows_by_hash.values(), key=lambda row: row["sourceSha256"])
    attach_experiment_blocks(rows)
    attach_early_evidence(rows, early_evidence_files)
    return rows


def load_evaluation_only_rows(
        dataset_dirs,
        training_rows,
        early_evidence_files=None,
):
    """Load external rows while rejecting any overlap with the fit corpus."""
    if not dataset_dirs:
        return []
    evaluation_rows = load_rows(
        dataset_dirs,
        early_evidence_files=early_evidence_files,
    )
    training_hashes = {row["sourceSha256"] for row in training_rows}
    overlap = training_hashes & {
        row["sourceSha256"] for row in evaluation_rows
    }
    if overlap:
        raise RuntimeError("evaluation_only_dataset_overlaps_training")
    # 中文注释：不同 source hash 也可能来自同一采集 session；
    # 外测与训练必须按完整 session 隔离，否则同源触摸会泄漏采集上下文。
    training_sessions = {
        str(row.get("groupId") or "").strip()
        for row in training_rows
        if str(row.get("groupId") or "").strip()
    }
    evaluation_sessions = {
        str(row.get("groupId") or "").strip()
        for row in evaluation_rows
        if str(row.get("groupId") or "").strip()
    }
    if training_sessions & evaluation_sessions:
        raise RuntimeError("evaluation_only_session_overlaps_training")
    return build_evaluation_only_rows(evaluation_rows)


def read_early_evidence(files):
    """Load sidecar evidence by source hash and reject conflicts."""
    evidence_by_hash = {}
    for value in files or []:
        path = Path(value).resolve()
        if not path.is_file():
            raise RuntimeError("missing_early_evidence_file")
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                source_sha256 = str(row.get("sourceSha256") or "")
                if not source_sha256:
                    raise RuntimeError(
                        "early_evidence_missing_source_sha256"
                    )
                if row.get("earlyEvidenceContract") != (
                        early_press_evidence.CONTRACT):
                    raise RuntimeError("invalid_early_evidence_contract")
                values = np.asarray(
                    row.get("earlyEvidence"),
                    dtype=np.float32,
                )
                early_press_evidence.validate([values])
                existing = evidence_by_hash.get(source_sha256)
                if existing is not None and not np.array_equal(
                        existing, values):
                    raise RuntimeError(
                        "conflicting_early_evidence_source"
                    )
                evidence_by_hash[source_sha256] = values
    return evidence_by_hash


def attach_early_evidence(rows, files):
    """Attach immutable sidecar vectors without changing source matrices."""
    if not files:
        return rows
    evidence_by_hash = read_early_evidence(files)
    for row in rows:
        values = evidence_by_hash.get(row["sourceSha256"])
        if values is not None:
            row["earlyEvidence"] = values.copy()
    return rows


def early_evidence_values(rows):
    """Return a zero-filled matrix while preserving the availability mask."""
    result = np.zeros(
        (len(rows), len(early_press_evidence.FEATURE_NAMES)),
        dtype=np.float32,
    )
    for index, row in enumerate(rows):
        values = row.get("earlyEvidence")
        if values is None:
            continue
        values = np.asarray(values, dtype=np.float32)
        early_press_evidence.validate([values])
        result[index] = values
    return result


def append_early_evidence(
        fit_rows,
        rows,
        features,
        mode,
):
    """Append normalized early evidence and return audit metadata."""
    if mode == "none":
        return features, {
            "mode": "none",
            "contract": None,
            "availableCount": 0,
            "totalCount": len(rows),
            "missingCount": len(rows),
            "missingRate": 1.0 if rows else 0.0,
        }
    fit_evidence = early_evidence_values(fit_rows)
    evidence = early_evidence_values(rows)
    normalized, metadata = early_press_evidence.normalize(
        fit_evidence,
        evidence,
        mode=mode,
    )
    result = np.concatenate((features, normalized), axis=1)
    return result, metadata


def early_evidence_coverage(rows):
    """Summarize sidecar availability without changing any model input."""
    available_count = sum(
        float(np.asarray(row.get("earlyEvidence"))[-1]) >= 0.5
        for row in rows
        if row.get("earlyEvidence") is not None
    )
    total_count = len(rows)
    return {
        "availableCount": int(available_count),
        "totalCount": int(total_count),
        "missingCount": int(total_count - available_count),
        "missingRate": (
            float(total_count - available_count) / total_count
            if total_count
            else 0.0
        ),
    }

def load_ordinary_click_rows(
        dataset_dirs,
        training_rows,
        evaluation_only_rows=None,
):
    """Load ordinary taps as evaluation-only negatives with no fit access."""
    if not dataset_dirs:
        return []
    rows_by_hash = {}
    for path in dataset_dirs:
        root = Path(path).resolve()
        source_path = root / "ordinary_touch23.jsonl"
        if not source_path.is_file():
            source_path = root / "ordinary_touch26.jsonl"
        if not source_path.is_file():
            continue
        identifier = dataset_id(root)
        with source_path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                source_sha256 = str(row.get("sourceSha256") or "")
                if not source_sha256:
                    raise RuntimeError(
                        "ordinary_click_missing_source_sha256"
                    )
                if row.get("classLabel") != ORDINARY_CLASS_LABEL:
                    raise RuntimeError("invalid_ordinary_click_class_label")
                if (
                    row.get("collectionProtocol")
                    != ORDINARY_COLLECTION_PROTOCOL
                ):
                    raise RuntimeError(
                        "invalid_ordinary_click_collection_protocol"
                    )
                matrix = np.asarray(row.get("matrix"), dtype=np.float32)
                expected_shape = (
                    contract.FRAME_COUNT,
                    len(lightweight.FEATURE_NAMES),
                )
                if matrix.shape != expected_shape:
                    raise RuntimeError(
                        "invalid_ordinary_click_matrix_shape"
                    )
                copied = dict(row)
                copied["matrix"] = matrix
                copied["_datasetDir"] = str(root)
                copied["_datasetId"] = identifier
                copied["_evaluationOnly"] = True
                copied["_ordinaryClick"] = True
                existing = rows_by_hash.get(source_sha256)
                if existing is None:
                    rows_by_hash[source_sha256] = copied
                    continue
                if not np.array_equal(existing["matrix"], matrix):
                    raise RuntimeError(
                        "conflicting_duplicate_ordinary_click_source"
                    )
    if not rows_by_hash:
        return []
    training_hashes = {row["sourceSha256"] for row in training_rows}
    evaluation_hashes = {
        row["sourceSha256"] for row in (evaluation_only_rows or [])
    }
    if training_hashes & set(rows_by_hash):
        raise RuntimeError("ordinary_click_dataset_overlaps_training")
    if evaluation_hashes & set(rows_by_hash):
        raise RuntimeError(
            "ordinary_click_dataset_overlaps_evaluation_only"
        )
    ordinary_sessions = {
        str(row.get("groupId") or "").strip()
        for row in rows_by_hash.values()
        if str(row.get("groupId") or "").strip()
    }
    training_sessions = {
        str(row.get("groupId") or "").strip()
        for row in training_rows
        if str(row.get("groupId") or "").strip()
    }
    evaluation_sessions = {
        str(row.get("groupId") or "").strip()
        for row in (evaluation_only_rows or [])
        if str(row.get("groupId") or "").strip()
    }
    if ordinary_sessions & training_sessions:
        raise RuntimeError("ordinary_click_session_overlaps_training")
    if ordinary_sessions & evaluation_sessions:
        raise RuntimeError("ordinary_click_session_overlaps_evaluation_only")
    return sorted(
        rows_by_hash.values(),
        key=lambda row: row["sourceSha256"],
    )


def validate_rows(rows):
    """Validate labels and freeze the product gate target."""
    invalid = sorted({
        str(row.get("classLabel") or "")
        for row in rows
        if row.get("classLabel") not in contract.CLASS_LABELS
    })
    if invalid:
        raise RuntimeError("invalid_training_class_label")
    labels = [
        lightweight.CLASS_LABELS.index(row["classLabel"])
        for row in rows
    ]
    gate_labels = [
        contract.class_strength_target(row["classLabel"])
        for row in rows
    ]
    context_labels = [
        contract.context_targets_for_row(row)
        for row in rows
    ]
    return labels, gate_labels, context_labels


def physical_device_key(row):
    """Use model name as the physical device identity."""
    return device_normalization.physical_device_key(row)


def feature_mask(mode):
    """Return retained physical channels without changing matrix width."""
    if mode == "full":
        enabled = set(contract.PHYSICAL_FEATURE_NAMES)
    elif mode == "compact":
        enabled = (
            set(contract.PHYSICAL_FEATURE_NAMES)
            - {"x_norm", "y_norm", "size", "log_axis_ratio"}
        )
    elif mode == "no_absolute_xy":
        enabled = set(contract.PHYSICAL_FEATURE_NAMES) - {"x_norm", "y_norm"}
    elif mode == "no_shape":
        enabled = (
            set(contract.PHYSICAL_FEATURE_NAMES)
            - {"size", "log_axis_ratio"}
        )
    elif mode == "motion_only":
        enabled = {
            "userAcceleration.x",
            "userAcceleration.y",
            "userAcceleration.z",
            "rotationRate.x",
            "rotationRate.y",
            "rotationRate.z",
        }
    elif mode == "shape_only":
        enabled = {"size", "log_axis_ratio"}
    elif mode == "size_only":
        # 中文注释：单独检验接触面积通道，避免与轴比或 IMU 的收益混在一起。
        enabled = {"size"}
    elif mode == "axis_ratio_only":
        # 中文注释：轴比只保留无量纲接触椭圆比例，用于检验形状证据。
        enabled = {"log_axis_ratio"}
    elif mode == "motion_size":
        # 中文注释：IMU 与接触面积组合，用于判断尺寸能否补充运动冲击。
        enabled = {
            "userAcceleration.x",
            "userAcceleration.y",
            "userAcceleration.z",
            "rotationRate.x",
            "rotationRate.y",
            "rotationRate.z",
            "size",
        }
    elif mode == "motion_axis_ratio":
        # 中文注释：IMU 与轴比组合，隔离尺寸量纲和设备驱动差异。
        enabled = {
            "userAcceleration.x",
            "userAcceleration.y",
            "userAcceleration.z",
            "rotationRate.x",
            "rotationRate.y",
            "rotationRate.z",
            "log_axis_ratio",
        }
    elif mode == "magnitude_only":
        # 中文注释：幅值模式只保留三轴模量；实际模量在
        # apply_feature_mode 中计算，避免丢失旋转不变信息。
        enabled = {"userAcceleration.x", "rotationRate.x"}
    else:
        raise RuntimeError("unsupported_feature_mode")
    return np.asarray(
        [name in enabled for name in contract.PHYSICAL_FEATURE_NAMES],
        dtype=bool,
    )


def apply_observation_mask(values, rows):
    """Zero unavailable touch channels before sequence normalization."""
    result = values.copy()
    source_size_available = lightweight.FEATURE_NAMES.index(
        "size_available"
    )
    source_major_available = lightweight.FEATURE_NAMES.index(
        "major_available"
    )
    source_minor_available = lightweight.FEATURE_NAMES.index(
        "minor_available"
    )
    source_touch_valid = lightweight.FEATURE_NAMES.index("touch_valid")
    physical_size = contract.PHYSICAL_FEATURE_NAMES.index("size")
    physical_ratio = contract.PHYSICAL_FEATURE_NAMES.index(
        "log_axis_ratio"
    )
    for index, row in enumerate(rows):
        source = np.asarray(row["matrix"], dtype=np.float32)
        touch_valid = source[:, source_touch_valid] >= 0.5
        result[index, ~touch_valid, 0:2] = 0.0
        size_available = source[:, source_size_available] >= 0.5
        result[index, ~size_available, physical_size] = 0.0
        ratio_available = (
            (source[:, source_major_available] >= 0.5)
            & (source[:, source_minor_available] >= 0.5)
        )
        result[index, ~ratio_available, physical_ratio] = 0.0
    return result


def apply_contact_relative_size(values, rows):
    """Express contact extent relative to the same touch.

    中文注释：不同驱动的 size 和 major 量纲不可直接比较。
    这里优先使用本次触摸内可用的长轴，其次使用面积，并除以
    该次触摸自身的中位数。整个过程只读取当前样本，不估计
    设备或 session 统计，因此能用于首次接触的新设备。
    长轴和面积都缺失时保持零值，避免把缺失当成小接触。
    """
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] != len(rows):
        raise RuntimeError("contact_relative_row_count_mismatch")
    result = values.copy()
    source_size = lightweight.FEATURE_NAMES.index("size")
    source_major = lightweight.FEATURE_NAMES.index("major_norm")
    source_size_available = lightweight.FEATURE_NAMES.index(
        "size_available"
    )
    source_major_available = lightweight.FEATURE_NAMES.index(
        "major_available"
    )
    physical_size = contract.PHYSICAL_FEATURE_NAMES.index("size")
    for index, row in enumerate(rows):
        source = np.asarray(row["matrix"], dtype=np.float32)
        size_values = source[:, source_size]
        major_values = source[:, source_major]
        size_available = (
            (source[:, source_size_available] >= 0.5)
            & np.isfinite(size_values)
            & (size_values > 0.0)
        )
        major_available = (
            (source[:, source_major_available] >= 0.5)
            & np.isfinite(major_values)
            & (major_values > 0.0)
        )
        if np.any(major_available):
            extent = major_values
            available = major_available
        else:
            extent = size_values
            available = size_available
        if not np.any(available):
            result[index, :, physical_size] = 0.0
            continue
        reference = float(np.median(extent[available]))
        if not np.isfinite(reference) or reference <= 1e-8:
            result[index, :, physical_size] = 0.0
            continue
        # 中文注释：限制上尾避免单帧驱动异常吞掉其余时序变化。
        relative = np.clip(extent / reference, 0.0, 8.0)
        result[index, :, physical_size] = np.where(
            available,
            relative,
            0.0,
        )
    return result


def apply_contact_relative_shape(values, rows):
    """Replace absolute contact scale with within-touch relative changes.

    中文注释：统计特征仍使用固定的十通道合同，只把 size 与
    log_axis_ratio 两列改为本次触摸内部的相对量。基线取首个有效
    触摸帧附近的中位数，缺失通道继续为零。这样同一设备内部的
    接触变化被保留，跨设备的绝对量纲被移除。
    """
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] != len(rows):
        raise RuntimeError("contact_shape_row_count_mismatch")
    result = values.copy()
    source_size = lightweight.FEATURE_NAMES.index("size")
    source_major = lightweight.FEATURE_NAMES.index("major_norm")
    source_minor = lightweight.FEATURE_NAMES.index("minor_norm")
    source_touch_valid = lightweight.FEATURE_NAMES.index("touch_valid")
    source_size_available = lightweight.FEATURE_NAMES.index(
        "size_available"
    )
    source_major_available = lightweight.FEATURE_NAMES.index(
        "major_available"
    )
    source_minor_available = lightweight.FEATURE_NAMES.index(
        "minor_available"
    )
    physical_size = contract.PHYSICAL_FEATURE_NAMES.index("size")
    physical_ratio = contract.PHYSICAL_FEATURE_NAMES.index(
        "log_axis_ratio"
    )
    for index, row in enumerate(rows):
        source = np.asarray(row["matrix"], dtype=np.float32)
        touch_valid = source[:, source_touch_valid] >= 0.5
        size_values = source[:, source_size]
        size_valid = (
            touch_valid
            & (source[:, source_size_available] >= 0.5)
            & np.isfinite(size_values)
            & (size_values > 0.0)
        )
        major_values = source[:, source_major]
        minor_values = source[:, source_minor]
        ratio_valid = (
            touch_valid
            & (source[:, source_major_available] >= 0.5)
            & (source[:, source_minor_available] >= 0.5)
            & np.isfinite(major_values)
            & np.isfinite(minor_values)
            & (major_values > 0.0)
            & (minor_values > 0.0)
        )
        result[index, :, physical_size] = 0.0
        result[index, :, physical_ratio] = 0.0
        if np.any(size_valid):
            first_valid = int(np.flatnonzero(size_valid)[0])
            baseline_values = size_values[
                first_valid:first_valid + 2
            ][size_valid[first_valid:first_valid + 2]]
            baseline = (
                float(np.median(baseline_values))
                if len(baseline_values)
                else float(size_values[first_valid])
            )
            if np.isfinite(baseline) and baseline > 1e-8:
                relative = np.clip(
                    np.log(
                        np.maximum(size_values, 1e-8) / baseline
                    ),
                    -4.0,
                    4.0,
                )
                result[index, :, physical_size] = np.where(
                    size_valid,
                    relative,
                    0.0,
                )
        if np.any(ratio_valid):
            first_valid = int(np.flatnonzero(ratio_valid)[0])
            ratio_values = source[:, contract.SOURCE_FEATURE_NAMES.index(
                "log_axis_ratio"
            )]
            baseline_values = ratio_values[
                first_valid:first_valid + 2
            ][ratio_valid[first_valid:first_valid + 2]]
            baseline = (
                float(np.median(baseline_values))
                if len(baseline_values)
                else float(ratio_values[first_valid])
            )
            relative = np.clip(
                ratio_values - baseline,
                -4.0,
                4.0,
            )
            result[index, :, physical_ratio] = np.where(
                ratio_valid,
                relative,
                0.0,
            )
    return result


def apply_contact_relative_magnitude(values, rows):
    """Add touch-local relative contact shape to a magnitude window.

    中文注释：不同驱动的 size 和轴比量纲不可直接比较，但同一次触摸内
    的接触扩张与收缩仍包含按压信息。这里仅使用当前样本的首个有效帧
    作为基线，将 size 转为对数相对变化，将 log_axis_ratio 转为相对
    偏移。缺失通道保持零值，不估计设备或 session 统计，因此不会把
    外测设备信息带入训练。绝对加速度和旋转模量保持不变，避免删除
    已经验证有效的真实增益。
    """
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] != len(rows):
        raise RuntimeError("contact_magnitude_row_count_mismatch")
    result = values.copy()
    source_size_available = lightweight.FEATURE_NAMES.index(
        "size_available"
    )
    source_major_available = lightweight.FEATURE_NAMES.index(
        "major_available"
    )
    source_minor_available = lightweight.FEATURE_NAMES.index(
        "minor_available"
    )
    source_touch_valid = lightweight.FEATURE_NAMES.index("touch_valid")
    physical_size = contract.PHYSICAL_FEATURE_NAMES.index("size")
    physical_ratio = contract.PHYSICAL_FEATURE_NAMES.index(
        "log_axis_ratio"
    )
    for index, row in enumerate(rows):
        source = np.asarray(row["matrix"], dtype=np.float32)
        touch_valid = source[:, source_touch_valid] >= 0.5
        size = result[index, :, physical_size]
        size_available = (
            touch_valid
            & (source[:, source_size_available] >= 0.5)
            & np.isfinite(size)
            & (size > 0.0)
        )
        if np.any(size_available):
            valid_indices = np.flatnonzero(size_available)
            reference = float(np.median(
                size[valid_indices[:min(2, len(valid_indices))]]
            ))
            if np.isfinite(reference) and reference > 1e-8:
                result[index, :, physical_size] = np.where(
                    size_available,
                    np.clip(
                        np.log(
                            np.maximum(size, 1e-8) / reference
                        ),
                        -4.0,
                        4.0,
                    ),
                    0.0,
                )
            else:
                result[index, :, physical_size] = 0.0
        else:
            result[index, :, physical_size] = 0.0

        ratio = result[index, :, physical_ratio]
        ratio_available = (
            touch_valid
            & (source[:, source_major_available] >= 0.5)
            & (source[:, source_minor_available] >= 0.5)
            & np.isfinite(ratio)
        )
        if np.any(ratio_available):
            valid_indices = np.flatnonzero(ratio_available)
            reference = float(np.median(
                ratio[valid_indices[:min(2, len(valid_indices))]]
            ))
            result[index, :, physical_ratio] = np.where(
                ratio_available,
                np.clip(ratio - reference, -4.0, 4.0),
                0.0,
            )
        else:
            result[index, :, physical_ratio] = 0.0
    return result


def apply_availability_mask(values, rows):
    """Zero normalized values whose source modality was unavailable."""
    result = values.copy()
    source_touch_valid = lightweight.FEATURE_NAMES.index("touch_valid")
    source_size_available = lightweight.FEATURE_NAMES.index(
        "size_available"
    )
    source_major_available = lightweight.FEATURE_NAMES.index(
        "major_available"
    )
    source_minor_available = lightweight.FEATURE_NAMES.index(
        "minor_available"
    )
    physical_size = contract.PHYSICAL_FEATURE_NAMES.index("size")
    physical_ratio = contract.PHYSICAL_FEATURE_NAMES.index(
        "log_axis_ratio"
    )
    for index, row in enumerate(rows):
        source = np.asarray(row["matrix"], dtype=np.float32)
        touch_valid = source[:, source_touch_valid] >= 0.5
        size_available = source[:, source_size_available] >= 0.5
        ratio_available = (
            (source[:, source_major_available] >= 0.5)
            & (source[:, source_minor_available] >= 0.5)
        )
        result[index, ~touch_valid, 0:2] = 0.0
        result[index, ~size_available, physical_size] = 0.0
        result[index, ~ratio_available, physical_ratio] = 0.0
    return result


def touch_local_shape_features(rows):
    """Extract scale-free contact-shape dynamics from one touch window.

    中文注释：不同设备的 size、major 和 minor 量纲不同，因此这里只计算
    同一次触摸内部的相对变化。每个通道独立使用可用性掩码；有效帧不足
    两帧时只保留可用率为零的标记，其余特征为零，避免把缺失误当成小接触。
    """
    channel_specs = (
        ("size", "size", "size_available"),
        ("major", "major_norm", "major_available"),
        ("minor", "minor_norm", "minor_available"),
    )
    touch_valid_index = lightweight.FEATURE_NAMES.index("touch_valid")
    output = np.zeros(
        (len(rows), len(TOUCH_LOCAL_FEATURE_NAMES)),
        dtype=np.float32,
    )
    for row_index, row in enumerate(rows):
        source = np.asarray(row["matrix"], dtype=np.float32)
        touch_valid = source[:, touch_valid_index] >= 0.5
        for channel_index, (
            prefix,
            value_name,
            available_name,
        ) in enumerate(channel_specs):
            value_index = lightweight.FEATURE_NAMES.index(value_name)
            available_index = lightweight.FEATURE_NAMES.index(
                available_name
            )
            channel = source[:, value_index]
            available = (
                touch_valid
                & (source[:, available_index] >= 0.5)
                & np.isfinite(channel)
                & (channel > 0.0)
            )
            base = channel_index * 6
            output[row_index, base] = float(available.mean())
            if int(np.count_nonzero(available)) < 2:
                continue
            valid_indices = np.flatnonzero(available)
            values = channel[valid_indices].astype(np.float64)
            times = np.asarray(
                [TOUCH_GRID_MS[index] for index in valid_indices],
                dtype=np.float64,
            )
            median = float(np.median(values))
            if not np.isfinite(median) or median <= 1e-8:
                continue
            minimum = float(np.min(values))
            maximum = float(np.max(values))
            extent = maximum - minimum
            peak_index = int(np.argmax(values))
            peak = float(values[peak_index])
            span = float(times[-1] - times[0])
            if span <= 0.0:
                continue
            rise_time = float(
                (times[peak_index] - times[0]) / span
            )
            half_level = median + 0.5 * max(peak - median, 0.0)
            decay_time = 1.0
            for index in range(peak_index + 1, len(values)):
                if values[index] <= half_level:
                    decay_time = float(
                        (times[index] - times[peak_index]) / span
                    )
                    break
            normalized = (
                (values - minimum) / extent
                if extent > 1e-8
                else np.zeros_like(values)
            )
            area = float(np.sum(
                0.5
                * (normalized[:-1] + normalized[1:])
                * np.diff(times)
            ) / span)
            output[row_index, base + 1:base + 6] = (
                extent / median,
                (peak - median) / median,
                rise_time,
                decay_time,
                area,
            )
    if not np.isfinite(output).all():
        raise RuntimeError("nonfinite_touch_local_shape_features")
    return output


def causal_motion_features(rows):
    """Extract touch-local motion evidence relative to the pretouch baseline.

    中文注释：所有统计只读取当前样本的时间网格，不使用 session 内其他触摸。
    按下前三帧作为基线，后续帧相对该基线计算加速度与旋转变化。
    比值特征同时缩放分子和分母，可降低不同设备传感器增益造成的量纲差异。
    缺少 IMU 或有效帧不足时明确置零并输出未就绪标记，禁止按静止处理。
    """
    acceleration_index = lightweight.FEATURE_NAMES.index(
        "userAcceleration.x"
    )
    rotation_index = lightweight.FEATURE_NAMES.index("rotationRate.x")
    output = np.zeros(
        (len(rows), len(CAUSAL_MOTION_FEATURE_NAMES)),
        dtype=np.float32,
    )
    for row_index, row in enumerate(rows):
        source = np.asarray(row["matrix"], dtype=np.float32)
        acceleration = source[:, acceleration_index:acceleration_index + 3]
        rotation = source[:, rotation_index:rotation_index + 3]
        explicit_available = row.get("imuAvailable")
        sensor_source = str(row.get("sensorSource") or "").strip().lower()
        if explicit_available is False or sensor_source in (
            "none",
            "unavailable",
            "missing",
        ):
            continue
        acc_valid = np.isfinite(acceleration).all(axis=1)
        rot_valid = np.isfinite(rotation).all(axis=1)
        acc_coverage = acc_valid.astype(np.float32)
        rot_coverage = rot_valid.astype(np.float32)
        combined_coverage = np.minimum(acc_coverage, rot_coverage)
        output[row_index, 0] = float(
            combined_coverage[:TOUCH_START_FRAME].mean()
        )
        output[row_index, 1] = float(
            combined_coverage[TOUCH_START_FRAME:].mean()
        )
        if (
            int(np.count_nonzero(combined_coverage[:TOUCH_START_FRAME]))
            < 2
            or int(np.count_nonzero(
                combined_coverage[TOUCH_START_FRAME:]
            )) < 2
        ):
            continue
        output[row_index, 2] = 1.0
        for values, valid, prefix_index in (
            (acceleration, acc_valid, 3),
            (rotation, rot_valid, 8),
        ):
            pre_values = values[
                :TOUCH_START_FRAME
            ][valid[:TOUCH_START_FRAME]]
            touch_values = values[
                TOUCH_START_FRAME:
            ][valid[TOUCH_START_FRAME:]]
            if len(pre_values) < 2 or len(touch_values) < 2:
                continue
            baseline = np.median(pre_values, axis=0)
            pre_centered = pre_values - baseline
            pre_rms = float(np.sqrt(np.mean(
                np.sum(pre_centered * pre_centered, axis=1)
            )))
            touch_std = float(np.sqrt(np.mean(np.var(
                touch_values,
                axis=0,
            ))))
            sequence = np.concatenate((
                pre_values,
                touch_values,
            ), axis=0)
            sequence_delta = np.linalg.norm(
                np.diff(sequence, axis=0),
                axis=1,
            )
            touch_delta = sequence_delta[
                max(0, len(pre_values) - 1):
            ]
            touch_jerk_max = (
                float(np.max(touch_delta))
                if touch_delta.size
                else 0.0
            )
            pre_delta = np.linalg.norm(
                np.diff(pre_values, axis=0),
                axis=1,
            )
            pre_jerk_max = (
                float(np.max(pre_delta))
                if pre_delta.size
                else 0.0
            )
            scale_floor = max(pre_rms, 1e-6)
            jerk_floor = max(pre_jerk_max, 1e-6)
            output[row_index, prefix_index] = np.log1p(touch_std)
            output[row_index, prefix_index + 1] = np.log1p(
                touch_jerk_max
            )
            output[row_index, prefix_index + 2] = np.log1p(
                pre_rms / max(touch_std, 1e-6)
            )
            output[row_index, prefix_index + 3] = np.log1p(
                touch_std / scale_floor
            )
            output[row_index, prefix_index + 4] = np.log1p(
                touch_jerk_max / jerk_floor
            )
    if not np.isfinite(output).all():
        raise RuntimeError("nonfinite_causal_motion_features")
    return output


def historical_contact_features(rows):
    """Express each touch against earlier completed touches in its session.

    中文注释：只用同一 session 中时间更早、且尺寸可用的触摸建立基线。
    当前触摸及未来触摸均不参与统计；基线不足时全部置零并输出准备状态，
    避免把冷启动或缺失传感器误当成小接触。中位数负责位置估计，MAD
    负责稳健尺度，单次重触不会污染下一条样本的基线。
    """
    source_size = lightweight.FEATURE_NAMES.index("size")
    source_major = lightweight.FEATURE_NAMES.index("major_norm")
    source_minor = lightweight.FEATURE_NAMES.index("minor_norm")
    source_size_available = lightweight.FEATURE_NAMES.index(
        "size_available"
    )
    source_major_available = lightweight.FEATURE_NAMES.index(
        "major_available"
    )
    source_minor_available = lightweight.FEATURE_NAMES.index(
        "minor_available"
    )
    source_touch_valid = lightweight.FEATURE_NAMES.index("touch_valid")
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        timestamp = timestamp_ms(row)
        if timestamp is None:
            timestamp = _source_timestamp_ms(row)
        if timestamp is None:
            continue
        grouped[str(row.get("groupId") or "")].append(
            (timestamp, str(row.get("sourceSha256") or ""), index)
        )
    order_position = {}
    for session_id, items in grouped.items():
        if not session_id:
            continue
        ordered = sorted(items)
        for position, (_, source_sha256, index) in enumerate(ordered):
            order_position[index] = position
    output = np.zeros(
        (len(rows), len(HISTORICAL_CONTACT_FEATURE_NAMES)),
        dtype=np.float32,
    )
    size_median_ratio = np.zeros(len(rows), dtype=np.float32)
    size_peak_ratio = np.zeros(len(rows), dtype=np.float32)
    size_mad_relative = np.zeros(len(rows), dtype=np.float32)
    ratio_median_offset = np.zeros(len(rows), dtype=np.float32)
    ratio_peak_offset = np.zeros(len(rows), dtype=np.float32)
    for index, row in enumerate(rows):
        source = np.asarray(row["matrix"], dtype=np.float32)
        touch_valid = source[:, source_touch_valid] >= 0.5
        size = source[:, source_size]
        major = source[:, source_major]
        minor = source[:, source_minor]
        ratio = source[:, contract.SOURCE_FEATURE_NAMES.index(
            "log_axis_ratio"
        )]
        size_valid = (
            touch_valid
            & (source[:, source_size_available] >= 0.5)
            & np.isfinite(size)
            & (size > 0.0)
        )
        ratio_valid = (
            touch_valid
            & (source[:, source_major_available] >= 0.5)
            & (source[:, source_minor_available] >= 0.5)
            & np.isfinite(major)
            & np.isfinite(minor)
            & (major > 0.0)
            & (minor > 0.0)
        )
        output[index, 0] = float(size_valid.mean())
        history_size = []
        history_ratio = []
        history_age = []
        position = order_position.get(index)
        if position is not None:
            session_id = str(row.get("groupId") or "")
            earlier = [
                (timestamp, source_sha256, earlier_index)
                for timestamp, source_sha256, earlier_index
                in grouped[session_id]
                if earlier_index != index
                and earlier_index in order_position
                and order_position[earlier_index] < position
            ]
            earlier.sort()
            earlier = earlier[-HISTORICAL_CONTACT_MAX_TOUCHES:]
            current_timestamp = timestamp_ms(row)
            if current_timestamp is None:
                current_timestamp = _source_timestamp_ms(row)
            for timestamp, _, earlier_index in earlier:
                earlier_source = np.asarray(
                    rows[earlier_index]["matrix"],
                    dtype=np.float32,
                )
                earlier_valid = (
                    earlier_source[:, source_touch_valid] >= 0.5
                )
                earlier_size = earlier_source[:, source_size]
                earlier_size_valid = (
                    earlier_valid
                    & (
                        earlier_source[:, source_size_available]
                        >= 0.5
                    )
                    & np.isfinite(earlier_size)
                    & (earlier_size > 0.0)
                )
                if np.any(earlier_size_valid):
                    history_size.append(float(np.median(
                        earlier_size[earlier_size_valid]
                    )))
                    history_age.append(
                        max(0.0, float(current_timestamp - timestamp))
                    )
                earlier_major = earlier_source[:, source_major]
                earlier_minor = earlier_source[:, source_minor]
                earlier_ratio_valid = (
                    earlier_valid
                    & (
                        earlier_source[:, source_major_available]
                        >= 0.5
                    )
                    & (
                        earlier_source[:, source_minor_available]
                        >= 0.5
                    )
                    & np.isfinite(earlier_major)
                    & np.isfinite(earlier_minor)
                    & (earlier_major > 0.0)
                    & (earlier_minor > 0.0)
                )
                if np.any(earlier_ratio_valid):
                    earlier_ratio = earlier_source[
                        :, contract.SOURCE_FEATURE_NAMES.index(
                            "log_axis_ratio"
                        )
                    ]
                    history_ratio.append(float(np.median(
                        earlier_ratio[earlier_ratio_valid]
                    )))
        output[index, 1] = float(min(
            1.0,
            len(history_size) / HISTORICAL_CONTACT_MIN_TOUCHES,
        ))
        output[index, 2] = float(min(
            1.0,
            len(history_ratio) / HISTORICAL_CONTACT_MIN_TOUCHES,
        ))
        ready = (
            len(history_size) >= HISTORICAL_CONTACT_MIN_TOUCHES
            and np.any(size_valid)
        )
        output[index, 3] = float(ready)
        output[index, 4] = float(np.log1p(len(history_size)))
        output[index, 5] = float(
            np.log1p(np.median(history_age) / 1000.0)
            if history_age
            else 0.0
        )
        if not ready:
            continue
        current_size = float(np.median(size[size_valid]))
        history_median = float(np.median(history_size))
        history_mad = float(np.median(np.abs(
            np.asarray(history_size) - history_median
        )))
        if not np.isfinite(history_median) or history_median <= 1e-8:
            continue
        current_peak = float(np.max(size[size_valid]))
        size_median_ratio[index] = np.log(
            max(current_size, 1e-8) / history_median
        )
        size_peak_ratio[index] = np.log(
            max(current_peak, 1e-8) / history_median
        )
        size_mad_relative[index] = history_mad / history_median
        if (
            len(history_ratio) >= HISTORICAL_CONTACT_MIN_TOUCHES
            and np.any(ratio_valid)
        ):
            current_ratio = float(np.median(ratio[ratio_valid]))
            ratio_median_offset[index] = (
                current_ratio - float(np.median(history_ratio))
            )
            ratio_peak_offset[index] = (
                float(np.max(ratio[ratio_valid]))
                - float(np.median(history_ratio))
            )
    if not np.isfinite(output).all():
        raise RuntimeError("nonfinite_historical_contact_features")
    return output, (
        size_median_ratio,
        size_peak_ratio,
        size_mad_relative,
        ratio_median_offset,
        ratio_peak_offset,
    )


def apply_historical_contact_relative(values, rows):
    """Replace contact scale with history-relative causal evidence."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] != len(rows):
        raise RuntimeError("historical_contact_row_count_mismatch")
    result = values.copy()
    (
        metadata,
        derived,
    ) = historical_contact_features(rows)
    physical_size = contract.PHYSICAL_FEATURE_NAMES.index("size")
    physical_ratio = contract.PHYSICAL_FEATURE_NAMES.index(
        "log_axis_ratio"
    )
    result[:, :, physical_size] = 0.0
    result[:, :, physical_ratio] = 0.0
    ready = metadata[:, 3] >= 0.5
    result[ready, :, physical_size] = derived[0][ready, None]
    result[ready, :, physical_ratio] = derived[3][ready, None]
    return result, (metadata, derived)


def historical_contact_metadata(derived):
    """Summarize causal history coverage without exposing touch labels."""
    metadata, ratios = derived
    ready = metadata[:, 3] >= 0.5
    return {
        "rowCount": int(len(metadata)),
        "readyCount": int(np.count_nonzero(ready)),
        "readyFraction": float(np.mean(ready)) if len(ready) else 0.0,
        "minimumHistoryTouches": HISTORICAL_CONTACT_MIN_TOUCHES,
        "maximumHistoryTouches": HISTORICAL_CONTACT_MAX_TOUCHES,
        "currentSizeCoverage": float(np.mean(metadata[:, 0])),
        "historySizeCoverage": float(np.mean(metadata[:, 1])),
        "historyRatioCoverage": float(np.mean(metadata[:, 2])),
        "medianSizeMedianRatio": (
            float(np.median(ratios[0][ready]))
            if np.any(ready)
            else None
        ),
        "medianSizeMadRelative": (
            float(np.median(ratios[2][ready]))
            if np.any(ready)
            else None
        ),
    }


def _contact_channel_values(row):
    """Extract available contact extent and axis values for one touch."""
    source = np.asarray(row["matrix"], dtype=np.float32)
    size_index = lightweight.FEATURE_NAMES.index("size")
    major_index = lightweight.FEATURE_NAMES.index("major_norm")
    minor_index = lightweight.FEATURE_NAMES.index("minor_norm")
    touch_valid = (
        source[:, lightweight.FEATURE_NAMES.index("touch_valid")] >= 0.5
    )
    size = source[:, size_index]
    size_valid = (
        touch_valid
        & (
            source[:, lightweight.FEATURE_NAMES.index("size_available")]
            >= 0.5
        )
        & np.isfinite(size)
        & (size > 0.0)
    )
    major = source[:, major_index]
    minor = source[:, minor_index]
    ratio_valid = (
        touch_valid
        & (
            source[:, lightweight.FEATURE_NAMES.index("major_available")]
            >= 0.5
        )
        & (
            source[:, lightweight.FEATURE_NAMES.index("minor_available")]
            >= 0.5
        )
        & np.isfinite(major)
        & np.isfinite(minor)
        & (major > 0.0)
        & (minor > 0.0)
    )
    ratio = source[:, contract.SOURCE_FEATURE_NAMES.index(
        "log_axis_ratio"
    )]
    return size, size_valid, ratio, ratio_valid


def device_causal_contact_features(rows):
    """Calibrate contact scale from earlier touches on the same device.

    中文注释：当前数据中 session 与力度标签高度绑定，同一 session 的历史
    基线只能看到相同力度。这里改用同一物理设备上严格早于当前触摸的已完成
    触摸，估计设备尺度；当前触摸和未来触摸都不进入基线。冷启动阶段保留
    原始通道，由训练集缩放器提供保守回退。校准后的相对值与当前触摸内部
    的时序统计组合，既不依赖绝对设备量纲，也不会把同一力度当成参照。
    """
    output = np.zeros((len(rows), 4), dtype=np.float32)
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        timestamp = timestamp_ms(row)
        if timestamp is None:
            timestamp = _source_timestamp_ms(row)
        if timestamp is None:
            continue
        grouped[physical_device_key(row)].append((
            timestamp,
            str(row.get("sourceSha256") or ""),
            index,
        ))
    positions = {}
    ordered_by_device = {}
    for device, items in grouped.items():
        ordered = sorted(items)
        ordered_by_device[device] = ordered
        for position, (_, _, index) in enumerate(ordered):
            positions[(device, index)] = position
    for index, row in enumerate(rows):
        device = physical_device_key(row)
        size, size_valid, ratio, ratio_valid = _contact_channel_values(row)
        output[index, 0] = float(size_valid.mean())
        output[index, 2] = float(ratio_valid.mean())
        position = positions.get((device, index))
        if position is None:
            continue
        ordered = ordered_by_device[device]
        history = ordered[
            max(0, position - DEVICE_CAUSAL_CONTACT_MAX_TOUCHES):
            position
        ]
        history_size = []
        history_ratio = []
        for _, _, earlier_index in history:
            earlier_size, earlier_size_valid, earlier_ratio, earlier_ratio_valid = (
                _contact_channel_values(rows[earlier_index])
            )
            if np.any(earlier_size_valid):
                history_size.append(float(np.median(
                    earlier_size[earlier_size_valid]
                )))
            if np.any(earlier_ratio_valid):
                history_ratio.append(float(np.median(
                    earlier_ratio[earlier_ratio_valid]
                )))
        output[index, 1] = float(min(
            1.0,
            len(history_size) / DEVICE_CAUSAL_CONTACT_MIN_TOUCHES,
        ))
        output[index, 3] = float(min(
            1.0,
            len(history_ratio) / DEVICE_CAUSAL_CONTACT_MIN_TOUCHES,
        ))
        size_ready = (
            len(history_size) >= DEVICE_CAUSAL_CONTACT_MIN_TOUCHES
            and np.any(size_valid)
        )
        if size_ready:
            history_median = float(np.median(history_size))
            if np.isfinite(history_median) and history_median > 1e-8:
                output[index, 0] = float(np.log(
                    max(float(np.median(size[size_valid])), 1e-8)
                    / history_median
                ))
        ratio_ready = (
            len(history_ratio) >= DEVICE_CAUSAL_CONTACT_MIN_TOUCHES
            and np.any(ratio_valid)
        )
        if ratio_ready:
            output[index, 2] = (
                float(np.median(ratio[ratio_valid]))
                - float(np.median(history_ratio))
            )
        output[index, 1] = float(size_ready)
        output[index, 3] = float(ratio_ready)
    if not np.isfinite(output).all():
        raise RuntimeError("nonfinite_device_causal_contact_features")
    return output


def apply_device_causal_contact(values, rows):
    """Replace absolute contact scale with causal device-relative evidence."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] != len(rows):
        raise RuntimeError("device_causal_contact_row_count_mismatch")
    result = values.copy()
    features = device_causal_contact_features(rows)
    size_channel = contract.PHYSICAL_FEATURE_NAMES.index("size")
    ratio_channel = contract.PHYSICAL_FEATURE_NAMES.index("log_axis_ratio")
    result[:, :, size_channel] = 0.0
    result[:, :, ratio_channel] = 0.0
    ready_size = features[:, 1] >= 1.0
    ready_ratio = features[:, 3] >= 1.0
    result[ready_size, :, size_channel] = features[
        ready_size,
        0,
    ][:, None]
    result[ready_ratio, :, ratio_channel] = (
        features[ready_ratio, 2][:, None]
    )
    return result, features


def device_causal_contact_metadata(features):
    """Describe causal device calibration coverage without class leakage."""
    size_ready = features[:, 1] >= 1.0
    ratio_ready = features[:, 3] >= 1.0
    return {
        "rowCount": int(len(features)),
        "sizeReadyCount": int(np.count_nonzero(size_ready)),
        "sizeReadyFraction": (
            float(np.mean(size_ready)) if len(features) else 0.0
        ),
        "ratioReadyCount": int(np.count_nonzero(ratio_ready)),
        "ratioReadyFraction": (
            float(np.mean(ratio_ready)) if len(features) else 0.0
        ),
        "minimumHistoryTouches": DEVICE_CAUSAL_CONTACT_MIN_TOUCHES,
        "maximumHistoryTouches": DEVICE_CAUSAL_CONTACT_MAX_TOUCHES,
    }


def append_touch_local_shape_features(values, rows):
    """Append explicit local-shape features to the statistics matrix."""
    values = np.asarray(values, dtype=np.float32)
    local = touch_local_shape_features(rows)
    if values.shape[0] != local.shape[0]:
        raise RuntimeError("touch_local_shape_row_count_mismatch")
    return np.concatenate((values, local), axis=1).astype(np.float32)


def append_causal_motion_features(values, rows, include_shape=False):
    """Append causal motion and optionally touch-local contact shape."""
    values = np.asarray(values, dtype=np.float32)
    motion = causal_motion_features(rows)
    if values.shape[0] != motion.shape[0]:
        raise RuntimeError("causal_motion_row_count_mismatch")
    pieces = [values, motion]
    if include_shape:
        pieces.append(touch_local_shape_features(rows))
    return np.concatenate(pieces, axis=1).astype(np.float32)


def apply_feature_mode(values, mode):
    """Zero excluded channels while retaining the frozen 60-value contract."""
    if mode == "historical_contact_relative_no_xy":
        result = values.copy()
        result[:, :, 0:2] = 0.0
        return result
    if mode in (
        "raw_contact_no_xy",
        "device_causal_contact_no_xy",
    ):
        result = values.copy()
        result[:, :, 0:2] = 0.0
        return result
    if mode in (
        "raw_contact_magnitude_no_xy",
        "device_causal_contact_magnitude_no_xy",
    ):
        result = values.copy()
        result[:, :, 2] = np.linalg.norm(
            values[:, :, 2:5],
            axis=2,
        )
        result[:, :, 5] = np.linalg.norm(
            values[:, :, 5:8],
            axis=2,
        )
        result[:, :, [0, 1, 3, 4, 6, 7]] = 0.0
        return result
    if mode == "historical_contact_magnitude_no_xy":
        result = values.copy()
        result[:, :, 2] = np.linalg.norm(
            values[:, :, 2:5],
            axis=2,
        )
        result[:, :, 5] = np.linalg.norm(
            values[:, :, 5:8],
            axis=2,
        )
        result[:, :, [0, 1, 3, 4, 6, 7]] = 0.0
        return result
    if mode == "historical_contact_relative_no_shape":
        result = values.copy()
        result[:, :, contract.PHYSICAL_FEATURE_NAMES.index(
            "log_axis_ratio"
        )] = 0.0
        return result
    if mode == "magnitude_historical_contact":
        result = values.copy()
        result[:, :, 2] = np.linalg.norm(
            values[:, :, 2:5],
            axis=2,
        )
        result[:, :, 5] = np.linalg.norm(
            values[:, :, 5:8],
            axis=2,
        )
        result[:, :, [0, 1, 3, 4, 6, 7]] = 0.0
        return result
    if mode == "historical_contact_motion_no_shape":
        result = values.copy()
        result[:, :, 2] = np.linalg.norm(
            values[:, :, 2:5],
            axis=2,
        )
        result[:, :, 5] = np.linalg.norm(
            values[:, :, 5:8],
            axis=2,
        )
        # 中文注释：保留历史相对尺寸，移除绝对坐标和轴比。
        result[:, :, [0, 1, 3, 4, 6, 7, 9]] = 0.0
        return result
    if mode == "historical_contact_motion_shape":
        result = values.copy()
        result[:, :, 2] = np.linalg.norm(
            values[:, :, 2:5],
            axis=2,
        )
        result[:, :, 5] = np.linalg.norm(
            values[:, :, 5:8],
            axis=2,
        )
        # 中文注释：保留历史相对尺寸与轴比，并追加局部接触时序。
        result[:, :, [0, 1, 3, 4, 6, 7]] = 0.0
        return result
    if mode == "historical_contact_local_shape":
        result = values.copy()
        # 中文注释：仅保留历史相对尺寸，局部时序随后单独追加。
        result[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 9]] = 0.0
        return result
    if mode in CAUSAL_MOTION_MODES:
        result = values.copy()
        # 中文注释：保留历史相对尺寸；形态模式同时保留历史相对轴比。
        retained_channels = {8}
        if mode in CAUSAL_MOTION_SHAPE_MODES:
            retained_channels.add(9)
        for channel in range(values.shape[2]):
            if channel not in retained_channels:
                result[:, :, channel] = 0.0
        return result
    if mode in (
        "physical_magnitude",
        "physical_magnitude_temporal",
        "physical_magnitude_shape",
        "physical_magnitude_temporal_shape",
        "device_relative_magnitude",
        "historical_contact_relative",
        "device_causal_contact",
    ):
        return values
    if mode == "touch_local_shape":
        return np.zeros_like(values)
    if mode in ("touch_local_motion", "touch_local_motion_shape"):
        mode = "magnitude_only"
    if mode in ("magnitude_only", "motion_size", "motion_axis_ratio"):
        result = values.copy()
        result[:, :, 2] = np.linalg.norm(
            values[:, :, 2:5],
            axis=2,
        )
        result[:, :, 5] = np.linalg.norm(
            values[:, :, 5:8],
            axis=2,
        )
        result[:, :, [0, 1, 3, 4, 6, 7]] = 0.0
        if mode == "magnitude_only":
            result[:, :, 8:10] = 0.0
        elif mode == "motion_size":
            result[:, :, 9] = 0.0
        else:
            result[:, :, 8] = 0.0
        return result
    if mode == "contact_relative_magnitude":
        result = values.copy()
        result[:, :, 2] = np.linalg.norm(
            values[:, :, 2:5],
            axis=2,
        )
        result[:, :, 5] = np.linalg.norm(
            values[:, :, 5:8],
            axis=2,
        )
        # 中文注释：保留相对接触尺寸列，其余非模量列清零。
        result[:, :, [0, 1, 3, 4, 6, 7, 9]] = 0.0
        return result
    if mode == "contact_relative_magnitude_shape":
        result = values.copy()
        result[:, :, 2] = np.linalg.norm(
            values[:, :, 2:5],
            axis=2,
        )
        result[:, :, 5] = np.linalg.norm(
            values[:, :, 5:8],
            axis=2,
        )
        result[:, :, [0, 1, 3, 4, 6, 7]] = 0.0
        return result
    if mode == "magnitude_contact_relative":
        result = values.copy()
        result[:, :, 2] = np.linalg.norm(
            values[:, :, 2:5],
            axis=2,
        )
        result[:, :, 5] = np.linalg.norm(
            values[:, :, 5:8],
            axis=2,
        )
        # 中文注释：仅保留两路模量与触摸局部相对形态。
        result[:, :, [0, 1, 3, 4, 6, 7]] = 0.0
        return result
    if mode == "magnitude_temporal":
        result = np.zeros_like(values)
        acceleration = values[:, :, 2:5]
        rotation = values[:, :, 5:8]
        acceleration_magnitude = np.linalg.norm(
            acceleration,
            axis=2,
        )
        rotation_magnitude = np.linalg.norm(rotation, axis=2)
        acceleration_delta = np.zeros_like(acceleration_magnitude)
        rotation_delta = np.zeros_like(rotation_magnitude)
        acceleration_delta[:, 1:] = np.abs(
            np.diff(acceleration_magnitude, axis=1)
        )
        rotation_delta[:, 1:] = np.abs(
            np.diff(rotation_magnitude, axis=1)
        )
        # 中文注释：前 3 帧是按下前基线，后续时间片才代表本次触摸。
        acceleration_relative = np.maximum(
            acceleration_magnitude
            - acceleration_magnitude[:, :TOUCH_START_FRAME].mean(
                axis=1,
                keepdims=True,
            ),
            0.0,
        )
        rotation_relative = np.maximum(
            rotation_magnitude
            - rotation_magnitude[:, :TOUCH_START_FRAME].mean(
                axis=1,
                keepdims=True,
            ),
            0.0,
        )
        ramp = np.linspace(0.0, 1.0, values.shape[1], dtype=np.float32)
        result[:, :, 0] = acceleration_magnitude
        result[:, :, 1] = rotation_magnitude
        result[:, :, 2] = acceleration_magnitude
        result[:, :, 3] = acceleration_delta
        result[:, :, 4] = acceleration_relative
        result[:, :, 5] = rotation_magnitude
        result[:, :, 6] = rotation_delta
        result[:, :, 7] = rotation_relative
        result[:, :, 8] = acceleration_relative * ramp[None, :]
        result[:, :, 9] = rotation_relative * ramp[None, :]
        return result
    retained = feature_mask(mode)
    if bool(retained.all()):
        return values
    result = values.copy()
    result[:, :, ~retained] = 0.0
    return result


def gravity_rotation_to_canonical(gravity):
    """Return a rotation that maps one device gravity vector to +Z.

    中文注释：这里只使用重力方向建立局部坐标系，不把重力向量本身送入分类器。
    同一物理动作在设备坐标系中整体旋转后，会得到一致的加速度与角速度方向，
    从训练源头消除正放、倒置和侧卧造成的坐标轴捷径。
    """
    vector = np.asarray(gravity, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        raise RuntimeError("invalid_gravity_vector")
    source = vector / norm
    target = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if cosine > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if cosine < -1.0 + 1e-12:
        # 中文注释：完全反向时叉积退化，选择与源向量正交的任意稳定轴。
        helper = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        if abs(float(np.dot(source, helper))) > 0.9:
            helper = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
        return (
            2.0 * np.outer(axis, axis)
            - np.eye(3, dtype=np.float64)
        )
    cross = np.cross(source, target)
    skew = np.asarray((
        (0.0, -cross[2], cross[1]),
        (cross[2], 0.0, -cross[0]),
        (-cross[1], cross[0], 0.0),
    ), dtype=np.float64)
    return (
        np.eye(3, dtype=np.float64)
        + skew
        + np.matmul(skew, skew) / (1.0 + cosine)
    )


def apply_gravity_alignment(values, rows):
    """Align acceleration and angular velocity using pre-touch gravity."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] != len(rows):
        raise RuntimeError("gravity_alignment_row_count_mismatch")
    result = values.copy()
    for index, row in enumerate(rows):
        source = np.asarray(row["matrix"], dtype=np.float32)
        if source.ndim != 2 or source.shape[1] < 11:
            raise RuntimeError("gravity_alignment_missing_source_matrix")
        pre_gravity = source[:TOUCH_START_FRAME, 8:11]
        valid = (
            np.isfinite(pre_gravity).all(axis=1)
            & (np.linalg.norm(pre_gravity, axis=1) >= 1e-8)
        )
        if not np.any(valid):
            raise RuntimeError("gravity_alignment_missing_valid_pretouch")
        gravity = np.median(pre_gravity[valid], axis=0)
        rotation = gravity_rotation_to_canonical(gravity)
        acceleration = result[index, :, 2:5].astype(np.float64)
        angular_velocity = result[index, :, 5:8].astype(np.float64)
        result[index, :, 2:5] = np.matmul(
            acceleration,
            rotation.T,
        ).astype(np.float32)
        result[index, :, 5:8] = np.matmul(
            angular_velocity,
            rotation.T,
        ).astype(np.float32)
    return result


def apply_device_relative_magnitude(values, rows):
    """Convert raw inertial channels into unit-consistent touch evidence.

    中文注释：加速度除以重力尺度，并同时除以按下前噪声基线；
    角速度也按本次触摸前的噪声归一化。这样分类器比较的是相对于
    用户和设备自身静止噪声的冲击倍数，避免固定绝对阈值漂移。
    """
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] != len(rows):
        raise RuntimeError("relative_magnitude_row_count_mismatch")
    result = np.zeros_like(values)
    ramp = np.linspace(
        0.0,
        1.0,
        values.shape[1],
        dtype=np.float32,
    )
    for index, row in enumerate(rows):
        source = np.asarray(row["matrix"], dtype=np.float32)
        if source.ndim != 2 or source.shape[1] < 11:
            raise RuntimeError("relative_magnitude_missing_gravity")
        gravity = source[:TOUCH_START_FRAME, 8:11]
        gravity_norm = np.linalg.norm(gravity, axis=1)
        valid_gravity = (
            np.isfinite(gravity_norm)
            & (gravity_norm >= 1e-6)
        )
        if not np.any(valid_gravity):
            raise RuntimeError("relative_magnitude_missing_gravity")
        gravity_scale = float(np.median(gravity_norm[valid_gravity]))
        acceleration_raw = np.linalg.norm(
            values[index, :, 2:5],
            axis=1,
        )
        rotation_raw = np.linalg.norm(
            values[index, :, 5:8],
            axis=1,
        )
        acceleration = acceleration_raw / gravity_scale
        acceleration_floor = 0.05 * gravity_scale
        acceleration_baseline = max(
            float(np.median(
                acceleration_raw[:TOUCH_START_FRAME]
            )),
            acceleration_floor,
        )
        rotation_baseline = max(
            float(np.median(
                rotation_raw[:TOUCH_START_FRAME]
            )),
            0.02,
        )
        acceleration_relative = acceleration_raw / acceleration_baseline
        rotation_relative = rotation_raw / rotation_baseline
        acceleration_delta = np.zeros_like(acceleration)
        rotation_delta = np.zeros_like(rotation_raw)
        acceleration_delta[1:] = np.abs(
            np.diff(acceleration_relative)
        )
        rotation_delta[1:] = np.abs(
            np.diff(rotation_relative)
        )
        acceleration_onset = np.maximum(
            acceleration_relative - 1.0,
            0.0,
        )
        rotation_onset = np.maximum(
            rotation_relative - 1.0,
            0.0,
        )
        result[index, :, 0] = np.log1p(acceleration_relative)
        result[index, :, 1] = np.log1p(rotation_relative)
        result[index, :, 2] = np.log1p(acceleration_relative)
        result[index, :, 3] = acceleration_delta
        result[index, :, 4] = acceleration_onset
        result[index, :, 5] = np.log1p(rotation_relative)
        result[index, :, 6] = rotation_delta
        result[index, :, 7] = rotation_onset
        result[index, :, 8] = acceleration_onset * ramp
        result[index, :, 9] = rotation_onset * ramp
    return result


def apply_physical_magnitude(values, temporal=False):
    """Build orientation-invariant magnitudes before per-axis scaling."""
    values = np.asarray(values, dtype=np.float32)
    result = np.zeros_like(values)
    acceleration = values[:, :, 2:5]
    rotation = values[:, :, 5:8]
    acceleration_magnitude = np.linalg.norm(acceleration, axis=2)
    rotation_magnitude = np.linalg.norm(rotation, axis=2)
    result[:, :, 0] = acceleration_magnitude
    result[:, :, 1] = rotation_magnitude
    result[:, :, 2] = acceleration_magnitude
    result[:, :, 5] = rotation_magnitude
    if not temporal:
        return result
    acceleration_delta = np.zeros_like(acceleration_magnitude)
    rotation_delta = np.zeros_like(rotation_magnitude)
    acceleration_delta[:, 1:] = np.abs(
        np.diff(acceleration_magnitude, axis=1)
    )
    rotation_delta[:, 1:] = np.abs(
        np.diff(rotation_magnitude, axis=1)
    )
    acceleration_baseline = np.median(
        acceleration_magnitude[:, :TOUCH_START_FRAME],
        axis=1,
        keepdims=True,
    )
    rotation_baseline = np.median(
        rotation_magnitude[:, :TOUCH_START_FRAME],
        axis=1,
        keepdims=True,
    )
    acceleration_onset = np.maximum(
        acceleration_magnitude - acceleration_baseline,
        0.0,
    )
    rotation_onset = np.maximum(
        rotation_magnitude - rotation_baseline,
        0.0,
    )
    ramp = np.linspace(
        0.0,
        1.0,
        values.shape[1],
        dtype=np.float32,
    )
    result[:, :, 3] = acceleration_delta
    result[:, :, 4] = acceleration_onset
    result[:, :, 6] = rotation_delta
    result[:, :, 7] = rotation_onset
    result[:, :, 8] = acceleration_onset * ramp[None, :]
    result[:, :, 9] = rotation_onset * ramp[None, :]
    return result


def gravity_alignment_coverage(rows):
    """Count rows with a usable pre-touch gravity direction."""
    available = 0
    for row in rows:
        source = np.asarray(row["matrix"], dtype=np.float32)
        if source.ndim != 2 or source.shape[1] < 11:
            continue
        gravity = source[:TOUCH_START_FRAME, 8:11]
        valid = (
            np.isfinite(gravity).all(axis=1)
            & (np.linalg.norm(gravity, axis=1) >= 1e-8)
        )
        if np.any(valid):
            available += 1
    return {
        "availableCount": available,
        "unavailableCount": len(rows) - available,
        "coverage": available / len(rows) if rows else 0.0,
    }


def apply_imu_mode(values, mode):
    """Remove pre-contact bias while guarding against press onset.

    中文注释：时间网格的前三帧名义上位于 ACTION_DOWN 之前，但传感器
    可能已经记录了手指开始加力的上升段。直接对三帧取中位数会把按压
    起始当成噪声基线，缩小真正的冲击幅值。受保护模式检测到持续上升时
    只使用最早一帧，否则仍使用三帧中位数；该选择只读取当前帧及其之前
    的帧，不引入未来标签或未来窗口。
    """
    if mode == "raw":
        return values
    if mode == "gravity_aligned":
        return values
    if mode == "gravity_aligned_pretouch_centered":
        mode = "pretouch_centered"
    if mode == "gravity_aligned_guarded_pretouch_centered":
        mode = "guarded_pretouch_centered"
    if mode == "gravity_aligned_earliest_pretouch_centered":
        mode = "earliest_pretouch_centered"
    if mode not in (
        "pretouch_centered",
        "guarded_pretouch_centered",
        "earliest_pretouch_centered",
    ):
        raise RuntimeError("unsupported_imu_mode")
    result = values.copy()
    pretouch = values[:, :TOUCH_START_FRAME, 2:8]
    baseline = np.median(pretouch, axis=1)
    if mode in (
        "guarded_pretouch_centered",
        "earliest_pretouch_centered",
    ):
        # 中文注释：标准化后相邻帧的中位差为正，表示六个惯性通道整体
        # 在按压前已持续上升。阈值只用于拒绝明显上升，避免把正常抖动
        # 误判为污染；没有上升时保留较稳的三帧中位数。
        diffs = np.diff(pretouch, axis=1)
        rising = (
            np.median(diffs, axis=(1, 2)) > 0.10
        ) & (
            np.median(pretouch[:, -1] - pretouch[:, 0], axis=1)
            > 0.20
        )
        if mode == "earliest_pretouch_centered":
            rising[:] = True
        baseline[rising] = pretouch[rising, 0]
    result[:, :, 2:8] -= baseline[:, None, :]
    return result


def uses_gravity_alignment(mode):
    """Return whether the mode rotates vectors into a gravity frame."""
    return mode in (
        "gravity_aligned",
        "gravity_aligned_pretouch_centered",
        "gravity_aligned_guarded_pretouch_centered",
        "gravity_aligned_earliest_pretouch_centered",
    )


def apply_pretouch_delta(values):
    """Remove the first frame while retaining relative motion and shape."""
    result = np.zeros_like(values)
    result[:, 1:, 2:8] = values[:, 1:, 2:8] - values[:, :-1, 2:8]
    result[:, 1:, 8:10] = values[:, 1:, 8:10]
    return result


def apply_adaptive_features(values, mode):
    """Build deployable residuals without changing the frozen input width."""
    if mode == "none":
        return values
    if mode == "imu_delta":
        return apply_pretouch_delta(values)
    if mode == "imu_delta_shape":
        result = apply_pretouch_delta(values)
        result[:, :, 8:10] = values[:, :, 8:10]
        return result
    raise RuntimeError("unsupported_adaptive_feature_mode")


def apply_modality_dropout(values, rows, probability, seed):
    """Randomly hide optional touch-shape channels for training only."""
    if probability <= 0.0:
        return values
    if not 0.0 <= probability < 1.0:
        raise RuntimeError("invalid_modality_dropout")
    result = values.copy()
    physical_size = contract.PHYSICAL_FEATURE_NAMES.index("size")
    physical_ratio = contract.PHYSICAL_FEATURE_NAMES.index(
        "log_axis_ratio"
    )
    rng = np.random.default_rng(seed)
    shape_present = np.asarray([
        np.any(np.asarray(row["matrix"], dtype=np.float32)[:, 12] != 0.0)
        for row in rows
    ], dtype=bool)
    drop_size = shape_present & (rng.random(len(rows)) < probability)
    drop_ratio = shape_present & (rng.random(len(rows)) < probability)
    result[drop_size, :, physical_size] = 0.0
    result[drop_ratio, :, physical_ratio] = 0.0
    return result


def apply_channel_scale_augmentation(
        values,
        rows,
        probability,
        seed,
        maximum_log_scale=0.35,
):
    """Randomly rescale sensor groups during fitting only.

    中文注释：不同设备的加速度、旋转率和接触尺寸存在整体增益差。
    训练时对整段触摸施加随机乘性缩放，让模型不能把某一设备的绝对
    量纲当作类别捷径。校准集、验证集和外测集保持原值，保证离线与
    端侧合同一致。缩放只作用于物理量，不改变时间形状和可用性掩码。
    """
    if probability <= 0.0:
        return values
    if not 0.0 <= probability <= 1.0:
        raise RuntimeError("invalid_channel_scale_probability")
    if not 0.0 < maximum_log_scale <= 1.0:
        raise RuntimeError("invalid_channel_scale_range")
    result = np.asarray(values, dtype=np.float32).copy()
    rng = np.random.default_rng(seed)
    groups = (
        (2, 5),
        (5, 8),
        (8, 10),
    )
    for start, stop in groups:
        selected = rng.random(len(rows)) < probability
        log_scale = rng.uniform(
            -maximum_log_scale,
            maximum_log_scale,
            size=len(rows),
        )
        factors = np.where(
            selected,
            np.exp(log_scale),
            1.0,
        ).astype(np.float32)
        result[selected, :, start:stop] *= factors[
            selected,
            None,
            None,
        ]
    return result


def ensure_fold_coverage(train_rows, valid_rows):
    """Require every class and both gate targets on both sides."""
    if not train_rows or not valid_rows:
        raise RuntimeError("empty_validation_fold")
    for side, rows in (("train", train_rows), ("valid", valid_rows)):
        class_labels = {row["classLabel"] for row in rows}
        gate_labels = {
            contract.class_strength_target(row["classLabel"])
            for row in rows
        }
        if class_labels != set(contract.CLASS_LABELS):
            raise RuntimeError(side + "_missing_diagnostic_class")
        if gate_labels != set(range(len(contract.GATE_LABELS))):
            raise RuntimeError(side + "_missing_gate_class")
    if (
        {row["groupId"] for row in train_rows}
        & {row["groupId"] for row in valid_rows}
    ):
        raise RuntimeError("cross_fold_group_leakage")
    train_blocks = {
        experiment_block_key(row)
        for row in train_rows
        if experiment_block_key(row)
    }
    valid_blocks = {
        experiment_block_key(row)
        for row in valid_rows
        if experiment_block_key(row)
    }
    if train_blocks & valid_blocks:
        raise RuntimeError("cross_fold_experiment_block_leakage")


def make_fold(kind, identifier, train_rows, valid_rows):
    ensure_fold_coverage(train_rows, valid_rows)
    return {
        "kind": kind,
        "id": identifier,
        "train": train_rows,
        "valid": valid_rows,
    }


def build_device_folds(rows):
    """Leave one physical device out at a time."""
    by_device = defaultdict(list)
    for row in rows:
        by_device[physical_device_key(row)].append(row)
    if len(by_device) < 2:
        return []
    folds = []
    for device in sorted(by_device):
        valid_rows = by_device[device]
        train_rows = [
            row for row in rows
            if physical_device_key(row) != device
        ]
        try:
            fold = make_fold(
                "device",
                "device:" + device,
                train_rows,
                valid_rows,
            )
            # 中文注释：设备折可能合并多个派生数据集，实验块会跨 session；
            # 校准集必须按完整实验块切分，否则会污染同块上下文。
            fold["calibrationGroupField"] = "_experimentBlock"
            folds.append(fold)
        except RuntimeError:
            continue
    return folds


def build_block_folds(rows):
    """按完整实验块留出，防止同批次正负样本跨折泄漏。"""
    grouped = defaultdict(list)
    for row in rows:
        block_id = experiment_block_key(row)
        if block_id:
            grouped[block_id].append(row)
    eligible = []
    for block_id, block_rows in sorted(grouped.items()):
        class_labels = {row.get("classLabel") for row in block_rows}
        if class_labels != set(contract.CLASS_LABELS):
            continue
        eligible.append((block_id, block_rows))
    if len(eligible) < 2:
        raise RuntimeError("insufficient_complete_experiment_blocks")
    folds = []
    for block_id, valid_rows in eligible:
        train_rows = [
            row for row in rows
            if experiment_block_key(row) != block_id
        ]
        fold = make_fold(
            "block",
            "block:" + block_id,
            train_rows,
            valid_rows,
        )
        fold["calibrationGroupField"] = "_experimentBlock"
        fold["experimentBlock"] = experiment_block_metadata(valid_rows)
        folds.append(fold)
    return folds


def block_session_partition(rows, valid_block_count):
    """在完整实验块内隔离训练、校准和验证 session。"""
    grouped = defaultdict(list)
    for row in rows:
        block_id = experiment_block_key(row)
        if not block_id:
            continue
        grouped[block_id].append(row)
    eligible_blocks = [
        block_id for block_id, block_rows in grouped.items()
        if {row.get("classLabel") for row in block_rows}
        == set(contract.CLASS_LABELS)
    ]
    if len(eligible_blocks) < 2:
        raise RuntimeError("insufficient_complete_experiment_blocks")
    folds = []
    for valid_index in range(valid_block_count):
        valid_blocks = {
            block_id for index, block_id in enumerate(eligible_blocks)
            if index % valid_block_count == valid_index
        }
        valid_rows = [
            row for row in rows
            if experiment_block_key(row) in valid_blocks
        ]
        remaining_blocks = [
            block_id for block_id in eligible_blocks
            if block_id not in valid_blocks
        ]
        calibration_blocks = {
            block_id for index, block_id in enumerate(remaining_blocks)
            if index % max(1, len(remaining_blocks) // 3) == 0
        }
        if not calibration_blocks:
            calibration_blocks = {remaining_blocks[0]}
        train_blocks = set(remaining_blocks) - calibration_blocks
        train_rows = [
            row for row in rows
            if experiment_block_key(row) in train_blocks
        ]
        calibration_rows = [
            row for row in rows
            if experiment_block_key(row) in calibration_blocks
        ]
        fold = make_fold(
            "block",
            "block:test:{}".format(valid_index + 1),
            train_rows + calibration_rows,
            valid_rows,
        )
        fold["fixedTrainRows"] = train_rows
        fold["fixedCalibrationRows"] = calibration_rows
        fold["experimentBlock"] = experiment_block_metadata(valid_rows)
        fold["blockPartitionAudit"] = {
            "trainBlocks": sorted(train_blocks),
            "calibrationBlocks": sorted(calibration_blocks),
            "validBlocks": sorted(valid_blocks),
            "trainSessionCount": len({
                row.get("groupId") for row in train_rows
            }),
            "calibrationSessionCount": len({
                row.get("groupId") for row in calibration_rows
            }),
            "validSessionCount": len({
                row.get("groupId") for row in valid_rows
            }),
        }
        folds.append(fold)
    return folds


def experiment_block_metadata(rows):
    """汇总实验块的 session、样本和类别覆盖。"""
    return {
        "blockId": sorted({
            experiment_block_key(row) for row in rows
        }),
        "sessionCount": len({
            str(row.get("groupId") or "") for row in rows
        }),
        "rowCount": len(rows),
        "classCounts": dict(sorted(Counter(
            str(row.get("classLabel") or "") for row in rows
        ).items())),
        "postures": sorted({
            posture_domain(row) for row in rows
        }),
        "collectors": sorted({
            str(row.get("collectorUserName") or "unknown") for row in rows
        }),
    }


def experiment_block_diagnostics(rows):
    """报告实验块隔离状态，便于识别 session 级泄漏。"""
    blocks = defaultdict(list)
    for row in rows:
        block_id = experiment_block_key(row)
        if block_id:
            blocks[block_id].append(row)
    complete_blocks = {
        block_id
        for block_id, block_rows in blocks.items()
        if {row.get("classLabel") for row in block_rows}
        == set(contract.CLASS_LABELS)
    }
    return {
        "blockCount": len(blocks),
        "completeBlockCount": len(complete_blocks),
        "incompleteBlockCount": len(blocks) - len(complete_blocks),
        "blocks": {
            block_id: experiment_block_metadata(block_rows)
            for block_id, block_rows in sorted(blocks.items())
        },
    }


def _normalized_mutual_information(pairs):
    """Compute session-label dependence without external dependencies."""
    count = len(pairs)
    if count == 0:
        return 0.0
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    joint_counts = Counter(pairs)

    def entropy(counts):
        return -sum(
            (value / count) * math.log(value / count)
            for value in counts.values()
            if value > 0
        )

    left_entropy = entropy(left_counts)
    right_entropy = entropy(right_counts)
    denominator = math.sqrt(left_entropy * right_entropy)
    if denominator <= 1e-12:
        return 0.0
    mutual_information = 0.0
    for (left, right), value in joint_counts.items():
        probability = value / count
        mutual_information += probability * math.log(
            probability
            / (
                (left_counts[left] / count)
                * (right_counts[right] / count)
            )
        )
    return float(np.clip(mutual_information / denominator, 0.0, 1.0))


def _paired_group_diagnostics(rows, key_function):
    """Summarize groups that contain both light and heavy samples."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[key_function(row)].append(row)
    paired = []
    for key, group_rows in grouped.items():
        counts = Counter(
            contract.class_strength_target(row["classLabel"])
            for row in group_rows
        )
        if counts[0] >= MIN_GROUP_NEGATIVES and counts[1] >= MIN_GROUP_POSITIVES:
            paired.append({
                "key": key,
                "rowCount": len(group_rows),
                "lightCount": counts[0],
                "heavyCount": counts[1],
            })
    return {
        "groupCount": len(grouped),
        "pairedGroupCount": len(paired),
        "pairedGroups": sorted(
            paired,
            key=lambda item: (-item["rowCount"], item["key"]),
        ),
    }


def data_contract_diagnostics(rows):
    """Audit whether the dataset can support cross-context generalization."""
    sessions = defaultdict(list)
    for row in rows:
        sessions[str(row.get("groupId") or "unknown")].append(row)
    session_labels = []
    multi_class_sessions = 0
    multi_strength_sessions = 0
    interleaved_sessions = 0
    interleaved_rows = 0
    for session_id, session_rows in sessions.items():
        class_labels = {
            str(row.get("classLabel") or "") for row in session_rows
        }
        strength_labels = {
            contract.class_strength_target(row["classLabel"])
            for row in session_rows
        }
        if len(class_labels) > 1:
            multi_class_sessions += 1
        if len(strength_labels) > 1:
            multi_strength_sessions += 1
        protocols = {
            str(row.get("collectionProtocol") or "") for row in session_rows
        }
        if "interleaved_v2" in protocols:
            interleaved_sessions += 1
            interleaved_rows += len(session_rows)
        for row in session_rows:
            session_labels.append((
                session_id,
                contract.class_strength_target(row["classLabel"]),
            ))
    dimensions = {
        "device": lambda row: physical_device_key(row),
        "posture": lambda row: posture_domain(row),
        "motion": lambda row: str(
            row.get("motionState") or "unknown"
        ).strip().lower(),
        "device_posture_motion": lambda row: "|".join((
            physical_device_key(row),
            posture_domain(row),
            str(row.get("motionState") or "unknown").strip().lower(),
        )),
    }
    paired_coverage = {
        name: _paired_group_diagnostics(rows, key_function)
        for name, key_function in dimensions.items()
    }
    motion_states = {
        str(row.get("motionState") or "").strip().lower()
        for row in rows
    }
    postures = {posture_domain(row) for row in rows}
    devices = {physical_device_key(row) for row in rows}
    session_dependence = _normalized_mutual_information(session_labels)
    blockers = []
    if len(devices) < DEPLOYMENT_MIN_DEVICES:
        blockers.append("insufficient_physical_devices")
    if len(postures) < DEPLOYMENT_MIN_POSTURES:
        blockers.append("insufficient_postures")
    if len(motion_states) < DEPLOYMENT_MIN_MOTION_STATES:
        blockers.append("insufficient_motion_states")
    if interleaved_sessions == 0:
        blockers.append("no_interleaved_sessions")
    if multi_strength_sessions == 0:
        blockers.append("session_label_confounding")
    if paired_coverage["device_posture_motion"]["pairedGroupCount"] < 2:
        blockers.append("insufficient_paired_context_groups")
    return {
        "rowCount": len(rows),
        "sessionCount": len(sessions),
        "physicalDeviceCount": len(devices),
        "postures": sorted(postures),
        "motionStates": sorted(motion_states),
        "multiClassSessionCount": multi_class_sessions,
        "multiStrengthSessionCount": multi_strength_sessions,
        "multiStrengthSessionFraction": (
            multi_strength_sessions / len(sessions)
            if sessions
            else 0.0
        ),
        "interleavedSessionCount": interleaved_sessions,
        "interleavedRowFraction": (
            interleaved_rows / len(rows)
            if rows
            else 0.0
        ),
        "sessionStrengthNormalizedMutualInformation": (
            session_dependence
        ),
        "pairedCoverage": paired_coverage,
        "proofReady": not blockers,
        "blockers": blockers,
    }


def build_external_device_folds(training_rows, evaluation_rows):
    """Hold out every external device while never fitting on external labels."""
    external_by_device = defaultdict(list)
    for row in evaluation_rows:
        external_by_device[physical_device_key(row)].append(row)
    folds = []
    for device in sorted(external_by_device):
        valid_rows = external_by_device[device]
        train_rows = [
            row for row in training_rows
            if physical_device_key(row) != device
        ]
        try:
            fold = make_fold(
                "device",
                "device:external:" + device,
                train_rows,
                valid_rows,
            )
        except RuntimeError:
            continue
        # 中文注释：外测设备折同样按实验块隔离校准上下文。
        fold["calibrationGroupField"] = "_experimentBlock"
        fold["evaluationOnlyRows"] = valid_rows
        folds.append(fold)
    if not folds:
        raise RuntimeError("no_eligible_external_device_fold")
    return folds

def select_ordinary_click_rows_for_fold(rows, fold):
    """Match the audit set to the device held out by the current fold."""
    target_devices = {
        physical_device_key(row) for row in fold["valid"]
    }
    return [
        row for row in rows
        if physical_device_key(row) in target_devices
    ]


def read_valid_hashes(path):
    """Read validation identities from a generated fold CSV."""
    grouped = defaultdict(set)
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row.get("split") == "valid":
                grouped[int(row["fold"])].add(row["sourceSha256"])
    return grouped


def build_csv_folds(rows, filename, kind):
    """Build folds separately inside every source dataset."""
    folds = []
    rows_by_dataset = defaultdict(list)
    for row in rows:
        rows_by_dataset[row["_datasetId"]].append(row)
    dataset_dirs = {
        row["_datasetId"]: Path(row["_datasetDir"])
        for row in rows
    }
    for identifier, dataset_rows in sorted(rows_by_dataset.items()):
        path = dataset_dirs[identifier] / filename
        if not path.is_file():
            continue
        valid_hashes = read_valid_hashes(path)
        all_hashes = {row["sourceSha256"] for row in dataset_rows}
        for fold_index, selected_hashes in sorted(valid_hashes.items()):
            unknown = selected_hashes - all_hashes
            if unknown:
                raise RuntimeError("fold_references_unknown_training_row")
            valid_rows = [
                row for row in dataset_rows
                if row["sourceSha256"] in selected_hashes
            ]
            train_rows = [
                row for row in dataset_rows
                if row["sourceSha256"] not in selected_hashes
            ]
            try:
                folds.append(make_fold(
                    kind,
                    kind + ":" + identifier + ":" + str(fold_index),
                    train_rows,
                    valid_rows,
                ))
            except RuntimeError:
                continue
    return folds


def build_folds(rows, requested_kinds):
    """构造设备、session、姿态和完整实验块验证折。"""
    folds = []
    if "device" in requested_kinds:
        folds.extend(build_device_folds(rows))
    if "block" in requested_kinds:
        folds.extend(block_session_partition(rows, 3))
    if "session" in requested_kinds:
        folds.extend(build_csv_folds(
            rows,
            "strict_group_folds.csv",
            "session",
        ))
    if "posture" in requested_kinds:
        folds.extend(build_csv_folds(
            rows,
            "pose_holdout_folds.csv",
            "posture",
        ))
    if not folds:
        raise RuntimeError("no_eligible_validation_fold")
    return folds


def calibration_split(
        rows,
        seed,
        ratio=0.22,
        min_scenario_gate_sessions=0,
        group_field="groupId",
):
    """Reserve whole sessions with motion, gate and optional quota coverage."""
    if not 0.0 < ratio < 0.5:
        raise RuntimeError("invalid_calibration_ratio")
    if min_scenario_gate_sessions < 0:
        raise RuntimeError("invalid_calibration_session_quota")
    grouped = defaultdict(list)
    classes_by_group = defaultdict(set)
    cells_by_group = defaultdict(set)
    groups_by_class = defaultdict(set)
    scenario_gates_by_group = defaultdict(set)
    groups_by_scenario_gate = defaultdict(set)
    for row in rows:
        group_id = str(row.get(group_field) or "")
        if not group_id:
            raise RuntimeError("missing_calibration_group")
        class_label = row["classLabel"]
        scenario_gate = (
            scenario_domain(row),
            contract.class_strength_target(class_label),
        )
        grouped[group_id].append(row)
        classes_by_group[group_id].add(class_label)
        groups_by_class[class_label].add(group_id)
        scenario_gates_by_group[group_id].add(scenario_gate)
        groups_by_scenario_gate[scenario_gate].add(group_id)
        cells_by_group[group_id].add((
            scenario_gate[0],
            contract.class_strength_target(class_label),
        ))
    if not grouped:
        raise RuntimeError("empty_calibration_rows")

    rng = random.Random(seed)
    calibration_groups = set()
    selected_class_counts = Counter()

    def can_select(group_id):
        # 中文注释：校准集与拟合集都必须保留四个诊断类别。
        # 不能为了覆盖一个运动层而抽空另一侧的稀缺类别。
        return all(
            selected_class_counts[class_label]
            < len(groups_by_class[class_label]) - 1
            for class_label in classes_by_group[group_id]
        )

    def select(group_id):
        calibration_groups.add(group_id)
        for class_label in classes_by_group[group_id]:
            selected_class_counts[class_label] += 1

    def selected_scenario_gate_count(scenario_gate):
        return sum(
            scenario_gate in scenario_gates_by_group[group_id]
            for group_id in calibration_groups
        )

    for class_label in contract.CLASS_LABELS:
        if selected_class_counts[class_label] > 0:
            continue
        candidates = [
            group_id
            for group_id in sorted(groups_by_class[class_label])
            if group_id not in calibration_groups and can_select(group_id)
        ]
        if not candidates:
            raise RuntimeError(
                "calibration_class_uncovered:" + class_label
            )
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda group_id: -len(classes_by_group[group_id])
        )
        select(candidates[0])

    available_cells = sorted({
        cell
        for cells in cells_by_group.values()
        for cell in cells
    })
    for cell in available_cells:
        # 中文注释：类别覆盖阶段可能已经选中当前组合，不能重复
        # 占用另一个完整实验块，否则稀缺场景会耗尽可训练组。
        if selected_scenario_gate_count(cell) > 0:
            continue
        candidates = [
            group_id
            for group_id in sorted(grouped)
            if group_id not in calibration_groups
            and cell in cells_by_group[group_id]
            and can_select(group_id)
        ]
        if not candidates:
            raise RuntimeError(
                "calibration_stratum_uncovered:"
                + cell[0]
                + ":"
                + str(cell[1])
            )
        rng.shuffle(candidates)
        # 中文注释：同一层有多个 session 时，优先保留包含稀缺类别的 session。
        candidates.sort(key=lambda group_id: sum(
            1.0 / max(1, len(groups_by_class[class_label]) - 1)
            for class_label in classes_by_group[group_id]
        ))
        select(candidates[0])

    if min_scenario_gate_sessions:
        # 中文注释：同一场景的强度类别可能被 session 绑定。仅覆盖场景
        # 不足以形成稳定的负样本阈值，必须为场景与强度组合保留独立
        # session 配额。优先选择样本较多的 session，降低稀有场景阈值
        # 因少量触摸而剧烈波动。
        for scenario_gate in sorted(groups_by_scenario_gate):
            available_groups = groups_by_scenario_gate[scenario_gate]
            quota = min(
                min_scenario_gate_sessions,
                max(0, len(available_groups) - 1),
            )
            while (
                selected_scenario_gate_count(scenario_gate)
                < quota
            ):
                candidates = [
                    group_id
                    for group_id in sorted(available_groups)
                    if group_id not in calibration_groups
                    and can_select(group_id)
                ]
                if not candidates:
                    raise RuntimeError(
                        "calibration_scenario_gate_quota_uncovered:"
                        + scenario_gate[0]
                        + ":"
                        + str(scenario_gate[1])
                    )
                rng.shuffle(candidates)
                candidates.sort(
                    key=lambda group_id: -len(grouped[group_id])
                )
                select(candidates[0])

    target = max(
        sum(len(grouped[group_id]) for group_id in calibration_groups),
        int(round(len(rows) * ratio)),
    )
    selected_count = sum(
        len(grouped[group_id]) for group_id in calibration_groups
    )
    remaining = [
        group_id
        for group_id in grouped
        if group_id not in calibration_groups
    ]
    rng.shuffle(remaining)
    for group_id in remaining:
        if selected_count >= target:
            break
        if not can_select(group_id):
            continue
        select(group_id)
        selected_count += len(grouped[group_id])

    fit_rows = [
        row for row in rows
        if str(row.get(group_field) or "") not in calibration_groups
    ]
    calibration_rows = [
        row for row in rows
        if str(row.get(group_field) or "") in calibration_groups
    ]
    ensure_fold_coverage(fit_rows, calibration_rows)
    return fit_rows, calibration_rows


def select_target_calibration_sessions(
        rows,
        gate_labels,
        count,
        seed,
):
    """Select complete target sessions for independent threshold calibration."""
    if count <= 0:
        raise RuntimeError("invalid_target_calibration_count")
    negatives_by_group = defaultdict(list)
    for index, (row, label) in enumerate(zip(rows, gate_labels)):
        group_id = str(row.get("groupId") or "").strip()
        if not group_id:
            raise RuntimeError("missing_target_session_id")
        if int(label) == contract.GATE_LABELS.index("light"):
            negatives_by_group[group_id].append(index)
    if not negatives_by_group:
        raise RuntimeError("missing_target_calibration_negative_proxy")
    if len(negatives_by_group) < 2:
        raise RuntimeError("insufficient_target_calibration_sessions")
    group_ids = sorted(negatives_by_group)
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    # 中文注释：按 session 而不是单条负样本切分。先随机打乱，
    # 再优先选择负样本较少的完整 session，尽量给留出集留下大 session。
    group_ids.sort(
        key=lambda group_id: len(negatives_by_group[group_id])
    )
    selected = []
    selected_negatives = 0
    # 中文注释：至少保留一个完整 session 给评估集，
    # 且所有 session 的负样本作为一个不可拆分单位进入同一侧。
    for group_id in group_ids:
        if len(selected) >= len(group_ids) - 1:
            break
        group_size = len(negatives_by_group[group_id])
        if selected and selected_negatives + group_size > count:
            continue
        selected.append(group_id)
        selected_negatives += group_size
        if selected_negatives >= count:
            break
    if not selected:
        selected.append(group_ids[0])
    return selected


def target_calibrated_reports(
        rows,
        gate_labels,
        heavy_probabilities,
        target_calibration_count,
        seed,
        repetitions,
):
    """Fit a target threshold on complete sessions and evaluate other sessions."""
    if repetitions <= 0:
        raise RuntimeError("invalid_target_calibration_repetitions")
    labels = np.asarray(gate_labels, dtype=np.int64)
    scores = np.asarray(heavy_probabilities, dtype=np.float64)
    negatives = np.flatnonzero(labels == 0)
    if not len(negatives):
        raise RuntimeError("missing_held_out_target_negative_proxy")
    # 中文注释：目标设备校准必须按完整 session 隔离；
    # 小设备没有足够的独立 session 时，报告只能作为描述性诊断。
    effective_count = min(
        target_calibration_count,
        max(1, len(negatives) // 2),
    )
    selection_seeds = [
        seed + index * 1009
        for index in range(repetitions)
    ]
    results = {}
    for target_fpr in TARGET_FPRS:
        minimum_proof_negative_count = int(
            np.ceil(5.0 / float(target_fpr))
        )
        target_reports = []
        for selection_seed in selection_seeds:
            calibration_sessions = select_target_calibration_sessions(
                rows,
                gate_labels,
                effective_count,
                selection_seed,
            )
            calibration_set = set(calibration_sessions)
            calibration_indices = [
                index for index, (row, label) in enumerate(
                    zip(rows, labels)
                )
                if (
                    str(row.get("groupId") or "").strip()
                    in calibration_set
                    and int(label) == 0
                )
            ]
            if not calibration_indices:
                raise RuntimeError("empty_target_calibration_set")
            evaluation_indices = [
                index for index in range(len(rows))
                if (
                    str(rows[index].get("groupId") or "").strip()
                    not in calibration_set
                )
            ]
            if not np.any(labels[evaluation_indices] == 0):
                raise RuntimeError(
                    "missing_held_out_target_negative_proxy"
                )
            calibration_negatives = scores[calibration_indices]
            threshold = gate_threshold_metrics.select_threshold_at_fpr(
                calibration_negatives,
                target_fpr,
            )
            evaluation_rows = [
                rows[index] for index in evaluation_indices
            ]
            evaluation_labels = labels[evaluation_indices]
            evaluation_scores = scores[evaluation_indices]
            evaluation_sessions = sorted({
                str(row.get("groupId") or "").strip()
                for row in evaluation_rows
            })
            held_out_negative_sessions = sorted({
                str(row.get("groupId") or "").strip()
                for row, label in zip(evaluation_rows, evaluation_labels)
                if int(label) == 0
            })
            report = gate_threshold_metrics.fixed_threshold_report(
                evaluation_rows,
                evaluation_labels,
                evaluation_scores,
                threshold,
            )
            report["thresholdPolicy"] = (
                "target_known_negative_calibration_with_held_out_evaluation"
            )
            report["targetCalibrationNegativeCount"] = len(
                calibration_indices
            )
            report["targetCalibrationRequestedNegativeCount"] = (
                target_calibration_count
            )
            report["calibrationSessionCount"] = len(calibration_sessions)
            report["heldOutSessionCount"] = len(evaluation_sessions)
            report["heldOutNegativeSessionCount"] = len(
                held_out_negative_sessions
            )
            report["calibrationSessions"] = sorted(calibration_sessions)
            report["heldOutSessions"] = evaluation_sessions
            report["sessionDisjoint"] = not bool(
                set(calibration_sessions) & set(evaluation_sessions)
            )
            calibration_rows = [
                row for row in rows
                if str(row.get("groupId") or "").strip()
                in calibration_set
            ]
            report["calibrationRowCount"] = len(calibration_rows)
            report["heldOutRowCount"] = len(evaluation_rows)
            report["calibrationDeviceNames"] = sorted({
                str(row.get("deviceName") or "")
                for row in calibration_rows
            })
            report["heldOutDeviceNames"] = sorted({
                str(row.get("deviceName") or "")
                for row in evaluation_rows
            })
            report["calibrationBodyPostures"] = sorted({
                str(row.get("bodyPosture") or "")
                for row in calibration_rows
            })
            report["heldOutBodyPostures"] = sorted({
                str(row.get("bodyPosture") or "")
                for row in evaluation_rows
            })
            report["calibrationMotionStates"] = sorted({
                str(row.get("motionState") or "")
                for row in calibration_rows
            })
            report["heldOutMotionStates"] = sorted({
                str(row.get("motionState") or "")
                for row in evaluation_rows
            })
            report["heldOutNegativeCount"] = int(
                np.count_nonzero(evaluation_labels == 0)
            )
            report["heldOutPositiveCount"] = int(
                np.count_nonzero(evaluation_labels == 1)
            )
            report["calibrationSelectionSeed"] = selection_seed
            report["minimumProofNegativeCount"] = (
                minimum_proof_negative_count
            )
            report["minimumProofPositiveCount"] = MIN_GROUP_POSITIVES
            report["minimumProofSessionCount"] = MIN_PROOF_SESSION_COUNT
            report["calibrationEvidenceStatus"] = (
                "proof_ready"
                if (
                    report["targetCalibrationNegativeCount"]
                    >= minimum_proof_negative_count
                    and report["heldOutNegativeCount"]
                    >= minimum_proof_negative_count
                    and report["heldOutPositiveCount"]
                    >= MIN_GROUP_POSITIVES
                    and report["calibrationSessionCount"]
                    >= MIN_PROOF_SESSION_COUNT
                    and report["heldOutNegativeSessionCount"]
                    >= MIN_PROOF_SESSION_COUNT
                    and report["sessionDisjoint"]
                )
                else "descriptive_only"
            )
            report["proofReady"] = (
                report["calibrationEvidenceStatus"] == "proof_ready"
            )
            target_reports.append(report)
        results[str(target_fpr)] = {
            "repetitions": repetitions,
            "minimumProofNegativeCount": minimum_proof_negative_count,
            "proofReady": all(
                report["proofReady"]
                for report in target_reports
            ),
            "targetCalibrationNegativeCount": (
                target_reports[0]["targetCalibrationNegativeCount"]
            ),
            "overall": {
                "heavyRecall": float(np.mean([
                    report["overall"]["heavyRecall"]
                    for report in target_reports
                ])),
                "lightFalsePositiveRate": float(np.mean([
                    report["overall"]["lightFalsePositiveRate"]
                    for report in target_reports
                ])),
            },
            "minimumHeavyRecall": float(min(
                report["overall"]["heavyRecall"]
                for report in target_reports
            )),
            "maximumHeavyRecall": float(max(
                report["overall"]["heavyRecall"]
                for report in target_reports
            )),
            "perRepetition": target_reports,
        }
    return results


def build_evaluation_only_rows(rows):
    """Use external samples only for metrics, never for fitting."""
    return [
        {
            **row,
            "_evaluationOnly": True,
        }
        for row in rows
    ]


def prepare_evaluation_values(
        fit_rows,
        evaluation_rows,
        feature_mode="full",
        imu_mode="raw",
        adaptive_feature_mode="none",
        feature_family="stats",
        evidence_contract=evidence_features.DEFAULT_EVIDENCE_CONTRACT,
        normalization_mode="none",
        early_evidence_mode="none",
        return_metadata=False,
):
    """Transform evaluation samples without reading external labels."""
    if not fit_rows:
        raise RuntimeError("empty_evaluation_reference_split")
    if not evaluation_rows:
        raise RuntimeError("empty_evaluation_only_dataset")
    fit_rows = [
        {
            **row,
            "matrix": np.asarray(row["matrix"], dtype=np.float32),
        }
        for row in fit_rows
    ]
    evaluation_rows = [
        {
            **row,
            "matrix": np.asarray(row["matrix"], dtype=np.float32),
        }
        for row in evaluation_rows
    ]
    indices = lightweight.PHYSICAL_INDICES
    fit_raw = lightweight.select_matrix(fit_rows, indices)
    evaluation_raw = lightweight.select_matrix(evaluation_rows, indices)
    fit_raw = apply_observation_mask(fit_raw, fit_rows)
    evaluation_raw = apply_observation_mask(
        evaluation_raw,
        evaluation_rows,
    )
    normalization_metadata = {
        "mode": normalization_mode,
        "targetCalibration": "none",
    }
    if feature_mode in (
        "physical_magnitude",
        "physical_magnitude_temporal",
        "physical_magnitude_shape",
        "physical_magnitude_temporal_shape",
    ):
        temporal = feature_mode in (
            "physical_magnitude_temporal",
            "physical_magnitude_temporal_shape",
        )
        shape = feature_mode in (
            "physical_magnitude_shape",
            "physical_magnitude_temporal_shape",
        )
        fit_raw = apply_physical_magnitude(fit_raw, temporal)
        evaluation_raw = apply_physical_magnitude(
            evaluation_raw,
            temporal,
        )
        if shape:
            # 中文注释：接触形态只在设备提供真实值时保留，
            # 缺失通道由可用性掩码清零，不能伪造统一量纲。
            fit_raw[:, :, 8:10] = np.asarray(
                lightweight.select_matrix(fit_rows, indices)
            )[:, :, 8:10]
            evaluation_raw[:, :, 8:10] = np.asarray(
                lightweight.select_matrix(evaluation_rows, indices)
            )[:, :, 8:10]
    if feature_mode == "device_relative_magnitude":
        fit_raw = apply_device_relative_magnitude(fit_raw, fit_rows)
        evaluation_raw = apply_device_relative_magnitude(
            evaluation_raw,
            evaluation_rows,
        )
    if feature_mode in (
        "contact_relative_magnitude",
        "contact_relative_magnitude_shape",
    ):
        fit_raw = apply_contact_relative_size(fit_raw, fit_rows)
        evaluation_raw = apply_contact_relative_size(
            evaluation_raw,
            evaluation_rows,
        )
    if feature_mode == "contact_relative_magnitude_shape":
        fit_raw = apply_contact_relative_shape(fit_raw, fit_rows)
        evaluation_raw = apply_contact_relative_shape(
            evaluation_raw,
            evaluation_rows,
        )
    if feature_mode == "magnitude_contact_relative":
        fit_raw = apply_contact_relative_magnitude(fit_raw, fit_rows)
        evaluation_raw = apply_contact_relative_magnitude(
            evaluation_raw,
            evaluation_rows,
        )
    historical_metadata = {}
    if feature_mode in HISTORICAL_CONTACT_MODES:
        fit_raw, fit_history = apply_historical_contact_relative(
            fit_raw,
            fit_rows,
        )
        evaluation_raw, evaluation_history = (
            apply_historical_contact_relative(
                evaluation_raw,
                evaluation_rows,
            )
        )
        historical_metadata = {
            "fit": historical_contact_metadata(fit_history),
            "evaluation": historical_contact_metadata(
                evaluation_history
            ),
        }
    device_causal_metadata = {}
    if feature_mode in DEVICE_CAUSAL_CONTACT_MODES:
        fit_raw, fit_device = apply_device_causal_contact(
            fit_raw,
            fit_rows,
        )
        evaluation_raw, evaluation_device = apply_device_causal_contact(
            evaluation_raw,
            evaluation_rows,
        )
        device_causal_metadata = {
            "fit": device_causal_contact_metadata(fit_device),
            "evaluation": device_causal_contact_metadata(
                evaluation_device
            ),
        }
    if uses_gravity_alignment(imu_mode):
        fit_raw = apply_gravity_alignment(fit_raw, fit_rows)
        evaluation_raw = apply_gravity_alignment(
            evaluation_raw,
            evaluation_rows,
        )
    if normalization_mode != "none":
        if feature_family != "stats":
            raise RuntimeError("normalization_requires_stats_family")
        source_reference = fit_raw.copy()
        source_profile = device_normalization.fit_profile(
            source_reference
        )
        fit_raw = device_normalization.apply_profile(
            source_reference,
            source_profile,
        )
        evaluation_raw, normalization_metadata = (
            device_normalization.normalize_values(
                normalization_mode,
                source_reference,
                evaluation_raw,
                evaluation_rows,
            )
        )
        normalization_metadata["sourceProfileSampleCount"] = int(
            source_profile["sampleCount"]
        )
    if historical_metadata:
        normalization_metadata["historicalContact"] = historical_metadata
    if device_causal_metadata:
        normalization_metadata["deviceCausalContact"] = (
            device_causal_metadata
        )
    if feature_family == "evidence":
        if imu_mode != "raw" or adaptive_feature_mode != "none":
            raise RuntimeError("unsupported_evidence_preprocessing")
        fit_stats = evidence_features.extract(
            fit_raw,
            fit_rows,
            evidence_contract,
        )
        evaluation_stats = evidence_features.extract(
            evaluation_raw,
            evaluation_rows,
            evidence_contract,
        )
        fit_stats = evidence_features.apply_feature_mode(
            fit_stats,
            feature_mode,
        )
        evaluation_stats = evidence_features.apply_feature_mode(
            evaluation_stats,
            feature_mode,
        )
    elif feature_family == "stats":
        center, scale = lightweight.fit_scaler(fit_raw)
        fit_normalized = lightweight.transform(fit_raw, center, scale)
        evaluation_normalized = lightweight.transform(
            evaluation_raw,
            center,
            scale,
        )
        fit_normalized = apply_imu_mode(fit_normalized, imu_mode)
        evaluation_normalized = apply_imu_mode(
            evaluation_normalized,
            imu_mode,
        )
        fit_normalized = apply_availability_mask(fit_normalized, fit_rows)
        evaluation_normalized = apply_availability_mask(
            evaluation_normalized,
            evaluation_rows,
        )
        fit_normalized = apply_feature_mode(fit_normalized, feature_mode)
        evaluation_normalized = apply_feature_mode(
            evaluation_normalized,
            feature_mode,
        )
        fit_normalized = apply_adaptive_features(
            fit_normalized,
            adaptive_feature_mode,
        )
        evaluation_normalized = apply_adaptive_features(
            evaluation_normalized,
            adaptive_feature_mode,
        )
        fit_stats = lightweight.stats_features(fit_normalized)
        evaluation_stats = lightweight.stats_features(
            evaluation_normalized
        )
        if feature_mode in (
            "touch_local_shape",
            "touch_local_motion_shape",
            "historical_contact_local_shape",
            "historical_contact_motion_shape",
        ):
            fit_stats = append_touch_local_shape_features(
                fit_stats,
                fit_rows,
            )
            evaluation_stats = append_touch_local_shape_features(
                evaluation_stats,
                evaluation_rows,
            )
        if feature_mode in CAUSAL_MOTION_MODES:
            include_shape = feature_mode in CAUSAL_MOTION_SHAPE_MODES
            fit_stats = append_causal_motion_features(
                fit_stats,
                fit_rows,
                include_shape=include_shape,
            )
            evaluation_stats = append_causal_motion_features(
                evaluation_stats,
                evaluation_rows,
                include_shape=include_shape,
            )
    else:
        raise RuntimeError("unsupported_feature_family")
    if feature_family == "evidence":
        result = evidence_features.normalize(fit_stats, evaluation_stats)
    else:
        stats_center = fit_stats.mean(axis=0)
        stats_scale = fit_stats.std(axis=0)
        stats_scale[stats_scale < 1e-6] = 1.0
        result = np.clip(
            (evaluation_stats - stats_center) / stats_scale,
            contract.STATS_CLIP_MIN,
            contract.STATS_CLIP_MAX,
        ).astype(np.float32)
    result, early_metadata = append_early_evidence(
        fit_rows,
        evaluation_rows,
        result,
        early_evidence_mode,
    )
    normalization_metadata["earlyPressEvidence"] = early_metadata
    if return_metadata:
        return result, normalization_metadata
    return result


def apply_evaluation_only_protocol(
        report,
        rows,
        model_runs,
):
    """Add an explicit external evaluation label to a validation report."""
    report["evaluationOnlyDataset"] = True
    report["evaluationOnlySampleCount"] = len(rows)
    report["evaluationOnlyDevices"] = sorted({
        str(row.get("deviceName") or "") for row in rows
    })
    report["evaluationOnlyMetrics"] = {
        "fixedFprModels": report.get("models", {}),
        "targetCalibratedModels": report.get(
            "targetCalibratedModels",
            {},
        ),
    }
    report["models"] = {}
    report["targetCalibratedModels"] = {}
    report["runs"] = {
        name: [
            {
                **run,
                "evaluationOnly": True,
            }
            for run in values
        ]
        for name, values in model_runs.items()
    }
    return report


def prepare_values(
        fit_rows,
        calibration_rows,
        valid_rows,
        feature_mode="full",
        modality_dropout=0.0,
        augmentation_mode="none",
        augmentation_probability=0.5,
        augmentation_max_log_scale=0.35,
        seed=0,
        imu_mode="raw",
        adaptive_feature_mode="none",
        feature_family="stats",
        evidence_contract=evidence_features.DEFAULT_EVIDENCE_CONTRACT,
        normalization_mode="none",
        early_evidence_mode="none",
        return_metadata=False,
):
    """Fit every scaler on the training split only."""
    if augmentation_mode not in AUGMENTATION_MODES:
        raise RuntimeError("unsupported_augmentation_mode")
    fit_rows = [
        {
            **row,
            "matrix": np.asarray(row["matrix"], dtype=np.float32),
        }
        for row in fit_rows
    ]
    calibration_rows = [
        {
            **row,
            "matrix": np.asarray(row["matrix"], dtype=np.float32),
        }
        for row in calibration_rows
    ]
    valid_rows = [
        {
            **row,
            "matrix": np.asarray(row["matrix"], dtype=np.float32),
        }
        for row in valid_rows
    ]
    indices = lightweight.PHYSICAL_INDICES
    fit_raw = lightweight.select_matrix(fit_rows, indices)
    calibration_raw = lightweight.select_matrix(
        calibration_rows,
        indices,
    )
    valid_raw = lightweight.select_matrix(valid_rows, indices)
    fit_raw = apply_observation_mask(fit_raw, fit_rows)
    calibration_raw = apply_observation_mask(
        calibration_raw,
        calibration_rows,
    )
    valid_raw = apply_observation_mask(valid_raw, valid_rows)
    if feature_mode in (
        "physical_magnitude",
        "physical_magnitude_temporal",
        "physical_magnitude_shape",
        "physical_magnitude_temporal_shape",
    ):
        temporal = feature_mode in (
            "physical_magnitude_temporal",
            "physical_magnitude_temporal_shape",
        )
        shape = feature_mode in (
            "physical_magnitude_shape",
            "physical_magnitude_temporal_shape",
        )
        fit_raw = apply_physical_magnitude(fit_raw, temporal)
        calibration_raw = apply_physical_magnitude(
            calibration_raw,
            temporal,
        )
        valid_raw = apply_physical_magnitude(valid_raw, temporal)
        if shape:
            fit_raw[:, :, 8:10] = np.asarray(
                lightweight.select_matrix(fit_rows, indices)
            )[:, :, 8:10]
            calibration_raw[:, :, 8:10] = np.asarray(
                lightweight.select_matrix(calibration_rows, indices)
            )[:, :, 8:10]
            valid_raw[:, :, 8:10] = np.asarray(
                lightweight.select_matrix(valid_rows, indices)
            )[:, :, 8:10]
    if feature_mode == "device_relative_magnitude":
        fit_raw = apply_device_relative_magnitude(fit_raw, fit_rows)
        calibration_raw = apply_device_relative_magnitude(
            calibration_raw,
            calibration_rows,
        )
        valid_raw = apply_device_relative_magnitude(valid_raw, valid_rows)
    if feature_mode in (
        "contact_relative_magnitude",
        "contact_relative_magnitude_shape",
    ):
        fit_raw = apply_contact_relative_size(fit_raw, fit_rows)
        calibration_raw = apply_contact_relative_size(
            calibration_raw,
            calibration_rows,
        )
        valid_raw = apply_contact_relative_size(
            valid_raw,
            valid_rows,
        )
    if feature_mode == "contact_relative_magnitude_shape":
        fit_raw = apply_contact_relative_shape(fit_raw, fit_rows)
        calibration_raw = apply_contact_relative_shape(
            calibration_raw,
            calibration_rows,
        )
        valid_raw = apply_contact_relative_shape(
            valid_raw,
            valid_rows,
        )
    if feature_mode == "magnitude_contact_relative":
        fit_raw = apply_contact_relative_magnitude(fit_raw, fit_rows)
        calibration_raw = apply_contact_relative_magnitude(
            calibration_raw,
            calibration_rows,
        )
        valid_raw = apply_contact_relative_magnitude(
            valid_raw,
            valid_rows,
        )
    historical_metadata = {}
    if feature_mode in HISTORICAL_CONTACT_MODES:
        fit_raw, fit_history = apply_historical_contact_relative(
            fit_raw,
            fit_rows,
        )
        calibration_raw, calibration_history = (
            apply_historical_contact_relative(
                calibration_raw,
                calibration_rows,
            )
        )
        valid_raw, valid_history = apply_historical_contact_relative(
            valid_raw,
            valid_rows,
        )
        historical_metadata = {
            "fit": historical_contact_metadata(fit_history),
            "calibration": historical_contact_metadata(
                calibration_history
            ),
            "validation": historical_contact_metadata(valid_history),
        }
    device_causal_metadata = {}
    if feature_mode in DEVICE_CAUSAL_CONTACT_MODES:
        fit_raw, fit_device = apply_device_causal_contact(
            fit_raw,
            fit_rows,
        )
        calibration_raw, calibration_device = (
            apply_device_causal_contact(
                calibration_raw,
                calibration_rows,
            )
        )
        valid_raw, valid_device = apply_device_causal_contact(
            valid_raw,
            valid_rows,
        )
        device_causal_metadata = {
            "fit": device_causal_contact_metadata(fit_device),
            "calibration": device_causal_contact_metadata(
                calibration_device
            ),
            "validation": device_causal_contact_metadata(valid_device),
        }
    if uses_gravity_alignment(imu_mode):
        fit_raw = apply_gravity_alignment(fit_raw, fit_rows)
        calibration_raw = apply_gravity_alignment(
            calibration_raw,
            calibration_rows,
        )
        valid_raw = apply_gravity_alignment(valid_raw, valid_rows)
    normalization_metadata = {
        "mode": normalization_mode,
        "targetCalibration": "none",
    }
    if historical_metadata:
        normalization_metadata["historicalContact"] = historical_metadata
    if device_causal_metadata:
        normalization_metadata["deviceCausalContact"] = (
            device_causal_metadata
        )
    if normalization_mode != "none":
        if feature_family != "stats":
            raise RuntimeError("normalization_requires_stats_family")
        source_reference = fit_raw.copy()
        source_profile = device_normalization.fit_profile(
            source_reference
        )
        fit_raw = device_normalization.apply_profile(
            source_reference,
            source_profile,
        )
        calibration_raw, calibration_metadata = (
            device_normalization.normalize_values(
                normalization_mode,
                source_reference,
                calibration_raw,
                calibration_rows,
            )
        )
        valid_raw, valid_metadata = (
            device_normalization.normalize_values(
                normalization_mode,
                source_reference,
                valid_raw,
                valid_rows,
            )
        )
        normalization_metadata = {
            "mode": normalization_mode,
            "sourceProfileSampleCount": int(
                source_profile["sampleCount"]
            ),
            "calibration": calibration_metadata,
            "validation": valid_metadata,
        }
    if feature_family == "evidence":
        if (
            imu_mode != "raw"
            or adaptive_feature_mode != "none"
            or modality_dropout > 0.0
            or augmentation_mode != "none"
        ):
            raise RuntimeError("unsupported_evidence_preprocessing")
        fit_stats = evidence_features.extract(
            fit_raw,
            fit_rows,
            evidence_contract,
        )
        calibration_stats = evidence_features.extract(
            calibration_raw,
            calibration_rows,
            evidence_contract,
        )
        valid_stats = evidence_features.extract(
            valid_raw,
            valid_rows,
            evidence_contract,
        )
        fit_stats = evidence_features.apply_feature_mode(
            fit_stats,
            feature_mode,
        )
        calibration_stats = evidence_features.apply_feature_mode(
            calibration_stats,
            feature_mode,
        )
        valid_stats = evidence_features.apply_feature_mode(
            valid_stats,
            feature_mode,
        )
    elif feature_family == "stats":
        center, scale = lightweight.fit_scaler(fit_raw)
        fit_normalized = lightweight.transform(fit_raw, center, scale)
        calibration_normalized = lightweight.transform(
            calibration_raw,
            center,
            scale,
        )
        valid_normalized = lightweight.transform(valid_raw, center, scale)
        clean_fit_normalized = (
            fit_normalized.copy()
            if augmentation_mode == "channel_scale"
            else None
        )
        if augmentation_mode == "channel_scale":
            # 中文注释：只在拟合集上随机改变传感器整体增益。
            # 校准集、验证集和外测集保留真实量纲，避免数据泄漏，
            # 也让阈值选择面对未经增强的分数分布。
            fit_normalized = apply_channel_scale_augmentation(
                fit_normalized,
                fit_rows,
                augmentation_probability,
                seed + 130000,
                augmentation_max_log_scale,
            )
        fit_normalized = apply_imu_mode(fit_normalized, imu_mode)
        calibration_normalized = apply_imu_mode(
            calibration_normalized,
            imu_mode,
        )
        valid_normalized = apply_imu_mode(valid_normalized, imu_mode)
        fit_normalized = apply_availability_mask(fit_normalized, fit_rows)
        calibration_normalized = apply_availability_mask(
            calibration_normalized,
            calibration_rows,
        )
        valid_normalized = apply_availability_mask(
            valid_normalized,
            valid_rows,
        )
        fit_normalized = apply_feature_mode(fit_normalized, feature_mode)
        calibration_normalized = apply_feature_mode(
            calibration_normalized,
            feature_mode,
        )
        valid_normalized = apply_feature_mode(
            valid_normalized,
            feature_mode,
        )
        fit_normalized = apply_adaptive_features(
            fit_normalized,
            adaptive_feature_mode,
        )
        calibration_normalized = apply_adaptive_features(
            calibration_normalized,
            adaptive_feature_mode,
        )
        valid_normalized = apply_adaptive_features(
            valid_normalized,
            adaptive_feature_mode,
        )
        fit_normalized = apply_modality_dropout(
            fit_normalized,
            fit_rows,
            modality_dropout,
            seed,
        )
        if clean_fit_normalized is not None:
            # 中文注释：增强只改变拟合样本，特征统计尺度仍从未增强的
            # 训练集估计。否则校准集虽然没有被缩放，其分数分布也会随
            # 增强参数漂移，等于间接改变误触阈值。
            clean_fit_normalized = apply_imu_mode(
                clean_fit_normalized,
                imu_mode,
            )
            clean_fit_normalized = apply_availability_mask(
                clean_fit_normalized,
                fit_rows,
            )
            clean_fit_normalized = apply_feature_mode(
                clean_fit_normalized,
                feature_mode,
            )
            clean_fit_normalized = apply_adaptive_features(
                clean_fit_normalized,
                adaptive_feature_mode,
            )
        fit_stats = lightweight.stats_features(fit_normalized)
        calibration_stats = lightweight.stats_features(
            calibration_normalized
        )
        valid_stats = lightweight.stats_features(valid_normalized)
        if feature_mode in (
            "touch_local_shape",
            "touch_local_motion_shape",
            "historical_contact_local_shape",
            "historical_contact_motion_shape",
        ):
            fit_stats = append_touch_local_shape_features(
                fit_stats,
                fit_rows,
            )
            calibration_stats = append_touch_local_shape_features(
                calibration_stats,
                calibration_rows,
            )
            valid_stats = append_touch_local_shape_features(
                valid_stats,
                valid_rows,
            )
        if feature_mode in CAUSAL_MOTION_MODES:
            include_shape = feature_mode in CAUSAL_MOTION_SHAPE_MODES
            fit_stats = append_causal_motion_features(
                fit_stats,
                fit_rows,
                include_shape=include_shape,
            )
            calibration_stats = append_causal_motion_features(
                calibration_stats,
                calibration_rows,
                include_shape=include_shape,
            )
            valid_stats = append_causal_motion_features(
                valid_stats,
                valid_rows,
                include_shape=include_shape,
            )
    else:
        raise RuntimeError("unsupported_feature_family")
    if feature_family == "evidence":
        def normalize(values):
            return evidence_features.normalize(fit_stats, values)
    else:
        reference_stats = (
            lightweight.stats_features(clean_fit_normalized)
            if clean_fit_normalized is not None
            else fit_stats
        )
        stats_center = reference_stats.mean(axis=0)
        stats_scale = reference_stats.std(axis=0)
        stats_scale[stats_scale < 1e-6] = 1.0

        def normalize(values):
            return np.clip(
                (values - stats_center) / stats_scale,
                contract.STATS_CLIP_MIN,
                contract.STATS_CLIP_MAX,
            ).astype(np.float32)

    split_values = (
        (fit_rows, normalize(fit_stats)),
        (calibration_rows, normalize(calibration_stats)),
        (valid_rows, normalize(valid_stats)),
    )
    split_results = tuple(
        append_early_evidence(
            fit_rows,
            rows,
            values,
            early_evidence_mode,
        )
        for rows, values in split_values
    )
    result = tuple(values for values, _ in split_results)
    normalization_metadata["earlyPressEvidence"] = {
        "mode": early_evidence_mode,
        "contract": (
            early_press_evidence.CONTRACT
            if early_evidence_mode != "none"
            else None
        ),
        "fit": split_results[0][1],
        "calibration": split_results[1][1],
        "validation": split_results[2][1],
    }
    normalization_metadata["channelScaleAugmentation"] = {
        "mode": augmentation_mode,
        "probability": (
            float(augmentation_probability)
            if augmentation_mode == "channel_scale"
            else 0.0
        ),
        "maximumLogScale": (
            float(augmentation_max_log_scale)
            if augmentation_mode == "channel_scale"
            else 0.0
        ),
        "appliedPartitions": (
            ["fit"]
            if augmentation_mode == "channel_scale"
            else []
        ),
    }
    if return_metadata:
        return result + (normalization_metadata,)
    return result


def normalized_motion_domain(row):
    """Return the canonical known motion domain or None for missing metadata."""
    value = str(row.get("motionState") or "")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in contract.CONTEXT_UNKNOWN_VALUES:
        return None
    try:
        return contract.MOTION_LABELS[contract._context_target(
            normalized,
            contract.MOTION_LABELS,
            contract.MOTION_ALIASES,
        )]
    except ValueError as error:
        raise RuntimeError("invalid_motion_domain") from error


def scenario_domain(row):
    """Return the collection scenario used for balanced model selection."""
    motion = normalized_motion_domain(row)
    if motion == "walking":
        return "walking"
    normalized_motion = str(row.get("motionState") or "").strip().lower()
    if normalized_motion == "placed_static":
        return "placed"
    # 中文注释：任务和身体姿势只用于训练期分层，不进入模型输入。
    # 固定字段优先级，避免同一批数据因不同元数据拆成多组。
    for field in ("taskId", "bodyPosture"):
        value = str(row.get(field) or "").strip().lower().replace("-", "_")
        if value and value not in contract.CONTEXT_UNKNOWN_VALUES:
            return value
    if motion == "still":
        return "handheld_static"
    return motion or "unknown"


def posture_domain(row):
    """Return a canonical posture label without inventing missing evidence."""
    for field in ("bodyPosture", "taskId", "requestedDevicePose"):
        value = str(row.get(field) or "").strip().lower().replace("-", "_")
        if not value or value in contract.CONTEXT_UNKNOWN_VALUES:
            continue
        try:
            target = contract._context_target(
                value,
                contract.POSTURE_LABELS,
                contract.POSTURE_ALIASES,
            )
            return contract.POSTURE_LABELS[target]
        except ValueError:
            continue
        try:
            target = contract._context_target(
                value,
                contract.DEVICE_POSE_LABELS,
                contract.DEVICE_POSE_ALIASES,
            )
            return contract.DEVICE_POSE_LABELS[target]
        except ValueError:
            continue
    return "unknown"


def balanced_sample_weights(
        rows,
        gate_labels,
        weighting="session_balanced",
):
    """Balance gate classes across physical devices and motion domains."""
    if weighting in ("session_balanced", "block_balanced"):
        labels = np.asarray(gate_labels, dtype=np.int64)
        session_counts = Counter(row["groupId"] for row in rows)
        raw = np.asarray([
            1.0 / np.sqrt(session_counts[row["groupId"]])
            for row in rows
        ], dtype=np.float64)
        if weighting == "block_balanced":
            # 中文注释：完整实验块等权，避免大块主导固定误触率阈值。
            block_keys = [
                experiment_block_key(row) or row["groupId"]
                for row in rows
            ]
            block_sums = Counter()
            for key, weight in zip(block_keys, raw):
                block_sums[key] += weight
            for index, key in enumerate(block_keys):
                raw[index] /= max(1.0, block_sums[key])
        for label in range(len(contract.GATE_LABELS)):
            selected = labels == label
            if not np.any(selected):
                raise RuntimeError("missing_gate_class")
            raw[selected] /= raw[selected].sum()
        raw /= raw.sum()
        return np.clip(raw, 1e-8, None).astype(np.float32)
    if weighting not in ("device_balanced", "device_motion_balanced"):
        raise RuntimeError("unsupported_weighting_mode")
    group_counts = Counter(row["groupId"] for row in rows)
    device_counts = Counter(physical_device_key(row) for row in rows)
    labels = np.asarray(gate_labels, dtype=np.int64)
    if weighting == "device_motion_balanced":
        # 中文注释：运动域键包含设备、运动和标签。
        # 未知运动保留为独立单元，不把缺失元数据误归入静止或行走。
        motion_keys = [
            (
                physical_device_key(row),
                scenario_domain(row),
                int(label),
            )
            for row, label in zip(rows, labels)
        ]
        cell_counts = Counter(motion_keys)
        cells_by_label = Counter(key[2] for key in cell_counts)
        base = np.asarray([
            1.0 / np.sqrt(group_counts[row["groupId"]])
            for row in rows
        ], dtype=np.float64)
        weights = np.zeros(len(rows), dtype=np.float64)
        for cell, count in cell_counts.items():
            selected = np.asarray([
                key == cell for key in motion_keys
            ])
            cell_sum = base[selected].sum()
            if cell_sum <= 0.0:
                raise RuntimeError("invalid_motion_domain_weights")
            weights[selected] = (
                base[selected]
                / cell_sum
                / cells_by_label[cell[2]]
            )
        weights /= weights.sum()
        return np.clip(weights, 1e-8, None).astype(np.float32)
    else:
        motion_keys = [None] * len(rows)
    weights = np.asarray([
        1.0
        / np.sqrt(group_counts[row["groupId"]])
        / device_counts[physical_device_key(row)]
        for row in rows
    ], dtype=np.float64)
    for label in range(len(contract.GATE_LABELS)):
        selected = labels == label
        if not np.any(selected):
            raise RuntimeError("missing_gate_class")
        weights[selected] /= weights[selected].sum()
    weights /= weights.sum()
    return np.clip(weights, 1e-8, None).astype(np.float32)


def gate_probabilities(model, values, batch_size=1024):
    """Return heavy probabilities from the strength head."""
    model.eval()
    if not isinstance(values, torch.Tensor):
        values = torch.from_numpy(values)
    output = []
    heavy_index = contract.GATE_LABELS.index("heavy")
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            model_output = model(values[start:start + batch_size])
            _, strength_logits, _ = lightweight.unpack_multitask_output(
                model_output
            )
            probabilities = torch.softmax(strength_logits, dim=1)
            output.extend(probabilities[:, heavy_index].cpu().tolist())
    return np.asarray(output, dtype=np.float64)


def select_group_balanced_threshold(
        rows,
        labels,
        scores,
        target_fpr,
        group_field,
):
    """Select an FPR threshold after giving every negative group equal weight."""
    truths = np.asarray(labels, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if len(rows) != len(truths) or len(rows) != len(values):
        raise RuntimeError("balanced_threshold_shape_mismatch")
    if not 0.0 <= float(target_fpr) <= 1.0:
        raise RuntimeError("invalid_target_fpr")
    negative_indices = np.flatnonzero(truths == 0)
    if not len(negative_indices):
        raise RuntimeError("missing_negative_proxy")
    group_counts = Counter()
    for index in negative_indices:
        row = rows[int(index)]
        group_counts[str(row.get(group_field) or "unknown")] += 1
    weights = np.asarray([
        1.0 / group_counts[str(rows[int(index)].get(group_field) or "unknown")]
        for index in negative_indices
    ], dtype=np.float64)
    weights /= weights.sum()
    ordered = np.argsort(values[negative_indices], kind="stable")[::-1]
    ordered_scores = values[negative_indices][ordered]
    ordered_weights = weights[ordered]
    allowed = float(target_fpr)
    cumulative = 0.0
    index = 0
    while index < len(ordered_scores):
        score = ordered_scores[index]
        end = index
        while end + 1 < len(ordered_scores) and (
            ordered_scores[end + 1] == score
        ):
            end += 1
        next_cumulative = cumulative + float(
            ordered_weights[index:end + 1].sum()
        )
        # 中文注释：判定使用严格大于，因此阈值可以取当前同分组的分数。
        if next_cumulative > allowed + 1e-12:
            return float(score)
        cumulative = next_cumulative
        index = end + 1
    return float(-math.inf)


def fit_temperature_scaling(probabilities, labels):
    """Fit one temperature on calibration labels and report NLL before and after."""
    values = np.asarray(probabilities, dtype=np.float64)
    truths = np.asarray(labels, dtype=np.int64)
    if values.shape != truths.shape:
        raise RuntimeError("temperature_shape_mismatch")
    if not len(values):
        raise RuntimeError("temperature_empty_partition")
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    logits = torch.from_numpy(
        np.log(clipped / (1.0 - clipped))
    ).to(dtype=torch.float64)
    targets = torch.from_numpy(truths.astype(np.float64))
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        (log_temperature,),
        lr=0.2,
        max_iter=40,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits / temperature,
            targets,
        )
        loss.backward()
        return loss

    before = float(torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
    ))
    optimizer.step(closure)
    temperature = float(torch.exp(log_temperature.detach()))
    after = float(torch.nn.functional.binary_cross_entropy_with_logits(
        logits / temperature,
        targets,
    ))
    return {
        "temperature": temperature,
        "calibrationNllBefore": before,
        "calibrationNllAfter": after,
    }


def temperature_threshold_report(
        calibration_scores,
        calibration_labels,
        valid_rows,
        valid_labels,
        valid_scores,
        target_fpr,
        temperature,
):
    """Evaluate whether temperature scaling improves threshold transfer."""
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise RuntimeError("invalid_temperature")
    calibration = apply_temperature(calibration_scores, temperature)
    valid = apply_temperature(valid_scores, temperature)
    calibration_labels = np.asarray(calibration_labels, dtype=np.int64)
    threshold = gate_threshold_metrics.select_threshold_at_fpr(
        calibration[calibration_labels == 0],
        target_fpr,
    )
    report = gate_threshold_metrics.fixed_threshold_report(
        valid_rows,
        valid_labels,
        valid,
        threshold,
    )
    return {
        "temperature": float(temperature),
        "threshold": float(threshold),
        "overall": report["overall"],
    }


def apply_temperature(scores, temperature):
    """Apply temperature scaling to binary heavy probabilities."""
    values = np.asarray(scores, dtype=np.float64)
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise RuntimeError("invalid_temperature")
    clipped = np.clip(values, 1e-8, 1.0 - 1e-8)
    logits = np.log(clipped / (1.0 - clipped))
    scaled = logits / float(temperature)
    return 1.0 / (1.0 + np.exp(-np.clip(scaled, -60.0, 60.0)))


def score_group_diagnostics(rows, labels, scores, threshold):
    """Summarize score transfer by session, block, collector and posture."""
    truths = np.asarray(labels, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if len(rows) != len(truths) or len(rows) != len(values):
        raise RuntimeError("score_group_shape_mismatch")
    group_fields = {
        "session": lambda row: str(row.get("groupId") or "unknown"),
        "experimentBlock": lambda row: (
            experiment_block_key(row) or "unknown"
        ),
        "collector": lambda row: str(
            row.get("collectorUserName") or "unknown"
        ),
        "posture": posture_domain,
        "finger": lambda row: str(
            row.get("classLabel") or "unknown"
        ).split("_")[0],
    }
    result = {}
    for field, resolver in group_fields.items():
        grouped = defaultdict(list)
        for index, row in enumerate(rows):
            grouped[resolver(row)].append(index)
        field_result = {}
        for name, indices in sorted(grouped.items()):
            selected = np.asarray(indices, dtype=np.int64)
            selected_labels = truths[selected]
            selected_scores = values[selected]
            metrics = gate_threshold_metrics.binary_metrics(
                selected_labels,
                selected_scores,
                threshold,
            )
            entry = {
                "rowCount": len(indices),
                "positiveCount": metrics["positiveCount"],
                "negativeCount": metrics["negativeCount"],
                "heavyRecall": metrics["heavyRecall"],
                "lightFalsePositiveRate": metrics[
                    "lightFalsePositiveRate"
                ],
                "rocAuc": metrics["rocAuc"],
                "negativeMedian": (
                    float(np.median(selected_scores[selected_labels == 0]))
                    if np.any(selected_labels == 0)
                    else None
                ),
                "positiveMedian": (
                    float(np.median(selected_scores[selected_labels == 1]))
                    if np.any(selected_labels == 1)
                    else None
                ),
            }
            if entry["positiveCount"] and entry["negativeCount"]:
                oracle = gate_threshold_metrics.fixed_fpr_report(
                    [rows[int(index)] for index in selected],
                    selected_labels,
                    selected_scores,
                    0.01,
                )
                entry["withinGroupOracleRecallAt1pct"] = (
                    oracle["overall"]["heavyRecall"]
                )
            field_result[name] = entry
        result[field] = field_result
    return result


def score_quantiles(values):
    """Return compact quantiles and saturation rates for one score vector."""
    scores = np.asarray(values, dtype=np.float64)
    if not len(scores):
        return {
            "count": 0,
            "minimum": None,
            "p01": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "maximum": None,
            "saturatedLow": None,
            "saturatedHigh": None,
        }
    levels = (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
    quantiles = np.quantile(scores, levels)
    return {
        "count": int(len(scores)),
        "minimum": float(quantiles[0]),
        "p01": float(quantiles[1]),
        "p05": float(quantiles[2]),
        "p50": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "maximum": float(quantiles[6]),
        # 中文注释：sigmoid 饱和会使不同触摸得到几乎相同的分数。
        "saturatedLow": float(np.mean(scores < 1e-4)),
        "saturatedHigh": float(np.mean(scores > 1.0 - 1e-4)),
    }


def score_distribution_diagnostics(
        fit_scores,
        fit_labels,
        calibration_scores,
        calibration_labels,
        valid_scores,
        valid_labels,
):
    """Report score shift between fit, calibration and validation partitions."""
    def summarize(scores, labels):
        values = np.asarray(scores, dtype=np.float64)
        truths = np.asarray(labels, dtype=np.int64)
        if values.shape != truths.shape:
            raise RuntimeError("score_distribution_shape_mismatch")
        return {
            "rocAuc": gate_threshold_metrics.roc_auc(truths, values),
            "negative": score_quantiles(values[truths == 0]),
            "positive": score_quantiles(values[truths == 1]),
        }
    return {
        "fit": summarize(fit_scores, fit_labels),
        "calibration": summarize(calibration_scores, calibration_labels),
        "validation": summarize(valid_scores, valid_labels),
    }


def group_threshold_diagnostics(rows, labels, scores, group_field, target_fpr):
    """Report per-group oracle and transfer at one target false-positive rate."""
    truths = np.asarray(labels, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if len(rows) != len(truths) or len(rows) != len(values):
        raise RuntimeError("group_threshold_shape_mismatch")
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row.get(group_field) or "unknown")].append(index)
    result = {}
    for name, indices in sorted(grouped.items()):
        selected = np.asarray(indices, dtype=np.int64)
        selected_labels = truths[selected]
        selected_scores = values[selected]
        negatives = selected_scores[selected_labels == 0]
        positives = selected_scores[selected_labels == 1]
        if not len(negatives) or not len(positives):
            result[name] = {
                "rowCount": int(len(indices)),
                "negativeCount": int(len(negatives)),
                "positiveCount": int(len(positives)),
                "threshold": None,
                "heavyRecall": None,
                "lightFalsePositiveRate": None,
            }
            continue
        threshold = gate_threshold_metrics.select_threshold_at_fpr(
            negatives,
            target_fpr,
        )
        metrics = gate_threshold_metrics.binary_metrics(
            selected_labels,
            selected_scores,
            threshold,
        )
        result[name] = {
            "rowCount": int(len(indices)),
            "negativeCount": int(len(negatives)),
            "positiveCount": int(len(positives)),
            "threshold": float(threshold),
            "heavyRecall": metrics["heavyRecall"],
            "lightFalsePositiveRate": metrics["lightFalsePositiveRate"],
            "negativeQuantiles": score_quantiles(negatives),
            "positiveQuantiles": score_quantiles(positives),
        }
    return result


def threshold_calibration_diagnostics(
        calibration_rows,
        calibration_labels,
        calibration_scores,
        valid_rows,
        valid_labels,
        valid_scores,
        target_fpr,
):
    """Compare sample-weighted and group-weighted threshold transfer."""
    policies = {
        "sample": gate_threshold_metrics.select_threshold_at_fpr(
            np.asarray(calibration_scores)[
                np.asarray(calibration_labels) == 0
            ],
            target_fpr,
        ),
        "sessionBalanced": select_group_balanced_threshold(
            calibration_rows,
            calibration_labels,
            calibration_scores,
            target_fpr,
            "groupId",
        ),
        "blockBalanced": select_group_balanced_threshold(
            calibration_rows,
            calibration_labels,
            calibration_scores,
            target_fpr,
            "_experimentBlock",
        ),
    }
    result = {}
    for name, threshold in policies.items():
        report = gate_threshold_metrics.fixed_threshold_report(
            valid_rows,
            valid_labels,
            valid_scores,
            threshold,
        )
        result[name] = {
            "threshold": float(threshold),
            "overall": report["overall"],
        }
    calibration_negative = np.asarray(calibration_scores)[
        np.asarray(calibration_labels) == 0
    ]
    valid_negative = np.asarray(valid_scores)[
        np.asarray(valid_labels) == 0
    ]
    valid_positive = np.asarray(valid_scores)[
        np.asarray(valid_labels) == 1
    ]
    return {
        "targetFalsePositiveRate": float(target_fpr),
        "policies": result,
        "calibrationNegativeQuantiles": score_quantiles(
            calibration_negative
        ),
        "validationNegativeQuantiles": score_quantiles(valid_negative),
        "validationPositiveQuantiles": score_quantiles(valid_positive),
        "negativeP99Drift": (
            score_quantiles(valid_negative)["p99"]
            - score_quantiles(calibration_negative)["p99"]
        ),
    }


def calibration_domain_scores(
        truths,
        probabilities,
        motion_domains=None,
):
    """Return balanced accuracy for every usable calibration domain."""
    labels = np.asarray(truths, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != scores.shape:
        raise RuntimeError("calibration_score_shape_mismatch")
    if motion_domains is None:
        motion_domains = np.full(len(labels), "all", dtype=object)
    domains = np.asarray(motion_domains, dtype=object)
    if domains.shape != labels.shape:
        raise RuntimeError("calibration_domain_shape_mismatch")
    domain_scores = {}
    for domain in sorted(set(str(value) for value in domains)):
        selected = np.asarray(
            [str(value) == domain for value in domains],
            dtype=bool,
        )
        domain_labels = labels[selected]
        if not np.any(domain_labels == 0) or not np.any(domain_labels == 1):
            continue
        domain_scores[domain] = trainer.gate_metrics(
            domain_labels,
            np.stack((
                1.0 - scores[selected],
                scores[selected],
            ), axis=1),
        )["balancedAccuracy"]
    if not domain_scores:
        raise RuntimeError("calibration_missing_motion_stratum")
    return domain_scores


def calibration_score(truths, probabilities, motion_domains=None):
    """Select epochs with equal weight for each observed motion domain."""
    domain_scores = calibration_domain_scores(
        truths,
        probabilities,
        motion_domains,
    )
    # 中文注释：静止样本数量远高于行走，直接使用全体平衡准确率
    # 会让静止层吞掉模型选择权。这里每个已覆盖运动域等权。
    return float(np.mean(list(domain_scores.values())))


def calibration_fixed_fpr_score(
        truths,
        probabilities,
        motion_domains=None,
        target_fpr=0.01,
):
    """Return calibration-domain recall at one global target FPR."""
    labels = np.asarray(truths, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != scores.shape:
        raise RuntimeError("calibration_fixed_fpr_shape_mismatch")
    if motion_domains is None:
        motion_domains = np.full(len(labels), "all", dtype=object)
    domains = np.asarray(motion_domains, dtype=object)
    if domains.shape != labels.shape:
        raise RuntimeError("calibration_fixed_fpr_domain_shape_mismatch")
    negative_scores = scores[labels == 0]
    if not len(negative_scores):
        raise RuntimeError("calibration_missing_negative_proxy")
    threshold = gate_threshold_metrics.select_threshold_at_fpr(
        negative_scores,
        target_fpr,
    )
    domain_recalls = {}
    for domain in sorted(set(str(value) for value in domains)):
        selected = np.asarray(
            [str(value) == domain for value in domains],
            dtype=bool,
        )
        domain_labels = labels[selected]
        if not np.any(domain_labels == 0) or not np.any(domain_labels == 1):
            continue
        domain_scores = scores[selected]
        domain_recalls[domain] = float(np.mean(
            domain_scores[domain_labels == 1] > threshold
        ))
    if not domain_recalls:
        raise RuntimeError("calibration_missing_fixed_fpr_stratum")
    return {
        "score": float(np.mean(list(domain_recalls.values()))),
        "threshold": float(threshold),
        "domainRecall": domain_recalls,
    }


def calibration_selection_score(
        selection_mode,
        truths,
        probabilities,
        motion_domains=None,
        target_fpr=0.01,
):
    """Return the epoch-selection objective and its diagnostic components."""
    if selection_mode not in MODEL_SELECTION_MODES:
        raise RuntimeError("unsupported_model_selection_mode")
    balanced = calibration_score(
        truths,
        probabilities,
        motion_domains,
    )
    fixed = calibration_fixed_fpr_score(
        truths,
        probabilities,
        motion_domains,
        target_fpr,
    )
    if selection_mode == "fixed_fpr_recall":
        # 中文注释：1% FPR 下的域等权召回是主目标，平衡准确率只用于同分时稳定排序。
        primary = fixed["score"]
        tie_break = balanced
    else:
        primary = balanced
        tie_break = fixed["score"]
    return {
        "score": float(primary),
        "tieBreakScore": float(tie_break),
        "balancedAccuracy": float(balanced),
        "fixedFprRecall": float(fixed["score"]),
        "fixedFprThreshold": float(fixed["threshold"]),
        "fixedFprDomainRecall": fixed["domainRecall"],
    }


def pairwise_ranking_loss(
        strength_logits,
        gate_targets,
        sample_weights,
        margin=0.0,
):
    """Rank every heavy touch above every light touch in one batch.

    中文注释：固定误触率最终依赖全局排序，而不是四分类交叉熵。
    这里直接约束 heavy 的强度边际大于 light；样本权重同时进入每个
    配对，避免大 session 主导。没有任一类别时返回可求导的零值。
    """
    if strength_logits.ndim != 2 or strength_logits.shape[1] != 2:
        raise RuntimeError("invalid_pairwise_strength_shape")
    targets = gate_targets.reshape(-1)
    weights = sample_weights.reshape(-1)
    if strength_logits.shape[0] != targets.shape[0]:
        raise RuntimeError("pairwise_target_shape_mismatch")
    if targets.shape != weights.shape:
        raise RuntimeError("pairwise_weight_shape_mismatch")
    heavy = int(contract.GATE_LABELS.index("heavy"))
    light = int(contract.GATE_LABELS.index("light"))
    scores = strength_logits[:, heavy] - strength_logits[:, light]
    positive = scores[targets == heavy]
    negative = scores[targets == light]
    if not len(positive) or not len(negative):
        return strength_logits.sum() * 0.0
    positive_weights = weights[targets == heavy]
    negative_weights = weights[targets == light]
    pair_weights = (
        positive_weights[:, None] * negative_weights[None, :]
    )
    denominator = pair_weights.sum()
    if float(denominator.detach()) <= 0.0:
        return strength_logits.sum() * 0.0
    differences = positive[:, None] - negative[None, :] - float(margin)
    losses = torch.nn.functional.softplus(-differences)
    return (losses * pair_weights).sum() / denominator


def asymmetric_focal_loss(
        strength_logits,
        gate_targets,
        sample_weights,
        positive_weight=1.0,
        negative_weight=1.0,
):
    """Apply asymmetric class weights with focal modulation.

    中文注释：固定误触率下，普通轻触误报的代价高于漏掉少数重触；
    但负样本不能压成唯一目标，否则正类排序退化。这里使用 focal
    降低已正确样本的影响，再用类别权重调节正负错误代价。
    """
    if not 0.0 <= positive_weight <= 100.0:
        raise RuntimeError("invalid_positive_focal_weight")
    if not 0.0 <= negative_weight <= 100.0:
        raise RuntimeError("invalid_negative_focal_weight")
    targets = gate_targets.reshape(-1)
    weights = sample_weights.reshape(-1)
    if strength_logits.shape[0] != targets.shape[0]:
        raise RuntimeError("focal_target_shape_mismatch")
    if targets.shape != weights.shape:
        raise RuntimeError("focal_weight_shape_mismatch")
    probabilities = torch.softmax(strength_logits, dim=1)
    selected = probabilities.gather(
        1,
        targets[:, None],
    ).reshape(-1)
    positive = targets == contract.GATE_LABELS.index("heavy")
    class_weights = torch.where(
        positive,
        torch.as_tensor(
            float(positive_weight),
            dtype=selected.dtype,
            device=selected.device,
        ),
        torch.as_tensor(
            float(negative_weight),
            dtype=selected.dtype,
            device=selected.device,
        ),
    )
    focal_weights = torch.pow(
        1.0 - selected.clamp(1e-8, 1.0 - 1e-8),
        2.0,
    )
    losses = -focal_weights * torch.log(selected.clamp(1e-8, 1.0))
    weighted = losses * weights * class_weights
    denominator = weights.sum()
    if float(denominator.detach()) <= 0.0:
        return strength_logits.sum() * 0.0
    return weighted.sum() / denominator


def train_model(
        values,
        labels,
        gate_labels,
        context_labels,
        sample_weights,
        calibration_values,
        calibration_gate_labels,
        calibration_motion_domains,
        input_feature_count,
        hidden_count,
        seed,
        epochs,
        context_heads,
        context_loss_weight,
        finger_loss_weight,
        patience,
        selection_mode="balanced_accuracy",
        selection_target_fpr=0.01,
        gate_loss_mode="cross_entropy",
        ranking_loss_weight=0.0,
        pairwise_margin=0.0,
        positive_focal_weight=1.0,
        negative_focal_weight=1.0,
):
    """Train one model with calibration-session early stopping."""
    if gate_loss_mode not in GATE_LOSS_MODES:
        raise RuntimeError("unsupported_gate_loss_mode")
    if not 0.0 <= ranking_loss_weight <= 10.0:
        raise RuntimeError("invalid_ranking_loss_weight")
    if not 0.0 <= pairwise_margin <= 10.0:
        raise RuntimeError("invalid_pairwise_margin")
    trainer.configure_determinism(seed)
    model = lightweight.StatsMlp(
        input_feature_count,
        hidden_count,
        True,
        contract.CONTEXT_CLASS_COUNTS if context_heads else None,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.002,
        weight_decay=0.003,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=0.00005,
    )
    x = torch.from_numpy(values)
    calibration_x = torch.from_numpy(calibration_values)
    class_targets = torch.tensor(labels, dtype=torch.long)
    gate_targets = torch.tensor(gate_labels, dtype=torch.long)
    context_targets = torch.tensor(context_labels, dtype=torch.long)
    weights = torch.from_numpy(sample_weights)
    best_state = None
    best_key = None
    best_epoch = 0
    selection_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(order), 128):
            batch = order[start:start + 128]
            model_output = model(x[batch])
            finger_logits, strength_logits, context_logits = (
                lightweight.unpack_multitask_output(model_output)
            )
            gate_losses = torch.nn.functional.cross_entropy(
                strength_logits,
                gate_targets[batch],
                reduction="none",
            )
            if gate_loss_mode == "pairwise_ranking":
                loss = pairwise_ranking_loss(
                    strength_logits,
                    gate_targets[batch],
                    weights[batch],
                    pairwise_margin,
                )
            elif gate_loss_mode == "asymmetric_focal":
                loss = asymmetric_focal_loss(
                    strength_logits,
                    gate_targets[batch],
                    weights[batch],
                    positive_focal_weight,
                    negative_focal_weight,
                )
            else:
                loss = (
                    gate_losses * weights[batch]
                ).sum() / weights[batch].sum()
                if gate_loss_mode == "ce_pairwise":
                    loss = loss + ranking_loss_weight * (
                        pairwise_ranking_loss(
                            strength_logits,
                            gate_targets[batch],
                            weights[batch],
                            pairwise_margin,
                        )
                    )
            if finger_loss_weight > 0.0:
                loss += finger_loss_weight * (
                    torch.nn.functional.cross_entropy(
                        finger_logits,
                        class_targets[batch] // 2,
                    )
                )
            if context_heads:
                if len(context_logits) != len(
                    contract.CONTEXT_CLASS_COUNTS
                ):
                    raise RuntimeError("invalid_context_head_count")
                context_loss = sum(
                    trainer.masked_context_loss(
                        logits,
                        context_targets[batch, index],
                    )
                    for index, logits in enumerate(context_logits)
                )
                loss += (
                    context_loss_weight
                    * context_loss
                    / len(context_logits)
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        calibration_probabilities = gate_probabilities(
            model,
            calibration_x,
        )
        selection_score = calibration_selection_score(
            selection_mode,
            calibration_gate_labels,
            calibration_probabilities,
            calibration_motion_domains,
            selection_target_fpr,
        )
        selection_history.append({
            "epoch": epoch,
            "selectionMode": selection_mode,
            "selectionScore": selection_score["score"],
            "tieBreakScore": selection_score["tieBreakScore"],
            "macroBalancedAccuracy": selection_score["balancedAccuracy"],
            "calibrationFixedFprRecall": selection_score[
                "fixedFprRecall"
            ],
            "calibrationFixedFprThreshold": selection_score[
                "fixedFprThreshold"
            ],
            "calibrationFixedFprDomainRecall": selection_score[
                "fixedFprDomainRecall"
            ],
        })
        selection_key = (
            selection_score["score"],
            selection_score["tieBreakScore"],
        )
        if best_key is None or selection_key > (
            best_key[0] + 1e-12,
            best_key[1] + 1e-12,
        ):
            best_key = selection_key
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
        if epoch - best_epoch >= patience:
            break
    if best_state is None:
        raise RuntimeError("model_selection_failed")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "bestEpoch": best_epoch,
        "selectionMode": selection_mode,
        "selectionTargetFalsePositiveRate": float(selection_target_fpr),
        "gateLossMode": gate_loss_mode,
        "rankingLossWeight": float(ranking_loss_weight),
        "pairwiseMargin": float(pairwise_margin),
        "positiveFocalWeight": float(positive_focal_weight),
        "negativeFocalWeight": float(negative_focal_weight),
        "calibrationBalancedAccuracy": (
            best_key[1]
            if selection_mode == "fixed_fpr_recall"
            else best_key[0]
        ),
        "calibrationFixedFprRecall": (
            best_key[0]
            if selection_mode == "fixed_fpr_recall"
            else best_key[1]
        ),
        "selectionHistory": selection_history,
    }


def run_fold_seed(
        fold,
        seed,
        hidden_count,
        epochs,
        context_heads,
        context_loss_weight,
        finger_loss_weight,
        calibration_ratio,
        patience,
        feature_mode,
        weighting,
        modality_dropout,
        imu_mode,
        adaptive_feature_mode,
        target_calibration_counts,
        target_calibration_repetitions=8,
        evaluation_only_rows=None,
        feature_family="stats",
        evidence_contract=evidence_features.DEFAULT_EVIDENCE_CONTRACT,
        ordinary_click_rows=None,
        normalization_mode="none",
        early_evidence_mode="none",
        augmentation_mode="none",
        augmentation_probability=0.5,
        augmentation_max_log_scale=0.35,
        selection_mode="balanced_accuracy",
        selection_target_fpr=0.01,
        gate_loss_mode="cross_entropy",
        ranking_loss_weight=0.0,
        pairwise_margin=0.0,
        positive_focal_weight=1.0,
        negative_focal_weight=1.0,
        threshold_calibration_fpr=DEFAULT_THRESHOLD_CALIBRATION_FPR,
        threshold_calibration_sweep=(),
        false_positive_limit=DEFAULT_FALSE_POSITIVE_LIMIT,
):
    """Evaluate one model specification without touching validation labels."""
    fold_evaluation_rows = fold.get(
        "evaluationOnlyRows",
        evaluation_only_rows,
    )
    if fold.get("fixedTrainRows") is not None:
        fit_rows = fold["fixedTrainRows"]
        calibration_rows = fold["fixedCalibrationRows"]
        ensure_fold_coverage(fit_rows, calibration_rows)
    else:
        fit_rows, calibration_rows = calibration_split(
            fold["train"],
            seed + 70000,
            calibration_ratio,
            group_field=fold.get("calibrationGroupField", "groupId"),
        )
    (
        fit_values,
        calibration_values,
        valid_values,
        normalization_metadata,
    ) = prepare_values(
        fit_rows=fit_rows,
        calibration_rows=calibration_rows,
        valid_rows=fold["valid"],
        feature_mode=feature_mode,
        modality_dropout=modality_dropout,
        augmentation_mode=augmentation_mode,
        augmentation_probability=augmentation_probability,
        augmentation_max_log_scale=augmentation_max_log_scale,
        seed=seed + 90000,
        imu_mode=imu_mode,
        adaptive_feature_mode=adaptive_feature_mode,
        feature_family=feature_family,
        evidence_contract=evidence_contract,
        normalization_mode=normalization_mode,
        early_evidence_mode=early_evidence_mode,
        return_metadata=True,
    )
    fit_labels, fit_gate_labels, fit_context_labels = validate_rows(fit_rows)
    _, calibration_gate_labels, _ = validate_rows(calibration_rows)
    calibration_motion_domains = [
        scenario_domain(row)
        for row in calibration_rows
    ]
    _, valid_gate_labels, _ = validate_rows(fold["valid"])
    sample_weights = balanced_sample_weights(
        fit_rows,
        fit_gate_labels,
        weighting,
    )
    model, selection = train_model(
        fit_values,
        fit_labels,
        fit_gate_labels,
        fit_context_labels,
        sample_weights,
        calibration_values,
        calibration_gate_labels,
        calibration_motion_domains,
        fit_values.shape[1],
        hidden_count,
        seed,
        epochs,
        context_heads,
        context_loss_weight,
        finger_loss_weight,
        patience,
        selection_mode,
        selection_target_fpr,
        gate_loss_mode,
        ranking_loss_weight,
        pairwise_margin,
        positive_focal_weight,
        negative_focal_weight,
    )
    fit_heavy = gate_probabilities(model, fit_values)
    calibration_heavy = gate_probabilities(model, calibration_values)
    valid_heavy = gate_probabilities(model, valid_values)
    calibration_distribution = score_distribution_diagnostics(
        fit_heavy,
        fit_gate_labels,
        calibration_heavy,
        calibration_gate_labels,
        valid_heavy,
        valid_gate_labels,
    )
    temperature_scaling = fit_temperature_scaling(
        calibration_heavy,
        calibration_gate_labels,
    )
    calibration_negatives = np.asarray([
        score
        for score, label in zip(calibration_heavy, calibration_gate_labels)
        if label == 0
    ])
    if not len(calibration_negatives):
        raise RuntimeError("calibration_missing_negative_proxy")
    fixed_reports = {}
    fixed_thresholds = {}
    for target_fpr in TARGET_FPRS:
        threshold = gate_threshold_metrics.select_threshold_at_fpr(
            calibration_negatives,
            threshold_calibration_fpr,
        )
        fixed_thresholds[str(target_fpr)] = threshold
        report = (
            gate_threshold_metrics.fixed_threshold_report(
                fold["valid"],
                valid_gate_labels,
                valid_heavy,
                threshold,
            )
        )
        report["withinFoldOracleDiagnostic"] = (
            gate_threshold_metrics.fixed_fpr_report(
                fold["valid"],
                valid_gate_labels,
                valid_heavy,
                target_fpr,
            )
        )
        minimum_proof_negative_count = int(
            np.ceil(5.0 / float(target_fpr))
        )
        report["minimumProofNegativeCount"] = (
            minimum_proof_negative_count
        )
        report["calibrationNegativeCount"] = len(calibration_negatives)
        report["heldOutNegativeCount"] = report["overall"]["negativeCount"]
        report["calibrationEvidenceStatus"] = (
            "proof_ready"
            if (
                len(calibration_negatives) >= minimum_proof_negative_count
                and report["heldOutNegativeCount"]
                >= minimum_proof_negative_count
            )
            else "descriptive_only"
        )
        report["proofReady"] = (
            report["calibrationEvidenceStatus"] == "proof_ready"
        )
        report["thresholdCalibrationDiagnostics"] = (
            threshold_calibration_diagnostics(
                calibration_rows,
                calibration_gate_labels,
                calibration_heavy,
                fold["valid"],
                valid_gate_labels,
                valid_heavy,
                target_fpr,
            )
        )
        report["temperatureScalingDiagnostic"] = (
            temperature_threshold_report(
                calibration_heavy,
                calibration_gate_labels,
                fold["valid"],
                valid_gate_labels,
                valid_heavy,
                target_fpr,
                temperature_scaling["temperature"],
            )
        )
        report["validationScoreGroups"] = score_group_diagnostics(
            fold["valid"],
            valid_gate_labels,
            valid_heavy,
            threshold,
        )
        report["validationBlockThresholdDiagnostics"] = (
            group_threshold_diagnostics(
                fold["valid"],
                valid_gate_labels,
                valid_heavy,
                "_experimentBlock",
                target_fpr,
            )
        )
        sweep_reports = {}
        for calibration_fpr in threshold_calibration_sweep:
            sweep_threshold = gate_threshold_metrics.select_threshold_at_fpr(
                calibration_negatives,
                calibration_fpr,
            )
            sweep_report = gate_threshold_metrics.fixed_threshold_report(
                fold["valid"],
                valid_gate_labels,
                valid_heavy,
                sweep_threshold,
            )
            measured_fpr = sweep_report["overall"][
                "lightFalsePositiveRate"
            ]
            sweep_reports[str(calibration_fpr)] = {
                "calibrationFalsePositiveRate": float(calibration_fpr),
                "threshold": float(sweep_threshold),
                "thresholdPolicy": sweep_report["thresholdPolicy"],
                "overall": sweep_report["overall"],
                "groups": sweep_report["groups"],
                "falsePositiveLimit": float(false_positive_limit),
                "falsePositiveLimitMet": bool(
                    measured_fpr is not None
                    and float(measured_fpr)
                    < float(false_positive_limit)
                ),
            }
        report["thresholdCalibrationSweep"] = sweep_reports
        fixed_reports[str(target_fpr)] = report
    evaluation_reports = {}
    ordinary_click_reports = {}
    evaluation_values = None
    if fold_evaluation_rows:
        (
            evaluation_values,
            evaluation_normalization_metadata,
        ) = prepare_evaluation_values(
            fit_rows=fit_rows,
            evaluation_rows=fold_evaluation_rows,
            feature_mode=feature_mode,
            imu_mode=imu_mode,
            adaptive_feature_mode=adaptive_feature_mode,
            feature_family=feature_family,
            evidence_contract=evidence_contract,
            normalization_mode=normalization_mode,
            early_evidence_mode=early_evidence_mode,
            return_metadata=True,
        )
        _, evaluation_gate_labels, _ = validate_rows(fold_evaluation_rows)
        evaluation_heavy = gate_probabilities(model, evaluation_values)
        for target_fpr in TARGET_FPRS:
            evaluation_reports[str(target_fpr)] = (
                gate_threshold_metrics.fixed_threshold_report(
                    fold_evaluation_rows,
                    evaluation_gate_labels,
                    evaluation_heavy,
                    fixed_thresholds[str(target_fpr)],
                )
            )
            evaluation_reports[str(target_fpr)][
                "thresholdSource"
            ] = "internal_fit_calibration_only"
            minimum_proof_negative_count = int(
                np.ceil(5.0 / float(target_fpr))
            )
            evaluation_report = evaluation_reports[str(target_fpr)]
            evaluation_report["minimumProofNegativeCount"] = (
                minimum_proof_negative_count
            )
            evaluation_report["calibrationNegativeCount"] = len(
                calibration_negatives
            )
            evaluation_report["heldOutNegativeCount"] = (
                evaluation_report["overall"]["negativeCount"]
            )
            evaluation_report["minimumProofSessionCount"] = (
                MIN_PROOF_SESSION_COUNT
            )
            evaluation_report["calibrationSessionCount"] = len({
                str(row.get("groupId") or "").strip()
                for row, label in zip(
                    calibration_rows,
                    calibration_gate_labels,
                )
                if label == 0
                and str(row.get("groupId") or "").strip()
            })
            evaluation_report["heldOutNegativeSessionCount"] = len({
                str(row.get("groupId") or "").strip()
                for row, label in zip(
                    fold_evaluation_rows,
                    evaluation_gate_labels,
                )
                if label == 0
                and str(row.get("groupId") or "").strip()
            })
            evaluation_report["minimumProofPositiveCount"] = (
                MIN_GROUP_POSITIVES
            )
            evaluation_report["heldOutPositiveCount"] = int(
                evaluation_report["overall"]["positiveCount"]
            )
            evaluation_report["calibrationEvidenceStatus"] = (
                "proof_ready"
                if (
                    len(calibration_negatives)
                    >= minimum_proof_negative_count
                    and evaluation_report["heldOutNegativeCount"]
                    >= minimum_proof_negative_count
                    and evaluation_report["heldOutPositiveCount"]
                    >= MIN_GROUP_POSITIVES
                    and evaluation_report["calibrationSessionCount"]
                    >= MIN_PROOF_SESSION_COUNT
                    and evaluation_report[
                        "heldOutNegativeSessionCount"
                    ] >= MIN_PROOF_SESSION_COUNT
                )
                else "descriptive_only"
            )
            evaluation_report["proofReady"] = (
                evaluation_report["calibrationEvidenceStatus"]
                == "proof_ready"
            )
    if ordinary_click_rows:
        ordinary_values = prepare_evaluation_values(
            fit_rows=fit_rows,
            evaluation_rows=ordinary_click_rows,
            feature_mode=feature_mode,
            imu_mode=imu_mode,
            adaptive_feature_mode=adaptive_feature_mode,
            feature_family=feature_family,
            evidence_contract=evidence_contract,
            normalization_mode=normalization_mode,
            early_evidence_mode=early_evidence_mode,
        )
        ordinary_heavy = gate_probabilities(model, ordinary_values)
        for target_fpr in TARGET_FPRS:
            ordinary_click_reports[str(target_fpr)] = (
                gate_threshold_metrics.fixed_negative_audit_report(
                    ordinary_click_rows,
                    ordinary_heavy,
                    fixed_thresholds[str(target_fpr)],
                )
            )
            ordinary_report = ordinary_click_reports[str(target_fpr)]
            ordinary_report["thresholdSource"] = (
                "internal_fit_calibration_only"
            )
            ordinary_report["minimumProofNegativeCount"] = int(
                np.ceil(5.0 / float(target_fpr))
            )
            ordinary_report["proofReady"] = (
                ordinary_report["overall"]["negativeCount"]
                >= ordinary_report["minimumProofNegativeCount"]
            )
            ordinary_report["evidenceStatus"] = (
                "proof_ready"
                if ordinary_report["proofReady"]
                else "descriptive_only"
            )
    runtime_state = trainer.runtime_state_dict(model.state_dict())
    target_calibrated = {
        str(count): target_calibrated_reports(
            fold["valid"],
            valid_gate_labels,
            valid_heavy,
            count,
            seed + 110000 + count,
            target_calibration_repetitions,
        )
        for count in target_calibration_counts
    }
    return {
        "seed": seed,
        "foldKind": fold["kind"],
        "foldId": fold["id"],
        "experimentBlock": fold.get("experimentBlock"),
        "blockPartitionAudit": fold.get("blockPartitionAudit"),
        "featureMode": feature_mode,
        "featureFamily": feature_family,
        "evidenceContract": (
            evidence_features.contract_metadata(evidence_contract)
            if feature_family == "evidence"
            else None
        ),
        "featureNames": (
            list(evidence_features.FEATURE_NAMES)
            if feature_family == "evidence"
            else (
                list(contract.STATS_FEATURE_NAMES)
                + (
                    list(TOUCH_LOCAL_FEATURE_NAMES)
                    if feature_mode in (
                        "touch_local_shape",
                        "touch_local_motion_shape",
                        "historical_contact_local_shape",
                        "historical_contact_motion_shape",
                    )
                    else []
                )
                + (
                    list(CAUSAL_MOTION_FEATURE_NAMES)
                    if feature_mode in CAUSAL_MOTION_MODES
                    else []
                )
                + (
                    list(TOUCH_LOCAL_FEATURE_NAMES)
                    if feature_mode in CAUSAL_MOTION_SHAPE_MODES
                    else []
                )
            )
        ),
        "weighting": weighting,
        "inputFeatureCount": int(fit_values.shape[1]),
        "modalityDropout": modality_dropout,
        "augmentationMode": augmentation_mode,
        "augmentationProbability": (
            augmentation_probability
            if augmentation_mode == "channel_scale"
            else 0.0
        ),
        "augmentationMaximumLogScale": (
            augmentation_max_log_scale
            if augmentation_mode == "channel_scale"
            else 0.0
        ),
        "imuMode": imu_mode,
        "adaptiveFeatureMode": adaptive_feature_mode,
        "normalizationMode": normalization_mode,
        "earlyEvidenceMode": early_evidence_mode,
        "earlyEvidenceContract": (
            early_press_evidence.CONTRACT
            if early_evidence_mode != "none"
            else None
        ),
        "normalizationAudit": normalization_metadata,
        "historicalContactContract": (
            {
                "featureNames": list(HISTORICAL_CONTACT_FEATURE_NAMES),
                "minimumHistoryTouches": (
                    HISTORICAL_CONTACT_MIN_TOUCHES
                ),
                "maximumHistoryTouches": (
                    HISTORICAL_CONTACT_MAX_TOUCHES
                ),
                "ordering": "same_session_timestamp_then_source_sha256",
                "futureRowsExcluded": True,
                "currentRowExcluded": True,
            }
            if feature_mode in HISTORICAL_CONTACT_MODES
            else None
        ),
        "causalMotionContract": (
            {
                "featureNames": list(CAUSAL_MOTION_FEATURE_NAMES),
                "baselineFrames": TOUCH_START_FRAME,
                "touchFrames": len(TOUCH_GRID_MS) - TOUCH_START_FRAME,
                "futureRowsExcluded": True,
                "missingImuPolicy": "fail_closed",
            }
            if feature_mode in CAUSAL_MOTION_MODES
            else None
        ),
        "fitSampleCount": len(fit_rows),
        "calibrationSampleCount": len(calibration_rows),
        "validSampleCount": len(fold["valid"]),
        "fitSessionCount": len({row["groupId"] for row in fit_rows}),
        "calibrationSessionCount": len({
            row["groupId"] for row in calibration_rows
        }),
        "validSessionCount": len({
            row["groupId"] for row in fold["valid"]
        }),
        "validExperimentBlockCount": len({
            experiment_block_key(row) for row in fold["valid"]
        }),
        "trainingParameterCount": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "runtimeParameterCount": sum(
            parameter.numel() for parameter in runtime_state.values()
        ),
        "selection": selection,
        "selectionMode": selection_mode,
        "selectionTargetFalsePositiveRate": float(
            selection_target_fpr
        ),
        "gateLossMode": gate_loss_mode,
        "rankingLossWeight": float(ranking_loss_weight),
        "pairwiseMargin": float(pairwise_margin),
        "positiveFocalWeight": float(positive_focal_weight),
        "negativeFocalWeight": float(negative_focal_weight),
        "scoreDistribution": {
            "partitions": calibration_distribution,
            "temperatureScaling": temperature_scaling,
        },
        "argmaxMetrics": trainer.gate_metrics(
            valid_gate_labels,
            np.stack((1.0 - valid_heavy, valid_heavy), axis=1),
        ),
        "fixedFprMetrics": fixed_reports,
        "fixedFprOracleDiagnostics": {
            target: fixed_reports[target][
                "withinFoldOracleDiagnostic"
            ]
            for target in fixed_reports
        },
        "targetDeviceThresholdDiagnostics": {
            str(target): target_device_threshold_diagnostics(
                valid_gate_labels,
                valid_heavy,
                target,
            )
            for target in TARGET_FPRS
        },
        "targetCalibratedMetrics": target_calibrated,
        "evaluationOnlyMetrics": evaluation_reports,
        "ordinaryClickAudit": ordinary_click_reports,
        "evaluationOnlySampleCount": len(fold_evaluation_rows or []),
        "evaluationOnlyDeviceNames": sorted({
            str(row.get("deviceName") or "")
            for row in (fold_evaluation_rows or [])
        }),
        "evaluationNormalizationAudit": (
            evaluation_normalization_metadata
            if fold_evaluation_rows
            else None
        ),
        "ordinaryClickSampleCount": len(ordinary_click_rows or []),
        "ordinaryClickDeviceNames": sorted({
            str(row.get("deviceName") or "")
            for row in (ordinary_click_rows or [])
        }),
    }


def target_device_threshold_diagnostics(
        truths,
        heavy_probabilities,
        target_fpr,
):
    """Measure threshold transfer using target negatives only as diagnostics."""
    labels = np.asarray(truths, dtype=np.int64)
    scores = np.asarray(heavy_probabilities, dtype=np.float64)
    negatives = scores[labels == 0]
    if not len(negatives):
        raise RuntimeError("missing_target_device_negative_proxy")
    threshold = gate_threshold_metrics.select_threshold_at_fpr(
        negatives,
        target_fpr,
    )
    return {
        "thresholdPolicy": "target_device_negative_replay",
        "targetFalsePositiveRate": float(target_fpr),
        "overall": gate_threshold_metrics.binary_metrics(
            labels,
            scores,
            threshold,
        ),
    }


def finite_values(values):
    return [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]


def verified_group_name(value):
    """Return whether one group value is an observed evidence bucket."""
    normalized = str(value or "").strip().lower()
    return normalized not in UNVERIFIED_GROUP_VALUES


def report_group_extreme(
        reports,
        field,
        metric,
        count_key,
        minimum_count,
        choose_min=True,
):
    """Aggregate a metric across observed groups without scoring unknowns."""
    values = []
    for report in reports:
        groups = report_group_map(report, field)
        for name, metrics in groups.items():
            if not verified_group_name(name):
                continue
            if metrics.get(count_key, 0) < minimum_count:
                continue
            value = metrics.get(metric)
            if value is not None and np.isfinite(float(value)):
                values.append(float(value))
    if not values:
        return None
    return min(values) if choose_min else max(values)


def report_group_names(
        reports,
        field,
        count_key,
        minimum_count,
):
    """Collect observed evidence groups that meet the count threshold."""
    names = set()
    for report in reports:
        for name, metrics in report_group_map(report, field).items():
            if not verified_group_name(name):
                continue
            if metrics.get(count_key, 0) >= minimum_count:
                names.add(str(name))
    return sorted(names)


def report_group_map(report, field):
    """Read a report group with a legacy posture fallback."""
    groups = report.get("groups", {})
    selected = groups.get(field)
    if selected is None and field == "postureDomain":
        # 中文注释：旧报告只生成 bodyPosture；兼容读取可让历史
        # 固定证据继续参与回归，而不改写原始报告。
        selected = groups.get("bodyPosture", {})
    return selected or {}


def collection_batch_key(row):
    """Extract a diagnostic collection batch without using it as a model input."""
    source_file = str(row.get("sourceFile") or "").replace("\\", "/")
    parts = [part for part in source_file.split("/") if part]
    if len(parts) >= 2:
        return parts[0]
    dataset = str(row.get("_datasetId") or "").strip()
    return dataset or "unknown"


def collection_batch_diagnostics(rows):
    """Summarize collection-domain drift for auditing, never for fitting."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(
            collection_batch_key(row),
            str(row.get("deviceName") or "unknown"),
        )].append(row)
    result = {}
    for (batch, device), selected in sorted(grouped.items()):
        labels = Counter(
            str(row.get("classLabel") or "unknown")
            for row in selected
        )
        motion = Counter(
            str(row.get("motionState") or "unknown")
            for row in selected
        )
        posture = Counter(
            str(row.get("bodyPosture") or "unknown")
            for row in selected
        )
        missing_shape = 0
        missing_pressure = 0
        key_signals = defaultdict(list)
        for row in selected:
            matrix = np.asarray(row["matrix"], dtype=np.float32)
            if matrix.shape[1] < 26:
                raise RuntimeError("batch_diagnostic_invalid_matrix")
            missing_shape += int(
                not np.any(matrix[:, 22:25] >= 0.5)
            )
            missing_pressure += int(
                not np.any(matrix[:, 20] >= 0.5)
            )
            acceleration = matrix[:, 2:5]
            rotation = matrix[:, 5:8]
            key_signals["accelerationJerkMean"].append(
                float(np.linalg.norm(
                    np.diff(acceleration, axis=0),
                    axis=1,
                ).mean())
            )
            key_signals["accelerationMagnitudeMean"].append(
                float(np.linalg.norm(
                    acceleration,
                    axis=1,
                ).mean())
            )
            key_signals["rotationMagnitudeMean"].append(
                float(np.linalg.norm(
                    rotation,
                    axis=1,
                ).mean())
            )
        result[batch + "|" + device] = {
            "batch": batch,
            "deviceName": device,
            "sampleCount": len(selected),
            "sessionCount": len({
                str(row.get("groupId") or "")
                for row in selected
            }),
            "classLabelCounts": dict(sorted(labels.items())),
            "motionStateCounts": dict(sorted(motion.items())),
            "bodyPostureCounts": dict(sorted(posture.items())),
            "missingShapeCount": missing_shape,
            "missingPressureCount": missing_pressure,
            "signalMedians": {
                name: float(np.median(values))
                for name, values in sorted(key_signals.items())
            },
        }
    return result


def group_recall_values(report, minimum_positives=MIN_GROUP_POSITIVES):
    """Return valid group recalls from one fixed-threshold report."""
    result = []
    for field_groups in report["groups"].values():
        for metrics in field_groups.values():
            if metrics["positiveCount"] >= minimum_positives:
                recall = metrics["heavyRecall"]
                if recall is not None:
                    result.append(float(recall))
    return result


def group_false_positive_rate_values(
        report,
        minimum_negatives=MIN_GROUP_NEGATIVES,
):
    """Return valid group false-positive rates from one report."""
    result = []
    for field_groups in report["groups"].values():
        for metrics in field_groups.values():
            if metrics.get("negativeCount", 0) >= minimum_negatives:
                rate = metrics.get("lightFalsePositiveRate")
                if rate is not None:
                    result.append(float(rate))
    return result


def calibration_reports(metric):
    """Return every calibration repetition under one stable metric shape."""
    if "perRepetition" in metric:
        reports = metric["perRepetition"]
        if not reports:
            raise RuntimeError("empty_target_calibration_repetitions")
        return reports
    return [metric]


def group_metric_values(
        report,
        field,
        metric,
        count_key,
        minimum_count,
):
    """Read one metric from known, sufficiently sampled groups."""
    values = []
    for name, metrics in report_group_map(report, field).items():
        normalized = str(name or "").strip().lower()
        if normalized in UNVERIFIED_GROUP_VALUES:
            continue
        if metrics.get(count_key, 0) < minimum_count:
            continue
        value = metrics.get(metric)
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    return values


def worst_group_metric(reports, field, metric, count_key, minimum_count):
    """Return the worst observed value or None when evidence is missing."""
    values = [
        value
        for report in reports
        for value in group_metric_values(
            report,
            field,
            metric,
            count_key,
            minimum_count,
        )
    ]
    return min(values) if values else None


def worst_group_false_positive_rate(reports):
    """Return the highest observed group FPR under one frozen threshold."""
    values = [
        value
        for report in reports
        for value in group_metric_values(
            report,
            "deviceName",
            "lightFalsePositiveRate",
            "negativeCount",
            MIN_GROUP_NEGATIVES,
        )
    ]
    return max(values) if values else None


def paired_group_coverage(reports):
    """Count known groups containing both positive and negative evidence."""
    result = {}
    # 中文注释：对外的 bodyPosture 键保持兼容，实际读取统一姿态域，
    # 这样倾斜手持等 taskId 派生组也能进入发布证据。
    field_sources = (
        ("deviceName", "deviceName"),
        ("bodyPosture", "postureDomain"),
        ("motionState", "motionState"),
    )
    for output_field, field in field_sources:
        positive_names = set()
        negative_names = set()
        for report in reports:
            for name, metrics in report_group_map(report, field).items():
                normalized = str(name or "").strip().lower()
                if normalized in UNVERIFIED_GROUP_VALUES:
                    continue
                if metrics.get("positiveCount", 0) >= MIN_GROUP_POSITIVES:
                    positive_names.add(str(name))
                if metrics.get("negativeCount", 0) >= MIN_GROUP_NEGATIVES:
                    negative_names.add(str(name))
        result[output_field] = {
            "positiveGroupCount": len(positive_names),
            "negativeGroupCount": len(negative_names),
            "pairedGroupCount": len(positive_names & negative_names),
        }
    return result


def evaluate_deployment_readiness(summary, target=DEPLOYMENT_PRIMARY_FPR):
    """Apply the strict product release gate without averaging away weak groups."""
    blockers = []
    seed_count = int(summary.get("seedCount", 0))
    if seed_count < DEPLOYMENT_MIN_SEEDS:
        blockers.append("insufficient_seed_evidence")

    recall_fields = {
        "deviceName": "worstDeviceHeavyRecallByTargetFpr",
        "bodyPosture": "worstPostureHeavyRecallByTargetFpr",
        "motionState": "worstMotionHeavyRecallByTargetFpr",
    }
    required_counts = {
        "deviceName": DEPLOYMENT_MIN_DEVICES,
        "bodyPosture": DEPLOYMENT_MIN_POSTURES,
        "motionState": DEPLOYMENT_MIN_MOTION_STATES,
    }
    coverage = summary.get("pairedGroupCoverageByTargetFpr", {}).get(
        target,
        {},
    )
    for field, key in recall_fields.items():
        field_coverage = coverage.get(field, {})
        paired_count = int(field_coverage.get("pairedGroupCount", 0))
        if paired_count < required_counts[field]:
            blockers.append(field + "_evidence_incomplete")
            continue
        value = summary.get(key, {}).get(target)
        if value is None:
            blockers.append(field + "_recall_unavailable")
        elif float(value) < DEPLOYMENT_RECALL_TARGET:
            blockers.append(field + "_recall_below_target")

    worst_fpr = summary.get(
        "worstGroupFalsePositiveRateByTargetFpr",
        {},
    ).get(target)
    if worst_fpr is None:
        blockers.append("group_false_positive_rate_unavailable")
    elif float(worst_fpr) > float(target) + 1e-12:
        blockers.append("group_false_positive_rate_above_target")

    if not summary.get("proofReadyByTargetFpr", {}).get(target, False):
        blockers.append("negative_evidence_not_proof_ready")

    return {
        "eligible": not blockers,
        "targetFalsePositiveRate": float(target),
        "minimumHeavyRecall": DEPLOYMENT_RECALL_TARGET,
        "minimumSeedCount": DEPLOYMENT_MIN_SEEDS,
        "requiredPairedGroups": required_counts,
        "worstGroupHeavyRecallByDimension": {
            field: summary.get(key, {}).get(target)
            for field, key in recall_fields.items()
        },
        "pairedGroupCoverage": coverage,
        "worstGroupFalsePositiveRate": worst_fpr,
        "proofReady": bool(
            summary.get("proofReadyByTargetFpr", {}).get(target, False)
        ),
        "blockers": blockers,
    }


def deployment_gate(summary, target=DEPLOYMENT_PRIMARY_FPR):
    """Keep the shorter public name while the report uses the explicit one."""
    return evaluate_deployment_readiness(summary, target)


def summarize_runs(runs, model_name):
    result = {}
    for fold_kind in sorted({run["foldKind"] for run in runs}):
        kind_runs = [
            run for run in runs
            if run["foldKind"] == fold_kind
        ]
        fold_ids = sorted({run["foldId"] for run in kind_runs})
        fold_recalls = {}
        for fold_id in fold_ids:
            selected = [
                run for run in kind_runs
                if run["foldId"] == fold_id
            ]
            fold_recalls[fold_id] = {
                "runIds": [run["foldId"] for run in selected],
                "recalls": {
                    target: finite_values([
                        run["fixedFprMetrics"][target]["overall"][
                            "heavyRecall"
                        ]
                        for run in selected
                    ])
                    for target in map(str, TARGET_FPRS)
                },
            }
        reports_by_target = {
            target: [
                run["fixedFprMetrics"][target]
                for run in kind_runs
            ]
            for target in map(str, TARGET_FPRS)
        }
        entry = {
            "foldCount": len(fold_ids),
            "runCount": len(kind_runs),
            # 中文注释：兼容历史报告中没有 seed 字段的旧 fixture；
            # 新训练结果始终显式写入 seed，旧数据不能因此阻断汇总。
            "seedCount": len({
                run.get("seed", 0) for run in kind_runs
            }),
            "folds": {
                fold_id: {
                    "foldId": fold_id,
                    "meanHeavyRecallByTargetFpr": {
                        target: (
                            float(np.mean(values))
                            if values
                            else None
                        )
                        for target, values in recalls["recalls"].items()
                    },
                }
                for fold_id, recalls in fold_recalls.items()
            },
            "meanHeavyRecallByTargetFpr": {
                target: float(np.mean([
                    run["fixedFprMetrics"][target]["overall"]["heavyRecall"]
                    for run in kind_runs
                ]))
                for target in map(str, TARGET_FPRS)
            },
            "worstFoldHeavyRecallByTargetFpr": {
                target: float(min(
                    np.mean(values)
                    for values in (
                        fold_recalls[fold_id]["recalls"][target]
                        for fold_id in fold_ids
                    )
                    if values
                ))
                for target in map(str, TARGET_FPRS)
            },
        "meanOracleHeavyRecallByTargetFpr": {
            target: float(np.mean([
                run.get("fixedFprOracleDiagnostics", {})[target][
                    "overall"
                ]["heavyRecall"]
                for run in kind_runs
                if target in run.get("fixedFprOracleDiagnostics", {})
            ])) if any(
                target in run.get("fixedFprOracleDiagnostics", {})
                for run in kind_runs
            ) else None
            for target in map(str, TARGET_FPRS)
        },
        "worstFoldOracleHeavyRecallByTargetFpr": {
            target: float(min(
                np.mean([
                    run["fixedFprOracleDiagnostics"][target][
                        "overall"
                    ]["heavyRecall"]
                    for run in kind_runs
                    if run["foldId"] == fold_id
                    and target in run.get(
                        "fixedFprOracleDiagnostics",
                        {},
                    )
                ])
                for fold_id in fold_ids
                if any(
                    run["foldId"] == fold_id
                    and target in run.get(
                        "fixedFprOracleDiagnostics",
                        {},
                    )
                    for run in kind_runs
                )
            )) if any(
                target in run.get("fixedFprOracleDiagnostics", {})
                for run in kind_runs
            ) else None
            for target in map(str, TARGET_FPRS)
        },
            "worstDeviceHeavyRecallByTargetFpr": {
                target: report_group_extreme(
                    reports_by_target[target],
                    "deviceName",
                    "heavyRecall",
                    "positiveCount",
                    MIN_GROUP_POSITIVES,
                )
                for target in map(str, TARGET_FPRS)
            },
            "worstPostureHeavyRecallByTargetFpr": {
                target: report_group_extreme(
                    reports_by_target[target],
                    "postureDomain",
                    "heavyRecall",
                    "positiveCount",
                    MIN_GROUP_POSITIVES,
                )
                for target in map(str, TARGET_FPRS)
            },
            "worstMotionHeavyRecallByTargetFpr": {
                target: report_group_extreme(
                    reports_by_target[target],
                    "motionState",
                    "heavyRecall",
                    "positiveCount",
                    MIN_GROUP_POSITIVES,
                )
                for target in map(str, TARGET_FPRS)
            },
            "worstGroupHeavyRecallByTargetFpr": {
                target: (
                    float(min(field_values))
                    if (field_values := finite_values([
                        report_group_extreme(
                            [
                                run["fixedFprMetrics"][target]
                                for run in kind_runs
                            ],
                            field,
                            "heavyRecall",
                            "positiveCount",
                            MIN_GROUP_POSITIVES,
                        )
                        for field in (
                            "deviceName",
                            "postureDomain",
                            "motionState",
                        )
                    ]))
                    else None
                )
                for target in map(str, TARGET_FPRS)
            },
            "pairedGroupCoverageByTargetFpr": {
                target: paired_group_coverage(reports_by_target[target])
                for target in map(str, TARGET_FPRS)
            },
            "worstGroupFalsePositiveRateByTargetFpr": {
                target: (
                    float(max(values))
                    if (values := finite_values([
                        worst_group_metric(
                            reports_by_target[target],
                            field,
                            "lightFalsePositiveRate",
                            "negativeCount",
                            MIN_GROUP_NEGATIVES,
                        )
                        for field in (
                            "deviceName",
                            "postureDomain",
                            "motionState",
                        )
                    ]))
                    else None
                )
                for target in map(str, TARGET_FPRS)
            },
            "proofReadyByTargetFpr": {
                target: bool(all(
                    report.get("proofReady", False)
                    for report in reports_by_target[target]
                ))
                for target in map(str, TARGET_FPRS)
            },
            "meanRuntimeParameterCount": float(np.mean([
                run["runtimeParameterCount"] for run in kind_runs
            ])),
            "meanTrainingParameterCount": float(np.mean([
                run["trainingParameterCount"] for run in kind_runs
            ])),
        }
        entry["deploymentGate"] = deployment_gate(entry)
        result[fold_kind] = entry
    return result


def summarize_evaluation_only_folds(runs):
    """Retain fold-level external recall and actual false-positive rates."""
    selected_runs = [
        run for run in runs
        if run.get("evaluationOnlyMetrics")
    ]
    result = {}
    for fold_kind in sorted({run["foldKind"] for run in selected_runs}):
        kind_runs = [
            run for run in selected_runs
            if run["foldKind"] == fold_kind
        ]
        result[fold_kind] = {}
        for fold_id in sorted({run["foldId"] for run in kind_runs}):
            fold_runs = [
                run for run in kind_runs
                if run["foldId"] == fold_id
            ]
            result[fold_kind][fold_id] = {
                target: {
                    "meanHeavyRecall": float(np.mean([
                        run["evaluationOnlyMetrics"][target]["overall"][
                            "heavyRecall"
                        ]
                        for run in fold_runs
                    ])),
                    "minimumHeavyRecall": float(min(
                        run["evaluationOnlyMetrics"][target]["overall"][
                            "heavyRecall"
                        ]
                        for run in fold_runs
                    )),
                    "maximumHeavyRecall": float(max(
                        run["evaluationOnlyMetrics"][target]["overall"][
                            "heavyRecall"
                        ]
                        for run in fold_runs
                    )),
                    "meanFalsePositiveRate": float(np.mean([
                        run["evaluationOnlyMetrics"][target]["overall"][
                            "lightFalsePositiveRate"
                        ]
                        for run in fold_runs
                    ])),
                    "minimumFalsePositiveRate": float(min(
                        run["evaluationOnlyMetrics"][target]["overall"][
                            "lightFalsePositiveRate"
                        ]
                        for run in fold_runs
                    )),
                    "maximumFalsePositiveRate": float(max(
                        run["evaluationOnlyMetrics"][target]["overall"][
                            "lightFalsePositiveRate"
                        ]
                        for run in fold_runs
                    )),
                    "negativeCount": int(min(
                        run["evaluationOnlyMetrics"][target]["overall"][
                            "negativeCount"
                        ]
                        for run in fold_runs
                    )),
                    "minimumProofNegativeCount": int(
                        np.ceil(5.0 / float(target))
                    ),
                    "calibrationNegativeCount": int(min(
                        run["evaluationOnlyMetrics"][target].get(
                            "calibrationNegativeCount",
                            0,
                        )
                        for run in fold_runs
                    )),
                    "heldOutNegativeCount": int(min(
                        run["evaluationOnlyMetrics"][target].get(
                            "heldOutNegativeCount",
                            run["evaluationOnlyMetrics"][target]["overall"][
                                "negativeCount"
                            ],
                        )
                        for run in fold_runs
                    )),
                    "minimumProofSessionCount": MIN_PROOF_SESSION_COUNT,
                    "calibrationSessionCount": int(min(
                        run["evaluationOnlyMetrics"][target].get(
                            "calibrationSessionCount",
                            0,
                        )
                        for run in fold_runs
                    )),
                    "heldOutNegativeSessionCount": int(min(
                        run["evaluationOnlyMetrics"][target].get(
                            "heldOutNegativeSessionCount",
                            0,
                        )
                        for run in fold_runs
                    )),
                    "heldOutPositiveCount": int(min(
                        run["evaluationOnlyMetrics"][target].get(
                            "heldOutPositiveCount",
                            run["evaluationOnlyMetrics"][target]["overall"]
                            .get("positiveCount", 0),
                        )
                        for run in fold_runs
                    )),
                    "proofReady": bool(all(
                        run["evaluationOnlyMetrics"][target].get(
                            "calibrationEvidenceStatus"
                        ) == "proof_ready"
                        and run["evaluationOnlyMetrics"][target].get(
                            "calibrationNegativeCount",
                            0,
                        ) >= np.ceil(5.0 / float(target))
                        and run["evaluationOnlyMetrics"][target].get(
                            "heldOutNegativeCount",
                            run["evaluationOnlyMetrics"][target]["overall"][
                                "negativeCount"
                            ],
                        ) >= np.ceil(5.0 / float(target))
                        and run["evaluationOnlyMetrics"][target].get(
                            "calibrationSessionCount",
                            0,
                        ) >= MIN_PROOF_SESSION_COUNT
                        and run["evaluationOnlyMetrics"][target].get(
                            "heldOutNegativeSessionCount",
                            0,
                        ) >= MIN_PROOF_SESSION_COUNT
                        and run["evaluationOnlyMetrics"][target].get(
                            "heldOutPositiveCount",
                            run["evaluationOnlyMetrics"][target]["overall"]
                            .get("positiveCount", 0),
                        ) >= MIN_GROUP_POSITIVES
                        for run in fold_runs
                    )),
                }
                for target in map(str, TARGET_FPRS)
            }
    return result

def summarize_ordinary_click_audits(runs):
    """Aggregate ordinary-tap acceptance without affecting model selection."""
    selected_runs = [
        run for run in runs
        if run.get("ordinaryClickAudit")
    ]
    if not selected_runs:
        return {}
    result = {}
    for fold_kind in sorted({run["foldKind"] for run in selected_runs}):
        kind_runs = [
            run for run in selected_runs
            if run["foldKind"] == fold_kind
        ]
        result[fold_kind] = {}
        for target in map(str, TARGET_FPRS):
            reports = [
                run["ordinaryClickAudit"][target]
                for run in kind_runs
            ]
            acceptance_rates = finite_values([
                report["ordinaryClickAcceptanceRate"]
                for report in reports
            ])
            device_rates = finite_values([
                metrics["ordinaryClickAcceptanceRate"]
                for report in reports
                for metrics in report["groups"]["deviceName"].values()
                if metrics["negativeCount"] >= MIN_GROUP_NEGATIVES
            ])
            posture_rates = finite_values([
                metrics["ordinaryClickAcceptanceRate"]
                for report in reports
                for metrics in report_group_map(
                    report,
                    "postureDomain",
                ).values()
                if metrics["negativeCount"] >= MIN_GROUP_NEGATIVES
            ])
            motion_rates = finite_values([
                metrics["ordinaryClickAcceptanceRate"]
                for report in reports
                for metrics in report["groups"]["motionState"].values()
                if metrics["negativeCount"] >= MIN_GROUP_NEGATIVES
            ])
            result[fold_kind][target] = {
                "runCount": len(reports),
                "sampleCount": int(min(
                    report["overall"]["negativeCount"]
                    for report in reports
                )),
                "meanAcceptanceRate": (
                    float(np.mean(acceptance_rates))
                    if acceptance_rates
                    else None
                ),
                "worstAcceptanceRate": (
                    float(max(acceptance_rates))
                    if acceptance_rates
                    else None
                ),
                "worstDeviceAcceptanceRate": (
                    float(max(device_rates))
                    if device_rates
                    else None
                ),
                "worstPostureAcceptanceRate": (
                    float(max(posture_rates))
                    if posture_rates
                    else None
                ),
                "worstMotionAcceptanceRate": (
                    float(max(motion_rates))
                    if motion_rates
                    else None
                ),
                "minimumProofNegativeCount": int(
                    np.ceil(5.0 / float(target))
                ),
                "proofReady": bool(all(
                    report.get("proofReady", False)
                    for report in reports
                )),
            }
    return result


def summarize_evaluation_only_runs(runs):
    """Aggregate external metrics without allowing them to select a model."""
    selected_runs = [
        run for run in runs
        if run.get("evaluationOnlyMetrics")
    ]
    if not selected_runs:
        return {}
    result = {}
    for fold_kind in sorted({run["foldKind"] for run in selected_runs}):
        kind_runs = [
            run for run in selected_runs
            if run["foldKind"] == fold_kind
        ]
        fold_metrics = summarize_evaluation_only_folds(kind_runs)[fold_kind]
        reports_by_target = {
            target: [
                run["evaluationOnlyMetrics"][target]
                for run in kind_runs
            ]
            for target in map(str, TARGET_FPRS)
        }
        worst_device_heavy_recall = {
            target: report_group_extreme(
                reports_by_target[target],
                "deviceName",
                "heavyRecall",
                "positiveCount",
                MIN_GROUP_POSITIVES,
            )
            for target in map(str, TARGET_FPRS)
        }
        worst_posture_heavy_recall = {
            target: report_group_extreme(
                reports_by_target[target],
                "postureDomain",
                "heavyRecall",
                "positiveCount",
                MIN_GROUP_POSITIVES,
            )
            for target in map(str, TARGET_FPRS)
        }
        worst_motion_heavy_recall = {
            target: report_group_extreme(
                reports_by_target[target],
                "motionState",
                "heavyRecall",
                "positiveCount",
                MIN_GROUP_POSITIVES,
            )
            for target in map(str, TARGET_FPRS)
        }
        worst_group_heavy_recall = {
            target: (
                float(min(field_values))
                if (field_values := finite_values([
                    worst_device_heavy_recall[target],
                    worst_posture_heavy_recall[target],
                    worst_motion_heavy_recall[target],
                ]))
                else None
            )
            for target in map(str, TARGET_FPRS)
        }
        worst_group_false_positive_rate = {
            target: (
                float(max(values))
                if (values := finite_values([
                    worst_group_metric(
                        reports_by_target[target],
                        field,
                        "lightFalsePositiveRate",
                        "negativeCount",
                        MIN_GROUP_NEGATIVES,
                    )
                    for field in (
                        "deviceName",
                        "postureDomain",
                        "motionState",
                    )
                ]))
                else None
            )
            for target in map(str, TARGET_FPRS)
        }
        proof_ready = {
            target: bool(all(
                metrics[target]["proofReady"]
                for metrics in fold_metrics.values()
            ))
            for target in map(str, TARGET_FPRS)
        }
        validated_group_names = {
            target: {
                field: report_group_names(
                    reports_by_target[target],
                    field,
                    (
                        "positiveCount"
                        if field != "deviceName"
                        else "positiveCount"
                    ),
                    MIN_GROUP_POSITIVES,
                )
                for field in (
                    "deviceName",
                    "postureDomain",
                    "motionState",
                )
            }
            for target in map(str, TARGET_FPRS)
        }
        entry = {
            "runCount": len(kind_runs),
            "foldCount": len(fold_metrics),
            "seedCount": len({run["seed"] for run in kind_runs}),
            "meanHeavyRecallByTargetFpr": {
                target: float(np.mean([
                    metrics[target]["meanHeavyRecall"]
                    for metrics in fold_metrics.values()
                ]))
                for target in map(str, TARGET_FPRS)
            },
            "meanFalsePositiveRateByTargetFpr": {
                target: float(np.mean([
                    run["evaluationOnlyMetrics"][target]["overall"][
                        "lightFalsePositiveRate"
                    ]
                    for run in kind_runs
                ]))
                for target in map(str, TARGET_FPRS)
            },
            "worstFoldFalsePositiveRateByTargetFpr": {
                target: float(max(
                    metrics[target]["meanFalsePositiveRate"]
                    for metrics in fold_metrics.values()
                ))
                for target in map(str, TARGET_FPRS)
            },
            "worstGroupFalsePositiveRateByTargetFpr": (
                worst_group_false_positive_rate
            ),
            "worstRunFalsePositiveRateByTargetFpr": {
                target: float(max(
                    run["evaluationOnlyMetrics"][target]["overall"][
                        "lightFalsePositiveRate"
                    ]
                    for run in kind_runs
                ))
                for target in map(str, TARGET_FPRS)
            },
            "proofReadyByTargetFpr": proof_ready,
            "minimumProofNegativeCountByTargetFpr": {
                target: int(max(
                    metrics[target]["minimumProofNegativeCount"]
                    for metrics in fold_metrics.values()
                ))
                for target in map(str, TARGET_FPRS)
            },
            "worstFoldHeavyRecallByTargetFpr": {
                target: float(min(
                    metrics[target]["meanHeavyRecall"]
                    for metrics in fold_metrics.values()
                ))
                for target in map(str, TARGET_FPRS)
            },
            "worstGroupHeavyRecallByTargetFpr": worst_group_heavy_recall,
            "worstDeviceHeavyRecallByTargetFpr": (
                worst_device_heavy_recall
            ),
            "worstPostureHeavyRecallByTargetFpr": (
                worst_posture_heavy_recall
            ),
            "worstMotionHeavyRecallByTargetFpr": (
                worst_motion_heavy_recall
            ),
            "validatedGroupNamesByTargetFpr": validated_group_names,
            "validatedGroupCountsByTargetFpr": {
                target: {
                    field: len(names)
                    for field, names in fields.items()
                }
                for target, fields in validated_group_names.items()
            },
            "pairedGroupCoverageByTargetFpr": {
                target: paired_group_coverage(reports_by_target[target])
                for target in map(str, TARGET_FPRS)
            },
            "worstRunHeavyRecallByTargetFpr": {
                target: float(min(
                    run["evaluationOnlyMetrics"][target]["overall"][
                        "heavyRecall"
                    ]
                    for run in kind_runs
                ))
                for target in map(str, TARGET_FPRS)
            },
            "worstSeedHeavyRecallByTargetFpr": {
                target: float(min(
                    np.mean([
                        run["evaluationOnlyMetrics"][target]["overall"][
                            "heavyRecall"
                        ]
                        for run in kind_runs
                        if run["seed"] == seed
                    ])
                    for seed in sorted({run["seed"] for run in kind_runs})
                ))
                for target in map(str, TARGET_FPRS)
            },
            "foldMetrics": fold_metrics,
        }
        entry["deploymentGate"] = deployment_gate(entry)
        result[fold_kind] = entry
    return result

def summarize_target_calibration_runs(runs):
    """Aggregate held-out target calibration without mixing threshold policies."""
    counts = sorted({
        str(count)
        for run in runs
        for count in run.get("targetCalibratedMetrics", {})
    }, key=int)
    result = {}
    for count in counts:
        selected_runs = [
            run for run in runs
            if count in run.get("targetCalibratedMetrics", {})
        ]
        result[count] = {}
        for fold_kind in sorted({run["foldKind"] for run in selected_runs}):
            kind_runs = [
                run for run in selected_runs
                if run["foldKind"] == fold_kind
            ]
            fold_ids = sorted({run["foldId"] for run in kind_runs})
            fold_recalls = {}
            fold_metrics = {}
            for fold_id in fold_ids:
                selected = [
                    run for run in kind_runs
                    if run["foldId"] == fold_id
                ]
                fold_recalls[fold_id] = {
                    target: finite_values([
                        report["overall"]["heavyRecall"]
                        for run in selected
                        for report in calibration_reports(
                            run["targetCalibratedMetrics"][count][target]
                        )
                    ])
                    for target in map(str, TARGET_FPRS)
                }
                fold_recalls[fold_id]["falsePositiveRates"] = {
                    target: finite_values([
                        report["overall"]["lightFalsePositiveRate"]
                        for run in selected
                        for report in calibration_reports(
                            run["targetCalibratedMetrics"][count][target]
                        )
                    ])
                    for target in map(str, TARGET_FPRS)
                }
                fold_metrics[fold_id] = {
                    "foldId": fold_id,
                    "meanHeavyRecallByTargetFpr": {
                        target: (
                            float(np.mean(
                                fold_recalls[fold_id][target]
                            ))
                            if fold_recalls[fold_id][target]
                            else None
                        )
                        for target in map(str, TARGET_FPRS)
                    },
                }
                for target in map(str, TARGET_FPRS):
                    recalls = fold_recalls[fold_id][target]
                    false_positive_rates = fold_recalls[fold_id][
                        "falsePositiveRates"
                    ][target]
                    device_false_positive_rates = finite_values([
                        metrics.get("lightFalsePositiveRate")
                        for run in selected
                        for report in calibration_reports(
                            run["targetCalibratedMetrics"][count][target]
                        )
                        for metrics in report["groups"][
                            "deviceName"
                        ].values()
                        if (
                            metrics.get("negativeCount", 0)
                            >= MIN_GROUP_NEGATIVES
                        )
                    ])
                    fold_metrics[fold_id][target] = {
                        "meanHeavyRecall": (
                            float(np.mean(recalls))
                            if recalls
                            else None
                        ),
                        "minimumHeavyRecall": (
                            float(min(recalls))
                            if recalls
                            else None
                        ),
                        "maximumHeavyRecall": (
                            float(max(recalls))
                            if recalls
                            else None
                        ),
                        "meanHeldOutFalsePositiveRate": (
                            float(np.mean(false_positive_rates))
                            if false_positive_rates
                            else None
                        ),
                        "maximumHeldOutFalsePositiveRate": (
                            float(max(false_positive_rates))
                            if false_positive_rates
                            else None
                        ),
                        "worstDeviceHeldOutFalsePositiveRate": (
                            float(max(device_false_positive_rates))
                            if device_false_positive_rates
                            else None
                        ),
                        "heldOutNegativeCount": int(min(
                            report.get(
                                "heldOutNegativeCount",
                                report["overall"].get("negativeCount", 0),
                            )
                            for run in selected
                            for report in calibration_reports(
                                run["targetCalibratedMetrics"][count][
                                    target
                                ]
                            )
                        )),
                        "heldOutPositiveCount": int(min(
                            report.get(
                                "heldOutPositiveCount",
                                report["overall"].get("positiveCount", 0),
                            )
                            for run in selected
                            for report in calibration_reports(
                                run["targetCalibratedMetrics"][count][
                                    target
                                ]
                            )
                        )),
                        "minimumProofSessionCount": (
                            MIN_PROOF_SESSION_COUNT
                        ),
                        "calibrationSessionCount": int(min(
                            report.get("calibrationSessionCount", 0)
                            for run in selected
                            for report in calibration_reports(
                                run["targetCalibratedMetrics"][count][
                                    target
                                ]
                            )
                        )),
                        "heldOutNegativeSessionCount": int(min(
                            report.get(
                                "heldOutNegativeSessionCount",
                                0,
                            )
                            for run in selected
                            for report in calibration_reports(
                                run["targetCalibratedMetrics"][count][
                                    target
                                ]
                            )
                        )),
                        "proofReady": bool(all(
                            report.get("proofReady", False)
                            for run in selected
                            for report in calibration_reports(
                                run["targetCalibratedMetrics"][count][
                                    target
                                ]
                            )
                        )),
                        "minimumProofNegativeCount": int(min(
                            report.get("minimumProofNegativeCount", 0)
                            for run in selected
                            for report in calibration_reports(
                                run["targetCalibratedMetrics"][count][
                                    target
                                ]
                            )
                        )),
                    }
            result[count][fold_kind] = {
                "foldCount": len(fold_ids),
                "runCount": len(kind_runs),
                "meanHeavyRecallByTargetFpr": {
                    target: float(np.mean([
                        report["overall"]["heavyRecall"]
                        for run in kind_runs
                        for report in calibration_reports(
                            run["targetCalibratedMetrics"][count][target]
                        )
                    ]))
                    for target in map(str, TARGET_FPRS)
                },
                "worstFoldHeavyRecallByTargetFpr": {
                    target: float(min(
                        np.mean(values)
                        for values in (
                            fold_recalls[fold_id][target]
                            for fold_id in fold_ids
                        )
                        if values
                    ))
                    for target in map(str, TARGET_FPRS)
                },
                "meanHeldOutFalsePositiveRateByTargetFpr": {
                    target: float(np.mean([
                        report["overall"]["lightFalsePositiveRate"]
                        for run in kind_runs
                        for report in calibration_reports(
                            run["targetCalibratedMetrics"][count][target]
                        )
                    ]))
                    for target in map(str, TARGET_FPRS)
                },
                "worstFoldHeldOutFalsePositiveRateByTargetFpr": {
                    target: float(max(
                        np.mean(values)
                        for values in (
                            fold_recalls[fold_id][
                                "falsePositiveRates"
                            ][target]
                            for fold_id in fold_ids
                        )
                        if values
                    ))
                    for target in map(str, TARGET_FPRS)
                },
                "worstGroupHeldOutFalsePositiveRateByTargetFpr": {
                    target: (
                        float(max(rates))
                        if rates
                        else None
                    )
                    for target in map(str, TARGET_FPRS)
                    for rates in [finite_values([
                        rate
                        for run in kind_runs
                        for report in calibration_reports(
                            run["targetCalibratedMetrics"][count][target]
                        )
                        for rate in group_false_positive_rate_values(report)
                    ])]
                },
                "proofReadyByTargetFpr": {
                    target: bool(all(
                        report.get("proofReady", False)
                        for run in kind_runs
                        for report in calibration_reports(
                            run["targetCalibratedMetrics"][count][target]
                        )
                    ))
                    for target in map(str, TARGET_FPRS)
                },
                "calibrationEvidenceStatus": {
                    target: (
                        "proof_ready"
                        if all(
                            report.get("proofReady", False)
                            for run in kind_runs
                            for report in calibration_reports(
                                run["targetCalibratedMetrics"][count][
                                    target
                                ]
                            )
                        )
                        else "descriptive_only"
                    )
                    for target in map(str, TARGET_FPRS)
                },
                "worstHeldOutFalsePositiveRateByTargetFpr": {
                    target: (
                        float(max(rates))
                        if rates
                        else None
                    )
                    for target in map(str, TARGET_FPRS)
                    for rates in [finite_values([
                        report["overall"]["lightFalsePositiveRate"]
                        for run in kind_runs
                        for report in calibration_reports(
                            run["targetCalibratedMetrics"][count][target]
                        )
                    ])]
                },
                "worstDeviceHeldOutFalsePositiveRateByTargetFpr": {
                    target: (
                        float(max(rates))
                        if rates
                        else None
                    )
                    for target in map(str, TARGET_FPRS)
                    for rates in [finite_values([
                        metrics.get("lightFalsePositiveRate")
                        for run in kind_runs
                        for report in calibration_reports(
                            run["targetCalibratedMetrics"][count][target]
                        )
                        for metrics in report["groups"][
                            "deviceName"
                        ].values()
                        if (
                            metrics.get("negativeCount", 0)
                            >= MIN_GROUP_NEGATIVES
                        )
                    ])]
                },
                "minimumProofNegativeCountByTargetFpr": {
                    target: int(min(
                        report.get("minimumProofNegativeCount", 0)
                        for run in kind_runs
                        for report in calibration_reports(
                            run["targetCalibratedMetrics"][count][target]
                        )
                    ))
                    for target in map(str, TARGET_FPRS)
                },
                "worstGroupHeavyRecallByTargetFpr": {
                    target: float(min([
                        recall
                        for run in kind_runs
                        for report in calibration_reports(
                            run["targetCalibratedMetrics"][count][target]
                        )
                        for recall in group_recall_values(report)
                    ]))
                    for target in map(str, TARGET_FPRS)
                },
                "worstSeedHeavyRecallByTargetFpr": {
                    target: float(min(
                        np.mean([
                            run["targetCalibratedMetrics"][count][target][
                                "overall"
                            ]["heavyRecall"]
                            for run in kind_runs
                            if run["seed"] == seed
                        ])
                        for seed in sorted({
                            run["seed"] for run in kind_runs
                        })
                    ))
                    for target in map(str, TARGET_FPRS)
                },
                "seedRecallRangeByTargetFpr": {
                    target: {
                        "min": float(min(
                            np.mean([
                                report["overall"]["heavyRecall"]
                                for run in kind_runs
                                if run["seed"] == seed
                                for report in calibration_reports(
                                    run["targetCalibratedMetrics"][count][
                                        target
                                    ]
                                )
                            ])
                            for seed in sorted({
                                run["seed"] for run in kind_runs
                            })
                        )),
                        "max": float(max(
                            np.mean([
                                report["overall"]["heavyRecall"]
                                for run in kind_runs
                                if run["seed"] == seed
                                for report in calibration_reports(
                                    run["targetCalibratedMetrics"][count][
                                        target
                                    ]
                                )
                            ])
                            for seed in sorted({
                                run["seed"] for run in kind_runs
                            })
                        )),
                        "maxMinusMin": float(
                            max(
                                np.mean([
                                    report["overall"]["heavyRecall"]
                                    for run in kind_runs
                                    if run["seed"] == seed
                                    for report in calibration_reports(
                                        run["targetCalibratedMetrics"][count][
                                            target
                                        ]
                                    )
                                ])
                                for seed in sorted({
                                    run["seed"] for run in kind_runs
                                })
                            )
                            - min(
                                np.mean([
                                    report["overall"]["heavyRecall"]
                                    for run in kind_runs
                                    if run["seed"] == seed
                                    for report in calibration_reports(
                                        run["targetCalibratedMetrics"][count][
                                            target
                                        ]
                                    )
                                ])
                                for seed in sorted({
                                    run["seed"] for run in kind_runs
                                })
                            )
                        ),
                    }
                    for target in map(str, TARGET_FPRS)
                },
                "foldMetrics": fold_metrics,
            }
    return result


def summarize_threshold_calibration_sweep(runs, false_positive_limit):
    """Rank nominal thresholds after enforcing the measured FPR limit."""
    grouped = defaultdict(list)
    for run in runs:
        for report in run.get("fixedFprMetrics", {}).values():
            for calibration_fpr, sweep in report.get(
                    "thresholdCalibrationSweep",
                    {}).items():
                grouped[float(calibration_fpr)].append(sweep)
    result = {}
    for calibration_fpr, reports in sorted(grouped.items()):
        overall_recalls = finite_values([
            report["overall"]["heavyRecall"] for report in reports
        ])
        overall_fprs = finite_values([
            report["overall"]["lightFalsePositiveRate"]
            for report in reports
        ])
        device_recalls = {}
        device_false_positive_rates = {}
        for report in reports:
            for device, metrics in report["groups"]["deviceName"].items():
                recall = metrics.get("heavyRecall")
                if recall is not None:
                    device_recalls.setdefault(device, []).append(
                        float(recall)
                    )
                false_positive_rate = metrics.get(
                    "lightFalsePositiveRate"
                )
                if false_positive_rate is not None:
                    device_false_positive_rates.setdefault(
                        device,
                        [],
                    ).append(float(false_positive_rate))
        mean_device_recall = (
            float(np.mean([
                np.mean(values)
                for values in device_recalls.values()
            ]))
            if device_recalls
            else None
        )
        worst_device_recall = (
            float(min(np.mean(values) for values in device_recalls.values()))
            if device_recalls
            else None
        )
        worst_overall_fpr = (
            float(max(overall_fprs)) if overall_fprs else None
        )
        mean_device_fprs = {
            device: float(np.mean(values))
            for device, values in device_false_positive_rates.items()
            if values
        }
        worst_device_fpr = (
            max(mean_device_fprs.values())
            if mean_device_fprs
            else None
        )
        false_positive_limit_met = bool(
            worst_overall_fpr is not None
            and worst_device_fpr is not None
            and worst_overall_fpr < float(false_positive_limit)
            and worst_device_fpr < float(false_positive_limit)
        )
        result[str(calibration_fpr)] = {
            "calibrationFalsePositiveRate": calibration_fpr,
            "runCount": len(reports),
            "meanOverallHeavyRecall": (
                float(np.mean(overall_recalls))
                if overall_recalls
                else None
            ),
            "meanDeviceHeavyRecall": mean_device_recall,
            "worstDeviceHeavyRecall": worst_device_recall,
            "meanMeasuredFalsePositiveRate": (
                float(np.mean(overall_fprs)) if overall_fprs else None
            ),
            "worstMeasuredFalsePositiveRate": worst_overall_fpr,
            "meanDeviceFalsePositiveRates": mean_device_fprs,
            "worstDeviceFalsePositiveRate": worst_device_fpr,
            "falsePositiveLimit": float(false_positive_limit),
            "falsePositiveLimitMet": false_positive_limit_met,
        }
    return result


def compare_models(report):
    """Report context-head deltas at each fixed false-positive rate."""
    baseline = report["models"]["gate_no_context"]
    context = report["models"]["gate_context"]
    deltas = {}
    for fold_kind in sorted(set(baseline) & set(context)):
        deltas[fold_kind] = {}
        for target in map(str, TARGET_FPRS):
            base_value = baseline[fold_kind][
                "meanHeavyRecallByTargetFpr"
            ][target]
            context_value = context[fold_kind][
                "meanHeavyRecallByTargetFpr"
            ][target]
            deltas[fold_kind][target] = {
                "meanHeavyRecallDelta": context_value - base_value,
                "worstFoldHeavyRecallDelta": (
                    context[fold_kind][
                        "worstFoldHeavyRecallByTargetFpr"
                    ][target]
                    - baseline[fold_kind][
                        "worstFoldHeavyRecallByTargetFpr"
                    ][target]
                ),
                "worstGroupHeavyRecallDelta": (
                    context[fold_kind][
                        "worstGroupHeavyRecallByTargetFpr"
                    ][target]
                    - baseline[fold_kind][
                        "worstGroupHeavyRecallByTargetFpr"
                    ][target]
                ),
            }
    return deltas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        action="append",
        required=True,
        help="Training dataset directory; repeat to merge datasets.",
    )
    parser.add_argument(
        "--evaluation-only-dataset-dir",
        action="append",
        default=[],
        help="External dataset used only for forward evaluation.",
    )
    parser.add_argument(
        "--metadata-manifest",
        action="append",
        default=[],
        help="可信清单，仅补充派生训练集的实验块时间与采集者。",
    )
    parser.add_argument(
        "--ordinary-click-dataset-dir",
        action="append",
        default=[],
        help=(
            "Dataset directory containing ordinary_touch26.jsonl; "
            "ordinary taps are evaluation-only negatives."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--fold-kind",
        action="append",
        choices=("device", "block", "session", "posture"),
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--hidden-count", type=int, default=24)
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=20261022)
    parser.add_argument("--calibration-ratio", type=float, default=0.22)
    parser.add_argument("--context-loss-weight", type=float, default=0.08)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default="full",
    )
    parser.add_argument(
        "--feature-family",
        choices=FEATURE_FAMILIES,
        default="stats",
    )
    parser.add_argument(
        "--evidence-contract",
        choices=EVIDENCE_CONTRACTS,
        default=evidence_features.DEFAULT_EVIDENCE_CONTRACT,
        help=(
            "Versioned causal pre/post window for evidence features. "
            "legacy_post20 is retained only for A/B comparison."
        ),
    )
    parser.add_argument(
        "--weighting",
        choices=WEIGHTING_MODES,
        default="session_balanced",
    )
    parser.add_argument(
        "--modality-dropout",
        type=float,
        default=0.0,
        help="Probability of hiding optional touch-shape channels in training.",
    )
    parser.add_argument(
        "--augmentation-mode",
        choices=AUGMENTATION_MODES,
        default="none",
        help="Training-only physical channel gain randomization.",
    )
    parser.add_argument(
        "--augmentation-probability",
        type=float,
        default=0.5,
        help="Per-row, per-sensor-group probability of channel scaling.",
    )
    parser.add_argument(
        "--augmentation-max-log-scale",
        type=float,
        default=0.35,
        help="Maximum absolute natural-log gain applied during training.",
    )
    parser.add_argument(
        "--imu-mode",
        choices=IMU_MODES,
        default="raw",
    )
    parser.add_argument(
        "--adaptive-feature-mode",
        choices=ADAPTIVE_FEATURE_MODES,
        default="none",
    )
    parser.add_argument(
        "--normalization-mode",
        choices=NORMALIZATION_MODES,
        default="none",
        help=(
            "Unlabeled per-channel pretouch calibration; target history "
            "is causal and falls back to source statistics during cold start."
        ),
    )
    parser.add_argument(
        "--early-evidence-mode",
        choices=EARLY_EVIDENCE_MODES,
        default="none",
        help=(
            "Append causal early-press sidecar evidence after the standard "
            "feature scaler."
        ),
    )
    parser.add_argument(
        "--early-evidence-file",
        action="append",
        default=[],
        help="JSONL sidecar produced by build_early_press_evidence.py.",
    )
    parser.add_argument(
        "--finger-loss-weight",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--model-selection-mode",
        choices=MODEL_SELECTION_MODES,
        default="fixed_fpr_recall",
        help=(
            "Epoch selection objective; fixed_fpr_recall uses calibration "
            "domain recall at the primary target FPR."
        ),
    )
    parser.add_argument(
        "--selection-target-fpr",
        type=float,
        default=0.01,
        help="Target FPR used by fixed_fpr_recall model selection.",
    )
    parser.add_argument(
        "--threshold-calibration-fpr",
        type=float,
        default=DEFAULT_THRESHOLD_CALIBRATION_FPR,
        help=(
            "Nominal FPR used to fit the deployment threshold. A lower value "
            "adds margin against target-domain score shift."
        ),
    )
    parser.add_argument(
        "--false-positive-limit",
        type=float,
        default=DEFAULT_FALSE_POSITIVE_LIMIT,
        help="Maximum measured held-out FPR allowed for a ranked candidate.",
    )
    parser.add_argument(
        "--threshold-calibration-sweep",
        action="append",
        type=float,
        default=[],
        help="Additional nominal calibration FPRs reported without retraining.",
    )
    parser.add_argument(
        "--gate-loss-mode",
        choices=GATE_LOSS_MODES,
        default="cross_entropy",
        help=(
            "Strength training objective; pairwise_ranking directly "
            "optimizes heavy-above-light ordering."
        ),
    )
    parser.add_argument(
        "--ranking-loss-weight",
        type=float,
        default=1.0,
        help="Pairwise term weight when --gate-loss-mode=ce_pairwise.",
    )
    parser.add_argument(
        "--pairwise-margin",
        type=float,
        default=0.0,
        help="Required score margin between heavy and light pairs.",
    )
    parser.add_argument(
        "--positive-focal-weight",
        type=float,
        default=1.0,
        help="Class weight for heavy samples in asymmetric focal loss.",
    )
    parser.add_argument(
        "--negative-focal-weight",
        type=float,
        default=1.0,
        help="Class weight for light samples in asymmetric focal loss.",
    )
    parser.add_argument(
        "--target-calibration-negative-count",
        action="append",
        type=int,
        default=[],
        help="Known target-domain negative taps used for threshold calibration.",
    )
    parser.add_argument(
        "--target-calibration-repetitions",
        type=int,
        default=8,
        help="Independent target-negative subsets used for calibration stability.",
    )
    parser.add_argument(
        "--max-folds-per-kind",
        type=int,
        default=0,
        help="Zero evaluates every eligible fold.",
    )
    parser.add_argument(
        "--fold-id",
        action="append",
        default=[],
        help="Evaluate only the exact fold IDs provided; repeat for multiple.",
    )
    args = parser.parse_args()
    if (
        args.epochs <= 0
        or args.hidden_count <= 0
        or args.seed_count <= 0
        or args.patience <= 0
    ):
        raise RuntimeError("invalid_validation_budget")
    if not 0.0 < args.context_loss_weight <= 0.20:
        raise RuntimeError("invalid_context_loss_weight")
    if not 0.0 <= args.modality_dropout < 1.0:
        raise RuntimeError("invalid_modality_dropout")
    if not 0.0 <= args.augmentation_probability <= 1.0:
        raise RuntimeError("invalid_augmentation_probability")
    if not 0.0 < args.augmentation_max_log_scale <= 1.0:
        raise RuntimeError("invalid_augmentation_max_log_scale")
    if not 0.0 <= args.finger_loss_weight <= 0.50:
        raise RuntimeError("invalid_finger_loss_weight")
    if not 0.0 < args.selection_target_fpr < 1.0:
        raise RuntimeError("invalid_selection_target_fpr")
    if not 0.0 < args.threshold_calibration_fpr <= 0.01:
        raise RuntimeError("invalid_threshold_calibration_fpr")
    if not 0.0 < args.false_positive_limit <= 0.01:
        raise RuntimeError("invalid_false_positive_limit")
    if any(
            not 0.0 < value <= 0.01
            for value in args.threshold_calibration_sweep):
        raise RuntimeError("invalid_threshold_calibration_sweep")
    if not 0.0 <= args.ranking_loss_weight <= 10.0:
        raise RuntimeError("invalid_ranking_loss_weight")
    if not 0.0 <= args.pairwise_margin <= 10.0:
        raise RuntimeError("invalid_pairwise_margin")
    if not 0.0 <= args.positive_focal_weight <= 100.0:
        raise RuntimeError("invalid_positive_focal_weight")
    if not 0.0 <= args.negative_focal_weight <= 100.0:
        raise RuntimeError("invalid_negative_focal_weight")
    if any(count <= 0 for count in args.target_calibration_negative_count):
        raise RuntimeError("invalid_target_calibration_count")
    if args.target_calibration_repetitions <= 0:
        raise RuntimeError("invalid_target_calibration_repetitions")
    rows_by_hash = {}
    for index, dataset_dir in enumerate(args.dataset_dir):
        metadata_manifest = (
            args.metadata_manifest[index]
            if index < len(args.metadata_manifest)
            else None
        )
        loaded = load_rows(
            [dataset_dir],
            early_evidence_files=args.early_evidence_file,
            metadata_manifest=metadata_manifest,
        )
        for row in loaded:
            existing = rows_by_hash.get(row["sourceSha256"])
            if existing is None:
                rows_by_hash[row["sourceSha256"]] = row
            elif (
                existing.get("classLabel") != row.get("classLabel")
                or existing.get("matrix") != row.get("matrix")
            ):
                raise RuntimeError("conflicting_duplicate_training_source")
    rows = sorted(
        rows_by_hash.values(),
        key=lambda row: row["sourceSha256"],
    )
    attach_experiment_blocks(rows)
    evaluation_only_rows = load_evaluation_only_rows(
        args.evaluation_only_dataset_dir,
        rows,
        early_evidence_files=args.early_evidence_file,
    )
    ordinary_click_rows = load_ordinary_click_rows(
        args.ordinary_click_dataset_dir,
        rows,
        evaluation_only_rows,
    )
    requested_kinds = args.fold_kind or [
        "device",
        "session",
        "posture",
    ]
    if evaluation_only_rows and "device" in requested_kinds:
        folds = build_external_device_folds(rows, evaluation_only_rows)
        other_kinds = set(requested_kinds) - {"device"}
        if other_kinds:
            folds.extend(build_folds(rows, other_kinds))
    else:
        folds = build_folds(rows, set(requested_kinds))
    if args.fold_id:
        requested_fold_ids = set(args.fold_id)
        folds = [
            fold for fold in folds
            if fold["id"] in requested_fold_ids
        ]
        found_fold_ids = {fold["id"] for fold in folds}
        missing_fold_ids = requested_fold_ids - found_fold_ids
        if missing_fold_ids:
            raise RuntimeError(
                "requested_fold_not_found:" + ",".join(
                    sorted(missing_fold_ids)
                )
            )
    if args.max_folds_per_kind:
        limited = []
        counts = Counter()
        for fold in folds:
            if counts[fold["kind"]] >= args.max_folds_per_kind:
                continue
            limited.append(fold)
            counts[fold["kind"]] += 1
        folds = limited
    report = {
        "scope": "nested_fold_external_calibration",
        "externalDeviceHoldout": bool(
            evaluation_only_rows and "device" in requested_kinds
        ),
        "externalRowsUsedForFitting": False,
        "primaryTask": contract.PRIMARY_TASK,
        "negativeProxy": "light_touch",
        "ordinaryClickFprAvailable": bool(ordinary_click_rows),
        "targetFalsePositiveRates": list(TARGET_FPRS),
        "featureFamily": args.feature_family,
        "evidenceContract": (
            evidence_features.contract_metadata(args.evidence_contract)
            if args.feature_family == "evidence"
            else None
        ),
        "featureMode": args.feature_mode,
        "weighting": args.weighting,
        "modalityDropout": args.modality_dropout,
        "augmentationMode": args.augmentation_mode,
        "augmentationProbability": (
            args.augmentation_probability
            if args.augmentation_mode == "channel_scale"
            else 0.0
        ),
        "augmentationMaximumLogScale": (
            args.augmentation_max_log_scale
            if args.augmentation_mode == "channel_scale"
            else 0.0
        ),
        "imuMode": args.imu_mode,
        "gravityAlignmentCoverage": gravity_alignment_coverage(rows),
        "adaptiveFeatureMode": args.adaptive_feature_mode,
        "normalizationMode": args.normalization_mode,
        "earlyEvidenceMode": args.early_evidence_mode,
        "earlyEvidenceContract": (
            early_press_evidence.CONTRACT
            if args.early_evidence_mode != "none"
            else None
        ),
        "earlyEvidenceFiles": [
            str(Path(path).resolve())
            for path in args.early_evidence_file
        ],
        "earlyEvidenceCoverage": early_evidence_coverage(rows),
        "evaluationOnlyEarlyEvidenceCoverage": early_evidence_coverage(
            evaluation_only_rows
        ),
        "fingerLossWeight": args.finger_loss_weight,
        "modelSelectionMode": args.model_selection_mode,
        "selectionTargetFalsePositiveRate": args.selection_target_fpr,
        "thresholdCalibrationFalsePositiveRate": (
            args.threshold_calibration_fpr
        ),
        "falsePositiveLimit": args.false_positive_limit,
        "thresholdCalibrationSweep": (
            args.threshold_calibration_sweep
        ),
        "gateLossMode": args.gate_loss_mode,
        "rankingLossWeight": args.ranking_loss_weight,
        "pairwiseMargin": args.pairwise_margin,
        "positiveFocalWeight": args.positive_focal_weight,
        "negativeFocalWeight": args.negative_focal_weight,
        "targetCalibrationNegativeCounts": (
            args.target_calibration_negative_count
        ),
        "targetCalibrationRepetitions": (
            args.target_calibration_repetitions
        ),
        "datasetDirs": [str(Path(path).resolve()) for path in args.dataset_dir],
        "evaluationOnlyDatasetDirs": [
            str(Path(path).resolve())
            for path in args.evaluation_only_dataset_dir
        ],
        "evaluationOnlyDataset": bool(evaluation_only_rows),
        "evaluationOnlySampleCount": len(evaluation_only_rows),
        # 中文注释：外测批次单独留档，防止训练与外测的采集域变化被平均值掩盖。
        "evaluationOnlyCollectionBatchDiagnostics": (
            collection_batch_diagnostics(evaluation_only_rows)
            if evaluation_only_rows
            else {}
        ),
        "evaluationOnlyDevices": sorted({
            str(row.get("deviceName") or "")
            for row in evaluation_only_rows
        }),
        "ordinaryClickDatasetDirs": [
            str(Path(path).resolve())
            for path in args.ordinary_click_dataset_dir
        ],
        "ordinaryClickSampleCount": len(ordinary_click_rows),
        "ordinaryClickDevices": sorted({
            str(row.get("deviceName") or "")
            for row in ordinary_click_rows
        }),
        "ordinaryClickUsedForFitting": False,
        "rowCount": len(rows),
        # 中文注释：批次只用于诊断采集域漂移，不参与模型拟合或阈值选择。
        "collectionBatchDiagnostics": collection_batch_diagnostics(rows),
        "dataContractDiagnostics": data_contract_diagnostics(rows),
        "evaluationOnlyDataContractDiagnostics": (
            data_contract_diagnostics(evaluation_only_rows)
            if evaluation_only_rows
            else None
        ),
        "experimentBlockDiagnostics": experiment_block_diagnostics(rows),
        "deviceNames": sorted({
            str(row.get("deviceName") or "") for row in rows
        }),
        "foldCounts": dict(Counter(fold["kind"] for fold in folds)),
        "models": {},
        "runs": {},
    }
    for model_name, context_heads in MODEL_SPECS:
        model_runs = []
        for fold in folds:
            for seed_offset in range(args.seed_count):
                seed = args.base_seed + seed_offset
                result = run_fold_seed(
                    fold=fold,
                    seed=seed,
                    hidden_count=args.hidden_count,
                    epochs=args.epochs,
                    context_heads=context_heads,
                    context_loss_weight=args.context_loss_weight,
                    finger_loss_weight=args.finger_loss_weight,
                    calibration_ratio=args.calibration_ratio,
                    patience=args.patience,
                    feature_mode=args.feature_mode,
                    weighting=args.weighting,
                    modality_dropout=args.modality_dropout,
                    imu_mode=args.imu_mode,
                    adaptive_feature_mode=args.adaptive_feature_mode,
                    target_calibration_counts=(
                        args.target_calibration_negative_count
                    ),
                    target_calibration_repetitions=(
                        args.target_calibration_repetitions
                    ),
                    evaluation_only_rows=evaluation_only_rows,
                    feature_family=args.feature_family,
                    evidence_contract=args.evidence_contract,
                    ordinary_click_rows=(
                        select_ordinary_click_rows_for_fold(
                            ordinary_click_rows,
                            fold,
                        )
                    ),
                    normalization_mode=args.normalization_mode,
                    early_evidence_mode=args.early_evidence_mode,
                    augmentation_mode=args.augmentation_mode,
                    augmentation_probability=args.augmentation_probability,
                    augmentation_max_log_scale=(
                        args.augmentation_max_log_scale
                    ),
                    selection_mode=args.model_selection_mode,
                    selection_target_fpr=args.selection_target_fpr,
                    gate_loss_mode=args.gate_loss_mode,
                    ranking_loss_weight=args.ranking_loss_weight,
                    pairwise_margin=args.pairwise_margin,
                    positive_focal_weight=args.positive_focal_weight,
                    negative_focal_weight=args.negative_focal_weight,
                    threshold_calibration_fpr=(
                        args.threshold_calibration_fpr
                    ),
                    threshold_calibration_sweep=(
                        args.threshold_calibration_sweep
                    ),
                    false_positive_limit=args.false_positive_limit,
                )
                model_runs.append(result)
                print(json.dumps({
                    "model": model_name,
                    "fold": fold["id"],
                    "seed": seed,
                    "epoch": result["selection"]["bestEpoch"],
                    "recallAt1pct": result["fixedFprMetrics"]["0.01"][
                        "overall"
                    ]["heavyRecall"],
                    "recallAt5pct": result["fixedFprMetrics"]["0.05"][
                        "overall"
                    ]["heavyRecall"],
                }), flush=True)
        report["runs"][model_name] = model_runs
        report["models"][model_name] = summarize_runs(
            model_runs,
            model_name,
        )
        report.setdefault("thresholdCalibrationSweepModels", {})[
            model_name
        ] = summarize_threshold_calibration_sweep(
            model_runs,
            args.false_positive_limit,
        )
        if args.target_calibration_negative_count:
            report.setdefault("targetCalibratedModels", {})[model_name] = (
                summarize_target_calibration_runs(model_runs)
            )
        if evaluation_only_rows:
            report.setdefault("evaluationOnlyModels", {})[model_name] = (
                summarize_evaluation_only_runs(model_runs)
            )
            report.setdefault("evaluationOnlyFoldMetrics", {})[model_name] = (
                summarize_evaluation_only_folds(model_runs)
            )
        if ordinary_click_rows:
            report.setdefault("ordinaryClickModels", {})[model_name] = (
                summarize_ordinary_click_audits(model_runs)
            )
    report["contextHeadComparison"] = compare_models(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
