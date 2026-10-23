#!/usr/bin/env python3
"""Separate contact-size signal from session and device scale identity."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from . import run_lightweight_time_grid_models as lightweight
    from . import run_stats_mlp_feature_experiments as feature_experiments
except ImportError:
    import run_lightweight_time_grid_models as lightweight
    import run_stats_mlp_feature_experiments as feature_experiments


TOUCH_START = 7
# 源矩阵保持 touch26 的 26 列布局，物理矩阵只保留十个运行通道。
# 两套矩阵的 size 列号不同，必须从名称派生，禁止共用同一个手写常量。
SOURCE_SIZE_CHANNEL = lightweight.FEATURE_NAMES.index("size")
SOURCE_MAJOR_CHANNEL = lightweight.FEATURE_NAMES.index("major_norm")
SOURCE_SIZE_AVAILABLE_CHANNEL = lightweight.FEATURE_NAMES.index(
    "size_available"
)
SOURCE_MAJOR_AVAILABLE_CHANNEL = lightweight.FEATURE_NAMES.index(
    "major_available"
)
PHYSICAL_SIZE_CHANNEL = lightweight.PHYSICAL_NAMES.index("size")
SOURCE_GRAVITY_INDICES = tuple(
    lightweight.FEATURE_NAMES.index(name)
    for name in ("gravity.x", "gravity.y", "gravity.z")
)
CAUSAL_WARMUP_COUNTS = (4, 8, 16)
TAIL_CALIBRATION_COUNTS = (8, 16, 32)
ORDINARY_CLASS_LABEL = "ordinary_touch"


def load_samples(dataset_dir):
    """Load folds while retaining session order metadata."""
    root = Path(dataset_dir)
    rows = lightweight.read_rows(root)
    by_hash = {row["sourceSha256"]: row for row in rows}
    grouped = defaultdict(list)
    with (root / "strict_group_folds.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as source:
        for item in csv.DictReader(source):
            row = by_hash[item["sourceSha256"]]
            grouped[int(item["fold"])].append({
                "sourceSha256": row["sourceSha256"],
                "sampleId": row["sampleId"],
                "sourceFile": row["sourceFile"],
                # 负样本来自独立文件，设备身份必须从正样本原样带入，
                # 否则同一物理设备的普通点击会被匹配到错误 fold。
                "platform": row.get("platform", ""),
                "collectorDeviceId": row.get(
                    "collectorDeviceId",
                    "",
                ),
                "deviceName": row["deviceName"],
                "classLabel": row["classLabel"],
                "label": lightweight.CLASS_LABELS.index(row["classLabel"]),
                "groupId": row["groupId"],
                "bodyPosture": row.get("bodyPosture", "unknown"),
                "matrix": np.asarray(row["matrix"], dtype=np.float32),
                "touchAxisRangeAudit": row.get("touchAxisRangeAudit"),
                "split": item["split"],
            })
    folds = {}
    for fold, samples in grouped.items():
        train = [sample for sample in samples if sample["split"] == "train"]
        valid = [sample for sample in samples if sample["split"] == "valid"]
        for sample in train:
            sample["split"] = "train"
        for sample in valid:
            sample["split"] = "valid"
        if {sample["groupId"] for sample in train}.intersection(
            {sample["groupId"] for sample in valid}
        ):
            raise RuntimeError("cross_split_group_leakage")
        folds[fold] = (train, valid)
    return folds


def load_ordinary_samples(dataset_dir):
    """Load rejection negatives without adding them to supervised folds."""
    path = Path(dataset_dir) / "ordinary_touch23.jsonl"
    if not path.is_file():
        path = Path(dataset_dir) / "ordinary_touch26.jsonl"
    if not path.is_file():
        return []
    result = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("classLabel") != ORDINARY_CLASS_LABEL:
                raise RuntimeError("invalid_ordinary_negative_label")
            result.append({
                "sourceSha256": row["sourceSha256"],
                "sampleId": row["sampleId"],
                "sourceFile": row["sourceFile"],
                "platform": row.get("platform", ""),
                "collectorDeviceId": row.get(
                    "collectorDeviceId",
                    "",
                ),
                "deviceName": row.get("deviceName", ""),
                "groupId": row["groupId"],
                "bodyPosture": row.get("bodyPosture", "unknown"),
                "matrix": np.asarray(row["matrix"], dtype=np.float32),
                "touchAxisRangeAudit": row.get(
                    "touchAxisRangeAudit"
                ),
            })
    return result


def ordinary_samples_for_fold(
    ordinary_samples,
    valid_samples,
    fold_kind,
):
    """Match negatives to the same held-out device or session domain."""
    if not ordinary_samples:
        return []
    if fold_kind == "held_out_device":
        valid_devices = {
            physical_sample_device_key(sample)
            for sample in valid_samples
        }
        return [
            sample
            for sample in ordinary_samples
            if physical_sample_device_key(sample) in valid_devices
        ]
    if fold_kind == "held_out_session":
        valid_groups = {
            sample["groupId"] for sample in valid_samples
        }
        return [
            sample
            for sample in ordinary_samples
            if sample["groupId"] in valid_groups
        ]
    return []


def ordinary_negative_audit(
    ordinary_samples,
    folds,
    fold_kind,
):
    """Count negatives that reach each validation domain before calibration."""
    per_fold = {}
    matched_keys = set()
    matched_fold_count = 0
    positive_fold_count = len(folds)
    for fold, (_, valid_samples) in sorted(folds.items()):
        selected = ordinary_samples_for_fold(
            ordinary_samples,
            valid_samples,
            fold_kind,
        )
        keys = {
            str(
                sample.get("sourceSha256")
                or sample.get("sampleId")
                or sample.get("sourceFile")
                or ""
            )
            for sample in selected
        }
        matched_keys.update(keys)
        if selected:
            matched_fold_count += 1
        per_fold[str(fold)] = {
            "validDevices": sorted({
                str(sample.get("deviceName") or "")
                for sample in valid_samples
            }),
            "validGroups": sorted({
                str(sample.get("groupId") or "")
                for sample in valid_samples
            }),
            "matchedCount": len(selected),
            "uniqueMatchedCount": len(keys),
        }
    total_count = len(ordinary_samples)
    matched_count = len(matched_keys)
    return {
        "totalCount": total_count,
        "matchedCount": matched_count,
        "unmatchedCount": max(0, total_count - matched_count),
        "positiveFoldCount": positive_fold_count,
        "matchedFoldCount": matched_fold_count,
        "foldCoverage": (
            matched_fold_count / positive_fold_count
            if positive_fold_count
            else 0.0
        ),
        "perFold": per_fold,
    }


def ordinary_negative_mode_audit(domain_audit, reports):
    """Add the calibration-scored negative count for one experiment mode."""
    matched_count = sum(
        int(report.get("ordinaryDomainSampleCount", 0))
        for report in reports
    )
    scored_count = sum(
        int(report.get("ordinarySampleCount", 0))
        for report in reports
    )
    matched_fold_count = sum(
        1
        for report in reports
        if int(report.get("ordinaryDomainSampleCount", 0)) > 0
    )
    scored_fold_count = sum(
        1
        for report in reports
        if int(report.get("ordinarySampleCount", 0)) > 0
    )
    positive_fold_count = int(domain_audit.get("positiveFoldCount", 0))
    return {
        "totalCount": int(domain_audit.get("totalCount", 0)),
        "matchedCount": matched_count,
        "scoredCount": scored_count,
        "positiveFoldCount": positive_fold_count,
        "matchedFoldCount": matched_fold_count,
        "scoredFoldCount": scored_fold_count,
        "foldCoverage": (
            matched_fold_count / positive_fold_count
            if positive_fold_count
            else 0.0
        ),
        "scoredFoldCoverage": (
            scored_fold_count / positive_fold_count
            if positive_fold_count
            else 0.0
        ),
    }


def prepare_ordinary_values(
    train_samples,
    valid_samples,
    ordinary_samples,
    mode,
    prepare,
    pose_mode,
    is_cross_device,
):
    """Prepare negatives with the same held-out calibration as positives."""
    if not ordinary_samples:
        return [], None
    if (
        is_cross_device
        and mode.startswith("causal_device_median_")
        and mode.endswith("_tail")
    ):
        # 负样本必须和正式验证集一起排序，共享同一个设备预热前缀。
        # 否则单独从普通点击估计尺度，会得到过于乐观的误接受率。
        combined = list(valid_samples) + list(ordinary_samples)
        _, combined_values, score_indices = (
            prepare_fold_leave_one_device_out_calibrated_tail(
                train_samples,
                combined,
                mode,
                pose_mode=pose_mode,
            )
        )
        offset = len(valid_samples)
        retained = [
            index - offset
            for index in score_indices
            if index >= offset
        ]
        if not retained:
            return [], None
        selected = np.asarray(
            [index + offset for index in retained],
            dtype=np.int64,
        )
        return (
            [ordinary_samples[index] for index in retained],
            combined_values[selected],
        )
    ordinary_values, _ = prepare(
        train_samples,
        ordinary_samples,
        mode,
        pose_mode=pose_mode,
    )
    return ordinary_samples, ordinary_values


def physical_sample_device_key(sample):
    """Mirror the dataset builder's physical-device grouping."""
    platform = str(sample.get("platform") or "")
    device_name = str(sample.get("deviceName") or "")
    if device_name:
        # 中文注释：设备名是留出设备的主身份；平台缺失不能把同名设备
        # 与另一台设备合并，否则普通点击会匹配到错误 fold。
        return platform + ":device:" + device_name
    return (
        platform
        + ":collector:"
        + str(sample.get("collectorDeviceId") or "")
    )


def fold_evaluation_scope(folds):
    """Read the builder contract without inferring it from fold numbers."""
    root = None
    for train, valid in folds.values():
        for sample in train + valid:
            root = Path(sample["datasetId"])
            break
        if root is not None:
            break
    if root is None:
        return "not_ready"
    report = json.loads(
        (root / "training_cleaning_report.json").read_text(
            encoding="utf-8"
        )
    )
    return str(report.get("evaluationScope") or "unknown")


def fold_metadata(folds):
    """Describe held-out device or session identity for each fold."""
    root = None
    for train, valid in folds.values():
        for sample in train + valid:
            root = Path(sample["datasetId"])
            break
        if root is not None:
            break
    fold_kind = "unknown"
    if root is not None:
        report = json.loads(
            (root / "training_cleaning_report.json").read_text(
                encoding="utf-8"
            )
        )
        fold_kind = str(report.get("validationFoldKind") or "unknown")
    result = {}
    for fold, (train, valid) in folds.items():
        valid_devices = {sample["deviceName"] for sample in valid}
        train_devices = {sample["deviceName"] for sample in train}
        result[str(fold)] = {
            "kind": fold_kind,
            "validDevices": sorted(valid_devices),
            "trainDevices": sorted(train_devices),
            "validGroups": len({sample["groupId"] for sample in valid}),
        }
    return result


def read_dataset_rows(dataset_dir):
    """Normalize only rows referenced by strict folds for audit probes."""
    root = Path(dataset_dir)
    rows = lightweight.read_rows(root)
    selected_hashes = set()
    with (root / "strict_group_folds.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as source:
        selected_hashes.update(
            item["sourceSha256"] for item in csv.DictReader(source)
        )
    result = []
    for row in rows:
        if row["sourceSha256"] not in selected_hashes:
            continue
        result.append({
            **row,
            "matrix": np.asarray(row["matrix"], dtype=np.float32),
        })
    if len(result) != len(selected_hashes):
        raise RuntimeError("training_matrix_missing_strict_fold_source")
    return result


def valid_size_values(sample):
    """Return only contact sizes marked available by the source contract."""
    values = sample["matrix"][:, SOURCE_SIZE_CHANNEL]
    available = (
        sample["matrix"][:, SOURCE_SIZE_AVAILABLE_CHANNEL] >= 0.5
    )
    return values[available & (values > 0.0)]


def robust_reference(values, quantile):
    """Build a positive scale while tolerating unsupported devices."""
    if len(values) == 0:
        return None
    reference = float(np.percentile(values, quantile))
    if not math.isfinite(reference) or reference <= 1e-8:
        return None
    return reference


def safe_ratio(numerator, denominator, epsilon=1e-8):
    """Divide only finite positive denominators and preserve zero elsewhere."""
    numerator = np.asarray(numerator, dtype=np.float32)
    denominator = np.asarray(denominator, dtype=np.float32)
    result = np.zeros_like(numerator, dtype=np.float32)
    np.divide(
        numerator,
        denominator,
        out=result,
        where=np.isfinite(numerator)
        & np.isfinite(denominator)
        & (np.abs(denominator) > epsilon),
    )
    return result


def chronological_session_samples(samples, count=None):
    """Select session samples by capture order without using labels."""
    ordered = sorted(
        samples,
        key=lambda sample: (
            sample.get("sourceFile", ""),
            sample["sourceSha256"],
        ),
    )
    return ordered if count is None else ordered[:count]


def replace_valid_size(sample, target, reference, center=0.0, scale=None):
    """Normalize only available values and leave missing values at zero."""
    values = sample["matrix"][:, SOURCE_SIZE_CHANNEL]
    available = (
        sample["matrix"][:, SOURCE_SIZE_AVAILABLE_CHANNEL] >= 0.5
    ) & (values > 0.0)
    if scale is None:
        transformed = values / reference
    else:
        transformed = (values - center) / scale
    target[:, PHYSICAL_SIZE_CHANNEL] = np.where(
        available,
        transformed,
        0.0,
    )


def apply_within_touch_mode(raw, samples, mode):
    """Normalize contact size by one touch so no session warmup is needed."""
    result = raw.copy()
    for index, sample in enumerate(samples):
        values = valid_size_values(sample)
        reference = robust_reference(values, 50)
        if reference is None:
            result[index, :, PHYSICAL_SIZE_CHANNEL] = 0.0
            continue
        if mode == "within_touch_median":
            replace_valid_size(sample, result[index], reference)
        elif mode == "within_touch_iqr_z":
            low, high = np.percentile(values, (25, 75))
            scale = float(high - low)
            if scale <= 1e-8:
                result[index, :, PHYSICAL_SIZE_CHANNEL] = 0.0
            else:
                replace_valid_size(
                    sample,
                    result[index],
                    reference,
                    center=reference,
                    scale=scale,
                )
        else:
            raise RuntimeError("unsupported_within_touch_mode")
    return result


def apply_session_scale_mode(
    raw,
    samples,
    mode,
    causal_count=None,
    quantile=50,
):
    """Normalize each session using unlabeled values from that session."""
    result = raw.copy()
    session_indices = defaultdict(list)
    for index, sample in enumerate(samples):
        session_indices[sample["groupId"]].append(index)
    for indices in session_indices.values():
        selected = chronological_session_samples(
            [samples[index] for index in indices],
            causal_count,
        )
        reference_values = np.concatenate(
            [valid_size_values(sample) for sample in selected]
        ) if selected else np.asarray([], dtype=np.float32)
        reference = robust_reference(reference_values, quantile)
        if reference is None:
            for index in indices:
                result[index, :, PHYSICAL_SIZE_CHANNEL] = 0.0
            continue
        for index in indices:
            if mode == "session_median":
                replace_valid_size(samples[index], result[index], reference)
            elif mode == "session_iqr_z":
                low, high = np.percentile(reference_values, (25, 75))
                scale = float(high - low)
                if scale <= 1e-8:
                    result[index, :, PHYSICAL_SIZE_CHANNEL] = 0.0
                else:
                    replace_valid_size(
                        samples[index],
                        result[index],
                        reference,
                        center=reference,
                        scale=scale,
                    )
            else:
                raise RuntimeError("unsupported_session_scale_mode")
    return result


def apply_major_ratio_mode(raw, samples):
    """Express area against the observed major axis to retain shape."""
    result = raw.copy()
    for index, sample in enumerate(samples):
        size = sample["matrix"][:, SOURCE_SIZE_CHANNEL]
        major = sample["matrix"][:, SOURCE_MAJOR_CHANNEL]
        available = (
            (sample["matrix"][:, SOURCE_SIZE_AVAILABLE_CHANNEL] >= 0.5)
            & (sample["matrix"][:, SOURCE_MAJOR_AVAILABLE_CHANNEL] >= 0.5)
            & (size > 0.0)
            & (major > 1e-8)
        )
        result[index, :, PHYSICAL_SIZE_CHANNEL] = np.where(
            available,
            safe_ratio(size, major),
            0.0,
        )
    return result


def declared_major_diagonal_ratio(sample):
    """Return declared touch-major range in view-diagonal units."""
    audit = sample.get("touchAxisRangeAudit")
    if isinstance(audit, dict):
        ranges = audit.get("ranges")
        if isinstance(ranges, dict):
            major = ranges.get("touchMajor")
            if isinstance(major, dict):
                ratio = major.get("declaredToViewDiagonalMedian")
                try:
                    ratio = float(ratio)
                except (TypeError, ValueError):
                    ratio = 0.0
                if math.isfinite(ratio) and ratio > 1e-8:
                    return ratio
    ratios = []
    for frame in sample.get("touchFrames") or ():
        if not isinstance(frame, dict):
            continue
        ranges = frame.get("axisMotionRanges")
        if not isinstance(ranges, dict):
            continue
        major = ranges.get("touchMajor")
        if not isinstance(major, dict):
            continue
        try:
            maximum = float(major.get("max") or 0.0)
            width = float(frame.get("viewWidthPx") or 0.0)
            height = float(frame.get("viewHeightPx") or 0.0)
        except (TypeError, ValueError):
            continue
        diagonal = math.hypot(width, height)
        ratio = maximum / diagonal if diagonal > 0.0 else 0.0
        if math.isfinite(ratio) and ratio > 1e-8:
            ratios.append(ratio)
    if not ratios:
        return None
    return float(np.median(np.asarray(ratios, dtype=np.float64)))


def apply_declared_major_range_mode(
    raw,
    samples,
):
    """Map major axis to the driver range without reading labels."""
    result = raw.copy()
    for index, sample in enumerate(samples):
        diagonal_ratio = declared_major_diagonal_ratio(sample)
        if diagonal_ratio is None:
            result[index, :, PHYSICAL_SIZE_CHANNEL] = 0.0
            continue
        values = sample["matrix"][:, SOURCE_MAJOR_CHANNEL]
        available = (
            sample["matrix"][:, SOURCE_MAJOR_AVAILABLE_CHANNEL] >= 0.5
        ) & (values > 0.0)
        # major_norm 已按视图对角线归一化，再除以声明的量程比例，
        # 等价于把原始 touchMajor 除以该设备的声明最大值。
        transformed = safe_ratio(values, diagonal_ratio)
        result[index, :, PHYSICAL_SIZE_CHANNEL] = np.where(
            available,
            transformed,
            0.0,
        )
    return result


def apply_major_within_touch_mode(raw, samples):
    """Normalize major axis by the same touch to remove device scale."""
    result = raw.copy()
    for index, sample in enumerate(samples):
        values = sample["matrix"][:, SOURCE_MAJOR_CHANNEL]
        available = (
            sample["matrix"][:, SOURCE_MAJOR_AVAILABLE_CHANNEL] >= 0.5
        ) & (values > 0.0)
        valid = values[available]
        reference = robust_reference(valid, 50)
        if reference is None:
            result[index, :, PHYSICAL_SIZE_CHANNEL] = 0.0
            continue
        result[index, :, PHYSICAL_SIZE_CHANNEL] = np.where(
            available,
            safe_ratio(values, reference),
            0.0,
        )
    return result


def apply_presence_mode(raw, samples):
    """Replace scale with whether the touch driver exposed size at all."""
    result = raw.copy()
    for index, sample in enumerate(samples):
        available = (
            sample["matrix"][:, SOURCE_SIZE_AVAILABLE_CHANNEL] >= 0.5
        )
        fraction = float(np.mean(available))
        result[index, :, PHYSICAL_SIZE_CHANNEL] = (
            fraction if fraction > 0.0 else 0.0
        )
    return result


def gravity_aligned_vectors(gravity, vectors):
    """Rotate vectors into a gravity-aligned frame without future data."""
    gravity = np.asarray(gravity, dtype=np.float32)
    vectors = np.asarray(vectors, dtype=np.float32)
    gravity_norm = float(np.linalg.norm(gravity))
    if not math.isfinite(gravity_norm) or gravity_norm <= 1e-6:
        return vectors.copy()
    vertical = gravity / gravity_norm
    reference = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    tangent = reference - float(np.dot(reference, vertical)) * vertical
    if float(np.linalg.norm(tangent)) <= 1e-6:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        tangent = reference - float(np.dot(reference, vertical)) * vertical
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-6:
        return vectors.copy()
    axis_x = tangent / tangent_norm
    axis_y = np.cross(vertical, axis_x)
    return np.stack(
        (
            vectors @ axis_x,
            vectors @ axis_y,
            vectors @ vertical,
        ),
        axis=1,
    ).astype(np.float32)


def apply_pose_mode(raw, samples, mode):
    """Remove device roll and pitch by rotating IMU vectors per frame."""
    if mode == "raw":
        return raw.copy()
    if mode != "gravity_aligned_axes":
        raise RuntimeError("unsupported_pose_mode")
    result = raw.copy()
    for index, sample in enumerate(samples):
        source = sample["matrix"]
        gravity = source[:, SOURCE_GRAVITY_INDICES]
        acceleration = source[:, 2:5]
        rotation = source[:, 5:8]
        for frame in range(result.shape[1]):
            aligned_acceleration = gravity_aligned_vectors(
                gravity[frame],
                acceleration[frame][None, :],
            )[0]
            aligned_rotation = gravity_aligned_vectors(
                gravity[frame],
                rotation[frame][None, :],
            )[0]
            result[index, frame, 2:5] = aligned_acceleration
            result[index, frame, 5:8] = aligned_rotation
    return result


def apply_device_scale_mode(raw, samples, mode, causal_count=None):
    """Normalize size by an unlabeled device baseline.

    The oracle modes use every unlabeled value from the evaluation device and
    are diagnostics only. The causal modes use the first N chronological
    touches and can be reproduced after a short on-device calibration period.
    """
    result = raw.copy()
    device_indices = defaultdict(list)
    for index, sample in enumerate(samples):
        # 同名机型可能来自不同平台或采集器，预热基线不能跨身份混合。
        device_indices[physical_sample_device_key(sample)].append(index)
    for indices in device_indices.values():
        selected = chronological_session_samples(
            [samples[index] for index in indices],
            causal_count,
        )
        reference_values = np.concatenate(
            [valid_size_values(sample) for sample in selected]
        ) if selected else np.asarray([], dtype=np.float32)
        reference = robust_reference(reference_values, 50)
        if reference is None:
            for index in indices:
                result[index, :, PHYSICAL_SIZE_CHANNEL] = 0.0
            continue
        for index in indices:
            if mode == "device_median":
                replace_valid_size(samples[index], result[index], reference)
            elif mode == "device_iqr_z":
                low, high = np.percentile(reference_values, (25, 75))
                scale = float(high - low)
                if scale <= 1e-8:
                    result[index, :, PHYSICAL_SIZE_CHANNEL] = 0.0
                else:
                    replace_valid_size(
                        samples[index],
                        result[index],
                        reference,
                        center=reference,
                        scale=scale,
                    )
            else:
                raise RuntimeError("unsupported_device_scale_mode")
    return result


def apply_size_mode(raw, samples, mode):
    """Route one experimental size policy without reading class labels."""
    if mode == "baseline":
        return raw.copy()
    if mode.startswith("declared_") and not any(
        declared_major_diagonal_ratio(sample) is not None
        for sample in samples
    ):
        # 旧集合没有量程摘要。让该策略回退到同一份输入基线，
        # 避免把它误报成“接触尺寸无效”的零特征实验结果。
        return raw.copy()
    if mode == "shape_only":
        result = raw.copy()
        result[:, :, PHYSICAL_SIZE_CHANNEL] = 0.0
        return result
    if mode in ("within_touch_median", "within_touch_iqr_z"):
        return apply_within_touch_mode(raw, samples, mode)
    if mode == "session_median_oracle":
        return apply_session_scale_mode(
            raw,
            samples,
            "session_median",
        )
    if mode == "session_iqr_z_oracle":
        return apply_session_scale_mode(
            raw,
            samples,
            "session_iqr_z",
        )
    if mode.startswith("causal_median_"):
        count = int(mode.rsplit("_", 1)[1])
        return apply_session_scale_mode(
            raw,
            samples,
            "session_median",
            causal_count=count,
        )
    if mode == "major_ratio":
        return apply_major_ratio_mode(raw, samples)
    if mode == "major_within_touch_median":
        return apply_major_within_touch_mode(raw, samples)
    if mode == "declared_major_range":
        return apply_declared_major_range_mode(raw, samples)
    if mode == "presence":
        return apply_presence_mode(raw, samples)
    if mode in ("device_median_oracle", "device_iqr_z_oracle"):
        return apply_device_scale_mode(
            raw,
            samples,
            "device_median" if mode.startswith("device_median") else "device_iqr_z",
        )
    if mode.startswith("causal_device_median_"):
        count = int(mode.rsplit("_", 1)[1])
        return apply_device_scale_mode(
            raw,
            samples,
            "device_median",
            causal_count=count,
        )
    raise RuntimeError("unsupported_size_mode")


def prepare_fold(train_samples, valid_samples, mode, pose_mode="raw"):
    """Fit all scalers on training data after the selected size transform."""
    if mode.startswith("declared_"):
        # 量程属于设备合同，按样本自身归一化；训练与验证共同处理，
        # 但不从验证样本估计任何参数，也不读取标签。
        samples = train_samples + valid_samples
        raw = lightweight.select_matrix(
            samples,
            lightweight.PHYSICAL_INDICES,
        )
        transformed = apply_size_mode(raw, samples, mode)
        split = len(train_samples)
        train_raw = transformed[:split]
        valid_raw = transformed[split:]
    else:
        train_raw = lightweight.select_matrix(
            train_samples,
            lightweight.PHYSICAL_INDICES,
        )
        valid_raw = lightweight.select_matrix(
            valid_samples,
            lightweight.PHYSICAL_INDICES,
        )
        train_raw = apply_size_mode(train_raw, train_samples, mode)
        valid_raw = apply_size_mode(valid_raw, valid_samples, mode)
    return finish_fold_features(
        train_raw,
        valid_raw,
        train_samples,
        valid_samples,
        pose_mode,
    )


def prepare_fold_leave_one_device_out(
    train_samples,
    valid_samples,
    mode,
    pose_mode="raw",
):
    """Apply a deployable transform without validation-device statistics.

    A held-out device has no training examples, so transforms that need that
    device's baseline are deliberately rejected instead of silently reading
    validation distribution.
    """
    if mode in (
        "device_median_oracle",
        "device_iqr_z_oracle",
    ):
        # 名称已经标明 oracle；保留该路径仅用于判断设备尺度上限，
        # 主排序和部署决策不得把它当作可用候选。
        return prepare_fold(
            train_samples,
            valid_samples,
            mode,
            pose_mode=pose_mode,
        )
    if mode.startswith("causal_device_median_"):
        train_raw = lightweight.select_matrix(
            train_samples,
            lightweight.PHYSICAL_INDICES,
        )
        train_raw = apply_size_mode(train_raw, train_samples, mode)
        raw_valid = lightweight.select_matrix(
            valid_samples,
            lightweight.PHYSICAL_INDICES,
        )
        valid_raw = raw_valid.copy()
        valid_raw[:, :, PHYSICAL_SIZE_CHANNEL] = 0.0
        return finish_fold_features(
            train_raw,
            valid_raw,
            train_samples,
            valid_samples,
            pose_mode,
        )
    return prepare_fold(
        train_samples,
        valid_samples,
        mode,
        pose_mode=pose_mode,
    )


def prepare_fold_leave_one_device_out_calibrated_tail(
    train_samples,
    valid_samples,
    mode,
    pose_mode="raw",
):
    """Calibrate an unseen device from early unlabeled touches.

    The first N validation touches estimate only device scale. They are
    removed from scoring, so no validation label or later distribution is
    used. This models a short first-run adaptation period.
    """
    prefix = "causal_device_median_"
    suffix = "_tail"
    if not mode.startswith(prefix) or not mode.endswith(suffix):
        raise RuntimeError("unsupported_tail_calibration_mode")
    count = int(mode[len(prefix):-len(suffix)])
    if count <= 0:
        raise RuntimeError("invalid_tail_calibration_count")
    train_mode = "causal_device_median_" + str(count)
    train_raw = lightweight.select_matrix(
        train_samples,
        lightweight.PHYSICAL_INDICES,
    )
    train_raw = apply_size_mode(
        train_raw,
        train_samples,
        train_mode,
    )
    valid_raw = lightweight.select_matrix(
        valid_samples,
        lightweight.PHYSICAL_INDICES,
    )
    valid_raw = apply_size_mode(
        valid_raw,
        valid_samples,
        train_mode,
    )
    evaluation_indices = []
    valid_by_device = defaultdict(list)
    for index, sample in enumerate(valid_samples):
        valid_by_device[physical_sample_device_key(sample)].append(index)
    for indices in valid_by_device.values():
        ordered = sorted(
            indices,
            key=lambda index: (
                valid_samples[index].get("sourceFile", ""),
                valid_samples[index]["sourceSha256"],
            ),
        )
        evaluation_indices.extend(ordered[count:])
    train_values, valid_values = finish_fold_features(
        train_raw,
        valid_raw,
        train_samples,
        valid_samples,
        pose_mode,
    )
    return train_values, valid_values, sorted(evaluation_indices)


def finish_fold_features(
    train_raw,
    valid_raw,
    train_samples,
    valid_samples,
    pose_mode,
):
    """Fit scalers on training data and freeze the runtime statistics."""
    train_raw = apply_pose_mode(train_raw, train_samples, pose_mode)
    valid_raw = apply_pose_mode(valid_raw, valid_samples, pose_mode)
    center, scale = feature_experiments.fit_scaler_windowed(
        train_raw,
        TOUCH_START,
    )
    train_stats = feature_experiments.stats_features_windowed(
        lightweight.transform(train_raw, center, scale),
        TOUCH_START,
    )
    valid_stats = feature_experiments.stats_features_windowed(
        lightweight.transform(valid_raw, center, scale),
        TOUCH_START,
    )
    stats_center = train_stats.mean(axis=0)
    stats_scale = train_stats.std(axis=0)
    stats_scale[stats_scale < 1e-6] = 1.0
    return (
        np.clip(
            (train_stats - stats_center) / stats_scale,
            -8.0,
            8.0,
        ).astype(np.float32),
        np.clip(
            (valid_stats - stats_center) / stats_scale,
            -8.0,
            8.0,
        ).astype(np.float32),
    )


def train_fixed_epochs(
    train_values,
    train_samples,
    epochs,
    seed,
):
    """Train without validation-label epoch selection."""
    lightweight.configure_determinism(seed)
    x_train = torch.from_numpy(train_values)
    y_train = torch.tensor(
        [sample["label"] for sample in train_samples],
        dtype=torch.long,
    )
    train_weights = torch.from_numpy(
        feature_experiments.sample_weights(train_samples, "current")
    )
    model = lightweight.StatsMlp(
        train_values.shape[1],
        24,
        True,
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
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), 128):
            batch = order[start:start + 128]
            losses, finger_logits, strength_logits = (
                lightweight.four_class_losses(
                    model,
                    x_train[batch],
                    y_train[batch],
                )
            )
            loss = (losses * train_weights[batch]).sum()
            loss = loss / train_weights[batch].sum()
            loss += 0.25 * (
                torch.nn.functional.cross_entropy(
                    finger_logits,
                    y_train[batch] // 2,
                )
                + torch.nn.functional.cross_entropy(
                    strength_logits,
                    y_train[batch] % 2,
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    return model


def aggregate(reports):
    """Report both central accuracy and the weakest cross-session result."""
    metrics = [report["validMetrics"] for report in reports]
    recalls = [
        value
        for item in metrics
        for value in item["recallByClass"].values()
    ]
    calibration = [
        report["calibration"]
        for report in reports
        if report.get("calibration")
        and report["calibration"].get("expectedCalibrationError")
        is not None
    ]
    selective = [
        report["selective"]
        for report in reports
        if report.get("selective")
        and report["selective"].get("validationCoverage") is not None
    ]
    ordinary = [
        report["ordinaryEvaluation"]
        for report in reports
        if report.get("ordinaryEvaluation")
        and report["ordinaryEvaluation"].get("ordinaryCount", 0) > 0
    ]
    ordinary_count = sum(
        item["ordinaryCount"] for item in ordinary
    )
    ordinary_false_acceptance = sum(
        item["falseAcceptanceCount"] or 0 for item in ordinary
    )
    layered_ordinary = [
        item
        for item in ordinary
        if item.get("layered")
    ]
    layered_count = sum(
        item["ordinaryCount"] for item in layered_ordinary
    )

    def layered_rate(count_key):
        count = sum(
            item["layered"][count_key] or 0
            for item in layered_ordinary
        )
        return (
            count / layered_count
            if layered_count
            else None
        )

    return {
        "meanBalancedAccuracy": float(np.mean([
            item["balancedAccuracy"] for item in metrics
        ])),
        "worstFoldBalancedAccuracy": float(np.min([
            item["balancedAccuracy"] for item in metrics
        ])),
        "meanAccuracy": float(np.mean([
            item["accuracy"] for item in metrics
        ])),
        "worstClassRecall": float(np.min(recalls)),
        "foldBalancedAccuracies": [
            item["balancedAccuracy"] for item in metrics
        ],
        "meanExpectedCalibrationError": (
            float(np.mean([
                item["expectedCalibrationError"]
                for item in calibration
            ]))
            if calibration
            else None
        ),
        "meanBrierScore": (
            float(np.mean([item["brierScore"] for item in calibration]))
            if calibration
            else None
        ),
        "meanSelectiveCoverage": (
            float(np.mean([
                item["validationCoverage"] for item in selective
            ]))
            if selective
            else None
        ),
        "meanSelectiveAccuracy": (
            float(np.mean([
                item["acceptedAccuracy"]
                for item in selective
                if item["acceptedAccuracy"] is not None
            ]))
            if any(
                item["acceptedAccuracy"] is not None
                for item in selective
            )
            else None
        ),
        "ordinaryFalseAcceptance": (
            {
                "ordinaryCount": int(ordinary_count),
                "falseAcceptanceCount": int(
                    ordinary_false_acceptance
                ),
                "falseAcceptanceRate": (
                    ordinary_false_acceptance / ordinary_count
                    if ordinary_count
                    else None
                ),
                "meanFoldFalseAcceptanceRate": float(np.mean([
                    item["falseAcceptanceRate"]
                    for item in ordinary
                    if item["falseAcceptanceRate"] is not None
                ])) if any(
                    item["falseAcceptanceRate"] is not None
                    for item in ordinary
                ) else None,
                "worstFoldFalseAcceptanceRate": (
                    float(max(
                        item["falseAcceptanceRate"]
                        for item in ordinary
                        if item["falseAcceptanceRate"] is not None
                    ))
                    if any(
                        item["falseAcceptanceRate"] is not None
                        for item in ordinary
                    )
                    else None
                ),
                "layeredFalseAcceptance": (
                    {
                        "finger": layered_rate(
                            "fingerFalseAcceptanceCount"
                        ),
                        "strength": layered_rate(
                            "strengthFalseAcceptanceCount"
                        ),
                        "anyHead": layered_rate(
                            "anyHeadFalseAcceptanceCount"
                        ),
                        "bothHeads": layered_rate(
                            "fullFalseAcceptanceCount"
                        ),
                    }
                    if layered_count
                    else None
                ),
            }
            if ordinary_count
            else None
        ),
    }


def predict_probabilities(model, values):
    """Return normalized four-class probabilities for runtime diagnostics."""
    model.eval()
    with torch.no_grad():
        logits = lightweight.logits_for_four_class(
            model,
            torch.from_numpy(values),
        )
        return torch.softmax(logits, dim=1).numpy()


def predict_head_probabilities(model, values):
    """Return finger and strength probabilities from the shared MLP."""
    if not model.multitask:
        raise RuntimeError("head_probabilities_require_multitask_model")
    model.eval()
    with torch.no_grad():
        finger_logits, strength_logits = model(torch.from_numpy(values))
        return (
            torch.softmax(finger_logits, dim=1).numpy(),
            torch.softmax(strength_logits, dim=1).numpy(),
        )


def ordinary_probability_metrics(
    probabilities,
    threshold,
    finger_threshold=None,
    strength_threshold=None,
    finger_probabilities=None,
    strength_probabilities=None,
):
    """Report both four-class and layered-head false acceptance."""
    if threshold is None:
        return {
            "threshold": None,
            "ordinaryCount": int(len(probabilities)),
            "falseAcceptanceRate": None,
            "falseAcceptanceCount": None,
            "meanConfidence": None,
            "medianConfidence": None,
            "p90Confidence": None,
            "predictedClassCounts": {},
            "layered": None,
        }
    result = false_acceptance_metrics(probabilities, threshold)
    if (
        finger_threshold is None
        or strength_threshold is None
        or finger_probabilities is None
        or strength_probabilities is None
    ):
        result["layered"] = None
        return result
    finger_probabilities = np.asarray(
        finger_probabilities,
        dtype=np.float64,
    )
    strength_probabilities = np.asarray(
        strength_probabilities,
        dtype=np.float64,
    )
    if (
        finger_probabilities.ndim != 2
        or finger_probabilities.shape[1] != 2
        or strength_probabilities.shape != finger_probabilities.shape
        or len(finger_probabilities) != len(probabilities)
    ):
        raise RuntimeError("invalid_head_probabilities")
    finger_pass = (
        finger_probabilities.max(axis=1) >= finger_threshold
    )
    strength_pass = (
        strength_probabilities.max(axis=1) >= strength_threshold
    )
    full_pass = finger_pass & strength_pass
    any_pass = finger_pass | strength_pass
    result["layered"] = {
        "fingerThreshold": float(finger_threshold),
        "strengthThreshold": float(strength_threshold),
        "fingerFalseAcceptanceCount": int(np.sum(finger_pass)),
        "fingerFalseAcceptanceRate": (
            float(np.mean(finger_pass)) if len(finger_pass) else None
        ),
        "strengthFalseAcceptanceCount": int(np.sum(strength_pass)),
        "strengthFalseAcceptanceRate": (
            float(np.mean(strength_pass))
            if len(strength_pass)
            else None
        ),
        "anyHeadFalseAcceptanceCount": int(np.sum(any_pass)),
        "anyHeadFalseAcceptanceRate": (
            float(np.mean(any_pass)) if len(any_pass) else None
        ),
        "fingerOnlyFalseAcceptanceCount": int(
            np.sum(finger_pass & ~strength_pass)
        ),
        "strengthOnlyFalseAcceptanceCount": int(
            np.sum(strength_pass & ~finger_pass)
        ),
        "fullFalseAcceptanceCount": int(np.sum(full_pass)),
        "fullFalseAcceptanceRate": (
            float(np.mean(full_pass)) if len(full_pass) else None
        ),
    }
    return result


def expected_calibration_error(truths, probabilities, bins=10):
    """Measure confidence calibration without selecting on validation labels."""
    if len(truths) == 0:
        return {
            "expectedCalibrationError": None,
            "brierScore": None,
            "negativeLogLikelihood": None,
            "binCounts": [],
        }
    probabilities = np.asarray(probabilities, dtype=np.float64)
    truths = np.asarray(truths, dtype=np.int64)
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == truths
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    counts = []
    ece = 0.0
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        selected = (
            (confidence >= lower)
            & (
                confidence < upper
                if index < bins - 1
                else confidence <= upper
            )
        )
        count = int(np.sum(selected))
        if count:
            accuracy = float(np.mean(correct[selected]))
            mean_confidence = float(np.mean(confidence[selected]))
            ece += count / len(truths) * abs(accuracy - mean_confidence)
        counts.append(count)
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(truths)), truths] = 1.0
    selected_truth = probabilities[
        np.arange(len(truths)),
        truths,
    ]
    return {
        "expectedCalibrationError": float(ece),
        "brierScore": float(np.mean(np.sum(
            (probabilities - one_hot) ** 2,
            axis=1,
        ))),
        "negativeLogLikelihood": float(np.mean(
            -np.log(np.clip(selected_truth, 1e-12, 1.0))
        )),
        "binCounts": counts,
    }


def selective_metrics(
    train_truths,
    train_probabilities,
    valid_truths,
    valid_probabilities,
    target_train_coverage=0.9,
):
    """Choose a confidence threshold on training data and score its tail."""
    thresholds = np.linspace(0.05, 0.95, 19)
    chosen = None
    for threshold in thresholds:
        train_coverage = float(np.mean(
            train_probabilities.max(axis=1) >= threshold
        ))
        if train_coverage >= target_train_coverage:
            chosen = float(threshold)
        else:
            break
    if chosen is None:
        return {
            "targetTrainCoverage": target_train_coverage,
            "threshold": None,
            "trainCoverage": None,
            "validationCoverage": None,
            "acceptedAccuracy": None,
            "acceptedBalancedAccuracy": None,
            "confidenceCurve": [],
        }
    accepted = valid_probabilities.max(axis=1) >= chosen
    accepted_count = int(np.sum(accepted))
    if accepted_count:
        predictions = valid_probabilities.argmax(axis=1)[accepted]
        truths = np.asarray(valid_truths)[accepted]
        metrics = lightweight.metrics(
            truths.tolist(),
            predictions.tolist(),
        )
        accepted_accuracy = metrics["accuracy"]
        accepted_balanced_accuracy = metrics["balancedAccuracy"]
    else:
        accepted_accuracy = None
        accepted_balanced_accuracy = None
    curve = []
    for threshold in thresholds:
        selected = valid_probabilities.max(axis=1) >= threshold
        count = int(np.sum(selected))
        accuracy = None
        if count:
            accuracy = float(np.mean(
                valid_probabilities.argmax(axis=1)[selected]
                == np.asarray(valid_truths)[selected]
            ))
        curve.append({
            "threshold": float(threshold),
            "coverage": count / len(valid_truths),
            "acceptedAccuracy": accuracy,
        })
    return {
        "targetTrainCoverage": target_train_coverage,
        "threshold": chosen,
        "trainCoverage": float(np.mean(
            train_probabilities.max(axis=1) >= chosen
        )),
        "validationCoverage": accepted_count / len(valid_truths),
        "acceptedAccuracy": accepted_accuracy,
        "acceptedBalancedAccuracy": accepted_balanced_accuracy,
        "acceptedCountsByClass": dict(Counter(
            lightweight.CLASS_LABELS[value]
            for value in np.asarray(valid_truths)[accepted]
        )),
        "confidenceCurve": curve,
    }


def binary_metrics(truths, predictions, labels):
    """Measure accepted quality for a two-way task head."""
    confusion = [[0 for _ in labels] for _ in labels]
    for truth, prediction in zip(truths, predictions):
        confusion[int(truth)][int(prediction)] += 1
    recalls = []
    for index in range(len(labels)):
        total = sum(confusion[index])
        recalls.append(
            confusion[index][index] / total if total else None
        )
    correct = sum(
        confusion[index][index] for index in range(len(labels))
    )
    known_recalls = [value for value in recalls if value is not None]
    return {
        "accuracy": correct / len(truths) if truths else None,
        "balancedAccuracy": (
            float(np.mean(known_recalls)) if known_recalls else None
        ),
        "recallByClass": dict(zip(labels, recalls)),
        "confusionMatrix": confusion,
    }


def false_acceptance_metrics(probabilities, threshold):
    """Measure whether ordinary touches would cross the positive-class gate."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise RuntimeError("invalid_probabilities")
    if threshold is None:
        raise RuntimeError("false_acceptance_threshold_missing")
    confidence = probabilities.max(axis=1)
    accepted = confidence >= threshold
    predictions = probabilities.argmax(axis=1)
    return {
        "threshold": float(threshold),
        "ordinaryCount": int(len(probabilities)),
        "falseAcceptanceRate": (
            float(np.mean(accepted)) if len(probabilities) else None
        ),
        "falseAcceptanceCount": int(np.sum(accepted)),
        "meanConfidence": (
            float(np.mean(confidence)) if len(confidence) else None
        ),
        "medianConfidence": (
            float(np.median(confidence)) if len(confidence) else None
        ),
        "p90Confidence": (
            float(np.percentile(confidence, 90))
            if len(confidence)
            else None
        ),
        "predictedClassCounts": dict(Counter(
            predictions.tolist()
        )),
    }


def selective_head_metrics(
    train_truths,
    train_probabilities,
    valid_truths,
    valid_probabilities,
    labels,
    target_train_coverage=0.9,
):
    """Select a head threshold on train data and score held-out touches."""
    thresholds = np.linspace(0.05, 0.95, 19)
    train_truths = np.asarray(train_truths, dtype=np.int64)
    valid_truths = np.asarray(valid_truths, dtype=np.int64)
    train_probabilities = np.asarray(
        train_probabilities,
        dtype=np.float64,
    )
    valid_probabilities = np.asarray(
        valid_probabilities,
        dtype=np.float64,
    )
    chosen = None
    for threshold in thresholds:
        train_coverage = float(np.mean(
            train_probabilities.max(axis=1) >= threshold
        ))
        if train_coverage >= target_train_coverage:
            chosen = float(threshold)
        else:
            break
    if chosen is None:
        return {
            "targetTrainCoverage": target_train_coverage,
            "threshold": None,
            "trainCoverage": None,
            "validationCoverage": None,
            "acceptedMetrics": None,
            "coverageByClass": dict.fromkeys(labels, None),
            "confidenceCurve": [],
        }
    accepted = valid_probabilities.max(axis=1) >= chosen
    accepted_truths = valid_truths[accepted]
    accepted_predictions = valid_probabilities.argmax(axis=1)[accepted]
    accepted_metrics = (
        binary_metrics(
            accepted_truths.tolist(),
            accepted_predictions.tolist(),
            labels,
        )
        if len(accepted_truths)
        else None
    )
    coverage_by_class = {}
    for index, label in enumerate(labels):
        selected = valid_truths == index
        coverage_by_class[label] = (
            float(np.mean(accepted[selected])) if np.any(selected) else None
        )
    curve = []
    for threshold in thresholds:
        selected = valid_probabilities.max(axis=1) >= threshold
        count = int(np.sum(selected))
        accuracy = None
        if count:
            accuracy = float(np.mean(
                valid_probabilities.argmax(axis=1)[selected]
                == valid_truths[selected]
            ))
        curve.append({
            "threshold": float(threshold),
            "coverage": count / len(valid_truths),
            "acceptedAccuracy": accuracy,
        })
    return {
        "targetTrainCoverage": target_train_coverage,
        "threshold": chosen,
        "trainCoverage": float(np.mean(
            train_probabilities.max(axis=1) >= chosen
        )),
        "validationCoverage": float(np.mean(accepted)),
        "acceptedMetrics": accepted_metrics,
        "coverageByClass": coverage_by_class,
        "confidenceCurve": curve,
    }


def layered_head_decision_metrics(
    valid_truths,
    finger_probabilities,
    strength_probabilities,
    finger_threshold,
    strength_threshold,
):
    """Evaluate full, finger-only, strength-only, and abstain outcomes."""
    valid_truths = np.asarray(valid_truths, dtype=np.int64)
    finger_probabilities = np.asarray(
        finger_probabilities,
        dtype=np.float64,
    )
    strength_probabilities = np.asarray(
        strength_probabilities,
        dtype=np.float64,
    )
    finger_accepted = (
        finger_probabilities.max(axis=1) >= finger_threshold
    )
    strength_accepted = (
        strength_probabilities.max(axis=1) >= strength_threshold
    )
    finger_truths = valid_truths // 2
    strength_truths = valid_truths % 2
    finger_predictions = finger_probabilities.argmax(axis=1)
    strength_predictions = strength_probabilities.argmax(axis=1)
    both = finger_accepted & strength_accepted
    finger_only = finger_accepted & ~strength_accepted
    strength_only = strength_accepted & ~finger_accepted
    abstain = ~finger_accepted & ~strength_accepted
    both_truths = valid_truths[both]
    both_predictions = (
        finger_predictions[both] * 2 + strength_predictions[both]
    )
    both_metrics = (
        lightweight.metrics(both_truths.tolist(), both_predictions.tolist())
        if len(both_truths)
        else None
    )
    finger_only_metrics = (
        binary_metrics(
            finger_truths[finger_only].tolist(),
            finger_predictions[finger_only].tolist(),
            ("thumb", "index"),
        )
        if np.any(finger_only)
        else None
    )
    strength_only_metrics = (
        binary_metrics(
            strength_truths[strength_only].tolist(),
            strength_predictions[strength_only].tolist(),
            ("light", "heavy"),
        )
        if np.any(strength_only)
        else None
    )
    total = len(valid_truths)
    return {
        "bothHeadCoverage": float(np.mean(both)),
        "bothHeadMetrics": both_metrics,
        "fingerOnlyCoverage": float(np.mean(finger_only)),
        "fingerOnlyMetrics": finger_only_metrics,
        "strengthOnlyCoverage": float(np.mean(strength_only)),
        "strengthOnlyMetrics": strength_only_metrics,
        "abstainCoverage": float(np.mean(abstain)),
        "decisionCounts": {
            "both": int(np.sum(both)),
            "fingerOnly": int(np.sum(finger_only)),
            "strengthOnly": int(np.sum(strength_only)),
            "abstain": int(np.sum(abstain)),
            "total": total,
        },
    }


def device_balanced_weights(samples):
    """Give each physical device the same mass during held-out-device training."""
    device_counts = Counter(
        str(sample.get("deviceName") or "unknown")
        for sample in samples
    )
    raw = np.asarray([
        1.0 / device_counts[str(sample.get("deviceName") or "unknown")]
        for sample in samples
    ], dtype=np.float32)
    device_names = np.asarray([
        str(sample.get("deviceName") or "unknown")
        for sample in samples
    ])
    labels = np.asarray([sample["label"] for sample in samples])
    # 中文注释：先按设备等权，再在每台设备内部按类别等权，
    # 保证样本量大的设备不会同时放大类别权重。
    for device in np.unique(device_names):
        for label in range(len(lightweight.CLASS_LABELS)):
            selected = (device_names == device) & (labels == label)
            if np.any(selected):
                raw[selected] *= (
                    1.0 / np.sum(raw[selected])
                )
    return np.clip(raw, 0.05, 20.0) / raw.mean()


def train_fixed_epochs_device_balanced(
    train_values,
    train_samples,
    epochs,
    seed,
):
    """Train the same MLP with equal physical-device and class mass."""
    lightweight.configure_determinism(seed)
    x_train = torch.from_numpy(train_values)
    y_train = torch.tensor(
        [sample["label"] for sample in train_samples],
        dtype=torch.long,
    )
    train_weights = torch.from_numpy(
        device_balanced_weights(train_samples)
    )
    model = lightweight.StatsMlp(
        train_values.shape[1],
        24,
        True,
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
    for epoch in range(1, epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), 128):
            batch = order[start:start + 128]
            losses, finger_logits, strength_logits = (
                lightweight.four_class_losses(
                    model,
                    x_train[batch],
                    y_train[batch],
                )
            )
            loss = (losses * train_weights[batch]).sum()
            loss = loss / train_weights[batch].sum()
            loss += 0.25 * (
                torch.nn.functional.cross_entropy(
                    finger_logits,
                    y_train[batch] // 2,
                )
                + torch.nn.functional.cross_entropy(
                    strength_logits,
                    y_train[batch] % 2,
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    return model


def session_size_probe(samples):
    """Measure whether raw size identifies sessions on held-out touches."""
    by_session = defaultdict(list)
    for sample in samples:
        current = sample
        if isinstance(sample["matrix"], list):
            current = dict(sample)
            current["matrix"] = np.asarray(
                sample["matrix"],
                dtype=np.float32,
            )
        values = valid_size_values(current)
        if len(values):
            by_session[sample["groupId"]].append(float(np.median(values)))
    fit_centers = {}
    probes = []
    for session, values in by_session.items():
        ordered = sorted(values)
        split = max(1, int(round(len(ordered) * 0.7)))
        if split >= len(ordered):
            continue
        fit_centers[session] = float(np.median(ordered[:split]))
        probes.extend((session, value) for value in ordered[split:])
    if not probes or len(fit_centers) < 2:
        return {
            "sessionCount": len(fit_centers),
            "probeCount": len(probes),
            "accuracy": None,
            "chanceAccuracy": None,
        }
    session_names = list(fit_centers)
    centers = np.asarray([fit_centers[name] for name in session_names])
    correct = 0
    for truth, value in probes:
        prediction = session_names[int(np.argmin(np.abs(centers - value)))]
        correct += int(prediction == truth)
    return {
        "sessionCount": len(session_names),
        "probeCount": len(probes),
        "accuracy": correct / len(probes),
        "chanceAccuracy": 1.0 / len(session_names),
    }


def feature_vectors(sample):
    """Flatten only the ten runtime channels for identity diagnostics."""
    matrix = sample["matrix"]
    if isinstance(matrix, list):
        matrix = np.asarray(matrix, dtype=np.float32)
    return matrix[:, lightweight.PHYSICAL_INDICES].reshape(-1)


def grouped_session_identity_probe(samples, size_only, seed):
    """Test whether a compact model can recognize held-out sessions."""
    session_names = sorted({sample["groupId"] for sample in samples})
    if len(session_names) < 4:
        return {
            "sessionCount": len(session_names),
            "trainCount": 0,
            "testCount": 0,
            "accuracy": None,
            "chanceAccuracy": None,
        }
    rng = np.random.default_rng(seed)
    # 验证标签必须来自已见 session；未见于训练集的类别不能被模型输出，
    # 否则该指标会被结构性压到零，无法回答“通道能否识别 session”。
    shuffled = list(session_names)
    rng.shuffle(shuffled)
    session_to_label = {
        session: index for index, session in enumerate(shuffled)
    }
    session_indices = defaultdict(list)
    for index, sample in enumerate(samples):
        session_indices[sample["groupId"]].append(index)
    valid_sessions = set(shuffled[:max(2, int(round(len(shuffled) * 0.25)))])
    train_indices = []
    valid_indices = []
    for session, indices in session_indices.items():
        ordered = sorted(
            indices,
            key=lambda index: (
                samples[index].get("sourceFile", ""),
                samples[index]["sourceSha256"],
            ),
        )
        split = max(1, int(round(len(ordered) * 0.7)))
        if split >= len(ordered):
            split = len(ordered) - 1
        if split <= 0:
            continue
        train_indices.extend(ordered[:split])
        if session in valid_sessions:
            valid_indices.extend(ordered[split:])
    if not train_indices or not valid_indices:
        return {
            "sessionCount": len(session_names),
            "trainCount": len(train_indices),
            "testCount": len(valid_indices),
            "accuracy": None,
            "chanceAccuracy": None,
        }
    train_samples = [samples[index] for index in train_indices]
    valid_samples = [samples[index] for index in valid_indices]
    train_labels = sorted({sample["groupId"] for sample in train_samples})
    label_index = session_to_label
    if size_only:
        train_x = np.stack([
            np.asarray([
                frame[PHYSICAL_SIZE_CHANNEL]
                for frame in sample["matrix"]
            ])
            for sample in train_samples
        ])
        valid_x = np.stack([
            np.asarray([
                frame[PHYSICAL_SIZE_CHANNEL]
                for frame in sample["matrix"]
            ])
            for sample in valid_samples
        ])
    else:
        train_x = np.stack([
            feature_vectors(sample) for sample in train_samples
        ])
        valid_x = np.stack([
            feature_vectors(sample) for sample in valid_samples
        ])
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    valid_x = scaler.transform(valid_x)
    train_y = np.asarray([
        label_index[sample["groupId"]] for sample in train_samples
    ])
    model = LogisticRegression(
        max_iter=300,
        C=0.5,
        random_state=seed,
    )
    model.fit(train_x, train_y)
    predictions = model.predict(valid_x)
    correct = sum(
        int(train_labels[int(prediction)] == sample["groupId"])
        for prediction, sample in zip(predictions, valid_samples)
    )
    predictions_by_session = defaultdict(list)
    for prediction, sample in zip(predictions, valid_samples):
        predictions_by_session[sample["groupId"]].append(
            train_labels[int(prediction)]
        )
    correct_by_session = sum(
        int(
            Counter(predictions_by_session[session]).most_common(1)[0][0]
            == session
        )
        for session in valid_sessions
    )
    return {
        "sessionCount": len(train_labels),
        "heldOutSessionCount": len(valid_sessions),
        "trainCount": len(train_samples),
        "testCount": len(valid_samples),
        "accuracy": correct / len(valid_samples),
        "sessionVoteAccuracy": (
            correct_by_session / len(valid_sessions)
        ),
        "chanceAccuracy": 1.0 / len(train_labels),
        "sizeOnly": size_only,
    }


def size_channel_audit(rows):
    """Summarize how much size varies by session and target class."""
    size_index = SOURCE_SIZE_CHANNEL
    available_index = SOURCE_SIZE_AVAILABLE_CHANNEL
    values = []
    sessions = []
    labels = []
    for row in rows:
        matrix = row["matrix"]
        for frame in matrix:
            if frame[available_index] >= 0.5 and frame[size_index] > 0.0:
                values.append(frame[size_index])
                sessions.append(row["groupId"])
                labels.append(row["classLabel"])
    values = np.asarray(values, dtype=np.float64)
    sessions = np.asarray(sessions)
    labels = np.asarray(labels)
    total_mean = float(values.mean())
    _, session_codes = np.unique(sessions, return_inverse=True)
    session_counts = np.bincount(session_codes)
    session_means = (
        np.bincount(session_codes, weights=values) / session_counts
    )
    total_ss = float(np.sum((values - total_mean) ** 2))
    between_ss = float(np.sum(
        (session_means[session_codes] - total_mean) ** 2
    ))
    class_between = 0.0
    for label in lightweight.CLASS_LABELS:
        selected = values[labels == label]
        if len(selected):
            class_between += len(selected) * (
                float(selected.mean()) - total_mean
            ) ** 2
    return {
        "validFrameCount": int(len(values)),
        "sessionCount": int(len(session_counts)),
        "sessionVarianceShare": between_ss / total_ss if total_ss else 0.0,
        "classEffectShare": (
            class_between / total_ss if total_ss else 0.0
        ),
        "classFrameCounts": dict(Counter(labels.tolist())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--imu-mode",
        choices=("raw", "gravity_aligned"),
        default="raw",
    )
    parser.add_argument(
        "--modes",
        default="",
        help="Comma-separated subset of size experiment names.",
    )
    parser.add_argument(
        "--weighting",
        choices=("session_balanced", "device_balanced"),
        default="session_balanced",
    )
    parser.add_argument(
        "--require-training-ready",
        action="store_true",
        help=(
            "Fail before training unless metadata, interleaved sessions, "
            "and posture holdout gates are ready."
        ),
    )
    args = parser.parse_args()
    if args.epochs <= 0:
        raise RuntimeError("invalid_epochs")
    rows = read_dataset_rows(args.dataset_dir)
    folds = load_samples(args.dataset_dir)
    ordinary_samples = load_ordinary_samples(args.dataset_dir)
    for samples in folds.values():
        for sample in samples[0] + samples[1]:
            sample["datasetId"] = str(Path(args.dataset_dir).resolve())
    default_modes = (
        "baseline",
        "shape_only",
        "within_touch_median",
        "within_touch_iqr_z",
        "session_median_oracle",
        "session_iqr_z_oracle",
        "causal_median_4",
        "causal_median_8",
        "causal_median_16",
        "major_ratio",
        "major_within_touch_median",
        "declared_major_range",
        "presence",
        "device_median_oracle",
        "device_iqr_z_oracle",
        "causal_device_median_8",
        "causal_device_median_16",
        "causal_device_median_32",
    )
    tail_modes = tuple(
        "causal_device_median_" + str(count) + "_tail"
        for count in TAIL_CALIBRATION_COUNTS
    )
    modes = tuple(
        mode.strip()
        for mode in args.modes.split(",")
        if mode.strip()
    ) if args.modes.strip() else default_modes
    known_modes = set(default_modes).union(tail_modes)
    unknown_modes = set(modes) - known_modes
    if unknown_modes:
        raise RuntimeError(
            "unknown_size_modes:" + ",".join(sorted(unknown_modes))
        )
    is_cross_device = fold_evaluation_scope(folds) == "cross_device"
    cleaning_report = json.loads(
        (
            Path(args.dataset_dir).resolve()
            / "training_cleaning_report.json"
        ).read_text(encoding="utf-8")
    )
    readiness = cleaning_report.get("trainingReadiness") or {}
    if args.require_training_ready and not readiness.get("ready", False):
        raise RuntimeError(
            "training_readiness_gate_failed:"
            + ",".join(
                readiness.get("blockingReasons") or ["unknown"]
            )
        )
    fold_kind = str(
        cleaning_report.get("validationFoldKind") or "unknown"
    )
    negative_audit = ordinary_negative_audit(
        ordinary_samples,
        folds,
        fold_kind,
    )
    output = {
        "scope": fold_evaluation_scope(folds),
        "foldMetadata": fold_metadata(folds),
        "imuMode": args.imu_mode,
        "weighting": args.weighting,
        "poseAlignment": (
            "per_frame_gravity_frame_device_x_projected_horizontal"
            if args.imu_mode == "gravity_aligned"
            else "raw_device_axes"
        ),
        "touchStartFrame": TOUCH_START,
        "epochSelection": "fixed_without_validation_labels",
        "epochs": args.epochs,
        "selectionMetric": (
            "worst_fold_balanced_accuracy_then_worst_class_recall"
        ),
        "trainingReadiness": readiness,
        "ordinaryNegativeAudit": {
            **negative_audit,
            "usage": "optional_false_acceptance_diagnostic",
            "foldMatching": (
                "held_out_device"
                if fold_kind == "held_out_device"
                else "held_out_session"
            ),
            "matchedByMode": {},
        },
        "sizeAudit": size_channel_audit(rows),
        "sizeSessionProbe": session_size_probe(rows),
        "groupedSessionIdentityProbe": {
            "allPhysicalChannels": grouped_session_identity_probe(
                [
                    {
                        **row,
                        "matrix": np.asarray(
                            row["matrix"],
                            dtype=np.float32,
                        ),
                    }
                    for row in rows
                ],
                size_only=False,
                seed=20261022,
            ),
            "sizeOnly": grouped_session_identity_probe(
                [
                    {
                        **row,
                        "matrix": np.asarray(
                            row["matrix"],
                            dtype=np.float32,
                        ),
                    }
                    for row in rows
                ],
                size_only=True,
                seed=20261022,
            ),
        },
        "experiments": {},
        "tailCalibrationModes": list(tail_modes),
    }
    for mode in modes:
        reports = []
        for fold, (train_samples, valid_samples) in sorted(folds.items()):
            print(json.dumps({
                "mode": mode,
                "fold": fold,
            }), flush=True)
            prepare = (
                prepare_fold_leave_one_device_out
                if is_cross_device
                else prepare_fold
            )
            pose_mode = (
                "gravity_aligned_axes"
                if args.imu_mode == "gravity_aligned"
                else "raw"
            )
            score_indices = None
            if mode in tail_modes:
                if not is_cross_device:
                    raise RuntimeError(
                        "tail_calibration_requires_cross_device_folds"
                    )
                (
                    train_values,
                    valid_values,
                    score_indices,
                ) = prepare_fold_leave_one_device_out_calibrated_tail(
                    train_samples,
                    valid_samples,
                    mode,
                    pose_mode=pose_mode,
                )
            else:
                train_values, valid_values = prepare(
                    train_samples,
                    valid_samples,
                    mode,
                    pose_mode=pose_mode,
                )
            trainer = (
                train_fixed_epochs_device_balanced
                if args.weighting == "device_balanced"
                else train_fixed_epochs
            )
            model = trainer(
                train_values,
                train_samples,
                args.epochs,
                20261022 + fold,
            )
            train_metrics = lightweight.metrics(
                [sample["label"] for sample in train_samples],
                lightweight.predict(
                    model,
                    torch.from_numpy(train_values),
                ),
            )
            scored_samples = (
                valid_samples
                if score_indices is None
                else [valid_samples[index] for index in score_indices]
            )
            scored_values = (
                valid_values
                if score_indices is None
                else valid_values[score_indices]
            )
            probabilities = predict_probabilities(model, scored_values)
            valid_metrics = lightweight.metrics(
                [sample["label"] for sample in scored_samples],
                probabilities.argmax(axis=1).tolist(),
            )
            calibration = expected_calibration_error(
                [sample["label"] for sample in scored_samples],
                probabilities,
            )
            train_probabilities = predict_probabilities(
                model,
                train_values,
            )
            selective = selective_metrics(
                [sample["label"] for sample in train_samples],
                train_probabilities,
                [sample["label"] for sample in scored_samples],
                probabilities,
            )
            train_finger_probabilities, train_strength_probabilities = (
                predict_head_probabilities(model, train_values)
            )
            valid_finger_probabilities, valid_strength_probabilities = (
                predict_head_probabilities(model, scored_values)
            )
            head_selective = {
                "finger": selective_head_metrics(
                    [
                        sample["label"] // 2
                        for sample in train_samples
                    ],
                    train_finger_probabilities,
                    [
                        sample["label"] // 2
                        for sample in scored_samples
                    ],
                    valid_finger_probabilities,
                    ("thumb", "index"),
                ),
                "strength": selective_head_metrics(
                    [
                        sample["label"] % 2
                        for sample in train_samples
                    ],
                    train_strength_probabilities,
                    [
                        sample["label"] % 2
                        for sample in scored_samples
                    ],
                    valid_strength_probabilities,
                    ("light", "heavy"),
                ),
            }
            finger_threshold = head_selective["finger"]["threshold"]
            strength_threshold = head_selective["strength"]["threshold"]
            layered_decision = (
                layered_head_decision_metrics(
                    [sample["label"] for sample in scored_samples],
                    valid_finger_probabilities,
                    valid_strength_probabilities,
                    finger_threshold,
                    strength_threshold,
                )
                if finger_threshold is not None
                and strength_threshold is not None
                else None
            )
            fold_ordinary_samples = ordinary_samples_for_fold(
                ordinary_samples,
                valid_samples,
                fold_kind,
            )
            ordinary_domain_sample_count = len(fold_ordinary_samples)
            ordinary_evaluation = None
            if fold_ordinary_samples:
                (
                    fold_ordinary_samples,
                    ordinary_values,
                ) = prepare_ordinary_values(
                    train_samples,
                    valid_samples,
                    fold_ordinary_samples,
                    mode,
                    prepare,
                    pose_mode,
                    is_cross_device,
                )
                if fold_ordinary_samples:
                    ordinary_probabilities = predict_probabilities(
                        model,
                        ordinary_values,
                    )
                    ordinary_finger_probabilities = None
                    ordinary_strength_probabilities = None
                    if (
                        finger_threshold is not None
                        and strength_threshold is not None
                    ):
                        (
                            ordinary_finger_probabilities,
                            ordinary_strength_probabilities,
                        ) = predict_head_probabilities(
                            model,
                            ordinary_values,
                        )
                    ordinary_evaluation = ordinary_probability_metrics(
                        ordinary_probabilities,
                        selective["threshold"],
                        finger_threshold,
                        strength_threshold,
                        ordinary_finger_probabilities,
                        ordinary_strength_probabilities,
                    )
            reports.append({
                "fold": fold,
                "validDevices": sorted({
                    sample.get("deviceName", "")
                    for sample in scored_samples
                }),
                "validPostures": sorted({
                    sample.get("bodyPosture", "unknown")
                    for sample in scored_samples
                }),
                "parameterCount": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                ),
                "trainMetrics": train_metrics,
                "validMetrics": valid_metrics,
                "scoredCount": len(scored_samples),
                "calibrationWarmupCount": (
                    len(valid_samples) - len(scored_samples)
                    if score_indices is not None
                    else 0
                ),
                "ordinaryDomainSampleCount": (
                    ordinary_domain_sample_count
                ),
                "ordinarySampleCount": len(fold_ordinary_samples),
                "calibration": calibration,
                "selective": selective,
                "headSelective": head_selective,
                "layeredDecision": layered_decision,
                "ordinaryEvaluation": ordinary_evaluation,
            })
        output["experiments"][mode] = {
            "aggregate": aggregate(reports),
            "folds": reports,
        }
        output["ordinaryNegativeAudit"]["matchedByMode"][mode] = (
            ordinary_negative_mode_audit(
                negative_audit,
                reports,
            )
        )
    pose_reports = []
    for fold, (train_samples, valid_samples) in sorted(folds.items()):
        print(json.dumps({
            "mode": "pose_aligned_baseline",
            "fold": fold,
        }), flush=True)
        train_values, valid_values = prepare_fold(
            train_samples,
            valid_samples,
            "baseline",
            pose_mode="gravity_aligned_axes",
        )
        model = train_fixed_epochs(
            train_values,
            train_samples,
            args.epochs,
            20261022 + fold,
        )
        train_metrics = lightweight.metrics(
            [sample["label"] for sample in train_samples],
            lightweight.predict(
                model,
                torch.from_numpy(train_values),
            ),
        )
        valid_metrics = lightweight.metrics(
            [sample["label"] for sample in valid_samples],
            lightweight.predict(
                model,
                torch.from_numpy(valid_values),
            ),
        )
        pose_reports.append({
            "fold": fold,
            "parameterCount": sum(
                parameter.numel()
                for parameter in model.parameters()
            ),
            "trainMetrics": train_metrics,
            "validMetrics": valid_metrics,
        })
    output["experiments"]["pose_aligned_baseline"] = {
        "aggregate": aggregate(pose_reports),
        "folds": pose_reports,
    }
    oracle_modes = {
        "session_median_oracle",
        "session_iqr_z_oracle",
        "device_median_oracle",
        "device_iqr_z_oracle",
    }
    deployment_eligible = [
        name
        for name in output["experiments"]
        if name not in oracle_modes
        and name != "pose_aligned_baseline"
    ]
    if args.imu_mode == "gravity_aligned":
        deployment_eligible = [
            name
            for name in deployment_eligible
            if name != "baseline"
        ]
    ranked = sorted(
        (
            (name, output["experiments"][name])
            for name in deployment_eligible
        ),
        key=lambda item: (
            item[1]["aggregate"]["worstFoldBalancedAccuracy"],
            item[1]["aggregate"]["worstClassRecall"],
            item[1]["aggregate"]["meanBalancedAccuracy"],
        ),
        reverse=True,
    )
    output["ranking"] = [name for name, _ in ranked]
    output["deploymentEligibleModes"] = deployment_eligible
    output["oracleModes"] = sorted(oracle_modes)
    # 中文注释：模式名称只表示实验可以复现；数据协议和指标门禁
    # 未全部通过时，不能把排名第一的实验模式称为部署候选。
    best_mode = output["ranking"][0] if output["ranking"] else None
    gate_reasons = list(
        readiness.get("blockingReasons")
        or ([] if readiness.get("ready", False) else ["training_readiness"])
    )
    ordinary_gate = None
    if best_mode is not None:
        ordinary_gate = output["experiments"][best_mode]["aggregate"].get(
            "ordinaryFalseAcceptance"
        )
        # 中文注释：普通点击不再是当前采集协议的一部分。
        # 历史负样本只输出诊断，不参与部署准入判断。
    output["deploymentGate"] = {
        "eligible": bool(readiness.get("ready", False)) and not gate_reasons,
        "blockingReasons": sorted(set(gate_reasons)),
        "bestReproducibleMode": best_mode,
        "ordinaryFalseAcceptance": ordinary_gate,
    }
    output["runtimeDecisionPolicy"] = {
        "defaultModel": "StatsMlp24",
        "confidenceGate": (
            "accept only probabilities above the threshold selected on "
            "training coverage; expose low confidence to the caller"
        ),
        "layeredHeads": {
            "full": "emit four classes only when finger and strength pass",
            "fingerOnly": "emit finger when only the finger head passes",
            "strengthOnly": (
                "emit strength when only the strength head passes"
            ),
            "abstain": "return no class when both heads fail",
        },
        "lowConfidenceAction": (
            "request another touch or use an explicit fallback; never "
            "force a four-class answer solely because argmax exists"
        ),
        "cascadePolicy": (
            "add a larger model only after held-out device and posture "
            "tests show a material gain at acceptable latency"
        ),
    }
    path = Path(args.output_dir) / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(path),
        "ranking": [
            {
                "mode": name,
                **experiment["aggregate"],
            }
            for name, experiment in ranked
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
