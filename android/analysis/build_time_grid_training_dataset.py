#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .collection_annotations import AUDIT_FIELDS
    from . import touch_label_quality
except ImportError:
    from collection_annotations import AUDIT_FIELDS
    import touch_label_quality

CLASS_LABELS = (
    "thumb_light",
    "thumb_heavy",
    "index_light",
    "index_heavy",
)
ORDINARY_CLASS_LABEL = "ordinary_touch"
CONTRACT_IMU11 = "causal_time_grid_imu11_v2"
CONTRACT_TOUCH20 = "causal_time_grid_imu11_touch9_v2"
CONTRACT_TOUCH26 = "causal_time_grid_imu11_touch9_mask6_v4"
CONTRACT_TOUCH23 = "causal_time_grid_imu11_touch7_mask5_v5"
MASK_SOURCE_NAMES = ("pressure", "size", "major", "minor", "orientation", "pointerCount")
MASK_FEATURE_NAMES = (
    "pressure_available",
    "size_available",
    "major_available",
    "minor_available",
    "orientation_available",
    "pointer_count_available",
)
V5_MASK_SOURCE_NAMES = ("pressure", "size", "major", "minor", "pointerCount")
V5_MASK_FEATURE_NAMES = (
    "pressure_available",
    "size_available",
    "major_available",
    "minor_available",
    "pointer_count_available",
)
IMU_FEATURE_NAMES = (
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
TOUCH_FEATURE_NAMES = (
    "pressure",
    "size",
    "major_norm",
    "minor_norm",
    "log_axis_ratio",
    "orientation_sin",
    "orientation_cos",
    "pointer_count",
    "touch_valid",
)
V5_TOUCH_FEATURE_NAMES = (
    "pressure",
    "size",
    "major_norm",
    "minor_norm",
    "log_axis_ratio",
    "pointer_count",
    "touch_valid",
)
CONTRACT_FEATURE_NAMES = {
    CONTRACT_IMU11: IMU_FEATURE_NAMES,
    CONTRACT_TOUCH26: IMU_FEATURE_NAMES + TOUCH_FEATURE_NAMES + MASK_FEATURE_NAMES,
    CONTRACT_TOUCH23: IMU_FEATURE_NAMES + V5_TOUCH_FEATURE_NAMES + V5_MASK_FEATURE_NAMES,
}
CANDIDATE_FIELDS = (
    "sourceFile",
    "sourceSha256",
    "sampleId",
    "classLabel",
    "motionState",
    "collectorUserName",
    "collectorDeviceId",
    "deviceName",
    "platform",
    "sensorSource",
    "sessionId",
    "groupId",
    "sourceTimestampMs",
    "trainingEligibility",
    "exclusionReason",
    "trainingWeight",
    "exactFeatureFingerprint",
    "duplicateClusterId",
    "duplicateClusterSize",
) + AUDIT_FIELDS
FOLD_FIELDS = (
    "fold",
    "split",
    "foldKind",
    "sourceSha256",
    "sampleId",
    "classLabel",
    "groupId",
    "trainingWeight",
)
POSE_FOLD_FIELDS = FOLD_FIELDS + (
    "bodyPosture",
)
CONTEXT_FIELDS = (
    "groupId",
    "platform",
    "collectorDeviceId",
    "deviceName",
    "sessionId",
    "taskId",
    "bodyPosture",
    "requestedDevicePose",
    "grip",
    "support",
    "collectionProtocol",
    "sampleCount",
    "classCount",
    "classLabels",
    "classCounts",
    "isMultiClassSession",
)
MISSINGNESS_FIELDS = (
    "maskName",
    "availableFrameCount",
    "missingFrameCount",
    "totalFrameCount",
    "missingRate",
)


def build_training_dataset(derived_dir, output_dir):
    derived_root = Path(derived_dir).resolve()
    output_root = Path(output_dir).resolve()
    if not derived_root.is_dir():
        raise RuntimeError("derived_directory_not_found")
    if output_root == derived_root or derived_root in output_root.parents:
        raise RuntimeError("output_directory_must_not_contain_derived_sources")

    source_rows = read_csv(derived_root / "source_manifest.csv")
    label_quality_report = touch_label_quality.read_quality_report(
        derived_root
    )
    source_by_hash = accepted_sources(source_rows)
    imu_rows = read_jsonl(derived_root / "canonical_imu11.jsonl")
    touch_rows = read_jsonl(
        derived_root / "canonical_imu11_touch7_mask5_v5.jsonl"
    )
    legacy_touch_path = (
        derived_root / "canonical_imu11_touch9_mask6_v4.jsonl"
    )
    # 中文注释：v4 仅用于复现历史实验，不是新候选的必需输入。
    # 缺失时仍应能构建 v5 训练集，避免旧格式阻塞主线。
    legacy_touch_rows = (
        read_jsonl(legacy_touch_path)
        if legacy_touch_path.is_file()
        else []
    )
    imu_by_hash = matrix_rows_by_hash(imu_rows, CONTRACT_IMU11, 11)
    touch_by_hash = matrix_rows_by_hash(
        touch_rows,
        CONTRACT_TOUCH23,
        23,
    )
    legacy_touch_by_hash = (
        matrix_rows_by_hash(
            legacy_touch_rows,
            CONTRACT_TOUCH26,
            26,
        )
        if legacy_touch_rows
        else {}
    )
    validate_source_matrix_identity(source_by_hash, imu_by_hash, touch_by_hash)
    if legacy_touch_by_hash:
        validate_source_matrix_identity(
            source_by_hash,
            imu_by_hash,
            legacy_touch_by_hash,
        )

    candidates = []
    for source_sha256, source in sorted(source_by_hash.items(), key=lambda item: item[1]["sourceFile"]):
        imu = imu_by_hash[source_sha256]
        touch = touch_by_hash[source_sha256]
        candidate = candidate_record(source, imu, touch)
        candidates.append(candidate)
    assign_exact_duplicate_weights(candidates, touch_by_hash)
    eligible = [candidate for candidate in candidates if candidate["trainingEligibility"] == "eligible"]
    ordinary_negatives = [
        candidate
        for candidate in candidates
        if (
            candidate["classLabel"] == ORDINARY_CLASS_LABEL
            and candidate["trainingEligibility"] == "excluded"
            and candidate["exclusionReason"]
            == "ordinary_touch_negative_sample"
        )
    ]
    folds, fold_count, fold_status, fold_kind = build_validation_folds(
        eligible
    )
    pose_folds, pose_fold_status = build_pose_holdout_folds(eligible)

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "training_candidate_manifest.csv", CANDIDATE_FIELDS, candidates)
    write_csv(
        output_root / "training_excluded_manifest.csv",
        CANDIDATE_FIELDS,
        [candidate for candidate in candidates if candidate["trainingEligibility"] != "eligible"],
    )
    write_jsonl(
        output_root / "training_imu11.jsonl",
        [training_imu_record(candidate, imu_by_hash[candidate["sourceSha256"]]) for candidate in eligible],
    )
    write_jsonl(
        output_root / "training_touch23.jsonl",
        [training_touch23_record(candidate, touch_by_hash[candidate["sourceSha256"]]) for candidate in eligible],
    )
    # 中文注释：仅在 v4 输入存在时保留冻结视图；
    # 新候选必须显式读取 training_touch23。
    if legacy_touch_by_hash:
        write_jsonl(
            output_root / "training_touch26.jsonl",
            [
                training_touch26_record(
                    candidate,
                    legacy_touch_by_hash[candidate["sourceSha256"]],
                )
                for candidate in eligible
            ],
        )
    # 中文注释：普通点击不进入监督训练，但必须保留同一特征合同，
    # 才能在对应验证设备或会话上测量模型误接受率。
    write_jsonl(
        output_root / "ordinary_touch23.jsonl",
        [
            training_touch23_record(
                candidate,
                touch_by_hash[candidate["sourceSha256"]],
            )
            for candidate in ordinary_negatives
        ],
    )
    if legacy_touch_by_hash:
        write_jsonl(
            output_root / "ordinary_touch26.jsonl",
            [
                training_touch26_record(
                    candidate,
                    legacy_touch_by_hash[candidate["sourceSha256"]],
                )
                for candidate in ordinary_negatives
            ],
        )
    write_csv(output_root / "strict_group_folds.csv", FOLD_FIELDS, folds)
    write_csv(
        output_root / "pose_holdout_folds.csv",
        POSE_FOLD_FIELDS,
        pose_folds,
    )
    write_csv(
        output_root / "collection_context_audit.csv",
        CONTEXT_FIELDS,
        context_audit_rows(eligible),
    )
    write_csv(output_root / "session_label_coverage.csv", None, session_coverage_rows(eligible))
    missingness = missingness_rows(eligible, touch_by_hash)
    write_csv(output_root / "feature_missingness_report.csv", MISSINGNESS_FIELDS, missingness)
    write_csv(output_root / "near_duplicate_audit.csv", CANDIDATE_FIELDS, candidates)
    write_training_contract(output_root / "training_feature_contract.json")
    write_source_hash_manifest(output_root / "source_hash_manifest.sha256", candidates)
    summary = summary_record(
        candidates,
        eligible,
        fold_count,
        fold_status,
        fold_kind,
        missingness,
        pose_folds,
        pose_fold_status,
        ordinary_negatives,
        label_quality_report,
    )
    (output_root / "training_cleaning_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def read_csv(path):
    if not path.is_file():
        raise RuntimeError("missing_source_manifest")
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def accepted_sources(rows):
    selected = {}
    for row in rows:
        if row.get("normalizationStatus") != "accepted":
            continue
        source_sha256 = str(row.get("sourceSha256") or "")
        if not source_sha256:
            raise RuntimeError("accepted_source_missing_sha256")
        if source_sha256 in selected:
            raise RuntimeError("duplicate_accepted_source_sha256")
        selected[source_sha256] = row
    if not selected:
        raise RuntimeError("no_accepted_sources")
    return selected


def read_jsonl(path):
    if not path.is_file():
        raise RuntimeError("missing_canonical_matrix")
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError("invalid_canonical_jsonl_line_" + str(line_number)) from error
            if not isinstance(value, dict):
                raise RuntimeError("invalid_canonical_jsonl_record")
            rows.append(value)
    return rows


def matrix_rows_by_hash(rows, contract, width):
    selected = {}
    for row in rows:
        if row.get("featureContract") != contract:
            raise RuntimeError("unexpected_matrix_feature_contract")
        source_sha256 = str(row.get("sourceSha256") or "")
        if not source_sha256 or source_sha256 in selected:
            raise RuntimeError("duplicate_or_missing_matrix_source_sha256")
        validate_matrix(row, width)
        selected[source_sha256] = row
    return selected


def validate_matrix(row, width):
    matrix = row.get("matrix")
    feature_names = row.get("featureNames")
    if (
        row.get("matrixShape") != [9, width]
        or not isinstance(matrix, list)
        or len(matrix) != 9
        or not isinstance(feature_names, list)
        or len(feature_names) != width
        or any(not isinstance(name, str) or not name for name in feature_names)
    ):
        raise RuntimeError("invalid_time_grid_matrix_shape")
    expected_names = CONTRACT_FEATURE_NAMES.get(str(row.get("featureContract") or ""))
    if expected_names is None or list(expected_names) != feature_names:
        raise RuntimeError("unexpected_matrix_feature_names")
    for values in matrix:
        if not isinstance(values, list) or len(values) != width:
            raise RuntimeError("invalid_time_grid_matrix_shape")
        for value in values:
            if not finite(value):
                raise RuntimeError("non_finite_time_grid_feature")
    if width in (23, 26):
        masks = row.get("missingMasks")
        if not isinstance(masks, list) or len(masks) != 9:
            raise RuntimeError("invalid_missing_masks")
        mask_names = (
            V5_MASK_SOURCE_NAMES
            if width == 23
            else MASK_SOURCE_NAMES
        )
        for mask in masks:
            if not isinstance(mask, dict) or any(mask.get(name) not in (0, 1) for name in mask_names):
                raise RuntimeError("invalid_missing_masks")


def validate_source_matrix_identity(source_by_hash, imu_by_hash, touch_by_hash):
    if set(imu_by_hash) != set(touch_by_hash):
        raise RuntimeError("matrix_source_identity_mismatch")
    if set(source_by_hash) != set(imu_by_hash):
        raise RuntimeError("accepted_source_matrix_missing")
    identity_fields = ("sourceFile", "sampleId", "classLabel", "sessionId", "platform", "collectorDeviceId") + AUDIT_FIELDS
    for source_sha256, source in source_by_hash.items():
        imu = imu_by_hash[source_sha256]
        touch = touch_by_hash[source_sha256]
        for field in identity_fields:
            source_value = str(source.get(field) or "")
            if str(imu.get(field) or "") != source_value or str(touch.get(field) or "") != source_value:
                raise RuntimeError("matrix_source_identity_mismatch")


def candidate_record(source, imu, touch):
    class_label = str(source.get("classLabel") or "")
    group_id = ":".join((
        str(source.get("platform") or ""),
        str(source.get("collectorDeviceId") or ""),
        str(source.get("sessionId") or ""),
    ))
    candidate = {
        **{key: str(source.get(key) or "") for key in AUDIT_FIELDS},
        "sourceFile": str(source.get("sourceFile") or ""),
        "sourceSha256": str(source.get("sourceSha256") or ""),
        "sampleId": str(source.get("sampleId") or ""),
        "classLabel": class_label,
        "motionState": str(source.get("motionState") or ""),
        "collectorUserName": str(source.get("collectorUserName") or ""),
        "collectorDeviceId": str(source.get("collectorDeviceId") or ""),
        "deviceName": str(source.get("deviceName") or ""),
        "platform": str(source.get("platform") or ""),
        "sensorSource": str(source.get("sensorSource") or ""),
        "sessionId": str(source.get("sessionId") or ""),
        "groupId": group_id,
        "sourceTimestampMs": source_timestamp_ms(source.get("sourceFile")),
        "trainingEligibility": "eligible",
        "exclusionReason": "",
        "trainingWeight": 1.0,
        "exactFeatureFingerprint": "",
        "duplicateClusterId": "",
        "duplicateClusterSize": 1,
    }
    if source.get("labelStatus") in {"mistake", "uncertain"}:
        exclude(candidate, "user_review_" + source["labelStatus"])
    elif class_label not in CLASS_LABELS and class_label != ORDINARY_CLASS_LABEL:
        exclude(candidate, "unsupported_class_label")
    elif not all((candidate["platform"], candidate["collectorDeviceId"], candidate["sessionId"])):
        exclude(candidate, "missing_group_identity")
    elif contains_multi_pointer_touch(touch):
        exclude(candidate, "multi_pointer_touch")
    elif class_label == ORDINARY_CLASS_LABEL:
        exclude(candidate, "ordinary_touch_negative_sample")
    return candidate


def source_timestamp_ms(source_file):
    name = Path(str(source_file or "")).name
    prefix = name.split("_", 1)[0]
    try:
        value = int(prefix)
    except ValueError:
        return 0
    return value if value > 0 else 0


def contains_multi_pointer_touch(row):
    feature_names = row.get("featureNames")
    matrix = row.get("matrix")
    if not isinstance(matrix, list):
        return False
    if isinstance(feature_names, list):
        try:
            touch_valid_index = feature_names.index("touch_valid")
            pointer_count_index = feature_names.index("pointer_count")
        except ValueError:
            return False
    else:
        # 中文注释：轻量测试夹具可能只提供矩阵；按冻结列宽恢复索引，
        # 正式数据仍由 validate_matrix 校验完整列名。
        width = len(matrix[0]) if matrix and isinstance(matrix[0], list) else 0
        if width == 20:
            touch_valid_index = 19
            pointer_count_index = 18
        elif width == 23:
            touch_valid_index = 17
            pointer_count_index = 16
        elif width == 26:
            touch_valid_index = 19
            pointer_count_index = 18
        else:
            return False
    for values in matrix:
        if not isinstance(values, list):
            continue
        if (
            max(touch_valid_index, pointer_count_index) >= len(values)
        ):
            continue
        if values[touch_valid_index] >= 0.5 and values[pointer_count_index] > 1.0:
            return True
    return False


def exclude(candidate, reason):
    candidate["trainingEligibility"] = "excluded"
    candidate["exclusionReason"] = reason


def assign_exact_duplicate_weights(candidates, touch_by_hash):
    clusters = defaultdict(list)
    for candidate in candidates:
        matrix = touch_by_hash[candidate["sourceSha256"]]["matrix"]
        fingerprint = hashlib.sha256(
            json.dumps(matrix, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        candidate["exactFeatureFingerprint"] = fingerprint
        key = candidate["groupId"], candidate["classLabel"], fingerprint
        clusters[key].append(candidate)
    for cluster in clusters.values():
        cluster_id = "exact-" + cluster[0]["exactFeatureFingerprint"][:12]
        size = len(cluster)
        for candidate in cluster:
            candidate["duplicateClusterId"] = cluster_id
            candidate["duplicateClusterSize"] = size
            if candidate["trainingEligibility"] == "eligible":
                candidate["trainingWeight"] = 1.0 / size


def build_validation_folds(candidates):
    """Choose device holdout whenever every class covers enough devices."""
    physical_devices = defaultdict(set)
    physical_device_labels = defaultdict(set)
    for candidate in candidates:
        device = physical_device_key(candidate)
        physical_devices[device].add(candidate["groupId"])
        physical_device_labels[device].add(candidate["classLabel"])
    eligible_devices = {
        device
        for device, labels in physical_device_labels.items()
        if labels == set(CLASS_LABELS)
    }
    # 中文注释：只有设备同时具备四类和多个独立 session 时才可留出；
    # 单条样本设备不能伪装成跨设备验证折。
    if (
        len(eligible_devices) >= 2
        and all(
            len(physical_devices[device]) >= 2
            for device in eligible_devices
        )
    ):
        rows = []
        for fold, held_out_device in enumerate(sorted(eligible_devices)):
            for candidate in candidates:
                split = (
                    "valid"
                    if physical_device_key(candidate) == held_out_device
                    else "train"
                )
                rows.append({
                    "fold": fold,
                    "split": split,
                    "foldKind": "held_out_device",
                    "sourceSha256": candidate["sourceSha256"],
                    "sampleId": candidate["sampleId"],
                    "classLabel": candidate["classLabel"],
                    "groupId": candidate["groupId"],
                    "trainingWeight": candidate["trainingWeight"],
                })
        return rows, len(eligible_devices), "ready", "held_out_device"
    rows, fold_count, status = build_strict_group_folds(candidates)
    for row in rows:
        row["foldKind"] = "held_out_session"
    return rows, fold_count, status, "held_out_session"


def physical_device_key(candidate):
    """Treat multiple collector ids of one model as one holdout domain."""
    platform = str(candidate.get("platform") or "")
    device_name = str(candidate.get("deviceName") or "")
    if platform and device_name:
        return platform + ":" + device_name
    return (
        platform
        + ":collector:"
        + str(candidate.get("collectorDeviceId") or "")
    )


def build_strict_group_folds(candidates):
    class_groups = {label: set() for label in CLASS_LABELS}
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["groupId"]].append(candidate)
        class_groups[candidate["classLabel"]].add(candidate["groupId"])
    minimum_group_count = min((len(groups) for groups in class_groups.values()), default=0)
    fold_count = min(5, minimum_group_count)
    if fold_count < 2:
        return [], 0, "insufficient_groups_per_class"

    total_label_counts = Counter(candidate["classLabel"] for candidate in candidates)
    fold_label_counts = [Counter() for _ in range(fold_count)]
    fold_sizes = [0 for _ in range(fold_count)]
    assignment = {}
    groups = sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    for group_id, group_candidates in groups:
        group_counts = Counter(candidate["classLabel"] for candidate in group_candidates)
        ranked = []
        for fold in range(fold_count):
            label_score = sum(
                fold_label_counts[fold][label] / total_label_counts[label]
                for label in group_counts
            )
            ranked.append((label_score, fold_sizes[fold], fold))
        _, _, selected_fold = min(ranked)
        assignment[group_id] = selected_fold
        fold_label_counts[selected_fold].update(group_counts)
        fold_sizes[selected_fold] += len(group_candidates)

    rows = []
    for fold in range(fold_count):
        valid_labels = set()
        train_labels = set()
        for candidate in candidates:
            split = "valid" if assignment[candidate["groupId"]] == fold else "train"
            if split == "valid":
                valid_labels.add(candidate["classLabel"])
            else:
                train_labels.add(candidate["classLabel"])
            rows.append({
                "fold": fold,
                "split": split,
                "foldKind": "held_out_session",
                "sourceSha256": candidate["sourceSha256"],
                "sampleId": candidate["sampleId"],
                "classLabel": candidate["classLabel"],
                "groupId": candidate["groupId"],
                "trainingWeight": candidate["trainingWeight"],
            })
        if valid_labels != set(CLASS_LABELS) or train_labels != set(CLASS_LABELS):
            return [], 0, "strict_fold_missing_class_coverage"
    return rows, fold_count, "ready"


def build_pose_holdout_folds(candidates):
    """Build diagnostic posture folds without replacing the primary split."""
    grouped = defaultdict(list)
    for candidate in candidates:
        posture = str(candidate.get("bodyPosture") or "unknown")
        if posture != "unknown":
            grouped[posture].append(candidate)
    eligible_postures = []
    for posture, values in grouped.items():
        labels = {candidate["classLabel"] for candidate in values}
        groups = {candidate["groupId"] for candidate in values}
        if labels == set(CLASS_LABELS) and len(groups) >= 2:
            eligible_postures.append(posture)
    if not eligible_postures:
        return [], "insufficient_posture_coverage"
    rows = []
    for fold, held_out_posture in enumerate(sorted(eligible_postures)):
        for candidate in candidates:
            posture = str(candidate.get("bodyPosture") or "unknown")
            rows.append({
                "fold": fold,
                "split": "valid" if posture == held_out_posture else "train",
                "foldKind": "held_out_posture",
                "sourceSha256": candidate["sourceSha256"],
                "sampleId": candidate.get("sampleId", ""),
                "classLabel": candidate["classLabel"],
                "groupId": candidate["groupId"],
                "trainingWeight": candidate["trainingWeight"],
                "bodyPosture": posture,
            })
    return rows, "ready"


def training_imu_record(candidate, row):
    return training_record(candidate, row["matrix"], CONTRACT_IMU11, IMU_FEATURE_NAMES)


def training_touch23_record(candidate, row):
    result = training_record(
        candidate,
        row["matrix"],
        CONTRACT_TOUCH23,
        IMU_FEATURE_NAMES + V5_TOUCH_FEATURE_NAMES + V5_MASK_FEATURE_NAMES,
        row["missingMasks"],
    )
    # 量程只用于设备归一化实验，不进入数值矩阵。
    audit = row.get("touchAxisRangeAudit")
    if isinstance(audit, dict):
        result["touchAxisRangeAudit"] = audit
    return result


def training_touch26_record(candidate, row):
    result = training_record(
        candidate,
        row["matrix"],
        CONTRACT_TOUCH26,
        IMU_FEATURE_NAMES + TOUCH_FEATURE_NAMES + MASK_FEATURE_NAMES,
        row["missingMasks"],
    )
    audit = row.get("touchAxisRangeAudit")
    if isinstance(audit, dict):
        result["touchAxisRangeAudit"] = audit
    return result


def training_record(candidate, matrix, contract, feature_names, masks=None):
    return {
        **{key: candidate.get(key, "") for key in AUDIT_FIELDS},
        "sourceFile": candidate["sourceFile"],
        "sourceSha256": candidate["sourceSha256"],
        "sampleId": candidate["sampleId"],
        "classLabel": candidate["classLabel"],
        "motionState": candidate["motionState"],
        "platform": candidate["platform"],
        "collectorDeviceId": candidate["collectorDeviceId"],
        "deviceName": candidate["deviceName"],
        "sessionId": candidate["sessionId"],
        "groupId": candidate["groupId"],
        "trainingWeight": candidate["trainingWeight"],
        "featureContract": contract,
        "featureNames": list(feature_names),
        "matrixShape": [len(matrix), len(matrix[0])],
        "matrix": matrix,
        "missingMasks": masks or [],
    }


def session_coverage_rows(candidates):
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["groupId"]].append(candidate)
    rows = []
    for group_id, values in sorted(grouped.items()):
        first = values[0]
        label_counts = Counter(value["classLabel"] for value in values)
        rows.append({
            "groupId": group_id,
            "platform": first["platform"],
            "collectorDeviceId": first["collectorDeviceId"],
            "sessionId": first["sessionId"],
            "sampleCount": len(values),
            "classLabels": ";".join(sorted(label_counts)),
            "classCounts": json.dumps(dict(sorted(label_counts.items())), sort_keys=True),
            "bodyPosture": first.get("bodyPosture", "unknown"),
            "grip": first.get("grip", "unknown"),
            "support": first.get("support", "unknown"),
            "collectionProtocol": first.get(
                "collectionProtocol",
                "unknown",
            ),
            "isMultiClassSession": len(label_counts) > 1,
        })
    return rows


def context_audit_rows(candidates):
    """Expose collection context separately from the primary split policy."""
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["groupId"]].append(candidate)
    rows = []
    for group_id, values in sorted(grouped.items()):
        first = values[0]
        label_counts = Counter(
            candidate["classLabel"] for candidate in values
        )
        rows.append({
            "groupId": group_id,
            "platform": first["platform"],
            "collectorDeviceId": first["collectorDeviceId"],
            "deviceName": first["deviceName"],
            "sessionId": first["sessionId"],
            "taskId": first.get("taskId", "unknown"),
            "bodyPosture": first.get("bodyPosture", "unknown"),
            "requestedDevicePose": first.get(
                "requestedDevicePose",
                "unknown",
            ),
            "grip": first.get("grip", "unknown"),
            "support": first.get("support", "unknown"),
            "collectionProtocol": first.get(
                "collectionProtocol",
                "unknown",
            ),
            "sampleCount": len(values),
            "classCount": len(label_counts),
            "classLabels": ";".join(sorted(label_counts)),
            "classCounts": json.dumps(
                dict(sorted(label_counts.items())),
                sort_keys=True,
            ),
            "isMultiClassSession": len(label_counts) > 1,
        })
    return rows


def missingness_rows(candidates, touch_by_hash):
    counts = {name: Counter() for name in ("touch",) + V5_MASK_SOURCE_NAMES}
    for candidate in candidates:
        for mask in touch_by_hash[candidate["sourceSha256"]]["missingMasks"]:
            for name in counts:
                counts[name]["total"] += 1
                counts[name]["available"] += int(mask[name])
    rows = []
    for name in counts:
        total = counts[name]["total"]
        available = counts[name]["available"]
        rows.append({
            "maskName": name,
            "availableFrameCount": available,
            "missingFrameCount": total - available,
            "totalFrameCount": total,
            "missingRate": (total - available) / total if total else 0.0,
        })
    return rows


def write_training_contract(path):
    payload = {
        "schemaVersion": 1,
        "classificationLabels": list(CLASS_LABELS),
        "gridOffsetsMs": [-15, -10, -5, 0, 5, 10, 15, 20, 25],
        "contracts": {
            CONTRACT_IMU11: {
                "featureNames": list(IMU_FEATURE_NAMES),
                "matrixShape": [9, 11],
            },
            CONTRACT_TOUCH23: {
                "featureNames": list(IMU_FEATURE_NAMES + V5_TOUCH_FEATURE_NAMES + V5_MASK_FEATURE_NAMES),
                "matrixShape": [9, 23],
                "maskSourceNames": list(V5_MASK_SOURCE_NAMES),
                "maskFeatureNames": list(V5_MASK_SOURCE_NAMES),
                "maskOutputFeatureNames": list(V5_MASK_FEATURE_NAMES),
            },
            CONTRACT_TOUCH26: {
                "featureNames": list(IMU_FEATURE_NAMES + TOUCH_FEATURE_NAMES + MASK_FEATURE_NAMES),
                "matrixShape": [9, 26],
                "maskSourceNames": list(MASK_SOURCE_NAMES),
                "maskFeatureNames": list(MASK_SOURCE_NAMES),
                "maskOutputFeatureNames": list(MASK_FEATURE_NAMES),
            },
        },
        "primaryContract": CONTRACT_TOUCH23,
        "maskFeatureNames": list(V5_MASK_SOURCE_NAMES),
        "prohibitedFeatureNames": [
            "sensorTimestampElapsedRealtimeNanos",
            "receivedElapsedRealtimeNanos",
            "timeSinceTouchDown",
            "timeSinceAnchor",
            "liftEventElapsedRealtimeNanos",
            "contactPatch",
            "contactEncoding",
            "touchNormalization",
        ],
        "splitPolicy": "group_by_platform_device_session",
        "scalerPolicy": "fit_only_on_training_partition",
        "duplicatePolicy": "downweight_exact_matrix_duplicates_within_group_and_class",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path, fields, rows):
    if not fields:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            output.write("\n")


def write_source_hash_manifest(path, candidates):
    lines = [
        candidate["sourceSha256"] + "  " + candidate["sourceFile"]
        for candidate in candidates
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def summary_record(
    candidates,
    eligible,
    fold_count,
    fold_status,
    fold_kind,
    missingness,
    pose_folds,
    pose_fold_status,
    ordinary_negatives,
    label_quality_report=None,
):
    labels_by_group = defaultdict(set)
    sessions_by_label = defaultdict(set)
    for candidate in eligible:
        labels_by_group[candidate["groupId"]].add(candidate["classLabel"])
        sessions_by_label[candidate["classLabel"]].add(candidate["groupId"])
    groups_per_class = {
        label: len({candidate["groupId"] for candidate in eligible if candidate["classLabel"] == label})
        for label in CLASS_LABELS
    }
    bound_groups = sum(len(labels) == 1 for labels in labels_by_group.values())
    session_label_binding_rate = (
        bound_groups / len(labels_by_group)
        if labels_by_group
        else 0.0
    )
    # 中文注释：会话与类别绑定会让跨会话留出同时改变标签和采集条件，
    # 因而只能衡量跨批次泛化，不能单独证明触摸力度识别能力。
    session_label_binding = bool(labels_by_group) and bound_groups == len(labels_by_group)
    multi_class_groups = sum(
        len(labels) > 1 for labels in labels_by_group.values()
    )
    multi_class_session_rate = (
        multi_class_groups / len(labels_by_group)
        if labels_by_group
        else 0.0
    )
    device_count = len({
        physical_device_key(candidate) for candidate in eligible
    })
    if fold_kind == "held_out_device":
        evaluation_scope = "cross_device"
    elif fold_count:
        evaluation_scope = "single_device_cross_session"
    else:
        evaluation_scope = "not_ready"
    known_context_count = sum(
        candidate.get("bodyPosture", "unknown") != "unknown"
        and candidate.get("grip", "unknown") != "unknown"
        and candidate.get("support", "unknown") != "unknown"
        and candidate.get("collectionProtocol", "unknown") != "unknown"
        for candidate in eligible
    )
    known_context_rate = (
        known_context_count / len(eligible) if eligible else 0.0
    )
    interleaved_count = sum(
        candidate.get("collectionProtocol") == "interleaved_v2"
        for candidate in eligible
    )
    interleaved_rate = (
        interleaved_count / len(eligible) if eligible else 0.0
    )
    readiness = {
        "contextMetadataReady": known_context_rate >= 0.95,
        "interleavedProtocolReady": interleaved_rate >= 0.95,
        "crossClassSessionReady": multi_class_session_rate >= 0.8,
        "poseHoldoutReady": bool(pose_folds),
    }
    readiness["ready"] = all(readiness.values())
    if label_quality_report is not None:
        readiness["labelQualityReady"] = bool(
            label_quality_report["trainingLabelQualityReady"]
        )
        readiness["ready"] = all(
            value
            for name, value in readiness.items()
            if name != "ready"
        )
    readiness["blockingReasons"] = [
        name for name, value in readiness.items()
        if name != "ready" and not value
    ]
    return {
        "candidateCount": len(candidates),
        "eligibleCount": len(eligible),
        "excludedCount": sum(
            candidate.get("trainingEligibility") == "excluded"
            for candidate in candidates
        ),
        "ordinaryNegativeCount": len(ordinary_negatives),
        "ordinaryNegativeDevices": sorted({
            candidate["deviceName"] for candidate in ordinary_negatives
        }),
        "ordinaryNegativeSessions": len({
            candidate["groupId"] for candidate in ordinary_negatives
        }),
        "reviewCount": sum(
            candidate.get("trainingEligibility") == "review"
            for candidate in candidates
        ),
        "exclusionReasons": dict(Counter(
            candidate.get("exclusionReason", "")
            for candidate in candidates
            if candidate.get("exclusionReason", "")
        )),
        "classCounts": dict(Counter(candidate["classLabel"] for candidate in eligible)),
        "groupCount": len({candidate["groupId"] for candidate in eligible}),
        "deviceCount": device_count,
        "devices": sorted({
            candidate["deviceName"] for candidate in eligible
        }),
        "groupsPerClass": groups_per_class,
        "sessionsPerClass": {
            label: len(sessions_by_label[label])
            for label in CLASS_LABELS
        },
        "sessionLabelBinding": session_label_binding,
        "sessionLabelBindingRate": session_label_binding_rate,
        "sessionLabelBindingPolicy": "exclude_cross_session_folds_from_label_effectiveness_claims",
        "multiClassSessionCount": multi_class_groups,
        "multiClassSessionRate": multi_class_session_rate,
        "knownCollectionContextCount": known_context_count,
        "knownCollectionContextRate": known_context_rate,
        "interleavedSampleCount": interleaved_count,
        "interleavedSampleRate": interleaved_rate,
        "collectionProtocolCounts": dict(Counter(
            candidate.get("collectionProtocol", "unknown")
            for candidate in eligible
        )),
        "bodyPostureCounts": dict(Counter(
            candidate.get("bodyPosture", "unknown")
            for candidate in eligible
        )),
        "taskCounts": dict(Counter(
            candidate.get("taskId", "unknown")
            for candidate in eligible
        )),
        "allMissingMaskNames": [row["maskName"] for row in missingness if row["availableFrameCount"] == 0],
        "strictFoldCount": fold_count,
        "strictGroupValidationStatus": fold_status,
        "validationFoldKind": fold_kind,
        "evaluationScope": evaluation_scope,
        "poseHoldoutFoldCount": len(eligible_posture_count(pose_folds)),
        "poseHoldoutValidationStatus": pose_fold_status,
        "trainingReadiness": readiness,
        "labelQualityReport": label_quality_report,
        "nearDuplicatePolicy": "exact_matrix_only",
    }


def eligible_posture_count(pose_folds):
    return {
        int(row["fold"])
        for row in pose_folds
        if row["split"] == "valid"
    }


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_training_dataset(args.derived_dir, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
