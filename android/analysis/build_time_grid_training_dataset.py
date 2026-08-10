#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


CLASS_LABELS = (
    "thumb_light",
    "thumb_heavy",
    "index_light",
    "index_heavy",
)
CONTRACT_IMU11 = "causal_time_grid_imu11_v2"
CONTRACT_TOUCH20 = "causal_time_grid_imu11_touch9_v2"
CONTRACT_TOUCH26 = "causal_time_grid_imu11_touch9_mask6_v3"
MASK_SOURCE_NAMES = ("pressure", "size", "major", "minor", "orientation", "pointerCount")
MASK_FEATURE_NAMES = (
    "pressure_available",
    "size_available",
    "major_available",
    "minor_available",
    "orientation_available",
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
)
FOLD_FIELDS = (
    "fold",
    "split",
    "sourceSha256",
    "sampleId",
    "classLabel",
    "groupId",
    "trainingWeight",
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
    source_by_hash = accepted_sources(source_rows)
    imu_rows = read_jsonl(derived_root / "canonical_imu11.jsonl")
    touch_rows = read_jsonl(derived_root / "canonical_imu11_touch9.jsonl")
    imu_by_hash = matrix_rows_by_hash(imu_rows, CONTRACT_IMU11, 11)
    touch_by_hash = matrix_rows_by_hash(touch_rows, CONTRACT_TOUCH20, 20)
    validate_source_matrix_identity(source_by_hash, imu_by_hash, touch_by_hash)

    candidates = []
    for source_sha256, source in sorted(source_by_hash.items(), key=lambda item: item[1]["sourceFile"]):
        imu = imu_by_hash[source_sha256]
        touch = touch_by_hash[source_sha256]
        candidate = candidate_record(source, imu, touch)
        candidates.append(candidate)
    assign_exact_duplicate_weights(candidates, touch_by_hash)
    eligible = [candidate for candidate in candidates if candidate["trainingEligibility"] == "eligible"]
    folds, fold_count, fold_status = build_strict_group_folds(eligible)

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
        output_root / "training_touch26.jsonl",
        [training_touch26_record(candidate, touch_by_hash[candidate["sourceSha256"]]) for candidate in eligible],
    )
    write_csv(output_root / "strict_group_folds.csv", FOLD_FIELDS, folds)
    write_csv(output_root / "session_label_coverage.csv", None, session_coverage_rows(eligible))
    missingness = missingness_rows(eligible, touch_by_hash)
    write_csv(output_root / "feature_missingness_report.csv", MISSINGNESS_FIELDS, missingness)
    write_csv(output_root / "near_duplicate_audit.csv", CANDIDATE_FIELDS, candidates)
    write_training_contract(output_root / "training_feature_contract.json")
    write_source_hash_manifest(output_root / "source_hash_manifest.sha256", candidates)
    summary = summary_record(candidates, eligible, fold_count, fold_status, missingness)
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
    if row.get("matrixShape") != [9, width] or not isinstance(matrix, list) or len(matrix) != 9:
        raise RuntimeError("invalid_time_grid_matrix_shape")
    for values in matrix:
        if not isinstance(values, list) or len(values) != width:
            raise RuntimeError("invalid_time_grid_matrix_shape")
        for value in values:
            if not finite(value):
                raise RuntimeError("non_finite_time_grid_feature")
    if width == 20:
        masks = row.get("missingMasks")
        if not isinstance(masks, list) or len(masks) != 9:
            raise RuntimeError("invalid_missing_masks")
        for mask in masks:
            if not isinstance(mask, dict) or any(mask.get(name) not in (0, 1) for name in MASK_SOURCE_NAMES):
                raise RuntimeError("invalid_missing_masks")


def validate_source_matrix_identity(source_by_hash, imu_by_hash, touch_by_hash):
    if set(imu_by_hash) != set(touch_by_hash):
        raise RuntimeError("matrix_source_identity_mismatch")
    if set(source_by_hash) != set(imu_by_hash):
        raise RuntimeError("accepted_source_matrix_missing")
    identity_fields = ("sourceFile", "sampleId", "classLabel", "sessionId", "platform", "collectorDeviceId")
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
    if class_label not in CLASS_LABELS:
        exclude(candidate, "unsupported_class_label")
    elif not all((candidate["platform"], candidate["collectorDeviceId"], candidate["sessionId"])):
        exclude(candidate, "missing_group_identity")
    elif contains_multi_pointer_touch(touch):
        exclude(candidate, "multi_pointer_touch")
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
    for values in row["matrix"]:
        if values[19] >= 0.5 and values[18] > 1.0:
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
                "sourceSha256": candidate["sourceSha256"],
                "sampleId": candidate["sampleId"],
                "classLabel": candidate["classLabel"],
                "groupId": candidate["groupId"],
                "trainingWeight": candidate["trainingWeight"],
            })
        if valid_labels != set(CLASS_LABELS) or train_labels != set(CLASS_LABELS):
            return [], 0, "strict_fold_missing_class_coverage"
    return rows, fold_count, "ready"


def training_imu_record(candidate, row):
    return training_record(candidate, row["matrix"], CONTRACT_IMU11, IMU_FEATURE_NAMES)


def training_touch26_record(candidate, row):
    matrix = []
    for values, masks in zip(row["matrix"], row["missingMasks"]):
        matrix.append(list(values) + [int(masks[name]) for name in MASK_SOURCE_NAMES])
    return training_record(
        candidate,
        matrix,
        CONTRACT_TOUCH26,
        IMU_FEATURE_NAMES + TOUCH_FEATURE_NAMES + MASK_FEATURE_NAMES,
        row["missingMasks"],
    )


def training_record(candidate, matrix, contract, feature_names, masks=None):
    return {
        "sourceFile": candidate["sourceFile"],
        "sourceSha256": candidate["sourceSha256"],
        "sampleId": candidate["sampleId"],
        "classLabel": candidate["classLabel"],
        "motionState": candidate["motionState"],
        "platform": candidate["platform"],
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
        })
    return rows


def missingness_rows(candidates, touch_by_hash):
    counts = {name: Counter() for name in ("touch",) + MASK_SOURCE_NAMES}
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
            CONTRACT_TOUCH26: {
                "featureNames": list(IMU_FEATURE_NAMES + TOUCH_FEATURE_NAMES + MASK_FEATURE_NAMES),
                "matrixShape": [9, 26],
                "maskSourceNames": list(MASK_SOURCE_NAMES),
                "maskFeatureNames": list(MASK_SOURCE_NAMES),
                "maskOutputFeatureNames": list(MASK_FEATURE_NAMES),
            },
        },
        "maskFeatureNames": list(MASK_SOURCE_NAMES),
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


def summary_record(candidates, eligible, fold_count, fold_status, missingness):
    labels_by_group = defaultdict(set)
    for candidate in eligible:
        labels_by_group[candidate["groupId"]].add(candidate["classLabel"])
    groups_per_class = {
        label: len({candidate["groupId"] for candidate in eligible if candidate["classLabel"] == label})
        for label in CLASS_LABELS
    }
    return {
        "candidateCount": len(candidates),
        "eligibleCount": len(eligible),
        "excludedCount": sum(candidate["trainingEligibility"] == "excluded" for candidate in candidates),
        "reviewCount": sum(candidate["trainingEligibility"] == "review" for candidate in candidates),
        "exclusionReasons": dict(Counter(
            candidate["exclusionReason"]
            for candidate in candidates
            if candidate["exclusionReason"]
        )),
        "classCounts": dict(Counter(candidate["classLabel"] for candidate in eligible)),
        "groupCount": len({candidate["groupId"] for candidate in eligible}),
        "groupsPerClass": groups_per_class,
        "sessionLabelBinding": bool(labels_by_group) and all(len(labels) == 1 for labels in labels_by_group.values()),
        "allMissingMaskNames": [row["maskName"] for row in missingness if row["availableFrameCount"] == 0],
        "strictFoldCount": fold_count,
        "strictGroupValidationStatus": fold_status,
        "evaluationScope": "single_device_cross_session" if fold_count else "not_ready",
        "nearDuplicatePolicy": "exact_matrix_only",
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
