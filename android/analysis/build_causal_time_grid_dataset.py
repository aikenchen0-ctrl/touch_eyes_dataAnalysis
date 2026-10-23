#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

try:
    from . import cross_device_causal_features as cross_device
    from . import collection_annotations as annotations
    from . import touch_label_quality as label_quality
except ImportError:
    import cross_device_causal_features as cross_device
    import collection_annotations as annotations
    import touch_label_quality as label_quality


CONTRACTS = (
    cross_device.CONTRACT_IMU11,
    cross_device.CONTRACT_IMU11_TOUCH9,
    cross_device.CONTRACT_IMU11_TOUCH9_MASK6_V4,
    cross_device.CONTRACT_IMU11_TOUCH7_MASK5_V5,
)
CLASS_LABELS = {
    "thumb_light",
    "thumb_heavy",
    "index_light",
    "index_heavy",
}
ORDINARY_CLASS_LABEL = "ordinary_touch"
ACCEPTED_CLASS_LABELS = CLASS_LABELS | {ORDINARY_CLASS_LABEL}
SOURCE_FIELDS = (
    "sourceFile",
    "sourceSha256",
    "normalizationStatus",
    "rejectionReason",
    "schemaVersion",
    "sampleId",
    "classLabel",
    "motionState",
    "collectorUserName",
    "collectorDeviceId",
    "deviceName",
    "platform",
    "sensorSource",
    "sessionId",
) + annotations.AUDIT_FIELDS
QUALITY_FIELDS = (
    "sourceFile",
    "sourceSha256",
    "sampleId",
    "classLabel",
    "featureContract",
    "gridRateHz",
    "windowDurationMs",
    "medianDtNs",
    "maxGapNs",
    "sourceSampleRateHz",
    "touchAxisRangeStatus",
    "touchMajorDeclaredMax",
    "touchMajorValueFractionMedian",
)


def build_dataset(input_dir, output_dir):
    source_root = Path(input_dir).resolve()
    derived_root = Path(output_dir).resolve()
    if not source_root.is_dir():
        raise RuntimeError("input_directory_not_found")
    if source_root == derived_root or source_root in derived_root.parents:
        raise RuntimeError("output_directory_must_not_contain_sources")

    source_paths = sorted(
        (path for path in source_root.rglob("*.json") if path.is_file() and annotations.is_raw_sample_path(path)),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    source_rows = []
    rejected_rows = []
    quality_rows = []
    source_label_quality_records = []
    canonical_label_quality_records = []
    matrices = {contract: [] for contract in CONTRACTS}
    seen_hashes = set()
    conflicting_reviews = annotations.conflicting_duplicate_reviews(source_paths)

    for path in source_paths:
        source_file = path.relative_to(source_root).as_posix()
        raw = path.read_bytes()
        source_sha256 = hashlib.sha256(raw).hexdigest()
        row, data = source_row(source_file, source_sha256, raw)
        if source_sha256 in conflicting_reviews:
            reject(row, "duplicate_source_review_conflict", rejected_rows)
            source_rows.append(row)
            continue
        if source_sha256 in seen_hashes:
            reject(row, "duplicate_source_sha256", rejected_rows)
            source_rows.append(row)
            continue
        seen_hashes.add(source_sha256)
        if data is None:
            reject(row, "invalid_json", rejected_rows)
            source_rows.append(row)
            continue
        update_row_metadata(row, data)
        try:
            annotation = annotations.read_annotation(path, data)
            row.update(annotation)
            reason = annotations.exclusion_reason(annotation)
            if reason:
                raise RuntimeError(reason)
        except RuntimeError as error:
            reject(row, str(error), rejected_rows)
            source_rows.append(row)
            continue
        class_label = row["classLabel"]
        if class_label not in ACCEPTED_CLASS_LABELS:
            reject(row, "unsupported_class_label", rejected_rows)
            source_rows.append(row)
            continue
        # 中文注释：原始标签质量必须在 canonical 网格构建前记录。
        # 否则被多指、窗口缺失等规则排除的异常样本会从审计中消失。
        source_label_quality_records.append(
            label_quality.sample_record(
                source_file,
                source_sha256,
                data,
            )
        )
        try:
            outputs = {
                contract: cross_device.build_time_grid_matrix(data, data, contract)
                for contract in CONTRACTS
            }
        except RuntimeError as error:
            reject(row, str(error), rejected_rows)
            source_rows.append(row)
            continue
        row["normalizationStatus"] = "accepted"
        source_rows.append(row)
        canonical_label_quality_records.append(
            label_quality.sample_record(
                source_file,
                source_sha256,
                data,
            )
        )
        for contract, result in outputs.items():
            matrices[contract].append(matrix_record(row, contract, result))
            quality_rows.append(quality_record(row, contract, result["gridQuality"], result))

    derived_root.mkdir(parents=True, exist_ok=True)
    write_csv(derived_root / "source_manifest.csv", SOURCE_FIELDS, source_rows)
    write_csv(derived_root / "rejected_samples.csv", SOURCE_FIELDS, rejected_rows)
    write_csv(derived_root / "grid_quality_report.csv", QUALITY_FIELDS, quality_rows)
    write_jsonl(derived_root / "canonical_imu11.jsonl", matrices[cross_device.CONTRACT_IMU11])
    write_jsonl(
        derived_root / "canonical_imu11_touch9.jsonl",
        matrices[cross_device.CONTRACT_IMU11_TOUCH9],
    )
    write_jsonl(
        derived_root / "canonical_imu11_touch9_mask6_v4.jsonl",
        matrices[cross_device.CONTRACT_IMU11_TOUCH9_MASK6_V4],
    )
    write_jsonl(
        derived_root / "canonical_imu11_touch7_mask5_v5.jsonl",
        matrices[cross_device.CONTRACT_IMU11_TOUCH7_MASK5_V5],
    )
    write_feature_contract(derived_root / "feature_contract.json")
    source_label_quality_summary = label_quality.write_quality_outputs(
        derived_root,
        source_label_quality_records,
    )
    canonical_label_quality_summary = label_quality.write_quality_outputs(
        derived_root,
        canonical_label_quality_records,
        prefix="canonical_touch_label_quality",
    )
    write_source_hash_manifest(derived_root / "source_hash_manifest.sha256", source_rows)
    summary = summary_record(
        source_rows,
        rejected_rows,
        quality_rows,
        source_label_quality_summary,
        canonical_label_quality_summary,
    )
    (derived_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def source_row(source_file, source_sha256, raw):
    row = {field: "" for field in SOURCE_FIELDS}
    row["sourceFile"] = source_file
    row["sourceSha256"] = source_sha256
    row["normalizationStatus"] = "rejected"
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return row, None
    return row, data if isinstance(data, dict) else None


def update_row_metadata(row, data):
    label = data.get("label") or {}
    collector = data.get("collector") or {}
    capability = data.get("deviceCapability") or {}
    sensor_metadata = data.get("sensorMetadata") or {}
    row.update({
        "schemaVersion": str(data.get("schemaVersion") or ""),
        "sampleId": str(data.get("id") or ""),
        "classLabel": class_label(label),
        "motionState": str(label.get("motionState") or ""),
        "collectorUserName": str(collector.get("userName") or ""),
        "collectorDeviceId": str(collector.get("deviceId") or ""),
        "deviceName": str(capability.get("deviceName") or ""),
        "platform": str(sensor_metadata.get("platform") or data.get("platform") or ""),
        "sensorSource": str(sensor_metadata.get("sensorSource") or ""),
        "sessionId": str(data.get("collectionSessionID") or ""),
    })


def class_label(label):
    finger = str(label.get("fingerType") or "")
    strength = str(label.get("touchStrength") or "")
    # 中文注释：普通点击是拒识负样本，不携带手指监督含义，
    # 因此统一归为独立类别，避免 thumb_ordinary 被误判为无效样本。
    if strength == "ordinary":
        return ORDINARY_CLASS_LABEL
    return finger + "_" + strength if finger and strength else ""


def reject(row, reason, rejected_rows):
    row["normalizationStatus"] = "rejected"
    row["rejectionReason"] = reason
    rejected_rows.append(dict(row))


def matrix_record(row, contract, result):
    matrix = result["matrix"]
    return {
        **{key: row.get(key, "") for key in annotations.AUDIT_FIELDS},
        "sampleId": row["sampleId"],
        "sourceFile": row["sourceFile"],
        "sourceSha256": row["sourceSha256"],
        "classLabel": row["classLabel"],
        "motionState": row["motionState"],
        "collectorUserName": row["collectorUserName"],
        "collectorDeviceId": row["collectorDeviceId"],
        "deviceName": row["deviceName"],
        "platform": row["platform"],
        "sensorSource": row["sensorSource"],
        "sessionId": row["sessionId"],
        "featureContract": contract,
        "featureNames": cross_device.feature_names(contract),
        "matrixShape": [len(matrix), len(matrix[0]) if matrix else 0],
        "matrix": matrix,
        "missingMasks": result["missingMasks"],
        "gridOffsetsMs": result["gridOffsetsMs"],
        "gridRateHz": result["gridRateHz"],
        "gridQuality": result["gridQuality"],
        "touchAxisRangeAudit": result["touchAxisRangeAudit"],
    }


def quality_record(row, contract, quality, result):
    range_audit = result["touchAxisRangeAudit"]
    major_audit = range_audit["ranges"]["touchMajor"]
    return {
        "sourceFile": row["sourceFile"],
        "sourceSha256": row["sourceSha256"],
        "sampleId": row["sampleId"],
        "classLabel": row["classLabel"],
        "featureContract": contract,
        "gridRateHz": result["gridRateHz"],
        "windowDurationMs": quality["windowDurationMs"],
        "medianDtNs": quality["medianDtNs"],
        "maxGapNs": quality["maxGapNs"],
        "sourceSampleRateHz": quality["sourceSampleRateHz"],
        "touchAxisRangeStatus": range_audit["framesWithMotionRanges"],
        "touchMajorDeclaredMax": major_audit["declaredMax"],
        "touchMajorValueFractionMedian": major_audit[
            "valueFractionMedian"
        ],
    }


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            output.write("\n")


def write_feature_contract(path):
    contracts = {
        contract: {
            "featureNames": cross_device.feature_names(contract),
            "matrixShape": [len(cross_device.TIME_GRID_OFFSETS_MS), len(cross_device.feature_names(contract))],
        }
        for contract in CONTRACTS
    }
    payload = {
        "schemaVersion": 1,
        "gridOffsetsMs": list(cross_device.TIME_GRID_OFFSETS_MS),
        "gridRateHz": 200,
        "imuInterpolation": "linear_sensor_timestamp",
        "touchInterpolation": {
            "default": "zero_order_hold_received_timestamp_with_event_time_gate",
            cross_device.CONTRACT_IMU11_TOUCH7_MASK5_V5:
                "batch_ready_zero_order_hold_with_event_time_gate",
        },
        "contracts": contracts,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_source_hash_manifest(path, source_rows):
    lines = [
        row["sourceSha256"] + "  " + row["sourceFile"]
        for row in source_rows
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def summary_record(
        source_rows,
        rejected_rows,
        quality_rows,
        source_label_quality_summary=None,
        canonical_label_quality_summary=None,
):
    return {
        "sourceCount": len(source_rows),
        "acceptedCount": sum(row["normalizationStatus"] == "accepted" for row in source_rows),
        "rejectedCount": len(rejected_rows),
        "rejectionReasons": dict(Counter(row["rejectionReason"] for row in rejected_rows)),
        "classCounts": dict(Counter(row["classLabel"] for row in source_rows if row["normalizationStatus"] == "accepted")),
        "sessionCount": len({row["sessionId"] for row in source_rows if row["normalizationStatus"] == "accepted"}),
        "qualityRowCount": len(quality_rows),
        "labelQualityStatus": (
            source_label_quality_summary["disposition"]
            if source_label_quality_summary is not None
            else "not_available"
        ),
        "trainingLabelQualityReady": (
            source_label_quality_summary["trainingLabelQualityReady"]
            if source_label_quality_summary is not None
            else None
        ),
        "labelQualityBlockers": (
            source_label_quality_summary["blockers"]
            if source_label_quality_summary is not None
            else []
        ),
        "sourceLabelQualitySampleCount": (
            source_label_quality_summary["sampleCount"]
            if source_label_quality_summary is not None
            else 0
        ),
        "canonicalLabelQualityStatus": (
            canonical_label_quality_summary["disposition"]
            if canonical_label_quality_summary is not None
            else "not_available"
        ),
        "canonicalLabelQualitySampleCount": (
            canonical_label_quality_summary["sampleCount"]
            if canonical_label_quality_summary is not None
            else 0
        ),
        "canonicalLabelQualityBlockers": (
            canonical_label_quality_summary["blockers"]
            if canonical_label_quality_summary is not None
            else []
        ),
        "contracts": list(CONTRACTS),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_dataset(args.input_dir, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
