"""Audit whether tap labels represent short, single-pointer contacts."""

import csv
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
LONG_CONTACT_MS = 250.0
VERY_LONG_CONTACT_MS = 1000.0
MAX_LONG_CONTACT_RATE = 0.01
MAX_MULTI_POINTER_RATE = 0.01
MAX_UNEXPECTED_TOUCH_KIND_RATE = 0.001
MIN_DURATION_COVERAGE = 0.99
RECORD_FIELDS = (
    "sourceFile",
    "sourceSha256",
    "sampleId",
    "classLabel",
    "sessionId",
    "deviceName",
    "bodyPosture",
    "taskId",
    "collectionProtocol",
    "strengthTruthSource",
    "labelStatus",
    "reviewSha256",
    "labelTrust",
    "sourceTouchKind",
    "durationMs",
    "maxPointerCount",
    "touchFrameCount",
    "distinctSizeCount",
    "sizeAvailableFrameCount",
    "sizeMedian",
    "qualityFlags",
)


def class_label(data):
    """Return the frozen four-class label or an empty string."""
    label = data.get("label") or {}
    finger = str(label.get("fingerType") or "")
    strength = str(label.get("touchStrength") or "")
    return finger + "_" + strength if finger and strength else ""


def sample_record(source_file, source_sha256, data):
    """Extract label-independent contact quality evidence from one sample."""
    label = class_label(data)
    context = data.get("collectionContext") or {}
    capability = data.get("deviceCapability") or {}
    duration = touch_duration_ms(data)
    touch_frames = data.get("touchFrames") or []
    pointer_counts = [
        nonnegative_int(frame.get("pointerCount"))
        for frame in touch_frames
        if isinstance(frame, dict)
    ]
    size_values = [
        finite_positive(frame.get("size"))
        for frame in touch_frames
        if isinstance(frame, dict)
    ]
    size_values = [
        value for value in size_values if value is not None
    ]
    flags = []
    if label not in CLASS_LABELS:
        flags.append("unsupported_class_label")
    if duration is None:
        flags.append("missing_touch_duration")
    elif duration >= VERY_LONG_CONTACT_MS:
        flags.append("very_long_contact")
    elif duration >= LONG_CONTACT_MS:
        flags.append("long_contact")
    if not touch_frames:
        flags.append("missing_touch_frames")
    if max(pointer_counts, default=0) > 1:
        flags.append("multi_pointer")
    if not size_values:
        flags.append("missing_contact_size")
    expected_kind = {
        "thumb_light": "lightTap",
        "index_light": "lightTap",
        "thumb_heavy": "heavyTap",
        "index_heavy": "heavyTap",
    }.get(label)
    source_touch_kind = str(data.get("sourceTouchKind") or "")
    if (
        expected_kind is not None
        and source_touch_kind
        and source_touch_kind != expected_kind
    ):
        flags.append("unexpected_touch_kind")
    record = {
        "sourceFile": str(source_file),
        "sourceSha256": str(source_sha256),
        "sampleId": str(data.get("id") or ""),
        "classLabel": label,
        "sessionId": str(data.get("collectionSessionID") or ""),
        "deviceName": str(capability.get("deviceName") or ""),
        "bodyPosture": str(context.get("bodyPosture") or "unknown"),
        "taskId": str(context.get("taskId") or "unknown"),
        "collectionProtocol": str(
            context.get("collectionProtocol") or "unknown"
        ),
        "strengthTruthSource": str(
            context.get("strengthTruthSource") or "unknown"
        ),
        "labelStatus": str(
            context.get("labelStatus") or "legacy_unreviewed"
        ),
        "reviewSha256": str(context.get("reviewSha256") or ""),
        "sourceTouchKind": source_touch_kind,
        "durationMs": duration,
        "maxPointerCount": max(pointer_counts, default=0),
        "touchFrameCount": len(touch_frames),
        "distinctSizeCount": len({
            round(value, 6) for value in size_values
        }),
        "sizeAvailableFrameCount": len(size_values),
        "sizeMedian": median(size_values),
        "qualityFlags": ";".join(flags),
    }
    # 中文注释：可信度在明细中显式落盘，后续训练门禁无需重复推导。
    record["labelTrust"] = (
        "trusted" if is_trusted_strength_label(record) else "untrusted"
    )
    return record


def touch_duration_ms(data):
    """Read event-clock contact duration and reject nonpositive intervals."""
    alignment = data.get("timeAlignment") or {}
    down = positive_int(
        alignment.get("downEventElapsedRealtimeNanos")
    )
    lift = positive_int(
        alignment.get("liftEventElapsedRealtimeNanos")
    )
    if down <= 0 or lift <= down:
        return None
    return float(lift - down) / 1_000_000.0


def summarize_quality(records):
    """Build class, posture, and session summaries with explicit blockers."""
    rows = [
        record for record in records
        if record.get("classLabel") in CLASS_LABELS
    ]
    if not rows:
        # 中文注释：普通点击或全部冲突样本可能没有监督类别。
        # 这种情况必须保留为可审计的阻断状态，不能伪装成异常或通过。
        return empty_quality_summary(records)
    flags = {
        flag: sum(
            flag in split_flags(record.get("qualityFlags"))
            for record in rows
        )
        for flag in (
            "missing_touch_duration",
            "long_contact",
            "very_long_contact",
            "missing_touch_frames",
            "multi_pointer",
            "missing_contact_size",
            "unexpected_touch_kind",
        )
    }
    total = len(rows)
    rates = {
        name: count / total for name, count in flags.items()
    }
    duration_coverage = 1.0 - rates["missing_touch_duration"]
    long_contact_count = sum(
        "long_contact" in split_flags(record.get("qualityFlags"))
        or "very_long_contact" in split_flags(record.get("qualityFlags"))
        for record in rows
    )
    long_contact_rate = long_contact_count / total
    sessions = session_summary(rows)
    label_status_counts = Counter(
        str(record.get("labelStatus") or "unknown") for record in rows
    )
    truth_source_counts = Counter(
        str(record.get("strengthTruthSource") or "unknown")
        for record in rows
    )
    protocol_counts = Counter(
        str(record.get("collectionProtocol") or "unknown")
        for record in rows
    )
    trusted_label_count = sum(
        is_trusted_strength_label(record) for record in rows
    )
    trusted_label_rate = trusted_label_count / total
    blockers = []
    if duration_coverage < MIN_DURATION_COVERAGE:
        blockers.append("touch_duration_coverage_below_target")
    if long_contact_rate > MAX_LONG_CONTACT_RATE:
        blockers.append("long_contact_rate_above_target")
    if rates["multi_pointer"] > MAX_MULTI_POINTER_RATE:
        blockers.append("multi_pointer_rate_above_target")
    if (
        rates["unexpected_touch_kind"]
        > MAX_UNEXPECTED_TOUCH_KIND_RATE
    ):
        blockers.append("unexpected_touch_kind_rate_above_target")
    if sessions["sessionLabelBindingRate"] >= 0.8:
        blockers.append("session_label_binding_above_target")
    if trusted_label_count == 0:
        # 中文注释：任务分配标签只表示采集要求，不能单独证明动作强度。
        # 未确认且不属于交错协议的数据必须阻断训练效果声明。
        blockers.append("no_trusted_strength_labels")
    return {
        "schemaVersion": 1,
        "contract": "touch_label_quality_v1",
        "thresholds": {
            "longContactMs": LONG_CONTACT_MS,
            "veryLongContactMs": VERY_LONG_CONTACT_MS,
            "maximumLongContactRate": MAX_LONG_CONTACT_RATE,
            "maximumMultiPointerRate": MAX_MULTI_POINTER_RATE,
            "maximumUnexpectedTouchKindRate": (
                MAX_UNEXPECTED_TOUCH_KIND_RATE
            ),
            "minimumDurationCoverage": MIN_DURATION_COVERAGE,
            "maximumSessionLabelBindingRate": 0.8,
            "minimumTrustedStrengthLabelRate": 1.0,
        },
        "sampleCount": total,
        "classCounts": dict(Counter(row["classLabel"] for row in rows)),
        "deviceCounts": dict(Counter(row["deviceName"] for row in rows)),
        "protocolCounts": dict(protocol_counts),
        "labelStatusCounts": dict(label_status_counts),
        "strengthTruthSourceCounts": dict(truth_source_counts),
        "trustedStrengthLabelCount": trusted_label_count,
        "trustedStrengthLabelRate": trusted_label_rate,
        "flagCounts": flags,
        "flagRates": rates,
        "durationCoverage": duration_coverage,
        "longContactCount": long_contact_count,
        "longContactRate": long_contact_rate,
        "classMetrics": grouped_metrics(rows, "classLabel"),
        "postureMetrics": grouped_metrics(rows, "bodyPosture"),
        "sessions": sessions,
        "trainingLabelQualityReady": not blockers,
        "blockers": blockers,
        "disposition": (
            "ready"
            if not blockers
            else "audit_required_before_training_claims"
        ),
    }


def is_trusted_strength_label(record):
    """Accept reviewed labels or the complete interleaved task contract."""
    status = str(record.get("labelStatus") or "")
    if status == "confirmed":
        return True
    return (
        status == "task_assigned"
        and str(record.get("collectionProtocol") or "") == "interleaved_v2"
        and str(record.get("strengthTruthSource") or "") == "user_task"
    )


def empty_quality_summary(records):
    """Return a fail-closed report when no supervised touch rows exist."""
    return {
        "schemaVersion": 1,
        "contract": "touch_label_quality_v1",
        "thresholds": {
            "longContactMs": LONG_CONTACT_MS,
            "veryLongContactMs": VERY_LONG_CONTACT_MS,
            "maximumLongContactRate": MAX_LONG_CONTACT_RATE,
            "maximumMultiPointerRate": MAX_MULTI_POINTER_RATE,
            "maximumUnexpectedTouchKindRate": (
                MAX_UNEXPECTED_TOUCH_KIND_RATE
            ),
            "minimumDurationCoverage": MIN_DURATION_COVERAGE,
            "maximumSessionLabelBindingRate": 0.8,
            "minimumTrustedStrengthLabelRate": 1.0,
        },
        "sampleCount": 0,
        "inputRecordCount": len(records),
        "classCounts": {},
        "deviceCounts": {},
        "protocolCounts": {},
        "labelStatusCounts": {},
        "strengthTruthSourceCounts": {},
        "trustedStrengthLabelCount": 0,
        "trustedStrengthLabelRate": 0.0,
        "flagCounts": {
            name: 0
            for name in (
                "missing_touch_duration",
                "long_contact",
                "very_long_contact",
                "missing_touch_frames",
                "multi_pointer",
                "missing_contact_size",
                "unexpected_touch_kind",
            )
        },
        "flagRates": {},
        "durationCoverage": 0.0,
        "longContactCount": 0,
        "longContactRate": 0.0,
        "classMetrics": {},
        "postureMetrics": {},
        "sessions": session_summary([]),
        "trainingLabelQualityReady": False,
        "blockers": ["no_label_quality_samples"],
        "disposition": "audit_required_before_training_claims",
    }


def grouped_metrics(rows, field):
    """Return duration and contamination metrics for one metadata field."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field) or "unknown")].append(row)
    result = {}
    for name, values in sorted(grouped.items()):
        durations = [
            float(row["durationMs"])
            for row in values
            if row.get("durationMs") is not None
        ]
        result[name] = {
            "sampleCount": len(values),
            "durationPercentilesMs": percentiles(durations),
            "longContactRate": rate(
                values,
                ("long_contact", "very_long_contact"),
            ),
            "multiPointerRate": rate(values, ("multi_pointer",)),
            "sizeAvailableRate": sum(
                int(row.get("sizeAvailableFrameCount", 0)) > 0
                for row in values
            ) / len(values),
            "medianDistinctSizeCount": median([
                float(row.get("distinctSizeCount", 0))
                for row in values
            ]),
        }
    return result


def session_summary(rows):
    """Measure whether classes are separable from collection sessions."""
    grouped = defaultdict(set)
    for row in rows:
        grouped[str(row.get("sessionId") or "unknown")].add(
            row["classLabel"]
        )
    session_count = len(grouped)
    multi_class_count = sum(
        len(labels) > 1 for labels in grouped.values()
    )
    binding_count = session_count - multi_class_count
    return {
        "sessionCount": session_count,
        "multiClassSessionCount": multi_class_count,
        "multiClassSessionRate": (
            multi_class_count / session_count
            if session_count
            else 0.0
        ),
        "sessionLabelBindingCount": binding_count,
        "sessionLabelBindingRate": (
            binding_count / session_count
            if session_count
            else 0.0
        ),
    }


def write_quality_outputs(output_dir, records, prefix="touch_label_quality"):
    """Write detail CSV and compact JSON summary for the training gate."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize_quality(records)
    write_csv(
        output / (prefix + "_records.csv"),
        RECORD_FIELDS,
        records,
    )
    (output / (prefix + "_report.json")).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return summary


def read_quality_report(directory):
    """Load a prior gate report without inventing missing evidence."""
    path = Path(directory) / "touch_label_quality_report.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("contract") != "touch_label_quality_v1"
        or not isinstance(
            report.get("trainingLabelQualityReady"),
            bool,
        )
    ):
        raise RuntimeError("invalid_touch_label_quality_report")
    return report


def write_csv(path, fields, rows):
    """Write stable columns so the audit remains diffable."""
    with Path(path).open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: "" if row.get(field) is None else row.get(field)
                for field in fields
            })


def rate(rows, flags):
    """Return the fraction containing any requested quality flag."""
    return sum(
        bool(set(flags) & split_flags(row.get("qualityFlags")))
        for row in rows
    ) / len(rows) if rows else 0.0


def percentiles(values):
    """Return stable descriptive percentiles without NumPy dependency."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "p10": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "max": None,
        }
    return {
        name: ordered[index]
        for name, index in (
            ("p10", int((len(ordered) - 1) * 0.10)),
            ("p50", int((len(ordered) - 1) * 0.50)),
            ("p90", int((len(ordered) - 1) * 0.90)),
            ("p99", int((len(ordered) - 1) * 0.99)),
            ("max", len(ordered) - 1),
        )
    }


def split_flags(value):
    """Decode a compact semicolon-separated flag set."""
    return {
        item for item in str(value or "").split(";") if item
    }


def finite_positive(value):
    """Accept only finite positive contact values."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def median(values):
    """Return a finite median or zero for an empty series."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def positive_int(value):
    """Parse positive integer clocks and fail closed on malformed values."""
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return parsed if parsed > 0 else 0


def nonnegative_int(value):
    """Parse pointer counts while treating malformed values as missing."""
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, parsed)
