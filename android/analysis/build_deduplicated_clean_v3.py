#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    from android.analysis.clean_android_touch_dataset import dict_value
    from android.analysis.clean_android_touch_dataset import finite_float
    from android.analysis.clean_android_touch_dataset import impact_frame_stats
    from android.analysis.clean_android_touch_dataset import list_value
    from android.analysis.clean_android_touch_dataset import percentile
    from android.analysis.clean_android_touch_dataset import touch_frame_stats
except ImportError:
    from clean_android_touch_dataset import dict_value
    from clean_android_touch_dataset import finite_float
    from clean_android_touch_dataset import impact_frame_stats
    from clean_android_touch_dataset import list_value
    from clean_android_touch_dataset import percentile
    from clean_android_touch_dataset import touch_frame_stats


CLASS_ORDER = ("thumb_light", "thumb_heavy", "index_light", "index_heavy")
CONTEXT_FIELDS = ("touch_frame_count", "max_abs_delta", "path_travel_px")
OUTPUT_FIELDS = (
    "id",
    "path",
    "file_name",
    "sha256",
    "schema_version",
    "collector_user_name",
    "embedded_collector_user_name",
    "device_name",
    "date",
    "collection_name",
    "domain_group",
    "raw_session_id",
    "created_at",
    "class_label",
    "finger_type",
    "touch_strength",
    "motion_state",
    "source_touch_kind",
    "sample_rate_hz",
    "window_frame_count",
    "observed_window_frames",
    "touch_frame_count",
    "path_travel_px",
    "max_pointer_count",
    "max_abs_delta",
    "impact_mean_abs_delta",
    "raw_major_px",
    "raw_minor_px",
    "touch_norm_valid",
    "touch_norm_fallback",
    "contact_patch_default",
    "quality_bucket",
    "quality_score",
    "quality_reasons",
    "sample_weight",
)


def safe_slug(value, fallback="unknown"):
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return slug or fallback


def sample_id_from_name(path):
    matches = re.findall(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        Path(path).name,
    )
    return matches[-1].lower() if matches else ""


def discover_origin_metadata(origin_root):
    origin_root = Path(origin_root)
    origins = {}

    def add_directory(path, collector, collection):
        if not path.exists():
            return
        for sample_path in path.glob("*.json"):
            sample_id = sample_id_from_name(sample_path)
            if not sample_id:
                continue
            metadata = {"collector": collector, "collection": collection}
            existing = origins.get(sample_id)
            if existing is not None and existing != metadata:
                raise RuntimeError("origin_id_conflict")
            origins[sample_id] = metadata

    add_directory(origin_root / "lemon" / "lemon", "lemon", "legacy-lemon")
    add_directory(origin_root / "tao" / "tao", "tao", "legacy-tao")
    collections_root = origin_root / "collections"
    if collections_root.exists():
        for collection_path in sorted(collections_root.iterdir()):
            if not collection_path.is_dir():
                continue
            collector = collection_path.name.rsplit("_", 1)[-1]
            add_directory(
                collection_path / "raw",
                collector,
                safe_slug(collection_path.name),
            )
    return origins


def context_key(record):
    return (
        record.get("device_name", "unknown"),
        record.get("motion_state", "unknown"),
        record.get("finger_type", "unknown"),
        record.get("touch_strength", "unknown"),
    )


def build_context_stats(records):
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        if not record.get("load_ok", True):
            continue
        key = context_key(record)
        for field in CONTEXT_FIELDS:
            value = finite_float(record.get(field), None)
            if value is not None:
                grouped[key][field].append(value)

    output = {}
    for key, fields in grouped.items():
        output[key] = {}
        for field, values in fields.items():
            middle = percentile(values, 0.50)
            deviations = [abs(value - middle) for value in values]
            output[key][field] = {
                "p05": percentile(values, 0.05),
                "p50": middle,
                "p95": percentile(values, 0.95),
                "mad": percentile(deviations, 0.50),
            }
    return output


def context_field(record, context_stats, field):
    return context_stats.get(context_key(record), {}).get(field, {})


def robust_upper(record, context_stats, field, absolute_floor):
    stats = context_field(record, context_stats, field)
    middle = finite_float(stats.get("p50"), 0.0) or 0.0
    mad = finite_float(stats.get("mad"), 0.0) or 0.0
    p95 = finite_float(stats.get("p95"), 0.0) or 0.0
    return max(absolute_floor, middle + 8.0 * mad, p95 * 1.5)


def robust_duration_lower(record, context_stats):
    stats = context_field(record, context_stats, "touch_frame_count")
    p05 = finite_float(stats.get("p05"), 0.0) or 0.0
    return max(1.0, p05 * 0.5)


def decide_quality(record, context_stats):
    output = dict(record)
    reject_reasons = []
    review_reasons = []

    if not output.get("load_ok", True):
        reject_reasons.append("json_load_failed")
    if not output.get("id"):
        reject_reasons.append("missing_id")
    if output.get("class_label") not in CLASS_ORDER:
        reject_reasons.append("unsupported_class")
    if int(finite_float(output.get("observed_window_frames"), 0) or 0) < 8:
        reject_reasons.append("unusable_impact_window")
    if int(finite_float(output.get("max_pointer_count"), 0) or 0) > 1:
        reject_reasons.append("multi_pointer")
    if output.get("source_touch_kind") in ("dragStart", "drag", "dragEnd", "cancel"):
        reject_reasons.append("drag_or_cancel")

    sample_rate = finite_float(output.get("sample_rate_hz"), 0.0) or 0.0
    if sample_rate and not 120.0 <= sample_rate <= 320.0:
        review_reasons.append("sample_rate_out_of_range")
    if not output.get("touch_norm_valid", False):
        review_reasons.append("invalid_touch_normalization")
    if output.get("touch_norm_fallback") or output.get("contact_patch_default"):
        review_reasons.append("touch_patch_fallback")
    if (
        finite_float(output.get("raw_major_px"), 0.0) <= 0
        or finite_float(output.get("raw_minor_px"), 0.0) <= 0
    ):
        review_reasons.append("missing_touch_radius")
    if finite_float(output.get("max_abs_delta"), 0.0) <= 0:
        review_reasons.append("missing_impact_delta")

    if finite_float(output.get("path_travel_px"), 0.0) > robust_upper(
        output, context_stats, "path_travel_px", 36.0
    ):
        review_reasons.append("large_touch_travel")
    if finite_float(output.get("max_abs_delta"), 0.0) > robust_upper(
        output, context_stats, "max_abs_delta", 5.0
    ):
        review_reasons.append("impact_outlier_high")
    if finite_float(output.get("touch_frame_count"), 0.0) < robust_duration_lower(
        output, context_stats
    ):
        review_reasons.append("touch_duration_low")

    if reject_reasons:
        bucket = "rejected"
        score = 0
        weight = 0.0
        reasons = reject_reasons + review_reasons
    elif review_reasons:
        bucket = "review"
        score = max(40, 80 - 8 * len(set(review_reasons)))
        weight = max(0.2, min(0.6, 0.6 - 0.08 * (len(set(review_reasons)) - 1)))
        reasons = review_reasons
    else:
        bucket = "high_quality"
        score = 100
        weight = 1.0
        reasons = ["clean"]

    output["quality_bucket"] = bucket
    output["quality_score"] = score
    output["quality_reasons"] = "|".join(dict.fromkeys(reasons))
    output["sample_weight"] = round(weight, 4)
    return output


def read_sample(path, source_row, origin_metadata=None):
    origin_metadata = origin_metadata or {}
    record = {
        "load_ok": False,
        "id": source_row.get("id", ""),
        "path": str(path.resolve()),
        "file_name": path.name,
        "sha256": source_row.get("sha256", ""),
        "collector_user_name": source_row.get("collector", "unknown"),
        "device_name": source_row.get("device", "unknown"),
        "date": source_row.get("date", "unknown-date"),
    }
    try:
        with path.open("r", encoding="utf-8") as source:
            data = json.load(source)
    except (OSError, json.JSONDecodeError):
        return record

    label = dict_value(data.get("label"))
    collector = dict_value(data.get("collector"))
    capability = dict_value(data.get("deviceCapability"))
    touch_normalization = dict_value(data.get("touchNormalization"))
    selected_normalization = dict_value(touch_normalization.get("selectedResult"))
    contact_patch = dict_value(data.get("contactPatch"))
    frames = list_value(data.get("frames"))
    touch_frames = list_value(data.get("touchFrames"))
    touch_stats = touch_frame_stats(touch_frames)
    impact_stats = impact_frame_stats(frames)

    finger_type = str(label.get("fingerType") or "thumb")
    touch_strength = str(label.get("touchStrength") or "")
    class_label = finger_type + "_" + touch_strength
    embedded_collector_name = str(collector.get("userName") or record["collector_user_name"])
    collector_name = str(origin_metadata.get("collector") or embedded_collector_name)
    device_name = str(capability.get("deviceName") or record["device_name"])
    date_value = record["date"]
    collection_name = str(
        origin_metadata.get("collection")
        or "__".join(
            (safe_slug(collector_name), safe_slug(device_name), safe_slug(date_value, "unknown-date"))
        )
    )
    domain_group = "__".join((safe_slug(collector_name), safe_slug(device_name)))
    observed_indices = set()
    for frame in frames:
        frame_index = finite_float(frame.get("frameIndex"), None)
        if frame_index is not None and -5 <= int(frame_index) <= 9:
            observed_indices.add(int(frame_index))
    observed_window_frames = len(observed_indices)

    record.update(
        {
            "load_ok": True,
            "id": str(data.get("id") or record["id"]),
            "schema_version": int(finite_float(data.get("schemaVersion"), 0) or 0),
            "collector_user_name": collector_name,
            "embedded_collector_user_name": embedded_collector_name,
            "device_name": device_name,
            "collection_name": collection_name,
            "domain_group": domain_group,
            "raw_session_id": str(data.get("collectionSessionID") or ""),
            "created_at": finite_float(data.get("createdAt"), 0.0) or 0.0,
            "class_label": class_label,
            "finger_type": finger_type,
            "touch_strength": touch_strength,
            "motion_state": str(label.get("motionState") or "unknown"),
            "source_touch_kind": str(data.get("sourceTouchKind") or ""),
            "sample_rate_hz": finite_float(data.get("sampleRateHz"), 0.0) or 0.0,
            "window_frame_count": len(frames),
            "observed_window_frames": observed_window_frames,
            "touch_frame_count": len(touch_frames),
            "path_travel_px": touch_stats["path_travel_px"],
            "max_pointer_count": touch_stats["max_pointer_count"],
            "max_abs_delta": finite_float(data.get("maxAbsDelta"), 0.0) or 0.0,
            "impact_mean_abs_delta": impact_stats["impact_mean_abs_delta"],
            "raw_major_px": finite_float(selected_normalization.get("rawMajorPx"), 0.0) or 0.0,
            "raw_minor_px": finite_float(selected_normalization.get("rawMinorPx"), 0.0) or 0.0,
            "touch_norm_valid": bool(selected_normalization.get("valid")),
            "touch_norm_fallback": bool(selected_normalization.get("usedFallback")),
            "contact_patch_default": bool(contact_patch.get("defaultApplied")),
        }
    )
    return record


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in OUTPUT_FIELDS})


def build_dataset(source_manifest, data_root, output_dir, origin_root=None):
    source_manifest = Path(source_manifest)
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    with source_manifest.open("r", encoding="utf-8", newline="") as source:
        source_rows = list(csv.DictReader(source))

    origins = discover_origin_metadata(origin_root) if origin_root is not None else {}
    records = []
    origin_matched = 0
    for source_row in source_rows:
        relative = Path(source_row.get("destination", ""))
        path = relative if relative.is_absolute() else data_root / relative
        sample_id = str(source_row.get("id") or "").lower()
        origin = origins.get(sample_id)
        if origin is not None:
            origin_matched += 1
        records.append(read_sample(path, source_row, origin))

    stats = build_context_stats(records)
    decided = [decide_quality(record, stats) for record in records]
    decided.sort(key=lambda record: (finite_float(record.get("created_at"), 0.0), record.get("id", "")))
    training = [record for record in decided if record["quality_bucket"] != "rejected"]
    write_csv(output_dir / "master_manifest.csv", decided)
    write_csv(output_dir / "quality_training_manifest.csv", training)
    write_csv(output_dir / "training_manifest.csv", training)

    reason_counts = Counter()
    for record in decided:
        for reason in str(record.get("quality_reasons") or "").split("|"):
            if reason:
                reason_counts[reason] += 1
    summary = {
        "total": len(decided),
        "trainingRows": len(training),
        "bucketCounts": dict(sorted(Counter(record["quality_bucket"] for record in decided).items())),
        "classCounts": dict(sorted(Counter(record.get("class_label", "") for record in decided).items())),
        "domainCounts": dict(sorted(Counter(record.get("domain_group", "") for record in decided).items())),
        "reasonCounts": dict(reason_counts.most_common()),
        "contextCount": len(stats),
        "originMatched": origin_matched,
        "originMissing": len(source_rows) - origin_matched if origin_root is not None else 0,
        "collectorMetadataMismatches": sum(
            record.get("collector_user_name") != record.get("embedded_collector_user_name")
            for record in decided
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Build the deduplicated Clean V3 dataset.")
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--origin-root", type=Path)
    return parser.parse_args()


def main():
    arguments = parse_args()
    summary = build_dataset(
        arguments.source_manifest,
        arguments.data_root,
        arguments.output_dir,
        origin_root=arguments.origin_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
