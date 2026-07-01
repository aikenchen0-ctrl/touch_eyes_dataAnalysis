#!/usr/bin/env python3
import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path


KEEP_DIR = "high_quality"
REVIEW_DIR = "review"
REJECT_DIR = "rejected"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--copy-files", action="store_true")
    parser.add_argument("--max-per-class", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path in sorted(input_dir.rglob("*.json")):
        records.append(read_record(path))

    valid_records = [record for record in records if record.get("load_ok")]
    class_stats = build_class_stats(valid_records)
    for record in valid_records:
        apply_quality_decision(record, class_stats)
    for record in records:
        if not record.get("load_ok"):
            record["quality_bucket"] = REJECT_DIR
            record["quality_score"] = 0
            record["quality_reasons"] = "json_load_failed"

    selected_records = [record for record in records if record.get("quality_bucket") == KEEP_DIR]
    balanced_records = balance_records(selected_records, args.max_per_class)
    write_manifest(output_dir / "manifest.csv", records)
    write_manifest(output_dir / "train_manifest.csv", selected_records)
    write_manifest(output_dir / "balanced_train_manifest.csv", balanced_records)
    write_summary(output_dir / "summary.json", records, selected_records, balanced_records)
    if args.copy_files:
        copy_grouped_files(output_dir, records)


def read_record(path):
    record = {
        "path": str(path),
        "file_name": path.name,
        "load_ok": False,
    }
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        record["load_error"] = error.__class__.__name__
        return record

    label = data.get("label") or {}
    touch_strength = str(label.get("touchStrength") or "")
    motion_state = str(label.get("motionState") or "")
    # 中文注释：历史 Android 数据采集时未写入手指字段，旧样本统一按拇指处理。
    finger_type = str(label.get("fingerType") or "thumb")
    frames = list_value(data.get("frames"))
    touch_frames = list_value(data.get("touchFrames"))
    touch_norm = dict_value(data.get("touchNormalization"))
    selected_norm = dict_value(touch_norm.get("selectedResult"))
    contact_patch = dict_value(data.get("contactPatch"))
    motion_signal = dict_value(data.get("motionSignal"))
    capability = dict_value(data.get("deviceCapability"))
    cnn = dict_value(data.get("cnnPrediction"))

    touch_values = touch_frame_stats(touch_frames)
    impact_values = impact_frame_stats(frames)
    label_id = motion_state + "_" + touch_strength
    if finger_type and finger_type != "unknown":
        label_id = label_id + "_" + finger_type
    cnn_heavy = finite_float(cnn.get("heavyProbability"), None)
    cnn_light = finite_float(cnn.get("lightProbability"), None)
    cnn_margin = None
    if cnn_heavy is not None and cnn_light is not None:
        cnn_margin = abs(cnn_heavy - cnn_light)

    record.update({
        "load_ok": True,
        "id": str(data.get("id") or ""),
        "created_at": finite_float(data.get("createdAt"), 0),
        "label_id": label_id,
        "motion_state": motion_state,
        "touch_strength": touch_strength,
        "finger_type": finger_type,
        "source_touch_kind": str(data.get("sourceTouchKind") or ""),
        "sample_rate_hz": finite_float(data.get("sampleRateHz"), 0),
        "motion_frame_count": int(finite_float(motion_signal.get("frameCount"), len(frames)) or 0),
        "window_frame_count": len(frames),
        "touch_frame_count": len(touch_frames),
        "max_abs_delta": finite_float(data.get("maxAbsDelta"), 0),
        "device_level": str(capability.get("level") or ""),
        "has_major": bool(capability.get("hasMajor")),
        "has_minor": bool(capability.get("hasMinor")),
        "touch_norm_valid": bool(selected_norm.get("valid")),
        "touch_norm_fallback": bool(selected_norm.get("usedFallback")),
        "touch_norm_confidence": finite_float(selected_norm.get("confidence"), 0),
        "raw_major_px": finite_float(selected_norm.get("rawMajorPx"), 0),
        "raw_minor_px": finite_float(selected_norm.get("rawMinorPx"), 0),
        "norm_area": finite_float(selected_norm.get("areaValue"), 0),
        "baseline_area": finite_float(dict_value(touch_norm.get("sessionBaseline")).get("areaValue"), 0),
        "contact_patch_valid": bool(contact_patch.get("valid")),
        "contact_patch_default": bool(contact_patch.get("defaultApplied")),
        "cnn_model_id": str(cnn.get("modelId") or ""),
        "cnn_decision": str(cnn.get("decision") or ""),
        "cnn_heavy_probability": cnn_heavy if cnn_heavy is not None else "",
        "cnn_light_probability": cnn_light if cnn_light is not None else "",
        "cnn_margin": cnn_margin if cnn_margin is not None else "",
        "x_travel_px": touch_values["x_travel_px"],
        "y_travel_px": touch_values["y_travel_px"],
        "path_travel_px": touch_values["path_travel_px"],
        "max_pointer_count": touch_values["max_pointer_count"],
        "duration_ms": touch_values["duration_ms"],
        "max_pressure": touch_values["max_pressure"],
        "max_size": touch_values["max_size"],
        "max_touch_major": touch_values["max_touch_major"],
        "max_touch_minor": touch_values["max_touch_minor"],
        "mean_touch_major": touch_values["mean_touch_major"],
        "mean_touch_minor": touch_values["mean_touch_minor"],
        "impact_mean_abs_delta": impact_values["impact_mean_abs_delta"],
        "impact_peak_index": impact_values["impact_peak_index"],
        "impact_peak_time": impact_values["impact_peak_time"],
    })
    return record


def build_class_stats(records):
    values = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record["touch_strength"] not in ("light", "heavy"):
            continue
        strength = record["touch_strength"]
        for key in (
                "touch_frame_count",
                "max_touch_major",
                "max_touch_minor",
                "max_abs_delta",
                "path_travel_px",
                "norm_area",
                "baseline_area"):
            value = finite_float(record.get(key), None)
            if value is not None:
                values[strength][key].append(value)

    stats = {}
    for strength, fields in values.items():
        stats[strength] = {}
        for key, items in fields.items():
            stats[strength][key] = {
                "p05": percentile(items, 0.05),
                "p10": percentile(items, 0.10),
                "p25": percentile(items, 0.25),
                "p50": percentile(items, 0.50),
                "p75": percentile(items, 0.75),
                "p90": percentile(items, 0.90),
                "p95": percentile(items, 0.95),
            }
    return stats


def apply_quality_decision(record, class_stats):
    reasons = []
    warnings = []
    score = 100

    if record["touch_strength"] not in ("light", "heavy"):
        reasons.append("missing_manual_strength")
        score -= 70
    if record["motion_state"] not in ("handheld_static", "placed_static", "walking"):
        reasons.append("missing_motion_state")
        score -= 20
    if record["window_frame_count"] != 21:
        reasons.append("bad_window_frame_count")
        score -= 40
    if not 120 <= record["sample_rate_hz"] <= 320:
        reasons.append("sample_rate_out_of_range")
        score -= 25
    if record["motion_frame_count"] < 120:
        reasons.append("short_motion_signal")
        score -= 20
    if record["device_level"] != "A":
        reasons.append("weak_device_capability")
        score -= 30
    if record["max_pointer_count"] > 1:
        reasons.append("multi_pointer")
        score -= 70
    if record["source_touch_kind"] in ("dragStart", "drag", "dragEnd", "cancel"):
        reasons.append("drag_or_cancel")
        score -= 70
    if record["path_travel_px"] > 36:
        reasons.append("large_touch_travel")
        score -= 25
    if record["touch_frame_count"] < 2:
        reasons.append("too_few_touch_frames")
        score -= 20
    if record["touch_strength"] == "heavy" and record["touch_frame_count"] < 4:
        reasons.append("heavy_too_short")
        score -= 30
    if not record["touch_norm_valid"]:
        reasons.append("invalid_touch_normalization")
        score -= 25
    if record["touch_norm_fallback"] or record["contact_patch_default"]:
        reasons.append("touch_patch_fallback")
        score -= 25
    if record["raw_major_px"] <= 0 or record["raw_minor_px"] <= 0:
        reasons.append("missing_touch_radius")
        score -= 20
    if record["max_abs_delta"] <= 0:
        reasons.append("missing_impact_delta")
        score -= 30
    if record["max_abs_delta"] > 5.0:
        reasons.append("impact_outlier_high")
        score -= 25

    label_conflict = label_conflict_reason(record, class_stats)
    if label_conflict:
        reasons.append(label_conflict)
        score -= 25

    cnn_decision = record.get("cnn_decision")
    margin = finite_float(record.get("cnn_margin"), None)
    if cnn_decision in ("light", "heavy") and margin is not None and margin >= 0.95 and cnn_decision != record["touch_strength"]:
        warnings.append("old_cnn_label_conflict")

    kind = record.get("source_touch_kind")
    strength = record.get("touch_strength")
    if kind == "lightTap" and strength == "heavy":
        warnings.append("tap_kind_manual_label_mismatch")
    if kind == "heavyTap" and strength == "light":
        warnings.append("tap_kind_manual_label_mismatch")

    score = max(0, min(100, score))
    if score >= 78 and not hard_reject(reasons):
        bucket = KEEP_DIR
    elif score >= 48 and not fatal_reject(reasons):
        bucket = REVIEW_DIR
    else:
        bucket = REJECT_DIR

    record["quality_score"] = score
    record["quality_bucket"] = bucket
    record["quality_reasons"] = "|".join(reasons) if reasons else "clean"
    record["quality_warnings"] = "|".join(warnings) if warnings else ""
    record["sample_weight"] = sample_weight_for_record(record, class_stats)


def label_conflict_reason(record, class_stats):
    strength = record["touch_strength"]
    if strength not in ("light", "heavy"):
        return ""
    light_stats = class_stats.get("light", {})
    heavy_stats = class_stats.get("heavy", {})
    if not light_stats or not heavy_stats:
        return ""

    major = finite_float(record.get("max_touch_major"), 0)
    minor = finite_float(record.get("max_touch_minor"), 0)
    delta = finite_float(record.get("max_abs_delta"), 0)
    frames = finite_float(record.get("touch_frame_count"), 0)
    area = finite_float(record.get("norm_area"), 0)

    light_major_p90 = get_stat(light_stats, "max_touch_major", "p90")
    light_area_p90 = get_stat(light_stats, "norm_area", "p90")
    heavy_major_p10 = get_stat(heavy_stats, "max_touch_major", "p10")
    heavy_frames_p10 = get_stat(heavy_stats, "touch_frame_count", "p10")
    heavy_delta_p10 = get_stat(heavy_stats, "max_abs_delta", "p10")

    if strength == "light":
        high_radius = light_major_p90 > 0 and major > light_major_p90 * 1.5
        high_area = light_area_p90 > 0 and area > light_area_p90 * 1.8
        if high_radius and high_area:
            return "light_label_heavy_shape"
    if strength == "heavy":
        weak_radius = heavy_major_p10 > 0 and major < heavy_major_p10 * 0.75 and minor < heavy_major_p10 * 0.75
        weak_duration = heavy_frames_p10 > 0 and frames < max(3, heavy_frames_p10 * 0.5)
        weak_impact = heavy_delta_p10 > 0 and delta < heavy_delta_p10 * 0.55
        if weak_radius and weak_duration and weak_impact:
            return "heavy_label_light_shape"
    return ""


def hard_reject(reasons):
    hard = {
        "missing_manual_strength",
        "bad_window_frame_count",
        "multi_pointer",
        "drag_or_cancel",
        "weak_device_capability",
        "invalid_touch_normalization",
        "missing_touch_radius",
    }
    return any(reason in hard for reason in reasons)


def fatal_reject(reasons):
    fatal = {
        "missing_manual_strength",
        "bad_window_frame_count",
        "multi_pointer",
        "drag_or_cancel",
    }
    return any(reason in fatal for reason in reasons)


def balance_records(records, max_per_class):
    keep = [record for record in records if record.get("quality_bucket") == KEEP_DIR]
    if max_per_class <= 0:
        grouped = defaultdict(list)
        for record in keep:
            grouped[record["label_id"]].append(record)
        if not grouped:
            return []
        limit = min(len(items) for items in grouped.values())
    else:
        limit = max_per_class
    grouped = defaultdict(list)
    for record in keep:
        grouped[record["label_id"]].append(record)
    selected = []
    for label_id in sorted(grouped):
        items = sorted(grouped[label_id], key=lambda item: (-item["quality_score"], item["created_at"], item["file_name"]))
        selected.extend(items[:limit])
    return sorted(selected, key=lambda item: item["path"])


def sample_weight_for_record(record, class_stats):
    weight = 1.0
    if record.get("quality_bucket") != KEEP_DIR:
        return weight
    reasons = set(str(record.get("quality_reasons") or "").split("|"))
    warnings = set(str(record.get("quality_warnings") or "").split("|"))
    if "light_label_heavy_shape" in reasons or "light_label_heavy_shape" in warnings:
        weight *= 2.5
    if "heavy_label_light_shape" in reasons or "heavy_label_light_shape" in warnings:
        weight *= 1.5
    light_major_p90 = get_stat(class_stats.get("light", {}), "max_touch_major", "p90")
    major = finite_float(record.get("max_touch_major"), 0)
    large_light_major = major > 18
    if light_major_p90 > 0:
        large_light_major = large_light_major or major >= light_major_p90
    if record.get("touch_strength") == "light" and large_light_major:
        weight *= 3.0
    return round(weight, 4)


def write_manifest(path, records):
    fields = [
        "quality_bucket",
        "quality_score",
        "quality_reasons",
        "quality_warnings",
        "sample_weight",
        "label_id",
        "motion_state",
        "touch_strength",
        "finger_type",
        "file_name",
        "path",
        "id",
        "created_at",
        "source_touch_kind",
        "device_level",
        "sample_rate_hz",
        "motion_frame_count",
        "window_frame_count",
        "touch_frame_count",
        "duration_ms",
        "path_travel_px",
        "max_pointer_count",
        "max_abs_delta",
        "impact_mean_abs_delta",
        "impact_peak_index",
        "impact_peak_time",
        "raw_major_px",
        "raw_minor_px",
        "max_touch_major",
        "max_touch_minor",
        "mean_touch_major",
        "mean_touch_minor",
        "norm_area",
        "baseline_area",
        "touch_norm_confidence",
        "cnn_model_id",
        "cnn_decision",
        "cnn_heavy_probability",
        "cnn_light_probability",
        "cnn_margin",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (item.get("quality_bucket", ""), item.get("label_id", ""), item.get("file_name", ""))):
            writer.writerow({field: record.get(field, "") for field in fields})


def write_summary(path, records, selected_records, balanced_records):
    bucket_counts = Counter(record.get("quality_bucket", "unknown") for record in records)
    label_counts = Counter(record.get("label_id", "unknown") for record in records)
    selected_label_counts = Counter(record.get("label_id", "unknown") for record in selected_records)
    balanced_label_counts = Counter(record.get("label_id", "unknown") for record in balanced_records)
    reason_counts = Counter()
    warning_counts = Counter()
    for record in records:
        for reason in str(record.get("quality_reasons", "")).split("|"):
            if reason:
                reason_counts[reason] += 1
        for warning in str(record.get("quality_warnings", "")).split("|"):
            if warning:
                warning_counts[warning] += 1
    summary = {
        "total": len(records),
        "selectedForTraining": len(selected_records),
        "balancedForTraining": len(balanced_records),
        "bucketCounts": dict(sorted(bucket_counts.items())),
        "labelCounts": dict(sorted(label_counts.items())),
        "selectedLabelCounts": dict(sorted(selected_label_counts.items())),
        "balancedLabelCounts": dict(sorted(balanced_label_counts.items())),
        "reasonCounts": dict(reason_counts.most_common()),
        "warningCounts": dict(warning_counts.most_common()),
        "oldCnnWarning": "old_cnn_predictions_are_diagnostics_only",
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)


def copy_grouped_files(output_dir, records):
    for bucket in (KEEP_DIR, REVIEW_DIR, REJECT_DIR):
        target = output_dir / bucket
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
    for record in records:
        bucket = record.get("quality_bucket", REJECT_DIR)
        label_id = safe_name(record.get("label_id") or "unknown")
        source = Path(record["path"])
        target_dir = output_dir / bucket / label_id
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_dir / source.name)


def touch_frame_stats(frames):
    values = {
        "x_travel_px": 0.0,
        "y_travel_px": 0.0,
        "path_travel_px": 0.0,
        "max_pointer_count": 0,
        "duration_ms": 0.0,
        "max_pressure": 0.0,
        "max_size": 0.0,
        "max_touch_major": 0.0,
        "max_touch_minor": 0.0,
        "mean_touch_major": 0.0,
        "mean_touch_minor": 0.0,
    }
    if not frames:
        return values
    xs = []
    ys = []
    times = []
    majors = []
    minors = []
    last_x = None
    last_y = None
    path = 0.0
    for frame in frames:
        x = finite_float(frame.get("x"), 0)
        y = finite_float(frame.get("y"), 0)
        xs.append(x)
        ys.append(y)
        times.append(finite_float(frame.get("eventTimeMillis"), 0))
        pressure = finite_float(frame.get("pressure"), 0)
        size = finite_float(frame.get("size"), 0)
        major = max(0.0, finite_float(frame.get("touchMajor"), 0), finite_float(frame.get("toolMajor"), 0))
        minor = max(0.0, finite_float(frame.get("touchMinor"), 0), finite_float(frame.get("toolMinor"), 0))
        majors.append(major)
        minors.append(minor)
        values["max_pressure"] = max(values["max_pressure"], pressure)
        values["max_size"] = max(values["max_size"], size)
        values["max_pointer_count"] = max(values["max_pointer_count"], int(finite_float(frame.get("pointerCount"), 0) or 0))
        if last_x is not None and last_y is not None:
            path += math.hypot(x - last_x, y - last_y)
        last_x = x
        last_y = y
    values["x_travel_px"] = max(xs) - min(xs)
    values["y_travel_px"] = max(ys) - min(ys)
    values["path_travel_px"] = path
    values["duration_ms"] = max(times) - min(times) if len(times) >= 2 else 0.0
    values["max_touch_major"] = max(majors) if majors else 0.0
    values["max_touch_minor"] = max(minors) if minors else 0.0
    values["mean_touch_major"] = sum(majors) / len(majors) if majors else 0.0
    values["mean_touch_minor"] = sum(minors) / len(minors) if minors else 0.0
    return values


def impact_frame_stats(frames):
    values = {
        "impact_mean_abs_delta": 0.0,
        "impact_peak_index": 0,
        "impact_peak_time": 0.0,
    }
    if not frames:
        return values
    abs_values = []
    best_abs = -1.0
    best_index = 0
    best_time = 0.0
    for frame in frames:
        value = abs(finite_float(frame.get("delta"), 0))
        abs_values.append(value)
        if value > best_abs:
            best_abs = value
            best_index = int(finite_float(frame.get("frameIndex"), 0) or 0)
            best_time = finite_float(frame.get("relativeTimestamp"), 0)
    values["impact_mean_abs_delta"] = sum(abs_values) / len(abs_values)
    values["impact_peak_index"] = best_index
    values["impact_peak_time"] = best_time
    return values


def percentile(values, fraction):
    clean = sorted(finite_float(value, None) for value in values)
    clean = [value for value in clean if value is not None]
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return clean[lower] * (1 - weight) + clean[upper] * weight


def get_stat(stats, key, name):
    return finite_float(dict_value(stats.get(key)).get(name), 0)


def list_value(value):
    return value if isinstance(value, list) else []


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


def safe_name(value):
    output = []
    for char in str(value):
        if char.isalnum() or char in ("-", "_", "."):
            output.append(char)
        else:
            output.append("_")
    return "".join(output) or "unknown"


if __name__ == "__main__":
    main()
