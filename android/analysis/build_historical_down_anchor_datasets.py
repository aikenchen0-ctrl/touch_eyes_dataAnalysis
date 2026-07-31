#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import shutil
import uuid
from collections import Counter
from copy import deepcopy
from pathlib import Path


SIGNAL_FIELDS = (
    "delta",
    "x",
    "y",
    "z",
    "rotationRateX",
    "rotationRateY",
    "rotationRateZ",
    "gravityX",
    "gravityY",
    "gravityZ",
    "x_norm",
    "y_norm",
)

A_MANIFEST_FIELDS = (
    "id",
    "sha256",
    "source_path",
    "schema_version",
    "collector",
    "device",
    "date",
    "session_id",
    "class_label",
    "down_timestamp",
    "down_timestamp_source",
    "anchor_timestamp",
    "sample_rate_hz",
    "virtual_down_frame_index",
    "nearest_source_frame_index",
    "nearest_time_error_seconds",
    "pre_down_observed_count",
    "post_down_observed_count",
    "observed_frame_count",
    "placeholder_frame_count",
)

B_MANIFEST_FIELDS = A_MANIFEST_FIELDS + ("derived_path",)


def finite_float(value, fallback=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def is_observed_frame(frame):
    return any(abs(finite_float(frame.get(name), 0.0)) > 1e-12 for name in SIGNAL_FIELDS)


def touch_down_timestamp(data):
    touch_frames = data.get("touchFrames") or []
    if touch_frames:
        event_time_millis = finite_float(touch_frames[0].get("eventTimeMillis"), None)
        if event_time_millis is not None:
            return event_time_millis / 1000.0, "touchFrames[0].eventTimeMillis"
    touch_events = data.get("touchEvents") or []
    if touch_events:
        timestamp = finite_float(touch_events[0].get("timestamp"), None)
        if timestamp is not None:
            return timestamp, "touchEvents[0].timestamp"
    return None, "missing"


def frame_time_since_down(frame, down_timestamp, anchor_timestamp):
    stored = finite_float(frame.get("timeSinceTouchDown"), None)
    if stored is not None:
        return stored
    relative = finite_float(frame.get("relativeTimestamp"), None)
    if relative is None:
        return None
    return anchor_timestamp + relative - down_timestamp


def analyze_sample(data):
    down_timestamp, down_source = touch_down_timestamp(data)
    anchor_timestamp = finite_float(data.get("anchorTimestamp"), None)
    sample_rate_hz = finite_float(data.get("sampleRateHz"), None)
    frames = data.get("frames") or []
    observed = []
    placeholder_count = 0
    if down_timestamp is not None and anchor_timestamp is not None:
        for position, frame in enumerate(frames):
            if not is_observed_frame(frame):
                placeholder_count += 1
                continue
            time_since_down = frame_time_since_down(frame, down_timestamp, anchor_timestamp)
            if time_since_down is None:
                continue
            observed.append({
                "frame": frame,
                "position": position,
                "time_since_down": time_since_down,
            })
    observed.sort(key=lambda item: (item["time_since_down"], item["position"]))
    result = {
        "eligible_a": False,
        "eligible_b": False,
        "exclusion_reason_a": "",
        "exclusion_reason_b": "not_eligible_a",
        "down_timestamp": down_timestamp,
        "down_timestamp_source": down_source,
        "anchor_timestamp": anchor_timestamp,
        "sample_rate_hz": sample_rate_hz,
        "virtual_down_frame_index": None,
        "nearest_source_frame_index": None,
        "nearest_time_error_seconds": None,
        "observed_frame_count": len(observed),
        "placeholder_frame_count": placeholder_count,
        "pre_down_observed_count": sum(1 for item in observed if item["time_since_down"] < 0),
        "post_down_observed_count": sum(1 for item in observed if item["time_since_down"] > 0),
        "down_frames": [],
    }
    if down_timestamp is None:
        result["exclusion_reason_a"] = "missing_down_timestamp"
        return result
    if anchor_timestamp is None:
        result["exclusion_reason_a"] = "missing_anchor_timestamp"
        return result
    if sample_rate_hz is None or sample_rate_hz <= 0:
        result["exclusion_reason_a"] = "invalid_sample_rate"
        return result
    result["virtual_down_frame_index"] = (down_timestamp - anchor_timestamp) * sample_rate_hz
    if len(observed) < 2:
        result["exclusion_reason_a"] = "insufficient_observed_frames"
        return result
    nearest_index = min(range(len(observed)), key=lambda index: abs(observed[index]["time_since_down"]))
    nearest = observed[nearest_index]
    result["nearest_source_frame_index"] = finite_float(nearest["frame"].get("frameIndex"), nearest["position"])
    result["nearest_time_error_seconds"] = abs(nearest["time_since_down"])
    if observed[0]["time_since_down"] > 0 or observed[-1]["time_since_down"] < 0:
        result["exclusion_reason_a"] = "down_outside_observed_range"
        return result
    result["eligible_a"] = True
    result["exclusion_reason_b"] = "insufficient_down_context"
    if nearest_index < 5 or len(observed) - nearest_index - 1 < 9:
        return result
    period = 1.0 / sample_rate_hz
    if abs(nearest["time_since_down"]) > period * 0.75:
        result["exclusion_reason_b"] = "down_frame_too_far"
        return result
    selected = observed[nearest_index - 5:nearest_index + 10]
    gaps = [
        selected[index + 1]["time_since_down"] - selected[index]["time_since_down"]
        for index in range(len(selected) - 1)
    ]
    if any(gap <= 0 or gap > period * 1.75 for gap in gaps):
        result["exclusion_reason_b"] = "non_contiguous_down_window"
        return result
    down_frames = []
    for frame_index, item in zip(range(-5, 10), selected):
        frame = deepcopy(item["frame"])
        frame["sourceFrameIndex"] = frame.get("frameIndex")
        frame["sourceRelativeTimestamp"] = frame.get("relativeTimestamp")
        frame["sourceTimeSinceTouchDown"] = item["time_since_down"]
        frame["frameIndex"] = frame_index
        frame["relativeTimestamp"] = item["time_since_down"]
        frame["timeSinceAnchor"] = item["time_since_down"]
        frame["timeSinceTouchDown"] = item["time_since_down"]
        down_frames.append(frame)
    result["eligible_b"] = True
    result["exclusion_reason_b"] = ""
    result["down_frames"] = down_frames
    return result


def resolve_source_path(row, manifest_path):
    value = row.get("destination") or row.get("path") or ""
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    resolved = manifest_path.parent / candidate
    return resolved if resolved.exists() else None


def class_label(data):
    label = data.get("label") or {}
    finger = str(label.get("fingerType") or "thumb")
    strength = str(label.get("touchStrength") or "")
    return finger + "_" + strength


def collector_name(row, data):
    value = data.get("collector")
    if isinstance(value, dict):
        value = value.get("userName")
    return str(value or row.get("collector") or "unknown")


def manifest_record(row, data, source_path, analysis):
    return {
        "id": str(data.get("id") or row.get("id") or ""),
        "sha256": str(row.get("sha256") or ""),
        "source_path": str(source_path.resolve()),
        "schema_version": int(finite_float(data.get("schemaVersion"), 0) or 0),
        "collector": collector_name(row, data),
        "device": str(row.get("device") or (data.get("deviceCapability") or {}).get("deviceName") or "unknown"),
        "date": str(row.get("date") or "unknown"),
        "session_id": str(data.get("collectionSessionID") or ""),
        "class_label": class_label(data),
        "down_timestamp": analysis["down_timestamp"],
        "down_timestamp_source": analysis["down_timestamp_source"],
        "anchor_timestamp": analysis["anchor_timestamp"],
        "sample_rate_hz": analysis["sample_rate_hz"],
        "virtual_down_frame_index": analysis["virtual_down_frame_index"],
        "nearest_source_frame_index": analysis["nearest_source_frame_index"],
        "nearest_time_error_seconds": analysis["nearest_time_error_seconds"],
        "pre_down_observed_count": analysis["pre_down_observed_count"],
        "post_down_observed_count": analysis["post_down_observed_count"],
        "observed_frame_count": analysis["observed_frame_count"],
        "placeholder_frame_count": analysis["placeholder_frame_count"],
    }


def derived_payload(record, data, analysis):
    return {
        "schemaVersion": 1,
        "sourceId": record["id"],
        "sourcePath": record["source_path"],
        "sourceSchemaVersion": record["schema_version"],
        "label": data.get("label") or {},
        "downAnchorTimestamp": analysis["down_timestamp"],
        "downReferenceFrameIndex": 0,
        "sampleRateHz": analysis["sample_rate_hz"],
        "downFrames": analysis["down_frames"],
    }


def safe_file_name(value):
    safe = "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in value)
    return safe or uuid.uuid4().hex


def write_manifest(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def counter_dict(counter):
    return dict(sorted(counter.items(), key=lambda item: str(item[0])))


def dataset_summary(rows, exclusion_counts):
    return {
        "retained": len(rows),
        "excludedReasons": counter_dict(exclusion_counts),
        "schemaCounts": counter_dict(Counter(str(row["schema_version"]) for row in rows)),
        "collectorCounts": counter_dict(Counter(row["collector"] for row in rows)),
        "deviceCounts": counter_dict(Counter(row["device"] for row in rows)),
        "classCounts": counter_dict(Counter(row["class_label"] for row in rows)),
    }


def publish_directory(temp_root, output_root):
    if output_root.exists():
        shutil.rmtree(output_root)
    os.replace(temp_root, output_root)


def build_datasets(source_manifest, output_root):
    source_manifest = Path(source_manifest).resolve()
    output_root = Path(output_root).resolve()
    temp_root = output_root.parent / (output_root.name + ".tmp-" + uuid.uuid4().hex)
    a_root = temp_root / "a_down_inside_saved_window"
    b_root = temp_root / "b_complete_down_minus5_plus9"
    a_rows = []
    b_rows = []
    a_ids = set()
    b_ids = set()
    a_exclusions = Counter()
    b_exclusions = Counter()
    discovered = 0
    load_failures = 0
    try:
        with source_manifest.open("r", encoding="utf-8-sig", newline="") as source:
            for source_row in csv.DictReader(source):
                discovered += 1
                source_path = resolve_source_path(source_row, source_manifest)
                if source_path is None:
                    load_failures += 1
                    a_exclusions["source_path_missing"] += 1
                    b_exclusions["source_path_missing"] += 1
                    continue
                try:
                    with source_path.open("r", encoding="utf-8") as input_file:
                        data = json.load(input_file)
                except (OSError, json.JSONDecodeError):
                    load_failures += 1
                    a_exclusions["json_load_failed"] += 1
                    b_exclusions["json_load_failed"] += 1
                    continue
                analysis = analyze_sample(data)
                if not analysis["eligible_a"]:
                    a_exclusions[analysis["exclusion_reason_a"] or "unknown"] += 1
                    b_exclusions["not_eligible_a"] += 1
                    continue
                record = manifest_record(source_row, data, source_path, analysis)
                sample_id = record["id"]
                if not sample_id or sample_id in a_ids:
                    a_exclusions["duplicate_or_missing_id"] += 1
                    b_exclusions["duplicate_or_missing_id"] += 1
                    continue
                a_ids.add(sample_id)
                a_rows.append(record)
                if not analysis["eligible_b"]:
                    b_exclusions[analysis["exclusion_reason_b"] or "unknown"] += 1
                    continue
                if sample_id in b_ids:
                    b_exclusions["duplicate_id"] += 1
                    continue
                b_ids.add(sample_id)
                file_name = safe_file_name(sample_id) + ".json"
                derived_relative = Path("samples") / file_name
                write_json(b_root / derived_relative, derived_payload(record, data, analysis))
                b_record = dict(record)
                b_record["derived_path"] = str(derived_relative)
                b_rows.append(b_record)
                if discovered % 5000 == 0:
                    print(json.dumps({"processed": discovered, "datasetA": len(a_rows), "datasetB": len(b_rows)}), flush=True)
        if not b_ids.issubset(a_ids):
            raise RuntimeError("dataset_b_not_subset_of_a")
        write_manifest(a_root / "manifest.csv", A_MANIFEST_FIELDS, a_rows)
        write_manifest(b_root / "manifest.csv", B_MANIFEST_FIELDS, b_rows)
        a_summary = dataset_summary(a_rows, a_exclusions)
        b_summary = dataset_summary(b_rows, b_exclusions)
        root_summary = {
            "sourceManifest": str(source_manifest),
            "discovered": discovered,
            "loadFailures": load_failures,
            "datasetA": a_summary,
            "datasetB": b_summary,
            "datasetBSubsetOfA": b_ids.issubset(a_ids),
        }
        write_json(a_root / "summary.json", a_summary)
        write_json(b_root / "summary.json", b_summary)
        write_json(temp_root / "summary.json", root_summary)
        publish_directory(temp_root, output_root)
        return root_summary
    except BaseException:
        if temp_root.exists():
            shutil.rmtree(temp_root)
        raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_datasets(args.source_manifest, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
