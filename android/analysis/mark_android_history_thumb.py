#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


TARGET_FINGER_TYPE = "thumb"
UNKNOWN_VALUES = {"", "unknown", "none", "null"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="android/analysis/datasets")
    args = parser.parse_args()

    root = Path(args.root)
    json_stats = mark_json_records(root)
    csv_stats = mark_csv_records(root)
    summary_stats = mark_summary_records(root)
    print(json.dumps({
        "json": json_stats,
        "csv": csv_stats,
        "summary": summary_stats,
    }, indent=2, sort_keys=True))


def mark_json_records(root):
    stats = {
        "scanned": 0,
        "sample_records": 0,
        "updated": 0,
        "kept": 0,
        "skipped": 0,
        "errors": 0,
    }
    for path in sorted(root.rglob("*.json")):
        stats["scanned"] += 1
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            stats["errors"] += 1
            continue

        if not is_android_sample_record(data):
            stats["skipped"] += 1
            continue

        stats["sample_records"] += 1
        label = data.setdefault("label", {})
        finger_type = normalize_value(label.get("fingerType"))
        if finger_type in UNKNOWN_VALUES:
            label["fingerType"] = TARGET_FINGER_TYPE
            write_json(path, data)
            stats["updated"] += 1
        else:
            stats["kept"] += 1
    return stats


def mark_csv_records(root):
    stats = {
        "scanned": 0,
        "updated": 0,
        "skipped": 0,
        "errors": 0,
    }
    for path in sorted(root.rglob("*.csv")):
        stats["scanned"] += 1
        try:
            with path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                if reader.fieldnames is None:
                    stats["skipped"] += 1
                    continue
                fieldnames = list(reader.fieldnames)
                rows = list(reader)
        except (OSError, csv.Error, UnicodeDecodeError):
            stats["errors"] += 1
            continue

        updated, output_fields = mark_csv_rows(fieldnames, rows)
        if not updated:
            stats["skipped"] += 1
            continue

        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=output_fields)
            writer.writeheader()
            writer.writerows(rows)
        stats["updated"] += 1
    return stats


def mark_csv_rows(fieldnames, rows):
    has_motion = "motion_state" in fieldnames
    has_strength = "touch_strength" in fieldnames
    has_label = "label_id" in fieldnames or "label_name" in fieldnames or "class_name" in fieldnames
    if not (has_motion and has_strength) and not has_label:
        return False, fieldnames

    output_fields = list(fieldnames)
    if "finger_type" not in output_fields:
        insert_index = output_fields.index("touch_strength") + 1 if "touch_strength" in output_fields else len(output_fields)
        output_fields.insert(insert_index, "finger_type")

    changed = "finger_type" not in fieldnames
    for row in rows:
        original_finger = normalize_value(row.get("finger_type"))
        if original_finger in UNKNOWN_VALUES:
            row["finger_type"] = TARGET_FINGER_TYPE
            changed = True
        for field in ("label_id", "label_name", "class_name"):
            if field in output_fields:
                new_value = append_finger_to_label(row.get(field, ""), row["finger_type"])
                if new_value != row.get(field):
                    row[field] = new_value
                    changed = True
    return changed, output_fields


def mark_summary_records(root):
    stats = {
        "scanned": 0,
        "updated": 0,
        "skipped": 0,
        "errors": 0,
    }
    for path in sorted(root.rglob("*summary.json")):
        stats["scanned"] += 1
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            stats["errors"] += 1
            continue
        if not isinstance(data, dict):
            stats["skipped"] += 1
            continue

        changed = False
        for key, value in list(data.items()):
            if not isinstance(value, dict):
                continue
            if not all(isinstance(item, (int, float)) for item in value.values()):
                continue
            updated = mark_summary_counter(value)
            if updated != value:
                data[key] = updated
                changed = True
        if add_finger_count_summary(data):
            changed = True

        if changed:
            write_json(path, data)
            stats["updated"] += 1
        else:
            stats["skipped"] += 1
    return stats


def mark_summary_counter(counter):
    updated = {}
    for key, value in counter.items():
        new_key = append_finger_to_compound_key(key)
        updated[new_key] = updated.get(new_key, 0) + value
    return updated


def append_finger_to_compound_key(key):
    label = normalize_value(key)
    if not label:
        return key
    if "_thumb" in label or "_index" in label or "::thumb" in label or "::index" in label:
        return label
    parts = label.split("::")
    if len(parts) >= 2:
        last = parts[-1]
        if last in ("light", "heavy"):
            return "::".join(parts + [TARGET_FINGER_TYPE])
        if last.endswith("_light") or last.endswith("_heavy"):
            parts[-1] = append_finger_to_label(last, TARGET_FINGER_TYPE)
            return "::".join(parts)
    if label.endswith("_light") or label.endswith("_heavy"):
        return append_finger_to_label(label, TARGET_FINGER_TYPE)
    return label


def add_finger_count_summary(data):
    sample_count = sample_count_from_summary(data)
    if sample_count <= 0:
        return False
    current = data.get("count_by_finger_type")
    expected = {TARGET_FINGER_TYPE: sample_count}
    if current == expected:
        return False
    data["count_by_finger_type"] = expected
    return True


def sample_count_from_summary(data):
    for key in ("total_raw_json_files", "total"):
        value = data.get(key)
        if isinstance(value, int):
            return value
    label_counts = data.get("labelCounts") or data.get("count_by_original_label") or data.get("count_by_label")
    if isinstance(label_counts, dict):
        total = 0
        for value in label_counts.values():
            if isinstance(value, int):
                total += value
        return total
    return 0


def append_finger_to_label(value, finger_type):
    label = normalize_value(value)
    if not label:
        return value
    suffix = "_" + finger_type
    if label.endswith("_thumb") or label.endswith("_index"):
        return label
    return label + suffix


def is_android_sample_record(data):
    if not isinstance(data, dict):
        return False
    if data.get("platform") != "android":
        return False
    label = data.get("label")
    if not isinstance(label, dict):
        return False
    return bool(label.get("motionState") and label.get("touchStrength"))


def normalize_value(value):
    if value is None:
        return ""
    return str(value).strip()


def write_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
