#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


CLASS_ORDER = ["thumb_light", "thumb_heavy", "index_light", "index_heavy"]


def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    rows = []
    if args.include_legacy:
        rows.extend(read_legacy(root))
    rows.extend(read_clean(root, Path(args.le2110_manifest), "le2110_new", "train"))
    rows.extend(read_clean(root, Path(args.phk110_manifest), "phk110_new", "valid"))
    rows = deduplicate(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, rows)
    summary = {
        "total": len(rows),
        "splitCounts": dict(Counter(row["split"] for row in rows)),
        "deviceCounts": dict(Counter(row["device_group"] for row in rows)),
        "classCounts": dict(Counter(row["class_label"] for row in rows)),
        "splitClassCounts": dict(Counter(row["split"] + ":" + row["class_label"] for row in rows)),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--le2110-manifest", required=True)
    parser.add_argument("--phk110-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-legacy", action="store_true")
    return parser.parse_args()


def read_legacy(root):
    manifest = root / "data" / "manifests" / "full_four_class_manifest.csv"
    rows = []
    with manifest.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            source = row.get("source_person", "")
            raw_path = root / "data" / source / source / row["source_file_name"]
            rows.append(normalized_row(
                row,
                raw_path,
                "xiaomi_legacy" if source == "lemon" else "le2110_legacy",
                "train",
                row.get("class_label", ""),
                "1.0",
            ))
    return rows


def read_clean(root, manifest, device_group, split):
    path = manifest if manifest.is_absolute() else root / manifest
    rows = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            class_label = row.get("finger_type", "") + "_" + row.get("touch_strength", "")
            if class_label not in CLASS_ORDER:
                continue
            raw_path = Path(row["path"])
            if not raw_path.is_absolute():
                raw_path = root / raw_path
            rows.append(normalized_row(
                row,
                raw_path,
                device_group,
                split,
                class_label,
                row.get("sample_weight", "1.0"),
            ))
    return rows


def normalized_row(source, path, device_group, split, class_label, sample_weight):
    return {
        "id": source.get("id", ""),
        "path": str(path.resolve()),
        "file_name": source.get("file_name", source.get("source_file_name", "")),
        "finger_type": source.get("finger_type", ""),
        "touch_strength": source.get("touch_strength", ""),
        "motion_state": source.get("motion_state", ""),
        "class_label": class_label,
        "created_at": source.get("created_at", ""),
        "sample_weight": sample_weight,
        "device_group": device_group,
        "split": split,
    }


def deduplicate(rows):
    selected = {}
    for row in rows:
        key = row["id"] or row["path"]
        selected[key] = row
    return sorted(selected.values(), key=lambda row: (row["split"], row["device_group"], row["class_label"], row["path"]))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
