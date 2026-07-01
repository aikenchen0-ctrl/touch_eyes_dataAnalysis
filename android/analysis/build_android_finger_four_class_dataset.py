#!/usr/bin/env python3
import argparse
import csv
import json
import random
from collections import Counter
from collections import defaultdict
from pathlib import Path


CLASS_ORDER = [
    "thumb_light",
    "thumb_heavy",
    "index_light",
    "index_heavy",
]


def main():
    args = parse_args()
    phone_raw_dir = Path(args.phone_raw_dir) if args.phone_raw_dir else None
    thumb_reference_dir = Path(args.thumb_reference_dir) if args.thumb_reference_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "balanced_json"
    if args.copy_files:
        sample_dir.mkdir(parents=True, exist_ok=True)

    thumb_file_names = set()
    if thumb_reference_dir:
        thumb_file_names = {path.name for path in thumb_reference_dir.glob("*.json")}
    if args.source_manifest:
        rows_by_class, candidate_rows = collect_manifest_rows(
            Path(args.source_manifest),
            thumb_file_names,
            args.motion_state,
            parse_csv_list(args.quality_buckets),
        )
    elif phone_raw_dir:
        rows_by_class, candidate_rows = collect_rows(phone_raw_dir, thumb_file_names, args.motion_state)
    else:
        raise RuntimeError("missing_source")
    per_class_count = args.per_class if args.per_class > 0 else min(len(rows_by_class[class_name]) for class_name in CLASS_ORDER)
    balanced_rows = select_balanced_rows(rows_by_class, per_class_count, args.seed)
    if args.copy_files:
        copy_balanced_files(balanced_rows, sample_dir)
        for row in balanced_rows:
            row["path"] = str(sample_dir / row["file_name"])

    write_csv(output_dir / "all_candidates_manifest.csv", candidate_rows)
    write_csv(output_dir / "balanced_four_class_manifest.csv", balanced_rows)
    write_json(output_dir / "four_class_summary.json", {
        "phoneRawDirectory": str(phone_raw_dir) if phone_raw_dir else "",
        "sourceCleanManifest": args.source_manifest,
        "thumbReferenceDirectory": str(thumb_reference_dir) if thumb_reference_dir else "",
        "qualityBuckets": parse_csv_list(args.quality_buckets),
        "motionState": args.motion_state,
        "classOrder": CLASS_ORDER,
        "candidateCounts": dict(Counter(row["class_label"] for row in candidate_rows)),
        "balancedBucketCounts": dict(Counter(row.get("quality_bucket", "") for row in balanced_rows if row.get("quality_bucket", ""))),
        "balancedCounts": dict(Counter(row["class_label"] for row in balanced_rows)),
        "perClassCount": per_class_count,
        "totalBalanced": len(balanced_rows),
        "seed": args.seed,
        "copyFiles": args.copy_files,
        "sampleDirectory": str(sample_dir) if args.copy_files else "",
    })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phone-raw-dir", default="")
    parser.add_argument("--thumb-reference-dir", default="")
    parser.add_argument("--source-manifest", default="")
    parser.add_argument("--quality-buckets", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--motion-state", default="handheld_static")
    parser.add_argument("--per-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--copy-files", action="store_true")
    return parser.parse_args()


def parse_csv_list(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def collect_rows(phone_raw_dir, thumb_file_names, motion_state):
    rows_by_class = defaultdict(list)
    candidate_rows = []
    for path in sorted(phone_raw_dir.glob("*.json")):
        data = read_json(path)
        label = data.get("label") or {}
        if label.get("motionState") != motion_state:
            continue
        strength = label.get("touchStrength")
        if strength not in ("light", "heavy"):
            continue
        finger_type = finger_type_for_sample(path.name, data, thumb_file_names)
        if finger_type not in ("thumb", "index"):
            continue
        class_label = finger_type + "_" + strength
        if class_label not in CLASS_ORDER:
            continue
        row = make_row(path, data, finger_type, class_label, {})
        rows_by_class[class_label].append(row)
        candidate_rows.append(row)
    for class_name in CLASS_ORDER:
        if not rows_by_class[class_name]:
            raise RuntimeError("missing_class:" + class_name)
    return rows_by_class, sorted(candidate_rows, key=lambda row: row["source_file_name"])


def collect_manifest_rows(manifest_path, thumb_file_names, motion_state, quality_buckets):
    rows_by_class = defaultdict(list)
    candidate_rows = []
    allowed_buckets = set(quality_buckets)
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        for manifest_row in csv.DictReader(file):
            if allowed_buckets and manifest_row.get("quality_bucket", "") not in allowed_buckets:
                continue
            if manifest_row.get("motion_state", "") != motion_state:
                continue
            strength = manifest_row.get("touch_strength", "")
            if strength not in ("light", "heavy"):
                continue
            path = resolve_manifest_path(manifest_row, manifest_path)
            if path is None:
                continue
            data = read_json(path)
            finger_type = manifest_row.get("finger_type", "")
            if finger_type not in ("thumb", "index"):
                finger_type = finger_type_for_sample(path.name, data, thumb_file_names)
            if finger_type not in ("thumb", "index"):
                continue
            class_label = finger_type + "_" + strength
            if class_label not in CLASS_ORDER:
                continue
            row = make_row(path, data, finger_type, class_label, manifest_row)
            rows_by_class[class_label].append(row)
            candidate_rows.append(row)
    for class_name in CLASS_ORDER:
        if not rows_by_class[class_name]:
            raise RuntimeError("missing_class:" + class_name)
    return rows_by_class, sorted(candidate_rows, key=lambda row: row["source_file_name"])


def resolve_manifest_path(row, manifest_path):
    value = row.get("path", "")
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate
    for root in [Path.cwd()] + list(manifest_path.resolve().parents):
        resolved = root / candidate
        if resolved.exists():
            return resolved
    return None


def finger_type_for_sample(file_name, data, thumb_file_names):
    # 中文注释：新版样本已写入 fingerType，旧样本才回退到文件名或参考目录推断。
    label = data.get("label") or {}
    finger_type = str(label.get("fingerType") or "")
    if finger_type in ("thumb", "index"):
        return finger_type
    if "_thumb_" in file_name:
        return "thumb"
    if "_index_" in file_name:
        return "index"
    if file_name in thumb_file_names:
        return "thumb"
    return "index"


def make_row(path, data, finger_type, class_label, manifest_row):
    label = data.get("label") or {}
    output_name = normalized_file_name(path.name, data, finger_type)
    row = {
        "id": str(data.get("id") or ""),
        "source_file_name": path.name,
        "file_name": output_name,
        "path": str(path),
        "finger_type": finger_type,
        "touch_strength": str(label.get("touchStrength") or ""),
        "motion_state": str(label.get("motionState") or ""),
        "class_label": class_label,
        "created_at": data.get("createdAt", ""),
        "source_touch_kind": str(data.get("sourceTouchKind") or ""),
        "sample_rate_hz": data.get("sampleRateHz", ""),
        "max_abs_delta": data.get("maxAbsDelta", ""),
    }
    for key in ("quality_bucket", "quality_score", "quality_reasons", "sample_weight"):
        if key in manifest_row:
            row[key] = manifest_row.get(key, "")
    return row


def select_balanced_rows(rows_by_class, per_class_count, seed):
    rng = random.Random(seed)
    selected = []
    for class_name in CLASS_ORDER:
        items = list(rows_by_class[class_name])
        rng.shuffle(items)
        chosen = sorted(items[:per_class_count], key=lambda row: row["source_file_name"])
        selected.extend(chosen)
    return sorted(selected, key=lambda row: (row["class_label"], row["source_file_name"]))


def copy_balanced_files(rows, sample_dir):
    for row in rows:
        source = Path(row["path"])
        data = read_json(source)
        label = data.setdefault("label", {})
        label["fingerType"] = row["finger_type"]
        destination = sample_dir / row["file_name"]
        if destination.exists():
            destination.unlink()
        with destination.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")


def normalized_file_name(file_name, data, finger_type):
    if "_thumb_" in file_name or "_index_" in file_name:
        return file_name
    label = data.get("label") or {}
    created_at = int(round(float(data.get("createdAt", 0)) * 1000))
    motion_state = str(label.get("motionState") or "motion")
    strength = str(label.get("touchStrength") or "strength")
    sample_id = str(data.get("id") or Path(file_name).stem)
    return f"{created_at}_{motion_state}_{strength}_{finger_type}_{sample_id}.json"


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


if __name__ == "__main__":
    main()
