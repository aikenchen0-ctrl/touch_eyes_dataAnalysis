#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


CLASS_ORDER = ("thumb_light", "thumb_heavy", "index_light", "index_heavy")


def finite_float(value):
    try:
        output = float(value)
    except (TypeError, ValueError):
        return 0.0
    return output if math.isfinite(output) else 0.0


def safe_slug(value):
    return re.sub(r"[^a-z0-9]+", "-", str(value or "unknown").lower()).strip("-") or "unknown"


def assign_sessions(rows, gap_seconds=60.0):
    output = [dict(row) for row in rows]
    legacy_by_collection = defaultdict(list)
    for row in output:
        collection = row.get("collection_name", "unknown")
        raw_session = row.get("raw_session_id", "")
        if raw_session:
            row["session_group"] = collection + ":raw:" + safe_slug(raw_session)
        else:
            legacy_by_collection[collection].append(row)

    for collection, items in legacy_by_collection.items():
        session_index = 0
        previous_time = None
        for row in sorted(items, key=lambda item: (finite_float(item.get("created_at")), item.get("id", ""))):
            current_time = finite_float(row.get("created_at"))
            if previous_time is not None and current_time - previous_time > gap_seconds:
                session_index += 1
            row["session_group"] = collection + ":legacy:" + str(session_index)
            previous_time = current_time
    return output


def thin_temporal(rows, min_gap=0.25):
    if min_gap <= 0:
        return list(rows), 0
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row.get("session_group", ""),
            row.get("class_label", ""),
            row.get("motion_state", ""),
        )
        grouped[key].append(row)

    selected = []
    thinned = 0
    for items in grouped.values():
        previous_time = None
        for row in sorted(items, key=lambda item: (finite_float(item.get("created_at")), item.get("id", ""))):
            current_time = finite_float(row.get("created_at"))
            if previous_time is not None and current_time - previous_time < min_gap:
                thinned += 1
                continue
            selected.append(dict(row))
            previous_time = current_time
    selected.sort(key=lambda row: (finite_float(row.get("created_at")), row.get("id", "")))
    return selected, thinned


def class_counts(rows):
    return Counter(row.get("class_label", "") for row in rows)


def has_class_coverage(rows, minimum_per_class):
    counts = class_counts(rows)
    return all(counts[class_label] >= minimum_per_class for class_label in CLASS_ORDER)


def split_for_value(rows, field, value):
    output = []
    for row in rows:
        item = dict(row)
        item["split"] = "valid" if item.get(field, "") == value else "train"
        output.append(item)
    return output


def build_domain_folds(rows, minimum_per_class=100):
    folds = {}
    for domain in sorted({row.get("domain_group", "") for row in rows if row.get("domain_group")}):
        fold_rows = split_for_value(rows, "domain_group", domain)
        train = [row for row in fold_rows if row["split"] == "train"]
        valid = [row for row in fold_rows if row["split"] == "valid"]
        if not has_class_coverage(train, minimum_per_class) or not has_class_coverage(valid, minimum_per_class):
            continue
        validate_no_session_leakage(fold_rows)
        folds["domain_" + safe_slug(domain)] = fold_rows
    return folds


def build_collection_folds(rows, minimum_per_class=100):
    folds = {}
    collections = sorted({row.get("collection_name", "") for row in rows if row.get("collection_name")})
    for collection in collections:
        fold_rows = split_for_value(rows, "collection_name", collection)
        train = [row for row in fold_rows if row["split"] == "train"]
        valid = [row for row in fold_rows if row["split"] == "valid"]
        if not has_class_coverage(train, minimum_per_class) or not has_class_coverage(valid, minimum_per_class):
            continue
        validate_no_session_leakage(fold_rows)
        folds["collection_" + safe_slug(collection)] = fold_rows
    return folds


def build_temporal_tail_fold(rows, validation_fraction=0.20):
    sessions_by_domain_class = defaultdict(set)
    session_start = {}
    for row in rows:
        session = row.get("session_group", "")
        domain = row.get("domain_group", "unknown")
        class_label = row.get("class_label", "")
        sessions_by_domain_class[(domain, class_label)].add(session)
        created_at = finite_float(row.get("created_at"))
        if session not in session_start or created_at < session_start[session]:
            session_start[session] = created_at

    valid_sessions = set()
    for sessions in sessions_by_domain_class.values():
        ordered = sorted(sessions, key=lambda session: (session_start.get(session, 0.0), session))
        if len(ordered) < 2:
            continue
        valid_count = max(1, int(math.ceil(len(ordered) * validation_fraction)))
        valid_count = min(valid_count, len(ordered) - 1)
        valid_sessions.update(ordered[-valid_count:])

    output = []
    for row in rows:
        item = dict(row)
        item["split"] = "valid" if item.get("session_group") in valid_sessions else "train"
        output.append(item)
    validate_no_session_leakage(output)
    if not has_class_coverage([row for row in output if row["split"] == "train"], 1):
        raise RuntimeError("temporal_train_missing_class")
    if not has_class_coverage([row for row in output if row["split"] == "valid"], 1):
        raise RuntimeError("temporal_valid_missing_class")
    return output


def validate_no_session_leakage(rows):
    splits_by_session = defaultdict(set)
    for row in rows:
        split = row.get("split", "")
        if split:
            splits_by_session[row.get("session_group", "")].add(split)
    leaking = [session for session, splits in splits_by_session.items() if len(splits) > 1]
    if leaking:
        raise RuntimeError("cross_split_session_group")


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("empty_manifest")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fold_summary(rows):
    sessions_by_split = defaultdict(set)
    for row in rows:
        sessions_by_split[row.get("split", "")].add(row.get("session_group", ""))
    return {
        "rows": len(rows),
        "splitCounts": dict(sorted(Counter(row.get("split", "") for row in rows).items())),
        "splitClassCounts": dict(
            sorted(Counter(row.get("split", "") + ":" + row.get("class_label", "") for row in rows).items())
        ),
        "sessionCounts": {
            split: len(sessions) for split, sessions in sorted(sessions_by_split.items())
        },
    }


def build_folds(input_manifest, output_dir, gap_seconds=60.0, min_gap=0.25, minimum_per_class=100):
    rows = read_csv(input_manifest)
    assigned = assign_sessions(rows, gap_seconds=gap_seconds)
    thinned, thinned_count = thin_temporal(assigned, min_gap=min_gap)
    output_dir = Path(output_dir)
    write_csv(output_dir / "training_manifest.csv", thinned)

    folds = {}
    folds.update(build_domain_folds(thinned, minimum_per_class=minimum_per_class))
    folds.update(build_collection_folds(thinned, minimum_per_class=minimum_per_class))
    folds["temporal_tail"] = build_temporal_tail_fold(thinned)
    fold_summaries = {}
    for name, fold_rows in folds.items():
        validate_no_session_leakage(fold_rows)
        write_csv(output_dir / "folds" / (name + ".csv"), fold_rows)
        fold_summaries[name] = fold_summary(fold_rows)

    summary = {
        "inputRows": len(rows),
        "trainingRows": len(thinned),
        "thinnedRows": thinned_count,
        "sessionGroups": len({row["session_group"] for row in thinned}),
        "domainFolds": len([name for name in folds if name.startswith("domain_")]),
        "collectionFolds": len([name for name in folds if name.startswith("collection_")]),
        "folds": fold_summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fold_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Build strict Clean V3 folds.")
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gap-seconds", type=float, default=60.0)
    parser.add_argument("--min-gap", type=float, default=0.25)
    parser.add_argument("--minimum-per-class", type=int, default=100)
    return parser.parse_args()


def main():
    arguments = parse_args()
    summary = build_folds(
        arguments.input_manifest,
        arguments.output_dir,
        gap_seconds=arguments.gap_seconds,
        min_gap=arguments.min_gap,
        minimum_per_class=arguments.minimum_per_class,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
