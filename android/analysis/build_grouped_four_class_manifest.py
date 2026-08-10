#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


QUALITY_FIELDS = [
    "quality_bucket",
    "quality_score",
    "quality_reasons",
    "quality_warnings",
    "sample_weight",
]


def main():
    args = parse_args()
    master_rows = read_rows(Path(args.master_manifest))
    quality_by_id = load_quality_rows([Path(value) for value in args.quality_manifest])
    rows, summary = transform_rows(
        master_rows,
        quality_by_id,
        holdout_collection=args.holdout_collection,
        block_seconds=args.block_seconds,
        min_sample_gap=args.min_sample_gap,
        unscored_collections=set(args.unscored_collection),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(output, rows)
    summary.update({
        "masterManifest": args.master_manifest,
        "qualityManifests": args.quality_manifest,
        "outputManifest": str(output),
        "holdoutCollection": args.holdout_collection,
        "blockSeconds": args.block_seconds,
        "minSampleGap": args.min_sample_gap,
    })
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-manifest", required=True)
    parser.add_argument("--quality-manifest", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--holdout-collection", required=True)
    parser.add_argument("--unscored-collection", action="append", default=[])
    parser.add_argument("--block-seconds", type=int, default=300)
    parser.add_argument("--min-sample-gap", type=float, default=0.25)
    return parser.parse_args()


def read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_quality_rows(paths):
    quality_by_id = {}
    for path in paths:
        for row in read_rows(path):
            sample_id = row.get("id", "")
            if sample_id:
                quality_by_id[sample_id] = row
    return quality_by_id


def transform_rows(
        master_rows,
        quality_by_id,
        holdout_collection,
        block_seconds,
        min_sample_gap,
        unscored_collections):
    unique_rows = []
    seen_ids = set()
    duplicate_ids = 0
    for row in master_rows:
        sample_id = row.get("id", "") or row.get("path", "")
        if sample_id in seen_ids:
            duplicate_ids += 1
            continue
        seen_ids.add(sample_id)
        unique_rows.append(dict(row))

    origins = collection_origins(unique_rows)
    candidates = []
    excluded_unscored = 0
    excluded_rejected = 0
    for row in unique_rows:
        sample_id = row.get("id", "")
        collection = row.get("collection_name", "")
        quality = quality_by_id.get(sample_id)
        if quality is None:
            if collection not in unscored_collections:
                excluded_unscored += 1
                continue
            quality = {
                "quality_bucket": "legacy_unscored",
                "quality_score": "100",
                "quality_reasons": "legacy_unscored",
                "quality_warnings": "",
                "sample_weight": "1.0",
            }
        if quality.get("quality_bucket") == "rejected":
            excluded_rejected += 1
            continue
        for field in QUALITY_FIELDS:
            row[field] = quality.get(field, "")
        row["split"] = "valid" if collection == holdout_collection else "train"
        row["session_group"] = session_group_for_row(row, origins, block_seconds)
        candidates.append(row)

    selected, thinned_rows = thin_temporal_bursts(candidates, min_sample_gap)
    selected.sort(key=lambda row: (
        row.get("split", ""),
        row.get("collection_name", ""),
        finite_float(row.get("created_at")),
        row.get("id", ""),
    ))
    summary = {
        "inputRows": len(master_rows),
        "outputRows": len(selected),
        "duplicateIds": duplicate_ids,
        "excludedUnscored": excluded_unscored,
        "excludedRejected": excluded_rejected,
        "thinnedRows": thinned_rows,
        "countsBySplit": dict(Counter(row["split"] for row in selected)),
        "countsByClass": dict(Counter(row.get("class_label", "") for row in selected)),
        "countsByCollection": dict(Counter(row.get("collection_name", "") for row in selected)),
        "countsByQuality": dict(Counter(row.get("quality_bucket", "") for row in selected)),
        "sessionGroupCount": len({row["session_group"] for row in selected}),
    }
    return selected, summary


def collection_origins(rows):
    origins = {}
    for row in rows:
        collection = row.get("collection_name", "")
        created_at = finite_float(row.get("created_at"))
        if collection not in origins or created_at < origins[collection]:
            origins[collection] = created_at
    return origins


def session_group_for_row(row, origins, block_seconds):
    collection = row.get("collection_name", "unknown")
    created_at = finite_float(row.get("created_at"))
    origin = origins.get(collection, created_at)
    block = int(math.floor(max(0.0, created_at - origin) / max(1, block_seconds)))
    return collection + ":" + str(block)


def thin_temporal_bursts(rows, min_sample_gap):
    if min_sample_gap <= 0:
        return list(rows), 0
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row.get("collection_name", ""),
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
            if previous_time is not None and current_time - previous_time < min_sample_gap:
                thinned += 1
                continue
            selected.append(row)
            previous_time = current_time
    return selected, thinned


def finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def write_rows(path, rows):
    if not rows:
        raise RuntimeError("empty_output_manifest")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
