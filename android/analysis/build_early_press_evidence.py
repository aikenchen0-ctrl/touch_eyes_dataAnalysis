#!/usr/bin/env python3
"""Build a sidecar of causal early-press evidence from raw schema v6 files."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from . import early_press_evidence
except ImportError:
    import early_press_evidence


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_index(raw_root):
    """Index basenames once so large datasets do not rescan the tree."""
    if raw_root is None:
        return {}
    result = {}
    for path in Path(raw_root).resolve().rglob("*.json"):
        result.setdefault(path.name, []).append(path)
    return result


def source_manifest_index(manifest):
    """Read absolute source paths from an existing evidence sidecar.

    The old sidecar is a provenance manifest. It lets later feature contracts
    reuse verified source paths even when the raw files are split across
    several roots. A hash may map to only one path; conflicting mappings are
    rejected instead of silently choosing an arbitrary source.
    """
    if manifest is None:
        return {}
    manifest = Path(manifest)
    if not manifest.is_file():
        raise RuntimeError("source_manifest_not_found")
    result = {}
    with manifest.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_sha256 = str(row.get("sourceSha256") or "")
            source_file = str(row.get("sourceFile") or "")
            if not source_sha256 or not source_file:
                raise RuntimeError(
                    f"source_manifest_incomplete:{line_number}"
                )
            path = str(Path(source_file).resolve())
            previous = result.get(source_sha256)
            if previous is not None and previous != path:
                raise RuntimeError(
                    f"source_manifest_conflict:{source_sha256}"
                )
            result[source_sha256] = path
    if not result:
        raise RuntimeError("source_manifest_empty")
    return result


def resolve_source(
        raw_root,
        source_file,
        basename_index=None,
        source_manifest=None):
    if source_manifest is not None:
        return Path(source_manifest)
    if raw_root is None:
        return None
    root = Path(raw_root).resolve()
    relative = Path(str(source_file))
    candidates = [root / relative]
    if relative != Path(relative.name):
        candidates.append(root / relative.name)
    # 中文注释：部分旧数据集只保留扁平文件名。
    # 仅在唯一匹配时回退，避免同名文件静默绑定到错误原始记录。
    matches = list((basename_index or {}).get(relative.name, []))
    if len(matches) == 1:
        candidates.append(matches[0])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def build_one(
        row,
        raw_root,
        basename_index=None,
        source_manifest=None):
    source_sha256 = str(row.get("sourceSha256") or "")
    if not source_sha256:
        raise RuntimeError("training_row_missing_source_sha256")
    manifest_path = None
    if source_manifest is not None:
        manifest_path = source_manifest.get(source_sha256)
        if manifest_path is None:
            raise RuntimeError("raw_source_not_in_manifest")
    path = resolve_source(
        raw_root,
        row.get("sourceFile") or "",
        basename_index,
        manifest_path,
    )
    if path is None:
        raise RuntimeError("raw_source_not_found")
    if sha256_file(path) != source_sha256:
        raise RuntimeError("raw_source_hash_mismatch")
    data = json.loads(path.read_text(encoding="utf-8"))
    # 中文注释：原始记录可能直接包含帧，也可能放在 records 容器中。
    record = data
    records = data.get("records")
    if isinstance(records, dict):
        if len(records) != 1:
            raise RuntimeError("ambiguous_raw_record_container")
        record = next(iter(records.values()))
    evidence = early_press_evidence.build_evidence(data, record)
    early_press_evidence.validate([evidence["values"]])
    result = {
        "sourceSha256": source_sha256,
        "sampleId": str(row.get("sampleId") or ""),
        "earlyEvidenceContract": evidence["contract"],
        "offsetsMs": evidence["offsetsMs"],
        "featureNames": evidence["featureNames"],
        "earlyEvidence": evidence["values"],
        "sourceFile": str(path),
    }
    if evidence.get("reason"):
        result["unavailableReason"] = evidence["reason"]
    return result


def read_rows(dataset_dir):
    path = Path(dataset_dir) / "training_touch26.jsonl"
    if not path.is_file():
        raise RuntimeError("missing_training_touch26")
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build(
        dataset_dir,
        raw_root,
        output,
        workers,
        source_manifest_path=None):
    rows = read_rows(dataset_dir)
    if not rows:
        raise RuntimeError("empty_training_dataset")
    basename_index = source_index(raw_root)
    manifest_index = source_manifest_index(source_manifest_path)

    def safe_build(row):
        try:
            return build_one(
                row,
                raw_root,
                basename_index,
                manifest_index or None,
            )
        except RuntimeError as error:
            reason = str(error)
            # 中文注释：数据缺失只生成带可用性掩码的 sidecar；
            # 哈希错误或记录歧义属于数据合同破坏，必须中止整批构建。
            unavailable_prefixes = (
                "insufficient_imu_frames_for_early_evidence",
                "missing_down_event_timestamp",
                "missing_time_alignment",
                "time_grid_missing_",
            )
            if not reason.startswith(unavailable_prefixes):
                raise
            return {
                "sourceSha256": str(row.get("sourceSha256") or ""),
                "sampleId": str(row.get("sampleId") or ""),
                "earlyEvidenceContract": early_press_evidence.CONTRACT,
                "offsetsMs": list(early_press_evidence.OFFSETS_MS),
                "featureNames": list(early_press_evidence.FEATURE_NAMES),
                "earlyEvidence": early_press_evidence.unavailable()["values"],
                "sourceFile": str(row.get("sourceFile") or ""),
                "unavailableReason": reason,
            }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(safe_build, rows))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for result in results:
            target.write(json.dumps(result, ensure_ascii=False) + "\n")
    temporary.replace(output)
    available = sum(
        float(result["earlyEvidence"][-1]) >= 0.5
        for result in results
    )
    unavailable_reasons = {}
    for result in results:
        reason = result.get("unavailableReason")
        if reason:
            unavailable_reasons[reason] = (
                unavailable_reasons.get(reason, 0) + 1
            )
    return {
        "datasetDir": str(Path(dataset_dir).resolve()),
        "rawRoot": (
            str(Path(raw_root).resolve())
            if raw_root is not None
            else None
        ),
        "sourceManifest": (
            str(Path(source_manifest_path).resolve())
            if source_manifest_path is not None
            else None
        ),
        "output": str(output.resolve()),
        "rowCount": len(results),
        "availableCount": available,
        "unavailableCount": len(results) - available,
        "unavailableReasons": unavailable_reasons,
        "contract": early_press_evidence.CONTRACT,
        "featureNames": list(early_press_evidence.FEATURE_NAMES),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--raw-root")
    parser.add_argument("--source-manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not args.raw_root and not args.source_manifest:
        raise RuntimeError("raw_root_or_source_manifest_required")
    if args.workers <= 0:
        raise RuntimeError("invalid_worker_count")
    report = build(
        args.dataset_dir,
        args.raw_root,
        args.output,
        args.workers,
        args.source_manifest,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
