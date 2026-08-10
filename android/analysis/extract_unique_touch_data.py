#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import os
import re
import tarfile
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence


REPORT_JSON_NAMES = {"summary.json"}
REPORT_CSV_NAMES = {"manifest.csv", "duplicates.csv"}
LOCAL_TIMEZONE = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class Candidate:
    source: str
    file_name: str
    content: bytes


def slugify(value: object, fallback: str = "unknown") -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value.lower()).strip("-")
    return slug or fallback


def infer_collector(source: str) -> str:
    normalized = source.replace("\\", "/").lower()
    if "/collections/" in normalized:
        batch = normalized.split("/collections/", 1)[1].split("/", 1)[0]
        return slugify(batch.rsplit("_", 1)[-1])
    for collector in ("lemon", "tao", "zhuo"):
        if collector in normalized:
            return collector
    return "unknown"


def sample_date(created_at: object) -> str:
    try:
        timestamp = float(created_at)
        return datetime.fromtimestamp(timestamp, LOCAL_TIMEZONE).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError, OverflowError):
        return "unknown-date"


def parse_sample(candidate: Candidate) -> tuple[Optional[dict], Optional[str]]:
    try:
        payload = json.loads(candidate.content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "invalid_json"
    if not isinstance(payload, dict):
        return None, "invalid_json_object"
    sample_id = payload.get("id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        return None, "missing_id"
    return payload, None


def metadata_for(payload: dict, source: str) -> tuple[str, str, str]:
    collector_block = payload.get("collector")
    collector_value = None
    if isinstance(collector_block, dict):
        collector_value = collector_block.get("userName")
    collector = slugify(collector_value, infer_collector(source))

    device_block = payload.get("deviceCapability")
    device_value = None
    if isinstance(device_block, dict):
        device_value = device_block.get("deviceName")
    device = slugify(device_value)
    date_value = sample_date(payload.get("createdAt"))
    return collector, device, date_value


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def iter_directory_candidates(root: Path) -> Iterator[Candidate]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*.json"), key=lambda item: str(item).lower()):
        if path.name in REPORT_JSON_NAMES:
            continue
        yield Candidate(str(path), path.name, path.read_bytes())


def iter_destination_candidates(destination: Path) -> Iterator[Candidate]:
    paths = [
        path
        for path in destination.rglob("*.json")
        if path.name not in REPORT_JSON_NAMES
    ]
    paths.sort(
        key=lambda path: (
            len(str(path.relative_to(destination))),
            str(path.relative_to(destination)).lower(),
        )
    )
    for path in paths:
        yield Candidate(str(path), path.name, path.read_bytes())


def iter_archive_candidates(path: Path) -> Iterator[Candidate]:
    try:
        archive = tarfile.open(path, "r:*")
    except (tarfile.TarError, OSError):
        return
    with archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith(".json"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            source = f"{path}!{member.name}"
            yield Candidate(source, Path(member.name).name, extracted.read())


def discover_sources(data_root: Path, destination: Path) -> list[Path]:
    sources = []
    for relative_path in (Path("lemon/lemon"), Path("tao/tao")):
        path = data_root / relative_path
        if path.exists():
            sources.append(path)
    collections_root = data_root / "collections"
    if collections_root.exists():
        sources.extend(
            sorted(
                (path for path in collections_root.glob("*/raw") if path.is_dir()),
                key=lambda item: str(item).lower(),
            )
        )
    archive_paths = []
    for path in data_root.rglob("*"):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if lower_name.endswith((".tar", ".tar.gz", ".tgz")):
            try:
                path.relative_to(destination)
            except ValueError:
                archive_paths.append(path)
    sources.extend(sorted(archive_paths, key=lambda item: str(item).lower()))
    return sources


def iter_source_candidates(source: Path) -> Iterator[Candidate]:
    if source.is_dir():
        yield from iter_directory_candidates(source)
    elif source.is_file():
        yield from iter_archive_candidates(source)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(file_descriptor, "wb") as output_file:
            output_file.write(content)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def destination_path(
    destination: Path,
    collector: str,
    device: str,
    date_value: str,
    file_name: str,
    sample_id: str,
) -> Path:
    normalized_name = Path(file_name).name
    if not normalized_name.lower().endswith(".json"):
        normalized_name = f"{normalized_name}.json"
    candidate = destination / collector / device / date_value / normalized_name
    if not candidate.exists():
        return candidate
    return candidate.with_name(f"{candidate.stem}-{slugify(sample_id)}.json")


def write_csv(path: Path, field_names: Sequence[str], rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def write_json(path: Path, payload: dict) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    atomic_write_bytes(path, content)


def run_extraction(
    data_root: Path,
    destination: Path,
    sources: Optional[Iterable[Path]] = None,
) -> dict:
    data_root = Path(data_root).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    id_index: dict[str, tuple[str, str]] = {}
    hash_index: dict[str, tuple[str, str]] = {}
    manifest_rows: list[dict] = []
    duplicate_rows: list[dict] = []
    summary = {
        "archivesScanned": 0,
        "copied": 0,
        "directoriesScanned": 0,
        "discoveredSamples": 0,
        "duplicateHashes": 0,
        "duplicateIdAndHash": 0,
        "existing": 0,
        "idConflicts": 0,
        "invalidSamples": 0,
        "repairedDestinationDuplicates": 0,
        "retained": 0,
    }

    for candidate in iter_destination_candidates(destination):
        payload, error = parse_sample(candidate)
        if error or payload is None:
            continue
        sample_id = payload["id"].strip()
        content_hash = sha256_bytes(candidate.content)
        collector, device, date_value = metadata_for(payload, candidate.source)
        relative_destination = str(Path(candidate.source).resolve().relative_to(destination))
        retained_by_id = id_index.get(sample_id)
        if retained_by_id is not None:
            if retained_by_id[0] != content_hash:
                raise ValueError(
                    f"conflicting sample already exists in destination: {sample_id}"
                )
            Path(candidate.source).unlink()
            summary["repairedDestinationDuplicates"] += 1
            continue
        retained_by_hash = hash_index.get(content_hash)
        if retained_by_hash is not None:
            Path(candidate.source).unlink()
            summary["repairedDestinationDuplicates"] += 1
            continue
        id_index[sample_id] = (content_hash, relative_destination)
        hash_index[content_hash] = (sample_id, relative_destination)
        manifest_rows.append(
            {
                "id": sample_id,
                "sha256": content_hash,
                "collector": collector,
                "device": device,
                "date": date_value,
                "source": "existing_destination",
                "destination": relative_destination,
            }
        )
        summary["existing"] += 1

    if sources is None:
        selected_sources = discover_sources(data_root, destination)
    else:
        selected_sources = sorted(
            (Path(path) for path in sources), key=lambda item: str(item).lower()
        )
    selected_sources = [Path(path).resolve() for path in selected_sources]

    for source in selected_sources:
        if source.is_dir():
            summary["directoriesScanned"] += 1
        elif source.is_file():
            summary["archivesScanned"] += 1
        for candidate in iter_source_candidates(source):
            summary["discoveredSamples"] += 1
            payload, error = parse_sample(candidate)
            if error or payload is None:
                summary["invalidSamples"] += 1
                duplicate_rows.append(
                    {
                        "id": "",
                        "sha256": sha256_bytes(candidate.content),
                        "reason": error,
                        "source": candidate.source,
                        "retained_destination": "",
                    }
                )
                continue

            sample_id = payload["id"].strip()
            content_hash = sha256_bytes(candidate.content)
            retained_by_id = id_index.get(sample_id)
            if retained_by_id is not None:
                if retained_by_id[0] == content_hash:
                    reason = "duplicate_id_and_hash"
                    summary["duplicateIdAndHash"] += 1
                else:
                    reason = "id_conflict"
                    summary["idConflicts"] += 1
                duplicate_rows.append(
                    {
                        "id": sample_id,
                        "sha256": content_hash,
                        "reason": reason,
                        "source": candidate.source,
                        "retained_destination": retained_by_id[1],
                    }
                )
                continue

            retained_by_hash = hash_index.get(content_hash)
            if retained_by_hash is not None:
                summary["duplicateHashes"] += 1
                duplicate_rows.append(
                    {
                        "id": sample_id,
                        "sha256": content_hash,
                        "reason": "duplicate_hash",
                        "source": candidate.source,
                        "retained_destination": retained_by_hash[1],
                    }
                )
                continue

            collector, device, date_value = metadata_for(payload, candidate.source)
            output_path = destination_path(
                destination,
                collector,
                device,
                date_value,
                candidate.file_name,
                sample_id,
            )
            atomic_write_bytes(output_path, candidate.content)
            relative_destination = str(output_path.relative_to(destination))
            id_index[sample_id] = (content_hash, relative_destination)
            hash_index[content_hash] = (sample_id, relative_destination)
            manifest_rows.append(
                {
                    "id": sample_id,
                    "sha256": content_hash,
                    "collector": collector,
                    "device": device,
                    "date": date_value,
                    "source": candidate.source,
                    "destination": relative_destination,
                }
            )
            summary["copied"] += 1

    manifest_rows.sort(key=lambda row: (row["id"], row["destination"]))
    duplicate_rows.sort(key=lambda row: (row["reason"] or "", row["id"] or "", row["source"]))
    summary["retained"] = len(manifest_rows)

    write_csv(
        destination / "manifest.csv",
        ("id", "sha256", "collector", "device", "date", "source", "destination"),
        manifest_rows,
    )
    write_csv(
        destination / "duplicates.csv",
        ("id", "sha256", "reason", "source", "retained_destination"),
        duplicate_rows,
    )
    write_json(destination / "summary.json", summary)
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract unique raw touch JSON samples.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--source", action="append", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    data_root = arguments.data_root.resolve()
    destination = (
        arguments.destination.resolve()
        if arguments.destination is not None
        else data_root / "deduplicated_raw"
    )
    summary = run_extraction(data_root, destination, arguments.source)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
