# Deduplicated Raw Extraction Design

## Goal

Create a unified `data/deduplicated_raw/` dataset containing every unique raw touch JSON sample currently available under `data/`, without duplicating samples already present in the destination or repeated across expanded directories and archives.

## Sources

The extractor scans these inputs in deterministic order:

1. Existing JSON files already stored in `data/deduplicated_raw/`.
2. Expanded legacy datasets under `data/lemon/lemon/` and `data/tao/tao/`.
3. Expanded collection datasets under `data/collections/*/raw/`.
4. `.tar` and `.tar.gz` archives under `data/`.

The destination directory is excluded from ordinary source discovery so rerunning the tool does not recursively ingest its own reports.

## Deduplication Rules

Each valid JSON sample is identified by:

- Primary key: top-level sample `id`.
- Secondary key: SHA-256 of the original JSON bytes.

The first deterministic occurrence is retained. Later records are skipped when either key is already present. If the same `id` appears with different bytes, the later record is classified as an ID conflict and is not copied. Invalid JSON and JSON objects without a sample `id` are not copied.

## Destination Layout

Samples are written to:

```text
data/deduplicated_raw/<collector>/<device>/<date>/<original-file-name>.json
```

Metadata comes from `collector.userName`, `deviceCapability.deviceName`, and `createdAt`. Missing values fall back to source-based collector names or `unknown`. Path components are normalized to lowercase ASCII-safe slugs.

## Audit Outputs

The extractor writes:

- `manifest.csv`: one row for every retained sample, including ID, hash, source, destination, collector, device, and date.
- `duplicates.csv`: every skipped duplicate or conflict with its reason and retained destination.
- `summary.json`: totals for discovered, copied, existing, duplicate, conflict, and invalid records.

Reports are replaced only after a successful run. Sample files are written through temporary files and atomically renamed.

## Verification

Automated tests cover duplicate IDs, duplicate hashes, ID conflicts, archive extraction, metadata fallback, and reruns. Full-data verification independently scans the destination and asserts that every retained JSON has a unique ID and SHA-256 hash.
