# Deduplicated Raw Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an auditable extractor that creates one unified raw touch dataset with no duplicate sample IDs or file hashes.

**Architecture:** A focused Python command-line tool discovers expanded JSON files and tar archive members, normalizes each record into a common candidate representation, and passes candidates through a destination-aware deduplication index. It copies only unique samples and emits manifest, duplicate, and summary reports that can be independently verified.

**Tech Stack:** Python 3 standard library, `unittest`, JSON, CSV, `tarfile`, SHA-256.

---

### Task 1: Specify deduplication behavior

**Files:**
- Create: `android/analysis/tests/test_extract_unique_touch_data.py`

- [ ] Write a test that creates directory and tar sources containing unique samples, duplicate IDs, duplicate hashes, and an ID conflict.
- [ ] Assert that only unique samples are copied and every skipped record receives the correct audit reason.
- [ ] Write a rerun test proving existing destination samples are indexed and not copied again.
- [ ] Run `python -m unittest android.analysis.tests.test_extract_unique_touch_data -v` and confirm failure because the extractor module does not exist.

### Task 2: Implement the extractor

**Files:**
- Create: `android/analysis/extract_unique_touch_data.py`

- [ ] Implement deterministic source discovery for expanded directories and `.tar` or `.tar.gz` archives.
- [ ] Implement JSON validation, metadata extraction, path slugging, ID and SHA-256 indexing, and atomic writes.
- [ ] Implement `manifest.csv`, `duplicates.csv`, and `summary.json` generation.
- [ ] Add CLI options for `--data-root`, `--destination`, and optional explicit `--source` paths.
- [ ] Run the focused unit test and confirm all cases pass.

### Task 3: Extract the complete local dataset

**Files:**
- Create: `data/deduplicated_raw/**/*.json`
- Create: `data/deduplicated_raw/manifest.csv`
- Create: `data/deduplicated_raw/duplicates.csv`
- Create: `data/deduplicated_raw/summary.json`

- [ ] Run the extractor against the repository `data/` directory.
- [ ] Confirm archives and expanded directories are both represented in discovery totals.
- [ ] Confirm duplicate and conflict records are skipped without overwriting retained samples.

### Task 4: Verify and document

**Files:**
- Modify: `FunctionAndDataStructureDescription.md`

- [ ] Independently scan every destination JSON and assert unique IDs and hashes.
- [ ] Compare destination file count with the manifest and summary retained count.
- [ ] Rerun the extractor and confirm it copies zero additional samples.
- [ ] Document the extractor, destination layout, reports, and final extraction totals.
- [ ] Run the complete relevant Python test suite.
