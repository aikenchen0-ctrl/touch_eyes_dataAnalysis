# Historical Down-Anchor Datasets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build separate A and B manifests from the immutable historical JSON set, where A contains samples whose touch-down time lies inside observed IMU coverage and B contains samples with a complete observed down-centered `-5...+9` window.

**Architecture:** A focused Python module streams the deduplicated manifest, distinguishes observed sensor frames from zero placeholders, analyzes touch-down coverage, and writes deterministic audit artifacts. Dataset A stores references and diagnostics only; dataset B additionally writes reindexed 15-frame derived JSON files while preserving source provenance.

**Tech Stack:** Python 3 standard library (`argparse`, `csv`, `json`, `pathlib`, `tempfile`, `unittest`), existing JSON/CSV dataset contracts.

---

## File Structure

- Create `android/analysis/build_historical_down_anchor_datasets.py`: frame validity, down-location analysis, streaming dataset construction, atomic output replacement, and CLI.
- Create `android/analysis/tests/test_build_historical_down_anchor_datasets.py`: focused unit and temporary-directory integration tests.
- Modify `FunctionAndDataStructureDescription.md`: A/B definitions, output schema, command, and change log.

### Task 1: Frame and sample analysis

**Files:**
- Create: `android/analysis/build_historical_down_anchor_datasets.py`
- Test: `android/analysis/tests/test_build_historical_down_anchor_datasets.py`

- [ ] **Step 1: Write failing placeholder and A/B classification tests**

```python
import unittest

from android.analysis.build_historical_down_anchor_datasets import analyze_sample
from android.analysis.build_historical_down_anchor_datasets import is_observed_frame


class DownAnchorAnalysisTests(unittest.TestCase):
    def test_zero_placeholder_is_not_observed(self):
        frame = {
            "delta": 0,
            "x": 0,
            "y": 0,
            "z": 0,
            "rotationRateX": 0,
            "rotationRateY": 0,
            "rotationRateZ": 0,
            "gravityX": 0,
            "gravityY": 0,
            "gravityZ": 0,
            "x_norm": 0,
            "y_norm": 0,
        }
        self.assertFalse(is_observed_frame(frame))

    def test_inside_window_without_full_context_is_a_only(self):
        data = sample_with_times([-0.010, -0.005, 0.000, 0.005, 0.010])
        result = analyze_sample(data)
        self.assertTrue(result["eligible_a"])
        self.assertFalse(result["eligible_b"])

    def test_complete_continuous_window_is_a_and_b(self):
        data = sample_with_times([index * 0.005 for index in range(-5, 10)])
        result = analyze_sample(data)
        self.assertTrue(result["eligible_a"])
        self.assertTrue(result["eligible_b"])
        self.assertEqual(list(range(-5, 10)), [frame["frameIndex"] for frame in result["down_frames"]])
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m unittest android.analysis.tests.test_build_historical_down_anchor_datasets -v
```

Expected: import failure because `build_historical_down_anchor_datasets.py` does not exist.

- [ ] **Step 3: Implement minimal analysis API**

Implement these public functions:

```python
def is_observed_frame(frame):
    return any(abs(finite_float(frame.get(name), 0.0)) > 1e-12 for name in SIGNAL_FIELDS)


def touch_down_timestamp(data):
    touch_frames = data.get("touchFrames") or []
    if touch_frames:
        return finite_float(touch_frames[0].get("eventTimeMillis"), None) / 1000.0
    touch_events = data.get("touchEvents") or []
    if touch_events:
        return finite_float(touch_events[0].get("timestamp"), None)
    return None


def analyze_sample(data):
    down_timestamp = touch_down_timestamp(data)
    anchor_timestamp = finite_float(data.get("anchorTimestamp"), None)
    sample_rate_hz = finite_float(data.get("sampleRateHz"), None)
    frames = [frame for frame in data.get("frames") or [] if is_observed_frame(frame)]
    return analyze_observed_frames(frames, down_timestamp, anchor_timestamp, sample_rate_hz)
```

The implementation must sort observed frames by absolute time, require the down timestamp to be bracketed for A, select the nearest observed frame as zero, and enforce pre/post counts, zero-frame tolerance, and adjacent-gap tolerance for B.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same unittest command. Expected: all Task 1 tests pass.

- [ ] **Step 5: Add edge-case tests before expanding behavior**

Add four named tests: `test_down_before_first_observed_frame_is_excluded_from_a`, `test_placeholder_frames_do_not_complete_b`, `test_large_timestamp_gap_excludes_b`, and `test_touch_event_timestamp_is_fallback_down_marker`. Each test must construct a complete fixture and assert the relevant eligibility or fallback field.

- [ ] **Step 6: Run edge tests and verify RED, then implement minimal fixes**

Expected RED reasons: missing fallback or missing continuity/exclusion behavior. Implement only the required checks and rerun until green.

- [ ] **Step 7: Commit Task 1**

```powershell
git add android/analysis/build_historical_down_anchor_datasets.py android/analysis/tests/test_build_historical_down_anchor_datasets.py
git commit -m "feat: classify historical down-anchor windows"
```

### Task 2: Deterministic A/B dataset writer

**Files:**
- Modify: `android/analysis/build_historical_down_anchor_datasets.py`
- Modify: `android/analysis/tests/test_build_historical_down_anchor_datasets.py`

- [ ] **Step 1: Write failing integration test**

Use `tempfile.TemporaryDirectory` to create three source JSON files: excluded, A-only, and A+B. Write a source manifest and call:

```python
summary = build_datasets(source_manifest, output_root)
```

Assert:

```python
self.assertEqual(2, summary["datasetA"]["retained"])
self.assertEqual(1, summary["datasetB"]["retained"])
self.assertTrue((output_root / "a_down_inside_saved_window" / "manifest.csv").exists())
self.assertTrue((output_root / "b_complete_down_minus5_plus9" / "samples" / "complete.json").exists())
self.assertEqual({"complete"}, set(summary["datasetB"]["sampleIds"]))
```

- [ ] **Step 2: Run integration test and verify RED**

Expected: `build_datasets` is missing.

- [ ] **Step 3: Implement streaming writer and CLI**

Add this public entry point and CLI contract:

```python
def build_datasets(source_manifest, output_root):
    source_manifest = Path(source_manifest)
    output_root = Path(output_root)
    return DatasetBuilder(source_manifest, output_root).build()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()
```

Write `manifest.csv` even when no rows qualify. Write per-dataset and root `summary.json` with discovered, retained, excluded-reason, schema, collector, device, and class counts. B derived files must include `sourceId`, `sourcePath`, `downAnchorTimestamp`, `sampleRateHz`, and exactly 15 `downFrames`.

- [ ] **Step 4: Run integration test and verify GREEN**

Expected: all tests pass.

- [ ] **Step 5: Add and verify empty-B test**

Create only A-only input and assert B manifest contains a header, B summary has `retained: 0`, and no synthetic frame files are emitted.

- [ ] **Step 6: Commit Task 2**

```powershell
git add android/analysis/build_historical_down_anchor_datasets.py android/analysis/tests/test_build_historical_down_anchor_datasets.py
git commit -m "feat: write separate historical down datasets"
```

### Task 3: Documentation and full dataset build

**Files:**
- Modify: `FunctionAndDataStructureDescription.md`
- Generate outside Git: `data/down_anchor_historical_20260731/`

- [ ] **Step 1: Update technical documentation**

Document the source contract, placeholder rule, A/B criteria, manifest fields, B derived JSON schema, CLI command, and append a 2026-07-31 change-log entry.

- [ ] **Step 2: Run all focused tests**

```powershell
python -m unittest android.analysis.tests.test_build_historical_down_anchor_datasets -v
```

Expected: all tests pass with zero failures.

- [ ] **Step 3: Run full build against the immutable source**

```powershell
python android/analysis/build_historical_down_anchor_datasets.py `
  --source-manifest "C:/Users/血饮/Desktop/待完成任务/original/touch_eyes_dataAnalysis/data/deduplicated_raw/manifest.csv" `
  --output-dir "C:/Users/血饮/Desktop/待完成任务/original/touch_eyes_dataAnalysis/data/down_anchor_historical_20260731"
```

Expected: process exits zero and root summary reports `discovered: 89385`.

- [ ] **Step 4: Verify outputs**

Verify:

```text
A IDs are unique
B IDs are unique
B IDs are a subset of A IDs
every B JSON has frameIndex -5...9
every B JSON has 15 frames
every source SHA-256 remains equal to deduplicated_raw/manifest.csv
all summaries and derived JSON parse successfully
```

- [ ] **Step 5: Commit documentation**

```powershell
git add FunctionAndDataStructureDescription.md
git commit -m "docs: describe historical down-anchor datasets"
```

- [ ] **Step 6: Final verification**

Run focused tests again, run `git diff --check`, inspect `git status --short`, and report exact A/B counts, output locations, exclusions, and any residual evidence limits.
