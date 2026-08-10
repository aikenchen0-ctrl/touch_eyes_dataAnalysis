# Schema v5 Full-Factorial Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended) or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resumable runner that completes the 12-preset x 7-window schema v5 down-anchor CNN matrix and all three strict motion holdouts.

**Architecture:** A new analysis-only runner enumerates stable temporal and strict specifications, reuses the existing model directory layout, delegates training to the existing CNN script, and aggregates completed reports after every successful task. Existing reports are treated as checkpoints only when their report, weights, and scaler exist for the generated model id.

**Tech Stack:** Python standard library, existing PyTorch training entry point, `unittest`, CSV, JSON, and `subprocess`.

---

### Task 1: Define the Experiment Contract in Tests

**Files:**
- Create: `android/analysis/tests/test_run_schema5_full_factorial_ablation.py`
- Create: `android/analysis/run_schema5_full_factorial_ablation.py`

- [x] **Step 1: Write a failing matrix-coverage test**

```python
def test_specs_cover_all_presets_windows_and_motion_folds(self):
    temporal = runner.temporal_specs(Path("manifests"), Path("models"), Path("cache"))
    strict = runner.strict_specs(Path("manifests"), Path("models"), Path("cache"))

    self.assertEqual(84, len(temporal))
    self.assertEqual(252, len(strict))
    self.assertEqual(set(runner.FEATURE_PRESET_NAMES), {item["preset"] for item in temporal})
    self.assertEqual(set(runner.WINDOWS), {item["window"] for item in temporal})
```

- [x] **Step 2: Run the test to verify RED**

Run: `python -m unittest android.analysis.tests.test_run_schema5_full_factorial_ablation`

Expected: FAIL because `run_schema5_full_factorial_ablation` does not exist.

- [x] **Step 3: Add the minimal constants and spec constructors**

Implement ordered preset names, seven fixed frame-index strings, three motion folds, and output path constructors that preserve the existing baseline/window/holdout layout.

- [x] **Step 4: Run the test to verify GREEN**

Run: `python -m unittest android.analysis.tests.test_run_schema5_full_factorial_ablation`

Expected: PASS for matrix coverage.

### Task 2: Test Resume and Training-Command Semantics

**Files:**
- Modify: `android/analysis/tests/test_run_schema5_full_factorial_ablation.py`
- Modify: `android/analysis/run_schema5_full_factorial_ablation.py`

- [x] **Step 1: Write failing tests for historical paths, completion, and flags**

```python
def test_baseline_and_nonbaseline_temporal_paths_match_existing_layout(self):
    specs = runner.temporal_specs(Path("manifests"), Path("models"), Path("cache"))
    baseline = next(item for item in specs if item["preset"] == "old13" and item["window"] == "pre5_post9")
    alternate = next(item for item in specs if item["preset"] == "old13" and item["window"] == "pre5_post0")

    self.assertEqual(Path("models") / "feature_temporal" / "old13", baseline["outputDir"])
    self.assertEqual(Path("models") / "window_temporal" / "old13__pre5_post0", alternate["outputDir"])
```

- [x] **Step 2: Run the tests to verify RED**

Run: `python -m unittest android.analysis.tests.test_run_schema5_full_factorial_ablation`

Expected: FAIL because command and completion helpers do not exist.

- [x] **Step 3: Implement the minimal command and completion helpers**

Build commands with `down`, `error`, `predefined`, locked optimizer parameters, stage-specific epoch/patience values, and the spec cache path. Require report, weights, and scaler files for a completed specification.

- [x] **Step 4: Run the tests to verify GREEN**

Run: `python -m unittest android.analysis.tests.test_run_schema5_full_factorial_ablation`

Expected: PASS for command semantics and resume checks.

### Task 3: Test Aggregation and Implement the Runner

**Files:**
- Modify: `android/analysis/tests/test_run_schema5_full_factorial_ablation.py`
- Modify: `android/analysis/run_schema5_full_factorial_ablation.py`

- [x] **Step 1: Write a failing strict-aggregation test**

```python
def test_strict_aggregate_ranks_mean_then_worst_holdout_score(self):
    specs = runner.strict_specs(Path("manifests"), Path("models"), Path("cache"))
    # Write three small reports for one pair and assert its fold count, mean, and worst score.
```

- [x] **Step 2: Run the test to verify RED**

Run: `python -m unittest android.analysis.tests.test_run_schema5_full_factorial_ablation`

Expected: FAIL because strict aggregation does not exist.

- [x] **Step 3: Implement serial execution and checkpoint summaries**

Add `--stage temporal|strict|all`, `--dry-run`, and `--force`. After each successful run, write temporal and strict CSV/JSON summaries plus a status JSON with planned, completed, and pending counts. Do not import models into Android assets.

- [x] **Step 4: Run the focused and existing tests**

Run: `python -m unittest android.analysis.tests.test_run_schema5_full_factorial_ablation android.analysis.tests.test_train_android_finger_four_class_cnn`

Expected: PASS with no failures.

### Task 4: Record the Change and Run the Matrix

**Files:**
- Modify: `FunctionAndDataStructureDescription.md`
- Output: `models/bbsiak_schema5_down_anchor_20260806/full_factorial_status.json`
- Output: `models/bbsiak_schema5_down_anchor_20260806/full_factorial_temporal_summary.csv`
- Output: `models/bbsiak_schema5_down_anchor_20260806/full_factorial_strict_summary.csv`

- [x] **Step 1: Update the technical description**

Document the full-matrix runner, its resume rules, locked contract, and the fact that its ranking remains offline-only.

- [x] **Step 2: Validate a dry run**

Run: `python android/analysis/run_schema5_full_factorial_ablation.py --dry-run --stage all`

Expected: emits the missing work without starting training.

- [ ] **Step 3: Execute the missing work**

Run: `python android/analysis/run_schema5_full_factorial_ablation.py --stage all`

Expected: resumes from existing reports, completes missing temporal and strict jobs, and continually rewrites status/summary artifacts.

- [ ] **Step 4: Verify final coverage**

Run: `python android/analysis/run_schema5_full_factorial_ablation.py --dry-run --stage all`

Expected: no remaining command output; status reports 84/84 temporal pairs and 252/252 strict runs.
