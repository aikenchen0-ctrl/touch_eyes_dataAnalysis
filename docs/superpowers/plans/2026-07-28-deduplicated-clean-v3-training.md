# Deduplicated Clean V3 Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild a leakage-resistant cleaning, preprocessing, strict-validation, and four-class CNN training pipeline for all 89,385 unique raw samples.

**Architecture:** A new catalog and cleaning module reads only paths listed by the deduplicated manifest, applies context-aware quality decisions, assigns real or inferred sessions, and produces immutable derived manifests. A fold builder creates domain, collection, and temporal holdouts without session overlap. The existing trainer gains one per-sample baseline IMU transform, while a resumable experiment runner executes and ranks probe candidates.

**Tech Stack:** Python 3.11 standard library, PyTorch, unittest, CSV, JSON, existing CNN training utilities.

---

### Task 1: Clean V3 catalog and quality policy

**Files:**
- Create: `android/analysis/build_deduplicated_clean_v3.py`
- Create: `android/analysis/tests/test_build_deduplicated_clean_v3.py`

- [ ] Write failing tests proving report JSON is excluded, context-specific thresholds do not apply the old global heavy-frame rule, structural failures are rejected, and review weights remain bounded.
- [ ] Run `python -m unittest android.analysis.tests.test_build_deduplicated_clean_v3 -v` and confirm the module import fails.
- [ ] Implement manifest-driven JSON loading, metadata extraction, context statistics, quality decisions, and deterministic CSV/JSON output.
- [ ] Run the focused tests and confirm all pass.

### Task 2: Session assignment and strict folds

**Files:**
- Create: `android/analysis/build_clean_v3_folds.py`
- Create: `android/analysis/tests/test_build_clean_v3_folds.py`

- [ ] Write failing tests for raw session IDs, 60-second legacy session breaks, within-session thinning, domain holdouts, eligible collection holdouts, and temporal-tail session isolation.
- [ ] Run the focused test and confirm failure because the fold module is absent.
- [ ] Implement session assignment, thinning, fold generation, split validation, and fold summaries.
- [ ] Run focused tests and confirm all pass.

### Task 3: Baseline IMU candidate

**Files:**
- Modify: `android/analysis/train_android_finger_four_class_cnn.py`
- Modify: `android/analysis/tests/test_train_android_finger_four_class_cnn.py`

- [ ] Write a failing test showing `baseline_imu` subtracts pre-touch medians, applies bounded per-channel scales, and retains normalized coordinates.
- [ ] Run the focused test and confirm the transform is unsupported.
- [ ] Implement `baseline_imu` without using labels, future samples, validation statistics, or absolute contact geometry.
- [ ] Run trainer tests and confirm all pass.

### Task 4: Resumable probe runner

**Files:**
- Create: `android/analysis/run_clean_v3_probe_experiments.py`
- Create: `android/analysis/tests/test_run_clean_v3_probe_experiments.py`

- [ ] Write failing tests for candidate specifications, strict-fold command construction, completed-run skipping, and comparison aggregation.
- [ ] Implement four candidates, resumable execution, and ranking by mean/worst balanced accuracy and minimum recall.
- [ ] Run focused tests and confirm all pass.

### Task 5: Build the derived dataset

**Files:**
- Generate: `android/analysis/datasets/deduplicated_clean_v3_20260728/`

- [ ] Run Clean V3 against `data/deduplicated_raw/manifest.csv`.
- [ ] Generate sessions, thinned training view, domain folds, eligible collection folds, and temporal-tail fold.
- [ ] Verify source IDs remain unique, no session crosses train/valid, and source JSON hashes remain unchanged.

### Task 6: Run candidate probes

**Files:**
- Generate: `models/deduplicated_clean_v3_20260728/`

- [ ] Confirm CUDA availability and record the runtime environment.
- [ ] Run all four candidates on the strict domain folds with 30 epochs, patience 8, batch size 128, seed 22, and resume enabled.
- [ ] Aggregate mean and worst-fold balanced accuracy, minimum recall, best epoch, and channel count.
- [ ] Select at most two candidates for full training; do not deploy any candidate automatically.

### Task 7: Documentation and verification

**Files:**
- Modify: `FunctionAndDataStructureDescription.md`

- [ ] Document Clean V3 policy, session grouping, fold definitions, candidate inputs, generated counts, and measured probe results.
- [ ] Run all analysis tests, Python compilation, manifest integrity checks, session leakage checks, and `git diff --check`.
