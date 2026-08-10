# Existing-Data Device Invariance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and compare three existing-data-only feature transformations that reduce device and collection shortcuts in four-class touch recognition.

**Architecture:** Extend the existing four-class trainer after cache loading with deterministic matrix transforms and training-fold-only group balancing. Keep the base cache immutable and export transformed feature names in each model scaler. Drive the existing predefined holdout manifests through a small batch runner and aggregate reports.

**Tech Stack:** Python 3.11, PyTorch 2.11, unittest, existing CSV/JSON/pickle artifacts.

---

### Task 1: Derived matrix transforms

**Files:**
- Modify: `android/analysis/train_android_finger_four_class_cnn.py`
- Modify: `android/analysis/tests/test_train_android_finger_four_class_cnn.py`

- [ ] Write failing tests for `imu11_no_time`, `invariant_shape`, and causal rolling `relative_touch` outputs.
- [ ] Verify the tests fail because transform APIs are absent.
- [ ] Implement dimensionless per-sample channels and prior-only rolling medians.
- [ ] Verify transform tests and existing cache tests pass.

### Task 2: Training-fold group balancing

**Files:**
- Modify: `android/analysis/train_android_finger_four_class_cnn.py`
- Modify: `android/analysis/tests/test_train_android_finger_four_class_cnn.py`

- [ ] Write a failing test proving small collection-class groups receive larger bounded factors and validation rows are not involved.
- [ ] Implement training-only normalized inverse-frequency factors multiplied by quality weights.
- [ ] Verify all trainer tests pass.

### Task 3: Four-fold candidate runner

**Files:**
- Create: `android/analysis/run_group_invariant_four_class_experiments.py`
- Create: `android/analysis/tests/test_run_group_invariant_four_class_experiments.py`

- [ ] Write failing tests for deterministic command construction and report aggregation.
- [ ] Implement three-candidate, four-holdout execution with resume support.
- [ ] Verify runner tests pass.

### Task 4: Train and evaluate

**Files:**
- Generate: `models/all_collections_invariant_v3_20260724/`

- [ ] Run all 12 candidate/fold combinations from the shared cache.
- [ ] Aggregate mean and worst-fold balanced accuracy, minimum recall, best epoch, and confusion matrices.
- [ ] Select a candidate only from strict holdout metrics.

### Task 5: Documentation and verification

**Files:**
- Modify: `FunctionAndDataStructureDescription.md`

- [ ] Document transforms, causal baseline behavior, weighting, outputs, and measured results.
- [ ] Run all analysis tests, Python compilation, path/model artifact checks, and `git diff --check`.
