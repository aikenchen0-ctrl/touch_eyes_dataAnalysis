# Grouped Touch Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean the four new Android collections with device-aware quality rules, build leakage-resistant grouped manifests, and train comparable four-class CNN holdout models with batch size 128.

**Architecture:** Keep raw JSON immutable. Extend the existing cleaner to emit high-quality and softly weighted review views, then add a manifest builder that joins quality metadata onto the existing all-collections manifest, assigns time-block session groups, removes temporally redundant samples, and creates one predefined holdout manifest per collection. Reuse the existing CNN trainer and exported report format.

**Tech Stack:** Python 3.11 standard library, PyTorch 2.11, unittest, CSV/JSON manifests.

---

### Task 1: Device-aware quality decisions

**Files:**
- Modify: `android/analysis/clean_android_touch_dataset.py`
- Create: `android/analysis/tests/test_clean_android_touch_dataset.py`

- [ ] Write tests proving impact and travel thresholds are derived independently for each device and motion state.
- [ ] Run `python -m unittest android.analysis.tests.test_clean_android_touch_dataset -v` and verify the new tests fail.
- [ ] Add context statistics, context-relative quality reasons, and soft review weights.
- [ ] Emit `weighted_train_manifest.csv` while preserving existing manifests.
- [ ] Run the cleaner tests and verify they pass.

### Task 2: Leakage-resistant grouped manifests

**Files:**
- Create: `android/analysis/build_grouped_four_class_manifest.py`
- Create: `android/analysis/tests/test_build_grouped_four_class_manifest.py`

- [ ] Write tests for quality joins, five-minute session groups, temporal thinning, deduplication, and complete-collection holdout assignment.
- [ ] Run `python -m unittest android.analysis.tests.test_build_grouped_four_class_manifest -v` and verify the tests fail.
- [ ] Implement the generic manifest transformation and summary export.
- [ ] Run both analysis test modules and verify they pass.

### Task 3: Generate cleaned datasets

**Files:**
- Generate: `data/collections/*/cleaned_v2/`

- [ ] Run the cleaner for all four new raw collection directories.
- [ ] Inspect counts by quality bucket, class, motion, device, and reason.
- [ ] Build one training manifest per held-out new collection using high-quality and weighted review samples.
- [ ] Verify no IDs or session groups cross predefined train and valid partitions.

### Task 4: Train controlled comparisons

**Files:**
- Generate: `models/all_collections_grouped_clean_v2_20260724/`

- [ ] Train `enhanced_patch21_no_confidence` with the previous hyperparameters and `batch_size=128` for each new-collection holdout.
- [ ] Train one grouped/random-control model only if needed to quantify the remaining split effect.
- [ ] Aggregate accuracy, balanced accuracy, minimum class recall, per-class recall, best epoch, and confusion matrices.

### Task 5: Documentation and verification

**Files:**
- Modify: `FunctionAndDataStructureDescription.md`

- [ ] Document the new quality policy, weighted manifest, session-group fields, holdout policy, and model results.
- [ ] Run all Python analysis unit tests.
- [ ] Validate generated CSV/JSON files and ensure all referenced sample paths exist.
- [ ] Compare the new models against the prior 0.8991 random baseline and 0.6936 collection holdout baseline.
