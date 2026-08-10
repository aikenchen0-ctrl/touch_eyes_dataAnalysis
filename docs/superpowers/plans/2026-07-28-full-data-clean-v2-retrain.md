# Full Data Clean V2 Retrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the deployable four-class model from every current unique sample by exactly reproducing the prior Clean V2 pooled `old13` training procedure.

**Architecture:** Treat `data/deduplicated_raw/manifest.csv` as the uniqueness authority and prove it contains the same IDs as the prior 89,385-row all-collections input. Reuse the prior Clean V2 processed manifest because the ID sets are identical, run a 120-epoch random validation model to select the epoch, then train a separate final model on all 79,878 eligible rows without overwriting any existing model.

**Tech Stack:** Python 3.11, PyTorch, CSV/JSON manifests, Android JSON model export.

---

### Task 1: Verify the complete input set

**Files:**
- Read: `data/deduplicated_raw/manifest.csv`
- Read: `android/analysis/datasets/all_collections_four_class_20260724/full_all_collections_manifest.csv`
- Read: `android/analysis/datasets/all_collections_grouped_clean_v2_20260724/le2110_zhuo_20260723/manifest.csv`

- [x] Compare unique IDs between the current deduplicated manifest and the prior all-collections input.
- [x] Verify the Clean V2 training manifest has no duplicate IDs, only supported four-class labels, and only existing sample paths.
- [x] Record total, eligible, quality-bucket, class, collector, and device counts.

### Task 2: Reproduce the prior validation training

**Files:**
- Generate: `models/full_data_clean_v2_old13_20260728/validation_120/`

- [x] Run `android/analysis/train_android_finger_four_class_cnn.py` with `old13`, frames `-5...+9`, zero missing-frame policy, batch `256`, learning rate `0.001`, weight decay `0.0001`, random `25%` validation, seed `22`, and `120` epochs.
- [x] Read `training_report.json` and select the epoch with maximum validation balanced accuracy.
- [x] Compare validation balanced accuracy and per-class recall with `full_old13_seen_devices_current_preprocessed_120`.

### Task 3: Train the final pooled model

**Files:**
- Generate: `models/full_data_clean_v2_old13_20260728/final_all/`

- [x] Train with the same feature, optimizer, weighting, seed, and missing-frame settings, using `split_strategy=all` and the epoch selected in Task 2.
- [x] Verify scaler, weights, catalog entry, and training report are valid JSON and reference the new model ID.
- [x] Keep all prior models unchanged and do not add the model to Android until its validation result is reviewed.

### Task 4: Document and verify

**Files:**
- Modify: `FunctionAndDataStructureDescription.md`

- [x] Document the exact data equivalence check, preprocessing policy, hyperparameters, selected epoch, validation metrics, and final artifact paths.
- [x] Run all analysis unit tests.
- [x] Validate generated reports and confirm the final model used every eligible row exactly once.
