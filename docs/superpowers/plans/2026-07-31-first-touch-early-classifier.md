# First-Touch Early Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and deploy an experimental four-class model that emits a prediction on `ACTION_DOWN` from the first touch frame.

**Architecture:** A Python trainer exports a standardized one-hidden-layer MLP JSON. A reusable Java classifier loads the asset, extracts the same ten features, and publishes an early prediction through the existing controller status channel without changing final lift-up inference.

**Tech Stack:** Python 3.11, PyTorch, Java, JUnit 4, Android Gradle, ADB.

---

### Task 1: First-touch training pipeline

**Files:**
- Create: `android/analysis/train_android_first_touch_mlp.py`
- Create: `android/analysis/tests/test_train_android_first_touch_mlp.py`

- [x] Write failing tests for first-frame feature extraction, missing-field fallback, deterministic split, and JSON export shape.
- [x] Implement the ten-feature extractor and MLP training/export entry point.
- [x] Run the focused Python tests until green.

### Task 2: Train and evaluate the experimental model

**Files:**
- Generate: `models/first_touch_early_v1_20260731/`

- [x] Train a validation model on all 79,878 Clean V2 rows and record balanced accuracy plus four class recalls.
- [x] Select the best epoch and fit all eligible rows for that epoch.
- [x] Preserve the final model as an experimental artifact even if it does not exceed the lift-up model.

### Task 3: Android first-touch inference

**Files:**
- Create: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectFirstTouchClassifier.java`
- Create: `android/aispect-core/src/test/java/com/zhifaios/eyes/aispect/AispectFirstTouchClassifierTest.java`
- Modify: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectCollectionController.java`
- Create: `android/aispect-core/src/main/assets/aispect_impact_models/AispectFirstTouchModel_first_touch_early_v1.json`

- [x] Write failing Java tests for feature parity, softmax inference, and unavailable-model behavior.
- [x] Implement the classifier and load the exported asset.
- [x] Emit one early prediction after the `ACTION_DOWN` frame is captured, without changing lift-up inference or completed-touch recall accounting.
- [x] Run focused Java tests until green.

### Task 4: Documentation and device verification

**Files:**
- Modify: `FunctionAndDataStructureDescription.md`
- Generate: `android/app/build/outputs/apk/crossDevice/debug/app-crossDevice-debug.apk`

- [x] Document the feature contract, metrics, runtime event order, and experimental limitations.
- [x] Run all Python and Java tests.
- [x] Build and install the crossDevice APK on the connected phone.
- [x] Verify the UI shows the first-touch model during press and the existing model after lift-up.
