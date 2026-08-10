# Full Data Clean V2 Android Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package `full_data_clean_v2_old13_all_118` into the crossDevice Android app, make it the default selectable model, and install the verified APK on the connected phone.

**Architecture:** Reuse the existing JSON asset catalog and flavor-specific default model mechanism. Protect the deployment with the existing Python asset consistency test, then build and verify the installed package through ADB.

**Tech Stack:** Python unittest, JSON assets, Gradle Android build, ADB.

---

### Task 1: Define the deployment expectation

**Files:**
- Modify: `android/analysis/tests/test_latest_model_deployment.py`

- [x] Change `MODEL_ID` to `full_data_clean_v2_old13_all_118`.
- [x] Run `python -m unittest android.analysis.tests.test_latest_model_deployment -v` and confirm the default/catalog test fails because the model is not deployed.

### Task 2: Import and select the model

**Files:**
- Create: `android/aispect-core/src/main/assets/aispect_impact_models/AispectImpactCNNScaler_full_data_clean_v2_old13_all_118.json`
- Create: `android/aispect-core/src/main/assets/aispect_impact_models/AispectImpactCNNWeights_full_data_clean_v2_old13_all_118.json`
- Modify: `android/aispect-core/src/main/assets/aispect_impact_models/AispectImpactCNNModelCatalog.json`
- Modify: `android/app/build.gradle`

- [x] Copy the two generated model JSON files without modifying their contents.
- [x] Insert the generated catalog entry at index zero.
- [x] Set the crossDevice flavor default to `full_data_clean_v2_old13_all_118` and leave the standard default unchanged.
- [x] Run the focused deployment test and confirm it passes.

### Task 3: Document and install

**Files:**
- Modify: `FunctionAndDataStructureDescription.md`
- Generate: `android/app/build/outputs/apk/crossDevice/debug/app-crossDevice-debug.apk`

- [x] Document the catalog/default/asset changes and installation target.
- [x] Run all 42 analysis tests.
- [x] Build the crossDevice debug APK.
- [x] Install it on ADB serial `a2d0a964`, launch the crossDevice package, and verify the new model is selected.
