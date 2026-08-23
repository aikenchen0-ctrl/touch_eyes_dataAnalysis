# Aispect Touch Android SDK Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable Aispect Android SDK at `C:/codeDev/cnn/sdk/aispect-touch-android` that preserves the public `com.zhifaios.eyes.touch` API while supporting the latest causal time-grid remote models.

**Architecture:** Start from the `eyes/android/aispect-core` product shell: its Maven publication, consumer rules, public touch facade, lifecycle management, canonical assignment protocol, and verified downloaded-model store remain the SDK boundary. Port only the latest causal runtime internals from the data-analysis module: the 9-frame time-grid feature builder, GroupNorm CNN runtime, model-contract dispatch, and causal input metadata. The collector, JSON dataset writer, and experimental asset catalog stay out of the staged SDK.

**Tech Stack:** Android Java 17, Android Library, WorkManager 2.9.1, `HttpURLConnection`, `org.json`, JUnit 4.

---

### Task 1: Establish the Product SDK Baseline

**Files:**
- Replace: `C:/codeDev/cnn/sdk/aispect-touch-android/build.gradle`
- Replace: `C:/codeDev/cnn/sdk/aispect-touch-android/consumer-rules.pro`
- Replace: `C:/codeDev/cnn/sdk/aispect-touch-android/src/main/**`
- Preserve: `com.zhifaios.eyes.touch.*` public facade from `C:/Users/血饮/Desktop/eyes/android/aispect-core`

- [ ] Copy the Maven-published Android Library configuration, manifest permissions, consumer rules, public API package, and existing public API tests from the product SDK source.
- [ ] Exclude all `build`, `.gradle`, `.git`, host-App, data-collection, Python-analysis, and historical model artifacts.
- [ ] Verify the target has the public classes `AispectTouchClassifier`, `AispectTouchConfig`, `AispectTouchListener`, `AispectTouchResult`, and model-update result classes.

### Task 2: Add a Causal Runtime Contract Test

**Files:**
- Create: `C:/codeDev/cnn/sdk/aispect-touch-android/src/test/java/com/zhifaios/eyes/aispect/AispectCausalRuntimeContractTest.java`

- [ ] Write a test that declares the two supported remote contracts: `causal_time_grid_imu11_touch7_mask5_delta_gravity_v1` with `9 x 23` input and `causal_time_grid_touch7_mask5_no_gravity_v1` with `9 x 20` input.
- [ ] Write a test that rejects a mismatched architecture, feature count, label order, and tensor shape.
- [ ] Run the new test before porting causal production code and record the expected missing-class failure.

### Task 3: Port the Latest Causal Feature and CNN Runtime

**Files:**
- Create: `C:/codeDev/cnn/sdk/aispect-touch-android/src/main/java/com/zhifaios/eyes/aispect/AispectCausalPressFeatureBuilder.java`
- Create: `C:/codeDev/cnn/sdk/aispect-touch-android/src/main/java/com/zhifaios/eyes/aispect/AispectTimeGridGroupNormRuntime.java`
- Create: `C:/codeDev/cnn/sdk/aispect-touch-android/src/main/java/com/zhifaios/eyes/aispect/AispectTouchInputMetadata.java`
- Modify: `C:/codeDev/cnn/sdk/aispect-touch-android/src/main/java/com/zhifaios/eyes/aispect/AispectImpactCNNClassifier.java`
- Modify: `C:/codeDev/cnn/sdk/aispect-touch-android/src/main/java/com/zhifaios/eyes/aispect/AispectCollectionController.java`

- [ ] Copy only the causal feature constructors and GroupNorm inference runtime needed for the two declared contracts.
- [ ] Extend model dispatch by explicit `featureContract` and `runtimeArchitecture`; never infer a contract from channel count.
- [ ] Keep the public classifier API unchanged; build the fixed `-15ms...+25ms` nine-slot window internally and emit one result only after causal coverage is available.
- [ ] Run the causal runtime and public mapper tests.

### Task 4: Extend Remote Validation Without Replacing the Existing Protocol

**Files:**
- Modify: `C:/codeDev/cnn/sdk/aispect-touch-android/src/main/java/com/zhifaios/eyes/aispect/AispectModelBundleValidator.java`
- Modify: `C:/codeDev/cnn/sdk/aispect-touch-android/src/main/java/com/zhifaios/eyes/aispect/AispectRemoteModelUpdater.java`
- Modify: `C:/codeDev/cnn/sdk/aispect-touch-android/src/main/java/com/zhifaios/eyes/aispect/AispectDownloadedModelStore.java`
- Modify: corresponding tests under `src/test/java/com/zhifaios/eyes/aispect/`

- [ ] Preserve canonical assignment parsing, HTTPS policy, SHA-256 checks, immutable version storage, current/previous rollback, and touch-safe deferred activation.
- [ ] Add a validation branch for the two causal GroupNorm contracts; retain the legacy release-CNN validation branch for backwards compatibility.
- [ ] Verify a failed causal model download cannot replace the verified current model.

### Task 5: Reduce Embedded Assets and Publishability Checks

**Files:**
- Replace: `C:/codeDev/cnn/sdk/aispect-touch-android/src/main/assets/aispect_impact_models/AispectImpactCNNModelCatalog.json`
- Replace: only the one selected fallback weights/scaler pair in the same asset directory
- Create: `C:/codeDev/cnn/sdk/aispect-touch-android/README.md`

- [ ] Retain exactly one explicitly deployable fallback model; omit experimental, ablation, historical, and `deploymentAllowed=false` assets.
- [ ] Document the public dependency coordinate, initialization path, canonical remote model configuration, local fallback, and known causal-model waiting semantics.
- [ ] Build the library and run all JVM tests. Confirm no copied collection-data or analysis artifacts remain.
