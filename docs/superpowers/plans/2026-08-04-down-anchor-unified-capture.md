# Down-Anchor Unified Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify Android and iOS capture around an ACTION_DOWN-anchored `-5...+9` window while preserving a `down-15...lift+15` raw record and disabling legacy model output until the new dataset is trained.

**Architecture:** Keep the variable-length raw collector and fixed runtime window as separate views. Replace the Android lift-triggered diagnostic path and the iOS delta-triggered path with touch-down window readiness; retain old assets only as legacy resources. Serialize a new raw schema without derived delta fields.

**Tech Stack:** Android Java/Gradle/JUnit, iOS Swift/CoreMotion, existing JSON data logger and CNN asset loaders.

---

### Task 1: Lock the new window and schema contract with tests

**Files:**
- Modify: `android/aispect-core/src/test/java/com/zhifaios/eyes/aispect/AispectSignalWindowBuilderTest.java`
- Create: `android/aispect-core/src/test/java/com/zhifaios/eyes/aispect/DownAnchorCaptureContractTest.java`
- Test: Android `:aispect-core:test`

- [x] **Step 1: Write failing tests** for a complete down-anchored `-5...+9` window, one-shot readiness, incomplete post-down data, and serialized frames without `delta`/`maxAbsDelta`.
- [x] **Step 2: Run the focused tests and confirm they fail because the current window/logger contract is lift-centered and still serializes delta fields.**
- [x] **Step 3: Keep the tests as the executable contract for the following implementation tasks.**

### Task 2: Implement Android down-trigger capture and schema v5

**Files:**
- Modify: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectModels.java`
- Modify: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectImpactSignalCollector.java`
- Modify: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectSignalWindowBuilder.java`
- Modify: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectCollectionController.java`
- Modify: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectDataLogger.java`
- Modify: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectImpactCNNClassifier.java`
- Modify: `android/app/src/main/java/com/zhifaios/eyes/android/MainActivity.java`
- Modify: `android/app/src/main/res/values/strings.xml`

- [x] **Step 1: Remove derived delta calculation from the Android signal collector and frame JSON contract, retaining raw acceleration and timestamps.**
- [x] **Step 2: Add a complete fixed-window builder for `-5...+9` that never classifies padded or incomplete frames.**
- [x] **Step 3: Schedule one down-window readiness check per touch sequence and stop using the lift timestamp for the early result.**
- [x] **Step 4: Disable old CNN/first-touch results while the new model is absent and expose an explicit pending-model status.**
- [x] **Step 5: Preserve the down/lift raw logger and write schema v5 without delta-derived fields.**
- [x] **Step 6: Run the focused Android tests and make them pass.**

### Task 3: Apply the same semantics to iOS

**Files:**
- Modify: `ZHIFAeyes/Modules/AispectImpactDetector.swift`
- Modify: `ZHIFAeyes/Modules/AispectImpactDatasetCollector.swift`
- Modify: `ZHIFAeyes/Modules/AispectImpactCNNClassifier.swift`
- Modify: `ZHIFAeyes/UI/RootViewController.swift`

- [x] **Step 1: Replace the delta threshold path with a down-anchor readiness path and a one-shot `-5...+9` window state.**
- [x] **Step 2: Align serialized frame fields and schema version with Android, omitting delta-derived fields.**
- [x] **Step 3: Keep post-lift raw collection independent from classification and expose the pending-model state.**
- [x] **Step 4: Run source-level checks and document that Apple build verification is unavailable on Windows.**

### Task 4: Documentation and verification

**Files:**
- Modify: `FunctionAndDataStructureDescription.md`
- Modify: `docs/superpowers/specs/2026-08-04-down-anchor-unified-capture-design.md`

- [x] **Step 1: Update the function/data description and change log with the new terminology and schema.**
- [x] **Step 2: Run all Android Python tests, direct JUnit tests, and APK assembly.**
- [x] **Step 3: Verify no new raw JSON writer emits `delta` or `maxAbsDelta`, and verify old data/model assets remain untouched.**
- [x] **Step 4: Record remaining limitation: no new model is trained or installed in this round.**

执行结果：Android standard/crossDevice debug APK 均已构建；Gradle 单元测试任务受 Java 21 worker 类路径环境限制，使用同一编译产物直接运行 8 个 JUnit 合同测试通过，Python 回归测试 59 个全部通过。Apple 真机构建仍需 macOS/Xcode，本 Windows 环境只完成 Swift 源码级检查。
