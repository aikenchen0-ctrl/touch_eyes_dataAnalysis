# Collector Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a manually entered collector name and an automatically generated anonymous stable device ID to every Android training sample.

**Architecture:** Keep identity state in the collection controller and freeze it at session start. Generate and persist the anonymous device ID through Android `SharedPreferences`, then pass a small structured identity object to the data logger.

**Tech Stack:** Java, Android SDK, `org.json`, Gradle

---

### Task 1: Identity Model And Persistence

**Files:**
- Modify: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectModels.java`
- Modify: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectCollectionController.java`

- [ ] Add a `CollectorIdentity` value object containing `userName` and `deviceId` with JSON serialization.
- [ ] Generate a UUID once and persist it in private `SharedPreferences`.
- [ ] Add controller methods to update the selected user name only while collection is stopped.
- [ ] Freeze the selected identity when a collection session starts.

### Task 2: Raw JSON Export

**Files:**
- Modify: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectDataLogger.java`
- Modify: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectCollectionController.java`

- [ ] Extend `writeSample` with a `CollectorIdentity` parameter.
- [ ] Change new records to `schemaVersion: 3`.
- [ ] Write the identity under the `collector` JSON object.
- [ ] Build the core module to catch signature and serialization errors.

### Task 3: User Name Input

**Files:**
- Modify: `android/app/src/main/java/com/zhifaios/eyes/android/MainActivity.java`
- Modify: `android/app/src/main/res/values/strings.xml`

- [ ] Add a single-line user-name input to the training collection section.
- [ ] Reject session start when the trimmed name is empty.
- [ ] Lock the input while collection is active and restore it after stopping.
- [ ] Show the current user name in collection status without exposing the device ID.

### Task 4: Documentation And Device Verification

**Files:**
- Modify: `FunctionAndDataStructureDescription.md`

- [ ] Document the `collector` object, schema version, persistence, and session-lock behavior.
- [ ] Build `:app:assembleDebug` and verify `BUILD SUCCESSFUL`.
- [ ] Install with `adb install -r` and launch the application.
- [ ] Capture one sample and verify `collector.userName`, stable `collector.deviceId`, and `schemaVersion: 3`.

