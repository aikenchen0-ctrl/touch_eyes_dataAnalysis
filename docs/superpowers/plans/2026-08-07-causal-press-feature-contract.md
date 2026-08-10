# Causal Press Feature Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build causal schema-v6 press-phase feature matrices in Python and Android Java, with a shared fixture that proves zero-order-hold alignment and feature parity.

**Architecture:** `causal_press_features.py` owns offline extraction and preset metadata. `AispectCausalPressFeatureBuilder` owns matching runtime extraction. The trainer serializes `featureContract`; the classifier dispatches by that value; legacy feature lookup remains isolated for old model assets.

**Tech Stack:** Python standard library, existing PyTorch trainer, Java, Android `org.json`, JUnit, `unittest`.

---

### Task 1: Fixture and Python RED Tests

**Files:**
- Create: `android/aispect-core/src/test/resources/causal_press_features_v1_fixture.json`
- Create: `android/analysis/tests/test_causal_press_features.py`

- [x] **Step 1: Add source values and expected vectors**

The fixture contains touch receipt times `100` and `300`, IMU receipt times `200` and `350`, per-frame view dimensions, distinct contact values, and expected raw 11/20 channel vectors. The second touch differs enough that nearest-neighbor selection would change the first vector.

- [x] **Step 2: Write failing tests**

```python
def test_zero_order_hold_uses_only_latest_received_touch_frame(self):
    actual = causal.build_matrix(fixture, "causal_imu11_touch9_v1", [0, 1], "error")
    self.assertEqual(fixture["expected"]["causal_imu11_touch9_v1"], actual)

def test_future_touch_and_lift_metadata_cannot_change_causal_matrix(self):
    self.assertEqual(baseline_matrix, mutated_matrix)
```

- [x] **Step 3: Run RED**

Run: `python -m unittest android.analysis.tests.test_causal_press_features`

Expected: FAIL because `causal_press_features` does not exist.

### Task 2: Python Builder and Trainer

**Files:**
- Create: `android/analysis/causal_press_features.py`
- Modify: `android/analysis/train_android_finger_four_class_cnn.py`
- Modify: `android/analysis/tests/test_train_android_finger_four_class_cnn.py`

- [x] **Step 1: Implement the builder contract**

```python
CAUSAL_FEATURE_PRESETS = {
    "causal_imu11_v1": CAUSAL_IMU11_V1,
    "causal_imu11_touch9_v1": CAUSAL_IMU11_TOUCH9_V1,
}

def build_matrix(data, preset, frame_indices, missing_frame_policy):
    return [
        feature_vector(frame, latest_received_touch_frame(touches, frame["receivedElapsedRealtimeNanos"]), preset)
        for frame in requested_imu_frames(data, frame_indices, missing_frame_policy)
    ]
```

Missing IMU receipt time raises `causal_v1_missing_imu_received_timestamp`; missing view dimensions for an aligned touch raises `causal_v1_missing_touch_view_size`.

- [x] **Step 2: Run builder tests GREEN**

Run: `python -m unittest android.analysis.tests.test_causal_press_features`

Expected: PASS.

- [x] **Step 3: Write failing trainer contract tests**

```python
def test_down_anchor_trainer_rejects_legacy_feature_preset(self):
    with self.assertRaisesRegex(RuntimeError, "causal_press_feature_preset_required"):
        trainer.validate_cli_args(args_for_old13)

def test_scaler_metadata_records_causal_feature_contract(self):
    self.assertEqual("causal_imu11_touch9_v1", scaler["featureContract"])
```

- [x] **Step 4: Integrate causal presets**

Make `causal_imu11_touch9_v1` the default; reject legacy/custom down-anchor inputs; route sample construction through the new builder; add `causalPressFeaturesV1` to cache signatures; export `featureContract` in scaler and report.

- [x] **Step 5: Run Python GREEN**

Run: `python -m unittest android.analysis.tests.test_causal_press_features android.analysis.tests.test_train_android_finger_four_class_cnn`

Expected: PASS.

### Task 3: Android Builder and Schema RED/GREEN

**Files:**
- Create: `android/aispect-core/src/test/java/com/zhifaios/eyes/aispect/AispectCausalPressFeatureBuilderTest.java`
- Modify: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectTouchFrame.java`
- Create: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectCausalPressFeatureBuilder.java`

- [x] **Step 1: Write failing schema and fixture-parity tests**

```java
assertEquals(1080, frame.toJson(0).getInt("viewWidthPx"));
assertEquals(2400, frame.toJson(0).getInt("viewHeightPx"));
assertArrayEquals(expectedVector, builder.build(frames, -5, frameIndices, touchFrames, "causal_imu11_touch9_v1")[0], 1e-6);
```

- [x] **Step 2: Run Android RED**

Run: `java -classpath gradle/wrapper/gradle-wrapper.jar org.gradle.wrapper.GradleWrapperMain :aispect-core:compileDebugUnitTestJavaWithJavac --no-daemon`

Expected: failure because JSON dimensions and builder are absent.

- [x] **Step 3: Implement Java contract**

Serialize dimensions. Use only receipt timestamps and raw per-frame fields. Preserve list order for equal timestamps. Do not reference aggregate contact, normalization, confidence, event time, lift state, or time-since fields.

- [x] **Step 4: Run Android GREEN**

Run: the compiled test class directly with the Android SDK, the real `org.json` JAR, JUnit, and Hamcrest on the classpath.

Expected: causal builder and raw frame schema tests pass.

### Task 4: Runtime Dispatch and Documentation

**Files:**
- Modify: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectImpactCNNClassifier.java`
- Modify: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectCollectionController.java`
- Modify: `FunctionAndDataStructureDescription.md`

- [x] **Step 1: Write failing causal-model dispatch test**

```java
assertTrue(classifier.usesCausalPressFeatureContract("causal_imu11_touch9_v1"));
```

- [x] **Step 2: Dispatch by scaler `featureContract`**

Read optional scaler contract. Causal models require raw touch frames and call the Java builder; old models retain legacy lookup. The controller passes active touch frames only for a ready causal model window.

- [x] **Step 3: Update documentation and final checks**

Run: `python -m unittest android.analysis.tests.test_causal_press_features android.analysis.tests.test_train_android_finger_four_class_cnn android.analysis.tests.test_latest_model_deployment`

Run: `python -m py_compile android/analysis/causal_press_features.py android/analysis/train_android_finger_four_class_cnn.py`

Expected: all tests pass and Python compilation exits `0`.

No commit is created because the user has an active shared dirty worktree with concurrent schema-v6 changes.
