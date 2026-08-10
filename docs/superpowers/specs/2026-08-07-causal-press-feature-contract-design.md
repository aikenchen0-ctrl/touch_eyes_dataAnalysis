# Causal Press Feature Contract Design

**Status:** Approved by the user's instruction to plan and execute on 2026-08-07.

## Goal

Replace press-phase input construction with a causal schema-v6 contract. An IMU frame may consume only the final touch sample already received when that IMU frame was received. The contract exposes `causal_imu11_v1` and `causal_imu11_touch9_v1`.

## Current Failure State

The legacy matrix path can read `timeSinceAnchor`, whole-touch `contactPatch`, whole-touch `touchNormalization`, and aggregated contact encodings. Those sources can describe information unavailable at the early press decision frame. In addition, `AispectTouchFrame` has `width` and `height` in memory but does not serialize them, so offline training cannot calculate per-frame view-diagonal normalization.

## Selected Architecture

- Python `android/analysis/causal_press_features.py` builds offline matrices from schema-v6 JSON.
- Java `AispectCausalPressFeatureBuilder` builds matching device matrices from raw `ImpactFrame` and `AispectTouchFrame` inputs.
- `android/aispect-core/src/test/resources/causal_press_features_v1_fixture.json` is the shared behavioral authority. Python and Java tests use the same input and expected vectors.
- A scaler declares `featureContract`. The Android classifier dispatches causal contracts only to the Java causal builder; legacy models retain their historical extractor and cannot claim causal-v1 compatibility.

The language implementations are separate because Python training and Android runtime must be independently deployable. The shared fixture prevents semantic drift.

## Causal Alignment

For IMU receipt time `t_imu`, select the last source-order touch frame satisfying:

```text
touch.receivedElapsedRealtimeNanos <= t_imu
```

Touch frames with non-positive receipt timestamps are ineligible. IMU frames with non-positive receipt timestamps are rejected. The rule never uses nearest-neighbor matching, interpolation, touch event time, lift state, lift time, event-time reconstruction, or frame-rate reconstruction.

When no preceding touch frame exists, IMU values remain available and all touch-derived values are zero with `touch_valid = 0`.

## Feature Contracts

`causal_imu11_v1` has this fixed channel order:

```text
x_norm, y_norm,
userAcceleration.x, userAcceleration.y, userAcceleration.z,
rotationRate.x, rotationRate.y, rotationRate.z,
gravity.x, gravity.y, gravity.z
```

`causal_imu11_touch9_v1` appends:

```text
pressure, size, major_norm, minor_norm, log_axis_ratio,
orientation_sin, orientation_cos, pointer_count, touch_valid
```

`x_norm` and `y_norm` come from the aligned touch frame. `major_norm` and `minor_norm` use that frame's `sqrt(viewWidthPx^2 + viewHeightPx^2)`. Pressure and size are clipped to `[0, 1]`; axes to `[0, diagonal]`; `log_axis_ratio` is zero when either axis is absent, otherwise `log(major/minor)` clipped to `[-4, 4]`. Valid orientation is transformed to sin/cos; absent orientation becomes `0, 0`. Pointer count is clipped to `[0, 10]`.

All channels are standardized only by the model scaler fitted on the training split. The exported scaler remains the runtime normalization authority.

## Hard Exclusions

No causal builder may read or emit `time_since_anchor`, `time_since_touch_down`, `contactPatch.*`, `touchNormalization.*`, `gaussian.*`, `covariance.*`, or `confidence`.

Down-anchor training accepts causal presets only. Existing schema5 data, models, and feature extractors remain historical compatibility artifacts and are not candidates for causal-v1 training.

## Acceptance Criteria

1. Python and Java produce the same fixture vectors and choose the same ZOH touch frame for every IMU frame.
2. Mutating any future touch sample, post-lift sample, lift metadata, top-level contact aggregate, normalization aggregate, Gaussian value, covariance value, confidence, or time feature leaves a causal matrix unchanged.
3. A causal matrix rejects missing receipt timestamps and missing selected-touch view dimensions.
4. The contracts contain exactly 11 and 20 channels and no excluded feature names.
5. New scaler JSON records `featureContract`; Android causal models route to the causal builder.
6. No causal model is trained, installed, catalogued, or made default in this change.
