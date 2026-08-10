# Existing-Data Device Invariance Design

## Scope

Use only the existing raw JSON, `cleaned_v2` quality metadata, grouped manifests, and patch21 sample cache. Do not modify Android collection behavior, request new samples, synthesize labels, or overwrite previous datasets and models.

## Problem

Random validation reaches about `0.8377` balanced accuracy while complete-collection holdouts range from about `0.2741` to `0.6728`. LE2110 and PHK110 expose incompatible absolute contact geometry and touch-event timing ranges, so patch21 can identify collection/device style instead of learning a stable four-class boundary.

## Derived Feature Candidates

All candidates start from the cached `enhanced_patch21_no_confidence` matrix and preserve the same `-5...+9` impact window.

1. `imu11_no_time`: retain normalized coordinates and nine acceleration/rotation/gravity channels; remove absolute contact geometry and the two absolute time channels.
2. `invariant_shape`: add only dimensionless contact geometry to `imu11_no_time`: log axis ratio, covariance correlation, and normalized eccentricity. Do not expose absolute radius, covariance magnitude, area, or duration.
3. `relative_touch`: extend `invariant_shape` with log major/minor/contact-area ratios against a rolling median of prior unlabeled samples from the same collection/source group, plus baseline readiness. The rolling window contains at most 256 prior samples and never uses labels or future samples.

## Weighting

Preserve quality weights from `cleaned_v2`. Inside each training fold only, multiply them by an inverse `collection x class` frequency factor normalized to mean one and capped to `[0.25, 4.0]`. Validation samples remain unweighted. This prevents legacy or large collections from dominating without discarding valid rows.

## Validation

Train each candidate on the same four complete-collection holdout manifests with batch size `128`, learning rate `0.001`, weight decay `0.0001`, seed `22`, maximum `120` epochs, and patience `20`. Reuse the signed patch21 cache and the existing CNN architecture.

Primary ranking is mean collection-holdout balanced accuracy. Secondary gates are worst-fold balanced accuracy and worst per-class recall. Random split is diagnostic only and cannot select the final candidate.

Target deployment-quality evidence is mean holdout balanced accuracy at least `0.70` and worst-fold balanced accuracy at least `0.60`. Missing the target is a valid result and must not trigger fabricated data or App model replacement.

## Outputs

- Derived models: `models/all_collections_invariant_v3_20260724/`
- Comparison report: `models/all_collections_invariant_v3_20260724/model_comparison.csv`
- Analysis report: `models/all_collections_invariant_v3_20260724/training_analysis.json`
- Existing raw, `cleaned_v2`, grouped manifests, and prior models remain unchanged.
