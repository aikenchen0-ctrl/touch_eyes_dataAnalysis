# Deduplicated Clean V3 Training Design

## Scope

Rebuild cleaning, preprocessing, strict validation manifests, and four-class CNN candidate training from the immutable `data/deduplicated_raw/` dataset. Keep every existing dataset, model, Android asset, and report unchanged.

## Dataset Contract

The source of truth is `data/deduplicated_raw/manifest.csv` plus its 89,385 referenced JSON files. The derived catalog records the sample ID, class label, collector, device, date, collection, session, schema version, quality fields, and raw path. Report JSON files are never treated as samples.

The four supported labels are `thumb_light`, `thumb_heavy`, `index_light`, and `index_heavy`. Samples with invalid JSON, missing IDs, unsupported labels, unusable impact windows, multi-pointer input, or drag/cancel endings are excluded from training but remain untouched in the source directory.

## Context-Aware Quality Policy

Quality thresholds are calculated within device, motion-state, and finger contexts. Impact and travel upper bounds use robust median, MAD, and p95 limits. Touch-duration checks use device, motion, finger, and strength distributions instead of a global four-frame heavy-touch rule.

Geometry fallback, missing radius, unusual duration, impact outliers, and manual-label shape conflicts become review reasons with bounded weights rather than automatic deletion. High-quality samples receive weight `1.0`; review samples receive bounded weights from `0.2` to `0.6`; rejected samples receive `0.0`.

## Session And Redundancy Policy

Use `collectionSessionID` when present. For legacy samples without it, infer sessions within each collector, device, and date collection, starting a new session after a 60-second gap. Temporally thin only within the same session, class, and motion state, retaining the first sample when adjacent timestamps are less than 0.25 seconds apart.

## Validation Manifests

Generate immutable manifests for:

1. Domain holdout: one collector-device combination is validation and every other domain is training.
2. Collection holdout: one collector-device-date collection is validation when all four classes have at least 100 usable samples.
3. Temporal-tail holdout: complete latest sessions from each domain form validation; sessions never cross splits.
4. Random split: diagnostic only and never used for model selection.

## Candidate Training

Stage one runs short probes for:

1. `old13`.
2. `enhanced_patch21_no_confidence`.
3. `enhanced_patch21_no_confidence` with causal `relative_touch` transformation.
4. `enhanced_patch21_no_confidence` with `baseline_imu` transformation, which retains normalized coordinates and converts IMU channels into per-sample deviations from pre-touch median with a bounded noise scale.

Rank candidates by mean strict-holdout balanced accuracy, then worst-fold balanced accuracy, minimum class recall, and runtime channel count. Random validation remains diagnostic.

## Full Training And Deployment Boundary

The top two probe candidates advance to 120 epochs with patience 20 and seeds 22, 47, and 83. A final pooled model may be trained only after strict validation selects a feature definition and epoch budget. New weights and scalers stay under `models/deduplicated_clean_v3_20260728/`; Android assets are not replaced automatically.

The deployment evidence target is mean strict-holdout balanced accuracy at least 0.70, worst fold at least 0.55, and every class recall at least 0.45. Missing the target is a valid experimental result.

## Outputs

- `android/analysis/datasets/deduplicated_clean_v3_20260728/master_manifest.csv`
- `android/analysis/datasets/deduplicated_clean_v3_20260728/training_manifest.csv`
- `android/analysis/datasets/deduplicated_clean_v3_20260728/folds/*.csv`
- `android/analysis/datasets/deduplicated_clean_v3_20260728/summary.json`
- `models/deduplicated_clean_v3_20260728/`
