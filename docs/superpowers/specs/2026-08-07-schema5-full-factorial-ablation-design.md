# Schema v5 Down-Anchor Full-Factorial Ablation Design

**Status:** Approved by the user's `继续` instruction on 2026-08-07.

## Goal

Complete the reproducible schema v5 press-phase CNN ablation over every defined feature preset and fixed down-anchor window, then validate every feature-window pair against every planned motion-state holdout.

## Scope

- Dataset: `android/analysis/datasets/bbsiak_schema5_down_anchor_20260806`.
- Feature presets: all 12 current four-class CNN presets.
- Down-anchor windows: `pre5_post0`, `pre5_post3`, `pre5_post6`, `pre5_post9`, `pre5_post15`, `pre3_post9`, and `pre0_post9`.
- Strict folds: `handheld_static`, `placed_static`, and `walking`.
- Training contract: `anchor-mode=down`, `missing-frame-policy=error`, `split-strategy=predefined`, seed `22`, batch size `128`, learning rate `0.001`, and weight decay `0.0001`.

The experiment must not import a candidate into Android assets or change the default App model.

## Current State

The existing output root already contains:

- 12 temporal baseline runs at `pre5_post9`.
- 24 additional temporal runs for four screened presets across six non-baseline windows.
- 15 strict motion-holdout runs for five screened feature-window pairs.

These reports are valid completed work and must be reused. The missing work is 48 temporal feature-window pairs and 237 strict fold runs.

## Alternatives Considered

1. Run only the missing temporal pairs and retain the current strict shortlist. This is fast, but it cannot rule out feature-window interactions under the deployment-relevant motion shift.
2. Run every strict fold manually from a command list. This reaches coverage but makes interruption recovery and aggregation error-prone.
3. Add a small resumable analysis runner that enumerates the full matrix, recognizes existing completed reports, runs only missing work, and rewrites summaries after each result. This is the selected approach.

## Design

The runner owns experiment enumeration, command construction, completion checks, and report aggregation. It delegates all feature extraction, normalization, training, and metric calculation to `train_android_finger_four_class_cnn.py`.

Temporal runs use the existing per-session tail manifest with at most 60 epochs and patience 12. Strict runs use the corresponding motion holdout manifest with at most 120 epochs and patience 20. The same sample cache is used for equal preset-window inputs, while the trainer reloads the manifest split before each run.

The runner uses the existing output layout so historical reports are recognized:

- `feature_temporal/<preset>` for the `pre5_post9` baseline.
- `window_temporal/<preset>__<window>` for other temporal windows.
- `motion_holdout/<preset>__<window>/<motion>` for strict folds.

It emits a durable status JSON and separate temporal and strict CSV/JSON summaries. A result is ranked only as an offline candidate; no summary may claim deployment eligibility.

## Acceptance Criteria

1. The enumerated matrix contains 84 temporal pairs and 252 strict runs.
2. Existing reports are skipped only when the expected report, weights, and scaler for the exact model id are present.
3. Every new command uses the locked training contract and a matching cache key.
4. The runner may be stopped and re-run without retraining completed items.
5. The final strict summary contains one aggregate for every feature-window pair, including fold count, mean and worst balanced accuracy, and worst class recall.
6. Unit tests cover matrix coverage, historical output paths, command flags, completion checks, and aggregate ranking.

## Risks and Controls

- Full strict validation is long-running. The runner executes serially and writes progress after each successful job.
- Temporal results are used only as an exploratory view; strict motion holdout determines the final offline ranking.
- Existing data are single collector/device data. Even complete motion holdout results must not be treated as cross-device deployment proof.
