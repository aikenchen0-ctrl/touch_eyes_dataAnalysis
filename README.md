# Touch Eyes Data Analysis

This repository contains the Android/iOS touch data collection code, labeled raw dataset archives, trained CNN model assets, and full ablation reports for thumb/index light/heavy touch classification.

## Layout

- `android/`: Android collection app, reusable `aispect-core` module, training and ablation scripts.
- `ZHIFAeyes/` and `ZHIFAeyes.xcodeproj/`: current iOS workspace.
- `data/lemon/`: labeled raw JSON archive for the Xiaomi device/person dataset.
- `data/tao/`: labeled raw JSON archive for the OnePlus device/person dataset.
- `data/manifests/`: full four-class manifest and dataset summary used for training.
- `models/lemon_tao_full4_top5/`: top 5 four-class CNN models from the full lemon + tao ablation.
- `models/reports/`: full CNN ablation CSV/JSON reports.
- `docs/`: GitHub Pages dashboard files.

## Dataset Policy

The raw expanded folders `lemon/`, `tao/`, and `lemon_raw_before_thumb_mark/` are local working copies and are intentionally ignored by Git. Upload uses compressed archives under `data/` to avoid committing tens of thousands of small JSON files.

The full four-class training manifest contains 30,214 labeled JSON samples:

- `lemon`: 17,872 samples
- `tao`: 12,342 samples
- Classes: `thumb_light`, `thumb_heavy`, `index_light`, `index_heavy`
- No quality filtering
- No downsampling
- No confidence channel in the best deployed model

## Current Deployed Model Order

The Android catalog is ordered with the latest full lemon + tao four-class models first:

1. `lemon_tao_full4_enhanced_patch21_no_confidence`
2. `lemon_tao_full4_enhanced_patch22`
3. `lemon_tao_full4_old13_multichannel`
4. `lemon_tao_full4_old13_gaussian_summary`
5. `lemon_tao_full4_old13_raw_radius`

These assets are also embedded in `android/aispect-core/src/main/assets/aispect_impact_models/` so the debug APK can switch models on-device.
