# Full Data Clean V2 Android Deployment Design

## Goal

Make `full_data_clean_v2_old13_all_118` immediately testable in the connected crossDevice Android app without removing any existing model.

## Design

Copy the exported scaler and weights into the existing `aispect_impact_models` asset directory. Insert the generated catalog entry at index zero and change only the crossDevice flavor's `DEFAULT_MODEL_ID`; the standard flavor remains unchanged. The existing model selector continues to expose every prior model for A/B testing.

## Verification

The deployment test must require the new model ID as the crossDevice default and first catalog item, with both asset files present. After unit tests and the crossDevice debug build pass, install the APK with ADB, clear only the app process, launch it, and verify the installed package plus current selected model from UI/runtime evidence.

## Constraints

- Do not delete or overwrite prior model assets.
- Do not change the standard flavor default.
- Update `FunctionAndDataStructureDescription.md` with the deployment and installation record.
- Preserve unrelated dirty-worktree changes.
