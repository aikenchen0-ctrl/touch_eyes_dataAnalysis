# First-Touch Early Classifier Design

## Goal

Use the existing 79,878 Clean V2 samples to classify thumb/index and light/heavy immediately from the `ACTION_DOWN` touch frame, before lift-up.

## Input

Every eligible sample contains a first `touchFrames` row. The model uses ten runtime-reproducible features: normalized x/y, pressure, size, normalized contact sigma x/y, log sigma ratio, log contact area, and orientation sine/cosine. Raw pixel radii are excluded to reduce device-scale leakage.

## Model

Train a small standardized MLP with one hidden ReLU layer and four softmax outputs. Use the existing sample weights and class weighting, first select the epoch on a deterministic random validation split, then fit all eligible rows for the selected epoch. Export one JSON file containing feature order, scaler, weights, biases, and label order.

## Android Runtime

Add an independent first-touch classifier that loads the JSON asset and reproduces the same feature extraction. On `ACTION_DOWN`, classify the newly created first `AispectTouchFrame`, publish one early prediction event, and keep collecting the touch normally. Lift-up inference remains unchanged and replaces the early result with the final impact prediction.

## Product Behavior

The diagnostic panel identifies the early result through its model ID and event state. Early predictions do not increment the existing completed-touch recall counter. Existing model selection and final model assets remain unchanged.

## Verification

Use Python unit tests for feature extraction/export and Java unit tests for feature parity and MLP inference. Compare offline validation metrics with class recall, build the crossDevice APK, install it on the connected phone, and verify that a finger-down event displays the early model before lift-up.
