# Dual-Anchor Raw Capture Design

## Goal

Change Android and iOS dataset collection from a lift-centered fixed window to a redundant raw sequence centered on both touch-down and lift-up:

```text
down-15 ... down ... touch duration ... lift+15
```

Training and on-device inference continue to consume separately derived fixed windows.

## Contract

- The down anchor is the first touch event timestamp.
- The lift anchor is the accepted touch-ending event timestamp.
- The raw sequence starts at the nearest real IMU frame to down, including up to 15 preceding real frames.
- The raw sequence ends after 15 additional real IMU frames following the nearest lift frame.
- The raw sequence is variable length because touch duration is variable.
- Normal completion never inserts synthetic zero frames. Interrupted captures retain available frames and set `windowComplete: false`.
- Each raw frame keeps a frame index relative to down, `timeSinceTouchDown`, and a lift-relative timestamp.
- New Android and iOS samples use schema version 4 and record both anchor timestamps, both reference indices, start/end indices, pre-down count, post-lift count, and completeness.

## Runtime Separation

The existing release-time CNN still receives its own fixed lift-centered `-5...+15` window. The raw dual-anchor sequence is passed only to the data logger. This prevents a variable-length down-centered sequence from being interpreted as the old fixed model input.

The iOS detector starts its sensor stream when dataset collection is enabled and keeps it alive between touches, so the rolling buffer can actually provide down-pre frames. The Android collector already maintains a rolling buffer and increases its post-touch wait to cover low-rate sensor fallback.

## Derived Training Views

Offline scripts select fixed windows by explicit reference metadata rather than assuming frame index zero is always the lift anchor. Existing lift-relative training remains compatible through `referenceFrameIndex`; future down-relative experiments can use `downReferenceFrameIndex`.

## Verification

- Java unit tests verify the dual-anchor window starts at down-15, ends at lift+15, preserves real frames, and marks incomplete post-roll without padding.
- Python regression tests verify training matrices align new schema frames to `referenceFrameIndex`.
- Android library compilation, APK assembly, direct JUnit execution, and all Python analysis tests are required.
- iOS source changes are reviewed against the existing collector lifecycle; an Apple build remains environment-dependent on this Windows workspace.
