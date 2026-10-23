"""Robust device normalization for short causal touch windows."""

from collections import defaultdict

import numpy as np


NORMALIZATION_MODES = (
    "none",
    "pretouch_global_iqr",
    "pretouch_device_iqr",
    "pretouch_causal_iqr",
    "pretouch_causal_iqr_shrunk",
    "pretouch_causal_quantile",
)
PRETOUCH_STOP = 3
CALIBRATION_START = 2
CALIBRATION_STOP = 8
DEFAULT_MIN_HISTORY_ROWS = 8
DEFAULT_REFRESH_EVERY_ROWS = 16
DEFAULT_PRIOR_STRENGTH_ROWS = 8.0
MINIMUM_SCALE = 1e-4


def physical_device_key(row):
    """Return a stable identity for physical-device grouping."""
    platform = str(row.get("platform") or "")
    device_name = str(row.get("deviceName") or "")
    if platform and device_name:
        return platform + ":" + device_name
    return platform + ":collector:" + str(
        row.get("collectorDeviceId") or ""
    )


def _calibration_samples(values):
    """Read only pre-touch inertial frames used for gain estimation."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise RuntimeError("invalid_normalization_values")
    if values.shape[1] < PRETOUCH_STOP:
        raise RuntimeError("normalization_missing_pretouch_frames")
    if values.shape[2] < CALIBRATION_STOP:
        raise RuntimeError("normalization_missing_channels")
    samples = values[
        :,
        :PRETOUCH_STOP,
        CALIBRATION_START:CALIBRATION_STOP,
    ]
    samples = samples.reshape(-1, CALIBRATION_STOP - CALIBRATION_START)
    samples = samples[np.isfinite(samples).all(axis=1)]
    if not len(samples):
        raise RuntimeError("normalization_missing_finite_samples")
    return samples


def fit_profile(
        values,
        prior=None,
        prior_strength_rows=0.0,
        quantile=False,
):
    """Fit robust per-channel location and scale from pre-touch frames."""
    samples = _calibration_samples(values)
    center = np.median(samples, axis=0)
    if quantile:
        low = np.percentile(samples, 10, axis=0)
        high = np.percentile(samples, 90, axis=0)
    else:
        low = np.percentile(samples, 25, axis=0)
        high = np.percentile(samples, 75, axis=0)
    scale = np.maximum(high - low, MINIMUM_SCALE)
    sample_count = float(len(samples))
    if prior is not None:
        if prior_strength_rows < 0.0:
            raise RuntimeError("invalid_normalization_prior_strength")
        prior_center = np.asarray(prior["center"], dtype=np.float32)
        prior_scale = np.asarray(prior["scale"], dtype=np.float32)
        if (
            prior_center.shape != center.shape
            or prior_scale.shape != scale.shape
        ):
            raise RuntimeError("normalization_prior_shape_mismatch")
        denominator = sample_count + float(prior_strength_rows)
        if denominator > 0.0:
            center = (
                sample_count * center
                + float(prior_strength_rows) * prior_center
            ) / denominator
            scale = (
                sample_count * scale
                + float(prior_strength_rows) * prior_scale
            ) / denominator
    return {
        "center": center.astype(np.float32),
        "scale": np.maximum(scale, MINIMUM_SCALE).astype(np.float32),
        "sampleCount": int(len(samples)),
    }


def apply_profile(values, profile):
    """Standardize calibrated channels while retaining impact residuals."""
    values = np.asarray(values, dtype=np.float32)
    result = values.copy()
    center = np.asarray(profile["center"], dtype=np.float32)
    scale = np.asarray(profile["scale"], dtype=np.float32)
    expected = CALIBRATION_STOP - CALIBRATION_START
    if center.shape != (expected,) or scale.shape != (expected,):
        raise RuntimeError("normalization_profile_shape_mismatch")
    result[:, :, CALIBRATION_START:CALIBRATION_STOP] = (
        result[:, :, CALIBRATION_START:CALIBRATION_STOP]
        - center[None, None, :]
    ) / scale[None, None, :]
    return result


def apply_global_profile(
        fit_values,
        values,
        quantile=False,
):
    """Fit one source-domain profile and apply it unchanged."""
    profile = fit_profile(fit_values, quantile=quantile)
    return apply_profile(values, profile), {
        "targetCalibration": "source_only",
        "sampleCount": profile["sampleCount"],
        "quantile": bool(quantile),
    }


def _group_indices(rows):
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[physical_device_key(row)].append(index)
    return grouped


def apply_device_profiles(
        values,
        rows,
        quantile=False,
):
    """Fit unlabeled target-device profiles as a diagnostic upper bound."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] != len(rows):
        raise RuntimeError("normalization_row_count_mismatch")
    result = values.copy()
    groups = {}
    for device, indices in _group_indices(rows).items():
        profile = fit_profile(values[indices], quantile=quantile)
        result[indices] = apply_profile(values[indices], profile)
        groups[device] = {
            "sampleCount": profile["sampleCount"],
            "rowCount": len(indices),
        }
    return result, {
        "targetCalibration": "target_unlabeled_diagnostic",
        "groups": groups,
        "quantile": bool(quantile),
    }


def _causal_order(rows, indices):
    """Order rows without consulting class or review labels."""
    return sorted(
        indices,
        key=lambda index: (
            str(rows[index].get("sourceFile") or ""),
            _nonnegative_int(rows[index].get("cycleIndex")),
            _nonnegative_int(rows[index].get("stepIndex")),
            str(rows[index].get("sourceSha256") or ""),
        ),
    )


def _nonnegative_int(value):
    """Treat legacy unknown or invalid sequence markers as missing."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed >= 0 else 0


def apply_causal_device_profiles(
        values,
        rows,
        fallback_profile,
        min_history_rows=DEFAULT_MIN_HISTORY_ROWS,
        refresh_every_rows=DEFAULT_REFRESH_EVERY_ROWS,
        prior_strength_rows=0.0,
        quantile=False,
):
    """Use only prior target touches and fall back during cold start."""
    if min_history_rows <= 0:
        raise RuntimeError("invalid_normalization_history_rows")
    if refresh_every_rows <= 0:
        raise RuntimeError("invalid_normalization_refresh_rows")
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] != len(rows):
        raise RuntimeError("normalization_row_count_mismatch")
    result = values.copy()
    groups = {}
    for device, indices in _group_indices(rows).items():
        ordered = _causal_order(rows, indices)
        profile = fallback_profile
        calibrated_rows = 0
        block_start = 0
        while block_start < len(ordered):
            if block_start >= min_history_rows:
                history = ordered[:block_start]
                profile = fit_profile(
                    values[history],
                    prior=(
                        fallback_profile
                        if prior_strength_rows > 0.0
                        else None
                    ),
                    prior_strength_rows=prior_strength_rows,
                    quantile=quantile,
                )
            block_end = min(
                len(ordered),
                block_start + refresh_every_rows,
            )
            block = ordered[block_start:block_end]
            result[block] = apply_profile(values[block], profile)
            if block_start >= min_history_rows:
                calibrated_rows += len(block)
            block_start = block_end
        groups[device] = {
            "rowCount": len(ordered),
            "calibratedRowCount": calibrated_rows,
            "coldStartRowCount": len(ordered) - calibrated_rows,
            "historyRows": min_history_rows,
            "refreshEveryRows": refresh_every_rows,
        }
    return result, {
        "targetCalibration": "causal_target_history",
        "groups": groups,
        "priorStrengthRows": float(prior_strength_rows),
        "quantile": bool(quantile),
    }


def normalize_values(
        mode,
        fit_values,
        values,
        rows,
        quantile=False,
):
    """Apply one normalization mode and return an audit description."""
    if mode == "none":
        return np.asarray(values, dtype=np.float32).copy(), {
            "mode": mode,
        }
    if mode == "pretouch_global_iqr":
        result, metadata = apply_global_profile(
            fit_values,
            values,
            quantile=False,
        )
    elif mode == "pretouch_device_iqr":
        result, metadata = apply_device_profiles(
            values,
            rows,
            quantile=False,
        )
    else:
        fallback = fit_profile(fit_values)
        if mode == "pretouch_causal_iqr":
            result, metadata = apply_causal_device_profiles(
                values,
                rows,
                fallback,
                quantile=False,
            )
        elif mode == "pretouch_causal_iqr_shrunk":
            result, metadata = apply_causal_device_profiles(
                values,
                rows,
                fallback,
                prior_strength_rows=DEFAULT_PRIOR_STRENGTH_ROWS,
                quantile=False,
            )
        elif mode == "pretouch_causal_quantile":
            result, metadata = apply_causal_device_profiles(
                values,
                rows,
                fallback,
                quantile=True,
            )
        else:
            raise RuntimeError("unsupported_normalization_mode")
    metadata["mode"] = mode
    return result, metadata
