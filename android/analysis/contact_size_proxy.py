"""Causal, device-relative contact-size proxy features."""

import re
from collections import defaultdict, deque

import numpy as np

try:
    from . import stats_mlp_contract as contract
except ImportError:
    import stats_mlp_contract as contract


MINIMUM_TOUCHES = 8
MAXIMUM_TOUCHES = 64
LOG_RATIO_CLIP = 4.0
DEFAULT_MISSING_VALUE = np.nan
TRAINING_SOURCE_FEATURE_NAMES = tuple(contract.SOURCE_FEATURE_NAMES)


def physical_device_key(row):
    """Build a device key without using a user or class label."""
    platform = str(row.get("platform") or "").strip()
    collector_device_id = str(
        row.get("collectorDeviceId") or ""
    ).strip()
    # 采集器 ID 能区分同型号的不同物理设备，必须优先于型号名。
    if collector_device_id:
        return platform + ":collector:" + collector_device_id
    device_name = str(row.get("deviceName") or "").strip()
    if device_name:
        return platform + ":" + device_name
    raise RuntimeError("missing_physical_device")


def _nonnegative_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return -1
    return parsed if parsed >= 0 else -1


def causal_order_key(row):
    """Order completed touches without consulting labels or split state."""
    timestamp = _nonnegative_int(row.get("sourceTimestampMs"))
    timestamp_key = timestamp if timestamp >= 0 else 2**63 - 1
    source_file_key = tuple(
        (
            0,
            int(token),
        )
        if token.isdigit()
        else (
            1,
            token.casefold(),
        )
        for token in re.split(r"(\d+)", str(row.get("sourceFile") or ""))
        if token
    )
    return (
        timestamp_key,
        source_file_key,
        _nonnegative_int(row.get("cycleIndex")),
        _nonnegative_int(row.get("stepIndex")),
        str(row.get("sourceSha256") or ""),
        str(row.get("sampleId") or ""),
    )


def _index(feature_names, name):
    try:
        return feature_names.index(name)
    except ValueError as error:
        raise RuntimeError("missing_source_feature_" + name) from error


def _valid_positive(values, availability):
    return (
        np.isfinite(values)
        & (values > 0.0)
        & (availability >= 0.5)
    )


def _touch_median(matrix, feature_names):
    size_index = _index(feature_names, "size")
    available_index = _index(feature_names, "size_available")
    size = matrix[:, size_index].astype(np.float64)
    available = _valid_positive(
        size,
        matrix[:, available_index].astype(np.float64),
    )
    if not np.any(available):
        return np.nan
    return float(np.median(size[available]))


def _baseline_median(history):
    if len(history) < MINIMUM_TOUCHES:
        return np.nan
    return float(np.median(np.asarray(history, dtype=np.float64)))


def _absolute_log_axis_ratio(matrix, feature_names):
    major_index = _index(feature_names, "major_norm")
    minor_index = _index(feature_names, "minor_norm")
    major_available_index = _index(
        feature_names,
        "major_available",
    )
    minor_available_index = _index(
        feature_names,
        "minor_available",
    )
    major = matrix[:, major_index].astype(np.float64)
    minor = matrix[:, minor_index].astype(np.float64)
    available = (
        _valid_positive(
            major,
            matrix[:, major_available_index].astype(np.float64),
        )
        & _valid_positive(
            minor,
            matrix[:, minor_available_index].astype(np.float64),
        )
    )
    result = np.full(matrix.shape[0], DEFAULT_MISSING_VALUE, dtype=np.float64)
    result[available] = np.abs(
        np.log(major[available] / minor[available])
    )
    return np.clip(result, -LOG_RATIO_CLIP, LOG_RATIO_CLIP)


def transform_row(row, history):
    """Transform one completed row using history collected before it."""
    matrix = np.asarray(row.get("matrix"), dtype=np.float64)
    feature_names = list(row.get("featureNames") or TRAINING_SOURCE_FEATURE_NAMES)
    if (
        matrix.shape != (
            contract.FRAME_COUNT,
            len(contract.SOURCE_FEATURE_NAMES),
        )
        or feature_names != list(contract.SOURCE_FEATURE_NAMES)
    ):
        raise RuntimeError("invalid_source_matrix")
    size_index = _index(feature_names, "size")
    size_available_index = _index(feature_names, "size_available")
    axis_ratio_index = _index(feature_names, "log_axis_ratio")
    indices = (0, 1, 2, 3, 4, 5, 6, 7, size_index, axis_ratio_index)
    physical = matrix[:, indices].copy()

    baseline = _baseline_median(history)
    size = matrix[:, size_index]
    size_available = _valid_positive(
        size,
        matrix[:, size_available_index],
    )
    relative = np.full(matrix.shape[0], DEFAULT_MISSING_VALUE, dtype=np.float64)
    if np.isfinite(baseline) and baseline > 0.0:
        relative[size_available] = np.clip(
            np.log(size[size_available] / baseline),
            -LOG_RATIO_CLIP,
            LOG_RATIO_CLIP,
        )
    physical[:, 8] = relative
    physical[:, 9] = _absolute_log_axis_ratio(matrix, feature_names)
    return physical.astype(np.float32)


def transform_rows(rows):
    """Return ten-channel features using only strictly earlier touches."""
    histories = defaultdict(lambda: deque(maxlen=MAXIMUM_TOUCHES))
    grouped = defaultdict(list)
    seen_order_keys = set()
    for index, row in enumerate(rows):
        device = physical_device_key(row)
        order_key = causal_order_key(row)
        identity = (device, order_key)
        if identity in seen_order_keys:
            raise RuntimeError("duplicate_causal_touch_order")
        seen_order_keys.add(identity)
        grouped[device].append((order_key, index))

    output = [None] * len(rows)
    for device_rows in grouped.values():
        for _, index in sorted(device_rows):
            row = rows[index]
            history = histories[physical_device_key(row)]
            output[index] = transform_row(row, history)
            matrix = np.asarray(row["matrix"], dtype=np.float64)
            median = _touch_median(matrix, list(row["featureNames"]))
            if np.isfinite(median) and median > 0.0:
                history.append(median)
    return np.stack(output, axis=0)


def fit_robust_scaler(values):
    """Fit median/IQR with NaN excluded from every channel."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3:
        raise RuntimeError("invalid_sequence_values")
    center = np.zeros(values.shape[2], dtype=np.float32)
    scale = np.ones(values.shape[2], dtype=np.float32)
    for channel in range(values.shape[2]):
        finite = values[:, :, channel]
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            continue
        center[channel] = float(np.median(finite))
        low, high = np.percentile(finite, (25, 75))
        delta = float(high - low)
        scale[channel] = delta if delta >= 1e-6 else 1.0
    return center, scale


def transform(values, center, scale):
    """Normalize finite values and preserve missing channels as NaN."""
    values = np.asarray(values, dtype=np.float64)
    return np.clip(
        (values - center[None, None, :]) / scale[None, None, :],
        -8.0,
        8.0,
    ).astype(np.float32)


def _channel_stats(values, channel, exclude_non_finite):
    channel_values = values[:, :, channel]
    if exclude_non_finite:
        channel_values = np.where(
            np.isfinite(channel_values),
            channel_values,
            np.nan,
        )
    result = np.empty((values.shape[0], 5), dtype=np.float64)
    for row_index, row in enumerate(channel_values):
        valid = row[np.isfinite(row)]
        if not len(valid):
            result[row_index] = 0.0
            continue
        result[row_index] = (
            float(valid.mean()),
            float(valid.std()),
            float(valid.min()),
            float(valid.max()),
            float(valid[-1] - valid[0]),
        )
    return result


def stats_features(values):
    """Match Android statistics: finite values only, empty channel is zero."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3:
        raise RuntimeError("invalid_stats_input")
    channel_stats = np.concatenate(
        [
            _channel_stats(values, channel, True)
            for channel in range(values.shape[2])
        ],
        axis=1,
    )
    acceleration = values[:, :, 2:5]
    rotation = values[:, :, 5:8]
    accel_magnitude = np.linalg.norm(acceleration, axis=2)
    rotation_magnitude = np.linalg.norm(rotation, axis=2)
    accel_jerk = np.linalg.norm(np.diff(acceleration, axis=1), axis=2)
    rotation_jerk = np.linalg.norm(np.diff(rotation, axis=1), axis=2)
    derived = np.stack(
        (
            accel_magnitude.mean(axis=1),
            accel_magnitude.std(axis=1),
            accel_magnitude.max(axis=1),
            accel_jerk.mean(axis=1),
            accel_jerk.max(axis=1),
            rotation_magnitude.mean(axis=1),
            rotation_magnitude.std(axis=1),
            rotation_magnitude.max(axis=1),
            rotation_jerk.mean(axis=1),
            rotation_jerk.max(axis=1),
        ),
        axis=1,
    )
    return np.concatenate((channel_stats, derived), axis=1).astype(np.float32)


def export_policy():
    """Return the compact policy frozen into the exported model."""
    return {
        "semantics": "contact_area_proxy",
        "forceMeasurement": False,
        "normalization": "log_causal_device_median",
        "historyScope": "completed_touches_only",
        "aggregation": "within_touch_median_then_cross_touch_median",
        "minimumTouches": MINIMUM_TOUCHES,
        "maximumTouches": MAXIMUM_TOUCHES,
        "logRatioClip": LOG_RATIO_CLIP,
        "missingValuePolicy": "nan_excluded_from_stats",
    }
