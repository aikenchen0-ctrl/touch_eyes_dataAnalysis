"""Causal early-press evidence extracted before the current touch grid."""

import math

import numpy as np

try:
    from . import cross_device_causal_features as cross_device
except ImportError:
    import cross_device_causal_features as cross_device


CONTRACT = "early_press_log_ratio_v2"
OFFSETS_MS = (-70, -60, -50, -40, -30, -20, 0, 5, 10)
DENSE_OFFSETS_MS = (
    -70, -65, -60, -55, -50, -45, -40, -35, -30, -25, -20,
    -15, -10, -5, 0, 5, 10,
)
# 中文注释：物理按压在系统 ACTION_DOWN 之前已经开始。
# 训练标签使用 0 至 10 ms 的固定因果窗口；离线审计可单独检查更晚的帧。
TRAINING_OFFSETS_MS = (-70, -60, -50, -40, -30, -20, 0, 5, 10)
BASELINE_END = 3
EARLY_START = 3
EARLY_END = 6
POST_START = 6
POST_END = 8
EPSILON = 0.02
CLIP_MIN = -8.0
CLIP_MAX = 8.0


def feature_names():
    """Freeze the scalar order used by training and Android inference."""
    names = []
    for signal in ("acceleration", "rotation"):
        for offset in (-40, -30, -20, 0, 5, 10):
            names.append(signal + ".log_ratio." + str(offset))
        names.extend((
            signal + ".early_slope",
            signal + ".rise_fraction",
            signal + ".early_peak_log_ratio",
            signal + ".post_peak_log_ratio",
        ))
    names.append("evidence.available")
    for signal in ("acceleration", "rotation"):
        names.extend((
            signal + ".dense_peak_position",
            signal + ".dense_onset_position",
            signal + ".dense_rise_area",
            signal + ".dense_decay_area",
            signal + ".dense_rise_decay_ratio",
            signal + ".dense_jerk_mean",
            signal + ".dense_jerk_max",
            signal + ".dense_energy_center",
            signal + ".dense_peak_to_end_ratio",
        ))
    names.extend((
        "cross.peak_lag",
        "cross.onset_lag",
        "cross.peak_height_ratio",
        "dense_evidence.available",
    ))
    return tuple(names)


FEATURE_NAMES = feature_names()

# 中文注释：消融模式只选择预先冻结的特征列。
# 所有模式共享同一原始证据合同，避免不同实验对时间片或计算公式产生漂移。
EVIDENCE_MODES = (
    "shape",
    "acceleration",
    "rotation",
    "rotation_log_ratio",
    "rotation_peak_shape",
    "rotation_peak_shape_no_gain",
    "acceleration_post",
    "onset_shape",
    "acceleration_shape",
    "rotation_shape",
    "acceleration_log_ratio",
    "log_ratio",
    "acceleration_onset",
    "acceleration_onset_post",
    "dense_shape",
    "dense_acceleration",
    "dense_rotation",
    "dense_cross",
    "dense_shape_no_gain",
    "dense_scale_invariant",
    "dense_scale_invariant_no_cross",
    "rotation_plus_dense_scale_invariant",
)
MODE_FEATURE_INDICES = {
    "shape": tuple(range(20)),
    "acceleration": tuple(range(10)),
    "rotation": tuple(range(10, 20)),
    "rotation_log_ratio": tuple(range(10, 16)),
    "rotation_peak_shape": tuple(range(10, 20)),
    "rotation_peak_shape_no_gain": tuple(range(10, 20)),
    "acceleration_post": (6, 7, 8, 9),
    "onset_shape": (3, 4, 5, 6, 7, 8, 16, 17, 18),
    "acceleration_shape": tuple(range(10)),
    "rotation_shape": tuple(range(10, 20)),
    "acceleration_log_ratio": tuple(range(6)),
    "log_ratio": tuple(range(6)) + tuple(range(10, 16)),
    "acceleration_onset": (0, 1, 2, 6, 7, 8),
    "acceleration_onset_post": (0, 1, 2, 6, 7, 8, 9),
    "dense_shape": tuple(range(21, 42)),
    "dense_acceleration": tuple(range(21, 30)),
    "dense_rotation": tuple(range(30, 39)),
    "dense_cross": (39, 40, 41),
    "dense_shape_no_gain": tuple(range(21, 39)),
    # 中文注释：只保留无量纲时序位置与比例，排除面积、jerk 等
    # 仍会随设备增益和姿态变化的绝对量。
    "dense_scale_invariant": (
        21,
        22,
        25,
        28,
        29,
        30,
        31,
        34,
        37,
        38,
        39,
        40,
        41,
    ),
    "dense_scale_invariant_no_cross": (
        21,
        22,
        25,
        28,
        29,
        30,
        31,
        34,
        37,
        38,
    ),
    "rotation_plus_dense_scale_invariant": (
        *range(10, 20),
        21,
        22,
        25,
        28,
        29,
        30,
        31,
        34,
        37,
        38,
        39,
        40,
        41,
    ),
}


def build_evidence(data, record=None, offsets_ms=None):
    """Extract early evidence from schema v6 raw sensor frames.

    中文注释：-70 至 -50 ms 仅建立本次按压前的局部基线。
    -40 至 +10 ms 保留相对上升，避免把按压起始误当噪声减掉。
    所有值只使用本次触摸开始前已到达的帧，以及固定的因果后续时间片。
    """
    record = record or data
    sensor_metadata = (
        record.get("sensorMetadata")
        or data.get("sensorMetadata")
    )
    if not isinstance(sensor_metadata, dict):
        raise RuntimeError("missing_sensor_metadata")
    cross_device.validate_sensor_metadata(sensor_metadata)
    alignment = data.get("timeAlignment") or record.get("timeAlignment")
    if not isinstance(alignment, dict):
        raise RuntimeError("missing_time_alignment")
    down_ns = cross_device.positive_int(
        alignment.get("downEventElapsedRealtimeNanos")
    )
    if down_ns <= 0:
        raise RuntimeError("missing_down_event_timestamp")
    offsets_ms = tuple(offsets_ms or OFFSETS_MS)
    if tuple(offsets_ms) != OFFSETS_MS:
        raise RuntimeError("unsupported_early_evidence_offsets")
    frames = cross_device.canonical_imu_frames(
        record.get("frames") or [],
        sensor_metadata,
    )
    if len(frames) < 2:
        raise RuntimeError("insufficient_imu_frames_for_early_evidence")
    acceleration = []
    rotation = []
    dense_acceleration = []
    dense_rotation = []
    try:
        for offset in offsets_ms:
            target_ns = down_ns + offset * 1_000_000
            left, right = cross_device.bracket_frame(frames, target_ns)
            values = cross_device.interpolate_imu(left, right, target_ns)
            acceleration.append(math.sqrt(sum(
                values[index] * values[index]
                for index in range(3)
            )))
            rotation.append(math.sqrt(sum(
                values[index] * values[index]
                for index in range(3, 6)
            )))
        for offset in DENSE_OFFSETS_MS:
            target_ns = down_ns + offset * 1_000_000
            left, right = cross_device.bracket_frame(frames, target_ns)
            values = cross_device.interpolate_imu(left, right, target_ns)
            dense_acceleration.append(math.sqrt(sum(
                values[index] * values[index]
                for index in range(3)
            )))
            dense_rotation.append(math.sqrt(sum(
                values[index] * values[index]
                for index in range(3, 6)
            )))
    except RuntimeError as error:
        if str(error).startswith("time_grid_missing_"):
            return unavailable(str(error))
        raise
    acceleration = np.asarray(acceleration, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    dense_acceleration = np.asarray(
        dense_acceleration,
        dtype=np.float64,
    )
    dense_rotation = np.asarray(dense_rotation, dtype=np.float64)
    values = []
    values.extend(log_ratios(acceleration))
    values.extend(log_ratios(rotation))
    values.extend(shape_features(acceleration))
    values.extend(shape_features(rotation))
    values.append(1.0)
    dense_acceleration_values = dense_shape_features(
        dense_acceleration
    )
    dense_rotation_values = dense_shape_features(dense_rotation)
    values.extend(dense_acceleration_values)
    values.extend(dense_rotation_values)
    values.extend(cross_shape_features(
        dense_acceleration_values,
        dense_rotation_values,
    ))
    values.append(1.0)
    return {
        "contract": CONTRACT,
        "offsetsMs": list(OFFSETS_MS),
        "featureNames": list(FEATURE_NAMES),
        "values": values,
    }


def unavailable(reason=None):
    """Return a fully masked row when the source lacks early support."""
    result = {
        "contract": CONTRACT,
        "offsetsMs": list(OFFSETS_MS),
        "featureNames": list(FEATURE_NAMES),
        "values": [0.0] * len(FEATURE_NAMES),
    }
    if reason:
        result["reason"] = reason
    return result


def log_ratios(magnitudes):
    """Return causal relative rise at the fixed comparison offsets."""
    baseline = float(np.median(magnitudes[:BASELINE_END]))
    denominator = baseline + EPSILON
    indices = (3, 4, 5, 6, 7, 8)
    return [
        float(np.clip(
            math.log((float(magnitudes[index]) + EPSILON) / denominator),
            CLIP_MIN,
            CLIP_MAX,
        ))
        for index in indices
    ]


def shape_features(magnitudes):
    """Return scale-free rise timing and strength summaries."""
    baseline = float(np.median(magnitudes[:BASELINE_END]))
    denominator = baseline + EPSILON
    ratios = np.log(
        np.maximum(magnitudes + EPSILON, EPSILON) / denominator
    )
    ratios = np.clip(ratios, CLIP_MIN, CLIP_MAX)
    early = ratios[EARLY_START:EARLY_END]
    post = ratios[POST_START:POST_END]
    early_diffs = np.diff(ratios[BASELINE_END:EARLY_END])
    slope = (float(early[2]) - float(early[0])) / 2.0
    rise_fraction = float(np.mean(early_diffs > 0.0))
    return (
        slope,
        rise_fraction,
        float(np.max(early)),
        float(np.max(post)),
    )


def dense_shape_features(magnitudes):
    """Return scale-free within-touch timing features.

    中文注释：先以按压前 15 ms 的中位数估计本次触摸自身基线，再把
    17 个固定时间片转换为相对对数幅值。输出只比较本次触摸内部的
    上升、峰值和衰减形状，因此不依赖设备绝对增益、屏幕朝向或坐标。
    """
    magnitudes = np.asarray(magnitudes, dtype=np.float64)
    if magnitudes.shape != (len(DENSE_OFFSETS_MS),):
        raise RuntimeError("invalid_dense_evidence_length")
    baseline = float(np.median(magnitudes[:3]))
    ratios = np.clip(
        np.log(
            np.maximum(magnitudes + EPSILON, EPSILON)
            / (baseline + EPSILON)
        ),
        CLIP_MIN,
        CLIP_MAX,
    )
    peak_index = int(np.argmax(ratios))
    peak_position = peak_index / (len(ratios) - 1)
    # 中文注释：onset 定义为第一次超过本次峰值相对增幅一半的位置。
    # 若整段没有上升，位置取 0，后续面积与斜率仍保持有限值。
    threshold = 0.5 * max(float(ratios[peak_index]), 0.0)
    onset_candidates = np.flatnonzero(ratios >= threshold)
    onset_index = (
        int(onset_candidates[0]) if len(onset_candidates) else 0
    )
    onset_position = onset_index / (len(ratios) - 1)
    positive = np.maximum(ratios, 0.0)
    rise_area = float(np.sum(positive[:peak_index + 1]))
    decay_area = float(np.sum(positive[peak_index:]))
    rise_decay_ratio = float(np.clip(
        rise_area / (decay_area + 1e-6),
        0.0,
        100.0,
    ))
    diffs = np.diff(ratios)
    jerk = np.abs(np.diff(diffs)) if len(diffs) > 1 else np.zeros(1)
    energy_total = float(np.sum(positive))
    energy_center = (
        float(np.sum(positive * np.arange(len(ratios))))
        / energy_total
        / (len(ratios) - 1)
        if energy_total > 1e-8
        else 0.0
    )
    peak_to_end_ratio = float(np.clip(
        (ratios[-1] + CLIP_MAX)
        / (float(ratios[peak_index]) + CLIP_MAX + EPSILON),
        0.0,
        2.0,
    ))
    return (
        peak_position,
        onset_position,
        rise_area,
        decay_area,
        rise_decay_ratio,
        float(np.mean(jerk)),
        float(np.max(jerk)),
        energy_center,
        peak_to_end_ratio,
    )


def cross_shape_features(acceleration, rotation):
    """Describe phase and relative peak height without absolute gain."""
    acceleration = np.asarray(acceleration, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    if acceleration.shape != (9,) or rotation.shape != (9,):
        raise RuntimeError("invalid_dense_cross_shape")
    peak_lag = float(acceleration[0] - rotation[0])
    onset_lag = float(acceleration[1] - rotation[1])
    acceleration_peak = max(float(acceleration[0]), 0.0)
    rotation_peak = max(float(rotation[0]), 0.0)
    peak_height_ratio = float(np.clip(
        (rotation_peak + 1e-3) / (acceleration_peak + 1e-3),
        0.0,
        10.0,
    ))
    return (
        peak_lag,
        onset_lag,
        peak_height_ratio,
    )


def selected_feature_names(mode):
    """Return the frozen feature order for one ablation mode."""
    if mode == "rotation_peak_shape":
        return (
            "rotation.peak_shape.-40",
            "rotation.peak_shape.-30",
            "rotation.peak_shape.-20",
            "rotation.peak_shape.0",
            "rotation.peak_shape.5",
            "rotation.peak_shape.10",
            "rotation.rise_fraction",
            "rotation.peak_position",
            "rotation.post_peak_ratio",
            "rotation.early_peak_log_ratio",
            "evidence.available",
        )
    if mode == "rotation_peak_shape_no_gain":
        return (
            "rotation.peak_shape.-40",
            "rotation.peak_shape.-30",
            "rotation.peak_shape.-20",
            "rotation.peak_shape.0",
            "rotation.peak_shape.5",
            "rotation.peak_shape.10",
            "rotation.rise_fraction",
            "rotation.peak_position",
            "rotation.post_peak_ratio",
            "evidence.available",
        )
    indices = MODE_FEATURE_INDICES.get(mode)
    if indices is None:
        raise RuntimeError("unsupported_early_evidence_mode")
    availability_name = (
        "dense_evidence.available"
        if mode.startswith("dense_")
        else "evidence.available"
    )
    return tuple(FEATURE_NAMES[index] for index in indices) + (
        availability_name,
    )


def select_mode(values, mode):
    """Select one ablation channel while preserving availability."""
    values = np.asarray(values, dtype=np.float64)
    if mode in (
        "rotation_peak_shape",
        "rotation_peak_shape_no_gain",
    ):
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise RuntimeError("invalid_early_evidence_width")
        rotation = values[:, 10:20]
        log_ratios = rotation[:, :6]
        # 中文注释：用早期峰值归一化上升曲线，剥离设备整体增益，
        # 只保留本次按压内部的相对时序形状。峰值本身另列保留，
        # 供含增益对照判断该归一化是否真的降低跨设备上尾。
        scale = np.maximum(np.abs(rotation[:, 8]), 0.25)
        peak_shape = np.clip(
            log_ratios / scale[:, None],
            -4.0,
            4.0,
        )
        peak_position = (
            np.argmax(log_ratios, axis=1).astype(np.float64) / 5.0
        )
        post_peak_ratio = np.clip(
            rotation[:, 9] / scale,
            -4.0,
            4.0,
        )
        columns = [
            peak_shape,
            rotation[:, 7:8],
            peak_position[:, None],
            post_peak_ratio[:, None],
        ]
        if mode == "rotation_peak_shape":
            columns.append(rotation[:, 8:9])
        columns.append(values[:, -1:])
        return np.concatenate(columns, axis=1)
    indices = MODE_FEATURE_INDICES.get(mode)
    if indices is None:
        raise RuntimeError("unsupported_early_evidence_mode")
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("invalid_early_evidence_width")
    availability_index = (
        len(FEATURE_NAMES) - 1
        if mode.startswith("dense_")
        else 20
    )
    return np.concatenate(
        (values[:, indices], values[:, availability_index:availability_index + 1]),
        axis=1,
    )


def normalize(fit_values, values, mode="shape"):
    """Fit robust scalars on available training rows and retain the mask."""
    fit_values = np.asarray(fit_values, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if fit_values.ndim != 2 or values.ndim != 2:
        raise RuntimeError("invalid_early_evidence_shape")
    if fit_values.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("invalid_early_evidence_width")
    if values.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("invalid_early_evidence_width")
    fit_values = select_mode(fit_values, mode)
    values = select_mode(values, mode)
    available = fit_values[:, -1] >= 0.5
    if not np.any(available):
        raise RuntimeError("empty_available_early_evidence")
    reference = fit_values[available, :-1]
    center = np.median(reference, axis=0)
    low = np.percentile(reference, 25, axis=0)
    high = np.percentile(reference, 75, axis=0)
    scale = high - low
    scale[scale < 1e-6] = 1.0
    result = np.zeros_like(values, dtype=np.float32)
    value_available = values[:, -1] >= 0.5
    if np.any(value_available):
        result[value_available, :-1] = np.clip(
            (values[value_available, :-1] - center) / scale,
            CLIP_MIN,
            CLIP_MAX,
        )
        result[value_available, -1] = 1.0
    metadata = {
        "contract": CONTRACT,
        "offsetsMs": list(OFFSETS_MS),
        "denseOffsetsMs": list(DENSE_OFFSETS_MS),
        "mode": mode,
        "featureNames": list(selected_feature_names(mode)),
        "fitAvailableCount": int(np.sum(available)),
        "fitMissingCount": int(len(fit_values) - np.sum(available)),
        "availableCount": int(np.sum(value_available)),
        "totalCount": int(len(values)),
    }
    metadata["missingCount"] = (
        metadata["totalCount"] - metadata["availableCount"]
    )
    metadata["missingRate"] = (
        float(metadata["missingCount"]) / metadata["totalCount"]
        if metadata["totalCount"]
        else 0.0
    )
    return result, metadata


def validate(values):
    """Reject malformed evidence rows before model fitting."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("invalid_early_evidence_shape")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("non_finite_early_evidence")
    if np.any((values[:, -1] != 0.0) & (values[:, -1] != 1.0)):
        raise RuntimeError("invalid_early_evidence_availability")
    return values
