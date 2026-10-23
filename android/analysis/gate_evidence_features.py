"""Causal post-touch evidence features for the light/heavy gate."""

import numpy as np


TIME_GRID_MS = (-15, -10, -5, 0, 5, 10, 15, 20, 25)
DEFAULT_EVIDENCE_CONTRACT = "legacy_post20"
EVIDENCE_CONTRACTS = {
    # 中文注释：该合同完整复现旧报告，仅用于 A/B 对照。
    "legacy_post20": {
        "pre_stop": 7,
        "post_start": 7,
        "gravity_pre_stop": 7,
        "gravity_endpoint": (6, 8),
        "touch_delta_mode": "legacy_adjacent",
        "touch_anchor": 6,
        "touch_end": 7,
    },
    # 中文注释：0 ms 槽位即按下时刻，前 3 帧才是真正的按下前观测。
    "touchdown_post0": {
        "pre_stop": 3,
        "post_start": 3,
        "gravity_pre_stop": 3,
        "gravity_endpoint": (3, 8),
        "touch_delta_mode": "anchor_to_end",
        "touch_anchor": 3,
        "touch_end": 8,
    },
    "impact_post5": {
        "pre_stop": 3,
        "post_start": 4,
        "gravity_pre_stop": 3,
        "gravity_endpoint": (3, 8),
        "touch_delta_mode": "anchor_to_end",
        "touch_anchor": 3,
        "touch_end": 8,
    },
    "impact_post10": {
        "pre_stop": 3,
        "post_start": 5,
        "gravity_pre_stop": 3,
        "gravity_endpoint": (3, 8),
        "touch_delta_mode": "anchor_to_end",
        "touch_anchor": 3,
        "touch_end": 8,
    },
}
EPSILON = 1e-5


def feature_names():
    """Freeze the order used by training and future Android inference."""
    names = []
    for signal in ("acceleration", "rotation"):
        names.extend((
            signal + ".pre_mean",
            signal + ".pre_std",
            signal + ".pre_max",
            signal + ".post_mean",
            signal + ".post_std",
            signal + ".post_max",
            signal + ".mean_delta",
            signal + ".max_delta",
            signal + ".robust_mean_z",
            signal + ".robust_max_z",
            signal + ".mean_ratio",
            signal + ".max_ratio",
        ))
        for axis in ("x", "y", "z"):
            names.extend((
                signal + "." + axis + ".pre_mean",
                signal + "." + axis + ".pre_std",
                signal + "." + axis + ".post_mean",
                signal + "." + axis + ".post_std",
                signal + "." + axis + ".mean_delta",
            ))
    names.extend((
        "gravity.mean_delta",
        "gravity.max_delta",
        "gravity.endpoint_delta",
        "shape.post_size_mean",
        "shape.post_axis_ratio_mean",
        "shape.availability",
        "shape.size_ratio",
        "shape.axis_ratio_delta",
        "touch.x_delta",
        "touch.y_delta",
        "touch.xy_delta",
    ))
    return tuple(names)


FEATURE_NAMES = feature_names()


def _mean(values):
    """Return a zero for an empty optional channel."""
    return float(values.mean()) if len(values) else 0.0


def _std(values):
    """Return a zero for an unavailable optional channel."""
    return float(values.std()) if len(values) > 1 else 0.0


def _max(values):
    """Return a zero for an empty optional channel."""
    return float(values.max()) if len(values) else 0.0


def _signal_features(pre, post):
    """Summarize one vector signal without future frames."""
    pre_magnitude = np.linalg.norm(pre, axis=1)
    post_magnitude = np.linalg.norm(post, axis=1)
    center = float(np.median(pre_magnitude))
    mad = float(1.4826 * np.median(np.abs(pre_magnitude - center)))
    scale = max(mad, EPSILON)
    pre_mean = _mean(pre_magnitude)
    post_mean = _mean(post_magnitude)
    pre_max = _max(pre_magnitude)
    post_max = _max(post_magnitude)
    result = [
        pre_mean,
        _std(pre_magnitude),
        pre_max,
        post_mean,
        _std(post_magnitude),
        post_max,
        post_mean - pre_mean,
        post_max - pre_max,
        (post_mean - center) / scale,
        (post_max - center) / scale,
        (post_mean + EPSILON) / (center + EPSILON),
        (post_max + EPSILON) / (pre_max + EPSILON),
    ]
    pre_vector_mean = pre.mean(axis=0)
    post_vector_mean = post.mean(axis=0)
    for axis in range(3):
        result.extend((
            float(pre_vector_mean[axis]),
            float(pre[:, axis].std()),
            float(post_vector_mean[axis]),
            float(post[:, axis].std()),
            float(post_vector_mean[axis] - pre_vector_mean[axis]),
        ))
    return result


def contract_metadata(name):
    """Return a copy of one versioned evidence-window contract."""
    if name not in EVIDENCE_CONTRACTS:
        raise RuntimeError("unsupported_evidence_contract")
    contract = dict(EVIDENCE_CONTRACTS[name])
    contract["timeGridMs"] = list(TIME_GRID_MS)
    contract["name"] = name
    return contract


def _touch_delta(raw, matrix, contract):
    """Measure displacement with an explicit, contract-owned anchor."""
    if contract["touch_delta_mode"] == "legacy_adjacent":
        anchor = contract["touch_anchor"]
        end = contract["touch_end"]
        x_delta = float(raw[end, 0] - raw[anchor, 0])
        y_delta = float(raw[end, 1] - raw[anchor, 1])
        return x_delta, y_delta
    touch_valid = matrix[:, 19] >= 0.5
    start = contract["touch_anchor"]
    stop = contract["touch_end"] + 1
    valid_indices = np.flatnonzero(touch_valid[start:stop]) + start
    if len(valid_indices) < 2:
        return 0.0, 0.0
    anchor = int(valid_indices[0])
    end = int(valid_indices[-1])
    x_delta = float(raw[end, 0] - raw[anchor, 0])
    y_delta = float(raw[end, 1] - raw[anchor, 1])
    return x_delta, y_delta


def extract_one(
        raw,
        row,
        evidence_contract=DEFAULT_EVIDENCE_CONTRACT,
):
    """Extract one sample with a versioned causal evidence window."""
    raw = np.asarray(raw, dtype=np.float32)
    matrix = np.asarray(row["matrix"], dtype=np.float32)
    contract = contract_metadata(evidence_contract)
    if raw.shape[0] != 9 or raw.shape[1] < 10:
        raise RuntimeError("invalid_evidence_sequence")
    pre_stop = contract["pre_stop"]
    post_start = contract["post_start"]
    pre = raw[:pre_stop]
    post = raw[post_start:]
    values = _signal_features(pre[:, 2:5], post[:, 2:5])
    values.extend(_signal_features(pre[:, 5:8], post[:, 5:8]))

    gravity = matrix[:, 8:11]
    gravity_pre = gravity[:contract["gravity_pre_stop"]]
    gravity_valid = (
        np.isfinite(gravity_pre).all(axis=1)
        & (np.linalg.norm(gravity_pre, axis=1) >= 1e-8)
    )
    if np.any(gravity_valid):
        gravity_reference = np.median(
            gravity_pre[gravity_valid],
            axis=0,
            keepdims=True,
        )
        gravity_delta = np.linalg.norm(
            gravity[post_start:] - gravity_reference,
            axis=1,
        )
    else:
        gravity_delta = np.zeros(len(gravity) - post_start)
    gravity_start, gravity_end = contract["gravity_endpoint"]
    values.extend((
        _mean(gravity_delta),
        _max(gravity_delta),
        float(np.linalg.norm(
            gravity[gravity_end] - gravity[gravity_start]
        )),
    ))

    size_available = matrix[:, 22] >= 0.5
    ratio_available = (
        (matrix[:, 23] >= 0.5)
        & (matrix[:, 24] >= 0.5)
    )
    post_size = matrix[post_start:, 12][
        size_available[post_start:]
    ]
    post_ratio = matrix[post_start:, 15][
        ratio_available[post_start:]
    ]
    pre_size = matrix[:pre_stop, 12][
        size_available[:pre_stop]
    ]
    pre_ratio = matrix[:pre_stop, 15][
        ratio_available[:pre_stop]
    ]
    post_size_mean = _mean(post_size)
    pre_size_mean = _mean(pre_size)
    values.extend((
        post_size_mean,
        _mean(post_ratio),
        float(size_available[post_start:].mean()),
        (
            post_size_mean / (pre_size_mean + EPSILON)
            if pre_size.size
            else 0.0
        ),
        _mean(post_ratio) - _mean(pre_ratio),
    ))

    x_delta, y_delta = _touch_delta(raw, matrix, contract)
    values.extend((
        x_delta,
        y_delta,
        float(np.hypot(x_delta, y_delta)),
    ))
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(FEATURE_NAMES),):
        raise RuntimeError("invalid_evidence_feature_count")
    return result


def extract(
        raw_values,
        rows,
        evidence_contract=DEFAULT_EVIDENCE_CONTRACT,
):
    """Extract a matrix while keeping every feature finite."""
    raw_values = np.asarray(raw_values, dtype=np.float32)
    if raw_values.shape[0] != len(rows):
        raise RuntimeError("evidence_row_count_mismatch")
    return np.stack([
        extract_one(raw_values[index], row, evidence_contract)
        for index, row in enumerate(rows)
    ], axis=0)


def apply_feature_mode(values, mode):
    """Apply a channel-family ablation to evidence features."""
    result = np.asarray(values, dtype=np.float32).copy()
    if mode == "full":
        return result
    if mode == "magnitude_only":
        # 中文注释：证据特征中每个信号的模量统计位于轴向特征之前，
        # 这里只保留两段模量统计，避免把设备坐标轴或形状作为捷径。
        retained = np.asarray([
            (
                name.startswith("acceleration.")
                or name.startswith("rotation.")
            )
            and not any(
                name.startswith("acceleration." + axis + ".")
                or name.startswith("rotation." + axis + ".")
                for axis in ("x", "y", "z")
            )
            for name in FEATURE_NAMES
        ])
        result[:, ~retained] = 0.0
        return result
    if mode == "compact":
        excluded = ("shape.", "touch.")
    elif mode == "no_shape":
        excluded = ("shape.",)
    elif mode == "no_absolute_xy":
        excluded = ("touch.",)
    elif mode == "motion_only":
        excluded = ("shape.", "touch.")
    elif mode == "shape_only":
        retained = np.asarray([
            name.startswith("shape.") for name in FEATURE_NAMES
        ])
        result[:, ~retained] = 0.0
        return result
    elif mode == "size_only":
        retained = np.asarray([
            name in (
                "shape.post_size_mean",
                "shape.size_ratio",
                "shape.availability",
            )
            for name in FEATURE_NAMES
        ])
        result[:, ~retained] = 0.0
        return result
    elif mode == "axis_ratio_only":
        retained = np.asarray([
            name in (
                "shape.post_axis_ratio_mean",
                "shape.axis_ratio_delta",
                "shape.availability",
            )
            for name in FEATURE_NAMES
        ])
        result[:, ~retained] = 0.0
        return result
    elif mode in ("motion_size", "motion_axis_ratio"):
        shape_prefix = (
            "shape.size_"
            if mode == "motion_size"
            else "shape.axis_ratio_"
        )
        retained = np.asarray([
            (
                (
                    name.startswith("acceleration.")
                    or name.startswith("rotation.")
                )
                and not any(
                    name.startswith(
                        "acceleration." + axis + "."
                    )
                    or name.startswith(
                        "rotation." + axis + "."
                    )
                    for axis in ("x", "y", "z")
                )
            )
            or name.startswith(shape_prefix)
            or (
                mode == "motion_size"
                and name == "shape.post_size_mean"
            )
            or (
                mode == "motion_axis_ratio"
                and name == "shape.post_axis_ratio_mean"
            )
            or name == "shape.availability"
            for name in FEATURE_NAMES
        ])
        result[:, ~retained] = 0.0
        return result
    else:
        raise RuntimeError("unsupported_evidence_feature_mode")
    retained = np.asarray([
        not name.startswith(excluded) for name in FEATURE_NAMES
    ])
    result[:, ~retained] = 0.0
    return result


def normalize(fit_values, values):
    """Normalize with statistics fitted only on the training split."""
    fit_values = np.asarray(fit_values, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    center = np.median(fit_values, axis=0)
    low = np.percentile(fit_values, 25, axis=0)
    high = np.percentile(fit_values, 75, axis=0)
    scale = high - low
    scale[scale < 1e-6] = 1.0
    return np.clip(
        (values - center) / scale,
        -8.0,
        8.0,
    ).astype(np.float32)
