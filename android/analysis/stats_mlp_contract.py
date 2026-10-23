"""Shared contract for the touch26 statistics MLP candidate."""

CLASS_LABELS = (
    "thumb_light",
    "thumb_heavy",
    "index_light",
    "index_heavy",
)
FINGER_LABELS = ("thumb", "index")
STRENGTH_LABELS = ("light", "heavy")
GATE_LABELS = ("light", "heavy")
NEGATIVE_CLASS_LABELS = ("thumb_light", "index_light")
POSITIVE_CLASS_LABELS = ("thumb_heavy", "index_heavy")
PRIMARY_TASK = "strength_gate"
MOTION_LABELS = ("still", "walking")
SUPPORT_LABELS = ("handheld", "placed", "body")
POSTURE_LABELS = (
    "tilted",
    "upright",
    "supine",
    "left_side",
    "right_side",
    "other",
)
DEVICE_POSE_LABELS = (
    "tilted_toward_user",
    "vertical_top_up",
    "vertical_top_down",
    "screen_down",
    "screen_up",
    "natural",
)
CONTEXT_TASK_NAMES = (
    "motion_state",
    "support",
    "body_posture",
    "device_pose",
)
CONTEXT_CLASS_COUNTS = (
    len(MOTION_LABELS),
    len(SUPPORT_LABELS),
    len(POSTURE_LABELS),
    len(DEVICE_POSE_LABELS),
)
CONTEXT_IGNORE_INDEX = -100
CONTEXT_UNKNOWN_VALUES = frozenset((
    "",
    "unknown",
    "unspecified",
    "unconstrained",
    "none",
    "null",
))
MOTION_ALIASES = {
    "still": "still",
    "static": "still",
    "stationary": "still",
    "rest": "still",
    "handheld_static": "still",
    "placed_static": "still",
    "seated_static": "still",
    "body_static": "still",
    "tabletop": "still",
    "walking": "walking",
    "walk": "walking",
    "moving": "walking",
    "motion": "walking",
}
SUPPORT_ALIASES = {
    "handheld": "handheld",
    "hand_held": "handheld",
    "held": "handheld",
    "same_hand": "handheld",
    "other_hand": "handheld",
    "two_hands": "handheld",
    "placed": "placed",
    "tabletop": "placed",
    "hard_surface": "placed",
    "soft_surface": "placed",
    "stand": "placed",
    "stand_support": "placed",
    "desk": "placed",
    "table": "placed",
    "body": "body",
    "body_support": "body",
    "on_body": "body",
    "bed": "body",
}
POSTURE_ALIASES = {
    "tilted": "tilted",
    "tilted_handheld": "tilted",
    "seated": "tilted",
    "seated_natural": "tilted",
    "upright": "upright",
    "vertical": "upright",
    "standing": "upright",
    "supine": "supine",
    "lying_back": "supine",
    "back": "supine",
    "side_left": "left_side",
    "left_side": "left_side",
    "left_lateral": "left_side",
    "lateral_left": "left_side",
    "side_right": "right_side",
    "right_side": "right_side",
    "right_lateral": "right_side",
    "lateral_right": "right_side",
    "other": "other",
}
DEVICE_POSE_ALIASES = {
    "tilted_toward_user": "tilted_toward_user",
    "tilted": "tilted_toward_user",
    "tilted_handheld": "tilted_toward_user",
    "vertical_top_up": "vertical_top_up",
    "upright": "vertical_top_up",
    "top_up": "vertical_top_up",
    "portrait": "vertical_top_up",
    "vertical_top_down": "vertical_top_down",
    "inverted": "vertical_top_down",
    "top_down": "vertical_top_down",
    "upside_down": "vertical_top_down",
    "screen_down": "screen_down",
    "face_down": "screen_down",
    "display_down": "screen_down",
    "supine": "screen_down",
    "screen_up": "screen_up",
    "face_up": "screen_up",
    "display_up": "screen_up",
    "tabletop": "screen_up",
    "flat": "screen_up",
    "natural": "natural",
    "handheld": "natural",
    "walking": "natural",
    "side_left": "natural",
    "side_right": "natural",
    "seated": "natural",
    "seated_natural": "natural",
}
TASK_CONTEXT_ALIASES = {
    "seated_natural": ("still", None, "tilted", "natural"),
    "upright": ("still", None, "upright", "vertical_top_up"),
    "inverted": ("still", None, None, "vertical_top_down"),
    "supine": ("still", None, "supine", "screen_down"),
    "side_left": ("still", None, "left_side", "natural"),
    "side_right": ("still", None, "right_side", "natural"),
    "handheld": ("still", "handheld", None, "natural"),
    "tabletop": ("still", "placed", None, "screen_up"),
    "walking": ("walking", "handheld", None, "natural"),
    "tilted_handheld": ("still", "handheld", "tilted",
                        "tilted_toward_user"),
}
PHYSICAL_FEATURE_NAMES = (
    "x_norm",
    "y_norm",
    "userAcceleration.x",
    "userAcceleration.y",
    "userAcceleration.z",
    "rotationRate.x",
    "rotationRate.y",
    "rotationRate.z",
    "log_causal_size_ratio",
    "log_axis_ratio",
)
SOURCE_FEATURE_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 12, 15)
CONTACT_SIZE_SEMANTICS = "contact_area_proxy"
CAUSAL_SIZE_MINIMUM_TOUCHES = 8
CAUSAL_SIZE_MAXIMUM_TOUCHES = 64
CAUSAL_SIZE_LOG_RATIO_CLIP = 4.0
CONTACT_SIZE_PROXY_POLICY = {
    "semantics": CONTACT_SIZE_SEMANTICS,
    "forceMeasurement": False,
    "normalization": "log_causal_device_median",
    "historyScope": "completed_touches_only",
    "aggregation": "within_touch_median_then_cross_touch_median",
    "minimumTouches": CAUSAL_SIZE_MINIMUM_TOUCHES,
    "maximumTouches": CAUSAL_SIZE_MAXIMUM_TOUCHES,
    "logRatioClip": CAUSAL_SIZE_LOG_RATIO_CLIP,
    "missingValuePolicy": "nan_excluded_from_stats",
}
CONTACT_SIZE_BASELINE_POLICY = {
    "causal": True,
    "currentTouchExcluded": True,
    "deviceScope": "physical_device",
    "historyScope": "completed_touches_only",
    "normalization": "log_causal_device_median",
    "aggregation": "median",
    "withinTouchAggregation": "median",
    "minimumTouches": CAUSAL_SIZE_MINIMUM_TOUCHES,
    "maximumTouches": CAUSAL_SIZE_MAXIMUM_TOUCHES,
    "logRatioClip": CAUSAL_SIZE_LOG_RATIO_CLIP,
    "touchStatistics": [
        "mean",
        "std",
        "min",
        "max",
        "last_delta",
    ],
}
LEGACY_V2_FEATURE_CONTRACT = (
    "causal_time_grid_imu11_touch7_mask5_stats_mlp_v2"
)
LEGACY_V2_SOURCE_FEATURE_NAMES = (
    "x_norm",
    "y_norm",
    "userAcceleration.x",
    "userAcceleration.y",
    "userAcceleration.z",
    "rotationRate.x",
    "rotationRate.y",
    "rotationRate.z",
    "gravity.x",
    "gravity.y",
    "gravity.z",
    "pressure",
    "size",
    "major_norm",
    "minor_norm",
    "log_axis_ratio",
    "orientation_sin",
    "orientation_cos",
    "pointer_count",
    "touch_valid",
    "pressure_available",
    "size_available",
    "major_available",
    "minor_available",
    "orientation_available",
    "pointer_count_available",
)
SOURCE_FEATURE_NAMES = (
    "x_norm",
    "y_norm",
    "userAcceleration.x",
    "userAcceleration.y",
    "userAcceleration.z",
    "rotationRate.x",
    "rotationRate.y",
    "rotationRate.z",
    "gravity.x",
    "gravity.y",
    "gravity.z",
    "pressure",
    "size",
    "major_norm",
    "minor_norm",
    "log_axis_ratio",
    "pointer_count",
    "touch_valid",
    "pressure_available",
    "size_available",
    "major_available",
    "minor_available",
    "pointer_count_available",
)
# 中文注释：源矩阵仍保存原始 size；运行时合同只暴露因果相对尺度，
# 避免把设备相关的触摸半径误当成压力或真实力度。
RUNTIME_SOURCE_FEATURE_NAMES = (
    *SOURCE_FEATURE_NAMES[:12],
    "log_causal_size_ratio",
    *SOURCE_FEATURE_NAMES[13:],
)
LEGACY_V3_FEATURE_CONTRACT = (
    "causal_time_grid_imu11_touch7_mask5_stats_mlp_v3"
)
LEGACY_V3_SOURCE_FEATURE_CONTRACT = (
    "causal_time_grid_imu11_touch7_mask5_v5"
)
LEGACY_V3_PHYSICAL_FEATURE_NAMES = (
    "x_norm",
    "y_norm",
    "userAcceleration.x",
    "userAcceleration.y",
    "userAcceleration.z",
    "rotationRate.x",
    "rotationRate.y",
    "rotationRate.z",
    "size",
    "log_axis_ratio",
)
FEATURE_CONTRACT = "causal_time_grid_imu11_touch7_mask5_stats_mlp_v4"
SOURCE_FEATURE_CONTRACT = "causal_time_grid_imu11_touch7_mask5_v6"
LEGACY_RUNTIME_ARCHITECTURE = "stats_mlp_v1"
RUNTIME_ARCHITECTURE = "stats_mlp_strength_gate_v1"
FRAME_COUNT = 9
STATS_FEATURE_COUNT = 60
STATS_CLIP_MIN = -8.0
STATS_CLIP_MAX = 8.0


def class_strength_target(class_label):
    """Map all light labels to negative and heavy labels to positive."""
    if class_label in NEGATIVE_CLASS_LABELS:
        return GATE_LABELS.index("light")
    if class_label in POSITIVE_CLASS_LABELS:
        return GATE_LABELS.index("heavy")
    raise ValueError("invalid_class_label")


def class_finger_target(class_label):
    """Map a finger-strength label to the auxiliary finger target."""
    if class_label in ("thumb_light", "thumb_heavy"):
        return FINGER_LABELS.index("thumb")
    if class_label in ("index_light", "index_heavy"):
        return FINGER_LABELS.index("index")
    raise ValueError("invalid_class_label")


def _normalize_context_value(value):
    """Normalize metadata without inventing a class for missing values."""
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _context_target(value, labels, aliases):
    """Return a supervised index, or the ignore index for unknown metadata."""
    normalized = _normalize_context_value(value)
    if normalized in CONTEXT_UNKNOWN_VALUES:
        return CONTEXT_IGNORE_INDEX
    canonical = aliases.get(normalized)
    if canonical is None:
        raise ValueError("invalid_context_label")
    return labels.index(canonical)


def _task_context(task_id):
    """Return weak hints only when the collection task explicitly defines them."""
    normalized = _normalize_context_value(task_id)
    if normalized in CONTEXT_UNKNOWN_VALUES:
        return (None, None, None, None)
    if normalized not in TASK_CONTEXT_ALIASES:
        raise ValueError("invalid_context_task")
    return TASK_CONTEXT_ALIASES[normalized]


def context_targets(
        motion_state,
        support,
        body_posture,
        device_pose,
        task_id=None,
):
    """Map direct fields first, then fill only explicit task-level hints."""
    direct_values = (motion_state, support, body_posture, device_pose)
    labels = (
        MOTION_LABELS,
        SUPPORT_LABELS,
        POSTURE_LABELS,
        DEVICE_POSE_LABELS,
    )
    aliases = (
        MOTION_ALIASES,
        SUPPORT_ALIASES,
        POSTURE_ALIASES,
        DEVICE_POSE_ALIASES,
    )
    task_values = _task_context(task_id)
    targets = []
    for value, task_value, group_labels, group_aliases in zip(
        direct_values,
        task_values,
        labels,
        aliases,
    ):
        target = _context_target(value, group_labels, group_aliases)
        if target == CONTEXT_IGNORE_INDEX and task_value is not None:
            target = _context_target(
                task_value,
                group_labels,
                group_aliases,
            )
        targets.append(target)
    return tuple(targets)


def context_targets_for_row(row):
    """Read the training-row metadata contract without mutating the row."""
    if not isinstance(row, dict):
        raise TypeError("invalid_training_row")
    return context_targets(
        row.get("motionState"),
        row.get("support"),
        row.get("bodyPosture"),
        row.get("requestedDevicePose"),
        row.get("taskId"),
    )


def stats_feature_names():
    """Freeze the order used by both training and Android inference."""
    names = []
    for channel in PHYSICAL_FEATURE_NAMES:
        for statistic in ("mean", "std", "min", "max", "last_delta"):
            names.append(channel + "." + statistic)
    names.extend((
        "userAcceleration.magnitude.mean",
        "userAcceleration.magnitude.std",
        "userAcceleration.magnitude.max",
        "userAcceleration.jerk.mean",
        "userAcceleration.jerk.max",
        "rotationRate.magnitude.mean",
        "rotationRate.magnitude.std",
        "rotationRate.magnitude.max",
        "rotationRate.jerk.mean",
        "rotationRate.jerk.max",
    ))
    if len(names) != STATS_FEATURE_COUNT:
        raise RuntimeError("invalid_stats_feature_count")
    return tuple(names)


STATS_FEATURE_NAMES = stats_feature_names()
