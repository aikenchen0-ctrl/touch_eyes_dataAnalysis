import math


TIME_GRID_OFFSETS_MS = (-15, -10, -5, 0, 5, 10, 15, 20, 25)
CONTRACT_IMU11 = "causal_time_grid_imu11_v2"
CONTRACT_IMU11_TOUCH9 = "causal_time_grid_imu11_touch9_v2"
CONTRACT_IMU11_TOUCH9_MASK6 = "causal_time_grid_imu11_touch9_mask6_v3"
CONTRACT_IMU11_TOUCH9_MASK6_V4 = "causal_time_grid_imu11_touch9_mask6_v4"
CONTRACT_IMU11_TOUCH7_MASK5_V5 = "causal_time_grid_imu11_touch7_mask5_v5"
IMU_FEATURE_NAMES = (
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
)
TOUCH_FEATURE_NAMES = (
    "pressure",
    "size",
    "major_norm",
    "minor_norm",
    "log_axis_ratio",
    "orientation_sin",
    "orientation_cos",
    "pointer_count",
    "touch_valid",
)
TOUCH_MASK6_FEATURE_NAMES = (
    "pressure_available",
    "size_available",
    "major_available",
    "minor_available",
    "orientation_available",
    "pointer_count_available",
)
TOUCH_MASK5_FEATURE_NAMES = (
    "pressure_available",
    "size_available",
    "major_available",
    "minor_available",
    "pointer_count_available",
)
FEATURE_PRESETS = {
    CONTRACT_IMU11: IMU_FEATURE_NAMES,
    CONTRACT_IMU11_TOUCH9: IMU_FEATURE_NAMES + TOUCH_FEATURE_NAMES,
    CONTRACT_IMU11_TOUCH9_MASK6: IMU_FEATURE_NAMES + TOUCH_FEATURE_NAMES + TOUCH_MASK6_FEATURE_NAMES,
    CONTRACT_IMU11_TOUCH9_MASK6_V4: IMU_FEATURE_NAMES + TOUCH_FEATURE_NAMES + TOUCH_MASK6_FEATURE_NAMES,
    # 中文注释：v5 删除姿态角三个通道，避免模型把设备朝向当作力度捷径。
    CONTRACT_IMU11_TOUCH7_MASK5_V5: (
        IMU_FEATURE_NAMES
        + TOUCH_FEATURE_NAMES[:5]
        + TOUCH_FEATURE_NAMES[7:]
        + TOUCH_MASK5_FEATURE_NAMES
    ),
}

ACCELERATION_SCALES = {
    "m/s^2": 1.0,
    "mps2": 1.0,
    "g": 9.80665,
}
ROTATION_SCALES = {
    "rad/s": 1.0,
    "radians_per_second": 1.0,
    "deg/s": math.pi / 180.0,
    "degrees_per_second": math.pi / 180.0,
}


def feature_names(contract):
    try:
        return list(FEATURE_PRESETS[contract])
    except KeyError as error:
        raise RuntimeError("unknown_time_grid_feature_contract") from error


def build_time_grid_matrix(data, record, contract):
    if int_value(data.get("schemaVersion"), 0) < 6:
        raise RuntimeError("time_grid_requires_schema6")
    feature_names(contract)
    sensor_metadata = record.get("sensorMetadata") or data.get("sensorMetadata")
    if not isinstance(sensor_metadata, dict):
        raise RuntimeError("missing_sensor_metadata")
    touch_metadata = record.get("touchMetadata") or data.get("touchMetadata")
    if not isinstance(touch_metadata, dict):
        raise RuntimeError("missing_touch_metadata")
    validate_sensor_metadata(sensor_metadata)
    validate_touch_metadata(touch_metadata)
    canonical_frames = canonical_imu_frames(record.get("frames") or [], sensor_metadata)
    if len(canonical_frames) < 2:
        raise RuntimeError("insufficient_imu_frames_for_time_grid")
    raw_touch_frames = record.get("touchFrames") or []
    touch_frames = []
    for frame in raw_touch_frames:
        if not isinstance(frame, dict):
            raise RuntimeError("invalid_touch_frame")
        if positive_int(frame.get("receivedElapsedRealtimeNanos")) <= 0 or positive_int(frame.get("eventElapsedRealtimeNanos")) <= 0:
            raise RuntimeError("missing_touch_clock")
        touch_frames.append(frame)
    touch_frames.sort(key=lambda frame: frame["receivedElapsedRealtimeNanos"])
    alignment = data.get("timeAlignment") or record.get("timeAlignment")
    if not isinstance(alignment, dict):
        raise RuntimeError("missing_time_alignment")
    down_ns = positive_int(alignment.get("downEventElapsedRealtimeNanos"))
    if down_ns <= 0:
        raise RuntimeError("missing_down_event_timestamp")
    lift_ns = positive_int(alignment.get("liftEventElapsedRealtimeNanos"))
    endpoint_ns = down_ns + max(TIME_GRID_OFFSETS_MS) * 1_000_000
    if lift_ns > 0 and lift_ns < endpoint_ns:
        raise RuntimeError("lift_before_time_grid_endpoint")
    grid_start_ns = down_ns + min(TIME_GRID_OFFSETS_MS) * 1_000_000
    grid_end_ns = down_ns + max(TIME_GRID_OFFSETS_MS) * 1_000_000
    gaps = quality_gaps(canonical_frames, grid_start_ns, grid_end_ns)
    if not gaps:
        raise RuntimeError("insufficient_imu_quality_frames")
    if any(gap <= 0 for gap in gaps):
        raise RuntimeError("non_monotonic_imu_sensor_timestamp")
    median_dt = upper_middle_median(gaps)
    max_gap = max(2 * 5_000_000, 4 * median_dt)
    brackets = []
    for offset_ms in TIME_GRID_OFFSETS_MS:
        target_ns = down_ns + offset_ms * 1_000_000
        left, right = bracket_frame(canonical_frames, target_ns)
        gap = right["sensorTimestampElapsedRealtimeNanos"] - left["sensorTimestampElapsedRealtimeNanos"]
        if left is not right and (gap <= 0 or gap > max_gap):
            raise RuntimeError("time_grid_gap_exceeds_threshold")
        brackets.append((target_ns, left, right))
    # 中文注释：v5 的批量推理必须等九个网格槽的全部 IMU 支撑到齐；
    # 其他合同保留逐槽可用性，确保历史缓存可以复现。
    batch_availability_ns = max(
        max(left["receivedElapsedRealtimeNanos"], right["receivedElapsedRealtimeNanos"])
        for _, left, right in brackets
    )
    rows = []
    missing_masks = []
    for target_ns, left, right in brackets:
        imu_values = interpolate_imu(left, right, target_ns)
        availability_time = (
            batch_availability_ns
            if contract == CONTRACT_IMU11_TOUCH7_MASK5_V5
            else max(left["receivedElapsedRealtimeNanos"], right["receivedElapsedRealtimeNanos"])
        )
        touch = latest_received_touch_frame(touch_frames, availability_time, target_ns)
        touch_values, mask = touch_feature_values(
            touch,
            touch_metadata,
            contract,
        )
        values = [touch_values[0], touch_values[1]] + imu_values
        if contract == CONTRACT_IMU11_TOUCH7_MASK5_V5:
            values.extend(touch_values[2:7])
            values.extend(touch_values[9:])
            values.extend(mask[name] for name in (
                "pressure",
                "size",
                "major",
                "minor",
                "pointerCount",
            ))
        elif contract in (
            CONTRACT_IMU11_TOUCH9,
            CONTRACT_IMU11_TOUCH9_MASK6,
            CONTRACT_IMU11_TOUCH9_MASK6_V4,
        ):
            values.extend(touch_values[2:])
            if contract in (
                CONTRACT_IMU11_TOUCH9_MASK6,
                CONTRACT_IMU11_TOUCH9_MASK6_V4,
            ):
                values.extend(mask[name] for name in (
                    "pressure",
                    "size",
                    "major",
                    "minor",
                    "orientation",
                    "pointerCount",
                ))
        rows.append(values)
        missing_masks.append(mask)
    return {
        "matrix": rows,
        "gridOffsetsMs": list(TIME_GRID_OFFSETS_MS),
        "gridRateHz": 200,
        "medianDtNs": median_dt,
        "maxGapNs": max_gap,
        "gridQuality": {
            "status": "accepted",
            "medianDtNs": median_dt,
            "maxGapNs": max_gap,
            "windowDurationMs": max(TIME_GRID_OFFSETS_MS) - min(TIME_GRID_OFFSETS_MS),
            "sourceSampleRateHz": 1_000_000_000.0 / median_dt,
        },
        "missingMasks": missing_masks,
        "touchAxisRangeAudit": touch_axis_range_audit(touch_frames),
    }


def canonical_imu_frames(frames, metadata):
    validate_sensor_metadata(metadata)
    source = str(metadata.get("sensorSource") or "").lower()
    acceleration_unit = str(metadata.get("accelerationUnit") or "")
    rotation_unit = str(metadata.get("rotationRateUnit") or "")
    if acceleration_unit not in ACCELERATION_SCALES:
        raise RuntimeError("unsupported_acceleration_unit")
    if rotation_unit not in ROTATION_SCALES:
        raise RuntimeError("unsupported_rotation_unit")
    matrix = axis_transform_matrix(metadata)
    output = []
    for frame in frames:
        sensor_time = positive_int(frame.get("sensorTimestampElapsedRealtimeNanos"))
        received_time = positive_int(frame.get("receivedElapsedRealtimeNanos"))
        if sensor_time <= 0 or received_time <= 0:
            raise RuntimeError("missing_imu_clock")
        acceleration = transform_vector(
            matrix,
            [finite(frame.get("x")), finite(frame.get("y")), finite(frame.get("z"))],
        )
        gravity = transform_vector(
            matrix,
            [finite(frame.get("gravityX")), finite(frame.get("gravityY")), finite(frame.get("gravityZ"))],
        )
        acceleration = [value * ACCELERATION_SCALES[acceleration_unit] for value in acceleration]
        gravity = [value * ACCELERATION_SCALES[acceleration_unit] for value in gravity]
        if source in ("accelerometer_with_gravity", "accelerometer"):
            if source == "accelerometer":
                raise RuntimeError("accelerometer_gravity_subtraction_required")
            acceleration = [value - gravity[index] for index, value in enumerate(acceleration)]
        rotation = transform_vector(
            matrix,
            [finite(frame.get("rotationRateX")), finite(frame.get("rotationRateY")), finite(frame.get("rotationRateZ"))],
        )
        rotation = [value * ROTATION_SCALES[rotation_unit] for value in rotation]
        output.append({
            "sensorTimestampElapsedRealtimeNanos": sensor_time,
            "receivedElapsedRealtimeNanos": received_time,
            "acceleration": acceleration,
            "rotation": rotation,
            "gravity": gravity,
        })
    return sorted(output, key=lambda frame: frame["sensorTimestampElapsedRealtimeNanos"])


def validate_sensor_metadata(metadata):
    if not isinstance(metadata, dict):
        raise RuntimeError("missing_sensor_metadata")
    platform = str(metadata.get("platform") or "").lower()
    source = str(metadata.get("sensorSource") or "").lower()
    if platform not in ("android", "ios") or not source:
        raise RuntimeError("incomplete_sensor_metadata")
    if not str(metadata.get("sensorAxisFrame") or ""):
        raise RuntimeError("missing_sensor_axis_frame")
    if not str(metadata.get("transformVersion") or ""):
        raise RuntimeError("missing_transform_version")
    if platform == "ios" and source not in ("user_acceleration", "useracceleration", "linear_acceleration"):
        raise RuntimeError("unsupported_ios_sensor_source")
    if platform == "android" and source not in (
        "linear_acceleration",
        "linear_acceleration_sensor",
        "accelerometer_with_gravity",
        "accelerometer",
    ):
        raise RuntimeError("unsupported_android_sensor_source")
    if source == "accelerometer_with_gravity" and metadata.get("gravityAvailable") is not True:
        raise RuntimeError("gravity_metadata_required")


def upper_middle_median(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("median_requires_values")
    return ordered[len(ordered) // 2]


def quality_gaps(frames, start_ns, end_ns):
    inside = [
        frame for frame in frames
        if start_ns <= frame["sensorTimestampElapsedRealtimeNanos"] <= end_ns
    ]
    before = [
        frame for frame in frames
        if frame["sensorTimestampElapsedRealtimeNanos"] < start_ns
    ]
    after = [
        frame for frame in frames
        if frame["sensorTimestampElapsedRealtimeNanos"] > end_ns
    ]
    selected = ([] if not before else [before[-1]]) + inside + ([] if not after else [after[0]])
    gaps = [
        right["sensorTimestampElapsedRealtimeNanos"] - left["sensorTimestampElapsedRealtimeNanos"]
        for left, right in zip(selected, selected[1:])
    ]
    if any(gap <= 0 for gap in gaps):
        raise RuntimeError("non_monotonic_imu_sensor_timestamp")
    return gaps


def validate_touch_metadata(metadata):
    if not isinstance(metadata, dict):
        raise RuntimeError("missing_touch_metadata")
    if str(metadata.get("touchUnit") or "") not in ("pixels", "points", "millimeters"):
        raise RuntimeError("unsupported_touch_unit")
    if not str(metadata.get("touchTransformVersion") or ""):
        raise RuntimeError("missing_touch_transform_version")


def axis_transform_matrix(metadata):
    frame_name = str(metadata.get("sensorAxisFrame") or "")
    if not frame_name:
        raise RuntimeError("missing_sensor_axis_frame")
    matrix = metadata.get("axisTransformMatrix")
    if frame_name == "canonical_device" and matrix is None:
        return identity_matrix()
    if not isinstance(matrix, list) or len(matrix) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in matrix):
        raise RuntimeError("missing_axis_transform_matrix")
    values = [[finite(value) for value in row] for row in matrix]
    if any(abs(sum(values[row][column] ** 2 for column in range(3)) - 1.0) > 1e-6 for row in range(3)):
        raise RuntimeError("invalid_axis_transform_matrix")
    for left in range(3):
        for right in range(left + 1, 3):
            dot = sum(values[left][column] * values[right][column] for column in range(3))
            if abs(dot) > 1e-6:
                raise RuntimeError("invalid_axis_transform_matrix")
    determinant = (
        values[0][0] * (values[1][1] * values[2][2] - values[1][2] * values[2][1])
        - values[0][1] * (values[1][0] * values[2][2] - values[1][2] * values[2][0])
        + values[0][2] * (values[1][0] * values[2][1] - values[1][1] * values[2][0])
    )
    if abs(abs(determinant) - 1.0) > 1e-6:
        raise RuntimeError("invalid_axis_transform_matrix")
    return values


def transform_vector(matrix, vector):
    return [sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3)]


def identity_matrix():
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def bracket_frame(frames, target_ns):
    left = None
    for frame in frames:
        timestamp = frame["sensorTimestampElapsedRealtimeNanos"]
        if timestamp == target_ns:
            return frame, frame
        if timestamp > target_ns:
            if left is None:
                raise RuntimeError("time_grid_missing_left_support")
            return left, frame
        left = frame
    raise RuntimeError("time_grid_missing_right_support")


def interpolate_imu(left, right, target_ns):
    left_ns = left["sensorTimestampElapsedRealtimeNanos"]
    right_ns = right["sensorTimestampElapsedRealtimeNanos"]
    ratio = 0.0 if right_ns == left_ns else (target_ns - left_ns) / (right_ns - left_ns)
    values = []
    for key in ("acceleration", "rotation", "gravity"):
        for index in range(3):
            values.append(left[key][index] + (right[key][index] - left[key][index]) * ratio)
    return values


def latest_received_touch_frame(frames, receipt_time, target_ns):
    selected = None
    for frame in frames:
        if frame["receivedElapsedRealtimeNanos"] > receipt_time:
            break
        if positive_int(frame.get("eventElapsedRealtimeNanos")) <= target_ns:
            selected = frame
    return selected


def positive_axis_value(frame, key):
    """Return true when an axis contains an observed positive finite value."""
    if key not in frame:
        return False
    # 中文注释：原始 JSON 的数值可能已经解析为浮点数，
    # 因此不能复用只接受字符串的 positive_int。
    value = finite(frame.get(key))
    return math.isfinite(value) and value > 0.0


def touch_feature_values(frame, metadata, contract=CONTRACT_IMU11_TOUCH9):
    if frame is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], {
            "touch": 0,
            "pressure": 0,
            "size": 0,
            "major": 0,
            "minor": 0,
            "orientation": 0,
            "pointerCount": 0,
        }
    unit = str(metadata.get("touchUnit") or "")
    if unit not in ("pixels", "points", "millimeters"):
        raise RuntimeError("unsupported_touch_unit")
    width, height = touch_dimensions(frame, unit)
    diagonal = math.hypot(width, height)
    availability = frame.get("axisAvailability") or {}
    # 中文注释：旧合同保留声明的量程门控，保证历史缓存可复现；
    # v4 才允许使用未声明量程但确为正数的相对尺寸。
    use_positive_shape_without_range = (
        contract in (
            CONTRACT_IMU11_TOUCH9_MASK6_V4,
            CONTRACT_IMU11_TOUCH7_MASK5_V5,
        )
    )
    # 中文注释：v4 只放宽尺寸轴的量程要求，压力仍须有声明量程。
    # 部分驱动会在没有压力量程时返回常量占位值，直接采信会产生伪信号。
    pressure_valid = (
        bool(availability.get("hasPressureRange", True))
        and "pressure" in frame
    )
    if use_positive_shape_without_range:
        size_valid = positive_axis_value(frame, "size")
        major_valid = positive_axis_value(frame, "touchMajor")
        minor_valid = positive_axis_value(frame, "touchMinor")
    else:
        # 中文注释：旧合同保持历史量程门控，避免旧缓存语义被静默改变。
        size_valid = (
            bool(availability.get("hasSizeRange", True))
            and "size" in frame
        )
        major_valid = (
            bool(availability.get("hasTouchMajorRange", True))
            and "touchMajor" in frame
        )
        minor_valid = (
            bool(availability.get("hasTouchMinorRange", True))
            and "touchMinor" in frame
        )
    orientation_valid = bool(availability.get("hasOrientationRange", False)) and "orientation" in frame
    pointer_valid = "pointerCount" in frame and positive_int(frame.get("pointerCount")) > 0
    major = finite(frame.get("touchMajor")) if major_valid else 0.0
    minor = finite(frame.get("touchMinor")) if minor_valid else 0.0
    major_norm = clamp(major / diagonal, 0.0, 1.0) if major_valid and diagonal > 0 else 0.0
    minor_norm = clamp(minor / diagonal, 0.0, 1.0) if minor_valid and diagonal > 0 else 0.0
    log_axis_ratio = math.log(major / minor) if major > 0 and minor > 0 else 0.0
    orientation = finite(frame.get("orientation")) if orientation_valid else 0.0
    x_norm, y_norm = transform_touch_coordinates(
        normalized_touch_coordinate(frame, "x_norm", "x", width),
        normalized_touch_coordinate(frame, "y_norm", "y", height),
        metadata.get("touchCoordinateTransformMatrix"),
    )
    values = [
        x_norm,
        y_norm,
        clamp(finite(frame.get("pressure")), 0.0, 1.0) if pressure_valid else 0.0,
        clamp(finite(frame.get("size")), 0.0, 1.0) if size_valid else 0.0,
        major_norm,
        minor_norm,
        clamp(log_axis_ratio, -4.0, 4.0),
        math.sin(orientation) if orientation_valid else 0.0,
        math.cos(orientation) if orientation_valid else 0.0,
        clamp(float(positive_int(frame.get("pointerCount"))), 0.0, 10.0) if pointer_valid else 0.0,
        1.0,
    ]
    mask = {
        "touch": 1,
        "pressure": int(pressure_valid),
        "size": int(size_valid),
        "major": int(major_valid),
        "minor": int(minor_valid),
        "orientation": int(orientation_valid),
        "pointerCount": int(pointer_valid),
    }
    return values, mask


def touch_axis_range_audit(frames):
    """Summarize declared touch-axis scales without changing model input."""
    names = (
        "touchMajor",
        "touchMinor",
        "toolMajor",
        "toolMinor",
        "pressure",
        "size",
        "orientation",
    )
    result = {
        "frameCount": len(frames),
        "framesWithMotionRanges": 0,
        "ranges": {},
    }
    for name in names:
        maxima = []
        resolutions = []
        ratios = []
        diagonal_ratios = []
        for frame in frames:
            ranges = frame.get("axisMotionRanges")
            if not isinstance(ranges, dict):
                continue
            axis_range = ranges.get(name)
            if not isinstance(axis_range, dict):
                continue
            maximum = finite(axis_range.get("max"))
            minimum = finite(axis_range.get("min"))
            resolution = finite(axis_range.get("resolution"))
            value = finite(frame.get(name))
            if maximum > minimum:
                maxima.append(maximum)
                resolutions.append(max(0.0, resolution))
                ratios.append(
                    clamp(
                        (value - minimum) / (maximum - minimum),
                        0.0,
                        1.0,
                    )
                )
                width = finite(frame.get("viewWidthPx"))
                height = finite(frame.get("viewHeightPx"))
                diagonal = math.hypot(width, height)
                if diagonal > 0.0:
                    diagonal_ratios.append(maximum / diagonal)
        result["ranges"][name] = {
            "declaredMax": max(maxima) if maxima else None,
            "declaredResolution": max(resolutions) if resolutions else None,
            "declaredToViewDiagonalMedian": (
                upper_middle_median(diagonal_ratios)
                if diagonal_ratios
                else None
            ),
            "valueFractionMedian": (
                upper_middle_median(ratios)
                if ratios
                else None
            ),
            "auditedFrameCount": len(ratios),
        }
    result["framesWithMotionRanges"] = sum(
        isinstance(frame.get("axisMotionRanges"), dict)
        and bool(frame.get("axisMotionRanges"))
        for frame in frames
    )
    return result


def normalized_touch_coordinate(frame, normalized_name, raw_name, dimension):
    if normalized_name in frame:
        return clamp(finite(frame.get(normalized_name)), 0.0, 1.0)
    if dimension <= 0:
        return 0.0
    return clamp(finite(frame.get(raw_name)) / dimension, 0.0, 1.0)


def transform_touch_coordinates(x_norm, y_norm, matrix):
    if matrix is None:
        return clamp(x_norm, 0.0, 1.0), clamp(y_norm, 0.0, 1.0)
    if not isinstance(matrix, list) or len(matrix) != 2 or any(not isinstance(row, list) or len(row) != 3 for row in matrix):
        raise RuntimeError("invalid_touch_coordinate_transform_matrix")
    x = finite(matrix[0][0]) * x_norm + finite(matrix[0][1]) * y_norm + finite(matrix[0][2])
    y = finite(matrix[1][0]) * x_norm + finite(matrix[1][1]) * y_norm + finite(matrix[1][2])
    return clamp(x, 0.0, 1.0), clamp(y, 0.0, 1.0)


def touch_dimensions(frame, unit):
    if unit == "pixels":
        width = positive_float(frame.get("viewWidthPx"))
        height = positive_float(frame.get("viewHeightPx"))
    elif unit == "points":
        width = positive_float(frame.get("viewWidthPoints"))
        height = positive_float(frame.get("viewHeightPoints"))
    else:
        width = positive_float(frame.get("viewWidthMm"))
        height = positive_float(frame.get("viewHeightMm"))
    if width <= 0 or height <= 0:
        raise RuntimeError("missing_touch_view_dimensions")
    return width, height


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, finite(value)))


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def positive_int(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 0
    return value if value > 0 else 0


def positive_float(value):
    value = finite(value)
    return value if value > 0 else 0.0


def int_value(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
