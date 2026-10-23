import math

from cross_device_causal_features import CONTRACT_IMU11 as TIME_GRID_CONTRACT_IMU11
from cross_device_causal_features import CONTRACT_IMU11_TOUCH9 as TIME_GRID_CONTRACT_IMU11_TOUCH9
from cross_device_causal_features import CONTRACT_IMU11_TOUCH9_MASK6 as TIME_GRID_CONTRACT_IMU11_TOUCH9_MASK6
from cross_device_causal_features import CONTRACT_IMU11_TOUCH9_MASK6_V4 as TIME_GRID_CONTRACT_IMU11_TOUCH9_MASK6_V4
from cross_device_causal_features import FEATURE_PRESETS as TIME_GRID_FEATURE_PRESETS
from cross_device_causal_features import build_time_grid_matrix


CAUSAL_IMU11_V1 = (
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

CAUSAL_IMU12_V1 = CAUSAL_IMU11_V1 + ("touch_valid",)

CAUSAL_IMU11_TOUCH9_V1 = CAUSAL_IMU11_V1 + (
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

CAUSAL_FEATURE_PRESETS = {
    "causal_imu12_v1": CAUSAL_IMU12_V1,
    "causal_imu11_touch9_v1": CAUSAL_IMU11_TOUCH9_V1,
    TIME_GRID_CONTRACT_IMU11: TIME_GRID_FEATURE_PRESETS[TIME_GRID_CONTRACT_IMU11],
    TIME_GRID_CONTRACT_IMU11_TOUCH9: TIME_GRID_FEATURE_PRESETS[TIME_GRID_CONTRACT_IMU11_TOUCH9],
    TIME_GRID_CONTRACT_IMU11_TOUCH9_MASK6: TIME_GRID_FEATURE_PRESETS[TIME_GRID_CONTRACT_IMU11_TOUCH9_MASK6],
    TIME_GRID_CONTRACT_IMU11_TOUCH9_MASK6_V4: TIME_GRID_FEATURE_PRESETS[TIME_GRID_CONTRACT_IMU11_TOUCH9_MASK6_V4],
}

TIME_GRID_FEATURE_CONTRACTS = {
    TIME_GRID_CONTRACT_IMU11,
    TIME_GRID_CONTRACT_IMU11_TOUCH9,
    TIME_GRID_CONTRACT_IMU11_TOUCH9_MASK6,
    TIME_GRID_CONTRACT_IMU11_TOUCH9_MASK6_V4,
}


def feature_names(preset):
    try:
        return list(CAUSAL_FEATURE_PRESETS[preset])
    except KeyError as error:
        raise RuntimeError("unknown_causal_feature_preset") from error


def is_causal_feature_preset(preset):
    return preset in CAUSAL_FEATURE_PRESETS


def build_matrix(data, preset, frame_indices, missing_frame_policy):
    names = feature_names(preset)
    if int_value(data.get("schemaVersion"), 0) < 6:
        raise RuntimeError("causal_v1_requires_schema6")
    if missing_frame_policy not in ("zero", "skip", "error"):
        raise RuntimeError("unknown_missing_frame_policy")
    if preset in TIME_GRID_FEATURE_CONTRACTS:
        try:
            return build_time_grid_matrix(data, data, preset)["matrix"]
        except RuntimeError:
            if missing_frame_policy == "skip":
                return None
            raise

    reference_index = int_value(data.get("downReferenceFrameIndex"), 0)
    frames_by_index = {
        int_value(frame.get("frameIndex"), 0): frame
        for frame in data.get("frames") or []
        if isinstance(frame, dict)
    }
    touch_frames = sorted(
        (
            (
                positive_int_value(frame.get("receivedElapsedRealtimeNanos")),
                positive_int_value(frame.get("eventElapsedRealtimeNanos")),
                index,
                frame,
            )
            for index, frame in enumerate(data.get("touchFrames") or [])
            if isinstance(frame, dict)
        ),
        key=lambda item: (item[0], item[2]),
    )

    output = []
    for relative_index in frame_indices:
        frame = frames_by_index.get(reference_index + int(relative_index))
        if frame is None:
            if missing_frame_policy == "skip":
                return None
            if missing_frame_policy == "error":
                raise RuntimeError("missing_frame_index")
            output.append([0.0 for _ in names])
            continue
        receipt_time = positive_int_value(frame.get("receivedElapsedRealtimeNanos"))
        if receipt_time <= 0:
            raise RuntimeError("causal_v1_missing_imu_received_timestamp")
        sensor_time = positive_int_value(frame.get("sensorTimestampElapsedRealtimeNanos"))
        if sensor_time <= 0:
            raise RuntimeError("causal_v1_missing_imu_sensor_timestamp")
        touch_frame = latest_received_touch_frame(touch_frames, receipt_time, sensor_time)
        output.append(feature_vector(frame, touch_frame, preset))
    return output


def latest_received_touch_frame(touch_frames, imu_received_nanos, imu_sensor_nanos):
    selected = None
    for receipt_time, event_time, _, frame in touch_frames:
        if receipt_time <= 0 or event_time <= 0:
            raise RuntimeError("causal_v1_missing_touch_timestamp")
        if receipt_time > imu_received_nanos:
            break
        if event_time <= imu_sensor_nanos:
            selected = frame
    return selected


def feature_vector(imu_frame, touch_frame, preset):
    if touch_frame is None:
        touch_values = [0.0, 0.0]
        touch_details = [0.0 for _ in range(9)]
    else:
        touch_values, touch_details = touch_feature_values(touch_frame)
    imu_values = [
        finite_float(imu_frame.get("x")),
        finite_float(imu_frame.get("y")),
        finite_float(imu_frame.get("z")),
        finite_float(imu_frame.get("rotationRateX")),
        finite_float(imu_frame.get("rotationRateY")),
        finite_float(imu_frame.get("rotationRateZ")),
        finite_float(imu_frame.get("gravityX")),
        finite_float(imu_frame.get("gravityY")),
        finite_float(imu_frame.get("gravityZ")),
    ]
    output = touch_values + imu_values
    if preset == "causal_imu12_v1":
        output.append(1.0 if touch_frame is not None else 0.0)
    if preset == "causal_imu11_touch9_v1":
        output.extend(touch_details)
    return output


def touch_feature_values(frame):
    width = positive_int_value(frame.get("viewWidthPx"))
    height = positive_int_value(frame.get("viewHeightPx"))
    if width <= 0 or height <= 0:
        raise RuntimeError("causal_v1_missing_touch_view_size")
    diagonal = math.hypot(width, height)
    major = clamp(finite_float(frame.get("touchMajor")), 0.0, diagonal)
    minor = clamp(finite_float(frame.get("touchMinor")), 0.0, diagonal)
    if major > 0.0 and minor > 0.0:
        log_axis_ratio = clamp(math.log(major / minor), -4.0, 4.0)
    else:
        log_axis_ratio = 0.0
    axis_availability = frame.get("axisAvailability")
    has_orientation = isinstance(axis_availability, dict) and bool(axis_availability.get("hasOrientationRange"))
    orientation = finite_float(frame.get("orientation"))
    if has_orientation:
        orientation_sin = math.sin(orientation)
        orientation_cos = math.cos(orientation)
    else:
        orientation_sin = 0.0
        orientation_cos = 0.0
    return [
        clamp(finite_float(frame.get("x_norm")), 0.0, 1.0),
        clamp(finite_float(frame.get("y_norm")), 0.0, 1.0),
    ], [
        clamp(finite_float(frame.get("pressure")), 0.0, 1.0),
        clamp(finite_float(frame.get("size")), 0.0, 1.0),
        major / diagonal,
        minor / diagonal,
        log_axis_ratio,
        orientation_sin,
        orientation_cos,
        clamp(float(positive_int_value(frame.get("pointerCount"))), 0.0, 10.0),
        1.0,
    ]


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def positive_int_value(value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def int_value(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0
