"""Audit Android schema v6 cross-device capture provenance and fallback rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


REQUIRED_SENSOR_FIELDS = (
    "sensorSource",
    "accelerationStatus",
    "gyroscopeStatus",
    "gravityStatus",
    "rotationStatus",
    "rotationSensorSource",
    "rotationSensorName",
    "accelerationTrainingInput",
    "rotationTrainingInput",
    "streamEpoch",
    "timestampRegressionCount",
)
REQUIRED_TOUCH_FIELDS = (
    "touchUnit",
    "touchCoordinateFrame",
    "touchTransformVersion",
    "apiLevel",
    "displayWidthPx",
    "displayHeightPx",
    "density",
    "xdpi",
    "ydpi",
    "metadataEpoch",
    "windowInsets",
)
REQUIRED_AXIS_SOURCES = (
    "touchMajorSource",
    "touchMinorSource",
    "toolMajorSource",
    "toolMinorSource",
    "pressureSource",
    "sizeSource",
    "orientationSource",
)
AXIS_CONTRACTS = {
    "touchMajor": "touchMajorSource",
    "touchMinor": "touchMinorSource",
    "toolMajor": "toolMajorSource",
    "toolMinor": "toolMinorSource",
    "pressure": "pressureSource",
    "size": "sizeSource",
    "orientation": "orientationSource",
}


def _issue(code: str, detail: str, severity: str = "error") -> Dict[str, str]:
    return {"code": code, "detail": detail, "severity": severity}


def audit_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return a serializable audit result without mutating the input record."""
    issues: List[Dict[str, str]] = []
    if not isinstance(record, dict):
        return {"ok": False, "issues": [_issue("record_not_object", "root value is not a JSON object")], "frameCount": 0}
    schema = record.get("schemaVersion")
    if not isinstance(schema, int) or schema < 6:
        issues.append(_issue("schema_too_old", "schemaVersion must be at least 6"))

    sensor = record.get("sensorMetadata")
    if not isinstance(sensor, dict):
        issues.append(_issue("sensor_metadata_missing", "sensorMetadata is missing"))
        sensor = {}
    for field in REQUIRED_SENSOR_FIELDS:
        if field not in sensor:
            issues.append(_issue("sensor_field_missing", field))
    if not isinstance(sensor.get("timestampRegressionCount"), int) or sensor.get("timestampRegressionCount", 0) != 0:
        issues.append(_issue("sensor_timestamp_regression", "timestampRegressionCount must be zero for an accepted raw record"))
    allowed_acceleration_sources = {
        "unavailable",
        "linear_acceleration",
        "accelerometer_minus_gravity",
        "accelerometer_raw_no_gravity",
    }
    source = sensor.get("sensorSource")
    if source not in allowed_acceleration_sources:
        issues.append(_issue("acceleration_source_unknown", str(source)))
    expected_acceleration_training = source in {"linear_acceleration", "accelerometer_minus_gravity"}
    if sensor.get("accelerationTrainingInput") is not expected_acceleration_training:
        code = "raw_acceleration_training_leak" if source == "accelerometer_raw_no_gravity" else "acceleration_training_flag_invalid"
        issues.append(_issue(code, "accelerationTrainingInput does not match sensorSource"))
    if source == "accelerometer_minus_gravity":
        if sensor.get("gravityAvailable") is not True:
            issues.append(_issue("gravity_metadata_required", "gravity support must be healthy for derived acceleration"))
        if sensor.get("accelerationTrainingInput") is not True:
            issues.append(_issue("derived_acceleration_training_flag_invalid", "derived acceleration must be marked as a training input"))
    if source == "unavailable" and sensor.get("accelerationStatus") not in ("UNAVAILABLE", "WAITING_FIRST_EVENT"):
        issues.append(_issue("acceleration_status_source_mismatch", "unavailable acceleration source has an incompatible status"))
    if source != "unavailable" and sensor.get("accelerationStatus") == "UNAVAILABLE":
        issues.append(_issue("acceleration_status_source_mismatch", "available acceleration source cannot be marked UNAVAILABLE"))
    rotation_frames = sensor.get("rotationFrames", [])
    if not isinstance(rotation_frames, list):
        issues.append(_issue("rotation_frames_invalid", "rotationFrames must be an array"))
        rotation_frames = []
    if sensor.get("rotationTrainingInput", False) is not False:
        issues.append(_issue("rotation_training_leak", "rotation observation stream must not be a training input"))
    allowed_rotation_sources = {
        "unavailable",
        "game_rotation_vector",
        "rotation_vector",
        "geomagnetic_rotation_vector",
    }
    if sensor.get("rotationSensorSource") not in allowed_rotation_sources:
        issues.append(_issue("rotation_source_unknown", str(sensor.get("rotationSensorSource"))))
    allowed_statuses = {"UNAVAILABLE", "WAITING_FIRST_EVENT", "HEALTHY", "DEGRADED", "STALLED", "UNKNOWN"}
    for status_name in ("accelerationStatus", "gyroscopeStatus", "gravityStatus", "rotationStatus"):
        if sensor.get(status_name) not in allowed_statuses:
            issues.append(_issue("sensor_status_unknown", status_name))
    if sensor.get("rotationSensorSource") == "unavailable" and sensor.get("rotationStatus") not in ("UNAVAILABLE", "WAITING_FIRST_EVENT"):
        issues.append(_issue("rotation_status_source_mismatch", "unavailable rotation source has an incompatible status"))
    if sensor.get("rotationSensorSource") != "unavailable" and sensor.get("rotationStatus") == "UNAVAILABLE":
        issues.append(_issue("rotation_status_source_mismatch", "available rotation source cannot be marked UNAVAILABLE"))
    if sensor.get("rotationSensorSource") == "unavailable" and rotation_frames:
        issues.append(_issue("rotation_source_frame_mismatch", "unavailable rotation source cannot have frames"))
    rotation_sources_by_epoch: Dict[int, str] = {}
    previous_rotation_epoch = 0
    last_rotation_source = None
    for index, frame in enumerate(rotation_frames):
        if not isinstance(frame, dict):
            issues.append(_issue("rotation_frame_invalid", f"rotationFrames[{index}]"))
            continue
        if frame.get("trainingInput") is not False:
            issues.append(_issue("rotation_frame_training_leak", f"rotationFrames[{index}]"))
        frame_source = frame.get("sourceType")
        if frame_source not in allowed_rotation_sources or frame_source == "unavailable":
            issues.append(_issue("rotation_source_frame_mismatch", f"rotationFrames[{index}].sourceType"))
        for field in ("sensorTimestampElapsedRealtimeNanos", "receivedElapsedRealtimeNanos"):
            if not isinstance(frame.get(field), int) or frame.get(field) <= 0:
                issues.append(_issue("rotation_timestamp_invalid", f"rotationFrames[{index}].{field}"))
        frame_epoch = frame.get("streamEpoch")
        if not isinstance(frame_epoch, int) or frame_epoch < 1:
            issues.append(_issue("rotation_stream_epoch_invalid", f"rotationFrames[{index}].streamEpoch"))
        else:
            if previous_rotation_epoch and frame_epoch < previous_rotation_epoch:
                issues.append(_issue("rotation_stream_epoch_regression", f"rotationFrames[{index}].streamEpoch"))
            previous_rotation_epoch = frame_epoch
            epoch_source = rotation_sources_by_epoch.get(frame_epoch)
            if epoch_source is not None and epoch_source != frame_source:
                issues.append(_issue("rotation_source_frame_mismatch", f"rotationFrames[{index}] mixes sources in one epoch"))
            elif isinstance(frame_source, str):
                rotation_sources_by_epoch[frame_epoch] = frame_source
                last_rotation_source = frame_source
    if (
        last_rotation_source is not None
        and sensor.get("rotationStatus") not in ("UNAVAILABLE", "WAITING_FIRST_EVENT")
        and last_rotation_source != sensor.get("rotationSensorSource")
    ):
        issues.append(_issue("rotation_source_frame_mismatch", "latest rotation frame does not match active source"))

    capability = record.get("deviceCapability")
    if not isinstance(capability, dict):
        issues.append(_issue("device_capability_missing", "deviceCapability is missing"))
        capability = {}
    for field in (
        "level",
        "touchAxesSemanticallyValid",
        "touchAxesDistinct",
        "toolAxesIndependent",
        "observedFrames",
    ):
        if field not in capability:
            issues.append(_issue("capability_quality_field_missing", f"deviceCapability.{field}"))

    touch = record.get("touchMetadata")
    if not isinstance(touch, dict):
        issues.append(_issue("touch_metadata_missing", "touchMetadata is missing"))
        touch = {}
    for field in REQUIRED_TOUCH_FIELDS:
        if field not in touch:
            issues.append(_issue("touch_field_missing", field))
    root_metadata_epoch = touch.get("metadataEpoch")
    if (
        not isinstance(root_metadata_epoch, int)
        or isinstance(root_metadata_epoch, bool)
        or root_metadata_epoch < 1
    ):
        issues.append(_issue("metadata_epoch_invalid", "touchMetadata.metadataEpoch must be a positive integer"))
    insets = touch.get("windowInsets")
    if not isinstance(insets, dict):
        issues.append(_issue("window_insets_missing", "touchMetadata.windowInsets is missing"))
    else:
        if not isinstance(insets.get("available"), bool):
            issues.append(_issue("window_insets_availability_missing", "windowInsets.available is missing"))
        for field in ("leftPx", "topPx", "rightPx", "bottomPx"):
            if field not in insets or not isinstance(insets.get(field), (int, float)):
                issues.append(_issue("window_insets_field_missing", f"windowInsets.{field}"))
    input_device = touch.get("inputDevice")
    if not isinstance(input_device, dict) or not isinstance(input_device.get("motionRanges"), list):
        issues.append(_issue("motion_ranges_missing", "InputDevice.motionRanges is missing"))
    else:
        for range_index, motion_range in enumerate(input_device["motionRanges"]):
            if not isinstance(motion_range, dict):
                issues.append(_issue("motion_range_invalid", f"touchMetadata.inputDevice.motionRanges[{range_index}]"))
                continue
            for field in ("axis", "source", "min", "max", "range", "flat", "fuzz", "resolution"):
                if field not in motion_range:
                    issues.append(_issue("motion_range_field_missing", f"motionRanges[{range_index}].{field}"))

    frames = record.get("touchFrames")
    if not isinstance(frames, list) or not frames:
        issues.append(_issue("touch_frames_missing", "touchFrames must be a non-empty array"))
        frames = []
    capability_observed_frames = capability.get("observedFrames")
    if not isinstance(capability_observed_frames, int) or isinstance(capability_observed_frames, bool):
        issues.append(_issue(
            "capability_frame_count_invalid",
            "deviceCapability.observedFrames must be an integer",
        ))
    elif capability_observed_frames != len(frames):
        issues.append(_issue(
            "capability_frame_count_mismatch",
            f"deviceCapability observed {capability_observed_frames} frames but record has {len(frames)} touchFrames",
        ))
    imu_frames = record.get("frames")
    if not isinstance(imu_frames, list) or not imu_frames:
        issues.append(_issue("imu_frames_missing", "frames must be a non-empty array"))
        imu_frames = []
    previous_sensor_time = 0
    previous_received_time = 0
    imu_sensor_times: List[int] = []
    imu_sources_by_epoch: Dict[int, str] = {}
    previous_imu_epoch = 0
    last_imu_source = None
    last_imu_epoch = 0
    for index, frame in enumerate(imu_frames):
        if not isinstance(frame, dict):
            issues.append(_issue("imu_frame_invalid", f"frames[{index}]"))
            continue
        sensor_time = frame.get("sensorTimestampElapsedRealtimeNanos")
        received_time = frame.get("receivedElapsedRealtimeNanos")
        if not isinstance(sensor_time, int) or sensor_time <= 0:
            issues.append(_issue("imu_sensor_timestamp_invalid", f"frames[{index}].sensorTimestampElapsedRealtimeNanos"))
        if not isinstance(received_time, int) or received_time <= 0:
            issues.append(_issue("imu_received_timestamp_invalid", f"frames[{index}].receivedElapsedRealtimeNanos"))
        if isinstance(sensor_time, int) and isinstance(received_time, int):
            if sensor_time > received_time:
                issues.append(_issue("imu_sensor_after_receive", f"frames[{index}]"))
            if previous_sensor_time and sensor_time < previous_sensor_time:
                issues.append(_issue("imu_sensor_timestamp_regression", f"frames[{index}]"))
            if previous_received_time and received_time < previous_received_time:
                issues.append(_issue("imu_received_timestamp_regression", f"frames[{index}]"))
            previous_sensor_time = sensor_time
            previous_received_time = received_time
            if sensor_time > 0:
                imu_sensor_times.append(sensor_time)
        frame_source = frame.get("sensorSource")
        if frame_source not in allowed_acceleration_sources:
            issues.append(_issue("imu_frame_source_unknown", f"frames[{index}].sensorSource"))
        frame_training = frame.get("trainingInput")
        expected_frame_training = frame_source in {"linear_acceleration", "accelerometer_minus_gravity"}
        if not isinstance(frame_training, bool):
            issues.append(_issue("imu_training_flag_missing", f"frames[{index}].trainingInput"))
        elif frame_training is not expected_frame_training:
            code = "raw_acceleration_training_leak" if frame_source == "accelerometer_raw_no_gravity" else "imu_training_flag_invalid"
            issues.append(_issue(code, f"frames[{index}].trainingInput"))
        frame_epoch = frame.get("streamEpoch")
        if not isinstance(frame_epoch, int) or frame_epoch < 1:
            issues.append(_issue("imu_stream_epoch_invalid", f"frames[{index}].streamEpoch"))
        else:
            if previous_imu_epoch and frame_epoch < previous_imu_epoch:
                issues.append(_issue("imu_stream_epoch_regression", f"frames[{index}].streamEpoch"))
            previous_imu_epoch = frame_epoch
            epoch_source = imu_sources_by_epoch.get(frame_epoch)
            if epoch_source is not None and epoch_source != frame_source:
                issues.append(_issue("imu_source_epoch_mismatch", f"frames[{index}] mixes sources in one epoch"))
            elif isinstance(frame_source, str):
                imu_sources_by_epoch[frame_epoch] = frame_source
                last_imu_source = frame_source
                last_imu_epoch = frame_epoch
    if (
        last_imu_source is not None
        and sensor.get("streamEpoch") == last_imu_epoch
        and source != last_imu_source
    ):
        issues.append(_issue("imu_source_epoch_mismatch", "latest IMU frame does not match active source"))
    if imu_sensor_times:
        imu_start = min(imu_sensor_times)
        imu_end = max(imu_sensor_times)
        outside_rotation_count = sum(
            1
            for frame in rotation_frames
            if isinstance(frame, dict)
            and isinstance(frame.get("sensorTimestampElapsedRealtimeNanos"), int)
            and not imu_start <= frame["sensorTimestampElapsedRealtimeNanos"] <= imu_end
        )
        if outside_rotation_count:
            issues.append(_issue(
                "rotation_frame_outside_imu_window",
                f"{outside_rotation_count} rotation frames are outside [{imu_start}, {imu_end}]",
            ))
    previous_event_time = 0
    previous_received_time = 0
    touch_axis_pair_count = 0
    inverted_touch_axis_pair_count = 0
    distinct_touch_axis_pair_count = 0
    tool_axis_comparison_count = 0
    tool_axis_alias_count = 0
    pressure_values: List[float] = []
    pressure_has_declared_range = False
    previous_touch_metadata_epoch = 0
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            issues.append(_issue("touch_frame_invalid", f"touchFrames[{index}]"))
            continue
        frame_metadata_epoch = frame.get("metadataEpoch")
        if (
            not isinstance(frame_metadata_epoch, int)
            or isinstance(frame_metadata_epoch, bool)
            or frame_metadata_epoch < 1
        ):
            issues.append(_issue(
                "touch_metadata_epoch_invalid",
                f"touchFrames[{index}].metadataEpoch must be a positive integer",
            ))
        else:
            if index == 0 and isinstance(root_metadata_epoch, int) and frame_metadata_epoch != root_metadata_epoch:
                issues.append(_issue(
                    "touch_metadata_epoch_root_mismatch",
                    "touchMetadata.metadataEpoch must match the first touch frame",
                ))
            if previous_touch_metadata_epoch and frame_metadata_epoch < previous_touch_metadata_epoch:
                issues.append(_issue(
                    "touch_metadata_epoch_regression",
                    f"touchFrames[{index}].metadataEpoch regressed",
                ))
            previous_touch_metadata_epoch = frame_metadata_epoch
        for field in ("eventElapsedRealtimeNanos", "receivedElapsedRealtimeNanos", "pointerId", "actionMasked", "actionIndex"):
            if field not in frame:
                issues.append(_issue("touch_frame_field_missing", f"touchFrames[{index}].{field}"))
        event_time = frame.get("eventElapsedRealtimeNanos")
        received_time = frame.get("receivedElapsedRealtimeNanos")
        if not isinstance(event_time, int) or event_time <= 0:
            issues.append(_issue("touch_event_timestamp_invalid", f"touchFrames[{index}].eventElapsedRealtimeNanos"))
        if not isinstance(received_time, int) or received_time <= 0:
            issues.append(_issue("touch_received_timestamp_invalid", f"touchFrames[{index}].receivedElapsedRealtimeNanos"))
        if isinstance(event_time, int) and isinstance(received_time, int):
            if event_time > received_time:
                issues.append(_issue("touch_event_after_receive", f"touchFrames[{index}]"))
            if previous_event_time and event_time < previous_event_time:
                issues.append(_issue("touch_event_timestamp_regression", f"touchFrames[{index}]"))
            if previous_received_time and received_time < previous_received_time:
                issues.append(_issue("touch_received_timestamp_regression", f"touchFrames[{index}]"))
            previous_event_time = event_time
            previous_received_time = received_time
        axes = frame.get("axisAvailability")
        if not isinstance(axes, dict):
            issues.append(_issue("axis_availability_missing", f"touchFrames[{index}]"))
            continue
        for field in REQUIRED_AXIS_SOURCES:
            if field not in axes:
                issues.append(_issue("axis_source_missing", f"touchFrames[{index}].axisAvailability.{field}"))
        for axis_name, source_name in AXIS_CONTRACTS.items():
            for suffix in ("Observed", "SynthesizedOrUnknown"):
                if f"{axis_name}{suffix}" not in axes:
                    issues.append(_issue("axis_contract_field_missing", f"touchFrames[{index}].axisAvailability.{axis_name}{suffix}"))
            declared_name = f"has{axis_name[0].upper()}{axis_name[1:]}Range"
            if declared_name not in axes:
                issues.append(_issue("axis_contract_field_missing", f"touchFrames[{index}].axisAvailability.{declared_name}"))
        for field in REQUIRED_AXIS_SOURCES:
            value = axes.get(field)
            if isinstance(value, str) and "tool" in value.lower() and field.startswith("touch"):
                issues.append(_issue("tool_axis_overrides_touch", f"touchFrames[{index}].axisAvailability.{field}"))
        for axis_name, source_name in AXIS_CONTRACTS.items():
            observed = axes.get(f"{axis_name}Observed")
            source_value = axes.get(source_name)
            if observed is False and source_value not in ("unavailable", None):
                issues.append(_issue("axis_observed_source_mismatch", f"touchFrames[{index}].axisAvailability.{source_name}"))
        touch_major = frame.get("touchMajor")
        touch_minor = frame.get("touchMinor")
        tool_major = frame.get("toolMajor")
        tool_minor = frame.get("toolMinor")
        if _positive_number(touch_major) and _positive_number(touch_minor):
            touch_axis_pair_count += 1
            if float(touch_major) + 0.0001 < float(touch_minor):
                inverted_touch_axis_pair_count += 1
            if not _same_number(touch_major, touch_minor):
                distinct_touch_axis_pair_count += 1
        if (
            _positive_number(touch_major)
            and _positive_number(touch_minor)
            and _positive_number(tool_major)
            and _positive_number(tool_minor)
        ):
            tool_axis_comparison_count += 1
            if _same_number(touch_major, tool_major) and _same_number(touch_minor, tool_minor):
                tool_axis_alias_count += 1
        pressure = frame.get("pressure")
        if isinstance(pressure, (int, float)):
            pressure_values.append(float(pressure))
        pressure_has_declared_range = pressure_has_declared_range or axes.get("hasPressureRange") is True

    touch_axes_semantically_valid = touch_axis_pair_count > 0 and inverted_touch_axis_pair_count == 0
    touch_axes_distinct = distinct_touch_axis_pair_count > 0
    tool_axes_independent = tool_axis_comparison_count > 0 and tool_axis_alias_count < tool_axis_comparison_count
    if inverted_touch_axis_pair_count:
        issues.append(_issue(
            "touch_axis_order_inverted",
            f"touchMajor < touchMinor in {inverted_touch_axis_pair_count}/{touch_axis_pair_count} paired frames",
            "warning",
        ))
    if touch_axis_pair_count and not touch_axes_distinct:
        issues.append(_issue(
            "touch_axes_not_distinct",
            f"touchMajor equals touchMinor in all {touch_axis_pair_count} paired frames; radius only",
            "warning",
        ))
    if tool_axis_comparison_count and tool_axis_alias_count == tool_axis_comparison_count:
        issues.append(_issue(
            "tool_axis_alias_detected",
            f"tool axes duplicate touch axes in all {tool_axis_alias_count} comparable frames",
            "warning",
        ))
    if pressure_values and not pressure_has_declared_range and max(pressure_values) - min(pressure_values) <= 0.0001:
        issues.append(_issue(
            "pressure_uninformative",
            "pressure is constant and the input device declares no pressure range",
            "warning",
        ))
    if inverted_touch_axis_pair_count and capability.get("touchAxesSemanticallyValid") is not False:
        issues.append(_issue(
            "touch_axis_capability_mismatch",
            "deviceCapability must mark observed inverted touch axes as semantically invalid",
        ))
    if touch_axis_pair_count and not touch_axes_distinct and capability.get("touchAxesDistinct") is not False:
        issues.append(_issue(
            "touch_axis_capability_mismatch",
            "deviceCapability must not claim independent shape when touch axes are identical",
        ))
    if inverted_touch_axis_pair_count and capability.get("level") in ("A", "B"):
        issues.append(_issue(
            "touch_axis_capability_mismatch",
            "deviceCapability level cannot rely on semantically invalid touch axes",
        ))
    if (
        tool_axis_comparison_count
        and tool_axis_alias_count == tool_axis_comparison_count
        and capability.get("toolAxesIndependent") is True
    ):
        issues.append(_issue(
            "tool_axis_capability_mismatch",
            "deviceCapability cannot claim independent tool axes when all observed values are aliases",
        ))
    return {
        "ok": not any(issue["severity"] == "error" for issue in issues),
        "schemaVersion": schema,
        "frameCount": len(record.get("frames") or []),
        "touchFrameCount": len(frames),
        "rotationFrameCount": len(rotation_frames),
        "touchAxesSemanticallyValid": touch_axes_semantically_valid,
        "touchAxesDistinct": touch_axes_distinct,
        "toolAxesIndependent": tool_axes_independent,
        "issues": issues,
    }


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and float(value) > 0


def _same_number(left: Any, right: Any) -> bool:
    return abs(float(left) - float(right)) <= 0.0001


def iter_json_files(source: Path) -> Iterable[Path]:
    if source.is_file() and source.suffix.lower() == ".json":
        yield source
        return
    if source.is_dir():
        yield from sorted(source.rglob("*.json"))


def audit_paths(source: Path) -> Dict[str, Any]:
    results = []
    for path in iter_json_files(source):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            result = audit_record(record)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            result = {"ok": False, "issues": [_issue("json_read_failed", str(error))]}
        result["path"] = str(path)
        results.append(result)
    return {
        "contract": "android_schema_v6_cross_device_capture_v1",
        "source": str(source),
        "recordCount": len(results),
        "passedCount": sum(1 for result in results if result.get("ok")),
        "failedCount": sum(1 for result in results if not result.get("ok")),
        "records": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_paths(args.source)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0 if report["failedCount"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
