#!/usr/bin/env python3

import math
from copy import deepcopy


SIGNAL_FIELDS = (
    "delta",
    "x",
    "y",
    "z",
    "rotationRateX",
    "rotationRateY",
    "rotationRateZ",
    "gravityX",
    "gravityY",
    "gravityZ",
    "x_norm",
    "y_norm",
)


def finite_float(value, fallback=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def is_observed_frame(frame):
    return any(abs(finite_float(frame.get(name), 0.0)) > 1e-12 for name in SIGNAL_FIELDS)


def touch_down_timestamp(data):
    touch_frames = data.get("touchFrames") or []
    if touch_frames:
        event_time_millis = finite_float(touch_frames[0].get("eventTimeMillis"), None)
        if event_time_millis is not None:
            return event_time_millis / 1000.0, "touchFrames[0].eventTimeMillis"
    touch_events = data.get("touchEvents") or []
    if touch_events:
        timestamp = finite_float(touch_events[0].get("timestamp"), None)
        if timestamp is not None:
            return timestamp, "touchEvents[0].timestamp"
    return None, "missing"


def frame_time_since_down(frame, down_timestamp, anchor_timestamp):
    stored = finite_float(frame.get("timeSinceTouchDown"), None)
    if stored is not None:
        return stored
    relative = finite_float(frame.get("relativeTimestamp"), None)
    if relative is None:
        return None
    return anchor_timestamp + relative - down_timestamp


def analyze_sample(data):
    down_timestamp, down_source = touch_down_timestamp(data)
    anchor_timestamp = finite_float(data.get("anchorTimestamp"), None)
    sample_rate_hz = finite_float(data.get("sampleRateHz"), None)
    frames = data.get("frames") or []
    observed = []
    placeholder_count = 0
    if down_timestamp is not None and anchor_timestamp is not None:
        for position, frame in enumerate(frames):
            if not is_observed_frame(frame):
                placeholder_count += 1
                continue
            time_since_down = frame_time_since_down(frame, down_timestamp, anchor_timestamp)
            if time_since_down is None:
                continue
            observed.append({
                "frame": frame,
                "position": position,
                "time_since_down": time_since_down,
            })
    observed.sort(key=lambda item: (item["time_since_down"], item["position"]))
    result = {
        "eligible_a": False,
        "eligible_b": False,
        "exclusion_reason_a": "",
        "exclusion_reason_b": "not_eligible_a",
        "down_timestamp": down_timestamp,
        "down_timestamp_source": down_source,
        "anchor_timestamp": anchor_timestamp,
        "sample_rate_hz": sample_rate_hz,
        "virtual_down_frame_index": None,
        "nearest_source_frame_index": None,
        "nearest_time_error_seconds": None,
        "observed_frame_count": len(observed),
        "placeholder_frame_count": placeholder_count,
        "pre_down_observed_count": sum(1 for item in observed if item["time_since_down"] < 0),
        "post_down_observed_count": sum(1 for item in observed if item["time_since_down"] > 0),
        "down_frames": [],
    }
    if down_timestamp is None:
        result["exclusion_reason_a"] = "missing_down_timestamp"
        return result
    if anchor_timestamp is None:
        result["exclusion_reason_a"] = "missing_anchor_timestamp"
        return result
    if sample_rate_hz is None or sample_rate_hz <= 0:
        result["exclusion_reason_a"] = "invalid_sample_rate"
        return result
    result["virtual_down_frame_index"] = (down_timestamp - anchor_timestamp) * sample_rate_hz
    if len(observed) < 2:
        result["exclusion_reason_a"] = "insufficient_observed_frames"
        return result
    nearest_index = min(range(len(observed)), key=lambda index: abs(observed[index]["time_since_down"]))
    nearest = observed[nearest_index]
    result["nearest_source_frame_index"] = finite_float(nearest["frame"].get("frameIndex"), nearest["position"])
    result["nearest_time_error_seconds"] = abs(nearest["time_since_down"])
    if observed[0]["time_since_down"] > 0 or observed[-1]["time_since_down"] < 0:
        result["exclusion_reason_a"] = "down_outside_observed_range"
        return result
    result["eligible_a"] = True
    result["exclusion_reason_b"] = "insufficient_down_context"
    if nearest_index < 5 or len(observed) - nearest_index - 1 < 9:
        return result
    period = 1.0 / sample_rate_hz
    if abs(nearest["time_since_down"]) > period * 0.75:
        result["exclusion_reason_b"] = "down_frame_too_far"
        return result
    selected = observed[nearest_index - 5:nearest_index + 10]
    gaps = [
        selected[index + 1]["time_since_down"] - selected[index]["time_since_down"]
        for index in range(len(selected) - 1)
    ]
    if any(gap <= 0 or gap > period * 1.75 for gap in gaps):
        result["exclusion_reason_b"] = "non_contiguous_down_window"
        return result
    down_frames = []
    for frame_index, item in zip(range(-5, 10), selected):
        frame = deepcopy(item["frame"])
        frame["sourceFrameIndex"] = frame.get("frameIndex")
        frame["sourceRelativeTimestamp"] = frame.get("relativeTimestamp")
        frame["sourceTimeSinceTouchDown"] = item["time_since_down"]
        frame["frameIndex"] = frame_index
        frame["relativeTimestamp"] = item["time_since_down"]
        frame["timeSinceAnchor"] = item["time_since_down"]
        frame["timeSinceTouchDown"] = item["time_since_down"]
        down_frames.append(frame)
    result["eligible_b"] = True
    result["exclusion_reason_b"] = ""
    result["down_frames"] = down_frames
    return result
