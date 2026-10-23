#!/usr/bin/env python3
"""Audit whether nominal pre-touch frames already contain press onset."""

import argparse
import bisect
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


GRID_OFFSETS_MS = (-15, -10, -5, 0)
EARLY_BINS_MS = (
    (-120, -80),
    (-80, -40),
    (-40, -20),
    (-20, -10),
    (-10, 0),
)
RISE_START_MS = -80
RISE_END_MS = -20
MIN_RISE_FRAME_COUNT = 4
RISE_ABSOLUTE_FLOOR = 0.10
RISE_NOISE_MULTIPLIER = 3.0
CLASS_IN_FILENAME = re.compile(
    r"_(?P<strength>light|heavy|ordinary)_"
    r"(?P<finger>thumb|index)"
)


def positive_int(value):
    """Convert a timestamp without accepting booleans or negatives."""
    if isinstance(value, bool):
        return 0
    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0
    return result if result > 0 else 0


def finite_float(value):
    """Return a finite float or raise a stable contract error."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError("non_finite_imu_value") from error
    if not math.isfinite(result):
        raise RuntimeError("non_finite_imu_value")
    return result


def frame_timestamp(frame):
    """Read the sensor clock used for causal interpolation."""
    timestamp = positive_int(
        frame.get("sensorTimestampElapsedRealtimeNanos")
    )
    if timestamp <= 0:
        raise RuntimeError("missing_imu_sensor_timestamp")
    return timestamp


def frame_acceleration_magnitude(frame):
    """Compute linear-acceleration magnitude in the collected source units."""
    return float(np.linalg.norm(np.asarray(
        (
            finite_float(frame.get("x")),
            finite_float(frame.get("y")),
            finite_float(frame.get("z")),
        ),
        dtype=np.float64,
    )))


def bracket_indices(timestamps, target):
    """Return the two sensor frames used by linear time-grid interpolation."""
    if len(timestamps) < 2:
        raise RuntimeError("insufficient_imu_frames")
    right = bisect.bisect_left(timestamps, target)
    if right <= 0:
        return 0, 1
    if right >= len(timestamps):
        return len(timestamps) - 2, len(timestamps) - 1
    return right - 1, right


def grid_interpolation_audit(frames, down_ns):
    """Report whether each nominal pre-touch slot reads through down time."""
    timestamps = [frame_timestamp(frame) for frame in frames]
    result = {}
    for offset_ms in GRID_OFFSETS_MS:
        target_ns = down_ns + offset_ms * 1_000_000
        left, right = bracket_indices(timestamps, target_ns)
        left_ns = timestamps[left]
        right_ns = timestamps[right]
        result[str(offset_ms)] = {
            "targetMs": offset_ms,
            "leftMs": (left_ns - down_ns) / 1_000_000.0,
            "rightMs": (right_ns - down_ns) / 1_000_000.0,
            "interpolationCrossesDown": right_ns >= down_ns,
            "strictlyPreDispatch": (
                offset_ms < 0 and right_ns < down_ns
            ),
        }
    return result


def binned_magnitudes(frames, down_ns):
    """Return median acceleration magnitude for each declared time bin."""
    values = defaultdict(list)
    for frame in frames:
        relative_ms = (
            frame_timestamp(frame) - down_ns
        ) / 1_000_000.0
        magnitude = frame_acceleration_magnitude(frame)
        for start_ms, stop_ms in EARLY_BINS_MS:
            if start_ms <= relative_ms < stop_ms:
                values[(start_ms, stop_ms)].append(magnitude)
                break
    return {
        str(start_ms) + ":" + str(stop_ms): (
            float(np.median(bin_values))
            if bin_values
            else None
        )
        for (start_ms, stop_ms), bin_values in (
            (key, values.get(key, []))
            for key in EARLY_BINS_MS
        )
    }


def rising_pretouch_audit(frames, down_ns):
    """Detect a sustained rise before dispatch without future frames."""
    selected = [
        (
            (frame_timestamp(frame) - down_ns) / 1_000_000.0,
            frame_acceleration_magnitude(frame),
        )
        for frame in frames
        if (
            RISE_START_MS
            <= (frame_timestamp(frame) - down_ns) / 1_000_000.0
            < RISE_END_MS
        )
    ]
    selected.sort(key=lambda item: item[0])
    if len(selected) < MIN_RISE_FRAME_COUNT:
        return {
            "available": False,
            "frameCount": len(selected),
            "rising": False,
            "delta": None,
            "noise": None,
        }
    values = np.asarray(
        [item[1] for item in selected],
        dtype=np.float64,
    )
    split = max(1, len(values) // 2)
    early = float(np.median(values[:split]))
    late = float(np.median(values[split:]))
    delta = late - early
    noise = float(np.median(np.abs(np.diff(values)))) if len(
        values
    ) > 1 else 0.0
    threshold = max(
        RISE_ABSOLUTE_FLOOR,
        RISE_NOISE_MULTIPLIER * noise,
    )
    return {
        "available": True,
        "frameCount": len(selected),
        "rising": bool(delta > threshold),
        "delta": delta,
        "noise": noise,
        "threshold": threshold,
    }


def class_label(record):
    """Return the training strength/finger label."""
    label = record.get("label") or {}
    finger = str(label.get("fingerType") or "").strip()
    strength = str(label.get("touchStrength") or "").strip()
    return finger + "_" + strength if finger and strength else "unknown"


def device_name(record):
    """Read the physical model name recorded with the sample."""
    capability = record.get("deviceCapability") or {}
    return str(capability.get("deviceName") or "unknown")


def audit_sample(path):
    """Build one sample-level contamination record."""
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise RuntimeError("invalid_raw_sample")
    alignment = record.get("timeAlignment") or {}
    down_ns = positive_int(
        alignment.get("downEventElapsedRealtimeNanos")
    )
    if down_ns <= 0:
        raise RuntimeError("missing_down_timestamp")
    frames = sorted(
        (
            frame
            for frame in (record.get("frames") or [])
            if isinstance(frame, dict)
        ),
        key=frame_timestamp,
    )
    grid = grid_interpolation_audit(frames, down_ns)
    bins = binned_magnitudes(frames, down_ns)
    rise = rising_pretouch_audit(frames, down_ns)
    return {
        "sourceFile": str(path),
        "sampleId": str(record.get("id") or ""),
        "classLabel": class_label(record),
        "deviceName": device_name(record),
        "motionState": str(
            (record.get("label") or {}).get("motionState") or ""
        ),
        "sessionId": str(
            record.get("collectionSessionID") or ""
        ),
        "grid": grid,
        "bins": bins,
        "rise": rise,
    }


def iter_json_files(root):
    """Yield raw sample JSON files without following directory symlinks."""
    for path in sorted(Path(root).rglob("*.json")):
        if path.is_file():
            yield path


def stratified_sample(paths, max_per_class, seed):
    """Keep a deterministic per-class sample across all physical devices."""
    grouped = defaultdict(list)
    for path in paths:
        match = CLASS_IN_FILENAME.search(path.name)
        label = (
            match.group("finger") + "_" + match.group("strength")
            if match
            else "unknown"
        )
        grouped[label].append(path)
    rng = random.Random(seed)
    selected = []
    for label in sorted(grouped):
        candidates = sorted(grouped[label])
        if max_per_class > 0 and len(candidates) > max_per_class:
            selected.extend(rng.sample(candidates, max_per_class))
        else:
            selected.extend(candidates)
    return sorted(selected)


def numeric_summary(values):
    """Summarize finite values without inventing a result for empty groups."""
    finite = np.asarray(
        [value for value in values if value is not None],
        dtype=np.float64,
    )
    if not len(finite):
        return {
            "count": 0,
            "median": None,
            "p90": None,
        }
    return {
        "count": int(len(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
    }


def summarize(records):
    """Aggregate class/device contamination evidence."""
    grouped = defaultdict(list)
    for record in records:
        grouped[
            (record["deviceName"], record["classLabel"])
        ].append(record)
    groups = {}
    for key, selected in sorted(grouped.items()):
        device, label = key
        groups[device + "|" + label] = {
            "deviceName": device,
            "classLabel": label,
            "sampleCount": len(selected),
            "sessionCount": len({
                record["sessionId"] for record in selected
                if record["sessionId"]
            }),
            "gridCrossingRate": {
                offset: float(np.mean([
                    record["grid"][offset][
                        "interpolationCrossesDown"
                    ]
                    for record in selected
                ]))
                for offset in ("-15", "-10", "-5")
            },
            "strictlyPreDispatchRate": {
                offset: float(np.mean([
                    record["grid"][offset][
                        "strictlyPreDispatch"
                    ]
                    for record in selected
                ]))
                for offset in ("-15", "-10", "-5")
            },
            "bins": {
                bin_name: numeric_summary([
                    record["bins"][bin_name]
                    for record in selected
                ])
                for bin_name in (
                    str(start) + ":" + str(stop)
                    for start, stop in EARLY_BINS_MS
                )
            },
            "risingPretouchRate": float(np.mean([
                bool(record["rise"]["rising"])
                for record in selected
            ])),
            "risingPretouchAvailableRate": float(np.mean([
                bool(record["rise"]["available"])
                for record in selected
            ])),
            "riseDelta": numeric_summary([
                record["rise"]["delta"]
                for record in selected
            ]),
        }
    return {
        "schemaVersion": 1,
        "sampleCount": len(records),
        "classCounts": dict(Counter(
            record["classLabel"] for record in records
        )),
        "deviceCounts": dict(Counter(
            record["deviceName"] for record in records
        )),
        "groups": groups,
    }


def render_markdown(report):
    """Render the key pre-touch contamination metrics."""
    lines = [
        "# Pre-touch contamination audit",
        "",
        "| device | class | samples | -15 crosses down | "
        "-10 crosses down | -5 crosses down | rising before down |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in report["groups"].values():
        crossing = group["gridCrossingRate"]
        lines.append(
            "| {} | {} | {} | {:.2f}% | {:.2f}% | {:.2f}% | {:.2f}% |".format(
                group["deviceName"],
                group["classLabel"],
                group["sampleCount"],
                100.0 * crossing["-15"],
                100.0 * crossing["-10"],
                100.0 * crossing["-5"],
                100.0 * group["risingPretouchRate"],
            )
        )
    lines.extend((
        "",
        "A crossing means the right interpolation frame is at or after "
        "ACTION_DOWN. Rising is a sustained acceleration increase in "
        "[-80, -20) ms before dispatch.",
        "",
    ))
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output")
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=400,
        help="Zero audits every raw sample.",
    )
    parser.add_argument("--seed", type=int, default=20261023)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_samples_per_class < 0:
        raise RuntimeError("invalid_sample_limit")
    paths = []
    for root in args.raw_dir:
        paths.extend(iter_json_files(root))
    selected = stratified_sample(
        paths,
        args.max_samples_per_class,
        args.seed,
    )
    records = []
    failures = Counter()
    for path in selected:
        try:
            records.append(audit_sample(path))
        except RuntimeError as error:
            failures[str(error)] += 1
    report = summarize(records)
    report["sourcePathCount"] = len(paths)
    report["selectedPathCount"] = len(selected)
    report["failureCounts"] = dict(failures)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    if args.markdown_output:
        markdown = Path(args.markdown_output)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(
            render_markdown(report) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "sampleCount": report["sampleCount"],
        "failureCounts": report["failureCounts"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
