#!/usr/bin/env python3
"""Audit channel information and session confounding in touch26 data."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


CLASS_LABELS = ("thumb_light", "thumb_heavy", "index_light", "index_heavy")
CHANNEL_GROUPS = {
    "x_norm": "touch_location",
    "y_norm": "touch_location",
    "userAcceleration.x": "linear_acceleration",
    "userAcceleration.y": "linear_acceleration",
    "userAcceleration.z": "linear_acceleration",
    "rotationRate.x": "angular_velocity",
    "rotationRate.y": "angular_velocity",
    "rotationRate.z": "angular_velocity",
    "gravity.x": "gravity_pose",
    "gravity.y": "gravity_pose",
    "gravity.z": "gravity_pose",
    "pressure": "contact_force",
    "size": "contact_area",
    "major_norm": "contact_shape",
    "minor_norm": "contact_shape",
    "log_axis_ratio": "contact_shape",
    "orientation_sin": "contact_orientation",
    "orientation_cos": "contact_orientation",
    "pointer_count": "pointer_state",
    "touch_valid": "availability",
    "pressure_available": "availability",
    "size_available": "availability",
    "major_available": "availability",
    "minor_available": "availability",
    "orientation_available": "availability",
    "pointer_count_available": "availability",
}
VALUE_MASK = {
    "pressure": ("pressure_available",),
    "size": ("size_available",),
    "major_norm": ("major_available",),
    "minor_norm": ("minor_available",),
    "log_axis_ratio": ("major_available", "minor_available"),
    "orientation_sin": ("orientation_available",),
    "orientation_cos": ("orientation_available",),
    "pointer_count": ("pointer_count_available",),
}


def load_dataset(dataset_dir):
    path = Path(dataset_dir) / "training_touch26.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def variance_share(values, group_values):
    total_mean = float(np.mean(values))
    _, group_codes = np.unique(group_values, return_inverse=True)
    counts = np.bincount(group_codes)
    sums = np.bincount(group_codes, weights=values)
    group_means = sums / counts
    predicted = group_means[group_codes]
    total_ss = float(np.sum((values - total_mean) ** 2))
    between_ss = float(np.sum((predicted - total_mean) ** 2))
    return between_ss / total_ss if total_ss > 0 else 0.0


def class_effect(values, labels):
    total_mean = float(np.mean(values))
    between = 0.0
    within = 0.0
    for label in CLASS_LABELS:
        selected = values[labels == label]
        if len(selected):
            between += len(selected) * (float(np.mean(selected)) - total_mean) ** 2
            within += float(np.sum((selected - np.mean(selected)) ** 2))
    return between / (between + within) if between + within > 0 else 0.0


def channel_mask(flat, names, channel):
    indices = {name: index for index, name in enumerate(names)}
    mask = np.ones(len(flat), dtype=bool)
    for mask_name in VALUE_MASK.get(channel, ()):
        mask &= flat[:, indices[mask_name]] >= 0.5
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = load_dataset(args.dataset_dir)
    names = rows[0]["featureNames"]
    matrix = np.asarray([row["matrix"] for row in rows], dtype=np.float64)
    group_ids, group_codes = np.unique(
        np.asarray([row["groupId"] for row in rows]),
        return_inverse=True,
    )
    label_ids, label_codes = np.unique(
        np.asarray([row["classLabel"] for row in rows]),
        return_inverse=True,
    )
    groups = np.repeat(group_codes, matrix.shape[1])
    labels = np.repeat(label_codes, matrix.shape[1])
    flat = matrix.reshape(-1, len(names))
    channels = []
    for index, name in enumerate(names):
        valid = channel_mask(flat, names, name)
        values = flat[valid, index]
        valid_groups = groups[valid]
        valid_labels = labels[valid]
        if not len(values):
            channels.append(
                {
                    "name": name,
                    "group": CHANNEL_GROUPS[name],
                    "validRate": 0.0,
                    "constant": True,
                }
            )
            continue
        group_counts = np.bincount(valid_groups)
        group_sums = np.bincount(valid_groups, weights=values)
        group_squares = np.bincount(valid_groups, weights=values * values)
        total_mean = float(np.mean(values))
        predicted = (group_sums / group_counts)[valid_groups]
        total_ss = float(np.sum((values - total_mean) ** 2))
        between_ss = float(np.sum((predicted - total_mean) ** 2))
        class_counts = np.bincount(valid_labels, minlength=len(label_ids))
        class_sums = np.bincount(
            valid_labels,
            weights=values,
            minlength=len(label_ids),
        )
        class_means = np.divide(
            class_sums,
            class_counts,
            out=np.zeros(len(label_ids), dtype=np.float64),
            where=class_counts > 0,
        )
        class_between = float(
            np.sum(class_counts * (class_means - total_mean) ** 2)
        )
        class_within = max(0.0, total_ss - class_between)
        channels.append(
            {
                "name": name,
                "group": CHANNEL_GROUPS[name],
                "validRate": float(len(values) / len(flat)),
                "mean": total_mean,
                "std": float(np.std(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "uniqueCount": int(len(np.unique(values))),
                "sessionVarianceShare": (
                    between_ss / total_ss if total_ss > 0 else 0.0
                ),
                "classEffectShare": (
                    class_between / (class_between + class_within)
                    if class_between + class_within > 0
                    else 0.0
                ),
                "classMeans": {
                    label_ids[index]: (
                        float(class_means[index])
                        if class_counts[index] > 0
                        else None
                    )
                    for index in range(len(label_ids))
                },
                "constant": bool(np.max(values) == np.min(values)),
            }
        )
    touch_valid = flat[:, names.index("touch_valid")] >= 0.5
    correlation = np.corrcoef(flat[touch_valid], rowvar=False)
    correlations = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            value = correlation[left, right]
            if np.isfinite(value):
                correlations.append(
                    {
                        "left": names[left],
                        "right": names[right],
                        "correlation": float(value),
                        "absoluteCorrelation": float(abs(value)),
                    }
                )
    correlations.sort(
        key=lambda item: item["absoluteCorrelation"], reverse=True
    )
    grid_offsets = (-15, -10, -5, 0, 5, 10, 15, 20, 25)
    validity_by_offset = []
    for offset_index, offset in enumerate(grid_offsets):
        validity_by_offset.append(
            {
                "offsetMs": offset,
                "touchValidRate": float(
                    np.mean(matrix[:, offset_index, names.index("touch_valid")] >= 0.5)
                ),
                "pressureAvailableRate": float(
                    np.mean(
                        matrix[:, offset_index, names.index("pressure_available")]
                        >= 0.5
                    )
                ),
                "orientationAvailableRate": float(
                    np.mean(
                        matrix[
                            :,
                            offset_index,
                            names.index("orientation_available"),
                        ]
                        >= 0.5
                    )
                ),
            }
        )
    group_labels = Counter(
        (row["groupId"], row["classLabel"]) for row in rows
    )
    sessions_per_class = Counter(
        group_label[1] for group_label in group_labels
    )
    payload = {
        "sampleCount": len(rows),
        "sessionCount": len({row["groupId"] for row in rows}),
        "sessionsPerClass": dict(sessions_per_class),
        "classCounts": dict(Counter(row["classLabel"] for row in rows)),
        "devices": dict(Counter(row["deviceName"] for row in rows)),
        "motionStates": dict(
            Counter(row.get("motionState", "") for row in rows)
        ),
        "postureCoverage": {
            key: dict(Counter(row.get(key, "") for row in rows))
            for key in (
                "bodyPosture",
                "grip",
                "support",
                "requestedDevicePose",
            )
        },
        "validityByOffset": validity_by_offset,
        "channels": channels,
        "topCorrelations": correlations[:30],
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "channel_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output / "channel_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        fields = (
            "name",
            "group",
            "validRate",
            "mean",
            "std",
            "minimum",
            "maximum",
            "uniqueCount",
            "sessionVarianceShare",
            "classEffectShare",
            "constant",
        )
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for channel in channels:
            writer.writerow(
                {
                    field: channel.get(field, "")
                    for field in fields
                }
            )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
