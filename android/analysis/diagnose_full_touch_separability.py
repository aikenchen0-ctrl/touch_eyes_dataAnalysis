#!/usr/bin/env python3
"""Audit full-touch separability before extending the causal runtime window."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def finite(value):
    """Return whether a JSON scalar is a finite number."""
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def raw_paths(root):
    """Return schema v6 raw samples without reading generated reports."""
    return sorted(
        path
        for path in Path(root).rglob("*.json")
        if path.is_file()
        and "annotations" not in path.relative_to(root).parts
    )


def load_sample(path):
    """Read one raw sample and reject incomplete records."""
    with path.open("r", encoding="utf-8") as source:
        data = json.load(source)
    if int(data.get("schemaVersion") or 0) != 6:
        raise RuntimeError("schema6_required")
    label = data.get("label") or {}
    strength = str(label.get("touchStrength") or "")
    finger = str(label.get("fingerType") or "")
    if strength not in ("light", "heavy") or finger not in ("thumb", "index"):
        raise RuntimeError("unsupported_label")
    alignment = data.get("timeAlignment") or {}
    down_ns = int(alignment.get("downEventElapsedRealtimeNanos") or 0)
    lift_ns = int(alignment.get("liftEventElapsedRealtimeNanos") or 0)
    if down_ns <= 0 or lift_ns <= down_ns:
        raise RuntimeError("missing_touch_interval")
    frames = data.get("frames") or []
    touches = data.get("touchFrames") or []
    if len(frames) < 2 or len(touches) < 2:
        raise RuntimeError("insufficient_frames")
    return {
        "path": path,
        "data": data,
        "label": 1 if strength == "heavy" else 0,
        "finger": finger,
        "downNs": down_ns,
        "liftNs": lift_ns,
        "frames": frames,
        "touches": touches,
    }


def ordered_frames(sample, key):
    """Sort a sensor stream by its source timestamp."""
    return sorted(
        sample[key],
        key=lambda frame: int(frame.get(
            "sensorTimestampElapsedRealtimeNanos"
            if key == "frames"
            else "eventElapsedRealtimeNanos"
        ) or 0),
    )


def interpolate_imu(frames, target_ns):
    """Linearly interpolate one IMU vector at a target timestamp."""
    times = np.asarray([
        int(frame["sensorTimestampElapsedRealtimeNanos"])
        for frame in frames
    ], dtype=np.int64)
    position = int(np.searchsorted(times, target_ns))
    if position <= 0:
        left = right = frames[0]
    elif position >= len(frames):
        left = right = frames[-1]
    else:
        left = frames[position - 1]
        right = frames[position]
    left_ns = int(left["sensorTimestampElapsedRealtimeNanos"])
    right_ns = int(right["sensorTimestampElapsedRealtimeNanos"])
    if right_ns <= left_ns:
        fraction = 0.0
    else:
        fraction = float(
            (target_ns - left_ns) / (right_ns - left_ns)
        )
    output = []
    for name in (
        "x",
        "y",
        "z",
        "rotationRateX",
        "rotationRateY",
        "rotationRateZ",
    ):
        a = float(left.get(name) or 0.0)
        b = float(right.get(name) or 0.0)
        output.append(a + fraction * (b - a))
    return np.asarray(output, dtype=np.float64)


def touch_at(touches, target_ns):
    """Select the latest received touch state that was already available."""
    selected = None
    for frame in touches:
        received = int(frame.get("receivedElapsedRealtimeNanos") or 0)
        event = int(frame.get("eventElapsedRealtimeNanos") or 0)
        if received <= 0 or event <= 0:
            continue
        if received <= target_ns + 12_000_000 and event <= target_ns:
            selected = frame
        else:
            break
    return selected


def sequence_features(sample, offsets_ms):
    """Extract causal fixed-grid features over one configurable window."""
    frames = ordered_frames(sample, "frames")
    touches = ordered_frames(sample, "touches")
    acceleration = []
    rotation = []
    size = []
    major = []
    minor = []
    x_norm = []
    y_norm = []
    availability = []
    for offset in offsets_ms:
        target_ns = sample["downNs"] + int(offset * 1_000_000)
        imu = interpolate_imu(frames, target_ns)
        acceleration.append(imu[:3])
        rotation.append(imu[3:])
        touch = touch_at(touches, target_ns)
        if touch is None:
            x_norm.append(0.0)
            y_norm.append(0.0)
            size.append(0.0)
            major.append(0.0)
            minor.append(0.0)
            availability.append(0.0)
        else:
            x_norm.append(float(touch.get("x_norm") or 0.0))
            y_norm.append(float(touch.get("y_norm") or 0.0))
            size.append(float(touch.get("size") or 0.0))
            major.append(float(touch.get("touchMajor") or 0.0))
            minor.append(float(touch.get("touchMinor") or 0.0))
            availability.append(1.0)
    acceleration = np.asarray(acceleration, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    major = np.asarray(major, dtype=np.float64)
    minor = np.asarray(minor, dtype=np.float64)
    x_norm = np.asarray(x_norm, dtype=np.float64)
    y_norm = np.asarray(y_norm, dtype=np.float64)
    availability = np.asarray(availability, dtype=np.float64)
    duration_ms = (sample["liftNs"] - sample["downNs"]) / 1e6
    accel_norm = np.linalg.norm(acceleration, axis=1)
    rotation_norm = np.linalg.norm(rotation, axis=1)
    dx = np.diff(x_norm)
    dy = np.diff(y_norm)
    path = float(np.sum(np.hypot(dx, dy)))
    return np.asarray([
        duration_ms,
        float(np.mean(accel_norm)),
        float(np.std(accel_norm)),
        float(np.max(accel_norm)),
        float(np.mean(rotation_norm)),
        float(np.std(rotation_norm)),
        float(np.max(rotation_norm)),
        float(np.mean(size)),
        float(np.std(size)),
        float(np.max(size) - np.min(size)),
        float(np.mean(major)),
        float(np.std(major)),
        float(np.max(major) - np.min(major)),
        float(np.mean(minor)),
        float(np.std(minor)),
        path,
        float(np.max(np.hypot(x_norm, y_norm))),
        float(np.mean(availability)),
    ], dtype=np.float64)


def auc(truths, scores):
    """Compute rank AUC with tied values receiving half credit."""
    truths = np.asarray(truths, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = scores[truths == 1]
    negatives = scores[truths == 0]
    if not len(positives) or not len(negatives):
        return None
    comparisons = (
        (positives[:, None] > negatives[None, :]).astype(np.float64)
        + 0.5 * (
            positives[:, None] == negatives[None, :]
        ).astype(np.float64)
    )
    return float(comparisons.mean())


def recall_at_fpr(truths, scores, target=0.01):
    """Return recall at a single threshold calibrated on these negatives."""
    truths = np.asarray(truths, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    negatives = np.sort(scores[truths == 0])[::-1]
    allowed = int(math.floor(len(negatives) * target))
    threshold = negatives[allowed] if allowed < len(negatives) else -math.inf
    predictions = scores > threshold
    positives = truths == 1
    return {
        "threshold": float(threshold),
        "recall": float(predictions[positives].mean()),
        "fpr": float(predictions[~positives].mean()),
    }


def within_group_center(values, groups):
    """Remove a label-blind session location from each feature."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    result = values.copy()
    for group in sorted(set(groups)):
        selected = groups == group
        result[selected] = (
            values[selected] - np.median(values[selected], axis=0)
        )
    return result


def train_logistic(train_values, train_labels, target_values):
    """Fit a small regularized linear model for an offline upper bound."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_values)
    x_target = scaler.transform(target_values)
    model = LogisticRegression(
        max_iter=2000,
        random_state=20261023,
    )
    model.fit(x_train, train_labels)
    return model.predict_proba(x_target)[:, 1]


def session_split(samples, seed=20261023, validation_ratio=0.22):
    """Hold out complete sessions while preserving class and posture."""
    by_session = defaultdict(list)
    for sample in samples:
        by_session[str(
            sample["data"].get("collectionSessionID") or ""
        )].append(sample)
    session_meta = {}
    for session, values in by_session.items():
        if not session:
            raise RuntimeError("sample_missing_session")
        labels = {sample["label"] for sample in values}
        if len(labels) != 1:
            raise RuntimeError("session_has_multiple_labels")
        postures = Counter(
            str(
                (sample["data"].get("collectionContext") or {}).get(
                    "bodyPosture"
                ) or "unknown"
            )
            for sample in values
        )
        session_meta[session] = {
            "label": next(iter(labels)),
            "posture": sorted(
                postures.items(),
                key=lambda item: (-item[1], item[0]),
            )[0][0],
        }
    labels_present = {
        meta["label"] for meta in session_meta.values()
    }
    if labels_present != {0, 1}:
        raise RuntimeError("dataset_missing_strength_class")
    rng = np.random.default_rng(seed)
    validation = set()
    # 中文注释：按标签和姿态分层留出。优先从含多个 session 的层中抽取，
    # 保证训练集和验证集都能覆盖两类；同一 session 始终只进入一侧。
    for label in (0, 1):
        strata = defaultdict(list)
        for session, meta in session_meta.items():
            if meta["label"] == label:
                strata[meta["posture"]].append(session)
        ordered_strata = sorted(strata)
        rng.shuffle(ordered_strata)
        for posture in ordered_strata:
            sessions = sorted(strata[posture])
            rng.shuffle(sessions)
            if len(sessions) < 2:
                continue
            count = max(1, int(round(len(sessions) * validation_ratio)))
            count = min(count, len(sessions) - 1)
            validation.update(sessions[:count])
        if not any(
            session_meta[session]["label"] == label
            for session in validation
        ):
            candidates = sorted(
                session
                for session, meta in session_meta.items()
                if meta["label"] == label
            )
            if len(candidates) < 2:
                raise RuntimeError(
                    "strength_class_needs_two_sessions_for_holdout"
                )
            validation.add(candidates[0])
    train = [
        sample
        for sample in samples
        if str(sample["data"].get("collectionSessionID") or "")
        not in validation
    ]
    valid = [
        sample
        for sample in samples
        if str(sample["data"].get("collectionSessionID") or "")
        in validation
    ]
    valid_labels = {sample["label"] for sample in valid}
    train_labels = {sample["label"] for sample in train}
    if valid_labels != {0, 1}:
        raise RuntimeError("validation_session_split_missing_class")
    if train_labels != {0, 1}:
        raise RuntimeError("training_session_split_missing_class")
    train_sessions = {
        str(sample["data"].get("collectionSessionID") or "")
        for sample in train
    }
    if train_sessions.intersection(validation):
        raise RuntimeError("session_split_leakage")
    return train, valid, sorted(validation)


def evaluate_window(samples, offsets_ms, session_centered):
    """Evaluate one window and return descriptive cross-session metrics."""
    values = np.stack([
        sequence_features(sample, offsets_ms)
        for sample in samples
    ])
    groups = np.asarray([
        str(sample["data"].get("collectionSessionID") or "")
        for sample in samples
    ], dtype=object)
    if session_centered:
        values = within_group_center(values, groups)
    train, valid, validation_sessions = session_split(samples)
    train_index = {
        sample["path"]: index for index, sample in enumerate(samples)
    }
    train_indices = np.asarray([
        train_index[sample["path"]] for sample in train
    ], dtype=np.int64)
    valid_indices = np.asarray([
        train_index[sample["path"]] for sample in valid
    ], dtype=np.int64)
    scores = train_logistic(
        values[train_indices],
        np.asarray([sample["label"] for sample in train]),
        values[valid_indices],
    )
    truths = np.asarray([sample["label"] for sample in valid])
    per_posture = {}
    for posture in sorted({
        str(
            (sample["data"].get("collectionContext") or {}).get(
                "bodyPosture"
            ) or "unknown"
        )
        for sample in valid
    }):
        selected = np.asarray([
            str(
                (sample["data"].get("collectionContext") or {}).get(
                    "bodyPosture"
                ) or "unknown"
            ) == posture
            for sample in valid
        ])
        if len(set(truths[selected])) < 2:
            continue
        per_posture[posture] = recall_at_fpr(
            truths[selected],
            scores[selected],
        )
    return {
        "offsetsMs": list(offsets_ms),
        "sessionCentered": session_centered,
        "trainCount": len(train),
        "validationCount": len(valid),
        "validationSessions": validation_sessions,
        "overall": recall_at_fpr(truths, scores),
        "auc": auc(truths, scores),
        "posture": per_posture,
        "featureAuc": {
            name: auc(truths, values[valid_indices, index])
            for index, name in enumerate((
                "duration_ms",
                "accel_mean",
                "accel_std",
                "accel_max",
                "rotation_mean",
                "rotation_std",
                "rotation_max",
                "size_mean",
                "size_std",
                "size_range",
                "major_mean",
                "major_std",
                "major_range",
                "minor_mean",
                "minor_std",
                "path",
                "radius",
                "availability",
            ))
        },
    }


def choose_mixed_prefix(samples, minimum_samples=64):
    """Keep whole early sessions without relying on adjacent file order."""
    by_session = defaultdict(list)
    session_order = []
    for sample in samples:
        session = str(
            sample["data"].get("collectionSessionID") or ""
        )
        if session not in by_session:
            session_order.append(session)
        by_session[session].append(sample)
    sessions_by_label = Counter()
    sessions_by_posture = Counter()
    session_postures = {}
    for session, values in by_session.items():
        label = values[0]["label"]
        posture = str(
            (values[0]["data"].get("collectionContext") or {}).get(
                "bodyPosture"
            ) or "unknown"
        )
        sessions_by_label[label] += 1
        sessions_by_posture[posture] += 1
        session_postures[session] = posture
    target_per_label = {
        label: min(2, sessions_by_label[label])
        for label in (0, 1)
    }
    target_postures = min(2, len(sessions_by_posture))
    selected = []
    selected_sessions = []
    seen_labels = set()
    label_sessions = Counter()
    selected_postures = set()
    for session in session_order:
        session_samples = by_session[session]
        selected.extend(session_samples)
        selected_sessions.append(session)
        seen_labels.update(sample["label"] for sample in session_samples)
        label_sessions[session_samples[0]["label"]] += 1
        selected_postures.add(session_postures[session])
        # 中文注释：至少覆盖两类；数据允许时，每类留出两个 session，
        # 并覆盖两个姿态，避免前两个文件恰好都来自同一姿态。
        if (
            len(selected) >= minimum_samples
            and seen_labels == {0, 1}
            and all(
                label_sessions[label] >= target_per_label[label]
                for label in (0, 1)
            )
            and len(selected_postures) >= target_postures
        ):
            break
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    samples = []
    rejected = Counter()
    for path in raw_paths(args.input_dir):
        try:
            samples.append(load_sample(path))
        except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as error:
            rejected[str(error)] += 1
        if args.max_samples and len(samples) >= args.max_samples:
            break
    if args.max_samples:
        samples = choose_mixed_prefix(
            samples,
            minimum_samples=args.max_samples,
        )
    windows = (
        (-15, -10, -5, 0, 5, 10, 15, 20, 25),
        (-15, -10, -5, 0, 5, 10, 15, 20, 25, 40, 55, 70),
        (-15, -10, -5, 0, 5, 10, 15, 20, 25, 40, 55, 70, 85, 100),
    )
    report = {
        "scope": "full_touch_descriptive_upper_bound",
        "negativeProxy": "light_touch",
        "ordinaryClickFprAvailable": False,
        "sampleCount": len(samples),
        "rejected": dict(rejected),
        "classCounts": dict(Counter(
            "heavy" if sample["label"] else "light"
            for sample in samples
        )),
        "fingerCounts": dict(Counter(
            sample["finger"] for sample in samples
        )),
        "postureCounts": dict(Counter(
            str(
                (sample["data"].get("collectionContext") or {}).get(
                    "bodyPosture"
                ) or "unknown"
            )
            for sample in samples
        )),
        "results": [],
    }
    for offsets in windows:
        for centered in (False, True):
            try:
                report["results"].append(
                    evaluate_window(samples, offsets, centered)
                )
            except RuntimeError as error:
                report["results"].append({
                    "offsetsMs": list(offsets),
                    "sessionCentered": centered,
                    "error": str(error),
                })
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "sampleCount": len(samples),
        "rejected": len(rejected),
        "results": len(report["results"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
