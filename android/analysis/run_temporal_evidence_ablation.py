"""Evaluate how much evidence duration is needed for posture generalization.

The production gate currently uses a short time grid around touch-down. This
script reads the complete source captures and builds causal feature sets at
several post-down horizons. It keeps threshold calibration session-disjoint
and evaluates every posture as a held-out domain.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


DEFAULT_WINDOWS_MS = (20, 40, 80, 160, 320, 640, 1000, 1600, 2400)
PRETOUCH_START_MS = -85.0
PRETOUCH_END_MS = 0.0
MIN_PRETOUCH_FRAMES = 3
MIN_WINDOW_FRAMES = 5
CALIBRATION_SESSION_RATIO = 0.22
MIN_CALIBRATION_NEGATIVES = 100
MIN_EVALUATION_NEGATIVES = 100
EPSILON = 1e-8


def finite(value, default=0.0):
    """Return a finite float without allowing malformed JSON to poison rows."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def stats(values):
    """Return fixed-order temporal statistics for one scalar channel."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return [0.0] * 16
    differences = np.diff(values) if len(values) > 1 else np.zeros(1)
    absolute = np.abs(values)
    return [
        float(values.mean()),
        float(values.std()),
        float(values.min()),
        float(values.max()),
        float(absolute.mean()),
        float(np.sqrt(np.mean(values * values))),
        float(np.percentile(values, 90.0)),
        float(np.percentile(values, 95.0)),
        float(values[-1] - values[0]),
        float(differences.mean()),
        float(differences.std()),
        float(np.max(np.abs(differences))),
        float(np.sum(absolute)),
        float(np.sum(differences * differences)),
        float(np.argmax(absolute)) / max(1, len(values) - 1),
        float(values[-1]),
    ]


STATS_FIELDS = (
    "mean",
    "std",
    "min",
    "max",
    "abs_mean",
    "rms",
    "p90",
    "p95",
    "delta",
    "diff_mean",
    "diff_std",
    "diff_abs_max",
    "abs_sum",
    "diff_squared_sum",
    "peak_position",
    "last",
)
RELATIVE_FIELDS = (
    "median",
    "p90",
    "p95",
    "max",
    "robust_effect",
    "p90_over_median",
)


def rotation_to_canonical(gravity):
    """Build a rotation that maps the pretouch gravity to canonical -Z."""
    gravity = np.asarray(gravity, dtype=np.float64)
    norm = float(np.linalg.norm(gravity))
    if norm <= EPSILON:
        return np.eye(3, dtype=np.float64)
    source = gravity / norm
    target = np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine <= EPSILON:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        return np.diag((1.0, -1.0, -1.0))
    skew = np.asarray((
        (0.0, -cross[2], cross[1]),
        (cross[2], 0.0, -cross[0]),
        (-cross[1], cross[0], 0.0),
    ), dtype=np.float64)
    return (
        np.eye(3, dtype=np.float64)
        + skew
        + (skew @ skew) * ((1.0 - cosine) / (sine * sine))
    )


def feature_names(windows_ms):
    """Return the complete frozen feature order for one extraction run."""
    names = []
    for axis in ("x_norm", "y_norm"):
        names.extend(axis + "." + field for field in STATS_FIELDS)
    for channel in (
        "acceleration.x",
        "acceleration.y",
        "acceleration.z",
        "rotation.x",
        "rotation.y",
        "rotation.z",
        "acceleration.magnitude",
        "rotation.magnitude",
    ):
        names.extend(channel + "." + field for field in STATS_FIELDS)
    for channel in ("acceleration", "rotation"):
        names.extend(
            channel + ".relative." + field
            for field in RELATIVE_FIELDS
        )
    for channel in ("acceleration", "rotation"):
        for axis in ("x", "y", "z"):
            names.append(channel + ".peak_direction." + axis)
    names.extend(("gravity.x", "gravity.y", "gravity.z"))
    names.extend(("cross.magnitude_correlation", "touch.duration_ms"))
    return tuple(names)


def feature_sets(names):
    """Define physically motivated subsets for the duration ablation."""
    magnitude_indices = []
    relative_indices = []
    direction_indices = []
    gravity_indices = []
    for index, name in enumerate(names):
        if name.startswith(("acceleration.magnitude.", "rotation.magnitude.")):
            magnitude_indices.append(index)
        if ".relative." in name:
            relative_indices.append(index)
        if ".peak_direction." in name or name.startswith("gravity."):
            direction_indices.append(index)
        if name.startswith("gravity."):
            gravity_indices.append(index)
    all_indices = tuple(range(len(names)))
    duration_index = names.index("touch.duration_ms")
    cross_index = names.index("cross.magnitude_correlation")
    scale_invariant = tuple(sorted(set(
        relative_indices
        + direction_indices
        + [cross_index, duration_index]
    )))
    magnitude_only = tuple(sorted(set(
        magnitude_indices + [cross_index, duration_index]
    )))
    no_gravity = tuple(sorted(set(
        all_indices
    ) - set(gravity_indices)))
    return {
        "full": all_indices,
        "no_gravity": no_gravity,
        "magnitude": magnitude_only,
        "scale_invariant": scale_invariant,
    }


def build_window_features(
    times_ms,
    acceleration,
    rotation,
    coordinates,
    gravity,
    window_end_ms,
):
    """Build one causal window from touch-down through the requested horizon."""
    pretouch = (
        (times_ms >= PRETOUCH_START_MS)
        & (times_ms <= PRETOUCH_END_MS)
    )
    if int(np.sum(pretouch)) < MIN_PRETOUCH_FRAMES:
        raise RuntimeError("insufficient_pretouch_frames")
    selected = (
        pretouch
        | ((times_ms >= 0.0) & (times_ms <= float(window_end_ms)))
    )
    if int(np.sum(selected)) < MIN_WINDOW_FRAMES:
        raise RuntimeError("insufficient_window_frames")

    gravity_reference = np.median(gravity[pretouch], axis=0)
    rotation_matrix = rotation_to_canonical(gravity_reference)
    aligned_acceleration = acceleration @ rotation_matrix.T
    aligned_rotation = rotation @ rotation_matrix.T
    acceleration_magnitude = np.linalg.norm(aligned_acceleration, axis=1)
    rotation_magnitude = np.linalg.norm(aligned_rotation, axis=1)
    acceleration_baseline = max(
        float(np.median(acceleration_magnitude[pretouch])),
        EPSILON,
    )
    rotation_baseline = max(
        float(np.median(rotation_magnitude[pretouch])),
        EPSILON,
    )

    values = []
    for axis in range(2):
        values.extend(stats(coordinates[selected, axis]))
    for axis in range(3):
        values.extend(stats(aligned_acceleration[selected, axis]))
    for axis in range(3):
        values.extend(stats(aligned_rotation[selected, axis]))
    window_acceleration = acceleration_magnitude[selected]
    window_rotation = rotation_magnitude[selected]
    values.extend(stats(window_acceleration))
    values.extend(stats(window_rotation))

    for channel, baseline in (
        (window_acceleration, acceleration_baseline),
        (window_rotation, rotation_baseline),
    ):
        relative = channel / baseline
        center = float(np.median(relative))
        mad = float(np.median(np.abs(relative - center)))
        values.extend((
            center,
            float(np.percentile(relative, 90.0)),
            float(np.percentile(relative, 95.0)),
            float(np.max(relative)),
            center / max(mad, EPSILON),
            float(np.percentile(relative, 90.0)) / max(center, EPSILON),
        ))

    for vector in (
        aligned_acceleration[selected],
        aligned_rotation[selected],
    ):
        peak = vector[int(np.argmax(np.linalg.norm(vector, axis=1)))]
        values.extend((peak / max(float(np.linalg.norm(peak)), EPSILON)).tolist())

    gravity_norm = float(np.linalg.norm(gravity_reference))
    canonical_gravity = (
        gravity_reference / gravity_norm
        if gravity_norm > EPSILON
        else np.zeros(3, dtype=np.float64)
    )
    values.extend(canonical_gravity.tolist())
    if (
        float(np.std(window_acceleration)) > EPSILON
        and float(np.std(window_rotation)) > EPSILON
    ):
        correlation = float(np.corrcoef(
            window_acceleration,
            window_rotation,
        )[0, 1])
    else:
        correlation = 0.0
    duration_ms = float(times_ms[selected][-1])
    values.extend((correlation, duration_ms))
    return np.asarray(values, dtype=np.float32)


def extract_one_capture(arguments):
    """Read one source capture and return all requested window features."""
    path_string, windows_ms = arguments
    path = Path(path_string)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        context = data.get("collectionContext") or {}
        label = data.get("label") or {}
        strength = str(label.get("touchStrength") or "")
        finger = str(label.get("fingerType") or "")
        if strength not in ("light", "heavy"):
            return None, "label_strength"
        if finger not in ("thumb", "index"):
            return None, "label_finger"
        frames = data.get("frames") or []
        if not frames:
            return None, "missing_frames"
        times_ms = np.asarray(
            [
                finite(frame.get("timeSinceTouchDown")) * 1000.0
                for frame in frames
            ],
            dtype=np.float64,
        )
        acceleration = np.asarray(
            [
                (
                    finite(frame.get("x")),
                    finite(frame.get("y")),
                    finite(frame.get("z")),
                )
                for frame in frames
            ],
            dtype=np.float64,
        )
        rotation = np.asarray(
            [
                (
                    finite(frame.get("rotationRateX")),
                    finite(frame.get("rotationRateY")),
                    finite(frame.get("rotationRateZ")),
                )
                for frame in frames
            ],
            dtype=np.float64,
        )
        gravity = np.asarray(
            [
                (
                    finite(frame.get("gravityX")),
                    finite(frame.get("gravityY")),
                    finite(frame.get("gravityZ")),
                )
                for frame in frames
            ],
            dtype=np.float64,
        )
        coordinates = np.asarray(
            [
                (
                    finite(frame.get("x_norm")),
                    finite(frame.get("y_norm")),
                )
                for frame in frames
            ],
            dtype=np.float64,
        )
        alignment = data.get("timeAlignment") or {}
        duration_ms = (
            finite(alignment.get("liftEventElapsedRealtimeNanos"))
            - finite(alignment.get("downEventElapsedRealtimeNanos"))
        ) / 1_000_000.0
        posture = str(context.get("bodyPosture") or "unspecified")
        task_id = str(context.get("taskId") or "unknown")
        if posture == "unspecified" and task_id == "tilted_handheld":
            posture = "tilted"
        metadata = {
            "classLabel": finger + "_" + strength,
            "finger": finger,
            "strength": strength,
            "posture": posture,
            "taskId": task_id,
            "sessionId": str(data.get("collectionSessionID") or ""),
            "sourceFile": path.name,
            "durationMs": duration_ms,
        }
        features = {}
        for window_end_ms in windows_ms:
            features[int(window_end_ms)] = build_window_features(
                times_ms,
                acceleration,
                rotation,
                coordinates,
                gravity,
                int(window_end_ms),
            )
        return (metadata, features), None
    except Exception as error:
        return None, type(error).__name__


def extract_dataset(
    dataset_dir,
    output_dir,
    windows_ms,
    workers,
):
    """Extract all source captures in parallel and persist compact arrays."""
    paths = sorted(Path(dataset_dir).glob("*.json"))
    if not paths:
        raise RuntimeError("source_dataset_has_no_json_files")
    output_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = output_dir / "windows"
    windows_dir.mkdir(exist_ok=True)
    started = time.time()
    records = []
    failures = Counter()
    arguments = [(str(path), tuple(windows_ms)) for path in paths]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(extract_one_capture, item) for item in arguments]
        for completed, future in enumerate(as_completed(futures), 1):
            record, failure = future.result()
            if failure:
                failures[failure] += 1
            else:
                records.append(record)
            if completed % 5000 == 0:
                print(
                    "extract {}/{} kept={} elapsed={:.1f}s".format(
                        completed,
                        len(paths),
                        len(records),
                        time.time() - started,
                    ),
                    flush=True,
                )
    records.sort(key=lambda record: record[0]["sourceFile"])
    names = feature_names(windows_ms)
    for window_end_ms in windows_ms:
        selected = [
            (metadata, features[int(window_end_ms)])
            for metadata, features in records
        ]
        np.savez_compressed(
            windows_dir / ("window_{}ms.npz".format(window_end_ms)),
            X=np.stack([features for _, features in selected]),
            y=np.asarray(
                [metadata["strength"] == "heavy" for metadata, _ in selected],
                dtype=np.int8,
            ),
            posture=np.asarray(
                [metadata["posture"] for metadata, _ in selected],
            ),
            finger=np.asarray(
                [metadata["finger"] for metadata, _ in selected],
            ),
            session=np.asarray(
                [metadata["sessionId"] for metadata, _ in selected],
            ),
            duration=np.asarray(
                [metadata["durationMs"] for metadata, _ in selected],
                dtype=np.float32,
            ),
            sourceFile=np.asarray(
                [metadata["sourceFile"] for metadata, _ in selected],
            ),
            featureNames=np.asarray(names),
        )
    summary = {
        "datasetDir": str(Path(dataset_dir).resolve()),
        "sourceFileCount": len(paths),
        "acceptedCount": len(records),
        "windowsMs": list(windows_ms),
        "failures": dict(sorted(failures.items())),
        "elapsedSeconds": time.time() - started,
        "classCounts": dict(Counter(
            record[0]["classLabel"] for record in records
        )),
        "postureCounts": dict(Counter(
            record[0]["posture"] for record in records
        )),
        "sessionCount": len({
            record[0]["sessionId"] for record in records
        }),
    }
    (output_dir / "extraction_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def choose_calibration_sessions(
    train_session_classes,
    seed,
    ratio=CALIBRATION_SESSION_RATIO,
):
    """Reserve whole sessions while keeping every class represented."""
    generator = np.random.default_rng(seed)
    selected = set()
    for class_label in sorted(train_session_classes):
        sessions = sorted(train_session_classes[class_label])
        generator.shuffle(sessions)
        count = max(1, int(round(len(sessions) * ratio)))
        count = min(count, len(sessions) - 1)
        selected.update(sessions[:count])
    return selected


def threshold_at_fpr(negative_scores, target_fpr):
    """Select the most inclusive threshold that still meets the target FPR."""
    negative_scores = np.sort(np.asarray(negative_scores, dtype=np.float64))
    if not len(negative_scores):
        raise RuntimeError("missing_threshold_negatives")
    allowed_false_positives = max(
        0,
        int(math.floor(float(target_fpr) * len(negative_scores))),
    )
    index = len(negative_scores) - allowed_false_positives - 1
    return float(negative_scores[max(0, index)])


def binary_metrics(truth, scores, threshold, target_fpr):
    """Return recall, false-positive rate, and proof counts."""
    truth = np.asarray(truth, dtype=np.int8)
    accepted = np.asarray(scores, dtype=np.float64) >= float(threshold)
    positive = truth == 1
    negative = truth == 0
    true_positive = int(np.sum(accepted & positive))
    false_positive = int(np.sum(accepted & negative))
    return {
        "targetFalsePositiveRate": float(target_fpr),
        "threshold": float(threshold),
        "positiveCount": int(np.sum(positive)),
        "negativeCount": int(np.sum(negative)),
        "truePositiveCount": true_positive,
        "falsePositiveCount": false_positive,
        "heavyRecall": (
            true_positive / int(np.sum(positive))
            if np.any(positive)
            else None
        ),
        "lightFalsePositiveRate": (
            false_positive / int(np.sum(negative))
            if np.any(negative)
            else None
        ),
    }


def auc(truth, scores):
    """Compute rank AUC without depending on a plotting library."""
    truth = np.asarray(truth, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = scores[truth == 1]
    negatives = scores[truth == 0]
    if not len(positives) or not len(negatives):
        return None
    combined = np.concatenate((positives, negatives))
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype=np.float64)
    sorted_values = combined[order]
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop + 1) / 2.0
        start = stop
    positive_rank_sum = float(np.sum(ranks[:len(positives)]))
    return (
        positive_rank_sum - len(positives) * (len(positives) + 1) / 2.0
    ) / (len(positives) * len(negatives))


def evaluate_window(
    window_file,
    feature_set,
    seeds,
    target_fpr,
    max_iter,
):
    """Evaluate one feature set with every posture held out in turn."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    payload = np.load(window_file, allow_pickle=False)
    X_all = payload["X"]
    y_all = payload["y"].astype(np.int8)
    posture_all = payload["posture"].astype(str)
    finger_all = payload["finger"].astype(str)
    session_all = payload["session"].astype(str)
    names = tuple(payload["featureNames"].tolist())
    selected_indices = feature_sets(names)[feature_set]
    X_all = X_all[:, selected_indices]

    results = []
    postures = sorted(set(posture_all.tolist()))
    for held_posture in postures:
        train_mask = posture_all != held_posture
        evaluation_mask = posture_all == held_posture
        training_session_classes = defaultdict(set)
        for session, label in zip(
            session_all[train_mask],
            y_all[train_mask],
        ):
            training_session_classes[int(label)].add(str(session))
        for seed in seeds:
            calibration_sessions = choose_calibration_sessions(
                training_session_classes,
                seed,
            )
            calibration_mask = np.asarray([
                str(session) in calibration_sessions
                for session in session_all
            ]) & train_mask
            fit_mask = train_mask & ~calibration_mask
            if not np.any(fit_mask) or not np.any(calibration_mask):
                raise RuntimeError("empty_fit_or_calibration_split")
            model = HistGradientBoostingClassifier(
                max_iter=max_iter,
                learning_rate=0.06,
                max_leaf_nodes=15,
                min_samples_leaf=50,
                l2_regularization=2.0,
                early_stopping=False,
                random_state=seed,
            )
            model.fit(X_all[fit_mask], y_all[fit_mask])
            calibration_scores = model.predict_proba(
                X_all[calibration_mask]
            )[:, 1]
            calibration_truth = y_all[calibration_mask]
            evaluation_scores = model.predict_proba(
                X_all[evaluation_mask]
            )[:, 1]
            evaluation_truth = y_all[evaluation_mask]
            calibration_negatives = calibration_scores[
                calibration_truth == 0
            ]
            evaluation_negatives = evaluation_scores[
                evaluation_truth == 0
            ]
            transferred_threshold = threshold_at_fpr(
                calibration_negatives,
                target_fpr,
            )
            oracle_threshold = threshold_at_fpr(
                evaluation_negatives,
                target_fpr,
            )
            groups = {}
            for finger in sorted(set(finger_all[evaluation_mask].tolist())):
                group_mask = finger_all[evaluation_mask] == finger
                groups[finger] = {
                    "auc": auc(
                        evaluation_truth[group_mask],
                        evaluation_scores[group_mask],
                    ),
                    "transferred": binary_metrics(
                        evaluation_truth[group_mask],
                        evaluation_scores[group_mask],
                        transferred_threshold,
                        target_fpr,
                    ),
                    "oracleThresholdDiagnostic": binary_metrics(
                        evaluation_truth[group_mask],
                        evaluation_scores[group_mask],
                        oracle_threshold,
                        target_fpr,
                    ),
                }
            results.append({
                "windowFile": str(window_file),
                "featureSet": feature_set,
                "featureCount": len(selected_indices),
                "heldPosture": held_posture,
                "seed": seed,
                "fitCount": int(np.sum(fit_mask)),
                "calibrationCount": int(np.sum(calibration_mask)),
                "evaluationCount": int(np.sum(evaluation_mask)),
                "fitSessionCount": len(set(session_all[fit_mask].tolist())),
                "calibrationSessionCount": len(calibration_sessions),
                "evaluationSessionCount": len(set(
                    session_all[evaluation_mask].tolist()
                )),
                "calibrationNegativeCount": int(
                    np.sum(calibration_truth == 0)
                ),
                "evaluationNegativeCount": int(
                    np.sum(evaluation_truth == 0)
                ),
                "proofReady": (
                    int(np.sum(calibration_truth == 0))
                    >= MIN_CALIBRATION_NEGATIVES
                    and int(np.sum(evaluation_truth == 0))
                    >= MIN_EVALUATION_NEGATIVES
                ),
                "auc": auc(evaluation_truth, evaluation_scores),
                "transferred": binary_metrics(
                    evaluation_truth,
                    evaluation_scores,
                    transferred_threshold,
                    target_fpr,
                ),
                "oracleThresholdDiagnostic": binary_metrics(
                    evaluation_truth,
                    evaluation_scores,
                    oracle_threshold,
                    target_fpr,
                ),
                "groups": groups,
            })
    return results


def summarize_results(results):
    """Aggregate the worst posture and worst finger for each configuration."""
    grouped = defaultdict(list)
    for result in results:
        window_ms = int(
            Path(result["windowFile"])
            .stem.removeprefix("window_")
            .removesuffix("ms")
        )
        grouped[(
            result["windowFile"],
            window_ms,
            result["featureSet"],
        )].append(result)
    summaries = []
    for (window_file, window_ms, feature_set), values in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][1],
            item[0][2],
        ),
    ):
        transferred_recalls = [
            value["transferred"]["heavyRecall"]
            for value in values
            if value["transferred"]["heavyRecall"] is not None
        ]
        oracle_recalls = [
            value["oracleThresholdDiagnostic"]["heavyRecall"]
            for value in values
            if value["oracleThresholdDiagnostic"]["heavyRecall"] is not None
        ]
        transferred_fprs = [
            value["transferred"]["lightFalsePositiveRate"]
            for value in values
            if value["transferred"]["lightFalsePositiveRate"] is not None
        ]
        group_recalls = []
        for value in values:
            for group in value["groups"].values():
                recall = group["transferred"]["heavyRecall"]
                if recall is not None:
                    group_recalls.append(recall)
        summaries.append({
            "windowFile": window_file,
            "windowMs": window_ms,
            "featureSet": feature_set,
            "featureCount": values[0]["featureCount"],
            "meanAuc": float(np.mean([
                value["auc"] for value in values
                if value["auc"] is not None
            ])),
            "meanTransferredRecall": float(np.mean(transferred_recalls)),
            "worstPostureTransferredRecall": float(min(transferred_recalls)),
            "worstFingerTransferredRecall": float(min(group_recalls)),
            "meanOracleRecall": float(np.mean(oracle_recalls)),
            "worstPostureOracleRecall": float(min(oracle_recalls)),
            "meanTransferredFpr": float(np.mean(transferred_fprs)),
            "worstTransferredFpr": float(max(transferred_fprs)),
            "proofReady": all(value["proofReady"] for value in values),
        })
    return summaries


def parse_integer_list(value):
    """Parse comma-separated integers with stable validation."""
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("list_must_not_be_empty")
    return parsed


def run(args):
    """Execute extraction and evaluation for the requested experiment."""
    windows_ms = tuple(args.windows_ms)
    output_dir = Path(args.output_dir)
    if args.action in ("extract", "all"):
        extraction = extract_dataset(
            dataset_dir=args.dataset_dir,
            output_dir=output_dir,
            windows_ms=windows_ms,
            workers=args.workers,
        )
        print(json.dumps(extraction, ensure_ascii=False, indent=2))
    if args.action in ("evaluate", "all"):
        all_results = []
        for window_end_ms in windows_ms:
            window_file = (
                output_dir
                / "windows"
                / ("window_{}ms.npz".format(window_end_ms))
            )
            if not window_file.exists():
                raise RuntimeError("missing_window_file:" + str(window_file))
            for feature_set in args.feature_sets:
                print(
                    "evaluate {} {}".format(window_end_ms, feature_set),
                    flush=True,
                )
                all_results.extend(evaluate_window(
                    window_file=window_file,
                    feature_set=feature_set,
                    seeds=args.seeds,
                    target_fpr=args.target_fpr,
                    max_iter=args.max_iter,
                ))
        summaries = summarize_results(all_results)
        report = {
            "scope": "posture_holdout_temporal_evidence_ablation",
            "targetFalsePositiveRate": args.target_fpr,
            "targetRecall": args.target_recall,
            "releaseEligible": any(
                summary["worstPostureTransferredRecall"]
                >= args.target_recall
                and summary["worstTransferredFpr"] <= args.target_fpr
                and summary["proofReady"]
                for summary in summaries
            ),
            "negativeProxy": "light_touch",
            "ordinaryClickEvidenceAvailable": False,
            "summaries": summaries,
            "results": all_results,
        }
        (output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summaries, ensure_ascii=False, indent=2))


def main():
    """Parse command-line options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=("extract", "evaluate", "all"),
        default="all",
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--windows-ms",
        type=parse_integer_list,
        default=DEFAULT_WINDOWS_MS,
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=("full", "no_gravity", "magnitude", "scale_invariant"),
        default=("full", "magnitude", "scale_invariant"),
    )
    parser.add_argument(
        "--seeds",
        type=parse_integer_list,
        default=(20261022,),
    )
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--max-iter", type=int, default=180)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("workers_must_be_positive")
    if not 0.0 < args.target_fpr < 1.0:
        parser.error("target_fpr_must_be_between_zero_and_one")
    if not 0.0 < args.target_recall <= 1.0:
        parser.error("target_recall_must_be_between_zero_and_one")
    run(args)


if __name__ == "__main__":
    main()
