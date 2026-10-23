#!/usr/bin/env python3
"""Compare heavy-touch gate candidates on one frozen external audit."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from . import gate_threshold_metrics
    from . import run_gate_nested_validation as gate
except ImportError:
    import gate_threshold_metrics
    import run_gate_nested_validation as gate


MODEL_NAMES = (
    "stats_mlp8_gate",
    "stats_mlp16",
    "stats_mlp16_gate",
    "stats_mlp16_context_gate",
    "stats_mlp24_gate",
    "stats_mlp24_multitask",
    "logistic_regression",
    "hist_gradient_boosting",
)
MIN_SCENARIO_CALIBRATION_NEGATIVES = 500
PROFILE_SPECS = {
    "magnitude_calibrated": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "normalizationMode": "pretouch_global_iqr",
    },
    "magnitude_device_iqr": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "normalizationMode": "pretouch_device_iqr",
    },
    "magnitude_causal_iqr": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "normalizationMode": "pretouch_causal_iqr",
    },
    "magnitude_causal_iqr_shrunk": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "normalizationMode": "pretouch_causal_iqr_shrunk",
    },
    "magnitude_causal_quantile": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "normalizationMode": "pretouch_causal_quantile",
    },
    "magnitude_delta": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "imu_delta",
    },
    "magnitude_raw": {
        "featureMode": "magnitude_only",
        "imuMode": "raw",
        "adaptiveFeatureMode": "none",
    },
    "magnitude_aligned": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned",
        "adaptiveFeatureMode": "none",
    },
    "raw": {
        "featureMode": "full",
        "imuMode": "raw",
    },
    "compact": {
        "featureMode": "compact",
        "imuMode": "gravity_aligned",
        "adaptiveFeatureMode": "none",
    },
    "relative": {
        "featureMode": "no_absolute_xy",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "imu_delta_shape",
    },
    "magnitude_only": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
    },
    "magnitude_early_press_shape": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "shape",
    },
    "magnitude_early_acceleration": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "acceleration",
    },
    "magnitude_early_rotation": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "rotation",
    },
    "magnitude_early_rotation_contact_relative": {
        "featureMode": "contact_relative_magnitude",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "rotation",
    },
    "magnitude_early_rotation_contact_shape": {
        "featureMode": "contact_relative_magnitude_shape",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "rotation",
    },
    "magnitude_early_rotation_contact_delta": {
        "featureMode": "magnitude_contact_relative",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "rotation",
    },
    "magnitude_early_rotation_channel_scale": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "rotation",
        "augmentationMode": "channel_scale",
        "augmentationProbability": 0.5,
        "augmentationMaximumLogScale": 0.35,
    },
    "magnitude_early_rotation_holdout_consensus": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "rotation",
        "calibrationConsensusRepeats": 5,
    },
    "magnitude_early_dense_pose_generalization": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "dense_shape_no_gain",
        "calibrationConsensusRepeats": 5,
    },
    "magnitude_early_dense_acceleration_pose": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "dense_acceleration",
        "calibrationConsensusRepeats": 5,
    },
    "magnitude_early_dense_rotation_pose": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "dense_rotation",
        "calibrationConsensusRepeats": 5,
    },
    "magnitude_early_dense_cross_pose": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "dense_cross",
        "calibrationConsensusRepeats": 5,
    },
    "magnitude_early_dense_scale_invariant_pose": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "dense_scale_invariant",
        "calibrationConsensusRepeats": 5,
    },
    "magnitude_early_dense_scale_invariant_no_cross_pose": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "dense_scale_invariant_no_cross",
        "calibrationConsensusRepeats": 5,
    },
    "magnitude_early_rotation_dense_scale_invariant_pose": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "rotation_plus_dense_scale_invariant",
        "calibrationConsensusRepeats": 5,
    },
    "magnitude_early_rotation_peak_shape": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "rotation_peak_shape",
    },
    "magnitude_early_rotation_peak_shape_no_gain": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "rotation_peak_shape_no_gain",
    },
    "magnitude_early_rotation_log_ratio": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "rotation_log_ratio",
    },
    "magnitude_early_acceleration_post": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "acceleration_post",
    },
    "magnitude_early_onset_shape": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "onset_shape",
    },
    "magnitude_early_acceleration_shape": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "acceleration_shape",
    },
    "magnitude_early_rotation_shape": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "rotation_shape",
    },
    "magnitude_early_acceleration_log_ratio": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "acceleration_log_ratio",
    },
    "magnitude_early_log_ratio": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "log_ratio",
    },
    "magnitude_early_acceleration_onset": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "acceleration_onset",
    },
    "magnitude_early_acceleration_onset_causal_iqr_shrunk": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "normalizationMode": "pretouch_causal_iqr_shrunk",
        "earlyEvidenceMode": "acceleration_onset",
    },
    "magnitude_early_acceleration_onset_causal_quantile": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "normalizationMode": "pretouch_causal_quantile",
        "earlyEvidenceMode": "acceleration_onset",
    },
    "magnitude_early_acceleration_onset_global_iqr": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "normalizationMode": "pretouch_global_iqr",
        "earlyEvidenceMode": "acceleration_onset",
    },
    "magnitude_early_acceleration_onset_post": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
        "earlyEvidenceMode": "acceleration_onset_post",
    },
    "magnitude_guarded_pretouch": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_guarded_pretouch_centered",
        "adaptiveFeatureMode": "none",
    },
    "magnitude_earliest_pretouch": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_earliest_pretouch_centered",
        "adaptiveFeatureMode": "none",
    },
    "magnitude_temporal": {
        "featureMode": "magnitude_temporal",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "none",
    },
    "magnitude_delta": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned_pretouch_centered",
        "adaptiveFeatureMode": "imu_delta",
    },
    "magnitude_raw": {
        "featureMode": "magnitude_only",
        "imuMode": "raw",
        "adaptiveFeatureMode": "none",
    },
    "magnitude_aligned": {
        "featureMode": "magnitude_only",
        "imuMode": "gravity_aligned",
        "adaptiveFeatureMode": "none",
    },
    "device_relative_magnitude": {
        "featureMode": "device_relative_magnitude",
        "imuMode": "raw",
        "adaptiveFeatureMode": "none",
    },
    "physical_magnitude": {
        "featureMode": "physical_magnitude",
        "imuMode": "raw",
        "adaptiveFeatureMode": "none",
    },
    "physical_magnitude_temporal": {
        "featureMode": "physical_magnitude_temporal",
        "imuMode": "raw",
        "adaptiveFeatureMode": "none",
    },
    "physical_magnitude_shape": {
        "featureMode": "physical_magnitude_shape",
        "imuMode": "raw",
        "adaptiveFeatureMode": "none",
    },
    "physical_magnitude_temporal_shape": {
        "featureMode": "physical_magnitude_temporal_shape",
        "imuMode": "raw",
        "adaptiveFeatureMode": "none",
    },
}


def parse_values(value, allowed, name):
    """Parse a comma-separated list and reject unknown names."""
    selected = tuple(
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    )
    invalid = sorted(set(selected) - set(allowed))
    if invalid:
        raise RuntimeError("invalid_" + name + ":" + ",".join(invalid))
    if not selected:
        raise RuntimeError("empty_" + name)
    return selected


def attach_evidence_fields(
        report,
        calibration_rows,
        calibration_gate_labels,
        valid_rows,
        valid_gate_labels,
        target_fpr,
):
    """Attach count and session sufficiency without changing the metric."""
    minimum_negatives = int(math.ceil(5.0 / float(target_fpr)))
    calibration_sessions = {
        str(row.get("groupId") or "").strip()
        for row, label in zip(
            calibration_rows,
            calibration_gate_labels,
        )
        if int(label) == 0
        and str(row.get("groupId") or "").strip()
    }
    valid_sessions = {
        str(row.get("groupId") or "").strip()
        for row, label in zip(valid_rows, valid_gate_labels)
        if int(label) == 0
        and str(row.get("groupId") or "").strip()
    }
    report["targetFalsePositiveRate"] = float(target_fpr)
    report["calibrationNegativeCount"] = int(
        np.count_nonzero(np.asarray(calibration_gate_labels) == 0)
    )
    report["heldOutNegativeCount"] = int(
        np.count_nonzero(np.asarray(valid_gate_labels) == 0)
    )
    report["heldOutPositiveCount"] = int(
        np.count_nonzero(np.asarray(valid_gate_labels) == 1)
    )
    report["calibrationSessionCount"] = len(calibration_sessions)
    report["heldOutNegativeSessionCount"] = len(valid_sessions)
    report["minimumProofNegativeCount"] = minimum_negatives
    report["minimumProofPositiveCount"] = gate.MIN_GROUP_POSITIVES
    report["minimumProofSessionCount"] = gate.MIN_PROOF_SESSION_COUNT
    report["sessionDisjoint"] = not bool(
        calibration_sessions & valid_sessions
    )
    report["proofReady"] = bool(
        report["calibrationNegativeCount"] >= minimum_negatives
        and report["heldOutNegativeCount"] >= minimum_negatives
        and report["heldOutPositiveCount"] >= gate.MIN_GROUP_POSITIVES
        and report["calibrationSessionCount"]
        >= gate.MIN_PROOF_SESSION_COUNT
        and report["heldOutNegativeSessionCount"]
        >= gate.MIN_PROOF_SESSION_COUNT
        and report["sessionDisjoint"]
    )
    report["evidenceStatus"] = (
        "proof_ready" if report["proofReady"] else "descriptive_only"
    )
    return report


def known_group_values(metrics, group_name):
    """Exclude unlabelled buckets from deployment proof summaries."""
    groups = metrics.get("groups", {}).get(group_name, {})
    return [
        value
        for name, value in groups.items()
        if str(name).strip().lower() not in ("", "unknown", "unspecified")
    ]


def fit_candidate(
        name,
        profile_name,
        fit_values,
        fit_labels,
        fit_gate_labels,
        fit_context_labels,
        sample_weights,
        calibration_values,
        calibration_gate_labels,
        calibration_motion_domains,
        seed,
        epochs,
        patience,
):
    """Fit one candidate without reading external evaluation labels."""
    if name in (
        "stats_mlp8_gate",
        "stats_mlp16",
        "stats_mlp16_gate",
        "stats_mlp16_context_gate",
        "stats_mlp24_gate",
        "stats_mlp24_multitask",
    ):
        hidden_count = {
            "stats_mlp8_gate": 8,
            "stats_mlp16": 16,
            "stats_mlp16_gate": 16,
            "stats_mlp16_context_gate": 16,
            "stats_mlp24_gate": 24,
            "stats_mlp24_multitask": 24,
        }[name]
        finger_loss_weight = (
            0.20
            if name in ("stats_mlp16", "stats_mlp24_multitask")
            else 0.0
        )
        context_heads = name == "stats_mlp16_context_gate"
        model, selection = gate.train_model(
            fit_values,
            fit_labels,
            fit_gate_labels,
            fit_context_labels,
            sample_weights,
            calibration_values,
            calibration_gate_labels,
            calibration_motion_domains,
            fit_values.shape[1],
            hidden_count,
            seed,
            epochs,
            context_heads,
            0.08 if context_heads else 0.0,
            finger_loss_weight,
            patience,
        )
        return (
            lambda values: gate.gate_probabilities(model, values),
            int(sum(
                parameter.numel()
                for parameter in model.parameters()
            )),
            {
                **selection,
                "family": "stats_mlp",
                "hiddenCount": hidden_count,
                "multitask": finger_loss_weight > 0.0,
                "contextHeads": context_heads,
            },
        )
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:
        raise RuntimeError("scikit_learn_is_required") from error
    labels = np.asarray(fit_gate_labels, dtype=np.int64)
    weights = np.asarray(sample_weights, dtype=np.float64)
    if name == "logistic_regression":
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
        )
        model.fit(fit_values, labels, sample_weight=weights)
        return (
            lambda values: model.predict_proba(values)[:, 1],
            int(model.coef_.size + model.intercept_.size),
            {
                "family": "linear",
                "solver": "lbfgs",
                "iterations": int(model.n_iter_[0]),
            },
        )
    if name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=240,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=seed,
        )
        model.fit(fit_values, labels, sample_weight=weights)
        tree_count = int(model.n_iter_)
        return (
            lambda values: model.predict_proba(values)[:, 1],
            tree_count,
            {
                "family": "boosted_trees",
                "iterations": tree_count,
            },
        )
    raise RuntimeError("unsupported_model:" + profile_name + ":" + name)


def metric_value(metrics, key):
    """Return a finite metric or None."""
    value = metrics.get(key)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def scenario_thresholds_at_fpr(
        negatives_by_scenario,
        fallback_threshold,
        target_fpr,
):
    """Fit scenario thresholds only when negative evidence is sufficient."""
    thresholds = {}
    for scenario, scores in sorted(negatives_by_scenario.items()):
        thresholds[str(scenario)] = (
            gate_threshold_metrics.select_threshold_at_fpr(
                scores,
                target_fpr,
            )
            if len(scores) >= MIN_SCENARIO_CALIBRATION_NEGATIVES
            else float(fallback_threshold)
        )
    return thresholds


def calibration_consensus_scores(
        rows,
        gate_labels,
        scores,
        repeats,
        seed,
):
    """Estimate a robust threshold from repeated complete-session subsets."""
    if repeats <= 1:
        raise RuntimeError("invalid_calibration_consensus_repeats")
    labels = np.asarray(gate_labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    sessions = defaultdict(list)
    for row, label, score in zip(rows, labels, scores):
        if int(label) != 0:
            continue
        # 中文注释：生产调用只传入校准集；这里再显式拒绝带分区标记的外测行，
        # 防止以后误把验证集或外测集拼进阈值估计。
        partition = str(row.get("partition") or "calibration").strip().lower()
        if partition != "calibration":
            continue
        session = str(row.get("groupId") or "").strip()
        if session:
            sessions[session].append(float(score))
    if len(sessions) < 2:
        raise RuntimeError("calibration_missing_negative_proxy")
    session_ids = sorted(sessions)
    subset_count = max(
        1,
        int(math.ceil(len(session_ids) * 0.8)),
    )
    thresholds = {}
    for target in gate.TARGET_FPRS:
        selected = []
        for repeat in range(repeats):
            chosen = np.random.default_rng(
                seed + repeat * 1009
            ).choice(
                session_ids,
                size=subset_count,
                replace=False,
            )
            values = [
                score
                for session in chosen
                for score in sessions[str(session)]
            ]
            selected.append(
                gate_threshold_metrics.select_threshold_at_fpr(
                    values,
                    target,
                )
            )
        # 中文注释：重复只用于估计阈值稳定性。阈值取中位数，
        # 不接触外测标签，也不使用外测分数。
        thresholds[str(target)] = float(np.median(selected))
    return thresholds


def summarize_candidates(runs):
    """Aggregate one model/profile pair across devices and seeds."""
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["model"], run["profile"])].append(run)
    summaries = []
    for (model_name, profile_name), selected in sorted(grouped.items()):
        for target in map(str, gate.TARGET_FPRS):
            target_runs = [
                run for run in selected
                if str(run["targetFalsePositiveRate"]) == target
            ]
            recalls = [
                metric_value(run["metrics"]["overall"], "heavyRecall")
                for run in target_runs
            ]
            fprs = [
                metric_value(
                    run["metrics"]["overall"],
                    "lightFalsePositiveRate",
                )
                for run in target_runs
            ]
            device_recalls = [
                metric_value(metrics, "heavyRecall")
                for run in target_runs
                for metrics in known_group_values(
                    run["metrics"],
                    "deviceName",
                )
            ]
            posture_recalls = [
                metric_value(metrics, "heavyRecall")
                for run in target_runs
                for metrics in known_group_values(
                    run["metrics"],
                    "postureDomain",
                )
            ]
            motion_recalls = [
                metric_value(metrics, "heavyRecall")
                for run in target_runs
                for metrics in known_group_values(
                    run["metrics"],
                    "motionState",
                )
            ]
            posture_group_names = {
                str(name).strip().lower()
                for run in target_runs
                for name in run["metrics"].get("groups", {}).get(
                    "postureDomain",
                    {},
                )
                if str(name).strip().lower() not in (
                    "",
                    "unknown",
                    "unspecified",
                )
            }
            motion_group_names = {
                str(name).strip().lower()
                for run in target_runs
                for name in run["metrics"].get("groups", {}).get(
                    "motionState",
                    {},
                )
                if str(name).strip().lower() not in (
                    "",
                    "unknown",
                    "unspecified",
                )
            }
            recalls = [value for value in recalls if value is not None]
            fprs = [value for value in fprs if value is not None]
            device_recalls = [
                value for value in device_recalls if value is not None
            ]
            posture_recalls = [
                value for value in posture_recalls if value is not None
            ]
            motion_recalls = [
                value for value in motion_recalls if value is not None
            ]
            summaries.append({
                "model": model_name,
                "profile": profile_name,
                "targetFalsePositiveRate": float(target),
                "runCount": len(target_runs),
                "seedCount": len({run["seed"] for run in target_runs}),
                "deviceCount": len({
                    run["foldId"] for run in target_runs
                }),
                "meanHeavyRecall": (
                    float(np.mean(recalls)) if recalls else None
                ),
                "worstDeviceHeavyRecall": (
                    float(min(device_recalls))
                    if device_recalls
                    else None
                ),
                "worstPostureHeavyRecall": (
                    float(min(posture_recalls))
                    if posture_recalls
                    else None
                ),
                "worstMotionHeavyRecall": (
                    float(min(motion_recalls))
                    if motion_recalls
                    else None
                ),
                "postureGroupCount": len(posture_group_names),
                "motionGroupCount": len(motion_group_names),
                "meanLightFalsePositiveRate": (
                    float(np.mean(fprs)) if fprs else None
                ),
                "worstDeviceLightFalsePositiveRate": (
                    float(max(fprs)) if fprs else None
                ),
                "proofReady": bool(
                    target_runs
                    and all(run["proofReady"] for run in target_runs)
                ),
                "meanRuntimeParameterCount": float(np.mean([
                    run["runtimeParameterCount"]
                    for run in target_runs
                ])),
            })
    return summaries


def rank_candidates(summaries):
    """Rank 1 percent FPR candidates by worst-device recall then size."""
    primary = [
        summary for summary in summaries
        if summary["targetFalsePositiveRate"] == (
            float(gate.DEPLOYMENT_PRIMARY_FPR)
        )
    ]

    def score(summary):
        def value_or_negative(key):
            value = summary.get(key)
            return value if value is not None else -1.0

        return (
            -value_or_negative("worstDeviceHeavyRecall"),
            -value_or_negative("worstPostureHeavyRecall"),
            -value_or_negative("worstMotionHeavyRecall"),
            -value_or_negative("meanHeavyRecall"),
            summary["meanRuntimeParameterCount"],
        )

    return sorted(primary, key=score)


def assess_release_candidate(candidate, device_count):
    """Evaluate one threshold policy with every deployment blocker visible."""
    blockers = []
    if candidate is None:
        return {
            "eligible": False,
            "blockers": ["candidate_unavailable"],
        }
    if int(device_count) < gate.DEPLOYMENT_MIN_DEVICES:
        blockers.append("insufficient_device_evidence")
    if int(candidate.get("seedCount", 0)) < gate.DEPLOYMENT_MIN_SEEDS:
        blockers.append("insufficient_seed_evidence")
    if not candidate.get("proofReady", False):
        blockers.append("negative_evidence_not_proof_ready")
    recall_fields = (
        ("worstDeviceHeavyRecall", "device_recall_below_target"),
        ("worstPostureHeavyRecall", "posture_recall_unavailable"),
        ("worstMotionHeavyRecall", "motion_recall_below_target"),
    )
    for field, blocker in recall_fields:
        value = candidate.get(field)
        if value is None:
            blockers.append(blocker.replace(
                "_below_target",
                "_unavailable",
            ))
        elif float(value) < gate.DEPLOYMENT_RECALL_TARGET:
            blockers.append(blocker)
    worst_fpr = candidate.get("worstDeviceLightFalsePositiveRate")
    if worst_fpr is None:
        blockers.append("false_positive_rate_unavailable")
    elif float(worst_fpr) > float(gate.DEPLOYMENT_PRIMARY_FPR) + 1e-12:
        blockers.append("false_positive_rate_above_target")
    return {
        "eligible": not blockers,
        "blockers": blockers,
        "minimumHeavyRecall": gate.DEPLOYMENT_RECALL_TARGET,
        "maximumLightFalsePositiveRate": float(
            gate.DEPLOYMENT_PRIMARY_FPR
        ),
        "minimumSeedCount": gate.DEPLOYMENT_MIN_SEEDS,
        "minimumDeviceCount": gate.DEPLOYMENT_MIN_DEVICES,
    }


def render_markdown(report):
    """Render global and scenario threshold comparisons."""
    lines = [
        "# Heavy gate candidate comparison",
        "",
        "## Global threshold",
        "",
        "| model | profile | mean recall | worst device | "
        "worst posture | worst motion | proof | parameters |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in report["rankedCandidates"]:
        def percent(value):
            return (
                "{:.2f}%".format(100.0 * value)
                if value is not None
                else "n/a"
            )

        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {:.0f} |".format(
                row["model"],
                row["profile"],
                percent(row["meanHeavyRecall"]),
                percent(row["worstDeviceHeavyRecall"]),
                percent(row["worstPostureHeavyRecall"]),
                percent(row["worstMotionHeavyRecall"]),
                "yes" if row["proofReady"] else "no",
                row["meanRuntimeParameterCount"],
            )
        )
    lines.extend((
        "",
        "## Scenario threshold",
        "",
        "| model | profile | mean recall | worst device | "
        "worst posture | worst motion | mean light FPR | proof | parameters |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ))
    for row in report.get("scenarioRankedCandidates", []):
        def percent(value):
            return (
                "{:.2f}%".format(100.0 * value)
                if value is not None
                else "n/a"
            )

        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {:.0f} |".format(
                row["model"],
                row["profile"],
                percent(row["meanHeavyRecall"]),
                percent(row["worstDeviceHeavyRecall"]),
                percent(row["worstPostureHeavyRecall"]),
                percent(row["worstMotionHeavyRecall"]),
                percent(row["meanLightFalsePositiveRate"]),
                "yes" if row["proofReady"] else "no",
                row["meanRuntimeParameterCount"],
            )
        )
    lines.extend((
        "",
        "## Release assessment",
        "",
        "| policy | eligible | blockers |",
        "| --- | --- | --- |",
    ))
    for policy, assessment in report.get(
            "releaseAssessments",
            {}).items():
        lines.append("| {} | {} | {} |".format(
            policy,
            "yes" if assessment["eligible"] else "no",
            ", ".join(assessment["blockers"]) or "none",
        ))
    lines.extend((
        "",
        "External rows are used for audit only. A proof-ready result also "
        "requires the session and negative-count conditions in the JSON.",
        "",
    ))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", action="append", required=True)
    parser.add_argument(
        "--evaluation-only-dataset-dir",
        action="append",
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output")
    parser.add_argument("--score-output")
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--profiles", default=",".join(PROFILE_SPECS))
    parser.add_argument(
        "--early-evidence-file",
        action="append",
        default=[],
        help="JSONL sidecar produced by build_early_press_evidence.py.",
    )
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20261023)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--calibration-ratio", type=float, default=0.22)
    parser.add_argument(
        "--min-scenario-gate-sessions",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--weighting",
        choices=gate.WEIGHTING_MODES,
        default="device_balanced",
    )
    args = parser.parse_args()
    if args.seed_count <= 0:
        raise RuntimeError("invalid_seed_count")
    if args.epochs <= 0 or args.patience <= 0:
        raise RuntimeError("invalid_training_budget")
    models = parse_values(args.models, MODEL_NAMES, "models")
    profiles = parse_values(args.profiles, tuple(PROFILE_SPECS), "profiles")

    training_rows = gate.load_rows(
        args.dataset_dir,
        early_evidence_files=args.early_evidence_file,
    )
    all_evaluation_rows = gate.load_evaluation_only_rows(
        args.evaluation_only_dataset_dir,
        training_rows,
        early_evidence_files=args.early_evidence_file,
    )
    training_devices = {
        gate.physical_device_key(row) for row in training_rows
    }
    evaluation_rows = [
        row for row in all_evaluation_rows
        if gate.physical_device_key(row) not in training_devices
    ]
    excluded_seen_devices = sorted({
        gate.physical_device_key(row)
        for row in all_evaluation_rows
        if gate.physical_device_key(row) in training_devices
    })
    excluded_seen_row_count = (
        len(all_evaluation_rows) - len(evaluation_rows)
    )
    if not evaluation_rows:
        raise RuntimeError("no_unseen_external_device")
    evaluation_by_device = defaultdict(list)
    for row in evaluation_rows:
        evaluation_by_device[
            gate.physical_device_key(row)
        ].append(row)
    runs = []
    scenario_runs = []
    score_rows = []
    for profile_name in profiles:
        profile = PROFILE_SPECS[profile_name]
        for seed_index in range(args.seed_count):
            seed = args.base_seed + seed_index * 7919
            fit_rows, calibration_rows = gate.calibration_split(
                training_rows,
                seed + 70000,
                args.calibration_ratio,
                min_scenario_gate_sessions=(
                    args.min_scenario_gate_sessions
                ),
            )
            (
                fit_values,
                calibration_values,
                _,
                fit_normalization_audit,
            ) = gate.prepare_values(
                fit_rows=fit_rows,
                calibration_rows=calibration_rows,
                valid_rows=calibration_rows,
                feature_mode=profile["featureMode"],
                modality_dropout=0.0,
                augmentation_mode=profile.get(
                    "augmentationMode",
                    "none",
                ),
                augmentation_probability=profile.get(
                    "augmentationProbability",
                    0.5,
                ),
                augmentation_max_log_scale=profile.get(
                    "augmentationMaximumLogScale",
                    0.35,
                ),
                seed=seed + 90000,
                imu_mode=profile["imuMode"],
                adaptive_feature_mode=profile.get(
                    "adaptiveFeatureMode",
                    "none",
                ),
                feature_family="stats",
                normalization_mode=profile.get(
                    "normalizationMode",
                    "none",
                ),
                early_evidence_mode=profile.get(
                    "earlyEvidenceMode",
                    "none",
                ),
                return_metadata=True,
            )
            fit_labels, fit_gate_labels, fit_context_labels = (
                gate.validate_rows(fit_rows)
            )
            _, calibration_gate_labels, _ = gate.validate_rows(
                calibration_rows
            )
            calibration_motion_domains = [
                gate.scenario_domain(row)
                for row in calibration_rows
            ]
            sample_weights = gate.balanced_sample_weights(
                fit_rows,
                fit_gate_labels,
                args.weighting,
            )
            device_values = {}
            evaluation_normalization_audits = {}
            for device, rows in evaluation_by_device.items():
                (
                    device_values[device],
                    evaluation_normalization_audits[device],
                ) = gate.prepare_evaluation_values(
                    fit_rows=fit_rows,
                    evaluation_rows=rows,
                    feature_mode=profile["featureMode"],
                    imu_mode=profile["imuMode"],
                    adaptive_feature_mode=profile.get(
                        "adaptiveFeatureMode",
                        "none",
                    ),
                    feature_family="stats",
                    normalization_mode=profile.get(
                        "normalizationMode",
                        "none",
                    ),
                    early_evidence_mode=profile.get(
                        "earlyEvidenceMode",
                        "none",
                    ),
                    return_metadata=True,
                )
            for model_name in models:
                predict, parameter_count, training_info = fit_candidate(
                    model_name,
                    profile_name,
                    fit_values,
                    fit_labels,
                    fit_gate_labels,
                    fit_context_labels,
                    sample_weights,
                    calibration_values,
                    calibration_gate_labels,
                    calibration_motion_domains,
                    seed,
                    args.epochs,
                    args.patience,
                )
                calibration_scores = np.asarray(
                    predict(calibration_values),
                    dtype=np.float64,
                )
                for row, label, score in zip(
                    calibration_rows,
                    calibration_gate_labels,
                    calibration_scores,
                ):
                    score_rows.append({
                        "model": model_name,
                        "profile": profile_name,
                        "seed": seed,
                        "partition": "calibration",
                        "device": "calibration",
                        "sampleId": str(row.get("sampleId") or ""),
                        "groupId": str(row.get("groupId") or ""),
                        "label": int(label),
                        "score": float(score),
                    })
                calibration_negatives = calibration_scores[
                    np.asarray(calibration_gate_labels) == 0
                ]
                if not len(calibration_negatives):
                    raise RuntimeError(
                        "calibration_missing_negative_proxy"
                    )
                consensus_repeats = int(profile.get(
                    "calibrationConsensusRepeats",
                    1,
                ))
                thresholds = (
                    calibration_consensus_scores(
                        calibration_rows,
                        calibration_gate_labels,
                        calibration_scores,
                        consensus_repeats,
                        seed + 150000,
                    )
                    if consensus_repeats > 1
                    else {
                        str(target): (
                            gate_threshold_metrics.select_threshold_at_fpr(
                                calibration_negatives,
                                target,
                            )
                        )
                        for target in gate.TARGET_FPRS
                    }
                )
                scenario_thresholds = {}
                for target in gate.TARGET_FPRS:
                    by_scenario = defaultdict(list)
                    for row, label, score in zip(
                        calibration_rows,
                        calibration_gate_labels,
                        calibration_scores,
                    ):
                        if int(label) == 0:
                            by_scenario[gate.scenario_domain(row)].append(
                                float(score)
                            )
                    scenario_thresholds[str(target)] = (
                        scenario_thresholds_at_fpr(
                            by_scenario,
                            thresholds[str(target)],
                            target,
                        )
                    )
                for device, rows in sorted(
                    evaluation_by_device.items()
                ):
                    _, valid_gate_labels, _ = gate.validate_rows(rows)
                    scores = np.asarray(
                        predict(device_values[device]),
                        dtype=np.float64,
                    )
                    for row, label, score in zip(
                        rows,
                        valid_gate_labels,
                        scores,
                    ):
                        score_rows.append({
                            "model": model_name,
                            "profile": profile_name,
                            "seed": seed,
                            "partition": "external",
                            "device": device,
                            "sampleId": str(row.get("sampleId") or ""),
                            "groupId": str(row.get("groupId") or ""),
                            "label": int(label),
                            "score": float(score),
                        })
                    for target in gate.TARGET_FPRS:
                        report = (
                            gate_threshold_metrics.fixed_threshold_report(
                                rows,
                                valid_gate_labels,
                                scores,
                                thresholds[str(target)],
                            )
                        )
                        attach_evidence_fields(
                            report,
                            calibration_rows,
                            calibration_gate_labels,
                            rows,
                            valid_gate_labels,
                            target,
                        )
                        runs.append({
                            "model": model_name,
                            "profile": profile_name,
                            "seed": seed,
                            "foldKind": "device",
                            "foldId": "device:external:" + device,
                            "targetFalsePositiveRate": target,
                            "runtimeParameterCount": parameter_count,
                            "trainingInfo": training_info,
                            "normalizationAudit": {
                                "source": fit_normalization_audit,
                                "evaluation": (
                                    evaluation_normalization_audits[device]
                                ),
                            },
                            "metrics": report,
                            "proofReady": report["proofReady"],
                        })
                        scenario_report = (
                            gate_threshold_metrics.fixed_scenario_threshold_report(
                                rows,
                                valid_gate_labels,
                                scores,
                                scenario_thresholds[str(target)],
                                thresholds[str(target)],
                                [
                                    gate.scenario_domain(row)
                                    for row in rows
                                ],
                            )
                        )
                        attach_evidence_fields(
                            scenario_report,
                            calibration_rows,
                            calibration_gate_labels,
                            rows,
                            valid_gate_labels,
                            target,
                        )
                        scenario_report["thresholdSource"] = (
                            "internal_fit_calibration_only"
                        )
                        scenario_runs.append({
                            "model": model_name,
                            "profile": profile_name,
                            "seed": seed,
                            "foldKind": "device",
                            "foldId": "device:external:" + device,
                            "targetFalsePositiveRate": target,
                            "runtimeParameterCount": parameter_count,
                            "trainingInfo": training_info,
                            "metrics": scenario_report,
                            "proofReady": scenario_report["proofReady"],
                        })
                print(
                    "completed",
                    model_name,
                    profile_name,
                    seed,
                    flush=True,
                )
    summaries = summarize_candidates(runs)
    ranked = rank_candidates(summaries)
    selected = ranked[0] if ranked else None
    scenario_summaries = summarize_candidates(scenario_runs)
    scenario_ranked = rank_candidates(scenario_summaries)
    scenario_selected = (
        scenario_ranked[0] if scenario_ranked else None
    )
    release_assessments = {
        "global": assess_release_candidate(
            selected,
            len(evaluation_by_device),
        ),
        "scenario": assess_release_candidate(
            scenario_selected,
            len(evaluation_by_device),
        ),
    }
    eligible_policies = [
        policy
        for policy, assessment in release_assessments.items()
        if assessment["eligible"]
    ]
    release_eligible = bool(eligible_policies)
    report = {
        "scope": "external_development_benchmark",
        "releaseEligible": release_eligible,
        "releaseDecision": (
            "eligible"
            if release_eligible
            else "blocked_by_proof_or_recall"
        ),
        "releasePolicy": "best_of_global_and_scenario_threshold_candidates",
        "eligiblePolicies": eligible_policies,
        "releaseAssessments": release_assessments,
        "targetRecall": gate.DEPLOYMENT_RECALL_TARGET,
        "primaryFalsePositiveRate": float(
            gate.DEPLOYMENT_PRIMARY_FPR
        ),
        "externalRowsUsedForFitting": False,
        "datasetDirs": [
            str(Path(path).resolve()) for path in args.dataset_dir
        ],
        "evaluationOnlyDatasetDirs": [
            str(Path(path).resolve())
            for path in args.evaluation_only_dataset_dir
        ],
        "rowCount": len(training_rows),
        "evaluationOnlySampleCount": len(evaluation_rows),
        "sourceEvaluationOnlySampleCount": len(all_evaluation_rows),
        "excludedSeenDevices": excluded_seen_devices,
        "excludedSeenRowCount": excluded_seen_row_count,
        "externalDeviceCount": len(evaluation_by_device),
        "deviceNames": sorted({
            str(row.get("deviceName") or "")
            for row in evaluation_rows
        }),
        "models": list(models),
        "weighting": args.weighting,
        "profiles": {
            name: PROFILE_SPECS[name] for name in profiles
        },
        "earlyEvidenceFiles": [
            str(Path(path).resolve())
            for path in args.early_evidence_file
        ],
        "trainingEarlyEvidenceCoverage": gate.early_evidence_coverage(
            training_rows
        ),
        "evaluationOnlyEarlyEvidenceCoverage": gate.early_evidence_coverage(
            evaluation_rows
        ),
        "seedCount": args.seed_count,
        "calibrationRatio": args.calibration_ratio,
        "minScenarioGateSessions": args.min_scenario_gate_sessions,
        "runs": runs,
        "scenarioRuns": scenario_runs,
        "candidateSummaries": summaries,
        "rankedCandidates": ranked,
        "selectedCandidate": selected,
        "scenarioCandidateSummaries": scenario_summaries,
        "scenarioRankedCandidates": scenario_ranked,
        "scenarioSelectedCandidate": scenario_selected,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(output)
    if args.markdown_output:
        markdown = Path(args.markdown_output)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown_temporary = markdown.with_suffix(
            markdown.suffix + ".tmp"
        )
        markdown_temporary.write_text(
            render_markdown(report),
            encoding="utf-8",
        )
        markdown_temporary.replace(markdown)
    if args.score_output:
        score_output = Path(args.score_output)
        score_output.parent.mkdir(parents=True, exist_ok=True)
        score_temporary = score_output.with_suffix(
            score_output.suffix + ".tmp"
        )
        score_temporary.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in score_rows
            ),
            encoding="utf-8",
        )
        score_temporary.replace(score_output)


if __name__ == "__main__":
    main()
