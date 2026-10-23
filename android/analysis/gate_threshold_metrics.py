"""Fixed false-positive-rate evaluation for the light/heavy product gate."""

import math

import numpy as np


def _validate_probabilities(probabilities):
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("invalid_heavy_probabilities")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("invalid_heavy_probabilities")
    return values


def _group_value(row, field):
    """Read one report group, with a compatibility fallback for old rows."""
    raw_value = row.get(field)
    if field == "postureDomain" and raw_value in (None, ""):
        # 中文注释：旧 fixture 和外部调用方可能尚未生成统一姿态域。
        # 此处只做报告分组兼容，不把原始字段写回训练特征。
        raw_value = row.get("bodyPosture")
    return (
        str(raw_value).strip()
        if raw_value not in (None, "")
        else "unknown"
    )


def wilson_interval(successes, total, z=1.96):
    """Return a two-sided Wilson score interval for a binomial rate."""
    count = int(total)
    if count <= 0:
        return None
    successes = int(successes)
    if successes < 0 or successes > count:
        raise ValueError("invalid_wilson_counts")
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (
        proportion + z * z / (2.0 * count)
    ) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return {
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
        "confidence": 0.95,
    }


def select_threshold_at_fpr(negative_probabilities, target_fpr):
    """Choose a deterministic threshold with at most the requested negative rate."""
    negatives = _validate_probabilities(negative_probabilities)
    if not 0.0 <= float(target_fpr) <= 1.0:
        raise ValueError("invalid_target_fpr")
    allowed = int(math.floor(len(negatives) * float(target_fpr)))
    descending = np.sort(negatives)[::-1]
    if allowed >= len(descending):
        return -math.inf
    # 中文注释：判定使用严格大于阈值，因此阈值取第 allowed+1 个分数时，
    # 即使存在同分触摸，轻触误触发数量也不会超过目标配额。
    return float(descending[allowed])


def roc_auc(truths, heavy_probabilities):
    """Return threshold-free ranking quality for binary gate scores."""
    labels = np.asarray(truths, dtype=np.int64)
    scores = _validate_probabilities(heavy_probabilities)
    if labels.shape != scores.shape:
        raise ValueError("gate_metric_shape_mismatch")
    if np.any((labels < 0) | (labels > 1)):
        raise ValueError("invalid_gate_truth")
    positive_count = int(np.count_nonzero(labels == 1))
    negative_count = int(np.count_nonzero(labels == 0))
    if not positive_count or not negative_count:
        return None
    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(ordered_scores):
        end = start + 1
        while (
            end < len(ordered_scores)
            and ordered_scores[end] == ordered_scores[start]
        ):
            end += 1
        # 中文注释：同分样本取平均秩，避免阈值边界改变排序结论。
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def binary_metrics(truths, heavy_probabilities, threshold):
    """Evaluate heavy recall and light false positives at one threshold."""
    labels = np.asarray(truths, dtype=np.int64)
    scores = _validate_probabilities(heavy_probabilities)
    if labels.shape != scores.shape:
        raise ValueError("gate_metric_shape_mismatch")
    if np.any((labels < 0) | (labels > 1)):
        raise ValueError("invalid_gate_truth")
    if not math.isfinite(float(threshold)) and threshold != -math.inf:
        raise ValueError("invalid_gate_threshold")
    predictions = (scores > float(threshold)).astype(np.int64)
    positives = labels == 1
    negatives = labels == 0
    positive_count = int(positives.sum())
    negative_count = int(negatives.sum())
    true_positive = int(np.count_nonzero(predictions[positives] == 1))
    false_positive = int(np.count_nonzero(predictions[negatives] == 1))
    true_negative = negative_count - false_positive
    false_negative = positive_count - true_positive
    recall = true_positive / positive_count if positive_count else None
    false_positive_rate = (
        false_positive / negative_count if negative_count else None
    )
    return {
        "threshold": float(threshold),
        "positiveCount": positive_count,
        "negativeCount": negative_count,
        "truePositiveCount": true_positive,
        "falsePositiveCount": false_positive,
        "trueNegativeCount": true_negative,
        "falseNegativeCount": false_negative,
        "heavyRecall": recall,
        "lightFalsePositiveRate": false_positive_rate,
        "rocAuc": roc_auc(labels, scores),
        "heavyRecallWilson95": wilson_interval(
            true_positive,
            positive_count,
        ),
        "lightFalsePositiveRateWilson95": wilson_interval(
            false_positive,
            negative_count,
        ),
        "precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        ),
    }


def binary_metrics_from_predictions(truths, predictions):
    """Evaluate explicit binary decisions without inventing a threshold."""
    labels = np.asarray(truths, dtype=np.int64)
    decisions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != decisions.shape:
        raise ValueError("gate_prediction_shape_mismatch")
    if np.any((labels < 0) | (labels > 1)):
        raise ValueError("invalid_gate_truth")
    if np.any((decisions < 0) | (decisions > 1)):
        raise ValueError("invalid_gate_prediction")
    positives = labels == 1
    negatives = labels == 0
    positive_count = int(positives.sum())
    negative_count = int(negatives.sum())
    true_positive = int(np.count_nonzero(decisions[positives] == 1))
    false_positive = int(np.count_nonzero(decisions[negatives] == 1))
    true_negative = negative_count - false_positive
    false_negative = positive_count - true_positive
    recall = true_positive / positive_count if positive_count else None
    false_positive_rate = (
        false_positive / negative_count if negative_count else None
    )
    return {
        "threshold": None,
        "positiveCount": positive_count,
        "negativeCount": negative_count,
        "truePositiveCount": true_positive,
        "falsePositiveCount": false_positive,
        "trueNegativeCount": true_negative,
        "falseNegativeCount": false_negative,
        "heavyRecall": recall,
        "lightFalsePositiveRate": false_positive_rate,
        "heavyRecallWilson95": wilson_interval(
            true_positive,
            positive_count,
        ),
        "lightFalsePositiveRateWilson95": wilson_interval(
            false_positive,
            negative_count,
        ),
        "precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        ),
    }


def fixed_fpr_report(
        rows,
        truths,
        heavy_probabilities,
        target_fpr,
        group_fields=("deviceName", "bodyPosture", "motionState", "groupId"),
):
    """Report recall at one global threshold, never per-group thresholds."""
    labels = np.asarray(truths, dtype=np.int64)
    scores = _validate_probabilities(heavy_probabilities)
    if len(rows) != len(labels) or len(rows) != len(scores):
        raise ValueError("fixed_fpr_shape_mismatch")
    negative_scores = scores[labels == 0]
    if negative_scores.size == 0:
        raise ValueError("missing_negative_proxy")
    threshold = select_threshold_at_fpr(negative_scores, target_fpr)
    report = {
        "targetFalsePositiveRate": float(target_fpr),
        "negativeProxy": "light_touch",
        "ordinaryClickFprAvailable": False,
        "thresholdPolicy": "single_global_threshold",
        "overall": binary_metrics(labels, scores, threshold),
        "groups": {},
    }
    for field in group_fields:
        groups = {}
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError("invalid_gate_metric_row")
            value = _group_value(row, field)
            groups.setdefault(value, []).append(index)
        group_report = {}
        for value, indices in sorted(groups.items()):
            selected = np.asarray(indices, dtype=np.int64)
            group_report[value] = binary_metrics(
                labels[selected],
                scores[selected],
                threshold,
            )
        report["groups"][field] = group_report
    return report


def fixed_threshold_report(
        rows,
        truths,
        heavy_probabilities,
        threshold,
        group_fields=("deviceName", "bodyPosture", "motionState", "groupId"),
):
    """Apply an externally calibrated threshold without refitting it."""
    labels = np.asarray(truths, dtype=np.int64)
    scores = _validate_probabilities(heavy_probabilities)
    if len(rows) != len(labels) or len(rows) != len(scores):
        raise ValueError("fixed_threshold_shape_mismatch")
    report = {
        "negativeProxy": "light_touch",
        "ordinaryClickFprAvailable": False,
        "thresholdPolicy": "external_threshold",
        "overall": binary_metrics(labels, scores, threshold),
        "groups": {},
    }
    for field in group_fields:
        groups = {}
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError("invalid_gate_metric_row")
            value = _group_value(row, field)
            groups.setdefault(value, []).append(index)
        group_report = {}
        for value, indices in sorted(groups.items()):
            selected = np.asarray(indices, dtype=np.int64)
            group_report[value] = binary_metrics(
                labels[selected],
                scores[selected],
                threshold,
            )
        report["groups"][field] = group_report
    return report


def fixed_scenario_threshold_report(
        rows,
        truths,
        heavy_probabilities,
        thresholds,
        fallback_threshold,
        scenario_values,
        group_fields=("deviceName", "bodyPosture", "motionState", "groupId"),
):
    """Apply calibration thresholds by scenario without target refitting."""
    labels = np.asarray(truths, dtype=np.int64)
    scores = _validate_probabilities(heavy_probabilities)
    scenarios = np.asarray(scenario_values, dtype=object)
    if (
        len(rows) != len(labels)
        or len(rows) != len(scores)
        or len(rows) != len(scenarios)
    ):
        raise ValueError("fixed_scenario_shape_mismatch")
    selected = np.asarray([
        float(thresholds.get(str(scenario), fallback_threshold))
        for scenario in scenarios
    ], dtype=np.float64)
    predictions = (scores > selected).astype(np.int64)
    report = {
        "negativeProxy": "light_touch",
        "ordinaryClickFprAvailable": False,
        "thresholdPolicy": "calibration_scenario_threshold",
        "fallbackThreshold": float(fallback_threshold),
        "thresholds": {
            str(key): float(value)
            for key, value in sorted(thresholds.items())
        },
        "overall": binary_metrics_from_predictions(
            labels,
            predictions,
        ),
        "groups": {},
    }
    for field in group_fields:
        groups = {}
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TypeError("invalid_gate_metric_row")
            value = _group_value(row, field)
            groups.setdefault(value, []).append(index)
        group_report = {}
        for value, indices in sorted(groups.items()):
            index_array = np.asarray(indices, dtype=np.int64)
            group_report[value] = binary_metrics_from_predictions(
                labels[index_array],
                predictions[index_array],
            )
        report["groups"][field] = group_report
    return report


def fixed_negative_audit_report(
        rows,
        heavy_probabilities,
        threshold,
        group_fields=("deviceName", "bodyPosture", "motionState", "groupId"),
):
    """Audit ordinary taps at an externally frozen heavy-gate threshold."""
    scores = _validate_probabilities(heavy_probabilities)
    if len(rows) != len(scores):
        raise ValueError("negative_audit_shape_mismatch")
    truths = np.zeros(len(rows), dtype=np.int64)
    report = fixed_threshold_report(
        rows,
        truths,
        scores,
        threshold,
        group_fields,
    )
    report["negativeProxy"] = "ordinary_touch"
    report["ordinaryClickFprAvailable"] = True
    report["ordinaryClickAcceptedCount"] = (
        report["overall"]["falsePositiveCount"]
    )
    report["ordinaryClickAcceptanceRate"] = (
        report["overall"]["lightFalsePositiveRate"]
    )
    report["overall"]["ordinaryClickAcceptanceRate"] = (
        report["overall"]["lightFalsePositiveRate"]
    )
    for field_groups in report["groups"].values():
        for metrics in field_groups.values():
            metrics["ordinaryClickAcceptanceRate"] = (
                metrics["lightFalsePositiveRate"]
            )
    return report
