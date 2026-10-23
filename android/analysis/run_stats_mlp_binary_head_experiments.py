#!/usr/bin/env python3
"""Evaluate each shared MLP head under an explicit acceptance policy."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from . import run_lightweight_time_grid_models as lightweight
    from . import run_stats_mlp_contact_size_experiments as contact
    from . import run_stats_mlp_feature_experiments as feature_experiments
except ImportError:
    import run_lightweight_time_grid_models as lightweight
    import run_stats_mlp_contact_size_experiments as contact
    import run_stats_mlp_feature_experiments as feature_experiments


TARGET_COVERAGES = (0.5, 0.6, 0.7, 0.8, 0.9)
TASKS = {
    "finger": {
        "labels": ("thumb", "index"),
        "target": lambda labels: labels // 2,
    },
    "strength": {
        "labels": ("light", "heavy"),
        "target": lambda labels: labels % 2,
    },
}


def threshold_for_coverage(probabilities, target_coverage):
    """Choose the lowest threshold that reaches train coverage."""
    thresholds = np.linspace(0.05, 0.95, 19)
    chosen = None
    for threshold in thresholds:
        coverage = float(np.mean(
            probabilities.max(axis=1) >= threshold
        ))
        if coverage >= target_coverage:
            chosen = float(threshold)
        else:
            break
    if chosen is None:
        raise RuntimeError("binary_head_threshold_unavailable")
    return chosen


def accepted_head_metrics(truths, probabilities, threshold, labels):
    truths = np.asarray(truths, dtype=np.int64)
    accepted = probabilities.max(axis=1) >= threshold
    selected_truths = truths[accepted]
    predictions = probabilities.argmax(axis=1)[accepted]
    metrics = (
        contact.binary_metrics(
            selected_truths.tolist(),
            predictions.tolist(),
            labels,
        )
        if len(selected_truths)
        else None
    )
    coverage_by_class = {}
    for index, label in enumerate(labels):
        selected = truths == index
        coverage_by_class[label] = (
            float(np.mean(accepted[selected])) if np.any(selected) else None
        )
    return {
        "threshold": float(threshold),
        "validationCoverage": float(np.mean(accepted)),
        "acceptedMetrics": metrics,
        "coverageByClass": coverage_by_class,
    }


def evaluate_head(
    train_truths,
    train_probabilities,
    valid_truths,
    valid_probabilities,
    labels,
):
    results = {}
    for coverage in TARGET_COVERAGES:
        threshold = threshold_for_coverage(
            train_probabilities,
            coverage,
        )
        results[str(coverage)] = accepted_head_metrics(
            valid_truths,
            valid_probabilities,
            threshold,
            labels,
        )
    return results


def ordinary_head_acceptance(
    finger_probabilities,
    strength_probabilities,
    finger_threshold,
    strength_threshold,
):
    """Report false acceptance for each deployable head combination."""
    finger_probabilities = np.asarray(
        finger_probabilities,
        dtype=np.float64,
    )
    strength_probabilities = np.asarray(
        strength_probabilities,
        dtype=np.float64,
    )
    if (
        finger_probabilities.ndim != 2
        or finger_probabilities.shape[1] != 2
        or strength_probabilities.shape != finger_probabilities.shape
    ):
        raise RuntimeError("invalid_ordinary_head_probabilities")
    if finger_threshold is None or strength_threshold is None:
        raise RuntimeError("ordinary_head_threshold_missing")
    count = len(finger_probabilities)
    finger_pass = finger_probabilities.max(axis=1) >= finger_threshold
    strength_pass = strength_probabilities.max(axis=1) >= strength_threshold
    any_pass = finger_pass | strength_pass
    both_pass = finger_pass & strength_pass
    return {
        "ordinaryCount": int(count),
        "fingerThreshold": float(finger_threshold),
        "strengthThreshold": float(strength_threshold),
        "fingerFalseAcceptanceCount": int(np.sum(finger_pass)),
        "strengthFalseAcceptanceCount": int(np.sum(strength_pass)),
        "anyHeadFalseAcceptanceCount": int(np.sum(any_pass)),
        "bothHeadsFalseAcceptanceCount": int(np.sum(both_pass)),
        "fingerFalseAcceptanceRate": (
            float(np.mean(finger_pass)) if count else None
        ),
        "strengthFalseAcceptanceRate": (
            float(np.mean(strength_pass)) if count else None
        ),
        "anyHeadFalseAcceptanceRate": (
            float(np.mean(any_pass)) if count else None
        ),
        "bothHeadsFalseAcceptanceRate": (
            float(np.mean(both_pass)) if count else None
        ),
    }


def evaluate_ordinary_head_acceptance(
    shared_train,
    ordinary_probabilities,
):
    """Evaluate ordinary clicks at every calibrated head threshold."""
    result = {}
    for coverage in TARGET_COVERAGES:
        key = str(coverage)
        result[key] = ordinary_head_acceptance(
            ordinary_probabilities[0],
            ordinary_probabilities[1],
            shared_train[0][key]["threshold"],
            shared_train[1][key]["threshold"],
        )
    return result


def aggregate_ordinary_head_results(reports):
    """Pool ordinary clicks by fold and expose worst-fold rejection."""
    result = {}
    for coverage in TARGET_COVERAGES:
        key = str(coverage)
        items = [
            report["ordinary"][key]
            for report in reports
            if report.get("ordinary")
            and report["ordinary"].get(key)
        ]
        count = sum(item["ordinaryCount"] for item in items)
        both_count = sum(
            item["bothHeadsFalseAcceptanceCount"] for item in items
        )
        any_count = sum(
            item["anyHeadFalseAcceptanceCount"] for item in items
        )
        four_class_items = [
            item["fourClass"]
            for item in items
            if item.get("fourClass")
            and item["fourClass"].get("falseAcceptanceRate") is not None
        ]
        four_class_count = sum(
            item["ordinaryCount"] for item in four_class_items
        )
        four_class_false_count = sum(
            item["falseAcceptanceCount"] for item in four_class_items
        )
        four_class_fold_rates = [
            item["falseAcceptanceRate"] for item in four_class_items
        ]
        fold_rates = [
            item["bothHeadsFalseAcceptanceRate"]
            for item in items
            if item["bothHeadsFalseAcceptanceRate"] is not None
        ]
        result[key] = {
            "ordinaryCount": int(count),
            "fingerFalseAcceptanceRate": (
                sum(
                    item["fingerFalseAcceptanceCount"]
                    for item in items
                ) / count
                if count
                else None
            ),
            "strengthFalseAcceptanceRate": (
                sum(
                    item["strengthFalseAcceptanceCount"]
                    for item in items
                ) / count
                if count
                else None
            ),
            "anyHeadFalseAcceptanceRate": (
                any_count / count if count else None
            ),
            "bothHeadsFalseAcceptanceRate": (
                both_count / count if count else None
            ),
            "fourClassFalseAcceptanceRate": (
                four_class_false_count / four_class_count
                if four_class_count
                else None
            ),
            "worstFoldFourClassFalseAcceptanceRate": (
                float(max(four_class_fold_rates))
                if four_class_fold_rates
                else None
            ),
            "meanFoldBothHeadsFalseAcceptanceRate": (
                float(np.mean(fold_rates)) if fold_rates else None
            ),
            "worstFoldBothHeadsFalseAcceptanceRate": (
                float(max(fold_rates)) if fold_rates else None
            ),
        }
    return result


def task_readiness(
    aggregate,
    training_readiness=None,
    ordinary_acceptance=None,
    max_ordinary_false_acceptance_rate=0.05,
    negative_audit=None,
):
    """Separate reproducible experiments from deployable product behavior."""
    stable = aggregate.get("0.9")
    if stable is None:
        return {
            "status": "unverified",
            "reason": "missing_aggregate",
        }
    reasons = []
    # 中文注释：数据集未满足交错采集、姿态留出和上下文元数据门禁时，
    # 只能证明模型在当前数据上可复现，不能证明产品场景已可部署。
    training_ready = (
        bool(training_readiness)
        and bool(training_readiness.get("ready", False))
    )
    if not training_ready:
        reasons.append("training_readiness_gate_failed")
    if stable.get("worstClassRecall") is None:
        reasons.append("missing_class_recall")
    elif stable["worstClassRecall"] <= 0.0:
        reasons.append("zero_worst_class_recall")
    elif stable["worstClassRecall"] < 0.60:
        reasons.append("worst_class_recall_below_0_60")
    if stable.get("worstFoldBalancedAccuracy") is None:
        reasons.append("missing_worst_fold_balanced_accuracy")
    elif stable["worstFoldBalancedAccuracy"] < 0.70:
        reasons.append("worst_fold_balanced_accuracy_below_0_70")
    if stable.get("meanAccuracy") is None:
        reasons.append("missing_accuracy")
    elif stable["meanAccuracy"] < 0.80:
        reasons.append("mean_accuracy_below_0_80")

    # 中文注释：普通点击只保留为可选的历史误接受诊断。
    # 当前采集协议不要求负样本，因此缺失或指标偏高都不阻断训练候选。
    ordinary = (ordinary_acceptance or {}).get("0.9")

    if reasons:
        status = "not_ready"
    elif stable["worstClassRecall"] < 0.70:
        status = "candidate"
    else:
        status = "deployment_eligible"
    return {
        "status": status,
        "reasons": reasons,
        "evaluatedAtCoverage": 0.9,
        "actualMeanCoverage": stable.get("meanCoverage"),
        "worstFoldBalancedAccuracy": stable.get(
            "worstFoldBalancedAccuracy"
        ),
        "worstClassRecall": stable.get("worstClassRecall"),
        "trainingReady": training_ready,
        "maxOrdinaryFalseAcceptanceRate": (
            max_ordinary_false_acceptance_rate
        ),
        "ordinaryFalseAcceptance": ordinary,
        "ordinaryNegativeAudit": negative_audit,
    }


def aggregate_head_results(reports, task, source):
    labels = TASKS[task]["labels"]
    result = {}
    for coverage in TARGET_COVERAGES:
        key = str(coverage)
        items = [
            report[source][task][key]
            for report in reports
        ]
        accuracies = [
            item["acceptedMetrics"]["accuracy"]
            for item in items
            if item["acceptedMetrics"] is not None
        ]
        balanced = [
            item["acceptedMetrics"]["balancedAccuracy"]
            for item in items
            if item["acceptedMetrics"] is not None
        ]
        class_recalls = [
            item["acceptedMetrics"]["recallByClass"][label]
            for item in items
            if item["acceptedMetrics"] is not None
            for label in labels
        ]
        result[key] = {
            "meanCoverage": float(np.mean([
                item["validationCoverage"] for item in items
            ])),
            "meanAccuracy": (
                float(np.mean(accuracies)) if accuracies else None
            ),
            "worstFoldAccuracy": (
                float(np.min(accuracies)) if accuracies else None
            ),
            "meanBalancedAccuracy": (
                float(np.mean(balanced)) if balanced else None
            ),
            "worstFoldBalancedAccuracy": (
                float(np.min(balanced)) if balanced else None
            ),
            "worstClassRecall": (
                float(np.min([
                    value for value in class_recalls
                    if value is not None
                ]))
                if any(value is not None for value in class_recalls)
                else None
            ),
        }
    return result


def class_balanced_indices(samples, seed, count_per_class):
    """Sample equal touches per class without changing model training."""
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for index, sample in enumerate(samples):
        by_class[int(sample["label"])].append(index)
    selected = []
    for label in sorted(by_class):
        candidates = np.asarray(by_class[label], dtype=np.int64)
        take = min(count_per_class, len(candidates))
        selected.extend(
            rng.choice(candidates, size=take, replace=False).tolist()
        )
    return np.asarray(sorted(selected), dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--tail-count", type=int, default=8)
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=0,
        help=(
            "Repeat held-out-device scoring with class-balanced device "
            "subsamples. Zero disables the extra experiment."
        ),
    )
    args = parser.parse_args()
    if (
        args.epochs <= 0
        or args.tail_count <= 0
        or args.bootstrap_repeats < 0
    ):
        raise RuntimeError("invalid_training_parameter")

    folds = contact.load_samples(args.dataset_dir)
    for train_samples, valid_samples in folds.values():
        for sample in train_samples + valid_samples:
            sample["datasetId"] = str(Path(args.dataset_dir).resolve())
    cleaning_report = json.loads(
        (
            Path(args.dataset_dir)
            / "training_cleaning_report.json"
        ).read_text(encoding="utf-8")
    )
    ordinary_samples = contact.load_ordinary_samples(args.dataset_dir)
    fold_kind = str(
        cleaning_report.get("validationFoldKind") or "unknown"
    )
    negative_audit = contact.ordinary_negative_audit(
        ordinary_samples,
        folds,
        fold_kind,
    )
    modes = {
        "baseline": None,
        "causal_device_tail": args.tail_count,
    }
    training_readiness = cleaning_report.get("trainingReadiness")
    output = {
        "scope": contact.fold_evaluation_scope(folds),
        "trainingReadiness": training_readiness,
        "targetCoverages": list(TARGET_COVERAGES),
        "tailCalibrationCount": args.tail_count,
        "ordinaryNegativeAudit": {
            **negative_audit,
            "usage": "false_acceptance_only",
            "foldMatching": fold_kind,
            "calibrationPolicy": (
                "ordinary negatives share the held-out device warmup "
                "prefix with positive validation samples"
            ),
            "matchedByMode": {},
        },
        "comparison": (
            "finger and strength heads are calibrated independently; the "
            "same shared model is used for both heads"
        ),
        "experiments": {},
    }

    for mode, tail_count in modes.items():
        reports = []
        bootstrap_reports = []
        for fold, (train_samples, valid_samples) in sorted(folds.items()):
            print(json.dumps({
                "mode": mode,
                "fold": fold,
            }), flush=True)
            if tail_count is None:
                train_values, valid_values = (
                    contact.prepare_fold_leave_one_device_out(
                        train_samples,
                        valid_samples,
                        "baseline",
                    )
                )
                scored_samples = valid_samples
                scored_values = valid_values
            else:
                (
                    train_values,
                    valid_values,
                    score_indices,
                ) = contact.prepare_fold_leave_one_device_out_calibrated_tail(
                    train_samples,
                    valid_samples,
                    "causal_device_median_{}_tail".format(tail_count),
                )
                scored_samples = [
                    valid_samples[index] for index in score_indices
                ]
                scored_values = valid_values[score_indices]

            labels = np.asarray(
                [sample["label"] for sample in train_samples],
                dtype=np.int64,
            )
            valid_labels = np.asarray(
                [sample["label"] for sample in scored_samples],
                dtype=np.int64,
            )
            shared_model = contact.train_fixed_epochs(
                train_values,
                train_samples,
                args.epochs,
                20261022 + fold,
            )
            shared_train = contact.predict_head_probabilities(
                shared_model,
                train_values,
            )
            shared_train_four_class = contact.predict_probabilities(
                shared_model,
                train_values,
            )
            shared_valid = contact.predict_head_probabilities(
                shared_model,
                scored_values,
            )
            shared = {}
            for task_index, task in enumerate(("finger", "strength")):
                task_labels = TASKS[task]["labels"]
                truth = TASKS[task]["target"](labels)
                valid_truth = TASKS[task]["target"](valid_labels)
                shared[task] = evaluate_head(
                    truth,
                    shared_train[task_index],
                    valid_truth,
                    shared_valid[task_index],
                    task_labels,
                )
            ordinary = {}
            fold_ordinary_samples = contact.ordinary_samples_for_fold(
                ordinary_samples,
                valid_samples,
                fold_kind,
            )
            ordinary_domain_sample_count = len(fold_ordinary_samples)
            if fold_ordinary_samples:
                prepare_mode = (
                    "baseline"
                    if tail_count is None
                    else "causal_device_median_{}_tail".format(
                        tail_count
                    )
                )
                (
                    fold_ordinary_samples,
                    ordinary_values,
                ) = contact.prepare_ordinary_values(
                    train_samples,
                    valid_samples,
                    fold_ordinary_samples,
                    prepare_mode,
                    contact.prepare_fold_leave_one_device_out,
                    "raw",
                    contact.fold_evaluation_scope(folds) == "cross_device",
                )
            if fold_ordinary_samples:
                ordinary_probabilities = (
                    contact.predict_head_probabilities(
                        shared_model,
                        ordinary_values,
                    )
                )
                ordinary_four_class_probabilities = (
                    contact.predict_probabilities(
                        shared_model,
                        ordinary_values,
                    )
                )
                for coverage in TARGET_COVERAGES:
                    key = str(coverage)
                    four_class_threshold = threshold_for_coverage(
                        shared_train_four_class,
                        coverage,
                    )
                    ordinary[key] = ordinary_head_acceptance(
                        ordinary_probabilities[0],
                        ordinary_probabilities[1],
                        shared["finger"][key]["threshold"],
                        shared["strength"][key]["threshold"],
                    )
                    ordinary[key]["fourClass"] = (
                        contact.false_acceptance_metrics(
                            ordinary_four_class_probabilities,
                            four_class_threshold,
                        )
                    )
            reports.append({
                "fold": fold,
                "validDevices": sorted({
                    sample.get("deviceName", "")
                    for sample in scored_samples
                }),
                "scoredCount": len(scored_samples),
                "ordinaryDomainSampleCount": (
                    ordinary_domain_sample_count
                ),
                "ordinarySampleCount": len(fold_ordinary_samples),
                "shared": shared,
                "ordinary": ordinary,
            })
            if args.bootstrap_repeats > 0:
                valid_labels = np.asarray([
                    sample["label"] for sample in scored_samples
                ])
                class_counts = np.bincount(
                    valid_labels,
                    minlength=len(lightweight.CLASS_LABELS),
                )
                count_per_class = int(np.min(class_counts))
                for repeat in range(args.bootstrap_repeats):
                    selected = class_balanced_indices(
                        scored_samples,
                        20261022 + fold * 100 + repeat,
                        count_per_class,
                    )
                    sampled = {}
                    for task_index, task in enumerate(
                        ("finger", "strength")
                    ):
                        task_labels = TASKS[task]["labels"]
                        sampled_truth = TASKS[task]["target"](
                            valid_labels[selected]
                        )
                        sampled[task] = evaluate_head(
                            TASKS[task]["target"](labels),
                            shared_train[task_index],
                            sampled_truth,
                            shared_valid[task_index][selected],
                            task_labels,
                        )
                    bootstrap_reports.append({
                        "fold": fold,
                        "repeat": repeat,
                        "countPerClass": count_per_class,
                        "shared": sampled,
                    })
        aggregate_heads = {
            task: aggregate_head_results(
                reports,
                task,
                "shared",
            )
            for task in ("finger", "strength")
        }
        ordinary_aggregate = aggregate_ordinary_head_results(
            reports
        )
        mode_negative_audit = contact.ordinary_negative_mode_audit(
            negative_audit,
            reports,
        )
        output["ordinaryNegativeAudit"]["matchedByMode"][mode] = (
            mode_negative_audit
        )
        output["experiments"][mode] = {
            "folds": reports,
            "aggregate": {
                task: {
                    "shared": aggregate_heads[task],
                }
                for task in ("finger", "strength")
            },
            "ordinaryFalseAcceptance": ordinary_aggregate,
            "taskReadiness": {
                task: task_readiness(
                    aggregate_heads[task],
                    training_readiness,
                    ordinary_aggregate,
                    negative_audit=mode_negative_audit,
                )
                for task in ("finger", "strength")
            },
        }
        if bootstrap_reports:
            output["experiments"][mode]["classBalancedBootstrap"] = {
                "repeatCount": args.bootstrap_repeats,
                "aggregate": {
                    task: aggregate_head_results(
                        bootstrap_reports,
                        task,
                        "shared",
                    )
                    for task in ("finger", "strength")
                },
                "taskReadiness": {
                    task: task_readiness(
                        aggregate_head_results(
                            bootstrap_reports,
                            task,
                            "shared",
                        ),
                        training_readiness,
                        ordinary_aggregate,
                        negative_audit=mode_negative_audit,
                    )
                    for task in ("finger", "strength")
                },
            }

    path = Path(args.output_dir) / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
