#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

try:
    from . import train_time_grid_strict_cnn as trainer
except ImportError:
    import train_time_grid_strict_cnn as trainer


def experiment_specs():
    return [
        {"contract": contract, "fold": fold}
        for contract in ("imu11", "touch23")
        for fold in range(3)
    ]


def promotion_decision(imu, touch):
    checks = {
        "meanBalancedAccuracyGain": touch["meanBalancedAccuracy"] - imu["meanBalancedAccuracy"] >= 0.01,
        "worstFoldRegression": touch["worstFoldBalancedAccuracy"] >= imu["worstFoldBalancedAccuracy"] - 0.005,
        "worstClassRecallRegression": touch["worstClassRecall"] >= imu["worstClassRecall"] - 0.02,
        "meanGeneralizationGap": touch["meanGeneralizationGap"] <= 0.15,
        "worstGeneralizationGap": touch["worstGeneralizationGap"] <= 0.25,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "decision": "touch23_candidate" if all(checks.values()) else "keep_imu11_baseline",
    }


def aggregate_contract(reports):
    balanced = [report["validMetrics"]["balancedAccuracy"] for report in reports]
    macro_f1 = [report["validMetrics"]["macroF1"] for report in reports]
    gaps = [report["generalizationGap"] for report in reports]
    recalls = [
        value
        for report in reports
        for value in report["validMetrics"]["recallByClass"].values()
    ]
    mean_recall_by_class = {
        label: mean(report["validMetrics"]["recallByClass"][label] for report in reports)
        for label in trainer.CLASS_LABELS
    }
    worst_recall_by_class = {
        label: min(report["validMetrics"]["recallByClass"][label] for report in reports)
        for label in trainer.CLASS_LABELS
    }
    return {
        "contract": reports[0]["contract"],
        "foldCount": len(reports),
        "parameterCount": reports[0]["parameterCount"],
        "inputShape": [9, reports[0]["featureCount"]],
        "meanBalancedAccuracy": mean(balanced),
        "worstFoldBalancedAccuracy": min(balanced),
        "meanMacroF1": mean(macro_f1),
        "worstClassRecall": min(recalls),
        "meanRecallByClass": mean_recall_by_class,
        "worstFoldRecallByClass": worst_recall_by_class,
        "meanGeneralizationGap": mean(gaps),
        "worstGeneralizationGap": max(gaps),
        "foldBalancedAccuracies": balanced,
        "bestEpochs": [report["bestEpoch"] for report in reports],
    }


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_all(dataset_dir, output_dir):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    config = trainer.default_config()
    write_json(output_root / "experiment_contract.json", {
        "evaluationScope": "single_device_cross_session_baseline",
        "datasetDir": str(Path(dataset_dir).resolve()),
        "tasks": experiment_specs(),
        "config": config,
        "deploymentAllowed": False,
    })
    reports = []
    for spec in experiment_specs():
        task_dir = output_root / spec["contract"] / ("fold_" + str(spec["fold"]))
        report = trainer.train_fold(dataset_dir, task_dir, spec["contract"], spec["fold"], config)
        reports.append(report)

    aggregates = {
        contract: aggregate_contract([report for report in reports if report["contract"] == contract])
        for contract in ("imu11", "touch23")
    }
    promotion = promotion_decision(aggregates["imu11"], aggregates["touch23"])
    probe_dir = output_root / "determinism_probe" / "imu11_fold_0"
    probe = trainer.train_fold(dataset_dir, probe_dir, "imu11", 0, config)
    reference = next(report for report in reports if report["contract"] == "imu11" and report["fold"] == 0)
    determinism = {
        "passed": (
            probe["weightsSha256"] == reference["weightsSha256"]
            and probe["bestEpoch"] == reference["bestEpoch"]
            and probe["validMetrics"] == reference["validMetrics"]
        ),
        "referenceWeightsSha256": reference["weightsSha256"],
        "probeWeightsSha256": probe["weightsSha256"],
        "referenceBestEpoch": reference["bestEpoch"],
        "probeBestEpoch": probe["bestEpoch"],
    }
    write_json(output_root / "determinism_report.json", determinism)
    aggregate_report = {
        "evaluationScope": "single_device_cross_session_baseline",
        "deploymentAllowed": False,
        "taskCount": len(reports),
        "aggregates": aggregates,
        "promotion": promotion,
        "determinism": determinism,
    }
    write_json(output_root / "aggregate_report.json", aggregate_report)
    with (output_root / "aggregate_results.csv").open("w", encoding="utf-8", newline="") as output:
        fields = (
            "contract",
            "foldCount",
            "meanBalancedAccuracy",
            "worstFoldBalancedAccuracy",
            "meanMacroF1",
            "worstClassRecall",
            "meanGeneralizationGap",
            "worstGeneralizationGap",
        )
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for contract in ("imu11", "touch23"):
            writer.writerow({field: aggregates[contract][field] for field in fields})
    if not determinism["passed"]:
        raise RuntimeError("determinism_probe_failed")
    return aggregate_report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_all(args.dataset_dir, args.output_dir)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
