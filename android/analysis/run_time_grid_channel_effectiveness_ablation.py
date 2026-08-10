#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    from . import train_time_grid_all_data_ablation as trainer
    from . import train_time_grid_strict_cnn as strict_trainer
except ImportError:
    import train_time_grid_all_data_ablation as trainer
    import train_time_grid_strict_cnn as strict_trainer


def experiment_specs():
    return [{"contract": contract} for contract in trainer.CHANNEL_CONTRACTS]


def bounded_worker_count(requested_workers, task_count):
    if int(requested_workers) <= 0 or int(task_count) <= 0:
        raise RuntimeError("worker_and_task_counts_must_be_positive")
    return min(int(requested_workers), int(task_count))


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_spec(dataset_dir, output_dir, contract, config):
    return trainer.train_all_data_contract(dataset_dir, output_dir, contract, config)


def ranked_rows(reports):
    return sorted(
        reports,
        key=lambda report: (
            -float(report["trainMetrics"]["balancedAccuracy"]),
            -float(report["trainMetrics"]["macroF1"]),
            float(report["finalTrainLoss"]),
            str(report["contract"]),
        ),
    )


def run_all(dataset_dir, output_dir, max_workers=7):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    specs = experiment_specs()
    worker_count = bounded_worker_count(max_workers, len(specs))
    config = trainer.default_all_data_config()
    snapshot_root = output_root / "configuration_snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    write_json(snapshot_root / "strict_baseline_default_config.json", strict_trainer.default_config())
    write_json(snapshot_root / "all_data_ablation_config.json", config)
    write_json(snapshot_root / "source_scripts.json", {
        "strictTrainer": str(Path(strict_trainer.__file__).resolve()),
        "strictTrainerSha256": file_sha256(strict_trainer.__file__),
        "allDataTrainer": str(Path(trainer.__file__).resolve()),
        "allDataTrainerSha256": file_sha256(trainer.__file__),
    })
    write_json(output_root / "experiment_contract.json", {
        "evaluationScope": "all_data_fit_only",
        "deploymentAllowed": False,
        "datasetDir": str(Path(dataset_dir).resolve()),
        "taskCount": len(specs),
        "maxWorkers": worker_count,
        "config": config,
        "tasks": specs,
    })
    reports_by_contract = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                train_spec,
                str(Path(dataset_dir).resolve()),
                str(output_root / spec["contract"]),
                spec["contract"],
                config,
            ): spec["contract"]
            for spec in specs
        }
        for future in as_completed(futures):
            contract = futures[future]
            reports_by_contract[contract] = future.result()
    reports = [reports_by_contract[spec["contract"]] for spec in specs]
    ranked = ranked_rows(reports)
    aggregate_rows = []
    for rank, report in enumerate(ranked, start=1):
        aggregate_rows.append({
            "rank": rank,
            "contract": report["contract"],
            "featureCount": report["featureCount"],
            "parameterCount": report["parameterCount"],
            "trainCount": report["trainCount"],
            "epochsCompleted": report["epochsCompleted"],
            "finalTrainLoss": report["finalTrainLoss"],
            "trainBalancedAccuracy": report["trainMetrics"]["balancedAccuracy"],
            "trainMacroF1": report["trainMetrics"]["macroF1"],
            "worstClassRecall": min(report["trainMetrics"]["recallByClass"].values()),
        })
    aggregate_report = {
        "evaluationScope": "all_data_fit_only",
        "deploymentAllowed": False,
        "warning": "Training metrics rank in-sample fit only and do not measure cross-session or cross-device generalization.",
        "taskCount": len(reports),
        "maxWorkers": worker_count,
        "ranking": aggregate_rows,
        "reports": reports,
    }
    write_json(output_root / "aggregate_report.json", aggregate_report)
    with (output_root / "aggregate_results.csv").open("w", encoding="utf-8", newline="") as output:
        fields = tuple(aggregate_rows[0])
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    return aggregate_report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-workers", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_all(args.dataset_dir, args.output_dir, args.max_workers)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
