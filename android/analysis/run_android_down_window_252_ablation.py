#!/usr/bin/env python3

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


FOLD_NAMES = (
    "handheld_static",
    "placed_static",
    "walking",
)

CLASS_ORDER = (
    "thumb_light",
    "thumb_heavy",
    "index_light",
    "index_heavy",
)

IMU_SAMPLE_RATE_HZ = 210.0


def frame_indices(pre_frames, post_frames):
    return list(range(-pre_frames, post_frames + 1))


def build_dense_short_window_specs(minimum_frames=4, maximum_frames=15):
    windows = []
    for total_frames in range(minimum_frames, maximum_frames + 1):
        for pre_frames in range(total_frames):
            post_frames = total_frames - pre_frames - 1
            windows.append({
                "id": "pre" + str(pre_frames) + "_post" + str(post_frames),
                "preFrames": pre_frames,
                "postFrames": post_frames,
                "frameCount": total_frames,
            })
    return windows


def canonical_pre_frames(total_frames):
    span = total_frames - 1
    minimum = max(0, span - 15)
    maximum = min(15, span)
    available = maximum - minimum + 1
    if available <= 3:
        return list(range(minimum, maximum + 1))
    return sorted({minimum, (minimum + maximum) // 2, maximum})


def build_window_specs():
    windows = {}
    for total_frames in range(4, 32):
        for pre_frames in canonical_pre_frames(total_frames):
            post_frames = total_frames - pre_frames - 1
            window_id = "pre" + str(pre_frames) + "_post" + str(post_frames)
            windows[window_id] = {
                "id": window_id,
                "preFrames": pre_frames,
                "postFrames": post_frames,
                "frameCount": total_frames,
            }

    for pre_frames, post_frames in ((3, 11), (5, 9), (10, 4)):
        total_frames = pre_frames + post_frames + 1
        window_id = "pre" + str(pre_frames) + "_post" + str(post_frames)
        windows[window_id] = {
            "id": window_id,
            "preFrames": pre_frames,
            "postFrames": post_frames,
            "frameCount": total_frames,
        }

    output = sorted(
        windows.values(),
        key=lambda item: (item["frameCount"], item["preFrames"], item["postFrames"]),
    )
    if len(output) != 84:
        raise RuntimeError("expected_84_window_specs")
    if {item["frameCount"] for item in output} != set(range(4, 32)):
        raise RuntimeError("window_lengths_not_covered")
    return output


def build_training_jobs(fold_root, output_root, cache_root, window_specs=None):
    fold_root = Path(fold_root)
    output_root = Path(output_root)
    cache_root = Path(cache_root)
    jobs = []
    for window in window_specs or build_window_specs():
        for fold in FOLD_NAMES:
            jobs.append({
                "windowId": window["id"],
                "preFrames": window["preFrames"],
                "postFrames": window["postFrames"],
                "frameCount": window["frameCount"],
                "frameIndices": frame_indices(window["preFrames"], window["postFrames"]),
                "fold": fold,
                "manifest": fold_root / ("motion_" + fold + ".csv"),
                "outputDir": output_root / "runs" / window["id"] / fold,
                "sampleCache": cache_root / (window["id"] + ".pkl"),
            })
    expected_count = len(window_specs or build_window_specs()) * len(FOLD_NAMES)
    if len(jobs) != expected_count:
        raise RuntimeError("training_job_count_mismatch")
    return jobs


def group_jobs_by_window(jobs):
    grouped = {}
    for job in jobs:
        grouped.setdefault(job["windowId"], []).append(job)
    return list(grouped.values())


def build_training_command(
        job,
        python_executable,
        feature_preset="causal_imu11_touch9_v1",
        epochs=120,
        early_stopping_patience=20,
        batch_size=128,
        learning_rate=0.001,
        weight_decay=0.0001,
        seed=22):
    model_id = "bbsiak_down_window_" + job["windowId"] + "_" + job["fold"]
    return [
        python_executable,
        "android/analysis/train_android_finger_four_class_cnn.py",
        "--manifest", str(job["manifest"]),
        "--output-dir", str(job["outputDir"]),
        "--model-id", model_id,
        "--display-name", model_id,
        "--feature-preset", feature_preset,
        "--frame-indices=" + ",".join(str(index) for index in job["frameIndices"]),
        "--anchor-mode", "down",
        "--require-press-active",
        "--deterministic",
        "--missing-frame-policy", "error",
        "--split-strategy", "predefined",
        "--epochs", str(epochs),
        "--early-stopping-patience", str(early_stopping_patience),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--weight-decay", str(weight_decay),
        "--seed", str(seed),
        "--sample-cache", str(job["sampleCache"]),
    ]


def report_path(job):
    return job["outputDir"] / "training_report.json"


def read_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def integer_value(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def availability_for_job(job, availability_cache):
    key = (str(job["manifest"]), job["postFrames"])
    if key in availability_cache:
        return availability_cache[key]
    rows = read_rows(job["manifest"])
    train_rows = [row for row in rows if row.get("split") == "train"]
    valid_rows = [row for row in rows if row.get("split") == "valid"]
    eligible_train_rows = [
        row for row in train_rows
        if integer_value(row.get("lift_reference_frame_index"), -1) >= job["postFrames"]
    ]
    eligible_valid_rows = [
        row for row in valid_rows
        if integer_value(row.get("lift_reference_frame_index"), -1) >= job["postFrames"]
    ]
    result = {
        "trainRows": len(train_rows),
        "eligibleTrainRows": len(eligible_train_rows),
        "validRows": len(valid_rows),
        "eligibleValidRows": len(eligible_valid_rows),
        "totalRows": len(rows),
        "eligibleRows": len(eligible_train_rows) + len(eligible_valid_rows),
        "eligibleValidClassCounts": {
            class_label: sum(1 for row in eligible_valid_rows if row.get("class_label") == class_label)
            for class_label in CLASS_ORDER
        },
    }
    availability_cache[key] = result
    return result


def completed_run_rows(jobs, availability_cache):
    rows = []
    for job in jobs:
        path = report_path(job)
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        train = report.get("trainMetrics") or {}
        valid = report.get("validMetrics") or {}
        availability = availability_for_job(job, availability_cache)
        train_count = integer_value(train.get("count"), 0)
        valid_count = integer_value(valid.get("count"), 0)
        if train_count != availability["eligibleTrainRows"]:
            raise RuntimeError("press_active_train_count_mismatch:" + job["windowId"] + ":" + job["fold"])
        if valid_count != availability["eligibleValidRows"]:
            raise RuntimeError("press_active_valid_count_mismatch:" + job["windowId"] + ":" + job["fold"])
        recalls = valid.get("classRecalls") or {}
        rows.append({
            "windowId": job["windowId"],
            "preFrames": job["preFrames"],
            "postFrames": job["postFrames"],
            "frameCount": job["frameCount"],
            "fold": job["fold"],
            "bestEpoch": integer_value(report.get("bestEpoch"), 0),
            "trainRows": availability["trainRows"],
            "eligibleTrainRows": availability["eligibleTrainRows"],
            "trainCoverage": train_count / max(1, availability["trainRows"]),
            "validRows": availability["validRows"],
            "eligibleValidRows": availability["eligibleValidRows"],
            "validCoverage": valid_count / max(1, availability["validRows"]),
            "totalRows": availability["totalRows"],
            "eligibleRows": availability["eligibleRows"],
            "outputCoverage": availability["eligibleRows"] / max(1, availability["totalRows"]),
            "validAccuracy": float(valid.get("accuracy", 0.0)),
            "validBalancedAccuracy": float(valid.get("balancedAccuracy", 0.0)),
            "validMinRecall": float(valid.get("minRecall", 0.0)),
            "thumbLightRecall": float(recalls.get("thumb_light", 0.0)),
            "thumbHeavyRecall": float(recalls.get("thumb_heavy", 0.0)),
            "indexLightRecall": float(recalls.get("index_light", 0.0)),
            "indexHeavyRecall": float(recalls.get("index_heavy", 0.0)),
        })
    return rows


def dominates(left, right):
    objectives = (
        (left["meanValidAccuracy"] >= right["meanValidAccuracy"], left["meanValidAccuracy"] > right["meanValidAccuracy"]),
        (left["worstValidMinRecall"] >= right["worstValidMinRecall"], left["worstValidMinRecall"] > right["worstValidMinRecall"]),
        (left["overallOutputCoverage"] >= right["overallOutputCoverage"], left["overallOutputCoverage"] > right["overallOutputCoverage"]),
        (left["waitFrames"] <= right["waitFrames"], left["waitFrames"] < right["waitFrames"]),
    )
    return all(not_worse for not_worse, _ in objectives) and any(strictly_better for _, strictly_better in objectives)


def assign_pareto_fronts(rows):
    remaining = list(rows)
    front = 1
    while remaining:
        current = [
            candidate for candidate in remaining
            if not any(dominates(other, candidate) for other in remaining if other is not candidate)
        ]
        for candidate in current:
            candidate["paretoFront"] = front
        remaining = [candidate for candidate in remaining if candidate not in current]
        front += 1


def aggregate_window_rows(run_rows):
    grouped = defaultdict(list)
    for row in run_rows:
        grouped[row["windowId"]].append(row)
    output = []
    for window_id, rows in grouped.items():
        first = rows[0]
        completed_folds = len(rows)
        output.append({
            "windowId": window_id,
            "preFrames": first["preFrames"],
            "postFrames": first["postFrames"],
            "frameCount": first["frameCount"],
            "completedFolds": completed_folds,
            "waitFrames": first["postFrames"],
            "estimatedWaitMs": first["postFrames"] * 1000.0 / IMU_SAMPLE_RATE_HZ,
            "meanValidAccuracy": sum(row["validAccuracy"] for row in rows) / completed_folds,
            "meanValidBalancedAccuracy": sum(row["validBalancedAccuracy"] for row in rows) / completed_folds,
            "worstValidBalancedAccuracy": min(row["validBalancedAccuracy"] for row in rows),
            "meanValidMinRecall": sum(row["validMinRecall"] for row in rows) / completed_folds,
            "worstValidMinRecall": min(row["validMinRecall"] for row in rows),
            "meanValidCoverage": sum(row["validCoverage"] for row in rows) / completed_folds,
            "worstValidCoverage": min(row["validCoverage"] for row in rows),
            "meanOutputCoverage": sum(row["outputCoverage"] for row in rows) / completed_folds,
            "overallOutputCoverage": sum(row["eligibleRows"] for row in rows) / max(1, sum(row["totalRows"] for row in rows)),
            "worstOutputCoverage": min(row["outputCoverage"] for row in rows),
            "medianBestEpoch": sorted(row["bestEpoch"] for row in rows)[completed_folds // 2],
        })
    completed = [row for row in output if row["completedFolds"] == len(FOLD_NAMES)]
    assign_pareto_fronts(completed)
    completed.sort(
        key=lambda row: (
            row["paretoFront"],
            row["waitFrames"],
            -row["meanValidAccuracy"],
            -row["worstValidMinRecall"],
            -row["overallOutputCoverage"],
        ),
    )
    for position, row in enumerate(completed, start=1):
        row["presentationOrder"] = position
    partial = [row for row in output if row["completedFolds"] != len(FOLD_NAMES)]
    for row in partial:
        row["paretoFront"] = 0
        row["presentationOrder"] = 0
    return completed + sorted(partial, key=lambda row: (row["frameCount"], row["preFrames"], row["postFrames"]))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_progress(output_root, jobs, availability_cache, window_specs=None, window_plan=None):
    output_root = Path(output_root)
    run_rows = completed_run_rows(jobs, availability_cache)
    window_rows = aggregate_window_rows(run_rows)
    write_csv(
        output_root / "window_run_summary.csv",
        run_rows,
        [
            "windowId", "preFrames", "postFrames", "frameCount", "fold", "bestEpoch",
            "trainRows", "eligibleTrainRows", "trainCoverage",
            "validRows", "eligibleValidRows", "validCoverage", "totalRows", "eligibleRows", "outputCoverage", "validAccuracy",
            "validBalancedAccuracy", "validMinRecall", "thumbLightRecall", "thumbHeavyRecall",
            "indexLightRecall", "indexHeavyRecall",
        ],
    )
    write_csv(
        output_root / "window_candidate_summary.csv",
        window_rows,
        [
            "paretoFront", "presentationOrder", "windowId", "preFrames", "postFrames", "frameCount", "completedFolds",
            "waitFrames", "estimatedWaitMs", "meanValidAccuracy", "meanValidBalancedAccuracy",
            "worstValidBalancedAccuracy", "meanValidMinRecall", "worstValidMinRecall", "meanValidCoverage",
            "worstValidCoverage", "meanOutputCoverage", "overallOutputCoverage", "worstOutputCoverage", "medianBestEpoch",
        ],
    )
    payload = {
        "policy": "causal_down_window_strict_ablation",
        "plannedTrainingJobs": len(jobs),
        "completedTrainingJobs": len(run_rows),
        "plannedWindowCount": len(window_specs or build_window_specs()),
        "completedWindowCount": sum(1 for row in window_rows if row["completedFolds"] == len(FOLD_NAMES)),
        "folds": list(FOLD_NAMES),
        "anchorMode": "down",
        "requirePressActive": True,
        "windowPlan": window_plan or "representative_84",
        "selectionPolicy": "pareto_four_objective_v1",
        "paretoObjectives": [
            "meanValidAccuracy",
            "worstValidMinRecall",
            "overallOutputCoverage",
            "lowerWaitFrames",
        ],
        "paretoFrontCandidates": [
            row for row in window_rows
            if row["completedFolds"] == len(FOLD_NAMES) and row["paretoFront"] == 1
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "ablation_progress.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def run_training_job(job, args):
    command = build_training_command(
        job,
        args.python,
        feature_preset=args.feature_preset,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    if args.dry_run:
        print(" ".join(command), flush=True)
        return job
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "4")
    environment.setdefault("MKL_NUM_THREADS", "4")
    job["outputDir"].mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "start", "windowId": job["windowId"], "fold": job["fold"]}, sort_keys=True), flush=True)
    subprocess.run(command, check=True, env=environment)
    return job


def run_window_group(jobs, args):
    return [run_training_job(job, args) for job in jobs]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=Path("android/analysis/datasets/bbsiak_schema5_down_anchor_20260806/folds"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("models/bbsiak_schema5_down_window_252_20260807"),
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--feature-preset", default="causal_imu11_touch9_v1")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--window-plan",
        choices=("representative_84", "dense_4_15"),
        default="representative_84",
    )
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cache_root = args.cache_root or args.output_root / "cache"
    window_specs = (
        build_dense_short_window_specs()
        if args.window_plan == "dense_4_15"
        else build_window_specs()
    )
    if args.workers < 1:
        parser = argparse.ArgumentParser()
        parser.error("workers_must_be_positive")
    all_jobs = build_training_jobs(args.fold_root, args.output_root, cache_root, window_specs)
    availability_cache = {}
    pending_jobs = [
        job for job in all_jobs
        if args.force or not report_path(job).exists()
    ]
    if args.max_runs:
        pending_jobs = pending_jobs[:args.max_runs]
    if args.workers == 1:
        completed_groups = (run_window_group([job], args) for job in pending_jobs)
    else:
        pending_groups = group_jobs_by_window(pending_jobs)
        executor = ThreadPoolExecutor(max_workers=args.workers)
        futures = [executor.submit(run_window_group, group, args) for group in pending_groups]
        completed_groups = (future.result() for future in as_completed(futures))
    try:
        for completed_jobs in completed_groups:
            if args.dry_run:
                continue
            progress = write_progress(
                args.output_root,
                all_jobs,
                availability_cache,
                window_specs,
                args.window_plan,
            )
            for job in completed_jobs:
                print(json.dumps({
                    "event": "complete",
                    "windowId": job["windowId"],
                    "fold": job["fold"],
                    "completedTrainingJobs": progress["completedTrainingJobs"],
                }, sort_keys=True), flush=True)
    finally:
        if args.workers > 1:
            executor.shutdown(wait=True, cancel_futures=True)
    if not args.dry_run:
        progress = write_progress(
            args.output_root,
            all_jobs,
            availability_cache,
            window_specs,
            args.window_plan,
        )
        print(json.dumps(progress, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
