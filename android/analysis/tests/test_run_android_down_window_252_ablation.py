import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_DIR))
import run_android_down_window_252_ablation as runner


class DownWindow252AblationTests(unittest.TestCase):
    def test_dense_short_window_specs_enumerate_every_4_to_15_frame_combination(self):
        windows = runner.build_dense_short_window_specs()

        self.assertEqual(114, len(windows))
        self.assertEqual(114, len({window["id"] for window in windows}))
        self.assertEqual(set(range(4, 16)), {window["frameCount"] for window in windows})
        self.assertEqual(
            {
                "preFrames": 5,
                "postFrames": 9,
                "frameCount": 15,
            },
            {
                key: value
                for key, value in next(window for window in windows if window["id"] == "pre5_post9").items()
                if key != "id"
            },
        )
        self.assertEqual(
            15,
            len([window for window in windows if window["frameCount"] == 15]),
        )

    def test_dense_short_window_jobs_expand_to_342_strict_folds(self):
        jobs = runner.build_training_jobs(
            Path("folds"),
            Path("models"),
            Path("cache"),
            window_specs=runner.build_dense_short_window_specs(),
        )

        self.assertEqual(342, len(jobs))
        self.assertEqual(3, len([job for job in jobs if job["windowId"] == "pre1_post2"]))

    def test_window_groups_keep_all_three_folds_together_for_parallel_execution(self):
        jobs = runner.build_training_jobs(
            Path("folds"),
            Path("models"),
            Path("cache"),
            window_specs=runner.build_dense_short_window_specs(),
        )

        groups = runner.group_jobs_by_window(jobs)

        self.assertEqual(114, len(groups))
        self.assertTrue(all(len(group) == 3 for group in groups))
        self.assertTrue(
            all(
                {job["fold"] for job in group} == set(runner.FOLD_NAMES)
                for group in groups
            )
        )

    def test_window_specs_cover_all_lengths_with_84_unique_windows(self):
        windows = runner.build_window_specs()

        self.assertEqual(84, len(windows))
        self.assertEqual(84, len({window["id"] for window in windows}))
        self.assertEqual(set(range(4, 32)), {window["frameCount"] for window in windows})
        self.assertIn({"id": "pre5_post9", "preFrames": 5, "postFrames": 9, "frameCount": 15}, windows)
        self.assertTrue(all(0 <= window["preFrames"] <= 15 for window in windows))
        self.assertTrue(all(0 <= window["postFrames"] <= 15 for window in windows))

    def test_training_jobs_expand_each_window_into_three_strict_folds(self):
        jobs = runner.build_training_jobs(Path("folds"), Path("models"), Path("cache"))

        self.assertEqual(252, len(jobs))
        self.assertEqual(set(runner.FOLD_NAMES), {job["fold"] for job in jobs})
        self.assertEqual(3, len([job for job in jobs if job["windowId"] == "pre5_post9"]))
        self.assertTrue(all(job["manifest"].name.startswith("motion_") for job in jobs))

    def test_training_command_enforces_causal_and_deterministic_contract(self):
        job = runner.build_training_jobs(Path("folds"), Path("models"), Path("cache"))[0]

        command = runner.build_training_command(job, "python")

        self.assertIn("--require-press-active", command)
        self.assertIn("--deterministic", command)
        self.assertEqual("causal_imu11_touch9_v1", command[command.index("--feature-preset") + 1])
        self.assertIn("--anchor-mode", command)
        self.assertEqual("down", command[command.index("--anchor-mode") + 1])
        self.assertIn("--split-strategy", command)
        self.assertEqual("predefined", command[command.index("--split-strategy") + 1])

    def test_completed_run_rows_rejects_train_count_that_includes_post_lift_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "fold.csv"
            output_dir = root / "run"
            output_dir.mkdir()
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=[
                    "split", "lift_reference_frame_index", "class_label",
                ])
                writer.writeheader()
                writer.writerows([
                    {"split": "train", "lift_reference_frame_index": "1", "class_label": "thumb_light"},
                    {"split": "train", "lift_reference_frame_index": "2", "class_label": "thumb_light"},
                    {"split": "valid", "lift_reference_frame_index": "2", "class_label": "thumb_light"},
                ])
            (output_dir / "training_report.json").write_text(json.dumps({
                "trainMetrics": {"count": 2},
                "validMetrics": {"count": 1, "classRecalls": {}},
            }), encoding="utf-8")
            job = {
                "windowId": "pre1_post2",
                "preFrames": 1,
                "postFrames": 2,
                "frameCount": 4,
                "fold": "walking",
                "manifest": manifest,
                "outputDir": output_dir,
            }

            with self.assertRaisesRegex(RuntimeError, "press_active_train_count_mismatch"):
                runner.completed_run_rows([job], {})

    def test_aggregate_marks_only_non_dominated_windows_as_pareto_front_one(self):
        rows = []
        rows.extend(self.run_rows("pre1_post2", 1, 2, 0.80, 0.60, 1.00))
        rows.extend(self.run_rows("pre5_post9", 5, 9, 0.79, 0.58, 0.95))
        rows.extend(self.run_rows("pre0_post1", 0, 1, 0.75, 0.55, 1.00))

        summaries = runner.aggregate_window_rows(rows)
        by_id = {row["windowId"]: row for row in summaries}

        self.assertEqual(1, by_id["pre1_post2"]["paretoFront"])
        self.assertEqual(1, by_id["pre0_post1"]["paretoFront"])
        self.assertGreater(by_id["pre5_post9"]["paretoFront"], 1)
        self.assertEqual(2, by_id["pre1_post2"]["waitFrames"])
        self.assertAlmostEqual(1.00, by_id["pre1_post2"]["meanOutputCoverage"])

    def run_rows(self, window_id, pre_frames, post_frames, accuracy, min_recall, coverage):
        output = []
        for fold in runner.FOLD_NAMES:
            output.append({
                "windowId": window_id,
                "preFrames": pre_frames,
                "postFrames": post_frames,
                "frameCount": pre_frames + post_frames + 1,
                "fold": fold,
                "bestEpoch": 8,
                "trainRows": 100,
                "eligibleTrainRows": int(100 * coverage),
                "trainCoverage": coverage,
                "validRows": 100,
                "eligibleValidRows": int(100 * coverage),
                "validCoverage": coverage,
                "totalRows": 200,
                "eligibleRows": int(200 * coverage),
                "outputCoverage": coverage,
                "validAccuracy": accuracy,
                "validBalancedAccuracy": accuracy,
                "validMinRecall": min_recall,
                "thumbLightRecall": min_recall,
                "thumbHeavyRecall": min_recall,
                "indexLightRecall": min_recall,
                "indexHeavyRecall": min_recall,
            })
        return output


if __name__ == "__main__":
    unittest.main()
