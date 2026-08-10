import json
import sys
import tempfile
import unittest
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_DIR))
import run_group_invariant_four_class_experiments as runner


class GroupInvariantRunnerTests(unittest.TestCase):
    def test_experiment_specs_cover_each_transform_and_fold(self):
        specs = runner.experiment_specs(Path("manifests"), Path("models"), Path("cache.pkl"))

        self.assertEqual(len(specs), 12)
        self.assertEqual({spec["transform"] for spec in specs}, set(runner.TRANSFORMS))
        self.assertEqual({spec["fold"] for spec in specs}, set(runner.FOLDS))

    def test_build_command_enables_transform_balance_cache_and_early_stop(self):
        spec = runner.experiment_specs(Path("manifests"), Path("models"), Path("cache.pkl"))[0]

        command = runner.build_command(spec, "python")

        self.assertIn("--matrix-transform", command)
        self.assertIn(spec["transform"], command)
        self.assertIn("--group-balance", command)
        self.assertIn("--sample-cache", command)
        self.assertIn("--early-stopping-patience", command)

    def test_aggregate_reports_computes_candidate_mean_and_worst_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_report(root, "imu11_no_time", "fold_a", 0.7, 0.4)
            self.write_report(root, "imu11_no_time", "fold_b", 0.5, 0.3)

            rows, candidates = runner.aggregate_reports(root)

        self.assertEqual(len(rows), 2)
        summary = candidates[0]
        self.assertEqual(summary["matrixTransform"], "imu11_no_time")
        self.assertAlmostEqual(summary["meanValidBalancedAccuracy"], 0.6)
        self.assertAlmostEqual(summary["worstFoldBalancedAccuracy"], 0.5)
        self.assertAlmostEqual(summary["worstClassRecall"], 0.3)

    def write_report(self, root, transform, fold, balanced_accuracy, min_recall):
        output = root / transform / fold
        output.mkdir(parents=True)
        report = {
            "args": {"matrix_transform": transform},
            "history": [{"epoch": 1, "validBalancedAccuracy": balanced_accuracy}],
            "trainMetrics": {"count": 10, "balancedAccuracy": 0.8},
            "validMetrics": {
                "count": 4,
                "accuracy": balanced_accuracy,
                "balancedAccuracy": balanced_accuracy,
                "classRecalls": {
                    "thumb_light": min_recall,
                    "thumb_heavy": 0.8,
                    "index_light": 0.7,
                    "index_heavy": 0.6,
                },
                "confusionMatrix": [[1, 0, 0, 0]] * 4,
            },
        }
        (output / "training_report.json").write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
