import json
import tempfile
import unittest
from pathlib import Path

from android.analysis import run_clean_v3_probe_experiments as runner


class CleanV3ProbeRunnerTests(unittest.TestCase):
    def test_specs_cover_candidates_and_discovered_domain_folds(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            folds = root / "folds"
            folds.mkdir()
            (folds / "domain_a.csv").write_text("id\n", encoding="utf-8")
            (folds / "domain_b.csv").write_text("id\n", encoding="utf-8")

            specs = runner.experiment_specs(folds, root / "models", root / "cache")

            self.assertEqual(8, len(specs))
            self.assertEqual(set(runner.CANDIDATES), {spec["candidate"] for spec in specs})
            self.assertEqual({"domain_a", "domain_b"}, {spec["fold"] for spec in specs})

    def test_build_command_uses_probe_controls_and_predefined_split(self):
        candidate = runner.CANDIDATES["baseline_imu"]
        spec = {
            "candidate": "baseline_imu",
            "candidateConfig": candidate,
            "fold": "domain_a",
            "manifest": Path("folds/domain_a.csv"),
            "outputDir": Path("models/baseline_imu/domain_a"),
            "sampleCache": Path("cache/enhanced_patch21_no_confidence.pkl"),
            "modelId": "model-id",
        }

        command = runner.build_command(
            spec,
            "python",
            epochs=30,
            patience=8,
            batch_size=128,
            seed=22,
        )

        self.assertIn("--split-strategy", command)
        self.assertIn("predefined", command)
        self.assertIn("--group-balance", command)
        self.assertIn("baseline_imu", command)
        self.assertIn("30", command)
        self.assertIn("8", command)

    def test_pending_specs_skip_completed_reports(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "candidate" / "fold"
            output.mkdir(parents=True)
            spec = {"outputDir": output}
            (output / "training_report.json").write_text("{}", encoding="utf-8")

            self.assertEqual([], runner.pending_specs([spec], force=False))
            self.assertEqual([spec], runner.pending_specs([spec], force=True))

    def test_failed_gate_selects_no_candidate_for_full_training(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rows = [{"candidate": "baseline_imu", "fold": "domain-a"}]
            candidates = [
                {
                    "candidate": "baseline_imu",
                    "completedFolds": 5,
                    "channelCount": 11,
                    "meanValidBalancedAccuracy": 0.42,
                    "worstFoldBalancedAccuracy": 0.32,
                    "bestFoldBalancedAccuracy": 0.57,
                    "worstClassRecall": 0.04,
                }
            ]

            runner.write_reports(root, rows, candidates)

            analysis = json.loads((root / "probe_analysis.json").read_text(encoding="utf-8"))
            self.assertFalse(analysis["targetMet"])
            self.assertEqual([], analysis["selectedCandidates"])
            self.assertFalse(analysis["fullTrainingRecommended"])

    def test_aggregate_reports_ranks_mean_then_worst_fold(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.write_report(root, "old13", "domain_a", 0.70, 0.40)
            self.write_report(root, "old13", "domain_b", 0.50, 0.30)
            self.write_report(root, "patch21", "domain_a", 0.61, 0.45)
            self.write_report(root, "patch21", "domain_b", 0.61, 0.42)

            rows, candidates = runner.aggregate_reports(root)

            self.assertEqual(4, len(rows))
            self.assertEqual("patch21", candidates[0]["candidate"])
            self.assertEqual(11, candidates[0]["channelCount"])
            self.assertAlmostEqual(0.61, candidates[0]["meanValidBalancedAccuracy"])
            self.assertAlmostEqual(0.61, candidates[0]["worstFoldBalancedAccuracy"])
            self.assertAlmostEqual(0.42, candidates[0]["worstClassRecall"])

    def write_report(self, root, candidate, fold, balanced_accuracy, min_recall):
        output = root / candidate / fold
        output.mkdir(parents=True)
        report = {
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
        (output / "training_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        (output / ("AispectImpactCNNScaler_" + candidate + ".json")).write_text(
            json.dumps({"featureNames": ["feature"] * 11}), encoding="utf-8"
        )


if __name__ == "__main__":
    unittest.main()
