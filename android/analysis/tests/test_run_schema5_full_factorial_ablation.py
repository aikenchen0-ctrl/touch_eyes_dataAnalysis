import contextlib
import io
import sys
import tempfile
import unittest
import json
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_DIR))
import run_schema5_full_factorial_ablation as runner


class Schema5FullFactorialAblationTests(unittest.TestCase):
    def test_specs_cover_all_presets_windows_and_motion_folds(self):
        temporal = runner.temporal_specs(Path("manifests"), Path("models"), Path("cache"))
        strict = runner.strict_specs(Path("manifests"), Path("models"), Path("cache"))

        self.assertEqual(84, len(temporal))
        self.assertEqual(252, len(strict))
        self.assertEqual(
            set(runner.FEATURE_PRESET_NAMES),
            {item["preset"] for item in temporal},
        )
        self.assertEqual(set(runner.WINDOWS), {item["window"] for item in temporal})

    def test_baseline_and_nonbaseline_paths_preserve_existing_layout(self):
        temporal = runner.temporal_specs(Path("manifests"), Path("models"), Path("cache"))
        strict = runner.strict_specs(Path("manifests"), Path("models"), Path("cache"))
        baseline = next(
            item for item in temporal
            if item["preset"] == "old13" and item["window"] == "pre5_post9"
        )
        alternate = next(
            item for item in temporal
            if item["preset"] == "old13" and item["window"] == "pre5_post0"
        )
        holdout = next(
            item for item in strict
            if item["preset"] == "old13"
            and item["window"] == "pre5_post9"
            and item["motion"] == "walking"
        )

        self.assertEqual(Path("models/feature_temporal/old13"), baseline["outputDir"])
        self.assertEqual(
            Path("models/window_temporal/old13__pre5_post0"),
            alternate["outputDir"],
        )
        self.assertEqual(
            Path("models/motion_holdout/old13__pre5_post9/walking"),
            holdout["outputDir"],
        )

    def test_build_command_uses_locked_stage_contract(self):
        temporal = runner.temporal_specs(Path("manifests"), Path("models"), Path("cache"))[0]
        strict = runner.strict_specs(Path("manifests"), Path("models"), Path("cache"))[0]

        temporal_command = runner.build_command(temporal, "python")
        strict_command = runner.build_command(strict, "python")

        self.assertIn("--anchor-mode", temporal_command)
        self.assertIn("down", temporal_command)
        self.assertIn("--require-press-active", temporal_command)
        self.assertIn("--require-press-active", strict_command)
        self.assertIn("--missing-frame-policy", temporal_command)
        self.assertIn("error", temporal_command)
        self.assertIn("--split-strategy", temporal_command)
        self.assertIn("predefined", temporal_command)
        self.assertIn("--epochs", temporal_command)
        self.assertEqual("60", temporal_command[temporal_command.index("--epochs") + 1])
        self.assertEqual(
            "12",
            temporal_command[temporal_command.index("--early-stopping-patience") + 1],
        )
        self.assertEqual("120", strict_command[strict_command.index("--epochs") + 1])
        self.assertEqual(
            "20",
            strict_command[strict_command.index("--early-stopping-patience") + 1],
        )

    def test_completed_spec_requires_report_weights_and_scaler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = runner.temporal_specs(root, root / "models", root / "cache")[0]
            spec["outputDir"].mkdir(parents=True)

            self.assertFalse(runner.completed_spec(spec))
            (spec["outputDir"] / "training_report.json").write_text("{}", encoding="utf-8")
            self.assertFalse(runner.completed_spec(spec))
            (spec["outputDir"] / (
                "AispectImpactCNNWeights_" + spec["modelId"] + ".json"
            )).write_text("{}", encoding="utf-8")
            self.assertFalse(runner.completed_spec(spec))
            (spec["outputDir"] / (
                "AispectImpactCNNScaler_" + spec["modelId"] + ".json"
            )).write_text("{}", encoding="utf-8")

            self.assertTrue(runner.completed_spec(spec))

    def test_strict_aggregate_ranks_mean_then_worst_holdout_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = [
                item for item in runner.strict_specs(root, root / "models", root / "cache")
                if item["preset"] == "old13" and item["window"] == "pre5_post9"
            ]
            self.write_complete_report(specs[0], 0.70, 0.30)
            self.write_complete_report(specs[1], 0.60, 0.20)
            self.write_complete_report(specs[2], 0.80, 0.40)

            rows, aggregates = runner.aggregate_strict_specs(specs)

        self.assertEqual(3, len(rows))
        self.assertEqual(1, len(aggregates))
        aggregate = aggregates[0]
        self.assertEqual("old13", aggregate["preset"])
        self.assertEqual("pre5_post9", aggregate["window"])
        self.assertEqual(3, aggregate["completedFolds"])
        self.assertEqual(0, aggregate["pendingFolds"])
        self.assertAlmostEqual(0.70, aggregate["meanValidBalancedAccuracy"])
        self.assertAlmostEqual(0.60, aggregate["worstValidBalancedAccuracy"])
        self.assertAlmostEqual(0.20, aggregate["worstValidMinRecall"])

    def test_dry_run_writes_full_matrix_status_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "models"
            with contextlib.redirect_stdout(io.StringIO()):
                runner.main([
                    "--manifest-root", str(root / "manifests"),
                    "--output-root", str(output_root),
                    "--stage", "temporal",
                    "--dry-run",
                ])
            status = json.loads((output_root / "full_factorial_status.json").read_text(
                encoding="utf-8",
            ))

        self.assertEqual("temporal", status["requestedStage"])
        self.assertEqual(84, status["temporal"]["planned"])
        self.assertEqual(0, status["temporal"]["completed"])
        self.assertEqual(84, status["temporal"]["pending"])
        self.assertEqual(252, status["strict"]["planned"])
        self.assertEqual(252, status["strict"]["pending"])

    def write_complete_report(self, spec, balanced_accuracy, min_recall):
        spec["outputDir"].mkdir(parents=True)
        report = {
            "bestEpoch": 8,
            "trainMetrics": {"count": 10, "balancedAccuracy": 0.9},
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
        (spec["outputDir"] / "training_report.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )
        (spec["outputDir"] / (
            "AispectImpactCNNWeights_" + spec["modelId"] + ".json"
        )).write_text("{}", encoding="utf-8")
        (spec["outputDir"] / (
            "AispectImpactCNNScaler_" + spec["modelId"] + ".json"
        )).write_text("{}", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
