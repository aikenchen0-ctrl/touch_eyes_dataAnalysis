import unittest

from android.analysis import run_time_grid_channel_effectiveness_ablation as runner


class ChannelEffectivenessRunnerTests(unittest.TestCase):
    def test_specs_cover_all_nine_channel_contracts(self):
        specs = runner.experiment_specs()

        self.assertEqual(9, len(specs))
        self.assertEqual("imu11", specs[0]["contract"])
        self.assertEqual("imu9_no_position", specs[-1]["contract"])

    def test_worker_count_is_bounded_to_requested_limit_and_task_count(self):
        self.assertEqual(7, runner.bounded_worker_count(7, 9))
        self.assertEqual(3, runner.bounded_worker_count(7, 3))

    def test_ranked_rows_prefer_higher_balanced_accuracy_then_lower_loss(self):
        reports = [
            {"contract": "a", "trainMetrics": {"balancedAccuracy": 0.7, "macroF1": 0.6}, "finalTrainLoss": 0.4},
            {"contract": "b", "trainMetrics": {"balancedAccuracy": 0.8, "macroF1": 0.5}, "finalTrainLoss": 0.5},
            {"contract": "c", "trainMetrics": {"balancedAccuracy": 0.7, "macroF1": 0.6}, "finalTrainLoss": 0.3},
        ]

        self.assertEqual(["b", "c", "a"], [row["contract"] for row in runner.ranked_rows(reports)])


if __name__ == "__main__":
    unittest.main()
