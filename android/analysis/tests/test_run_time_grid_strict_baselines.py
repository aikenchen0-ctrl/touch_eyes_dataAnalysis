import unittest

from android.analysis import run_time_grid_strict_baselines as runner


class StrictBaselineRunnerTests(unittest.TestCase):
    def test_specs_cover_two_contracts_and_three_folds(self):
        specs = runner.experiment_specs()

        self.assertEqual(6, len(specs))
        self.assertEqual({"imu11", "touch23"}, {spec["contract"] for spec in specs})
        self.assertEqual({0, 1, 2}, {spec["fold"] for spec in specs})

    def test_touch_candidate_requires_stable_relative_gain(self):
        imu = {
            "meanBalancedAccuracy": 0.70,
            "worstFoldBalancedAccuracy": 0.65,
            "worstClassRecall": 0.60,
            "meanGeneralizationGap": 0.10,
            "worstGeneralizationGap": 0.20,
        }
        touch = {
            "meanBalancedAccuracy": 0.72,
            "worstFoldBalancedAccuracy": 0.65,
            "worstClassRecall": 0.59,
            "meanGeneralizationGap": 0.12,
            "worstGeneralizationGap": 0.22,
        }

        self.assertTrue(runner.promotion_decision(imu, touch)["passed"])
        touch["worstClassRecall"] = 0.55
        self.assertFalse(runner.promotion_decision(imu, touch)["passed"])

    def test_aggregate_reports_each_class_recall_and_parameter_count(self):
        reports = []
        for fold in range(3):
            reports.append({
                "contract": "imu11",
                "bestEpoch": fold + 1,
                "parameterCount": 4452,
                "featureCount": 11,
                "generalizationGap": 0.1,
                "validMetrics": {
                    "balancedAccuracy": 0.5 + fold * 0.1,
                    "macroF1": 0.4,
                    "recallByClass": {
                        "thumb_light": 0.4 + fold * 0.1,
                        "thumb_heavy": 0.5,
                        "index_light": 0.6,
                        "index_heavy": 0.7,
                    },
                },
            })

        result = runner.aggregate_contract(reports)

        self.assertEqual(4452, result["parameterCount"])
        self.assertEqual([9, 11], result["inputShape"])
        self.assertAlmostEqual(0.5, result["meanRecallByClass"]["thumb_light"])
        self.assertAlmostEqual(0.4, result["worstFoldRecallByClass"]["thumb_light"])


if __name__ == "__main__":
    unittest.main()
