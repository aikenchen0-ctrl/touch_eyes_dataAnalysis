import unittest

import numpy as np

try:
    from android.analysis import run_stats_mlp_binary_head_experiments as subject
except ImportError:
    import run_stats_mlp_binary_head_experiments as subject


class BinaryHeadExperimentTests(unittest.TestCase):
    def test_threshold_uses_requested_training_coverage(self):
        probabilities = np.asarray([
            [0.95, 0.05],
            [0.90, 0.10],
            [0.70, 0.30],
            [0.60, 0.40],
        ])

        threshold = subject.threshold_for_coverage(
            probabilities,
            0.75,
        )

        self.assertEqual(0.7, threshold)

    def test_accepted_metrics_report_missing_class_coverage(self):
        result = subject.accepted_head_metrics(
            [0, 1, 0, 1],
            np.asarray([
                [0.95, 0.05],
                [0.40, 0.60],
                [0.90, 0.10],
                [0.20, 0.80],
            ]),
            0.75,
            ("thumb", "index"),
        )

        self.assertEqual(0.75, result["validationCoverage"])
        self.assertEqual(
            {"thumb": 1.0, "index": 0.5},
            result["coverageByClass"],
        )
        self.assertEqual(1.0, result["acceptedMetrics"]["accuracy"])

    def test_task_readiness_rejects_zero_class_recall(self):
        ordinary = {
            "0.9": {
                "ordinaryCount": 12,
                "bothHeadsFalseAcceptanceRate": 0.0,
                "worstFoldBothHeadsFalseAcceptanceRate": 0.0,
                "fourClassFalseAcceptanceRate": 0.0,
                "worstFoldFourClassFalseAcceptanceRate": 0.0,
            },
        }
        result = subject.task_readiness({
            "0.9": {
                "meanCoverage": 0.9,
                "meanAccuracy": 0.9,
                "worstFoldBalancedAccuracy": 0.8,
                "worstClassRecall": 0.0,
            },
        }, {"ready": True}, ordinary)

        self.assertEqual("not_ready", result["status"])
        self.assertIn(
            "zero_worst_class_recall",
            result["reasons"],
        )

    def test_task_readiness_requires_dataset_gate_and_class_floor(self):
        aggregate = {
            "0.9": {
                "meanCoverage": 0.9,
                "meanAccuracy": 0.9,
                "worstFoldBalancedAccuracy": 0.8,
                "worstClassRecall": 0.65,
            },
        }
        ordinary = {
            "0.9": {
                "ordinaryCount": 12,
                "bothHeadsFalseAcceptanceRate": 0.0,
                "worstFoldBothHeadsFalseAcceptanceRate": 0.0,
                "fourClassFalseAcceptanceRate": 0.0,
                "worstFoldFourClassFalseAcceptanceRate": 0.0,
            },
        }

        gated = subject.task_readiness(
            aggregate,
            {
                "ready": False,
                "blockingReasons": ["poseHoldoutReady"],
            },
            ordinary,
        )
        self.assertEqual("not_ready", gated["status"])
        self.assertIn(
            "training_readiness_gate_failed",
            gated["reasons"],
        )

        candidate = subject.task_readiness(
            aggregate,
            {"ready": True},
            ordinary,
        )
        self.assertEqual("candidate", candidate["status"])
        self.assertEqual([], candidate["reasons"])

        deployable = subject.task_readiness(
            {
                "0.9": {
                    **aggregate["0.9"],
                    "worstClassRecall": 0.72,
                },
            },
            {"ready": True},
            ordinary,
        )
        self.assertEqual("deployment_eligible", deployable["status"])

    def test_ordinary_false_acceptance_is_optional_diagnostic(self):
        result = subject.task_readiness(
            {
                "0.9": {
                    "meanCoverage": 0.9,
                    "meanAccuracy": 0.9,
                    "worstFoldBalancedAccuracy": 0.8,
                    "worstClassRecall": 0.72,
                },
            },
            {"ready": True},
            {
                "0.9": {
                    "ordinaryCount": 20,
                    "bothHeadsFalseAcceptanceRate": 0.06,
                    "worstFoldBothHeadsFalseAcceptanceRate": 0.12,
                    "fourClassFalseAcceptanceRate": 0.0,
                    "worstFoldFourClassFalseAcceptanceRate": 0.0,
                },
            },
        )

        self.assertEqual("deployment_eligible", result["status"])
        self.assertEqual([], result["reasons"])
        self.assertEqual(
            0.06,
            result["ordinaryFalseAcceptance"][
                "bothHeadsFalseAcceptanceRate"
            ],
        )

    def test_ordinary_head_acceptance_counts_each_policy(self):
        result = subject.ordinary_head_acceptance(
            np.asarray([
                [0.95, 0.05],
                [0.80, 0.20],
                [0.40, 0.60],
            ]),
            np.asarray([
                [0.90, 0.10],
                [0.30, 0.70],
                [0.20, 0.80],
            ]),
            finger_threshold=0.85,
            strength_threshold=0.85,
        )

        self.assertEqual(3, result["ordinaryCount"])
        self.assertEqual(1, result["fingerFalseAcceptanceCount"])
        self.assertEqual(1, result["strengthFalseAcceptanceCount"])
        self.assertEqual(1, result["anyHeadFalseAcceptanceCount"])
        self.assertEqual(1, result["bothHeadsFalseAcceptanceCount"])
        self.assertAlmostEqual(
            1.0 / 3.0,
            result["bothHeadsFalseAcceptanceRate"],
        )

    def test_ordinary_aggregate_reports_pooled_and_worst_fold(self):
        reports = [
            {
                "ordinary": {
                    "0.9": {
                        "ordinaryCount": 10,
                        "fingerFalseAcceptanceCount": 2,
                        "strengthFalseAcceptanceCount": 2,
                        "anyHeadFalseAcceptanceCount": 2,
                        "bothHeadsFalseAcceptanceCount": 1,
                        "bothHeadsFalseAcceptanceRate": 0.1,
                        "fourClass": {
                            "ordinaryCount": 10,
                            "falseAcceptanceCount": 1,
                            "falseAcceptanceRate": 0.1,
                        },
                    },
                },
            },
            {
                "ordinary": {
                    "0.9": {
                        "ordinaryCount": 30,
                        "fingerFalseAcceptanceCount": 6,
                        "strengthFalseAcceptanceCount": 6,
                        "anyHeadFalseAcceptanceCount": 6,
                        "bothHeadsFalseAcceptanceCount": 3,
                        "bothHeadsFalseAcceptanceRate": 0.1,
                        "fourClass": {
                            "ordinaryCount": 30,
                            "falseAcceptanceCount": 3,
                            "falseAcceptanceRate": 0.1,
                        },
                    },
                },
            },
        ]

        result = subject.aggregate_ordinary_head_results(reports)["0.9"]

        self.assertEqual(40, result["ordinaryCount"])
        self.assertAlmostEqual(
            0.2,
            result["fingerFalseAcceptanceRate"],
        )
        self.assertAlmostEqual(
            0.2,
            result["strengthFalseAcceptanceRate"],
        )
        self.assertAlmostEqual(0.2, result["anyHeadFalseAcceptanceRate"])
        self.assertAlmostEqual(0.1, result["bothHeadsFalseAcceptanceRate"])
        self.assertAlmostEqual(
            0.1,
            result["fourClassFalseAcceptanceRate"],
        )
        self.assertAlmostEqual(0.1, result["fourClassFalseAcceptanceRate"])
        self.assertAlmostEqual(
            0.1,
            result["worstFoldBothHeadsFalseAcceptanceRate"],
        )

    def test_missing_ordinary_audit_does_not_block_deployment(self):
        result = subject.task_readiness(
            {
                "0.9": {
                    "meanCoverage": 0.9,
                    "meanAccuracy": 0.9,
                    "worstFoldBalancedAccuracy": 0.8,
                    "worstClassRecall": 0.72,
                },
            },
            {"ready": True},
            None,
        )

        self.assertEqual("deployment_eligible", result["status"])
        self.assertNotIn(
            "ordinary_negative_samples_missing",
            result["reasons"],
        )

    def test_incomplete_negative_fold_coverage_is_not_a_gate(self):
        result = subject.task_readiness(
            {
                "0.9": {
                    "meanCoverage": 0.9,
                    "meanAccuracy": 0.9,
                    "worstFoldBalancedAccuracy": 0.8,
                    "worstClassRecall": 0.72,
                },
            },
            {"ready": True},
            {
                "0.9": {
                    "ordinaryCount": 20,
                    "bothHeadsFalseAcceptanceRate": 0.0,
                    "worstFoldBothHeadsFalseAcceptanceRate": 0.0,
                    "fourClassFalseAcceptanceRate": 0.0,
                    "worstFoldFourClassFalseAcceptanceRate": 0.0,
                },
            },
            negative_audit={
                "totalCount": 20,
                "matchedCount": 15,
                "scoredCount": 12,
                "positiveFoldCount": 4,
                "matchedFoldCount": 3,
                "scoredFoldCount": 2,
            },
        )

        self.assertEqual("deployment_eligible", result["status"])
        self.assertEqual([], result["reasons"])
        self.assertEqual(
            2,
            result["ordinaryNegativeAudit"]["scoredFoldCount"],
        )

    def test_class_balanced_sampling_keeps_each_class(self):
        samples = [
            {"label": index % 4}
            for index in range(16)
        ]

        selected = subject.class_balanced_indices(
            samples,
            seed=17,
            count_per_class=2,
        )
        labels = [samples[index]["label"] for index in selected]

        self.assertEqual(8, len(selected))
        self.assertEqual(
            {0: 2, 1: 2, 2: 2, 3: 2},
            {
                label: labels.count(label)
                for label in range(4)
            },
        )


if __name__ == "__main__":
    unittest.main()
