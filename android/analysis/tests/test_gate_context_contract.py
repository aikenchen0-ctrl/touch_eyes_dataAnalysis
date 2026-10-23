import unittest

import numpy as np

from android.analysis import gate_threshold_metrics as threshold_metrics
from android.analysis import stats_mlp_contract as contract


class GateContextContractTests(unittest.TestCase):
    def test_scenario_threshold_report_uses_explicit_thresholds(self):
        rows = [
            {"deviceName": "a", "motionState": "handheld_static"},
            {"deviceName": "a", "motionState": "walking"},
        ]
        report = threshold_metrics.fixed_scenario_threshold_report(
            rows,
            [0, 1],
            [0.2, 0.9],
            {"walking": 0.8},
            fallback_threshold=0.3,
            scenario_values=["handheld_static", "walking"],
        )

        self.assertEqual(1, report["overall"]["truePositiveCount"])
        self.assertEqual(0, report["overall"]["falsePositiveCount"])
        self.assertEqual(
            "calibration_scenario_threshold",
            report["thresholdPolicy"],
        )
        self.assertEqual(0.8, report["thresholds"]["walking"])

    def test_scenario_threshold_report_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(
                ValueError,
                "fixed_scenario_shape_mismatch"):
            threshold_metrics.fixed_scenario_threshold_report(
                [{"deviceName": "a"}],
                [0],
                [0.4],
                {},
                fallback_threshold=0.3,
                scenario_values=[],
            )

    def test_direct_context_aliases_are_normalized(self):
        self.assertEqual(
            (0, 1, 3, 0),
            contract.context_targets(
                "handheld_static",
                "hard_surface",
                "side_left",
                "tilted_toward_user",
            ),
        )
        self.assertEqual(
            (1, 0, 4, 5),
            contract.context_targets(
                "walking",
                "handheld",
                "right_side",
                "natural",
            ),
        )

    def test_task_metadata_fills_only_missing_direct_fields(self):
        self.assertEqual(
            (0, -100, 2, 3),
            contract.context_targets(
                "unknown",
                "unspecified",
                "unknown",
                "unconstrained",
                task_id="supine",
            ),
        )
        self.assertEqual(
            (0, -100, 2, 3),
            contract.context_targets_for_row({
                "motionState": "unknown",
                "support": "unspecified",
                "bodyPosture": "unknown",
                "requestedDevicePose": "unconstrained",
                "taskId": "supine",
            }),
        )
        self.assertEqual(
            (0, 0, 0, 0),
            contract.context_targets(
                "unknown",
                "unspecified",
                "unknown",
                "unconstrained",
                task_id="tilted_handheld",
            ),
        )

    def test_unknown_metadata_is_masked_and_not_a_class(self):
        self.assertNotIn("unknown", contract.MOTION_LABELS)
        self.assertNotIn("unknown", contract.SUPPORT_LABELS)
        self.assertNotIn("unknown", contract.POSTURE_LABELS)
        self.assertNotIn("unknown", contract.DEVICE_POSE_LABELS)
        self.assertEqual(
            (
                contract.CONTEXT_IGNORE_INDEX,
                contract.CONTEXT_IGNORE_INDEX,
                contract.CONTEXT_IGNORE_INDEX,
                contract.CONTEXT_IGNORE_INDEX,
            ),
            contract.context_targets(
                "unknown",
                "unspecified",
                "unconstrained",
                "",
                task_id="unknown",
            ),
        )

    def test_invalid_context_is_rejected_instead_of_silently_masked(self):
        with self.assertRaisesRegex(ValueError, "invalid_context_label"):
            contract.context_targets(
                "not-a-motion-state",
                "unknown",
                "unknown",
                "unknown",
            )
        with self.assertRaisesRegex(ValueError, "invalid_context_task"):
            contract.context_targets(
                "unknown",
                "unknown",
                "unknown",
                "unknown",
                task_id="not-a-task",
            )

    def test_fixed_fpr_uses_one_global_threshold(self):
        rows = [
            {"deviceName": "a", "bodyPosture": "supine", "motionState": "still",
             "groupId": "s1"},
            {"deviceName": "a", "bodyPosture": "supine", "motionState": "still",
             "groupId": "s1"},
            {"deviceName": "b", "bodyPosture": "side_left", "motionState": "walking",
             "groupId": "s2"},
            {"deviceName": "b", "bodyPosture": "side_left", "motionState": "walking",
             "groupId": "s2"},
        ]
        truths = [0, 0, 1, 1]
        scores = [0.10, 0.90, 0.80, 0.95]
        report = threshold_metrics.fixed_fpr_report(
            rows,
            truths,
            scores,
            target_fpr=0.5,
        )
        self.assertEqual("light_touch", report["negativeProxy"])
        self.assertFalse(report["ordinaryClickFprAvailable"])
        self.assertEqual("single_global_threshold", report["thresholdPolicy"])
        self.assertEqual(1, report["overall"]["falsePositiveCount"])
        self.assertEqual(2, report["overall"]["truePositiveCount"])
        self.assertIn("a", report["groups"]["deviceName"])
        self.assertIn("b", report["groups"]["deviceName"])
        self.assertEqual(
            report["overall"]["threshold"],
            report["groups"]["deviceName"]["a"]["threshold"],
        )
        self.assertEqual(
            report["overall"]["threshold"],
            report["groups"]["deviceName"]["b"]["threshold"],
        )

    def test_fixed_fpr_is_deterministic_and_bounded(self):
        scores = np.asarray([0.2, 0.2, 0.7, 0.8, 0.9])
        threshold = threshold_metrics.select_threshold_at_fpr(
            scores,
            target_fpr=0.0,
        )
        self.assertEqual(0.9, threshold)
        metrics = threshold_metrics.binary_metrics(
            [0, 0, 1, 1, 1],
            scores,
            threshold,
        )
        self.assertEqual(0, metrics["falsePositiveCount"])
        self.assertEqual(0, metrics["truePositiveCount"])

    def test_roc_auc_is_threshold_free_and_handles_ties(self):
        truths = [0, 0, 1, 1, 1]
        scores = [0.2, 0.8, 0.2, 0.8, 0.8]
        self.assertAlmostEqual(
            0.5833333333333334,
            threshold_metrics.roc_auc(truths, scores),
        )
        metrics = threshold_metrics.binary_metrics(
            truths,
            scores,
            threshold=0.5,
        )
        self.assertEqual(metrics["rocAuc"], threshold_metrics.roc_auc(
            truths,
            scores,
        ))

    def test_roc_auc_requires_both_classes(self):
        self.assertIsNone(
            threshold_metrics.roc_auc([0, 0], [0.1, 0.2])
        )
        self.assertIsNone(
            threshold_metrics.roc_auc([1, 1], [0.8, 0.9])
        )


if __name__ == "__main__":
    unittest.main()
