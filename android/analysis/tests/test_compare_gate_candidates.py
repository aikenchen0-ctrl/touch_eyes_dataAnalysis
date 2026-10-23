import unittest
from collections import Counter
from unittest import mock

try:
    from android.analysis import compare_gate_candidates as candidates
except ImportError:
    import compare_gate_candidates as candidates


class CompareGateCandidatesTest(unittest.TestCase):
    def test_parse_values_rejects_unknown_values(self):
        with self.assertRaisesRegex(RuntimeError, "invalid_models"):
            candidates.parse_values(
                "logistic_regression,unknown",
                candidates.MODEL_NAMES,
                "models",
            )

    def test_evidence_fields_require_independent_proof_sessions(self):
        calibration_rows = [
            {"groupId": "calibration-" + str(index)}
            for index in range(5)
            for _ in range(100)
        ]
        valid_rows = [
            {"groupId": "valid-" + str(index)}
            for index in range(4)
            for _ in range(125)
        ]
        report = {
            "overall": {
                "heavyRecall": 0.8,
                "lightFalsePositiveRate": 0.01,
            },
        }
        candidates.attach_evidence_fields(
            report,
            calibration_rows,
            [0] * len(calibration_rows),
            valid_rows,
            [0] * len(valid_rows),
            0.01,
        )

        self.assertFalse(report["proofReady"])
        self.assertEqual(4, report["heldOutNegativeSessionCount"])

    def test_rank_prefers_worst_device_recall(self):
        summaries = [
            {
                "model": "model-a",
                "profile": "raw",
                "targetFalsePositiveRate": 0.01,
                "worstDeviceHeavyRecall": 0.4,
                "worstPostureHeavyRecall": 0.5,
                "worstMotionHeavyRecall": 0.5,
                "meanHeavyRecall": 0.8,
                "meanRuntimeParameterCount": 100,
            },
            {
                "model": "model-b",
                "profile": "compact",
                "targetFalsePositiveRate": 0.01,
                "worstDeviceHeavyRecall": 0.7,
                "worstPostureHeavyRecall": 0.3,
                "worstMotionHeavyRecall": 0.3,
                "meanHeavyRecall": 0.75,
                "meanRuntimeParameterCount": 1000,
            },
        ]

        ranked = candidates.rank_candidates(summaries)

        self.assertEqual("model-b", ranked[0]["model"])

    def test_known_group_values_exclude_missing_metadata(self):
        metrics = {
            "groups": {
                "bodyPosture": {
                    "unknown": {"heavyRecall": 0.2},
                    "": {"heavyRecall": 0.3},
                    "supine": {"heavyRecall": 0.8},
                },
            },
        }

        values = candidates.known_group_values(metrics, "bodyPosture")

        self.assertEqual([{"heavyRecall": 0.8}], values)

    def test_scenario_calibration_requires_independent_negative_floor(self):
        rows = [
            {
                "groupId": "still-" + str(index),
                "motionState": "handheld_static",
            }
            for index in range(6)
            for _ in range(100)
        ]
        rows.extend([
            {
                "groupId": "walking-" + str(index),
                "motionState": "walking",
            }
            for index in range(4)
            for _ in range(100)
        ])
        by_scenario = {}
        for row in rows:
            by_scenario.setdefault(
                candidates.gate.scenario_domain(row),
                [],
            ).append(0.25)

        thresholds = candidates.scenario_thresholds_at_fpr(
            by_scenario,
            fallback_threshold=0.8,
            target_fpr=0.01,
        )

        self.assertEqual(0.25, thresholds["handheld_static"])
        self.assertEqual(0.8, thresholds["walking"])

    def test_calibration_consensus_is_deterministic_and_uses_complete_sessions(self):
        session_sizes = {
            "session-a": 2,
            "session-b": 3,
            "session-c": 4,
            "session-d": 5,
            "session-e": 6,
        }
        session_scores = {
            "session-a": 0.11,
            "session-b": 0.22,
            "session-c": 0.33,
            "session-d": 0.44,
            "session-e": 0.55,
        }
        rows = []
        labels = []
        scores = []
        for session, size in session_sizes.items():
            for _ in range(size):
                rows.append({
                    "groupId": session,
                    "partition": "calibration",
                })
                labels.append(0)
                scores.append(session_scores[session])
        # 中文注释：正样本和带外测分区标记的行都不能影响校准阈值。
        rows.extend([
            {"groupId": "positive", "partition": "calibration"},
            {"groupId": "external", "partition": "valid"},
        ])
        labels.extend([1, 0])
        scores.extend([0.99, 0.99])

        calls = []

        def record_threshold(values, target):
            calls.append((float(target), tuple(sorted(values))))
            return 0.5

        with mock.patch.object(
                candidates.gate_threshold_metrics,
                "select_threshold_at_fpr",
                side_effect=record_threshold):
            first = candidates.calibration_consensus_scores(
                rows,
                labels,
                scores,
                repeats=5,
                seed=17,
            )
            calls.clear()
            second = candidates.calibration_consensus_scores(
                rows,
                labels,
                scores,
                repeats=5,
                seed=17,
            )

        self.assertEqual(first, second)
        self.assertEqual(len(calls), 10)
        for target, values in calls:
            self.assertIn(target, {float(value) for value in candidates.gate.TARGET_FPRS})
            counts = Counter(values)
            selected_sessions = [
                session
                for session, value in session_scores.items()
                if counts[value]
            ]
            self.assertEqual(4, len(selected_sessions))
            for session, value in session_scores.items():
                self.assertIn(
                    counts[value],
                    (0, session_sizes[session]),
                )
            self.assertNotIn(0.99, counts)

    def test_calibration_consensus_requires_two_negative_sessions(self):
        with self.assertRaisesRegex(
                RuntimeError,
                "calibration_missing_negative_proxy"):
            candidates.calibration_consensus_scores(
                [{"groupId": "only-session"}],
                [0],
                [0.2],
                repeats=3,
                seed=1,
            )

        with self.assertRaisesRegex(
                RuntimeError,
                "calibration_missing_negative_proxy"):
            candidates.calibration_consensus_scores(
                [
                    {"groupId": "positive", "partition": "calibration"},
                    {"groupId": "external", "partition": "valid"},
                ],
                [1, 0],
                [0.9, 0.2],
                repeats=3,
                seed=1,
            )

    def test_release_gate_reports_scenario_candidate_blockers(self):
        candidate = {
            "seedCount": 1,
            "proofReady": False,
            "worstDeviceHeavyRecall": 0.88,
            "worstPostureHeavyRecall": None,
            "worstMotionHeavyRecall": 0.86,
            "worstDeviceLightFalsePositiveRate": 0.06,
        }

        assessment = candidates.assess_release_candidate(candidate, 3)

        self.assertFalse(assessment["eligible"])
        self.assertIn("insufficient_seed_evidence", assessment["blockers"])
        self.assertIn(
            "negative_evidence_not_proof_ready",
            assessment["blockers"],
        )
        self.assertIn(
            "device_recall_below_target",
            assessment["blockers"],
        )
        self.assertIn("posture_recall_unavailable", assessment["blockers"])
        self.assertIn(
            "false_positive_rate_above_target",
            assessment["blockers"],
        )

    def test_release_gate_accepts_complete_candidate(self):
        candidate = {
            "seedCount": 3,
            "proofReady": True,
            "worstDeviceHeavyRecall": 0.90,
            "worstPostureHeavyRecall": 0.91,
            "worstMotionHeavyRecall": 0.92,
            "worstDeviceLightFalsePositiveRate": 0.01,
        }

        assessment = candidates.assess_release_candidate(candidate, 3)

        self.assertTrue(assessment["eligible"])
        self.assertEqual([], assessment["blockers"])



if __name__ == "__main__":
    unittest.main()
