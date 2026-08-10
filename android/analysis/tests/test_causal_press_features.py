import copy
import json
import sys
import unittest
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ANALYSIS_DIR.parent / "aispect-core" / "src" / "test" / "resources" / "causal_press_features_v1_fixture.json"
sys.path.insert(0, str(ANALYSIS_DIR))

try:
    import causal_press_features as causal
except ModuleNotFoundError:
    causal = None


class CausalPressFeatureTests(unittest.TestCase):
    def load_fixture(self):
        return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def assert_matrix_close(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for expected_row, actual_row in zip(expected, actual):
            self.assertEqual(len(expected_row), len(actual_row))
            for expected_value, actual_value in zip(expected_row, actual_row):
                self.assertAlmostEqual(expected_value, actual_value, places=12)

    def test_zero_order_hold_uses_latest_received_touch_frame(self):
        self.assertIsNotNone(causal)
        fixture = self.load_fixture()

        actual = causal.build_matrix(
            fixture,
            "causal_imu11_touch9_v1",
            [0, 1],
            "error",
        )

        self.assert_matrix_close(fixture["expected"]["causal_imu11_touch9_v1"], actual)

    def test_future_touch_and_lift_metadata_cannot_change_causal_matrix(self):
        self.assertIsNotNone(causal)
        fixture = self.load_fixture()
        baseline = causal.build_matrix(fixture, "causal_imu11_touch9_v1", [0, 1], "error")
        mutated = copy.deepcopy(fixture)
        mutated["liftReferenceFrameIndex"] = 99
        mutated["timeSinceAnchor"] = 99.0
        mutated["timeSinceTouchDown"] = 99.0
        mutated["frames"][0]["timeSinceAnchor"] = 99.0
        mutated["frames"][0]["timeSinceTouchDown"] = 99.0
        mutated["contactPatch"]["covariance"]["cov_11"] = -999.0
        mutated["touchNormalization"]["confidence"] = -999.0
        mutated["touchFrames"][0]["contactPatch"] = {"confidence": -999.0}
        mutated["touchFrames"][2]["x_norm"] = 0.0
        mutated["touchFrames"][2]["touchMajor"] = 0.0

        actual = causal.build_matrix(mutated, "causal_imu11_touch9_v1", [0, 1], "error")

        self.assert_matrix_close(baseline, actual)

    def test_selected_touch_requires_view_dimensions(self):
        self.assertIsNotNone(causal)
        fixture = self.load_fixture()
        del fixture["touchFrames"][0]["viewWidthPx"]

        with self.assertRaisesRegex(RuntimeError, "causal_v1_missing_touch_view_size"):
            causal.build_matrix(fixture, "causal_imu11_touch9_v1", [0, 1], "error")

    def test_imu_frame_requires_received_elapsed_realtime_timestamp(self):
        self.assertIsNotNone(causal)
        fixture = self.load_fixture()
        del fixture["frames"][0]["receivedElapsedRealtimeNanos"]

        with self.assertRaisesRegex(RuntimeError, "causal_v1_missing_imu_received_timestamp"):
            causal.build_matrix(fixture, "causal_imu12_v1", [0, 1], "error")

    def test_received_touch_with_future_event_is_not_selected(self):
        self.assertIsNotNone(causal)
        fixture = self.load_fixture()
        fixture["frames"][0]["receivedElapsedRealtimeNanos"] = 350
        fixture["frames"][0]["sensorTimestampElapsedRealtimeNanos"] = 250

        actual = causal.build_matrix(fixture, "causal_imu11_touch9_v1", [0], "error")

        self.assertAlmostEqual(0.25, actual[0][0], places=12)
        self.assertAlmostEqual(0.50, actual[0][1], places=12)

    def test_imu_frame_requires_sensor_timestamp_for_physical_time_gate(self):
        self.assertIsNotNone(causal)
        fixture = self.load_fixture()
        del fixture["frames"][0]["sensorTimestampElapsedRealtimeNanos"]

        with self.assertRaisesRegex(RuntimeError, "causal_v1_missing_imu_sensor_timestamp"):
            causal.build_matrix(fixture, "causal_imu12_v1", [0], "error")

    def test_base_contract_exposes_touch_valid_for_empty_history(self):
        self.assertIsNotNone(causal)
        fixture = self.load_fixture()
        fixture["touchFrames"] = []

        actual = causal.build_matrix(fixture, "causal_imu12_v1", [0], "error")

        self.assertEqual(12, len(actual[0]))
        self.assertEqual(0.0, actual[0][0])
        self.assertEqual(0.0, actual[0][1])
        self.assertEqual(0.0, actual[0][-1])

    def test_contracts_have_fixed_channel_counts_and_no_excluded_features(self):
        self.assertIsNotNone(causal)
        self.assertEqual(12, len(causal.feature_names("causal_imu12_v1")))
        self.assertEqual(20, len(causal.feature_names("causal_imu11_touch9_v1")))
        excluded = ("time", "contactpatch", "touchnormalization", "gaussian", "covariance", "confidence")
        for preset in ("causal_imu12_v1", "causal_imu11_touch9_v1"):
            for name in causal.feature_names(preset):
                self.assertFalse(any(token in name.lower() for token in excluded))


if __name__ == "__main__":
    unittest.main()
