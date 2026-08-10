import json
import tempfile
import unittest
from pathlib import Path

from android.analysis.train_android_first_touch_mlp import FEATURE_NAMES
from android.analysis.train_android_first_touch_mlp import build_export_payload
from android.analysis.train_android_first_touch_mlp import first_touch_features
from android.analysis.train_android_first_touch_mlp import should_update_best_state
from android.analysis.train_android_first_touch_mlp import stratified_indices


class FirstTouchFeatureTests(unittest.TestCase):
    def test_extracts_normalized_first_touch_features(self):
        data = {
            "touchFrames": [
                {
                    "x_norm": 0.25,
                    "y_norm": 0.75,
                    "pressure": 0.8,
                    "size": 0.12,
                    "contactPatch": {
                        "normalized": {
                            "sigmaX": 0.02,
                            "sigmaY": 0.01,
                            "orientationRad": 0.5,
                        }
                    },
                }
            ]
        }

        values = first_touch_features(data)

        self.assertEqual(len(FEATURE_NAMES), len(values))
        self.assertAlmostEqual(0.25, values[0])
        self.assertAlmostEqual(0.75, values[1])
        self.assertAlmostEqual(0.8, values[2])
        self.assertAlmostEqual(0.12, values[3])
        self.assertAlmostEqual(0.02, values[4])
        self.assertAlmostEqual(0.01, values[5])

    def test_uses_default_patch_when_first_frame_has_no_patch(self):
        data = {
            "touchFrames": [
                {
                    "x_norm": 0.4,
                    "y_norm": 0.6,
                    "pressure": 1.0,
                    "size": 0.02,
                }
            ]
        }

        values = first_touch_features(data)

        self.assertAlmostEqual(0.012, values[4])
        self.assertAlmostEqual(0.012, values[5])
        self.assertAlmostEqual(0.0, values[6])

    def test_returns_none_without_touch_frames(self):
        self.assertIsNone(first_touch_features({"touchFrames": []}))


class FirstTouchTrainingControlTests(unittest.TestCase):
    def test_all_split_always_keeps_latest_epoch(self):
        self.assertTrue(should_update_best_state(False, 0.4, 0.5))
        self.assertTrue(should_update_best_state(True, 0.6, 0.5))
        self.assertFalse(should_update_best_state(True, 0.4, 0.5))

    def test_stratified_indices_are_deterministic_and_disjoint(self):
        labels = [0] * 8 + [1] * 8 + [2] * 8 + [3] * 8

        first = stratified_indices(labels, 0.25, 22)
        second = stratified_indices(labels, 0.25, 22)

        self.assertEqual(first, second)
        train, valid = first
        self.assertFalse(set(train) & set(valid))
        self.assertEqual(set(range(len(labels))), set(train) | set(valid))
        self.assertEqual([2, 2, 2, 2], [sum(labels[index] == label for index in valid) for label in range(4)])

    def test_export_payload_has_runtime_shapes(self):
        hidden_weight = [[float(row + column) for column in range(10)] for row in range(8)]
        hidden_bias = [0.1] * 8
        output_weight = [[float(row + column) for column in range(8)] for row in range(4)]
        output_bias = [0.2] * 4

        payload = build_export_payload(
            "first_touch_test",
            [0.0] * 10,
            [1.0] * 10,
            hidden_weight,
            hidden_bias,
            output_weight,
            output_bias,
        )

        self.assertEqual("first_touch_test", payload["modelId"])
        self.assertEqual(list(FEATURE_NAMES), payload["featureNames"])
        self.assertEqual([8, 10], payload["hiddenWeightShape"])
        self.assertEqual([4, 8], payload["outputWeightShape"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(payload, json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
