import tempfile
import unittest
from pathlib import Path

import torch

from android.analysis import train_time_grid_strict_cnn as trainer


class StrictTimeGridTrainerTests(unittest.TestCase):
    def test_touch23_selection_removes_orientation_channels(self):
        matrix = [[float(index) for index in range(26)] for _ in range(9)]

        selected = trainer.select_touch23_matrix(matrix)

        self.assertEqual(9, len(selected))
        self.assertEqual(23, len(selected[0]))
        self.assertNotIn(16.0, selected[0])
        self.assertNotIn(17.0, selected[0])
        self.assertNotIn(24.0, selected[0])
        self.assertEqual([18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 25.0], selected[0][16:])

    def test_masked_scaler_ignores_missing_values_and_keeps_them_zero(self):
        first = [[0.0] * 23 for _ in range(9)]
        second = [[0.0] * 23 for _ in range(9)]
        first[0][11] = 2.0
        first[0][18] = 1.0
        second[0][11] = 6.0
        second[0][18] = 1.0
        second[1][11] = 9999.0
        second[1][18] = 0.0
        samples = [{"matrix": first}, {"matrix": second}]

        scaler = trainer.fit_masked_scaler(samples, trainer.TOUCH23_FEATURE_NAMES)
        transformed = trainer.transform_matrix(second, scaler)

        self.assertEqual(4.0, scaler["channels"][11]["center"])
        self.assertEqual(4.0, scaler["channels"][11]["scale"])
        self.assertEqual(0.0, transformed[1][11])
        self.assertEqual(0.0, transformed[1][18])

    def test_validation_values_cannot_change_training_scaler(self):
        train = [{"matrix": [[float(index)] * 11 for index in range(9)]}]
        valid = [{"matrix": [[999999.0] * 11 for _ in range(9)]}]

        before = trainer.fit_masked_scaler(train, trainer.IMU11_FEATURE_NAMES)
        valid[0]["matrix"][0][2] = -999999.0
        after = trainer.fit_masked_scaler(train, trainer.IMU11_FEATURE_NAMES)

        self.assertEqual(before, after)

    def test_session_and_class_weights_are_bounded_and_normalized(self):
        samples = []
        for label in trainer.CLASS_LABELS:
            for index in range(4):
                samples.append({
                    "classLabel": label,
                    "groupId": label + ("-large" if index < 3 else "-small"),
                    "trainingWeight": 1.0,
                })

        weights = trainer.compute_sample_weights(samples)

        self.assertAlmostEqual(1.0, sum(weights) / len(weights), places=6)
        self.assertTrue(all(0.25 <= weight <= 4.0 for weight in weights))
        for label in trainer.CLASS_LABELS:
            indices = [index for index, sample in enumerate(samples) if sample["classLabel"] == label]
            self.assertAlmostEqual(4.0, sum(weights[index] for index in indices), places=6)

    def test_model_has_four_class_output(self):
        model = trainer.StrictTimeGridCnn(torch, 23)

        output = model(torch.zeros(4, 9, 23))

        self.assertEqual([4, 4], list(output.shape))
        self.assertEqual(5604, sum(parameter.numel() for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
