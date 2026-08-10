import unittest

from android.analysis import train_time_grid_all_data_ablation as trainer


class AllDataAblationTrainerTests(unittest.TestCase):
    def test_contracts_select_expected_feature_groups(self):
        expected = {
            "imu11": 11,
            "touch23": 23,
            "touch16_deduplicated": 16,
            "touch15_geometry": 15,
            "imu14_geometry": 14,
            "imu12_touch_valid": 12,
            "imu8_no_gravity": 8,
            "imu6_motion": 6,
            "imu9_no_position": 9,
        }

        self.assertEqual(expected, {
            name: len(spec["sourceIndices"])
            for name, spec in trainer.CHANNEL_CONTRACTS.items()
        })
        self.assertEqual(
            len(trainer.CHANNEL_CONTRACTS),
            len({tuple(spec["sourceIndices"]) for spec in trainer.CHANNEL_CONTRACTS.values()}),
        )

    def test_all_data_config_disables_validation_driven_early_stopping(self):
        config = trainer.default_all_data_config()

        self.assertEqual(80, config["epochs"])
        self.assertIsNone(config["earlyStoppingPatience"])
        self.assertEqual("all_data_fit_only", config["evaluationScope"])

    def test_geometry_scaler_uses_touch_availability_without_model_mask_channel(self):
        samples = [{
            "matrix": [[0.0, 0.03] for _ in range(9)],
            "touchAvailable": [0.0] + [1.0] * 8,
        }]
        feature_names = ("size", "major_norm")

        scaler = trainer.fit_all_data_scaler(samples, feature_names)
        transformed = trainer.transform_all_data_matrix(samples[0], scaler)

        self.assertEqual(8, scaler["channels"][0]["fitCount"])
        self.assertEqual(0.0, transformed[0][0])
        self.assertEqual(0.0, transformed[0][1])


if __name__ == "__main__":
    unittest.main()
