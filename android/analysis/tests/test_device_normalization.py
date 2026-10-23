import unittest

import numpy as np

from android.analysis import device_normalization as normalization


def make_rows(count, device="device-a"):
    return [
        {
            "platform": "android",
            "deviceName": device,
            "sourceFile": "{:04d}.json".format(index),
            "sourceSha256": "{:04d}".format(index),
        }
        for index in range(count)
    ]


def make_values(count, gain=1.0, post_impact=1.0):
    values = np.zeros((count, 9, 10), dtype=np.float32)
    pattern = np.asarray((1.8, 2.0, 2.2), dtype=np.float32)
    values[:, :3, 2:8] = pattern[None, :, None]
    values[:, 3:, 2:5] = post_impact
    values[:, 3:, 5:8] = post_impact * 0.5
    values[:, :, 2:8] *= gain
    return values


class DeviceNormalizationTests(unittest.TestCase):
    def test_device_gain_is_removed_without_removing_impact(self):
        rows = make_rows(2)
        low = make_values(2, gain=1.0, post_impact=4.0)
        high = make_values(2, gain=4.0, post_impact=16.0)
        # 构造同形态、四倍增益的冲击帧，避免前后静默增益叠加。
        low[:, 3:, 2:5] = 4.0
        high[:, 3:, 2:5] = 16.0
        result, metadata = normalization.apply_device_profiles(
            np.concatenate((low, high), axis=0),
            rows + make_rows(2, device="device-b"),
        )

        self.assertAlmostEqual(
            float(result[0, 4, 2]),
            float(result[2, 4, 2]),
            places=5,
        )
        self.assertGreater(float(result[0, 4, 2]), 0.0)
        self.assertEqual(2, metadata["groups"]["android:device-a"]["rowCount"])

    def test_gravity_axis_rotation_is_not_removed_by_channel_gain(self):
        rows = make_rows(2)
        values = make_values(2, post_impact=4.0)
        rotated = values.copy()
        rotated[:, :, 2:5] = values[:, :, (4, 3, 2)]
        rotated[:, :, 5:8] = -values[:, :, 5:8]

        baseline, _ = normalization.apply_device_profiles(values, rows)
        rotated_result, _ = normalization.apply_device_profiles(rotated, rows)

        # 通道稳健校准只处理各轴位置和尺度，不能替代重力坐标系对齐。
        self.assertFalse(
            np.allclose(baseline[:, :, 2:8], rotated_result[:, :, 2:8])
        )

    def test_causal_profile_uses_only_prior_rows(self):
        rows = make_rows(4)
        values = make_values(4, gain=1.0)
        baseline, _ = normalization.apply_causal_device_profiles(
            values,
            rows,
            normalization.fit_profile(values),
            min_history_rows=2,
            refresh_every_rows=1,
        )
        changed = values.copy()
        changed[3, :, 2:8] *= 100.0
        altered, metadata = normalization.apply_causal_device_profiles(
            changed,
            rows,
            normalization.fit_profile(values),
            min_history_rows=2,
            refresh_every_rows=1,
        )

        self.assertTrue(np.allclose(baseline[:3], altered[:3]))
        self.assertGreater(altered[3, 4, 2], baseline[3, 4, 2])
        group = metadata["groups"]["android:device-a"]
        self.assertEqual(2, group["coldStartRowCount"])
        self.assertEqual(2, group["calibratedRowCount"])

    def test_cold_start_uses_source_profile(self):
        rows = make_rows(1)
        values = make_values(1)
        fallback = normalization.fit_profile(values)
        result, metadata = normalization.apply_causal_device_profiles(
            values,
            rows,
            fallback,
            min_history_rows=8,
            refresh_every_rows=16,
        )

        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(1, metadata["groups"]["android:device-a"][
            "coldStartRowCount"
        ])

    def test_causal_order_accepts_legacy_unknown_sequence_markers(self):
        rows = make_rows(3)
        rows[0]["cycleIndex"] = "unknown"
        rows[1]["cycleIndex"] = 2
        rows[2]["cycleIndex"] = None
        values = make_values(3)

        result, metadata = normalization.apply_causal_device_profiles(
            values,
            rows,
            normalization.fit_profile(values),
            min_history_rows=1,
            refresh_every_rows=1,
        )

        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(
            3,
            metadata["groups"]["android:device-a"]["rowCount"],
        )

    def test_quantile_profile_stays_finite_under_extreme_gain(self):
        values = make_values(12, gain=1e6, post_impact=1e6)
        result, metadata = normalization.apply_global_profile(
            values,
            values,
            quantile=True,
        )

        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(metadata["quantile"])

    def test_missing_touch_shape_is_outside_inertial_calibration(self):
        values = make_values(2)
        values[:, :, 8:10] = 0.0
        result, _ = normalization.apply_global_profile(values, values)

        self.assertTrue(np.all(result[:, :, 8:10] == 0.0))

    def test_profile_rejects_wrong_channel_shape(self):
        values = make_values(2)
        profile = normalization.fit_profile(values)
        profile["scale"] = profile["scale"][:2]

        with self.assertRaisesRegex(
                RuntimeError,
                "normalization_profile_shape_mismatch"):
            normalization.apply_profile(values, profile)


if __name__ == "__main__":
    unittest.main()
