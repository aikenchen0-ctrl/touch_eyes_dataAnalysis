import unittest

import numpy as np

from android.analysis import gate_evidence_features as evidence
from android.analysis import run_lightweight_time_grid_models as lightweight


def sample():
    matrix = np.zeros(
        (9, len(lightweight.FEATURE_NAMES)),
        dtype=np.float32,
    )
    matrix[:, 8] = 1.0
    matrix[:, 12] = 1.5
    matrix[:, 15] = 0.2
    matrix[:, 19] = 1.0
    matrix[:, 22] = 1.0
    matrix[:, 23] = 1.0
    matrix[:, 24] = 1.0
    raw = np.zeros(
        (9, len(lightweight.PHYSICAL_NAMES)),
        dtype=np.float32,
    )
    raw[:, 0] = 0.2
    raw[:, 1] = 0.3
    raw[:, 2:5] = 0.25
    raw[:, 5:8] = 0.1
    raw[:, 8] = 1.5
    raw[:, 9] = 0.2
    return raw, {"matrix": matrix}


class GateEvidenceFeatureTests(unittest.TestCase):
    def test_size_and_axis_ratio_modes_keep_distinct_contracts(self):
        values = np.ones((2, len(evidence.FEATURE_NAMES)), dtype=np.float32)

        size_values = evidence.apply_feature_mode(values, "size_only")
        ratio_values = evidence.apply_feature_mode(
            values,
            "axis_ratio_only",
        )
        size_retained = {
            evidence.FEATURE_NAMES[index]
            for index in np.flatnonzero(size_values[0] > 0.0)
        }
        ratio_retained = {
            evidence.FEATURE_NAMES[index]
            for index in np.flatnonzero(ratio_values[0] > 0.0)
        }

        self.assertTrue(size_retained)
        self.assertTrue(ratio_retained)
        self.assertTrue(all(
            name in (
                "shape.post_size_mean",
                "shape.size_ratio",
                "shape.availability",
            )
            for name in size_retained
        ))
        self.assertTrue(all(
            name in (
                "shape.post_axis_ratio_mean",
                "shape.axis_ratio_delta",
                "shape.availability",
            )
            for name in ratio_retained
        ))
        self.assertEqual({"shape.availability"}, size_retained & ratio_retained)
    def test_extracts_finite_frozen_feature_contract(self):
        raw, row = sample()

        values = evidence.extract_one(raw, row)

        self.assertEqual(65, len(evidence.FEATURE_NAMES))
        self.assertEqual((65,), values.shape)
        self.assertTrue(np.isfinite(values).all())

    def test_post_touch_frames_change_post_evidence_only(self):
        raw, row = sample()
        baseline = evidence.extract_one(raw, row)
        changed = raw.copy()
        changed[7:, 2:5] = 2.0
        result = evidence.extract_one(changed, row)

        post_index = evidence.FEATURE_NAMES.index("acceleration.post_max")
        pre_index = evidence.FEATURE_NAMES.index("acceleration.pre_max")
        self.assertGreater(result[post_index], baseline[post_index])
        self.assertEqual(baseline[pre_index], result[pre_index])

    def test_contract_selects_the_declared_impact_window(self):
        raw, row = sample()
        baseline_legacy = evidence.extract_one(raw, row, "legacy_post20")
        baseline_touchdown = evidence.extract_one(
            raw,
            row,
            "touchdown_post0",
        )
        changed = raw.copy()
        changed[3, 2:5] = 9.0
        legacy = evidence.extract_one(changed, row, "legacy_post20")
        touchdown = evidence.extract_one(
            changed,
            row,
            "touchdown_post0",
        )
        post_index = evidence.FEATURE_NAMES.index("acceleration.post_max")

        self.assertEqual(
            baseline_legacy[post_index],
            legacy[post_index],
        )
        self.assertGreater(
            touchdown[post_index],
            baseline_touchdown[post_index],
        )

    def test_legacy_contract_ignores_true_touchdown_frame(self):
        raw, row = sample()
        baseline = evidence.extract_one(raw, row, "legacy_post20")
        changed = raw.copy()
        changed[3, 2:5] = 9.0
        result = evidence.extract_one(changed, row, "legacy_post20")
        post_index = evidence.FEATURE_NAMES.index("acceleration.post_max")

        self.assertEqual(baseline[post_index], result[post_index])

    def test_touch_displacement_uses_the_contract_anchor(self):
        raw, row = sample()
        raw[:, 0] = np.arange(9, dtype=np.float32)
        raw[:, 1] = np.arange(9, dtype=np.float32) * 2.0
        x_index = evidence.FEATURE_NAMES.index("touch.x_delta")
        y_index = evidence.FEATURE_NAMES.index("touch.y_delta")

        legacy = evidence.extract_one(raw, row, "legacy_post20")
        touchdown = evidence.extract_one(
            raw,
            row,
            "touchdown_post0",
        )

        self.assertEqual(1.0, legacy[x_index])
        self.assertEqual(2.0, legacy[y_index])
        self.assertEqual(5.0, touchdown[x_index])
        self.assertEqual(10.0, touchdown[y_index])

    def test_new_contracts_use_only_true_pretouch_gravity(self):
        raw, row = sample()
        row["matrix"][:, 8:11] = (0.0, 0.0, 9.8)
        baseline = evidence.extract_one(raw, row, "touchdown_post0")
        changed = row["matrix"].copy()
        changed[3, 8:11] = (9.8, 0.0, 0.0)
        row["matrix"] = changed
        result = evidence.extract_one(raw, row, "touchdown_post0")
        mean_index = evidence.FEATURE_NAMES.index("gravity.mean_delta")

        self.assertGreater(result[mean_index], baseline[mean_index])

    def test_non_legacy_contracts_keep_the_true_pretouch_window(self):
        raw, row = sample()
        baseline = evidence.extract_one(raw, row, "impact_post5")
        changed = raw.copy()
        changed[3, 2:5] = 9.0
        result = evidence.extract_one(changed, row, "impact_post5")
        pre_index = evidence.FEATURE_NAMES.index("acceleration.pre_max")

        self.assertEqual(baseline[pre_index], result[pre_index])

    def test_touch_displacement_uses_the_contract_anchor(self):
        raw, row = sample()
        raw[:, 0] = np.arange(9, dtype=np.float32)
        raw[:, 1] = np.arange(9, dtype=np.float32) * 2.0
        x_index = evidence.FEATURE_NAMES.index("touch.x_delta")
        y_index = evidence.FEATURE_NAMES.index("touch.y_delta")

        legacy = evidence.extract_one(raw, row, "legacy_post20")
        touchdown = evidence.extract_one(raw, row, "touchdown_post0")

        self.assertEqual(1.0, legacy[x_index])
        self.assertEqual(2.0, legacy[y_index])
        self.assertEqual(5.0, touchdown[x_index])
        self.assertEqual(10.0, touchdown[y_index])

    def test_new_contracts_use_true_pretouch_gravity_reference(self):
        raw, row = sample()
        row["matrix"][:, 8:11] = (0.0, 0.0, 9.8)
        baseline = evidence.extract_one(raw, row, "impact_post5")
        changed = row["matrix"].copy()
        changed[3:5, 8:11] = (9.8, 0.0, 0.0)
        row["matrix"] = changed
        result = evidence.extract_one(raw, row, "impact_post5")
        mean_index = evidence.FEATURE_NAMES.index("gravity.mean_delta")

        self.assertEqual(0.0, baseline[mean_index])
        self.assertGreater(result[mean_index], 0.0)

    def test_missing_shape_is_masked_and_marked_unavailable(self):
        raw, row = sample()
        row["matrix"][:, 12] = 0.0
        row["matrix"][:, 15] = 0.0
        row["matrix"][:, 22:25] = 0.0

        values = evidence.extract_one(raw, row)
        availability = evidence.FEATURE_NAMES.index("shape.availability")
        size_mean = evidence.FEATURE_NAMES.index("shape.post_size_mean")
        ratio_mean = evidence.FEATURE_NAMES.index(
            "shape.post_axis_ratio_mean"
        )

        self.assertEqual(0.0, values[availability])
        self.assertEqual(0.0, values[size_mean])
        self.assertEqual(0.0, values[ratio_mean])

    def test_feature_modes_only_zero_excluded_families(self):
        raw, row = sample()
        values = evidence.extract(raw[None, ...], [row])
        no_shape = evidence.apply_feature_mode(values, "no_shape")
        motion_only = evidence.apply_feature_mode(values, "motion_only")
        shape_only = evidence.apply_feature_mode(values, "shape_only")
        magnitude_only = evidence.apply_feature_mode(
            values,
            "magnitude_only",
        )

        shape_mask = np.asarray([
            name.startswith("shape.") for name in evidence.FEATURE_NAMES
        ])
        touch_mask = np.asarray([
            name.startswith("touch.") for name in evidence.FEATURE_NAMES
        ])
        self.assertTrue(np.all(no_shape[:, shape_mask] == 0.0))
        self.assertTrue(
            np.all(motion_only[:, shape_mask | touch_mask] == 0.0)
        )
        self.assertTrue(np.all(shape_only[:, ~shape_mask] == 0.0))

    def test_magnitude_only_keeps_signal_magnitudes_without_axes(self):
        raw, row = sample()
        raw[:, 2:5] = (3.0, 4.0, 12.0)
        raw[:, 5:8] = (0.0, 5.0, 12.0)
        values = evidence.extract(raw[None, ...], [row])

        masked = evidence.apply_feature_mode(values, "magnitude_only")
        retained = np.asarray([
            (
                name.startswith("acceleration.")
                or name.startswith("rotation.")
            )
            and not any(
                name.startswith("acceleration." + axis + ".")
                or name.startswith("rotation." + axis + ".")
                for axis in ("x", "y", "z")
            )
            for name in evidence.FEATURE_NAMES
        ])

        self.assertTrue(np.all(masked[:, ~retained] == 0.0))
        self.assertTrue(np.array_equal(
            masked[:, retained],
            values[:, retained],
        ))
        self.assertEqual(
            0.0,
            masked[
                0,
                evidence.FEATURE_NAMES.index("acceleration.x.pre_mean"),
            ],
        )

    def test_normalization_uses_fit_statistics_and_clips_extremes(self):
        fit = np.asarray([[0.0], [10.0]], dtype=np.float32)
        stable = evidence.normalize(
            fit,
            np.asarray([[10.0]], dtype=np.float32),
        )
        extreme = evidence.normalize(
            fit,
            np.asarray([[1000000.0]], dtype=np.float32),
        )

        self.assertGreater(extreme[0, 0], stable[0, 0])
        self.assertEqual(8.0, extreme[0, 0])


if __name__ == "__main__":
    unittest.main()
