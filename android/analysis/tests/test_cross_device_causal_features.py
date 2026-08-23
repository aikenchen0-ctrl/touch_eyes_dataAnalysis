import copy
import json
import sys
import unittest
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ANALYSIS_DIR / "tests" / "fixtures" / "cross_device_causal_feature_fixture.json"
sys.path.insert(0, str(ANALYSIS_DIR))

try:
    import cross_device_causal_features as cross_device
except ModuleNotFoundError:
    cross_device = None


class CrossDeviceCausalFeatureTests(unittest.TestCase):
    def load_fixture(self):
        return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def test_time_grid_is_invariant_to_source_sample_rate(self):
        self.assertIsNotNone(cross_device)
        fixture = self.load_fixture()
        first = cross_device.build_time_grid_matrix(
            fixture,
            fixture["records"]["android_200hz"],
            "causal_time_grid_imu11_touch9_v2",
        )
        second = cross_device.build_time_grid_matrix(
            fixture,
            fixture["records"]["android_100hz"],
            "causal_time_grid_imu11_touch9_v2",
        )
        self.assertEqual(len(first["matrix"]), 9)
        self.assertEqual(len(first["matrix"][0]), 20)
        self.assertEqual("accepted", first["gridQuality"]["status"])
        self.assertEqual(40, first["gridQuality"]["windowDurationMs"])
        for first_row, second_row in zip(first["matrix"], second["matrix"]):
            for first_value, second_value in zip(first_row, second_row):
                self.assertAlmostEqual(first_value, second_value, places=6)

    def test_future_frames_after_interpolation_support_do_not_change_output(self):
        self.assertIsNotNone(cross_device)
        fixture = self.load_fixture()
        record = fixture["records"]["android_200hz"]
        baseline = cross_device.build_time_grid_matrix(
            fixture,
            record,
            "causal_time_grid_imu11_v2",
        )
        mutated = copy.deepcopy(record)
        mutated["frames"].append({
            "sensorTimestampElapsedRealtimeNanos": 4000000000,
            "receivedElapsedRealtimeNanos": 4002000000,
            "x": 9999.0,
            "y": 9999.0,
            "z": 9999.0,
            "rotationRateX": 9999.0,
            "rotationRateY": 9999.0,
            "rotationRateZ": 9999.0,
            "gravityX": 9999.0,
            "gravityY": 9999.0,
            "gravityZ": 9999.0,
        })
        actual = cross_device.build_time_grid_matrix(
            fixture,
            mutated,
            "causal_time_grid_imu11_v2",
        )
        self.assertEqual(baseline["matrix"], actual["matrix"])

    def test_missing_platform_metadata_is_rejected(self):
        self.assertIsNotNone(cross_device)
        fixture = self.load_fixture()
        record = copy.deepcopy(fixture["records"]["android_200hz"])
        del fixture["sensorMetadata"]
        with self.assertRaisesRegex(RuntimeError, "missing_sensor_metadata"):
            cross_device.build_time_grid_matrix(
                fixture,
                record,
                "causal_time_grid_imu11_v2",
            )

    def test_missing_coordinate_transform_version_is_rejected(self):
        self.assertIsNotNone(cross_device)
        fixture = self.load_fixture()
        record = fixture["records"]["android_200hz"]
        del fixture["sensorMetadata"]["transformVersion"]
        with self.assertRaisesRegex(RuntimeError, "missing_transform_version"):
            cross_device.build_time_grid_matrix(
                fixture,
                record,
                "causal_time_grid_imu11_v2",
            )

    def test_missing_touch_transform_version_is_rejected(self):
        self.assertIsNotNone(cross_device)
        fixture = self.load_fixture()
        record = fixture["records"]["android_200hz"]
        del fixture["touchMetadata"]["touchTransformVersion"]
        with self.assertRaisesRegex(RuntimeError, "missing_touch_transform_version"):
            cross_device.build_time_grid_matrix(
                fixture,
                record,
                "causal_time_grid_imu11_v2",
            )

    def test_axis_transform_normalizes_swapped_and_signed_platform_axes(self):
        self.assertIsNotNone(cross_device)
        fixture = self.load_fixture()
        baseline = cross_device.build_time_grid_matrix(
            fixture,
            fixture["records"]["android_200hz"],
            "causal_time_grid_imu11_v2",
        )
        transformed = copy.deepcopy(fixture)
        transformed["sensorMetadata"]["sensorAxisFrame"] = "ios_device"
        transformed["sensorMetadata"]["axisTransformMatrix"] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        record = transformed["records"]["android_200hz"]
        for frame in record["frames"]:
            for first, second in (("x", "y"), ("rotationRateX", "rotationRateY"), ("gravityX", "gravityY")):
                original_first = frame[first]
                original_second = frame[second]
                frame[first] = original_second
                frame[second] = -original_first
        actual = cross_device.build_time_grid_matrix(
            transformed,
            record,
            "causal_time_grid_imu11_v2",
        )

        self.assertEqual(baseline["matrix"], actual["matrix"])

    def test_missing_radius_is_reported_by_mask_not_as_valid_geometry(self):
        self.assertIsNotNone(cross_device)
        fixture = self.load_fixture()
        record = copy.deepcopy(fixture["records"]["android_200hz"])
        record["touchFrames"][0]["axisAvailability"]["hasTouchMajorRange"] = False
        record["touchFrames"][0]["axisAvailability"]["hasTouchMinorRange"] = False
        actual = cross_device.build_time_grid_matrix(
            fixture,
            record,
            "causal_time_grid_imu11_touch9_v2",
        )

        self.assertEqual(0, actual["missingMasks"][1]["major"])
        self.assertEqual(0, actual["missingMasks"][1]["minor"])
        self.assertEqual(0.0, actual["matrix"][1][13])
        self.assertEqual(0.0, actual["matrix"][1][14])

    def test_pointer_count_missing_is_reported_by_mask(self):
        self.assertIsNotNone(cross_device)
        fixture = self.load_fixture()
        record = copy.deepcopy(fixture["records"]["android_200hz"])
        record["touchFrames"][0].pop("pointerCount")
        actual = cross_device.build_time_grid_matrix(
            fixture,
            record,
            "causal_time_grid_imu11_touch9_v2",
        )

        self.assertEqual(0, actual["missingMasks"][1]["pointerCount"])
        self.assertEqual(0.0, actual["matrix"][1][18])

    def test_touch_coordinate_transform_maps_landscape_to_canonical_view(self):
        self.assertIsNotNone(cross_device)
        fixture = self.load_fixture()
        transformed = copy.deepcopy(fixture)
        transformed["touchMetadata"]["touchCoordinateTransformMatrix"] = [[0, 1, 0], [-1, 0, 1]]
        record = transformed["records"]["android_200hz"]
        record["touchFrames"][0]["x_norm"] = 0.3
        record["touchFrames"][0]["y_norm"] = 0.2

        result = cross_device.build_time_grid_matrix(
            transformed,
            record,
            "causal_time_grid_imu11_v2",
        )

        self.assertAlmostEqual(0.2, result["matrix"][1][0], places=6)
        self.assertAlmostEqual(0.7, result["matrix"][1][1], places=6)

    def test_invalid_touch_clock_is_rejected_instead_of_treated_as_touch_down(self):
        self.assertIsNotNone(cross_device)
        fixture = self.load_fixture()
        record = copy.deepcopy(fixture["records"]["android_200hz"])
        record["touchFrames"][0]["eventElapsedRealtimeNanos"] = 0
        with self.assertRaisesRegex(RuntimeError, "missing_touch_clock"):
            cross_device.build_time_grid_matrix(
                fixture,
                record,
                "causal_time_grid_imu11_v2",
            )

    def test_non_orthogonal_axis_transform_is_rejected(self):
        self.assertIsNotNone(cross_device)
        fixture = self.load_fixture()
        fixture["sensorMetadata"]["sensorAxisFrame"] = "ios_device"
        fixture["sensorMetadata"]["axisTransformMatrix"] = [[1, 0, 0], [1, 0, 0], [0, 0, 1]]
        with self.assertRaisesRegex(RuntimeError, "invalid_axis_transform_matrix"):
            cross_device.build_time_grid_matrix(
                fixture,
                fixture["records"]["android_200hz"],
                "causal_time_grid_imu11_v2",
            )

    def test_touch_coordinates_can_be_derived_from_raw_units(self):
        self.assertIsNotNone(cross_device)
        fixture = self.load_fixture()
        record = copy.deepcopy(fixture["records"]["android_200hz"])
        touch = record["touchFrames"][0]
        touch.pop("x_norm")
        touch.pop("y_norm")
        touch["x"] = 200.0
        touch["y"] = 600.0
        result = cross_device.build_time_grid_matrix(
            fixture,
            record,
            "causal_time_grid_imu11_v2",
        )
        self.assertAlmostEqual(0.2, result["matrix"][1][0], places=6)
        self.assertAlmostEqual(0.3, result["matrix"][1][1], places=6)

    def test_lift_before_positive_grid_endpoint_is_rejected(self):
        self.assertIsNotNone(cross_device)
        fixture = self.load_fixture()
        fixture["timeAlignment"]["liftEventElapsedRealtimeNanos"] = 1_020_000_000
        with self.assertRaisesRegex(RuntimeError, "lift_before_time_grid_endpoint"):
            cross_device.build_time_grid_matrix(
                fixture,
                fixture["records"]["android_200hz"],
                "causal_time_grid_imu11_v2",
            )

    def test_accelerometer_fallback_requires_gravity_availability_metadata(self):
        self.assertIsNotNone(cross_device)
        fixture = self.load_fixture()
        fixture["sensorMetadata"]["sensorSource"] = "accelerometer_with_gravity"
        fixture["sensorMetadata"]["gravityAvailable"] = False
        with self.assertRaisesRegex(RuntimeError, "gravity_metadata_required"):
            cross_device.build_time_grid_matrix(
                fixture,
                fixture["records"]["android_200hz"],
                "causal_time_grid_imu11_v2",
            )

    def test_sensor_source_matching_is_case_insensitive(self):
        self.assertIsNotNone(cross_device)
        fixture = self.load_fixture()
        fixture["sensorMetadata"]["sensorSource"] = "ACCELEROMETER_WITH_GRAVITY"
        fixture["sensorMetadata"]["gravityAvailable"] = True
        frames = cross_device.canonical_imu_frames(
            fixture["records"]["android_200hz"]["frames"],
            fixture["sensorMetadata"],
        )
        self.assertAlmostEqual(-2.0, frames[0]["acceleration"][0], places=6)
        self.assertAlmostEqual(-9.8, frames[0]["acceleration"][2], places=6)

    def test_accelerometer_minus_gravity_is_not_subtracted_twice(self):
        fixture = self.load_fixture()
        fixture["sensorMetadata"]["sensorSource"] = "accelerometer_minus_gravity"
        fixture["sensorMetadata"]["gravityAvailable"] = True
        frames = cross_device.canonical_imu_frames(
            fixture["records"]["android_200hz"]["frames"],
            fixture["sensorMetadata"],
        )
        self.assertAlmostEqual(-2.0, frames[0]["acceleration"][0], places=6)
        self.assertAlmostEqual(0.0, frames[0]["acceleration"][2], places=6)

    def test_raw_accelerometer_without_gravity_is_rejected_for_training(self):
        fixture = self.load_fixture()
        fixture["sensorMetadata"]["sensorSource"] = "accelerometer_raw_no_gravity"
        fixture["sensorMetadata"]["accelerationTrainingInput"] = False
        with self.assertRaisesRegex(RuntimeError, "raw_acceleration_training_input_forbidden"):
            cross_device.canonical_imu_frames(
                fixture["records"]["android_200hz"]["frames"],
                fixture["sensorMetadata"],
            )

    def test_median_gap_uses_same_upper_middle_rule_as_android(self):
        self.assertEqual(20, cross_device.upper_middle_median([10, 20]))
        self.assertEqual(20, cross_device.upper_middle_median([20, 10, 20]))


if __name__ == "__main__":
    unittest.main()
