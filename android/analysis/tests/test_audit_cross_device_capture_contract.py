import unittest
import sys
import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_cross_device_capture_contract import audit_record


def valid_record():
    return {
        "schemaVersion": 6,
        "sensorMetadata": {
            "sensorSource": "linear_acceleration",
            "accelerationStatus": "HEALTHY",
            "gyroscopeStatus": "HEALTHY",
            "gravityStatus": "HEALTHY",
            "rotationStatus": "HEALTHY",
            "rotationSensorSource": "game_rotation_vector",
            "rotationSensorName": "rotation",
            "gravityAvailable": True,
            "accelerationTrainingInput": True,
            "rotationTrainingInput": False,
            "streamEpoch": 1,
            "timestampRegressionCount": 0,
            "rotationFrames": [{
                "trainingInput": False,
                "sourceType": "game_rotation_vector",
                "sensorTimestampElapsedRealtimeNanos": 1,
                "receivedElapsedRealtimeNanos": 2,
                "streamEpoch": 1,
            }],
        },
        "deviceCapability": {
            "level": "A",
            "touchAxesSemanticallyValid": True,
            "touchAxesDistinct": True,
            "toolAxesIndependent": True,
            "observedFrames": 1,
        },
        "touchMetadata": {
            "touchUnit": "pixels",
            "touchCoordinateFrame": "android_view",
            "touchTransformVersion": "cross_device_v2",
            "apiLevel": 36,
            "displayWidthPx": 1240,
            "displayHeightPx": 2772,
            "density": 3.0,
            "xdpi": 400.0,
            "ydpi": 400.0,
            "metadataEpoch": 1,
            "windowInsets": {
                "available": False,
                "leftPx": 0,
                "topPx": 0,
                "rightPx": 0,
                "bottomPx": 0,
            },
            "inputDevice": {"motionRanges": []},
        },
        "frames": [{
            "sensorTimestampElapsedRealtimeNanos": 1,
            "receivedElapsedRealtimeNanos": 2,
            "sensorSource": "linear_acceleration",
            "streamEpoch": 1,
            "trainingInput": True,
        }],
        "touchFrames": [
            {
                "metadataEpoch": 1,
                "eventElapsedRealtimeNanos": 1,
                "receivedElapsedRealtimeNanos": 2,
                "pointerId": 0,
                "actionMasked": 0,
                "actionIndex": 0,
                "touchMajor": 40.0,
                "touchMinor": 20.0,
                "toolMajor": 60.0,
                "toolMinor": 30.0,
                "pressure": 0.5,
                "size": 0.4,
                "axisAvailability": {
                    "hasTouchMajorRange": True,
                    "hasTouchMinorRange": True,
                    "hasToolMajorRange": True,
                    "hasToolMinorRange": True,
                    "hasPressureRange": True,
                    "hasSizeRange": True,
                    "hasOrientationRange": False,
                    "touchMajorObserved": True,
                    "touchMinorObserved": True,
                    "toolMajorObserved": True,
                    "toolMinorObserved": True,
                    "pressureObserved": True,
                    "sizeObserved": True,
                    "orientationObserved": False,
                    "touchMajorSynthesizedOrUnknown": False,
                    "touchMinorSynthesizedOrUnknown": False,
                    "toolMajorSynthesizedOrUnknown": False,
                    "toolMinorSynthesizedOrUnknown": False,
                    "pressureSynthesizedOrUnknown": False,
                    "sizeSynthesizedOrUnknown": False,
                    "orientationSynthesizedOrUnknown": False,
                    "touchMajorSource": "android_motion_event_declared_range_unverified",
                    "touchMinorSource": "framework_value_undeclared_range",
                    "toolMajorSource": "android_motion_event_declared_range_unverified",
                    "toolMinorSource": "android_motion_event_declared_range_unverified",
                    "pressureSource": "android_motion_event_declared_range_unverified",
                    "sizeSource": "android_motion_event_declared_range_unverified",
                    "orientationSource": "unavailable",
                },
            }
        ],
    }


class CrossDeviceCaptureContractTest(unittest.TestCase):
    def test_valid_record_passes_and_rotation_stays_observation_only(self):
        result = audit_record(valid_record())
        self.assertTrue(result["ok"])
        self.assertEqual(result["rotationFrameCount"], 1)

    def test_tool_axis_cannot_override_touch_axis(self):
        record = valid_record()
        record["touchFrames"][0]["axisAvailability"]["touchMajorSource"] = "tool_major_fallback"
        result = audit_record(record)
        self.assertFalse(result["ok"])
        self.assertTrue(any(issue["code"] == "tool_axis_overrides_touch" for issue in result["issues"]))

    def test_rotation_training_input_is_rejected(self):
        record = valid_record()
        record["sensorMetadata"]["rotationTrainingInput"] = True
        result = audit_record(record)
        self.assertFalse(result["ok"])
        self.assertTrue(any(issue["code"] == "rotation_training_leak" for issue in result["issues"]))

    def test_raw_acceleration_is_observation_only(self):
        record = valid_record()
        record["sensorMetadata"]["sensorSource"] = "accelerometer_raw_no_gravity"
        record["sensorMetadata"]["accelerationTrainingInput"] = True
        result = audit_record(record)
        self.assertFalse(result["ok"])
        self.assertTrue(any(issue["code"] == "raw_acceleration_training_leak" for issue in result["issues"]))

    def test_raw_acceleration_can_be_saved_when_training_is_disabled(self):
        record = valid_record()
        record["sensorMetadata"]["sensorSource"] = "accelerometer_raw_no_gravity"
        record["sensorMetadata"]["accelerationStatus"] = "DEGRADED"
        record["sensorMetadata"]["accelerationTrainingInput"] = False
        record["frames"][0]["sensorSource"] = "accelerometer_raw_no_gravity"
        record["frames"][0]["trainingInput"] = False
        result = audit_record(record)
        self.assertTrue(result["ok"])

    def test_imu_frame_requires_training_input_flag(self):
        record = valid_record()
        del record["frames"][0]["trainingInput"]
        result = audit_record(record)
        self.assertFalse(result["ok"])
        self.assertTrue(any(issue["code"] == "imu_training_flag_missing" for issue in result["issues"]))

    def test_rotation_source_must_match_frames(self):
        record = valid_record()
        record["sensorMetadata"]["rotationFrames"][0]["sourceType"] = "rotation_vector"
        result = audit_record(record)
        self.assertFalse(result["ok"])
        self.assertTrue(any(issue["code"] == "rotation_source_frame_mismatch" for issue in result["issues"]))

    def test_rotation_fallback_can_cross_epoch_without_losing_old_frames(self):
        record = valid_record()
        record["sensorMetadata"]["rotationSensorSource"] = "rotation_vector"
        record["sensorMetadata"]["streamEpoch"] = 2
        record["sensorMetadata"]["rotationFrames"].append({
            "trainingInput": False,
            "sourceType": "rotation_vector",
            "sensorTimestampElapsedRealtimeNanos": 3,
            "receivedElapsedRealtimeNanos": 4,
            "streamEpoch": 2,
        })
        record["frames"].append({
            "sensorTimestampElapsedRealtimeNanos": 3,
            "receivedElapsedRealtimeNanos": 4,
            "sensorSource": "linear_acceleration",
            "streamEpoch": 2,
            "trainingInput": True,
        })
        result = audit_record(record)
        self.assertTrue(result["ok"])

    def test_touch_event_cannot_be_after_receive_or_regress(self):
        record = valid_record()
        record["touchFrames"][0]["eventElapsedRealtimeNanos"] = 3
        result = audit_record(record)
        self.assertFalse(result["ok"])
        self.assertTrue(any(issue["code"] == "touch_event_after_receive" for issue in result["issues"]))

    def test_axis_contract_requires_observed_and_declared_flags(self):
        record = valid_record()
        del record["touchFrames"][0]["axisAvailability"]["touchMajorObserved"]
        result = audit_record(record)
        self.assertFalse(result["ok"])
        self.assertTrue(any(issue["code"] == "axis_contract_field_missing" for issue in result["issues"]))

    def test_imu_frame_requires_source_and_stream_epoch(self):
        record = valid_record()
        del record["frames"][0]["sensorSource"]
        del record["frames"][0]["streamEpoch"]
        result = audit_record(record)
        self.assertFalse(result["ok"])
        self.assertTrue(any(issue["code"] == "imu_frame_source_unknown" for issue in result["issues"]))
        self.assertTrue(any(issue["code"] == "imu_stream_epoch_invalid" for issue in result["issues"]))

    def test_rotation_frame_cannot_extend_outside_imu_window(self):
        record = valid_record()
        record["sensorMetadata"]["rotationFrames"][0]["sensorTimestampElapsedRealtimeNanos"] = 3
        record["sensorMetadata"]["rotationFrames"][0]["receivedElapsedRealtimeNanos"] = 4
        result = audit_record(record)
        self.assertFalse(result["ok"])
        self.assertTrue(any(issue["code"] == "rotation_frame_outside_imu_window" for issue in result["issues"]))

    def test_inverted_touch_axes_are_reported_as_degraded_evidence(self):
        record = valid_record()
        record["touchFrames"][0]["touchMajor"] = 20.0
        record["touchFrames"][0]["touchMinor"] = 40.0
        record["deviceCapability"]["touchAxesSemanticallyValid"] = False
        record["deviceCapability"]["level"] = "C"
        result = audit_record(record)
        self.assertTrue(result["ok"])
        issue = next(issue for issue in result["issues"] if issue["code"] == "touch_axis_order_inverted")
        self.assertEqual("warning", issue["severity"])

    def test_tool_axis_alias_is_reported_and_must_match_capability(self):
        record = valid_record()
        record["touchFrames"][0]["toolMajor"] = 40.0
        record["touchFrames"][0]["toolMinor"] = 20.0
        record["deviceCapability"]["toolAxesIndependent"] = False
        result = audit_record(record)
        self.assertTrue(result["ok"])
        self.assertTrue(any(issue["code"] == "tool_axis_alias_detected" for issue in result["issues"]))

    def test_capability_quality_flags_must_match_observed_axes(self):
        record = valid_record()
        record["touchFrames"][0]["touchMajor"] = 20.0
        record["touchFrames"][0]["touchMinor"] = 40.0
        result = audit_record(record)
        self.assertFalse(result["ok"])
        self.assertTrue(any(issue["code"] == "touch_axis_capability_mismatch" for issue in result["issues"]))

    def test_capability_frame_count_cannot_include_previous_touch(self):
        record = valid_record()
        record["deviceCapability"]["observedFrames"] = 3

        result = audit_record(record)

        self.assertFalse(result["ok"])
        self.assertTrue(any(
            issue["code"] == "capability_frame_count_mismatch"
            for issue in result["issues"]
        ))

    def test_touch_frame_requires_positive_metadata_epoch(self):
        record = valid_record()
        record["touchFrames"][0]["metadataEpoch"] = 0

        result = audit_record(record)

        self.assertFalse(result["ok"])
        self.assertTrue(any(
            issue["code"] == "touch_metadata_epoch_invalid"
            for issue in result["issues"]
        ))

    def test_touch_frame_metadata_epoch_cannot_regress(self):
        record = valid_record()
        first = record["touchFrames"][0]
        first["metadataEpoch"] = 2
        record["touchMetadata"]["metadataEpoch"] = 2
        second = copy.deepcopy(first)
        second["eventElapsedRealtimeNanos"] = 3
        second["receivedElapsedRealtimeNanos"] = 4
        second["metadataEpoch"] = 1
        record["touchFrames"].append(second)
        record["deviceCapability"]["observedFrames"] = 2

        result = audit_record(record)

        self.assertFalse(result["ok"])
        self.assertTrue(any(
            issue["code"] == "touch_metadata_epoch_regression"
            for issue in result["issues"]
        ))

    def test_record_metadata_epoch_matches_first_touch_frame(self):
        record = valid_record()
        record["touchMetadata"]["metadataEpoch"] = 2

        result = audit_record(record)

        self.assertFalse(result["ok"])
        self.assertTrue(any(
            issue["code"] == "touch_metadata_epoch_root_mismatch"
            for issue in result["issues"]
        ))

    def test_equal_touch_axes_are_reported_as_radius_only(self):
        record = valid_record()
        record["touchFrames"][0]["touchMinor"] = record["touchFrames"][0]["touchMajor"]
        record["deviceCapability"]["touchAxesDistinct"] = False
        record["deviceCapability"]["level"] = "B"
        result = audit_record(record)
        self.assertTrue(result["ok"])
        issue = next(issue for issue in result["issues"] if issue["code"] == "touch_axes_not_distinct")
        self.assertEqual("warning", issue["severity"])


if __name__ == "__main__":
    unittest.main()
