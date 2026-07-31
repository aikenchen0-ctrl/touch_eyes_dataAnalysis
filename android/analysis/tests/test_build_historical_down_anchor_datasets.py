import unittest

from android.analysis.build_historical_down_anchor_datasets import analyze_sample
from android.analysis.build_historical_down_anchor_datasets import is_observed_frame


def observed_frame(time_since_down, frame_index, gravity_z=9.8):
    return {
        "frameIndex": frame_index,
        "relativeTimestamp": time_since_down,
        "timeSinceTouchDown": time_since_down,
        "delta": 0.1,
        "x": 0.01,
        "y": 0.02,
        "z": 0.03,
        "rotationRateX": 0.04,
        "rotationRateY": 0.05,
        "rotationRateZ": 0.06,
        "gravityX": 0.0,
        "gravityY": 0.0,
        "gravityZ": gravity_z,
        "x_norm": 0.5,
        "y_norm": 0.5,
    }


def placeholder_frame(time_since_down, frame_index):
    return {
        "frameIndex": frame_index,
        "relativeTimestamp": time_since_down,
        "timeSinceTouchDown": time_since_down,
        "delta": 0.0,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "rotationRateX": 0.0,
        "rotationRateY": 0.0,
        "rotationRateZ": 0.0,
        "gravityX": 0.0,
        "gravityY": 0.0,
        "gravityZ": 0.0,
        "x_norm": 0.0,
        "y_norm": 0.0,
    }


def sample_with_times(times, sample_rate_hz=200.0, use_touch_frame=True):
    data = {
        "id": "sample",
        "anchorTimestamp": 10.0,
        "sampleRateHz": sample_rate_hz,
        "frames": [observed_frame(value, index) for index, value in enumerate(times)],
        "touchEvents": [{"kind": "sample", "timestamp": 10.0}],
    }
    if use_touch_frame:
        data["touchFrames"] = [{"frameIndex": 0, "eventTimeMillis": 10000, "relativeTimestamp": 0.0}]
    return data


class DownAnchorAnalysisTests(unittest.TestCase):
    def test_zero_placeholder_is_not_observed(self):
        self.assertFalse(is_observed_frame(placeholder_frame(0.0, 0)))

    def test_inside_window_without_full_context_is_a_only(self):
        data = sample_with_times([-0.010, -0.005, 0.000, 0.005, 0.010])

        result = analyze_sample(data)

        self.assertTrue(result["eligible_a"])
        self.assertFalse(result["eligible_b"])

    def test_complete_continuous_window_is_a_and_b(self):
        data = sample_with_times([index * 0.005 for index in range(-5, 10)])

        result = analyze_sample(data)

        self.assertTrue(result["eligible_a"])
        self.assertTrue(result["eligible_b"])
        self.assertEqual(list(range(-5, 10)), [frame["frameIndex"] for frame in result["down_frames"]])

    def test_down_before_first_observed_frame_is_excluded_from_a(self):
        data = sample_with_times([0.050, 0.055, 0.060, 0.065])

        result = analyze_sample(data)

        self.assertFalse(result["eligible_a"])
        self.assertEqual("down_outside_observed_range", result["exclusion_reason_a"])

    def test_placeholder_frames_do_not_complete_b(self):
        times = [index * 0.005 for index in range(-5, 10)]
        data = sample_with_times(times)
        data["frames"][0] = placeholder_frame(times[0], 0)

        result = analyze_sample(data)

        self.assertTrue(result["eligible_a"])
        self.assertFalse(result["eligible_b"])
        self.assertEqual(14, result["observed_frame_count"])

    def test_large_timestamp_gap_excludes_b(self):
        times = [index * 0.005 for index in range(-5, 10)]
        times[10:] = [value + 0.020 for value in times[10:]]
        data = sample_with_times(times)

        result = analyze_sample(data)

        self.assertTrue(result["eligible_a"])
        self.assertFalse(result["eligible_b"])
        self.assertEqual("non_contiguous_down_window", result["exclusion_reason_b"])

    def test_touch_event_timestamp_is_fallback_down_marker(self):
        data = sample_with_times([-0.005, 0.000, 0.005], use_touch_frame=False)

        result = analyze_sample(data)

        self.assertTrue(result["eligible_a"])
        self.assertEqual("touchEvents[0].timestamp", result["down_timestamp_source"])


if __name__ == "__main__":
    unittest.main()
