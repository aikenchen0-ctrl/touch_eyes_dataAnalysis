import unittest

from android.analysis import audit_pretouch_contamination as audit


def frame(relative_ms, acceleration):
    return {
        "sensorTimestampElapsedRealtimeNanos": (
            1_000_000_000 + int(relative_ms * 1_000_000)
        ),
        "x": float(acceleration),
        "y": 0.0,
        "z": 0.0,
    }


class PretouchContaminationAuditTests(unittest.TestCase):
    def test_minus_five_slot_detects_interpolation_crossing_down(self):
        down_ns = 1_000_000_000
        frames = [
            frame(-20, 0.1),
            frame(-8, 0.2),
            frame(2, 1.0),
        ]

        result = audit.grid_interpolation_audit(frames, down_ns)

        self.assertTrue(result["-5"]["interpolationCrossesDown"])
        self.assertFalse(result["-5"]["strictlyPreDispatch"])
        self.assertTrue(result["-15"]["strictlyPreDispatch"])

    def test_pre_dispatch_frame_remains_strictly_causal(self):
        down_ns = 1_000_000_000
        frames = [
            frame(-20, 0.1),
            frame(-8, 0.2),
            frame(-1, 0.3),
            frame(2, 1.0),
        ]

        result = audit.grid_interpolation_audit(frames, down_ns)

        self.assertFalse(result["-5"]["interpolationCrossesDown"])
        self.assertTrue(result["-5"]["strictlyPreDispatch"])

    def test_rising_pretouch_is_detected_from_early_frames(self):
        down_ns = 1_000_000_000
        frames = [
            frame(-75, 0.10),
            frame(-65, 0.12),
            frame(-55, 0.18),
            frame(-45, 0.26),
            frame(-35, 0.40),
            frame(-25, 0.60),
            frame(5, 2.00),
        ]

        result = audit.rising_pretouch_audit(frames, down_ns)

        self.assertTrue(result["available"])
        self.assertTrue(result["rising"])
        self.assertGreater(result["delta"], result["threshold"])

    def test_flat_pretouch_is_not_marked_as_rising(self):
        down_ns = 1_000_000_000
        frames = [
            frame(-75, 0.10),
            frame(-65, 0.09),
            frame(-55, 0.10),
            frame(-45, 0.11),
            frame(-35, 0.10),
            frame(-25, 0.09),
        ]

        result = audit.rising_pretouch_audit(frames, down_ns)

        self.assertTrue(result["available"])
        self.assertFalse(result["rising"])

    def test_sparse_pretouch_is_reported_as_unavailable(self):
        down_ns = 1_000_000_000
        frames = [
            frame(-70, 0.1),
            frame(-40, 0.2),
        ]

        result = audit.rising_pretouch_audit(frames, down_ns)

        self.assertFalse(result["available"])
        self.assertFalse(result["rising"])


if __name__ == "__main__":
    unittest.main()
