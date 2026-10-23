import json
import tempfile
import unittest
from pathlib import Path

from android.analysis import clean_android_touch_dataset as cleaner


class ContextQualityTests(unittest.TestCase):
    def record(self, device_name, motion_state, delta, travel=4.0):
        return {
            "device_name": device_name,
            "motion_state": motion_state,
            "max_abs_delta": delta,
            "path_travel_px": travel,
        }

    def test_context_limit_accepts_device_specific_normal_range(self):
        records = [
            self.record("PHK110", "walking", 6.0 + index * 0.01)
            for index in range(100)
        ]
        records.append(self.record("PHK110", "walking", 30.0))

        stats = cleaner.build_context_stats(records)
        limit = cleaner.context_upper_limit(
            records[0], stats, "max_abs_delta", absolute_floor=5.0
        )

        self.assertGreater(limit, 6.9)
        self.assertLess(limit, 30.0)

    def test_context_limit_keeps_floor_for_low_noise_device(self):
        records = [
            self.record("LE2110", "handheld_static", 0.1 + index * 0.005)
            for index in range(100)
        ]

        stats = cleaner.build_context_stats(records)
        limit = cleaner.context_upper_limit(
            records[0], stats, "max_abs_delta", absolute_floor=5.0
        )

        self.assertEqual(limit, 5.0)

    def test_weighted_records_include_review_without_rejected(self):
        records = [
            {"id": "high", "quality_bucket": cleaner.KEEP_DIR},
            {"id": "review", "quality_bucket": cleaner.REVIEW_DIR},
            {"id": "rejected", "quality_bucket": cleaner.REJECT_DIR},
        ]

        selected = cleaner.select_weighted_records(records)

        self.assertEqual([record["id"] for record in selected], ["high", "review"])
        self.assertEqual(selected[0]["sample_weight"], 1.0)
        self.assertEqual(selected[1]["sample_weight"], 0.35)

    def test_suspected_label_conflict_is_downweighted(self):
        record = {
            "quality_bucket": cleaner.REVIEW_DIR,
            "quality_reasons": "light_label_heavy_shape",
        }

        weight = cleaner.sample_weight_for_record(record, {})

        self.assertLess(weight, 0.35)

    def test_ordinary_touch_is_kept_for_rejection_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "ordinary.json"
            path.write_text(
                json.dumps({
                    "id": "ordinary-a",
                    "label": {
                        "touchStrength": "ordinary",
                        "motionState": "handheld_static",
                        "fingerType": "thumb",
                    },
                }),
                encoding="utf-8",
            )

            record = cleaner.read_record(path)

        self.assertTrue(record["load_ok"])
        self.assertEqual(
            cleaner.ORDINARY_CLASS_LABEL,
            record["label_id"],
        )
        self.assertEqual(0.0, cleaner.sample_weight_for_record(record, {}))


if __name__ == "__main__":
    unittest.main()
