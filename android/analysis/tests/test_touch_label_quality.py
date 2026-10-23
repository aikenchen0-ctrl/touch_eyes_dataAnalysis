import json
import tempfile
import unittest
from pathlib import Path

from android.analysis import touch_label_quality as quality


def sample(
        sample_id,
        session_id,
        class_label,
        duration_ms=80.0,
        pointers=(1, 1),
        sizes=(0.04, 0.05),
):
    finger, strength = class_label.split("_")
    down = 1_000_000_000
    lift = down + int(duration_ms * 1_000_000)
    touch_frames = [
        {
            "pointerCount": pointer,
            "size": size,
        }
        for pointer, size in zip(pointers, sizes)
    ]
    return {
        "id": sample_id,
        "collectionSessionID": session_id,
        "label": {
            "fingerType": finger,
            "touchStrength": strength,
        },
        "collectionContext": {
            "taskId": "tilted_handheld",
            "bodyPosture": "tilted_handheld",
            "collectionProtocol": "interleaved_v2",
            "strengthTruthSource": "user_task",
            "labelStatus": "task_assigned",
        },
        "deviceCapability": {"deviceName": "Device A"},
        "sourceTouchKind": (
            "lightTap" if strength == "light" else "heavyTap"
        ),
        "timeAlignment": {
            "downEventElapsedRealtimeNanos": down,
            "liftEventElapsedRealtimeNanos": lift,
        },
        "touchFrames": touch_frames,
    }


class TouchLabelQualityTests(unittest.TestCase):
    def test_clean_interleaved_records_pass_quality_gate(self):
        records = [
            quality.sample_record(
                label + ".json",
                label,
                sample("id-" + label, "session-a", label),
            )
            for label in quality.CLASS_LABELS
        ]

        report = quality.summarize_quality(records)

        self.assertTrue(report["trainingLabelQualityReady"])
        self.assertEqual(1, report["sessions"]["multiClassSessionCount"])
        self.assertEqual(0, report["flagCounts"]["long_contact"])
        self.assertEqual(4, report["trustedStrengthLabelCount"])

    def test_long_and_multi_pointer_contacts_block_training_claims(self):
        records = [
            quality.sample_record(
                "long.json",
                "long",
                sample("long", "session-a", "thumb_light", 1200.0),
            ),
            quality.sample_record(
                "multi.json",
                "multi",
                sample(
                    "multi",
                    "session-b",
                    "thumb_heavy",
                    pointers=(1, 2),
                ),
            ),
        ]

        report = quality.summarize_quality(records)

        self.assertFalse(report["trainingLabelQualityReady"])
        self.assertIn(
            "long_contact_rate_above_target",
            report["blockers"],
        )
        self.assertIn(
            "multi_pointer_rate_above_target",
            report["blockers"],
        )
        self.assertIn(
            "session_label_binding_above_target",
            report["blockers"],
        )

    def test_missing_duration_and_wrong_touch_kind_are_auditable(self):
        value = sample("bad", "session-a", "index_heavy")
        value["timeAlignment"]["liftEventElapsedRealtimeNanos"] = 0
        value["sourceTouchKind"] = "lightTap"
        record = quality.sample_record("bad.json", "bad", value)

        self.assertIsNone(record["durationMs"])
        self.assertIn("missing_touch_duration", record["qualityFlags"])
        self.assertIn("unexpected_touch_kind", record["qualityFlags"])

    def test_empty_supervised_rows_fail_closed(self):
        report = quality.summarize_quality([])

        self.assertFalse(report["trainingLabelQualityReady"])
        self.assertEqual(["no_label_quality_samples"], report["blockers"])
        self.assertEqual(0, report["sampleCount"])

    def test_unverified_task_labels_block_training_claims(self):
        value = sample("weak", "session-a", "thumb_heavy")
        value["collectionContext"]["collectionProtocol"] = "unknown"
        value["collectionContext"]["strengthTruthSource"] = "unknown"
        record = quality.sample_record("weak.json", "weak", value)

        report = quality.summarize_quality([record])

        self.assertFalse(report["trainingLabelQualityReady"])
        self.assertEqual(0, report["trustedStrengthLabelCount"])
        self.assertIn(
            "no_trusted_strength_labels",
            report["blockers"],
        )
        self.assertEqual(
            {"task_assigned": 1},
            report["labelStatusCounts"],
        )

    def test_writer_and_reader_preserve_gate_status(self):
        with tempfile.TemporaryDirectory() as directory:
            records = [
                quality.sample_record(
                    label + ".json",
                    label,
                    sample("id-" + label, "session-a", label),
                )
                for label in quality.CLASS_LABELS
            ]
            written = quality.write_quality_outputs(directory, records)
            loaded = quality.read_quality_report(directory)
            csv_text = (
                Path(directory) / "touch_label_quality_records.csv"
            ).read_text(encoding="utf-8")

            self.assertEqual(written, loaded)
            self.assertTrue((Path(directory) / "touch_label_quality_records.csv").is_file())
            self.assertIn("labelTrust", csv_text.splitlines()[0])
            self.assertEqual(
                {"trusted"},
                {
                    record["labelTrust"]
                    for record in records
                },
            )
            self.assertEqual(
                "touch_label_quality_v1",
                json.loads(
                    (Path(directory) / "touch_label_quality_report.json").read_text(
                        encoding="utf-8"
                    )
                )["contract"],
            )


if __name__ == "__main__":
    unittest.main()
