import csv
import json
import tempfile
import unittest
from pathlib import Path

from android.analysis.build_deduplicated_clean_v3 import build_context_stats
from android.analysis.build_deduplicated_clean_v3 import build_dataset
from android.analysis.build_deduplicated_clean_v3 import decide_quality


def quality_record(**overrides):
    record = {
        "load_ok": True,
        "id": "sample-id",
        "class_label": "thumb_heavy",
        "touch_strength": "heavy",
        "finger_type": "thumb",
        "motion_state": "handheld_static",
        "device_name": "OnePlus LE2110",
        "window_frame_count": 21,
        "observed_window_frames": 15,
        "sample_rate_hz": 200.0,
        "max_pointer_count": 1,
        "source_touch_kind": "heavyTap",
        "path_travel_px": 4.0,
        "touch_frame_count": 2,
        "max_abs_delta": 0.8,
        "raw_major_px": 12.0,
        "raw_minor_px": 10.0,
        "touch_norm_valid": True,
        "touch_norm_fallback": False,
        "contact_patch_default": False,
    }
    record.update(overrides)
    return record


def sample_payload(sample_id="sample-id"):
    frames = []
    for index in range(-5, 16):
        frames.append(
            {
                "frameIndex": index,
                "relativeTimestamp": index / 200.0,
                "delta": 0.1,
                "x": 0.1,
                "y": 0.2,
                "z": 0.3,
            }
        )
    return {
        "schemaVersion": 3,
        "id": sample_id,
        "createdAt": 1_784_779_169.624,
        "collectionSessionID": "session-a",
        "sampleRateHz": 200.0,
        "sourceTouchKind": "heavyTap",
        "maxAbsDelta": 0.8,
        "collector": {"userName": "tester"},
        "label": {
            "motionState": "handheld_static",
            "touchStrength": "heavy",
            "fingerType": "thumb",
        },
        "deviceCapability": {"level": "A", "deviceName": "OnePlus LE2110"},
        "frames": frames,
        "motionSignal": {"frameCount": 300},
        "touchFrames": [
            {
                "eventTimeMillis": 1.0,
                "x": 10.0,
                "y": 20.0,
                "pointerCount": 1,
                "touchMajor": 12.0,
                "touchMinor": 10.0,
            },
            {
                "eventTimeMillis": 10.0,
                "x": 11.0,
                "y": 20.0,
                "pointerCount": 1,
                "touchMajor": 12.0,
                "touchMinor": 10.0,
            },
        ],
        "touchNormalization": {
            "selectedResult": {
                "valid": True,
                "usedFallback": False,
                "rawMajorPx": 12.0,
                "rawMinorPx": 10.0,
            }
        },
        "contactPatch": {"valid": True, "defaultApplied": False},
    }


class CleanV3Tests(unittest.TestCase):
    def test_context_duration_does_not_apply_global_four_frame_rule(self):
        records = [
            quality_record(id="le-a", touch_frame_count=2),
            quality_record(id="le-b", touch_frame_count=2),
            quality_record(id="le-c", touch_frame_count=3),
            quality_record(
                id="phk-a", device_name="OnePlus PHK110", touch_frame_count=8
            ),
            quality_record(
                id="phk-b", device_name="OnePlus PHK110", touch_frame_count=9
            ),
        ]

        stats = build_context_stats(records)
        decided = decide_quality(quality_record(touch_frame_count=2), stats)

        self.assertEqual("high_quality", decided["quality_bucket"])
        self.assertNotIn("touch_duration_low", decided["quality_reasons"])

    def test_structural_and_multi_pointer_failures_are_rejected(self):
        stats = build_context_stats([quality_record()])

        unsupported = decide_quality(
            quality_record(class_label="unknown", touch_strength="unknown"), stats
        )
        multi_pointer = decide_quality(
            quality_record(max_pointer_count=2), stats
        )

        self.assertEqual("rejected", unsupported["quality_bucket"])
        self.assertIn("unsupported_class", unsupported["quality_reasons"])
        self.assertEqual("rejected", multi_pointer["quality_bucket"])
        self.assertIn("multi_pointer", multi_pointer["quality_reasons"])

    def test_review_weights_are_bounded(self):
        stats = build_context_stats([quality_record()])

        decided = decide_quality(
            quality_record(
                raw_major_px=0.0,
                raw_minor_px=0.0,
                touch_norm_valid=False,
                contact_patch_default=True,
            ),
            stats,
        )

        self.assertEqual("review", decided["quality_bucket"])
        self.assertGreaterEqual(decided["sample_weight"], 0.2)
        self.assertLessEqual(decided["sample_weight"], 0.6)

    def test_build_dataset_reads_only_manifest_references(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sample_path = root / "sample.json"
            sample_path.write_text(json.dumps(sample_payload()), encoding="utf-8")
            (root / "summary.json").write_text(
                json.dumps({"retained": 1}), encoding="utf-8"
            )
            source_manifest = root / "manifest.csv"
            with source_manifest.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=(
                        "id",
                        "sha256",
                        "collector",
                        "device",
                        "date",
                        "source",
                        "destination",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "id": "sample-id",
                        "sha256": "hash",
                        "collector": "tester",
                        "device": "oneplus-le2110",
                        "date": "2026-07-23",
                        "source": "source",
                        "destination": "sample.json",
                    }
                )

            output_dir = root / "cleaned"
            summary = build_dataset(source_manifest, root, output_dir)

            self.assertEqual(1, summary["total"])
            self.assertEqual(1, summary["bucketCounts"]["high_quality"])
            with (output_dir / "master_manifest.csv").open(
                "r", encoding="utf-8", newline=""
            ) as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(["sample-id"], [row["id"] for row in rows])
            self.assertEqual("15", rows[0]["observed_window_frames"])

    def test_origin_directory_overrides_incorrect_embedded_collector(self):
        sample_id = "12345678-1234-1234-1234-123456789abc"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            deduplicated_root = root / "data" / "deduplicated_raw"
            deduplicated_root.mkdir(parents=True)
            sample_path = deduplicated_root / (sample_id + ".json")
            sample_path.write_text(
                json.dumps(sample_payload(sample_id)), encoding="utf-8"
            )
            origin = (
                root
                / "data"
                / "collections"
                / "2026-07-22_oneplus_phk110_serial_actual-person"
                / "raw"
            )
            origin.mkdir(parents=True)
            (origin / ("timestamp_label_" + sample_id + ".json")).write_text(
                "{}", encoding="utf-8"
            )
            source_manifest = deduplicated_root / "manifest.csv"
            with source_manifest.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=("id", "sha256", "collector", "device", "date", "source", "destination"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "id": sample_id,
                        "sha256": "hash",
                        "collector": "embedded-person",
                        "device": "oneplus-phk110",
                        "date": "2026-07-22",
                        "source": "source",
                        "destination": sample_path.name,
                    }
                )

            output_dir = root / "cleaned"
            summary = build_dataset(
                source_manifest,
                deduplicated_root,
                output_dir,
                origin_root=root / "data",
            )

            with (output_dir / "master_manifest.csv").open(
                "r", encoding="utf-8", newline=""
            ) as source:
                result = next(csv.DictReader(source))
            self.assertEqual("actual-person", result["collector_user_name"])
            self.assertEqual(
                "2026-07-22-oneplus-phk110-serial-actual-person",
                result["collection_name"],
            )
            self.assertEqual(1, summary["collectorMetadataMismatches"])


if __name__ == "__main__":
    unittest.main()
