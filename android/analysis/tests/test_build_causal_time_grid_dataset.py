import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


from android.analysis import build_causal_time_grid_dataset as dataset_builder


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ANALYSIS_DIR / "tests" / "fixtures" / "cross_device_causal_feature_fixture.json"


def valid_sample(sample_id):
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    record = fixture["records"]["android_200hz"]
    return {
        "schemaVersion": fixture["schemaVersion"],
        "id": sample_id,
        "createdAt": "2026-08-10T12:00:00Z",
        "collectionSessionID": "session-a",
        "recordStatus": "accepted",
        "collector": {"userName": "test0810", "deviceId": "device-a"},
        "label": {
            "motionState": "handheld_static",
            "touchStrength": "light",
            "fingerType": "thumb",
        },
        "deviceCapability": {"deviceName": "Test Device"},
        "timeAlignment": fixture["timeAlignment"],
        "sensorMetadata": fixture["sensorMetadata"],
        "touchMetadata": fixture["touchMetadata"],
        "frames": record["frames"],
        "touchFrames": record["touchFrames"],
    }


class CausalTimeGridDatasetTests(unittest.TestCase):
    def test_label_quality_keeps_sources_rejected_by_canonical_builder(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = root / "raw"
            output_dir = root / "derived"
            source_dir.mkdir()
            light = valid_sample("light-id")
            heavy = valid_sample("heavy-id")
            heavy["label"]["touchStrength"] = "heavy"
            heavy["sourceTouchKind"] = "heavyTap"
            heavy["timeAlignment"]["liftEventElapsedRealtimeNanos"] = (
                heavy["timeAlignment"]["downEventElapsedRealtimeNanos"]
                + 1_200_000_000
            )
            (source_dir / "light.json").write_text(
                json.dumps(light),
                encoding="utf-8",
            )
            (source_dir / "heavy.json").write_text(
                json.dumps(heavy),
                encoding="utf-8",
            )
            original_builder = dataset_builder.cross_device.build_time_grid_matrix

            def build_with_one_canonical_rejection(data, *args, **kwargs):
                if data.get("id") == "heavy-id":
                    raise RuntimeError("canonical_window_rejected")
                return original_builder(data, *args, **kwargs)

            with mock.patch.object(
                dataset_builder.cross_device,
                "build_time_grid_matrix",
                side_effect=build_with_one_canonical_rejection,
            ):
                summary = dataset_builder.build_dataset(
                    source_dir,
                    output_dir,
                )

            self.assertEqual(2, summary["sourceLabelQualitySampleCount"])
            self.assertEqual(1, summary["canonicalLabelQualitySampleCount"])
            self.assertIn(
                "long_contact_rate_above_target",
                summary["labelQualityBlockers"],
            )
            self.assertTrue(
                (
                    output_dir
                    / "canonical_touch_label_quality_report.json"
                ).is_file()
            )

    def test_build_dataset_writes_contracts_and_partitions_sources_once(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = root / "raw"
            output_dir = root / "derived"
            source_dir.mkdir()
            payload = json.dumps(valid_sample("accepted-id"), sort_keys=True)
            accepted_path = source_dir / "accepted.json"
            accepted_path.write_text(payload, encoding="utf-8")
            (source_dir / "duplicate.json").write_text(payload, encoding="utf-8")
            rejected = valid_sample("rejected-id")
            rejected["timeAlignment"]["liftEventElapsedRealtimeNanos"] = 1_020_000_000
            (source_dir / "rejected.json").write_text(json.dumps(rejected), encoding="utf-8")

            summary = dataset_builder.build_dataset(source_dir, output_dir)

            self.assertEqual(3, summary["sourceCount"])
            self.assertEqual(1, summary["acceptedCount"])
            self.assertEqual(2, summary["rejectedCount"])
            self.assertTrue((output_dir / "feature_contract.json").is_file())
            self.assertTrue((output_dir / "source_hash_manifest.sha256").is_file())
            self.assertEqual(
                1,
                len((output_dir / "canonical_imu11.jsonl").read_text(encoding="utf-8").splitlines()),
            )
            touch_row = json.loads(
                (output_dir / "canonical_imu11_touch9.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual("causal_time_grid_imu11_touch9_v2", touch_row["featureContract"])
            self.assertEqual([9, 20], touch_row["matrixShape"])
            self.assertEqual("thumb_light", touch_row["classLabel"])
            self.assertEqual(9, len(touch_row["matrix"]))
            self.assertEqual(20, len(touch_row["matrix"][0]))
            self.assertEqual(
                0,
                touch_row["touchAxisRangeAudit"]["framesWithMotionRanges"],
            )
            v4_row = json.loads(
                (
                    output_dir
                    / "canonical_imu11_touch9_mask6_v4.jsonl"
                ).read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(
                "causal_time_grid_imu11_touch9_mask6_v4",
                v4_row["featureContract"],
            )
            self.assertEqual([9, 26], v4_row["matrixShape"])
            self.assertGreater(v4_row["matrix"][1][13], 0.0)
            self.assertGreater(v4_row["matrix"][1][14], 0.0)
            with (output_dir / "source_manifest.csv").open("r", encoding="utf-8", newline="") as source:
                source_rows = list(csv.DictReader(source))
            self.assertEqual(3, len(source_rows))
            self.assertEqual(
                {"accepted", "rejected"},
                {row["normalizationStatus"] for row in source_rows},
            )
            accepted_row = next(row for row in source_rows if row["sourceFile"] == "accepted.json")
            self.assertEqual(hashlib.sha256(payload.encode("utf-8")).hexdigest(), accepted_row["sourceSha256"])
            duplicate_row = next(row for row in source_rows if row["sourceFile"] == "duplicate.json")
            self.assertEqual("duplicate_source_sha256", duplicate_row["rejectionReason"])
            with (output_dir / "rejected_samples.csv").open("r", encoding="utf-8", newline="") as source:
                rejected_rows = list(csv.DictReader(source))
            self.assertEqual(2, len(rejected_rows))
            with (output_dir / "grid_quality_report.csv").open("r", encoding="utf-8", newline="") as source:
                quality_rows = list(csv.DictReader(source))
            self.assertTrue(quality_rows)
            self.assertEqual(
                "0",
                quality_rows[0]["touchAxisRangeStatus"],
            )

    def test_ordinary_touch_is_accepted_as_a_rejection_negative(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = root / "raw"
            output_dir = root / "derived"
            source_dir.mkdir()
            sample = valid_sample("ordinary-id")
            sample["label"]["touchStrength"] = "ordinary"
            sample["collectionContext"] = {
                "schemaVersion": 1,
                "contract": "touch_collection_context_v1",
                "collectionProtocol": "negative_tap_v1",
                "strengthTruthSource": "user_task",
                "taskId": "tilted_handheld",
                "bodyPosture": "unspecified",
                "requestedDevicePose": "tilted_toward_user",
                "sceneSource": "task_instruction_not_verified",
                "grip": "same_hand",
                "support": "handheld",
                "contactMode": "natural",
                "gestureIntent": "tap",
                "strengthOnset": "at_contact",
                "labelStatus": "task_assigned",
                "targetPerCell": 10,
            }
            (source_dir / "ordinary.json").write_text(
                json.dumps(sample),
                encoding="utf-8",
            )

            summary = dataset_builder.build_dataset(
                source_dir,
                output_dir,
            )

            self.assertEqual(1, summary["acceptedCount"])
            self.assertEqual(
                {dataset_builder.ORDINARY_CLASS_LABEL: 1},
                summary["classCounts"],
            )
            touch_row = json.loads(
                (
                    output_dir / "canonical_imu11_touch9.jsonl"
                ).read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(
                dataset_builder.ORDINARY_CLASS_LABEL,
                touch_row["classLabel"],
            )
            self.assertEqual(
                "negative_tap_v1",
                touch_row["collectionProtocol"],
            )
            self.assertEqual(
                "user_task",
                touch_row["strengthTruthSource"],
            )

if __name__ == "__main__":
    unittest.main()
