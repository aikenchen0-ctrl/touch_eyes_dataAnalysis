import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path


from android.analysis import build_time_grid_training_dataset as training_builder


CLASS_LABELS = (
    "thumb_light",
    "thumb_heavy",
    "index_light",
    "index_heavy",
)
MASK_KEYS = ("pressure", "size", "major", "minor", "orientation", "pointerCount")


def matrix(width, pointer_count=1.0):
    values = []
    for index in range(9):
        row = [float(index + column) for column in range(width)]
        if width == 20:
            row[18] = pointer_count
            row[19] = 1.0
        values.append(row)
    return values


def touch_masks():
    return [
        {
            "touch": 1,
            "pressure": 1,
            "size": 1,
            "major": 1,
            "minor": 1,
            "orientation": 0,
            "pointerCount": 1,
        }
        for _ in range(9)
    ]


def source_row(source_file, source_sha256, sample_id, label, session_id):
    finger, strength = label.split("_")
    return {
        "sourceFile": source_file,
        "sourceSha256": source_sha256,
        "normalizationStatus": "accepted",
        "rejectionReason": "",
        "schemaVersion": "6",
        "sampleId": sample_id,
        "classLabel": label,
        "motionState": "handheld_static",
        "collectorUserName": "tester",
        "collectorDeviceId": "device-a",
        "deviceName": "Test Device",
        "platform": "android",
        "sensorSource": "linear_acceleration",
        "sessionId": session_id,
        "finger": finger,
        "strength": strength,
    }


def matrix_row(source, width, pointer_count=1.0):
    contract = "causal_time_grid_imu11_v2" if width == 11 else "causal_time_grid_imu11_touch9_v2"
    return {
        "sampleId": source["sampleId"],
        "sourceFile": source["sourceFile"],
        "sourceSha256": source["sourceSha256"],
        "classLabel": source["classLabel"],
        "motionState": source["motionState"],
        "collectorUserName": source["collectorUserName"],
        "collectorDeviceId": source["collectorDeviceId"],
        "deviceName": source["deviceName"],
        "platform": source["platform"],
        "sensorSource": source["sensorSource"],
        "sessionId": source["sessionId"],
        "featureContract": contract,
        "featureNames": ["f" + str(index) for index in range(width)],
        "matrixShape": [9, width],
        "matrix": matrix(width, pointer_count),
        "missingMasks": touch_masks() if width == 20 else [],
        "gridOffsetsMs": [-15, -10, -5, 0, 5, 10, 15, 20, 25],
        "gridRateHz": 200,
        "gridQuality": {"status": "accepted", "windowDurationMs": 40},
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


class TimeGridTrainingDatasetTests(unittest.TestCase):
    def write_derived_dataset(self, root, mismatch=False):
        root.mkdir()
        source_rows = []
        imu_rows = []
        touch_rows = []
        index = 0
        for label in CLASS_LABELS:
            for session_index in range(2):
                index += 1
                sample_id = label + "-" + str(session_index)
                source_file = str(1_700_000_000_000 + index) + "_" + sample_id + ".json"
                payload = source_file + sample_id
                source = source_row(
                    source_file,
                    hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                    sample_id,
                    label,
                    label + "-session-" + str(session_index),
                )
                source_rows.append(source)
                imu_rows.append(matrix_row(source, 11))
                touch_rows.append(matrix_row(source, 20))
        excluded_source = source_row(
            "1700000000999_multi.json",
            hashlib.sha256(b"multi").hexdigest(),
            "multi-pointer",
            "thumb_light",
            "multi-session",
        )
        source_rows.append(excluded_source)
        imu_rows.append(matrix_row(excluded_source, 11))
        touch_rows.append(matrix_row(excluded_source, 20, pointer_count=2.0))
        if mismatch:
            touch_rows[0]["sourceSha256"] = "mismatch"

        with (root / "source_manifest.csv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(source_rows[0].keys()))
            writer.writeheader()
            writer.writerows(source_rows)
        write_jsonl(root / "canonical_imu11.jsonl", imu_rows)
        write_jsonl(root / "canonical_imu11_touch9.jsonl", touch_rows)

    def test_builds_masked_training_view_and_leak_free_group_folds(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            derived = root / "derived"
            output = root / "training"
            self.write_derived_dataset(derived)
            source_hash_before = hashlib.sha256((derived / "source_manifest.csv").read_bytes()).hexdigest()

            summary = training_builder.build_training_dataset(derived, output)

            self.assertEqual(9, summary["candidateCount"])
            self.assertEqual(8, summary["eligibleCount"])
            self.assertEqual(1, summary["excludedCount"])
            self.assertEqual(2, summary["strictFoldCount"])
            self.assertTrue(summary["sessionLabelBinding"])
            self.assertEqual(["orientation"], summary["allMissingMaskNames"])
            self.assertEqual(source_hash_before, hashlib.sha256((derived / "source_manifest.csv").read_bytes()).hexdigest())
            touch_rows = [json.loads(line) for line in (output / "training_touch26.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(8, len(touch_rows))
            self.assertEqual([9, 26], touch_rows[0]["matrixShape"])
            self.assertEqual([1, 1, 1, 1, 0, 1], touch_rows[0]["matrix"][0][-6:])
            self.assertEqual(1.0, touch_rows[0]["trainingWeight"])
            with (output / "training_candidate_manifest.csv").open("r", encoding="utf-8", newline="") as source:
                candidates = list(csv.DictReader(source))
            multi = next(row for row in candidates if row["sampleId"] == "multi-pointer")
            self.assertEqual("excluded", multi["trainingEligibility"])
            self.assertEqual("multi_pointer_touch", multi["exclusionReason"])
            with (output / "strict_group_folds.csv").open("r", encoding="utf-8", newline="") as source:
                folds = list(csv.DictReader(source))
            for fold in ("0", "1"):
                fold_rows = [row for row in folds if row["fold"] == fold]
                train_groups = {row["groupId"] for row in fold_rows if row["split"] == "train"}
                valid_groups = {row["groupId"] for row in fold_rows if row["split"] == "valid"}
                self.assertFalse(train_groups.intersection(valid_groups))
                self.assertEqual(set(CLASS_LABELS), {row["classLabel"] for row in fold_rows if row["split"] == "valid"})
            contract = json.loads((output / "training_feature_contract.json").read_text(encoding="utf-8"))
            self.assertEqual(26, contract["contracts"]["causal_time_grid_imu11_touch9_mask6_v3"]["matrixShape"][1])
            self.assertEqual(list(MASK_KEYS), contract["maskFeatureNames"])
            self.assertIn("timeSinceAnchor", contract["prohibitedFeatureNames"])

    def test_rejects_mismatched_matrix_source_identity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            derived = root / "derived"
            self.write_derived_dataset(derived, mismatch=True)

            with self.assertRaisesRegex(RuntimeError, "matrix_source_identity_mismatch"):
                training_builder.build_training_dataset(derived, root / "training")

    def test_excludes_source_without_session_identity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            derived = root / "derived"
            self.write_derived_dataset(derived)
            with (derived / "source_manifest.csv").open("r", encoding="utf-8", newline="") as source:
                sources = list(csv.DictReader(source))
            sources[0]["sessionId"] = ""
            with (derived / "source_manifest.csv").open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(sources[0].keys()))
                writer.writeheader()
                writer.writerows(sources)
            for name in ("canonical_imu11.jsonl", "canonical_imu11_touch9.jsonl"):
                rows = [json.loads(line) for line in (derived / name).read_text(encoding="utf-8").splitlines()]
                rows[0]["sessionId"] = ""
                write_jsonl(derived / name, rows)

            training_builder.build_training_dataset(derived, root / "training")

            with (root / "training" / "training_candidate_manifest.csv").open("r", encoding="utf-8", newline="") as source:
                candidates = list(csv.DictReader(source))
            self.assertEqual("excluded", candidates[0]["trainingEligibility"])
            self.assertEqual("missing_group_identity", candidates[0]["exclusionReason"])


if __name__ == "__main__":
    unittest.main()
