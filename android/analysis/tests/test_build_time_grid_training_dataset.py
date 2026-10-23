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
        if width in (20, 26):
            row[18] = pointer_count
            row[19] = 1.0
        if width == 26:
            row[20:26] = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
        if width == 23:
            # 中文注释：v5 中 pointer_count 和 touch_valid 位于 16、17，
            # 后续五列是 availability 掩码。
            row[16] = pointer_count
            row[17] = 1.0
            row[18:23] = [1.0, 1.0, 1.0, 1.0, 1.0]
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
    result = {
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
    for field in training_builder.AUDIT_FIELDS:
        result[field] = ""
    return result


def matrix_row(source, width, pointer_count=1.0):
    contract = (
        "causal_time_grid_imu11_v2"
        if width == 11
        else (
            "causal_time_grid_imu11_touch7_mask5_v5"
            if width == 23
            else "causal_time_grid_imu11_touch9_mask6_v4"
        )
    )
    result = {
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
        "featureNames": (
            list(training_builder.IMU_FEATURE_NAMES)
            if width == 11
            else (
                list(
                    training_builder.IMU_FEATURE_NAMES
                    + training_builder.V5_TOUCH_FEATURE_NAMES
                    + training_builder.V5_MASK_FEATURE_NAMES
                )
                if width == 23
                else list(
                    training_builder.IMU_FEATURE_NAMES
                    + training_builder.TOUCH_FEATURE_NAMES
                    + training_builder.MASK_FEATURE_NAMES
                )
            )
        ),
        "matrixShape": [9, width],
        "matrix": matrix(width, pointer_count),
        "missingMasks": touch_masks() if width in (23, 26) else [],
        "gridOffsetsMs": [-15, -10, -5, 0, 5, 10, 15, 20, 25],
        "gridRateHz": 200,
        "gridQuality": {"status": "accepted", "windowDurationMs": 40},
    }
    if width in (23, 26):
        result["touchAxisRangeAudit"] = {
            "framesWithMotionRanges": 9,
            "ranges": {
                "touchMajor": {
                    "declaredToViewDiagonalMedian": 0.125,
                }
            },
        }
    for field in training_builder.AUDIT_FIELDS:
        result[field] = source.get(field, "")
    return result


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


class TimeGridTrainingDatasetTests(unittest.TestCase):
    def test_ordinary_touch_is_excluded_as_a_negative_sample(self):
        source = source_row(
            "1700000001000_ordinary.json",
            hashlib.sha256(b"ordinary").hexdigest(),
            "ordinary",
            "thumb_light",
            "ordinary-session",
        )
        source["classLabel"] = training_builder.ORDINARY_CLASS_LABEL
        source["collectionProtocol"] = "negative_tap_v1"
        source["strengthTruthSource"] = "user_task"
        source["labelStatus"] = "task_assigned"
        source["taskId"] = "tilted_handheld"
        candidate = {
            "classLabel": training_builder.ORDINARY_CLASS_LABEL,
            "trainingEligibility": "eligible",
            "exclusionReason": "",
        }
        record = training_builder.candidate_record(
            source,
            {"sourceSha256": "imu"},
            {"matrix": [[0.0] * 20 for _ in range(9)]},
        )

        self.assertEqual("excluded", record["trainingEligibility"])
        self.assertEqual(
            "ordinary_touch_negative_sample",
            record["exclusionReason"],
        )
        self.assertEqual(
            "negative_tap_v1",
            record["collectionProtocol"],
        )

    def write_derived_dataset(self, root, mismatch=False, devices=None):
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
                if devices:
                    device_index = (
                        CLASS_LABELS.index(label) + session_index
                    ) % len(devices)
                    source["collectorDeviceId"] = (
                        "device-" + str(device_index)
                    )
                    source["deviceName"] = devices[device_index]
                source_rows.append(source)
                imu_rows.append(matrix_row(source, 11))
                touch_rows.append(matrix_row(source, 23))
        excluded_source = source_row(
            "1700000000999_multi.json",
            hashlib.sha256(b"multi").hexdigest(),
            "multi-pointer",
            "thumb_light",
            "multi-session",
        )
        source_rows.append(excluded_source)
        imu_rows.append(matrix_row(excluded_source, 11))
        touch_rows.append(matrix_row(excluded_source, 23, pointer_count=2.0))
        ordinary_source = source_row(
            "1700000001000_ordinary.json",
            hashlib.sha256(b"ordinary").hexdigest(),
            "ordinary",
            "thumb_light",
            "ordinary-session",
        )
        ordinary_source["classLabel"] = (
            training_builder.ORDINARY_CLASS_LABEL
        )
        ordinary_source.update({
            "taskId": "tilted_handheld",
            "bodyPosture": "unspecified",
            "requestedDevicePose": "tilted_toward_user",
            "grip": "same_hand",
            "support": "handheld",
            "collectionProtocol": "negative_tap_v1",
            "strengthTruthSource": "user_task",
            "labelStatus": "task_assigned",
        })
        source_rows.append(ordinary_source)
        imu_rows.append(matrix_row(ordinary_source, 11))
        touch_rows.append(matrix_row(ordinary_source, 23))
        if mismatch:
            touch_rows[0]["sourceSha256"] = "mismatch"

        with (root / "source_manifest.csv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(source_rows[0].keys()))
            writer.writeheader()
            writer.writerows(source_rows)
        write_jsonl(root / "canonical_imu11.jsonl", imu_rows)
        write_jsonl(
            root / "canonical_imu11_touch7_mask5_v5.jsonl",
            touch_rows,
        )

    def test_builds_masked_training_view_and_leak_free_group_folds(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            derived = root / "derived"
            output = root / "training"
            self.write_derived_dataset(derived)
            source_hash_before = hashlib.sha256((derived / "source_manifest.csv").read_bytes()).hexdigest()

            summary = training_builder.build_training_dataset(derived, output)

            self.assertEqual(10, summary["candidateCount"])
            self.assertEqual(8, summary["eligibleCount"])
            self.assertEqual(2, summary["excludedCount"])
            self.assertEqual(1, summary["ordinaryNegativeCount"])
            self.assertEqual(2, summary["strictFoldCount"])
            self.assertEqual("held_out_session", summary["validationFoldKind"])
            self.assertEqual("single_device_cross_session", summary["evaluationScope"])
            self.assertEqual(1, summary["deviceCount"])
            self.assertTrue(summary["sessionLabelBinding"])
            self.assertEqual([], summary["allMissingMaskNames"])
            self.assertEqual(source_hash_before, hashlib.sha256((derived / "source_manifest.csv").read_bytes()).hexdigest())
            touch_rows = [json.loads(line) for line in (output / "training_touch23.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(8, len(touch_rows))
            ordinary_rows = [
                json.loads(line)
                for line in (
                    output / "ordinary_touch23.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(1, len(ordinary_rows))
            self.assertEqual(
                training_builder.ORDINARY_CLASS_LABEL,
                ordinary_rows[0]["classLabel"],
            )
            self.assertEqual(
                "negative_tap_v1",
                ordinary_rows[0]["collectionProtocol"],
            )
            self.assertEqual(
                "user_task",
                ordinary_rows[0]["strengthTruthSource"],
            )
            self.assertEqual(
                "tilted_handheld",
                ordinary_rows[0]["taskId"],
            )
            self.assertEqual(
                "same_hand",
                ordinary_rows[0]["grip"],
            )
            self.assertEqual(
                "handheld",
                ordinary_rows[0]["support"],
            )
            self.assertEqual(
                "task_assigned",
                ordinary_rows[0]["labelStatus"],
            )
            self.assertEqual([9, 23], touch_rows[0]["matrixShape"])
            self.assertEqual([1, 1, 1, 1, 1], touch_rows[0]["matrix"][0][-5:])
            self.assertEqual(1.0, touch_rows[0]["trainingWeight"])
            self.assertEqual(
                0.125,
                touch_rows[0]["touchAxisRangeAudit"]["ranges"][
                    "touchMajor"
                ]["declaredToViewDiagonalMedian"],
            )
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
            self.assertEqual(
                23,
                contract["contracts"][
                    "causal_time_grid_imu11_touch7_mask5_v5"
                ]["matrixShape"][1],
            )
            self.assertEqual(
                ["pressure", "size", "major", "minor", "pointerCount"],
                contract["maskFeatureNames"],
            )
            self.assertIn("timeSinceAnchor", contract["prohibitedFeatureNames"])

    def test_uses_device_holdout_when_each_device_has_all_classes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            derived = root / "derived"
            output = root / "training"
            self.write_derived_dataset(
                derived,
                devices=["Device A", "Device B"],
            )

            summary = training_builder.build_training_dataset(
                derived,
                output,
            )

            self.assertEqual(2, summary["strictFoldCount"])
            self.assertEqual("held_out_device", summary["validationFoldKind"])
            self.assertEqual("cross_device", summary["evaluationScope"])
            self.assertEqual(2, summary["deviceCount"])
            with (output / "strict_group_folds.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            ) as source:
                folds = list(csv.DictReader(source))
            for fold in ("0", "1"):
                fold_rows = [
                    row for row in folds if row["fold"] == fold
                ]
                valid_devices = {
                    row["groupId"].split(":")[1]
                    for row in fold_rows
                    if row["split"] == "valid"
                }
                train_devices = {
                    row["groupId"].split(":")[1]
                    for row in fold_rows
                    if row["split"] == "train"
                }
                self.assertEqual(1, len(valid_devices))
                self.assertFalse(valid_devices.intersection(train_devices))
                self.assertEqual(
                    set(CLASS_LABELS),
                    {
                        row["classLabel"]
                        for row in fold_rows
                        if row["split"] == "valid"
                    },
                )

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
            for name in (
                "canonical_imu11.jsonl",
                "canonical_imu11_touch7_mask5_v5.jsonl",
            ):
                rows = [json.loads(line) for line in (derived / name).read_text(encoding="utf-8").splitlines()]
                rows[0]["sessionId"] = ""
                write_jsonl(derived / name, rows)

            training_builder.build_training_dataset(derived, root / "training")

            with (root / "training" / "training_candidate_manifest.csv").open("r", encoding="utf-8", newline="") as source:
                candidates = list(csv.DictReader(source))
            self.assertEqual("excluded", candidates[0]["trainingEligibility"])
            self.assertEqual("missing_group_identity", candidates[0]["exclusionReason"])

    def test_collection_readiness_rejects_legacy_block_sessions(self):
        candidates = []
        for label in CLASS_LABELS:
            candidates.append({
                "classLabel": label,
                "groupId": label + "-session",
                "platform": "android",
                "collectorDeviceId": "device-a",
                "deviceName": "Device A",
                "sessionId": label + "-session",
                "taskId": "unknown",
                "bodyPosture": "unknown",
                "requestedDevicePose": "unknown",
                "grip": "unknown",
                "support": "unknown",
                "collectionProtocol": "unknown",
            })

        pose_folds, pose_status = training_builder.build_pose_holdout_folds(
            candidates
        )
        summary = training_builder.summary_record(
            candidates,
            candidates,
            0,
            "insufficient_groups_per_class",
            "held_out_session",
            [],
            pose_folds,
            pose_status,
            [],
        )

        self.assertEqual(
            "insufficient_posture_coverage",
            pose_status,
        )
        self.assertFalse(summary["trainingReadiness"]["ready"])
        self.assertEqual(
            [
                "contextMetadataReady",
                "interleavedProtocolReady",
                "crossClassSessionReady",
                "poseHoldoutReady",
            ],
            summary["trainingReadiness"]["blockingReasons"],
        )
        self.assertEqual(0.0, summary["multiClassSessionRate"])

    def test_interleaved_pose_data_can_satisfy_readiness_gates(self):
        candidates = []
        for posture in ("upright", "supine"):
            for session_index in range(2):
                group_id = posture + "-session-" + str(session_index)
                for class_index, label in enumerate(CLASS_LABELS):
                    candidates.append({
                        "classLabel": label,
                        "groupId": group_id,
                        "platform": "android",
                        "collectorDeviceId": "device-a",
                        "deviceName": "Device A",
                        "sessionId": group_id,
                        "taskId": posture,
                        "bodyPosture": posture,
                        "requestedDevicePose": posture,
                        "grip": "same_hand",
                        "support": "handheld",
                        "collectionProtocol": "interleaved_v2",
                        "sourceSha256": (
                            group_id + "-" + str(class_index)
                        ),
                        "trainingWeight": 1.0,
                    })

        pose_folds, pose_status = training_builder.build_pose_holdout_folds(
            candidates
        )
        summary = training_builder.summary_record(
            candidates,
            candidates,
            2,
            "ready",
            "held_out_session",
            [],
            pose_folds,
            pose_status,
            [],
        )

        self.assertEqual("ready", pose_status)
        self.assertEqual(2, summary["poseHoldoutFoldCount"])
        self.assertEqual(1.0, summary["multiClassSessionRate"])
        self.assertEqual(1.0, summary["knownCollectionContextRate"])
        self.assertEqual(1.0, summary["interleavedSampleRate"])
        self.assertTrue(summary["trainingReadiness"]["ready"])


if __name__ == "__main__":
    unittest.main()
