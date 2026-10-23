import csv
import hashlib
import json
import math
import pickle
import sys
import tempfile
import unittest
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_DIR))
import train_android_finger_four_class_cnn as trainer


class TrainingControlTests(unittest.TestCase):
    def test_checkpoint_selection_prioritizes_worst_class_recall(self):
        balanced = {
            "accuracy": 0.80,
            "balancedAccuracy": 0.80,
            "minRecall": 0.20,
        }
        robust = {
            "accuracy": 0.76,
            "balancedAccuracy": 0.76,
            "minRecall": 0.55,
        }

        self.assertGreater(trainer.checkpoint_score(robust), trainer.checkpoint_score(balanced))

    def schema5_record(
            self,
            sample_id="sample-a",
            motion_state="walking",
            finger_type="thumb",
            touch_strength="light",
            session_id="session-a",
            created_at=100.0,
            lift_index=18):
        return {
            "schemaVersion": 5,
            "id": sample_id,
            "createdAt": created_at,
            "recordStatus": "accepted",
            "collectionSessionID": session_id,
            "windowComplete": True,
            "referenceFrameIndex": 0,
            "downReferenceFrameIndex": 0,
            "liftReferenceFrameIndex": lift_index,
            "startFrameIndex": -15,
            "endFrameIndex": lift_index + 15,
            "preDownFrameCount": 15,
            "postLiftFrameCount": 15,
            "label": {
                "motionState": motion_state,
                "fingerType": finger_type,
                "touchStrength": touch_strength,
            },
            "frames": [
                {"frameIndex": index, "delta": float(index)}
                for index in range(-15, lift_index + 16)
            ],
        }

    def test_sample_matrix_aligns_new_dual_anchor_frames_to_reference_index(self):
        data = {
            "referenceFrameIndex": 10,
            "frames": [
                {"frameIndex": index, "delta": float(index)}
                for index in range(21)
            ],
        }

        matrix = trainer.sample_matrix(data, ["delta"], [-5, 0, 5], "error")

        self.assertEqual([5.0, 10.0, 15.0], [row[0] for row in matrix])

    def test_sample_matrix_for_lift_anchor_uses_lift_reference_index(self):
        data = self.schema5_record(lift_index=12)

        matrix = trainer.sample_matrix_for_anchor(
            data,
            ["delta"],
            [-1, 0, 1],
            "error",
            "lift",
        )

        self.assertEqual([11.0, 12.0, 13.0], [row[0] for row in matrix])

    def test_press_active_window_rejects_requested_frame_after_lift(self):
        data = self.schema5_record(lift_index=8)

        self.assertTrue(trainer.is_press_active_window(data, [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]))
        self.assertFalse(trainer.is_press_active_window(data, [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        self.assertTrue(trainer.is_press_active_window(data, [-5, -4, -3, -2, -1, 0]))

    def test_load_samples_always_filters_windows_that_finish_after_lift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            early_lift_path = root / "early_lift.json"
            complete_path = root / "complete.json"
            early_lift_path.write_text(
                json.dumps(self.schema5_record(sample_id="early", lift_index=8)),
                encoding="utf-8",
            )
            complete_path.write_text(
                json.dumps(self.schema5_record(sample_id="complete", lift_index=9)),
                encoding="utf-8",
            )
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["id", "path", "class_label"])
                writer.writeheader()
                writer.writerow({"id": "early", "path": str(early_lift_path), "class_label": "thumb_light"})
                writer.writerow({"id": "complete", "path": str(complete_path), "class_label": "thumb_light"})

            samples = trainer.load_samples(
                manifest,
                ["delta"],
                [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "error",
                "down",
            )

        self.assertEqual(["complete"], [sample["id"] for sample in samples])

    def test_load_samples_rejects_attempt_to_disable_press_active_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "sample.json"
            sample_path.write_text(
                json.dumps(self.schema5_record(sample_id="sample", lift_index=9)),
                encoding="utf-8",
            )
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["id", "path", "class_label"])
                writer.writeheader()
                writer.writerow({"id": "sample", "path": str(sample_path), "class_label": "thumb_light"})

            with self.assertRaisesRegex(RuntimeError, "press_active_filter_required"):
                trainer.load_samples(
                    manifest,
                    ["delta"],
                    [0, 1],
                    "error",
                    "down",
                    require_press_active=False,
                )

    def test_old_sample_cache_cannot_bypass_press_active_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "early_lift.json"
            sample_path.write_text(
                json.dumps(self.schema5_record(sample_id="early", lift_index=8)),
                encoding="utf-8",
            )
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["id", "path", "class_label"])
                writer.writeheader()
                writer.writerow({"id": "early", "path": str(sample_path), "class_label": "thumb_light"})
            cache_path = root / "samples.pkl"
            old_signature = {
                "featureNames": ["delta"],
                "frameIndices": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "missingFramePolicy": "error",
                "anchorMode": "down",
                "requirePressActive": True,
            }
            with cache_path.open("wb") as file:
                pickle.dump({
                    "signature": old_signature,
                    "samples": [{"id": "early", "path": str(sample_path)}],
                }, file)

            with self.assertRaisesRegex(RuntimeError, "sample_cache_signature_mismatch"):
                trainer.load_or_cache_samples(
                    manifest,
                    ["delta"],
                    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    "error",
                    cache_path,
                    "down",
                    True,
                )

    def test_training_cli_requires_press_active_and_accepts_deterministic_flag(self):
        args = trainer.parse_args([
            "--manifest", "manifest.csv",
            "--output-dir", "out",
            "--deterministic",
        ])

        trainer.validate_cli_args(args)

        self.assertTrue(args.require_press_active)
        self.assertTrue(args.deterministic)
        self.assertEqual("causal_imu11_touch9_v1", args.feature_preset)

    def test_training_cli_rejects_legacy_down_anchor_features(self):
        args = trainer.parse_args([
            "--manifest", "manifest.csv",
            "--output-dir", "out",
            "--feature-preset", "old13",
        ])

        with self.assertRaisesRegex(RuntimeError, "causal_press_feature_preset_required"):
            trainer.validate_cli_args(args)

    def test_training_cli_rejects_noncausal_matrix_transform(self):
        args = trainer.parse_args([
            "--manifest", "manifest.csv",
            "--output-dir", "out",
            "--matrix-transform", "relative_touch",
        ])

        with self.assertRaisesRegex(RuntimeError, "causal_press_matrix_transform_required"):
            trainer.validate_cli_args(args)

    def test_causal_feature_contract_matches_selected_feature_names(self):
        names = trainer.causal_feature_names("causal_imu11_touch9_v1")

        self.assertEqual("causal_imu11_touch9_v1", trainer.feature_contract_for_names(names))

    def test_scaler_metadata_records_causal_feature_contract(self):
        names = trainer.causal_feature_names("causal_imu11_touch9_v1")

        scaler = trainer.scaler_metadata(
            [0.0 for _ in names],
            [1.0 for _ in names],
            names,
            [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        )

        self.assertEqual("causal_imu11_touch9_v1", scaler["featureContract"])

    def test_time_grid_cache_signature_captures_adapter_and_gap_contract(self):
        names = trainer.causal_feature_names("causal_time_grid_imu11_touch9_v2")
        signature = trainer.sample_cache_signature(
            names,
            list(range(-3, 6)),
            "error",
            "down",
            True,
            "causal_time_grid_imu11_touch9_v2",
        )

        self.assertEqual("cross_device_v1", signature["platformAdapterVersion"])
        self.assertEqual("cross_device_v1", signature["coordinateTransformVersion"])
        self.assertEqual(200, signature["gridRateHz"])
        self.assertEqual("linear_sensor_time", signature["interpolation"])
        self.assertEqual("max(2*5ms,4*median_dt)", signature["maxGapRule"])

    def test_time_grid_sample_payload_exposes_quality_metadata(self):
        fixture_path = ANALYSIS_DIR / "tests" / "fixtures" / "cross_device_causal_feature_fixture.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        record = fixture["records"]["android_200hz"]
        record = dict(record)
        record.update({
            "schemaVersion": fixture["schemaVersion"],
            "timeAlignment": fixture["timeAlignment"],
            "sensorMetadata": fixture["sensorMetadata"],
            "touchMetadata": fixture["touchMetadata"],
        })

        payload = trainer.sample_matrix_for_anchor(
            record,
            trainer.causal_feature_names("causal_time_grid_imu11_v2"),
            list(range(-3, 6)),
            "error",
            "down",
            "causal_time_grid_imu11_v2",
            return_quality=True,
        )

        self.assertEqual("accepted", payload["gridQuality"]["status"])
        self.assertEqual(9, len(payload["matrix"]))

    def test_time_grid_contract_requires_nine_time_slots_and_is_serialized_explicitly(self):
        args = trainer.parse_args([
            "--manifest", "manifest.csv",
            "--output-dir", "out",
            "--feature-preset", "causal_time_grid_imu11_touch9_v2",
            "--frame-indices=-3,-2,-1,0,1,2,3,4,5",
        ])

        trainer.validate_cli_args(args)
        names = trainer.causal_feature_names(args.feature_preset)
        scaler = trainer.scaler_metadata(
            [0.0 for _ in names],
            [1.0 for _ in names],
            names,
            [-3, -2, -1, 0, 1, 2, 3, 4, 5],
            args.feature_preset,
        )

        self.assertEqual("causal_time_grid_imu11_touch9_v2", scaler["featureContract"])

    def test_mask6_contract_matches_prebuilt_training_matrix(self):
        args = trainer.parse_args([
            "--manifest", "manifest.csv",
            "--output-dir", "out",
            "--feature-preset", "causal_time_grid_imu11_touch9_mask6_v3",
            "--frame-indices=-3,-2,-1,0,1,2,3,4,5",
        ])

        trainer.validate_cli_args(args)
        names = trainer.causal_feature_names(args.feature_preset)
        scaler = trainer.scaler_metadata(
            [0.0 for _ in names],
            [1.0 for _ in names],
            names,
            [-3, -2, -1, 0, 1, 2, 3, 4, 5],
            args.feature_preset,
        )

        self.assertEqual(26, len(names))
        self.assertEqual("causal_time_grid_imu11_touch9_mask6_v3", scaler["featureContract"])

    def test_time_grid_press_active_filter_uses_physical_lift_timestamp(self):
        data = {
            "schemaVersion": 6,
            "timeAlignment": {
                "downEventElapsedRealtimeNanos": 1_000_000_000,
                "liftEventElapsedRealtimeNanos": 1_020_000_000,
            },
        }

        self.assertFalse(trainer.is_time_grid_press_active(data))
        data["timeAlignment"]["liftEventElapsedRealtimeNanos"] = 1_030_000_000
        self.assertTrue(trainer.is_time_grid_press_active(data))

    def test_schema5_manifest_rows_require_complete_down_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid.json"
            invalid_path = root / "invalid.json"
            valid_path.write_text(
                json.dumps(self.schema5_record(sample_id="valid")),
                encoding="utf-8",
            )
            invalid = self.schema5_record(sample_id="invalid")
            invalid["frames"] = [
                frame for frame in invalid["frames"]
                if frame["frameIndex"] != 9
            ]
            invalid_path.write_text(json.dumps(invalid), encoding="utf-8")

            rows, summary = trainer.build_schema5_manifest_rows(
                root,
                list(range(-5, 10)),
            )

        self.assertEqual(1, len(rows))
        self.assertEqual("valid", rows[0]["id"])
        self.assertEqual("thumb_light", rows[0]["class_label"])
        self.assertEqual("walking", rows[0]["motion_state"])
        self.assertEqual("session-a", rows[0]["raw_session_id"])
        self.assertEqual("q2", rows[0]["duration_bucket"])
        self.assertEqual(2, summary["inputFiles"])
        self.assertEqual(1, summary["acceptedRows"])
        self.assertEqual(1, summary["rejectionReasons"]["missing_required_frame"])

    def test_duration_bucket_uses_planned_lift_quartiles(self):
        self.assertEqual("q1", trainer.duration_bucket(14))
        self.assertEqual("q2", trainer.duration_bucket(18))
        self.assertEqual("q3", trainer.duration_bucket(23))
        self.assertEqual("q4", trainer.duration_bucket(24))
        self.assertEqual("extreme", trainer.duration_bucket(61))

    def test_motion_holdout_folds_keep_sessions_in_one_split(self):
        rows = []
        for motion_index, motion_state in enumerate(trainer.PLANNED_MOTION_STATES):
            for class_index, class_label in enumerate(trainer.CLASS_ORDER):
                rows.append({
                    "id": motion_state + "-" + class_label,
                    "class_label": class_label,
                    "motion_state": motion_state,
                    "raw_session_id": "session-" + str(motion_index) + "-" + str(class_index),
                    "created_at": str(100 + motion_index * 10 + class_index),
                })

        folds = trainer.build_motion_holdout_folds(rows)

        self.assertEqual(set(trainer.PLANNED_MOTION_STATES), set(folds))
        for motion_state, fold_rows in folds.items():
            valid = [row for row in fold_rows if row["split"] == "valid"]
            train = [row for row in fold_rows if row["split"] == "train"]
            self.assertEqual(set(trainer.CLASS_ORDER), {row["class_label"] for row in valid})
            self.assertEqual(set(trainer.CLASS_ORDER), {row["class_label"] for row in train})
            self.assertTrue(all(row["motion_state"] == motion_state for row in valid))
            trainer.validate_no_session_split_leakage(fold_rows)

    def test_temporal_tail_split_keeps_each_class_in_both_splits(self):
        rows = []
        for class_index, class_label in enumerate(trainer.CLASS_ORDER):
            for sample_index in range(10):
                rows.append({
                    "id": class_label + "-" + str(sample_index),
                    "class_label": class_label,
                    "motion_state": "walking",
                    "raw_session_id": "session-" + str(class_index),
                    "created_at": str(100 + sample_index),
                })

        split_rows = trainer.build_temporal_tail_split(rows, 0.2)

        for class_label in trainer.CLASS_ORDER:
            train = [
                row for row in split_rows
                if row["class_label"] == class_label and row["split"] == "train"
            ]
            valid = [
                row for row in split_rows
                if row["class_label"] == class_label and row["split"] == "valid"
            ]
            self.assertEqual(8, len(train))
            self.assertEqual(2, len(valid))
            self.assertLess(max(float(row["created_at"]) for row in train), min(float(row["created_at"]) for row in valid))

    def test_prepare_planned_manifests_writes_master_and_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "raw"
            output_dir = root / "plan"
            dataset_dir.mkdir()
            sample_index = 0
            for motion_state in trainer.PLANNED_MOTION_STATES:
                for class_label in trainer.CLASS_ORDER:
                    finger_type, touch_strength = class_label.split("_", 1)
                    for time_index in range(5):
                        record = self.schema5_record(
                            sample_id="sample-" + str(sample_index),
                            motion_state=motion_state,
                            finger_type=finger_type,
                            touch_strength=touch_strength,
                            session_id="session-" + motion_state + "-" + class_label,
                            created_at=1000 + sample_index,
                        )
                        (dataset_dir / (record["id"] + ".json")).write_text(
                            json.dumps(record),
                            encoding="utf-8",
                        )
                        sample_index += 1

            summary = trainer.prepare_planned_manifests(
                dataset_dir,
                output_dir,
                list(range(-5, 10)),
                0.2,
            )

            with (output_dir / "master_manifest.csv").open(encoding="utf-8") as file:
                master = list(csv.DictReader(file))
            with (output_dir / "folds" / "temporal_tail.csv").open(encoding="utf-8") as file:
                temporal = list(csv.DictReader(file))

        self.assertEqual(60, len(master))
        self.assertEqual(60, len(temporal))
        self.assertEqual(60, summary["dataset"]["acceptedRows"])
        self.assertEqual(4, summary["folds"]["temporal_tail"]["validClassCount"])
        self.assertEqual(3, len(summary["motionHoldoutFolds"]))

    def test_grouped_metrics_report_motion_and_duration(self):
        truths = [0, 0, 1, 1]
        guesses = [0, 1, 1, 1]
        samples = [
            {"motionState": "walking", "durationBucket": "q1"},
            {"motionState": "walking", "durationBucket": "q2"},
            {"motionState": "placed_static", "durationBucket": "q1"},
            {"motionState": "placed_static", "durationBucket": "q2"},
        ]

        metrics = trainer.grouped_metrics_from_predictions(truths, guesses, samples)

        self.assertEqual(2, metrics["byMotionState"]["walking"]["count"])
        self.assertEqual(0.5, metrics["byMotionState"]["walking"]["accuracy"])
        self.assertEqual(2, metrics["byDurationBucket"]["q1"]["count"])

    def test_load_samples_uses_anchor_mode_and_subgroup_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "sample.json"
            data_path.write_text(
                json.dumps(self.schema5_record(lift_index=12)),
                encoding="utf-8",
            )
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=[
                    "id",
                    "path",
                    "class_label",
                    "motion_state",
                    "duration_bucket",
                    "raw_session_id",
                    "split",
                ])
                writer.writeheader()
                writer.writerow({
                    "id": "sample-a",
                    "path": str(data_path),
                    "class_label": "thumb_light",
                    "motion_state": "walking",
                    "duration_bucket": "q1",
                    "raw_session_id": "session-a",
                    "split": "valid",
                })

            samples = trainer.load_samples(
                manifest,
                ["delta"],
                [-1, 0, 1],
                "error",
                "lift",
            )

        self.assertEqual([11.0, 12.0, 13.0], [row[0] for row in samples[0]["matrix"]])
        self.assertEqual("walking", samples[0]["motionState"])
        self.assertEqual("q1", samples[0]["durationBucket"])
        self.assertEqual("session-a", samples[0]["rawSessionId"])

    def test_prebuilt_matrix_loader_joins_by_source_hash_and_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "sample.json"
            source_path.write_text(
                json.dumps(self.schema5_record(sample_id="sample-a")),
                encoding="utf-8",
            )
            source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
            manifest = root / "fold.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=[
                    "id",
                    "path",
                    "class_label",
                    "source_sha256",
                    "raw_session_id",
                    "split",
                ])
                writer.writeheader()
                writer.writerow({
                    "id": "sample-a",
                    "path": str(source_path),
                    "class_label": "thumb_light",
                    "source_sha256": source_sha256,
                    "raw_session_id": "session-a",
                    "split": "valid",
                })
            matrix_path = root / "matrix.jsonl"
            matrix_path.write_text(json.dumps({
                "sourceSha256": source_sha256,
                "sampleId": "sample-a",
                "classLabel": "thumb_light",
                "featureContract": "causal_time_grid_imu11_v2",
                "featureNames": trainer.causal_feature_names("causal_time_grid_imu11_v2"),
                "matrixShape": [9, 11],
                "matrix": [[float(index)] * 11 for index in range(9)],
                "trainingWeight": 1.0,
                "labelStatus": "legacy_unreviewed",
                "reviewSha256": "",
            }) + "\n", encoding="utf-8")

            samples = trainer.load_matrix_samples(
                manifest,
                matrix_path,
                trainer.causal_feature_names("causal_time_grid_imu11_v2"),
                list(range(-3, 6)),
                "causal_time_grid_imu11_v2",
            )

        self.assertEqual(["sample-a"], [sample["id"] for sample in samples])
        self.assertEqual("valid", samples[0]["split"])
        self.assertEqual("session-a", samples[0]["rawSessionId"])
        self.assertEqual([9, 11], [len(samples[0]["matrix"]), len(samples[0]["matrix"][0])])

    def test_prebuilt_matrix_loader_rejects_manifest_source_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "sample.json"
            source_path.write_text(
                json.dumps(self.schema5_record(sample_id="sample-a")),
                encoding="utf-8",
            )
            manifest = root / "fold.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=[
                    "id", "path", "class_label", "source_sha256", "split"
                ])
                writer.writeheader()
                writer.writerow({
                    "id": "sample-a",
                    "path": str(source_path),
                    "class_label": "thumb_light",
                    "source_sha256": "0" * 64,
                    "split": "train",
                })
            matrix_path = root / "matrix.jsonl"
            matrix_path.write_text(json.dumps({
                "sourceSha256": "0" * 64,
                "sampleId": "sample-a",
                "classLabel": "thumb_light",
                "featureContract": "causal_time_grid_imu11_v2",
                "featureNames": trainer.causal_feature_names("causal_time_grid_imu11_v2"),
                "matrixShape": [9, 11],
                "matrix": [[0.0] * 11 for _ in range(9)],
            }) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "matrix_source_sha256_mismatch"):
                trainer.load_matrix_samples(
                    manifest,
                    matrix_path,
                    trainer.causal_feature_names("causal_time_grid_imu11_v2"),
                    list(range(-3, 6)),
                    "causal_time_grid_imu11_v2",
                )

    def test_prepare_cli_does_not_require_manifest(self):
        args = trainer.parse_args([
            "--prepare-planned-manifests",
            "--dataset-dir",
            "raw",
            "--output-dir",
            "out",
            "--anchor-mode",
            "down",
        ])

        trainer.validate_cli_args(args)

        self.assertTrue(args.prepare_planned_manifests)
        self.assertEqual("raw", args.dataset_dir)
        self.assertEqual("down", args.anchor_mode)

    def test_training_cli_requires_manifest(self):
        args = trainer.parse_args(["--output-dir", "out"])

        with self.assertRaisesRegex(RuntimeError, "manifest_required_for_training"):
            trainer.validate_cli_args(args)

    def source_features(self):
        return trainer.preset_features("enhanced_patch21_no_confidence")

    def sample(self, sample_id, created_at, major, minor, collection="collection"):
        values = {name: 0.0 for name in self.source_features()}
        values.update({
            "x_norm": 0.25,
            "y_norm": 0.75,
            "userAcceleration.x": 1.0,
            "contactPatch.cov_11": 4.0,
            "contactPatch.cov_12": 1.0,
            "contactPatch.cov_22": 1.0,
            "contactPatch.sigmaX": 4.0,
            "contactPatch.sigmaY": 2.0,
            "touchNormalization.rawMajorPx": major,
            "touchNormalization.rawMinorPx": minor,
        })
        row = [values[name] for name in self.source_features()]
        return {
            "id": sample_id,
            "path": sample_id + ".json",
            "matrix": [list(row), list(row)],
            "createdAt": created_at,
            "baselineGroup": collection,
            "collectionName": collection,
            "classLabel": "thumb_light",
            "sampleWeight": 1.0,
        }

    def test_early_stopping_waits_for_patience(self):
        self.assertFalse(trainer.should_stop_training(21, 2, 20))
        self.assertTrue(trainer.should_stop_training(22, 2, 20))
        self.assertFalse(trainer.should_stop_training(120, 2, 0))

    def test_all_split_uses_every_sample_for_training(self):
        samples = [
            {"label": 0, "path": "a.json"},
            {"label": 1, "path": "b.json"},
        ]

        train, valid = trainer.split_samples(samples, 0.25, 22, "all")

        self.assertEqual(train, samples)
        self.assertEqual(valid, [])

    def test_session_split_integrity_rejects_overlap(self):
        with self.assertRaisesRegex(RuntimeError, "cross_split_session"):
            trainer.validate_session_split_integrity(
                [{"rawSessionId": "session-a"}],
                [{"rawSessionId": "session-a"}],
            )

    def test_session_split_integrity_rejects_missing_session(self):
        with self.assertRaisesRegex(RuntimeError, "missing_raw_session_id"):
            trainer.validate_session_split_integrity(
                [{"rawSessionId": ""}],
                [{"rawSessionId": "session-b"}],
            )

    def test_no_validation_always_keeps_latest_epoch(self):
        self.assertTrue(trainer.should_update_best_state(False, 0.0, 0.9))
        self.assertTrue(trainer.should_update_best_state(True, 0.9, 0.8))
        self.assertFalse(trainer.should_update_best_state(True, 0.7, 0.8))

    def test_cached_samples_receive_split_from_current_manifest(self):
        samples = [
            {"path": "a.json", "split": "old"},
            {"path": "b.json", "split": "old"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["path", "split"])
                writer.writeheader()
                writer.writerow({"path": "a.json", "split": "train"})
                writer.writerow({"path": "b.json", "split": "valid"})

            trainer.apply_manifest_splits(samples, manifest)

        self.assertEqual(samples[0]["split"], "train")
        self.assertEqual(samples[1]["split"], "valid")

    def test_cached_samples_reject_manifest_with_extra_rows(self):
        samples = [{"path": "a.json", "split": "old"}]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["path", "split"])
                writer.writeheader()
                writer.writerow({"path": "a.json", "split": "train"})
                writer.writerow({"path": "b.json", "split": "valid"})

            with self.assertRaisesRegex(RuntimeError, "sample_cache_manifest_mismatch"):
                trainer.apply_manifest_splits(samples, manifest)

    def test_press_active_cache_allows_manifest_rows_filtered_before_cache(self):
        samples = [{"id": "eligible", "path": "eligible.json", "split": "old"}]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["id", "path", "split"])
                writer.writeheader()
                writer.writerow({"id": "ineligible", "path": "ineligible.json", "split": "train"})
                writer.writerow({"id": "eligible", "path": "eligible.json", "split": "valid"})

            trainer.apply_manifest_splits(samples, manifest, allow_manifest_subset=True)

        self.assertEqual("valid", samples[0]["split"])

    def test_imu_transform_removes_time_and_contact_channels(self):
        samples = [self.sample("a", 1.0, 4.0, 2.0)]

        names = trainer.transform_sample_matrices(
            samples, self.source_features(), "imu11_no_time"
        )

        self.assertEqual(names, self.source_features()[:11])
        self.assertEqual(len(samples[0]["matrix"][0]), 11)
        self.assertEqual(samples[0]["matrix"][0][:3], [0.25, 0.75, 1.0])

    def test_invariant_shape_uses_only_dimensionless_geometry(self):
        samples = [self.sample("a", 1.0, 4.0, 2.0)]

        names = trainer.transform_sample_matrices(
            samples, self.source_features(), "invariant_shape"
        )
        row = samples[0]["matrix"][0]

        self.assertEqual(names[-3:], [
            "invariant.log_axis_ratio",
            "invariant.covariance_correlation",
            "invariant.eccentricity",
        ])
        self.assertAlmostEqual(row[-3], math.log(2.0))
        self.assertAlmostEqual(row[-2], 0.5)
        self.assertAlmostEqual(row[-1], 1.0 / 3.0)

    def test_relative_touch_uses_prior_samples_only(self):
        samples = [
            self.sample("a", 1.0, 2.0, 1.0),
            self.sample("b", 2.0, 4.0, 2.0),
        ]

        names = trainer.transform_sample_matrices(
            samples,
            self.source_features(),
            "relative_touch",
            baseline_window=8,
            baseline_warmup=1,
        )
        first = samples[0]["matrix"][0]
        second = samples[1]["matrix"][0]

        self.assertEqual(names[-4:], [
            "relative.log_major_ratio",
            "relative.log_minor_ratio",
            "relative.log_area_ratio",
            "relative.baseline_readiness",
        ])
        self.assertEqual(first[-4:], [0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(second[-4], math.log(2.0))
        self.assertAlmostEqual(second[-3], math.log(2.0))
        self.assertAlmostEqual(second[-2], math.log(4.0))
        self.assertEqual(second[-1], 1.0)

    def test_baseline_imu_uses_pre_touch_frames_and_keeps_coordinates(self):
        sample = self.sample("a", 1.0, 4.0, 2.0)
        source_names = self.source_features()
        acceleration_index = source_names.index("userAcceleration.x")
        sample["matrix"] = []
        for frame_index, acceleration in enumerate([1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 5.0]):
            values = [0.0] * len(source_names)
            values[source_names.index("x_norm")] = 0.25
            values[source_names.index("y_norm")] = 0.75
            values[acceleration_index] = acceleration
            sample["matrix"].append(values)

        names = trainer.transform_sample_matrices(
            [sample], source_names, "baseline_imu"
        )

        self.assertEqual("x_norm", names[0])
        self.assertEqual("y_norm", names[1])
        self.assertEqual("baseline.userAcceleration.x", names[2])
        self.assertEqual(11, len(names))
        self.assertEqual([0.25, 0.75], sample["matrix"][5][:2])
        self.assertEqual(0.0, sample["matrix"][0][2])
        self.assertAlmostEqual(8.0, sample["matrix"][5][2])
        self.assertAlmostEqual(16.0, sample["matrix"][6][2])

    def test_group_balance_equalizes_collection_class_contribution(self):
        samples = [self.sample("a" + str(index), index, 2.0, 1.0, "large") for index in range(4)]
        samples.append(self.sample("b", 5.0, 2.0, 1.0, "small"))

        trainer.apply_collection_class_balance(samples)

        large_total = sum(sample["sampleWeight"] for sample in samples if sample["collectionName"] == "large")
        small_total = sum(sample["sampleWeight"] for sample in samples if sample["collectionName"] == "small")
        self.assertAlmostEqual(large_total, small_total)
        self.assertGreater(samples[-1]["sampleWeight"], samples[0]["sampleWeight"])

    def test_device_batched_training_and_evaluation_keep_data_on_cpu(self):
        import torch

        model = trainer.FingerFourClassCnn(torch, 2, 8, 4)
        x = torch.randn(8, 2, 8)
        y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        weights = torch.ones(8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        loss = trainer.train_one_epoch_on_device(
            model, x, y, weights, optimizer, loss_fn, 2, torch, torch.device("cpu")
        )
        metrics = trainer.evaluate(model, x, y, torch, batch_size=3)

        self.assertGreaterEqual(loss, 0.0)
        self.assertEqual(metrics["count"], 8)
        self.assertEqual(x.device.type, "cpu")
        self.assertEqual(y.device.type, "cpu")

    def test_evaluate_subgroups_uses_model_predictions(self):
        import torch

        convolution = torch.nn.Conv1d(4, 4, kernel_size=1, bias=False)
        model = torch.nn.Sequential(convolution, torch.nn.Flatten(1))
        with torch.no_grad():
            convolution.weight.zero_()
            for index in range(4):
                convolution.weight[index, index, 0] = 1.0
        x = torch.eye(4).unsqueeze(-1)
        y = torch.tensor([0, 1, 2, 3])
        samples = [
            {"motionState": "walking", "durationBucket": "q1"},
            {"motionState": "walking", "durationBucket": "q2"},
            {"motionState": "placed_static", "durationBucket": "q1"},
            {"motionState": "placed_static", "durationBucket": "q2"},
        ]

        metrics = trainer.evaluate_subgroups(model, x, y, samples, torch, batch_size=2)

        self.assertEqual(1.0, metrics["byMotionState"]["walking"]["accuracy"])
        self.assertEqual(1.0, metrics["byDurationBucket"]["q1"]["accuracy"])


if __name__ == "__main__":
    unittest.main()
