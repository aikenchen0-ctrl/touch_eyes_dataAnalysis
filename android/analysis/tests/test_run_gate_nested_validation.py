import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from android.analysis import gate_threshold_metrics
from android.analysis import run_gate_nested_validation as validation
from android.analysis import stats_mlp_contract as contract


def training_row(index, label, group_id, device_name):
    return {
        "sourceSha256": str(index),
        "classLabel": label,
        "groupId": group_id,
        "deviceName": device_name,
        "platform": "android",
        "collectorDeviceId": device_name,
        "matrix": np.zeros(
            (contract.FRAME_COUNT, len(validation.lightweight.FEATURE_NAMES)),
            dtype=np.float32,
        ),
    }


class GateNestedValidationTests(unittest.TestCase):
    def test_channel_scale_augmentation_is_grouped_and_reproducible(self):
        values = np.ones((2, contract.FRAME_COUNT, 10), dtype=np.float32)
        values[:, :, 0] = 0.25
        values[:, :, 1] = 0.50
        for channel in range(2, 10):
            values[:, :, channel] = float(channel + 1)

        first = validation.apply_channel_scale_augmentation(
            values,
            [{}, {}],
            probability=1.0,
            seed=17,
            maximum_log_scale=0.35,
        )
        second = validation.apply_channel_scale_augmentation(
            values,
            [{}, {}],
            probability=1.0,
            seed=17,
            maximum_log_scale=0.35,
        )

        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.array_equal(first[:, :, :2], values[:, :, :2]))
        for start, stop in ((2, 5), (5, 8), (8, 10)):
            ratios = first[:, :, start:stop] / values[:, :, start:stop]
            self.assertTrue(np.allclose(
                ratios,
                ratios[:, :, :1],
                rtol=0.0,
                atol=1e-6,
            ))

    def test_channel_scale_augmentation_changes_fit_partition_only(self):
        def make_rows(offset):
            rows = []
            for index in range(2):
                row = training_row(
                    offset + index,
                    "thumb_heavy",
                    "session-" + str(offset + index),
                    "device-a",
                )
                matrix = row["matrix"]
                for channel in range(len(matrix[0])):
                    matrix[:, channel] = (
                        np.linspace(0.2, 1.0, matrix.shape[0])
                        * float(channel + 1)
                        + offset * 0.1
                    )
                rows.append(row)
            return rows

        fit_rows = make_rows(1)
        calibration_rows = make_rows(11)
        valid_rows = make_rows(21)
        base = validation.prepare_values(
            fit_rows,
            calibration_rows,
            valid_rows,
            feature_mode="magnitude_only",
            seed=23,
        )
        augmented = validation.prepare_values(
            fit_rows,
            calibration_rows,
            valid_rows,
            feature_mode="magnitude_only",
            augmentation_mode="channel_scale",
            augmentation_probability=1.0,
            augmentation_max_log_scale=0.35,
            seed=23,
            return_metadata=True,
        )

        self.assertFalse(np.array_equal(base[0], augmented[0]))
        self.assertTrue(np.array_equal(base[1], augmented[1]))
        self.assertTrue(np.array_equal(base[2], augmented[2]))
        self.assertTrue(np.isfinite(augmented[0]).all())
        self.assertEqual(
            ["fit"],
            augmented[3]["channelScaleAugmentation"][
                "appliedPartitions"
            ],
        )

    def test_channel_scale_augmentation_rejects_evidence_family(self):
        row = training_row(1, "thumb_heavy", "session-a", "device-a")
        with self.assertRaisesRegex(
                RuntimeError,
                "unsupported_evidence_preprocessing"):
            validation.prepare_values(
                [row],
                [row],
                [row],
                feature_family="evidence",
                augmentation_mode="channel_scale",
            )

    def test_ordinary_click_audit_uses_frozen_threshold(self):
        rows = [
            {"deviceName": "device-a", "bodyPosture": "tilted",
             "motionState": "still", "groupId": "negative-a"},
            {"deviceName": "device-a", "bodyPosture": "tilted",
             "motionState": "still", "groupId": "negative-a"},
        ]
        report = gate_threshold_metrics.fixed_negative_audit_report(
            rows,
            [0.2, 0.8],
            threshold=0.5,
        )
        self.assertEqual("ordinary_touch", report["negativeProxy"])
        self.assertTrue(report["ordinaryClickFprAvailable"])
        self.assertEqual(0.5, report["ordinaryClickAcceptanceRate"])
        self.assertEqual(1, report["ordinaryClickAcceptedCount"])
        self.assertEqual(
            0.5,
            report["groups"]["deviceName"]["device-a"][
                "ordinaryClickAcceptanceRate"
            ],
        )

    def test_ordinary_click_rows_are_evaluation_only_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            row = training_row(1, "ordinary_touch", "negative-a", "device-a")
            row["collectionProtocol"] = "negative_tap_v1"
            serialized = {**row, "matrix": row["matrix"].tolist()}
            path = dataset / "ordinary_touch26.jsonl"
            path.write_text(
                json.dumps(serialized) + "\n" + json.dumps(serialized) + "\n",
                encoding="utf-8",
            )
            rows = validation.load_ordinary_click_rows(
                [dataset],
                [training_row(2, "thumb_light", "train-a", "device-b")],
            )
            self.assertEqual(1, len(rows))
            self.assertTrue(rows[0]["_evaluationOnly"])
            self.assertTrue(rows[0]["_ordinaryClick"])

    def test_ordinary_click_rows_reject_training_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            dataset.mkdir()
            row = training_row(1, "ordinary_touch", "negative-a", "device-a")
            row["collectionProtocol"] = "negative_tap_v1"
            (dataset / "ordinary_touch26.jsonl").write_text(
                json.dumps({**row, "matrix": row["matrix"].tolist()}) + "\n",
                encoding="utf-8",
            )
            training = training_row(1, "thumb_light", "train-a", "device-a")
            with self.assertRaisesRegex(
                    RuntimeError,
                    "ordinary_click_dataset_overlaps_training"):
                validation.load_ordinary_click_rows([dataset], [training])

    def test_ordinary_click_rows_reject_wrong_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            dataset.mkdir()
            row = training_row(1, "ordinary_touch", "negative-a", "device-a")
            row["collectionProtocol"] = "interleaved_v2"
            (dataset / "ordinary_touch26.jsonl").write_text(
                json.dumps({**row, "matrix": row["matrix"].tolist()}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    RuntimeError,
                    "invalid_ordinary_click_collection_protocol"):
                validation.load_ordinary_click_rows([dataset], [])

    def test_ordinary_click_summary_tracks_worst_groups(self):
        runs = []
        for fold_id, acceptance in (
                ("device:a", 0.0), ("device:b", 0.02)):
            metrics = {}
            for target in ("0.01", "0.05"):
                metrics[target] = {
                    "ordinaryClickAcceptanceRate": acceptance,
                    "proofReady": False,
                    "overall": {"negativeCount": 100},
                    "groups": {
                        "deviceName": {
                            fold_id: {
                                "negativeCount": 100,
                                "ordinaryClickAcceptanceRate": acceptance,
                            },
                        },
                        "bodyPosture": {
                            "supine": {
                                "negativeCount": 50,
                                "ordinaryClickAcceptanceRate": acceptance,
                            },
                        },
                        "motionState": {
                            "walking": {
                                "negativeCount": 50,
                                "ordinaryClickAcceptanceRate": acceptance,
                            },
                        },
                    },
                }
            runs.append({
                "foldKind": "device",
                "foldId": fold_id,
                "ordinaryClickAudit": metrics,
            })
        summary = validation.summarize_ordinary_click_audits(runs)
        audit = summary["device"]["0.01"]
        self.assertAlmostEqual(0.01, audit["meanAcceptanceRate"], places=6)
        self.assertEqual(0.02, audit["worstAcceptanceRate"])
        self.assertEqual(0.02, audit["worstDeviceAcceptanceRate"])
        self.assertEqual(0.02, audit["worstPostureAcceptanceRate"])
        self.assertEqual(0.02, audit["worstMotionAcceptanceRate"])
        self.assertFalse(audit["proofReady"])

    def test_feature_modes_preserve_physical_width(self):
        self.assertTrue(validation.feature_mask("full").all())
        self.assertEqual(
            8,
            int(validation.feature_mask("no_absolute_xy").sum()),
        )
        self.assertEqual(
            6,
            int(validation.feature_mask("motion_only").sum()),
        )
        self.assertEqual(
            2,
            int(validation.feature_mask("shape_only").sum()),
        )
        values = np.ones((2, contract.FRAME_COUNT, 10), dtype=np.float32)
        masked = validation.apply_feature_mode(values, "motion_only")
        self.assertEqual(values.shape, masked.shape)
        self.assertTrue(np.all(masked[:, :, 0:2] == 0.0))
        self.assertTrue(np.all(masked[:, :, 8:10] == 0.0))

    def test_magnitude_mode_retains_axis_norms_only(self):
        values = np.zeros(
            (1, contract.FRAME_COUNT, 10),
            dtype=np.float32,
        )
        values[0, :, 0:2] = 7.0
        values[0, :, 2:5] = (3.0, 4.0, 12.0)
        values[0, :, 5:8] = (0.0, 5.0, 12.0)
        values[0, :, 8:10] = 9.0

        result = validation.apply_feature_mode(values, "magnitude_only")

        self.assertEqual(13.0, result[0, 0, 2])
        self.assertEqual(13.0, result[0, 0, 5])
        self.assertTrue(np.all(result[:, :, [0, 1, 3, 4, 6, 7, 8, 9]] == 0.0))

    def test_magnitude_temporal_mode_adds_onset_channels(self):
        values = np.zeros(
            (1, contract.FRAME_COUNT, 10),
            dtype=np.float32,
        )
        values[0, :, 2] = np.arange(
            contract.FRAME_COUNT,
            dtype=np.float32,
        )
        values[0, :, 5] = np.arange(
            contract.FRAME_COUNT,
            dtype=np.float32,
        ) * 0.5

        result = validation.apply_feature_mode(
            values,
            "magnitude_temporal",
        )

        self.assertEqual(values.shape, result.shape)
        self.assertGreater(result[0, -1, 4], 0.0)
        self.assertGreater(result[0, -1, 7], 0.0)
        self.assertGreater(result[0, -1, 8], 0.0)

    def test_magnitude_contact_relative_keeps_absolute_motion(self):
        values = np.zeros(
            (1, contract.FRAME_COUNT, 10),
            dtype=np.float32,
        )
        values[0, :, 2:5] = (3.0, 4.0, 12.0)
        values[0, :, 5:8] = (0.0, 5.0, 12.0)
        values[0, :, 8] = (1.0, 1.1, 1.2, 1.5, 1.8, 1.4, 1.2, 1.0, 0.9)
        values[0, :, 9] = 0.2

        result = validation.apply_feature_mode(
            values,
            "magnitude_contact_relative",
        )

        self.assertEqual(13.0, result[0, 0, 2])
        self.assertEqual(13.0, result[0, 0, 5])
        self.assertTrue(np.all(result[:, :, [0, 1, 3, 4, 6, 7]] == 0.0))

    def test_contact_relative_magnitude_is_scale_invariant(self):
        def build_row(scale):
            row = training_row(1, "thumb_heavy", "group-a", "device-a")
            row["matrix"][:, validation.lightweight.FEATURE_NAMES.index(
                "size"
            )] = np.asarray(
                (1.0, 1.1, 1.2, 1.4, 1.6, 1.3, 1.1, 1.0, 0.9),
                dtype=np.float32,
            ) * scale
            row["matrix"][:, validation.lightweight.FEATURE_NAMES.index(
                "log_axis_ratio"
            )] = (
                np.asarray(
                    (0.1, 0.2, 0.3, 0.5, 0.6, 0.4, 0.2, 0.1, 0.0),
                    dtype=np.float32,
                )
                + scale * 0.01
            )
            row["matrix"][:, validation.lightweight.FEATURE_NAMES.index(
                "size_available"
            )] = 1.0
            row["matrix"][:, validation.lightweight.FEATURE_NAMES.index(
                "major_available"
            )] = 1.0
            row["matrix"][:, validation.lightweight.FEATURE_NAMES.index(
                "minor_available"
            )] = 1.0
            row["matrix"][:, validation.lightweight.FEATURE_NAMES.index(
                "touch_valid"
            )] = 1.0
            return row

        values = np.zeros(
            (2, contract.FRAME_COUNT, 10),
            dtype=np.float32,
        )
        values[0, :, 8] = build_row(1.0)["matrix"][:, 12]
        values[0, :, 9] = build_row(1.0)["matrix"][:, 15]
        values[1, :, 8] = build_row(37.0)["matrix"][:, 12]
        values[1, :, 9] = build_row(37.0)["matrix"][:, 15]

        result = validation.apply_contact_relative_magnitude(
            values,
            (build_row(1.0), build_row(37.0)),
        )

        np.testing.assert_allclose(
            result[0, :, 8],
            result[1, :, 8],
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            result[0, :, 9],
            result[1, :, 9],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_contact_relative_magnitude_masks_missing_shape(self):
        row = training_row(1, "thumb_heavy", "group-a", "device-a")
        values = np.zeros(
            (1, contract.FRAME_COUNT, 10),
            dtype=np.float32,
        )
        values[0, :, 8:10] = 5.0

        result = validation.apply_contact_relative_magnitude(
            values,
            (row,),
        )

        self.assertTrue(np.all(result[0, :, 8:10] == 0.0))

    def test_contact_relative_mode_does_not_return_all_channels(self):
        values = np.zeros(
            (1, contract.FRAME_COUNT, 10),
            dtype=np.float32,
        )
        values[0, :, 0] = 7.0
        values[0, :, 1] = 9.0
        values[0, :, 2:5] = (3.0, 4.0, 12.0)
        values[0, :, 5:8] = (0.0, 5.0, 12.0)
        values[0, :, 8] = 1.5
        values[0, :, 9] = 0.3

        result = validation.apply_feature_mode(
            values,
            "contact_relative_magnitude",
        )

        self.assertTrue(np.all(result[:, :, [0, 1, 3, 4, 6, 7, 9]] == 0.0))
        self.assertTrue(np.all(result[:, :, 2] > 0.0))
        self.assertTrue(np.all(result[:, :, 5] > 0.0))

    def test_historical_contact_uses_only_earlier_touches(self):
        def build_row(index, timestamp, size, group_id="session-a"):
            row = training_row(index, "thumb_heavy", group_id, "device-a")
            row["sourceTimestampMs"] = timestamp
            row["sourceFile"] = str(timestamp) + "_sample.json"
            source = row["matrix"]
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "size"
            )] = size
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "size_available"
            )] = 1.0
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "touch_valid"
            )] = 1.0
            return row

        rows = [
            build_row(1, 1000, 10.0),
            build_row(2, 2000, 10.0),
            build_row(3, 3000, 10.0),
            build_row(4, 4000, 10.0),
            build_row(5, 5000, 40.0),
        ]
        metadata, derived = validation.historical_contact_features(rows)
        self.assertEqual(0.0, metadata[3, 3])
        self.assertEqual(1.0, metadata[4, 3])
        self.assertAlmostEqual(np.log(4.0), derived[0][4], places=6)
        self.assertEqual(0.0, derived[0][0])

        future_changed = [dict(row) for row in rows]
        future_changed[4] = build_row(5, 5000, 400.0)
        future_metadata, future_derived = (
            validation.historical_contact_features(future_changed)
        )
        np.testing.assert_allclose(
            metadata[:4],
            future_metadata[:4],
        )
        np.testing.assert_allclose(
            derived[0][:4],
            future_derived[0][:4],
        )

    def test_historical_contact_is_session_isolated_and_scale_invariant(self):
        def build_row(index, timestamp, size, group_id):
            row = training_row(index, "thumb_light", group_id, "device-a")
            row["sourceTimestampMs"] = timestamp
            source = row["matrix"]
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "size"
            )] = size
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "size_available"
            )] = 1.0
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "touch_valid"
            )] = 1.0
            return row

        rows_a = [
            build_row(index, index * 1000, 10.0, "session-a")
            for index in range(1, 5)
        ]
        rows_a.append(build_row(5, 5000, 20.0, "session-a"))
        rows_b = [
            build_row(index, index * 1000, 10.0, "session-b")
            for index in range(1, 5)
        ]
        rows_b.append(build_row(5, 5000, 40.0, "session-b"))
        rows = rows_a + rows_b

        metadata, derived = validation.historical_contact_features(rows)
        self.assertEqual(1.0, metadata[4, 3])
        self.assertEqual(0.0, metadata[5, 3])
        self.assertAlmostEqual(np.log(2.0), derived[0][4], places=6)
        self.assertEqual(0.0, derived[0][5])

    def test_historical_contact_missing_shape_is_gated(self):
        row = training_row(1, "thumb_heavy", "session-a", "device-a")
        row["sourceTimestampMs"] = 1000
        values = np.ones(
            (1, contract.FRAME_COUNT, 10),
            dtype=np.float32,
        )

        result, derived = validation.apply_historical_contact_relative(
            values,
            (row,),
        )
        metadata = derived[0]

        self.assertEqual(0.0, metadata[0, 3])
        self.assertTrue(np.all(result[0, :, 8:10] == 0.0))

    def test_device_causal_contact_uses_only_earlier_device_touches(self):
        def build_row(index, timestamp, size, group_id, device):
            row = training_row(
                index,
                "thumb_light",
                group_id,
                device,
            )
            row["sourceTimestampMs"] = timestamp
            source = row["matrix"]
            source[:, validation.lightweight.FEATURE_NAMES.index("size")] = (
                size
            )
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "size_available"
            )] = 1.0
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "touch_valid"
            )] = 1.0
            return row

        rows = [
            build_row(
                index,
                index * 1000,
                10.0,
                "session-" + str(index),
                "device-a",
            )
            for index in range(1, 10)
        ]
        rows.append(
            build_row(
                10,
                10000,
                40.0,
                "session-10",
                "device-a",
            )
        )
        features = validation.device_causal_contact_features(rows)

        self.assertEqual(0.0, features[7, 1])
        self.assertEqual(1.0, features[8, 1])
        self.assertAlmostEqual(
            np.log(4.0),
            features[9, 0],
            places=6,
        )

        changed = [dict(row) for row in rows]
        changed[9] = build_row(
            10,
            10000,
            400.0,
            "session-10",
            "device-a",
        )
        changed_features = validation.device_causal_contact_features(
            changed
        )
        np.testing.assert_allclose(
            features[8],
            changed_features[8],
        )

    def test_device_causal_contact_isolates_devices_and_cold_start(self):
        def build_row(index, timestamp, size, group_id, device):
            row = training_row(
                index,
                "thumb_light",
                group_id,
                device,
            )
            row["sourceTimestampMs"] = timestamp
            source = row["matrix"]
            source[:, validation.lightweight.FEATURE_NAMES.index("size")] = (
                size
            )
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "size_available"
            )] = 1.0
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "touch_valid"
            )] = 1.0
            return row

        rows = [
            build_row(index, index * 1000, 10.0, "a-" + str(index), "a")
            for index in range(1, 10)
        ]
        rows.extend([
            build_row(
                index,
                index * 1000,
                100.0,
                "b-" + str(index),
                "b",
            )
            for index in range(1, 10)
        ])
        features = validation.device_causal_contact_features(rows)

        self.assertEqual(1.0, features[8, 1])
        self.assertEqual(0.0, features[9, 1])
        self.assertEqual(1.0, features[17, 1])

    def test_device_causal_contact_replaces_absolute_scale(self):
        row = training_row(1, "thumb_heavy", "session-a", "device-a")
        row["sourceTimestampMs"] = 1000
        values = np.ones(
            (1, contract.FRAME_COUNT, 10),
            dtype=np.float32,
        )
        values[0, :, 8] = 7.0
        values[0, :, 9] = 3.0

        result, features = validation.apply_device_causal_contact(
            values,
            (row,),
        )

        self.assertEqual(0.0, features[0, 1])
        self.assertTrue(np.all(result[0, :, 8] == 0.0))
        self.assertTrue(np.all(result[0, :, 9] == 0.0))

    def test_touch_local_shape_is_scale_invariant_and_missing_aware(self):
        def build_row(scale, available=True):
            row = training_row(1, "thumb_heavy", "group-a", "device-a")
            source = row["matrix"]
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "size"
            )] = np.asarray(
                (1.0, 1.2, 1.5, 2.0, 1.4, 1.0, 0.8, 0.7, 0.7),
                dtype=np.float32,
            ) * scale
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "major_norm"
            )] = np.asarray(
                (1.0, 1.3, 1.7, 2.2, 1.5, 1.1, 0.9, 0.8, 0.8),
                dtype=np.float32,
            ) * scale
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "minor_norm"
            )] = np.asarray(
                (0.8, 0.9, 1.0, 1.2, 1.0, 0.9, 0.8, 0.8, 0.8),
                dtype=np.float32,
            ) * scale
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "size_available"
            )] = float(available)
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "major_available"
            )] = float(available)
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "minor_available"
            )] = float(available)
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "touch_valid"
            )] = 1.0
            return row

        features = validation.touch_local_shape_features(
            (build_row(1.0), build_row(37.0))
        )

        self.assertEqual(
            (2, len(validation.TOUCH_LOCAL_FEATURE_NAMES)),
            features.shape,
        )
        np.testing.assert_allclose(
            features[0],
            features[1],
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertTrue(np.all(features[0] >= 0.0))
        self.assertGreater(features[0, 1], 0.0)

        missing = validation.touch_local_shape_features(
            (build_row(1.0, available=False),)
        )
        size_base = 0
        self.assertEqual(0.0, missing[0, size_base])
        self.assertTrue(np.all(missing[0, size_base + 1:size_base + 6] == 0.0))

    def test_touch_local_modes_append_only_the_requested_feature_family(self):
        row = training_row(1, "thumb_heavy", "group-a", "device-a")
        source = row["matrix"]
        source[:, 2:5] = (3.0, 4.0, 12.0)
        source[:, 5:8] = (0.0, 5.0, 12.0)
        source[:, validation.lightweight.FEATURE_NAMES.index("size")] = 2.0
        source[:, validation.lightweight.FEATURE_NAMES.index("major_norm")] = 2.0
        source[:, validation.lightweight.FEATURE_NAMES.index("minor_norm")] = 1.0
        source[:, 21:25] = 1.0
        source[:, validation.lightweight.FEATURE_NAMES.index("touch_valid")] = 1.0
        values = source[:, contract.SOURCE_FEATURE_INDICES][None, ...]

        shape_only = validation.append_touch_local_shape_features(
            np.zeros((1, contract.STATS_FEATURE_COUNT), dtype=np.float32),
            (row,),
        )
        self.assertEqual(
            (1, contract.STATS_FEATURE_COUNT + len(
                validation.TOUCH_LOCAL_FEATURE_NAMES
            )),
            shape_only.shape,
        )
        self.assertTrue(np.any(shape_only[:, contract.STATS_FEATURE_COUNT:] != 0.0))

        motion_only = validation.apply_feature_mode(
            values,
            "touch_local_motion",
        )
        self.assertTrue(np.all(motion_only[:, :, [0, 1, 3, 4, 6, 7, 8, 9]] == 0.0))

    def test_causal_motion_is_future_independent_and_scale_invariant(self):
        def build_row(scale, future_value=10.0):
            row = training_row(1, "thumb_heavy", "group-a", "device-a")
            source = row["matrix"]
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "userAcceleration.x"
            )] = np.asarray(
                (0.1, 0.2, 0.3, 1.0, 2.0, 3.0, 2.0, 1.0, 0.5),
                dtype=np.float32,
            ) * scale
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "userAcceleration.y"
            )] = np.asarray(
                (0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 1.0, 0.5, 0.2),
                dtype=np.float32,
            ) * scale
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "userAcceleration.z"
            )] = 0.0
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "rotationRate.x"
            )] = np.asarray(
                (0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05),
                dtype=np.float32,
            ) * scale
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "rotationRate.y"
            )] = 0.0
            source[:, validation.lightweight.FEATURE_NAMES.index(
                "rotationRate.z"
            )] = 0.0
            source[-1, validation.lightweight.FEATURE_NAMES.index(
                "userAcceleration.x"
            )] = future_value
            return row

        features = validation.causal_motion_features(
            (build_row(1.0), build_row(25.0))
        )
        self.assertEqual(
            (2, len(validation.CAUSAL_MOTION_FEATURE_NAMES)),
            features.shape,
        )
        self.assertEqual(1.0, features[0, 2])
        # 中文注释：当前帧绝对值可随传感器偏移变化，比值列必须一致。
        np.testing.assert_allclose(
            features[0, [10, 11, 12]],
            features[1, [10, 11, 12]],
            rtol=1e-5,
            atol=1e-5,
        )

        changed = build_row(1.0, future_value=1000.0)
        unchanged = validation.causal_motion_features((build_row(1.0),))
        future_changed = validation.causal_motion_features((changed,))
        # 中文注释：未来帧改变后，旋转特征保持不变。
        np.testing.assert_allclose(unchanged[0, 8:], future_changed[0, 8:])

    def test_causal_motion_missing_imu_is_fail_closed(self):
        row = training_row(1, "thumb_heavy", "group-a", "device-a")
        row["imuAvailable"] = False
        row["matrix"][:, 2:8] = 5.0
        features = validation.causal_motion_features((row,))
        self.assertTrue(np.all(features == 0.0))

    def test_causal_motion_modes_keep_history_relative_contact_only(self):
        values = np.ones(
            (1, contract.FRAME_COUNT, 10),
            dtype=np.float32,
        )
        no_shape = validation.apply_feature_mode(
            values,
            "historical_contact_causal_motion",
        )
        with_shape = validation.apply_feature_mode(
            values,
            "historical_contact_causal_motion_shape",
        )
        self.assertTrue(np.all(no_shape[:, :, 8] == 1.0))
        self.assertTrue(np.all(no_shape[:, :, 9] == 0.0))
        self.assertTrue(np.all(no_shape[:, :, [0, 1, 2, 3, 4, 5, 6, 7]] == 0.0))
        self.assertTrue(np.all(with_shape[:, :, [8, 9]] == 1.0))

    def test_historical_local_shape_keeps_only_history_relative_size(self):
        values = np.ones(
            (1, contract.FRAME_COUNT, 10),
            dtype=np.float32,
        )

        result = validation.apply_feature_mode(
            values,
            "historical_contact_local_shape",
        )

        self.assertTrue(np.all(result[:, :, 8] == 1.0))
        self.assertTrue(np.all(
            result[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 9]] == 0.0
        ))

    def test_historical_motion_ablation_removes_absolute_axes_and_shape(self):
        values = np.ones(
            (1, contract.FRAME_COUNT, 10),
            dtype=np.float32,
        )
        values[0, :, 2:5] = (3.0, 4.0, 12.0)
        values[0, :, 5:8] = (0.0, 5.0, 12.0)

        no_shape = validation.apply_feature_mode(
            values,
            "historical_contact_motion_no_shape",
        )
        with_shape = validation.apply_feature_mode(
            values,
            "historical_contact_motion_shape",
        )

        self.assertEqual(13.0, no_shape[0, 0, 2])
        self.assertEqual(13.0, no_shape[0, 0, 5])
        self.assertEqual(1.0, no_shape[0, 0, 8])
        self.assertTrue(np.all(no_shape[:, :, 9] == 0.0))
        self.assertTrue(np.all(with_shape[:, :, 9] == 1.0))

    def test_collection_batch_uses_source_root_and_dataset_fallback(self):
        self.assertEqual(
            "batch-a",
            validation.collection_batch_key({
                "sourceFile": "batch-a/session/sample.json",
            }),
        )
        self.assertEqual(
            "batch-b",
            validation.collection_batch_key({
                "sourceFile": "batch-b\\session\\sample.json",
            }),
        )
        self.assertEqual(
            "parent/dataset-root",
            validation.collection_batch_key({
                "sourceFile": "sample.json",
                "_datasetId": "parent/dataset-root",
            }),
        )

    def test_collection_batch_diagnostics_keep_audit_fields(self):
        matrix = np.zeros(
            (contract.FRAME_COUNT, len(validation.lightweight.FEATURE_NAMES)),
            dtype=np.float32,
        )
        matrix[:, 2] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        matrix[:, 5] = 2.0
        matrix[:, 22:25] = 1.0
        rows = [{
            "sourceFile": "batch-a/session/sample.json",
            "deviceName": "device-a",
            "groupId": "session-a",
            "classLabel": "thumb_heavy",
            "motionState": "walking",
            "bodyPosture": "tilted",
            "matrix": matrix,
        }]

        diagnostics = validation.collection_batch_diagnostics(rows)
        entry = diagnostics["batch-a|device-a"]

        self.assertEqual(1, entry["sampleCount"])
        self.assertEqual(1, entry["sessionCount"])
        self.assertEqual(1, entry["classLabelCounts"]["thumb_heavy"])
        self.assertEqual(1, entry["motionStateCounts"]["walking"])
        self.assertEqual(1, entry["bodyPostureCounts"]["tilted"])
        self.assertEqual(0, entry["missingShapeCount"])
        self.assertIn("accelerationJerkMean", entry["signalMedians"])

    def test_data_contract_detects_session_label_confounding(self):
        rows = []
        for session_index in range(6):
            label = "thumb_light" if session_index % 2 == 0 else "thumb_heavy"
            for offset in range(10):
                row = training_row(
                    session_index * 20 + offset + 1,
                    label,
                    "session-" + str(session_index),
                    "device-a",
                )
                row["bodyPosture"] = "tilted"
                row["motionState"] = "handheld_static"
                row["collectionProtocol"] = "unknown"
                rows.append(row)

        diagnostics = validation.data_contract_diagnostics(rows)

        self.assertEqual(0, diagnostics["multiStrengthSessionCount"])
        self.assertEqual(0, diagnostics["interleavedSessionCount"])
        self.assertGreater(
            diagnostics[
                "sessionStrengthNormalizedMutualInformation"
            ],
            0.60,
        )
        self.assertIn(
            "session_label_confounding",
            diagnostics["blockers"],
        )
        self.assertIn(
            "no_interleaved_sessions",
            diagnostics["blockers"],
        )
        self.assertFalse(diagnostics["proofReady"])

    def test_data_contract_accepts_paired_interleaved_sessions(self):
        rows = []
        devices = ("device-a", "device-b", "device-c")
        postures = ("tilted", "supine")
        motions = ("handheld_static", "walking")
        for device_index, device in enumerate(devices):
            for posture_index, posture in enumerate(postures):
                for motion_index, motion in enumerate(motions):
                    for label_index, label in enumerate((
                        "thumb_light",
                        "thumb_heavy",
                    )):
                        for repeat in range(
                            validation.MIN_GROUP_POSITIVES
                        ):
                            row = training_row(
                                (
                                    device_index * 10000
                                    + posture_index * 1000
                                    + motion_index * 100
                                    + label_index * 50
                                    + repeat
                                ),
                                label,
                                "session-{}-{}-{}".format(
                                    device_index,
                                    posture_index,
                                    motion_index,
                                ),
                                device,
                            )
                            row["bodyPosture"] = posture
                            row["motionState"] = motion
                            row["collectionProtocol"] = "interleaved_v2"
                            rows.append(row)

        diagnostics = validation.data_contract_diagnostics(rows)

        self.assertEqual(
            len(devices) * len(postures) * len(motions),
            diagnostics["multiStrengthSessionCount"],
        )
        self.assertEqual(
            len(devices) * len(postures) * len(motions),
            diagnostics["interleavedSessionCount"],
        )
        self.assertEqual(1.0, diagnostics["interleavedRowFraction"])
        self.assertGreaterEqual(
            diagnostics["pairedCoverage"][
                "device_posture_motion"
            ]["pairedGroupCount"],
            2,
        )
        self.assertTrue(diagnostics["proofReady"])
        self.assertEqual([], diagnostics["blockers"])

    def test_device_balanced_weights_equalize_physical_devices(self):
        rows = [
            {"groupId": "a1", "deviceName": "a", "platform": "android"},
            {"groupId": "a2", "deviceName": "a", "platform": "android"},
            {"groupId": "b1", "deviceName": "b", "platform": "android"},
            {"groupId": "b2", "deviceName": "b", "platform": "android"},
        ]
        weights = validation.balanced_sample_weights(
            rows,
            [0, 0, 1, 1],
            "device_balanced",
        )
        self.assertAlmostEqual(
            float(weights[:2].sum()),
            float(weights[2:].sum()),
            places=6,
        )
        self.assertAlmostEqual(
            0.5,
            float(weights[:2].sum()),
            places=6,
        )

    def test_motion_balanced_weights_equalize_device_motion_cells(self):
        rows = [
            {"groupId": "a1", "deviceName": "a", "platform": "android",
             "motionState": "handheld_static"},
            {"groupId": "a2", "deviceName": "a", "platform": "android",
             "motionState": "handheld_static"},
            {"groupId": "a3", "deviceName": "a", "platform": "android",
             "motionState": "walking"},
            {"groupId": "b1", "deviceName": "b", "platform": "android",
             "motionState": "handheld_static"},
        ]
        weights = validation.balanced_sample_weights(
            rows,
            [0, 0, 1, 1],
            "device_motion_balanced",
        )
        self.assertAlmostEqual(1.0, float(weights.sum()), places=6)
        self.assertTrue(np.allclose(weights, 0.25))

    def test_motion_domain_keeps_unknown_metadata_distinct(self):
        self.assertEqual(
            "still",
            validation.normalized_motion_domain({
                "motionState": "handheld_static",
            }),
        )
        self.assertIsNone(validation.normalized_motion_domain({
            "motionState": "unknown",
        }))

    def test_pretouch_centering_uses_only_past_frames(self):
        values = np.zeros(
            (1, contract.FRAME_COUNT, len(contract.PHYSICAL_FEATURE_NAMES)),
            dtype=np.float32,
        )
        values[0, :3, 2:8] = np.arange(3)[:, None]
        values[0, 3:, 2:8] = 100.0

        centered = validation.apply_imu_mode(values, "pretouch_centered")
        expected = np.arange(3)[:, None] - 1.0

        self.assertTrue(np.allclose(
            centered[0, :3, 2:8],
            np.broadcast_to(expected, (3, 6)),
        ))
        self.assertTrue(np.allclose(centered[0, 3:, 2:8], 99.0))

    def test_guarded_pretouch_keeps_stable_three_frame_baseline(self):
        values = np.zeros(
            (1, contract.FRAME_COUNT, len(contract.PHYSICAL_FEATURE_NAMES)),
            dtype=np.float32,
        )
        values[0, :3, 2:8] = 2.0
        values[0, 3:, 2:8] = 8.0

        guarded = validation.apply_imu_mode(
            values,
            "guarded_pretouch_centered",
        )
        control = validation.apply_imu_mode(
            values,
            "pretouch_centered",
        )

        self.assertTrue(np.allclose(guarded, control))

    def test_guarded_pretouch_rejects_rising_contact_onset(self):
        values = np.zeros(
            (1, contract.FRAME_COUNT, len(contract.PHYSICAL_FEATURE_NAMES)),
            dtype=np.float32,
        )
        values[0, :3, 2:8] = np.asarray((1.0, 3.0, 6.0))[:, None]
        values[0, 3:, 2:8] = 10.0

        guarded = validation.apply_imu_mode(
            values,
            "guarded_pretouch_centered",
        )
        earliest = validation.apply_imu_mode(
            values,
            "earliest_pretouch_centered",
        )

        self.assertTrue(np.allclose(guarded, earliest))
        self.assertTrue(np.allclose(guarded[0, 0, 2:8], 0.0))
        self.assertTrue(np.allclose(guarded[0, 2, 2:8], 5.0))

    def test_guarded_pretouch_does_not_inspect_post_touch_frames(self):
        values = np.zeros(
            (2, contract.FRAME_COUNT, len(contract.PHYSICAL_FEATURE_NAMES)),
            dtype=np.float32,
        )
        values[:, :3, 2:8] = np.asarray((1.0, 3.0, 6.0))[:, None]
        values[0, 3:, 2:8] = 10.0
        values[1, 3:, 2:8] = 100.0

        first = validation.apply_imu_mode(
            values[:1],
            "guarded_pretouch_centered",
        )
        second = validation.apply_imu_mode(
            values[1:],
            "guarded_pretouch_centered",
        )

        self.assertTrue(np.allclose(first[0, :3], second[0, :3]))

    def test_gravity_guarded_modes_enable_alignment(self):
        self.assertTrue(validation.uses_gravity_alignment(
            "gravity_aligned_guarded_pretouch_centered",
        ))
        self.assertTrue(validation.uses_gravity_alignment(
            "gravity_aligned_earliest_pretouch_centered",
        ))

    def test_gravity_rotation_maps_source_direction_to_canonical_axis(self):
        rotation = validation.gravity_rotation_to_canonical((0.0, 9.8, 0.0))
        mapped = rotation @ np.asarray((0.0, 1.0, 0.0))
        self.assertTrue(np.allclose(mapped, (0.0, 0.0, 1.0)))
        self.assertTrue(np.allclose(
            rotation @ rotation.T,
            np.eye(3),
        ))

    def test_gravity_alignment_is_invariant_to_device_coordinate_rotation(self):
        source_gravity = np.asarray((0.0, 0.0, 9.8), dtype=np.float32)
        source_acceleration = np.asarray((1.0, 2.0, 3.0), dtype=np.float32)
        source_rotation = np.asarray((0.1, -0.2, 0.3), dtype=np.float32)
        device_rotation = np.asarray((
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, -1.0, 0.0),
        ), dtype=np.float32)
        values = np.zeros(
            (2, contract.FRAME_COUNT, 10),
            dtype=np.float32,
        )
        rows = []
        for index, rotation in enumerate((np.eye(3), device_rotation)):
            matrix = np.zeros((contract.FRAME_COUNT, 26), dtype=np.float32)
            matrix[:3, 8:11] = rotation @ source_gravity
            values[index, :, 2:5] = rotation @ source_acceleration
            values[index, :, 5:8] = rotation @ source_rotation
            rows.append({"matrix": matrix})

        aligned = validation.apply_gravity_alignment(values, rows)
        self.assertTrue(np.allclose(aligned[0], aligned[1], atol=1e-5))
        self.assertTrue(np.allclose(
            aligned[0, 0, 2:5],
            source_acceleration,
            atol=1e-5,
        ))
        self.assertTrue(np.allclose(
            aligned[0, 0, 5:8],
            source_rotation,
            atol=1e-5,
        ))

    def test_gravity_alignment_ignores_post_touch_gravity(self):
        matrix = np.zeros((contract.FRAME_COUNT, 26), dtype=np.float32)
        matrix[:3, 8:11] = (0.0, 0.0, 9.8)
        changed = matrix.copy()
        changed[3:, 8:11] = (9.8, 0.0, 0.0)
        values = np.ones((2, contract.FRAME_COUNT, 10), dtype=np.float32)

        aligned = validation.apply_gravity_alignment(
            values,
            [{"matrix": matrix}, {"matrix": changed}],
        )
        self.assertTrue(np.allclose(aligned[0], aligned[1]))

    def test_gravity_alignment_requires_pretouch_gravity(self):
        values = np.ones((1, contract.FRAME_COUNT, 10), dtype=np.float32)
        with self.assertRaisesRegex(
                RuntimeError,
                "gravity_alignment_missing_valid_pretouch"):
            validation.apply_gravity_alignment(
                values,
                [{"matrix": np.zeros((contract.FRAME_COUNT, 26))}],
            )

    def test_gravity_alignment_coverage_reports_missing_rows(self):
        valid = np.zeros((contract.FRAME_COUNT, 26), dtype=np.float32)
        valid[:, 8:11] = (0.0, 0.0, 9.8)
        coverage = validation.gravity_alignment_coverage([
            {"matrix": valid},
            {"matrix": np.zeros_like(valid)},
        ])
        self.assertEqual(1, coverage["availableCount"])
        self.assertEqual(1, coverage["unavailableCount"])
        self.assertAlmostEqual(0.5, coverage["coverage"])

    def test_device_threshold_replay_uses_only_validation_negatives(self):
        scores = np.asarray([0.10, 0.20, 0.30, 0.80, 0.90])
        truths = np.asarray([0, 0, 0, 1, 1])
        diagnostics = validation.target_device_threshold_diagnostics(
            truths,
            scores,
            target_fpr=0.34,
        )
        self.assertEqual(
            "target_device_negative_replay",
            diagnostics["thresholdPolicy"],
        )
        self.assertEqual(0.34, diagnostics["targetFalsePositiveRate"])
        self.assertEqual(1, diagnostics["overall"]["falsePositiveCount"])
        self.assertEqual(
            1.0,
            diagnostics["overall"]["heavyRecall"],
        )

    def test_target_calibration_uses_complete_negative_sessions(self):
        rows = [
            {"groupId": "negative-a", "deviceName": "a"},
            {"groupId": "negative-a", "deviceName": "a"},
            {"groupId": "negative-a", "deviceName": "a"},
            {"groupId": "negative-b", "deviceName": "a"},
            {"groupId": "negative-b", "deviceName": "a"},
            {"groupId": "positive-a", "deviceName": "a"},
        ]
        sessions = validation.select_target_calibration_sessions(
            rows,
            [0, 0, 0, 0, 0, 1],
            count=2,
            seed=7,
        )
        self.assertEqual(1, len(sessions))
        self.assertIn(sessions[0], {"negative-a", "negative-b"})
        reports = validation.target_calibrated_reports(
            rows,
            [0, 0, 0, 0, 0, 1],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.9],
            target_calibration_count=2,
            seed=7,
            repetitions=1,
        )
        report = reports["0.01"]["perRepetition"][0]
        self.assertEqual(sessions, report["calibrationSessions"])
        self.assertEqual(
            {"negative-a", "negative-b", "positive-a"}
            - set(sessions),
            set(report["heldOutSessions"]),
        )
        self.assertTrue(report["sessionDisjoint"])
        self.assertFalse(
            set(report["calibrationSessions"])
            & set(report["heldOutSessions"])
        )
        calibration_counts = {"negative-a": 3, "negative-b": 2}
        self.assertEqual(
            calibration_counts[sessions[0]],
            report["targetCalibrationNegativeCount"],
        )
        self.assertEqual(
            5 - calibration_counts[sessions[0]],
            report["heldOutNegativeCount"],
        )

    def test_target_calibration_preserves_held_out_negatives(self):
        rows = [
            {"groupId": "negative-a"},
            {"groupId": "negative-a"},
            {"groupId": "negative-b"},
            {"groupId": "positive-a"},
        ]
        reports = validation.target_calibrated_reports(
            rows,
            [0, 0, 0, 1],
            [0.1, 0.2, 0.3, 0.9],
            target_calibration_count=32,
            seed=7,
            repetitions=1,
        )
        report = reports["0.01"]["perRepetition"][0]
        self.assertGreaterEqual(
            report["targetCalibrationNegativeCount"],
            1,
        )
        self.assertGreaterEqual(
            report["heldOutNegativeCount"],
            1,
        )
        self.assertEqual(
            3,
            report["targetCalibrationNegativeCount"]
            + report["heldOutNegativeCount"],
        )
        self.assertTrue(
            report["sessionDisjoint"]
        )
        self.assertFalse(reports["0.01"]["proofReady"])
        self.assertEqual(
            "descriptive_only",
            report["calibrationEvidenceStatus"],
        )

    def test_target_calibration_keeps_whole_sessions_disjoint(self):
        rows = [
            {
                "groupId": "calibration-session",
                "deviceName": "device-a",
                "bodyPosture": "tilted",
                "motionState": "still",
            },
            {
                "groupId": "calibration-session",
                "deviceName": "device-a",
                "bodyPosture": "tilted",
                "motionState": "still",
            },
            {
                "groupId": "held-out-session",
                "deviceName": "device-a",
                "bodyPosture": "tilted",
                "motionState": "walking",
            },
            {
                "groupId": "held-out-session",
                "deviceName": "device-a",
                "bodyPosture": "tilted",
                "motionState": "walking",
            },
        ]
        reports = validation.target_calibrated_reports(
            rows,
            [0, 1, 0, 1],
            [0.1, 0.2, 0.3, 0.9],
            target_calibration_count=1,
            seed=7,
            repetitions=1,
        )
        report = reports["0.01"]["perRepetition"][0]
        calibration_sessions = set(report["calibrationSessions"])
        held_out_sessions = set(report["heldOutSessions"])
        self.assertTrue(report["sessionDisjoint"])
        self.assertFalse(calibration_sessions & held_out_sessions)
        self.assertEqual(1, report["calibrationSessionCount"])
        self.assertEqual(1, report["heldOutSessionCount"])
        self.assertEqual(2, report["calibrationRowCount"])
        self.assertEqual(2, report["heldOutRowCount"])
        self.assertEqual(1, report["heldOutNegativeCount"])
        self.assertEqual(["device-a"], report["calibrationDeviceNames"])
        self.assertEqual(["device-a"], report["heldOutDeviceNames"])

    def test_calibration_repetitions_use_independent_subsets(self):
        rows = [
            {"groupId": "negative-" + str(index % 3)}
            for index in range(30)
        ] + [{"groupId": "positive"}]
        sessions = validation.select_target_calibration_sessions(
            rows,
            [0] * 30 + [1],
            count=10,
            seed=7,
        )
        self.assertEqual(1, len(sessions))
        self.assertIn(sessions[0], {"negative-0", "negative-1", "negative-2"})
        selections = {
            tuple(validation.select_target_calibration_sessions(
                rows,
                [0] * 30 + [1],
                count=10,
                seed=seed,
            ))
            for seed in range(7, 27)
        }
        self.assertGreater(len(selections), 1)

    def test_target_calibration_proof_requires_both_splits(self):
        rows = [
            {"groupId": "negative-" + str(index // 100)}
            for index in range(1000)
        ] + [
            {"groupId": "positive-" + str(index // 100)}
            for index in range(100)
        ]
        labels = [0] * 1000 + [1] * 100
        scores = [0.1] * 1000 + [0.9] * 100

        reports = validation.target_calibrated_reports(
            rows,
            labels,
            scores,
            target_calibration_count=500,
            seed=7,
            repetitions=1,
        )

        self.assertTrue(reports["0.01"]["proofReady"])
        self.assertEqual(
            "proof_ready",
            reports["0.01"]["perRepetition"][0][
                "calibrationEvidenceStatus"
            ],
        )

    def test_target_calibration_proof_requires_independent_sessions(self):
        rows = [
            {"groupId": "negative-a"}
            for _ in range(500)
        ] + [
            {"groupId": "negative-b"}
            for _ in range(500)
        ] + [
            {"groupId": "positive-a"}
            for _ in range(100)
        ]
        labels = [0] * 1000 + [1] * 100
        scores = [0.1] * 500 + [0.2] * 500 + [0.9] * 100
        reports = validation.target_calibrated_reports(
            rows,
            labels,
            scores,
            target_calibration_count=500,
            seed=7,
            repetitions=1,
        )
        report = reports["0.01"]["perRepetition"][0]
        self.assertEqual(500, report["heldOutNegativeCount"])
        self.assertEqual(1, report["calibrationSessionCount"])
        self.assertEqual(1, report["heldOutNegativeSessionCount"])
        self.assertTrue(report["sessionDisjoint"])
        self.assertFalse(report["proofReady"])
        self.assertEqual("descriptive_only", report["calibrationEvidenceStatus"])

    def test_evaluation_only_dataset_rejects_training_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            training_dir = Path(directory) / "training"
            evaluation_dir = Path(directory) / "evaluation"
            training_dir.mkdir()
            evaluation_dir.mkdir()
            row = training_row(1, "thumb_light", "group-a", "device-a")
            serializable_row = {
                **row,
                "matrix": row["matrix"].tolist(),
            }
            for root in (training_dir, evaluation_dir):
                (root / "training_touch26.jsonl").write_text(
                    json.dumps(serializable_row) + "\n",
                    encoding="utf-8",
                )
            training_rows = validation.load_rows([training_dir])
            with self.assertRaisesRegex(
                    RuntimeError,
                    "evaluation_only_dataset_overlaps_training"):
                validation.load_evaluation_only_rows(
                    [evaluation_dir],
                    training_rows,
                )

    def test_evaluation_only_dataset_rejects_shared_session(self):
        training_rows = [
            training_row(1, "thumb_light", "shared-session", "device-a"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            evaluation_dir = Path(directory)
            evaluation_row = training_row(
                2,
                "thumb_heavy",
                "shared-session",
                "device-a",
            )
            (evaluation_dir / "training_touch26.jsonl").write_text(
                json.dumps({
                    **evaluation_row,
                    "matrix": evaluation_row["matrix"].tolist(),
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    RuntimeError,
                    "evaluation_only_session_overlaps_training"):
                validation.load_evaluation_only_rows(
                    [evaluation_dir],
                    training_rows,
                )

    def test_evaluation_values_freeze_training_statistics(self):
        fit_rows = [
            training_row(1, "thumb_light", "fit-a", "device-a"),
            training_row(2, "thumb_heavy", "fit-b", "device-a"),
        ]
        fit_rows[0]["matrix"][:] = 1.0
        fit_rows[1]["matrix"][:] = 3.0
        evaluation_rows = validation.build_evaluation_only_rows([
            training_row(3, "index_heavy", "evaluation-a", "device-b"),
        ])
        evaluation_rows[0]["matrix"][:] = 1000.0
        values = validation.prepare_evaluation_values(
            fit_rows,
            evaluation_rows,
        )
        self.assertEqual((1, contract.STATS_FEATURE_COUNT), values.shape)
        self.assertTrue(np.all(values <= contract.STATS_CLIP_MAX))
        self.assertTrue(np.all(values >= contract.STATS_CLIP_MIN))

    def test_early_evidence_uses_fit_rows_for_normalization_only(self):
        fit_rows = [
            training_row(1, "thumb_light", "fit-a", "device-a"),
            training_row(2, "thumb_heavy", "fit-b", "device-a"),
        ]
        fit_evidence = np.zeros(
            (2, len(validation.early_press_evidence.FEATURE_NAMES)),
            dtype=np.float32,
        )
        fit_evidence[:, 0] = (0.0, 2.0)
        fit_evidence[:, 20] = 1.0
        fit_evidence[:, -1] = 1.0
        for row, evidence in zip(fit_rows, fit_evidence):
            row["earlyEvidence"] = evidence
        calibration_rows = [
            training_row(3, "thumb_light", "cal-a", "device-a"),
        ]
        calibration_rows[0]["earlyEvidence"] = fit_evidence[0].copy()
        evaluation_rows = validation.build_evaluation_only_rows([
            training_row(4, "index_heavy", "evaluation-a", "device-b"),
        ])
        evaluation_rows[0]["earlyEvidence"] = fit_evidence[1].copy()

        fit_values, calibration_values, valid_values, metadata = (
            validation.prepare_values(
            fit_rows=fit_rows,
            calibration_rows=calibration_rows,
            valid_rows=fit_rows,
            early_evidence_mode="shape",
            return_metadata=True,
            )
        )
        changed_rows = [
            {**row, "earlyEvidence": row["earlyEvidence"].copy()}
            for row in evaluation_rows
        ]
        changed_rows.append({
            **evaluation_rows[0],
            "sourceSha256": "5",
            "groupId": "evaluation-b",
            "earlyEvidence": evaluation_rows[0]["earlyEvidence"].copy(),
        })
        changed_rows[-1]["earlyEvidence"][0] = 1_000_000.0
        first = validation.prepare_evaluation_values(
            fit_rows,
            evaluation_rows,
            early_evidence_mode="shape",
        )
        second = validation.prepare_evaluation_values(
            fit_rows,
            changed_rows,
            early_evidence_mode="shape",
        )

        self.assertEqual(
            contract.STATS_FEATURE_COUNT
            + 21,
            fit_values.shape[1],
        )
        self.assertEqual(2, metadata["earlyPressEvidence"]["fit"]["availableCount"])
        self.assertEqual(
            fit_values.shape[1],
            calibration_values.shape[1],
        )
        self.assertEqual(
            fit_values.shape[1],
            valid_values.shape[1],
        )
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.all(first <= contract.STATS_CLIP_MAX))

    def test_device_threshold_replay_rejects_missing_negatives(self):
        with self.assertRaisesRegex(
                RuntimeError,
                "missing_target_device_negative_proxy"):
            validation.target_device_threshold_diagnostics(
                [1, 1],
                [0.2, 0.8],
                target_fpr=0.01,
            )

    def test_adaptive_delta_preserves_original_input_count(self):
        residuals = np.zeros(
            (2, contract.FRAME_COUNT, len(contract.PHYSICAL_FEATURE_NAMES)),
            dtype=np.float32,
        )
        residuals[:, :, 2:8] = (
            np.arange(contract.FRAME_COUNT, dtype=np.float32)[None, :, None]
        )
        residuals[1, :, 0:2] = 100.0
        values = validation.apply_pretouch_delta(residuals)
        self.assertEqual(residuals.shape, values.shape)
        self.assertTrue(np.all(values[:, 0] == 0.0))
        self.assertTrue(np.allclose(values[:, 1:7, 2:8], 1.0))
        self.assertTrue(np.allclose(values[:, 7:, 2:8], 1.0))
        self.assertTrue(np.all(values[:, :, 8:] == 0.0))

    def test_external_threshold_does_not_refit_on_validation(self):
        rows = [
            {"deviceName": "a", "bodyPosture": "supine",
             "motionState": "still", "groupId": "s1"},
            {"deviceName": "a", "bodyPosture": "supine",
             "motionState": "still", "groupId": "s1"},
        ]
        report = gate_threshold_metrics.fixed_threshold_report(
            rows,
            [0, 1],
            [0.2, 0.8],
            threshold=0.5,
        )
        self.assertEqual("external_threshold", report["thresholdPolicy"])
        self.assertEqual(0.5, report["overall"]["threshold"])
        self.assertEqual(1, report["overall"]["truePositiveCount"])
        self.assertEqual(
            0.5,
            report["groups"]["deviceName"]["a"]["threshold"],
        )

    def test_calibration_groups_are_disjoint_from_fit_groups(self):
        rows = []
        index = 0
        for label in contract.CLASS_LABELS:
            for session_index in range(3):
                index += 1
                rows.append(training_row(
                    index,
                    label,
                    label + "-session-" + str(session_index),
                    "device-a",
                ))

        fit_rows, calibration_rows = validation.calibration_split(
            rows,
            seed=7,
            ratio=0.34,
        )

        self.assertFalse(
            {row["groupId"] for row in fit_rows}
            & {row["groupId"] for row in calibration_rows}
        )
        self.assertTrue(calibration_rows)
        validation.ensure_fold_coverage(fit_rows, calibration_rows)

    def test_calibration_covers_motion_gate_cells_by_whole_session(self):
        rows = []
        index = 0
        for motion_state in ("handheld_static", "walking"):
            for label in contract.CLASS_LABELS:
                for session_index in range(3):
                    index += 1
                    row = training_row(
                        index,
                        label,
                        motion_state + "-" + label + "-" + str(session_index),
                        "device-a",
                    )
                    row["motionState"] = motion_state
                    rows.append(row)

        fit_rows, calibration_rows = validation.calibration_split(
            rows,
            seed=11,
            ratio=0.22,
        )

        fit_groups = {row["groupId"] for row in fit_rows}
        calibration_groups = {
            row["groupId"] for row in calibration_rows
        }
        self.assertFalse(fit_groups & calibration_groups)
        calibration_cells = {
            (
                validation.scenario_domain(row),
                contract.class_strength_target(row["classLabel"]),
            )
            for row in calibration_rows
        }
        self.assertEqual({
            ("handheld_static", 0),
            ("handheld_static", 1),
            ("walking", 0),
            ("walking", 1),
        }, calibration_cells)
        validation.ensure_fold_coverage(fit_rows, calibration_rows)

    def test_calibration_honors_scenario_gate_session_quota(self):
        rows = []
        index = 0
        for motion_state in ("handheld_static", "walking"):
            for label in contract.CLASS_LABELS:
                for session_index in range(5):
                    index += 1
                    row = training_row(
                        index,
                        label,
                        motion_state + "-" + label + "-"
                        + str(session_index),
                        "device-a",
                    )
                    row["motionState"] = motion_state
                    rows.append(row)

        fit_rows, calibration_rows = validation.calibration_split(
            rows,
            seed=19,
            ratio=0.22,
            min_scenario_gate_sessions=2,
        )

        calibration_groups = {
            row["groupId"] for row in calibration_rows
        }
        fit_groups = {row["groupId"] for row in fit_rows}
        self.assertFalse(calibration_groups & fit_groups)
        for scenario in ("handheld_static", "walking"):
            for gate_label in (0, 1):
                selected = {
                    row["groupId"]
                    for row in calibration_rows
                    if validation.scenario_domain(row) == scenario
                    and contract.class_strength_target(
                        row["classLabel"]
                    ) == gate_label
                }
                self.assertGreaterEqual(len(selected), 2)
        validation.ensure_fold_coverage(fit_rows, calibration_rows)

    def test_calibration_does_not_duplicate_covered_scenario_gate(self):
        rows = []
        index = 0
        for block_index in range(3):
            for motion_state in ("handheld_static", "walking"):
                for label in contract.CLASS_LABELS:
                    index += 1
                    row = training_row(
                        index,
                        label,
                        "block-{}-{}-{}".format(
                            block_index,
                            motion_state,
                            label,
                        ),
                        "device-a",
                    )
                    row["motionState"] = motion_state
                    row["_experimentBlock"] = "block-" + str(block_index)
                    rows.append(row)

        fit_rows, calibration_rows = validation.calibration_split(
            rows,
            seed=23,
            ratio=0.22,
            group_field="_experimentBlock",
        )

        calibration_blocks = {
            row["_experimentBlock"] for row in calibration_rows
        }
        self.assertEqual(1, len(calibration_blocks))
        self.assertEqual(
            {
                ("handheld_static", 0),
                ("handheld_static", 1),
                ("walking", 0),
                ("walking", 1),
            },
            {
                (
                    validation.scenario_domain(row),
                    contract.class_strength_target(row["classLabel"]),
                )
                for row in calibration_rows
            },
        )
        validation.ensure_fold_coverage(fit_rows, calibration_rows)

    def test_calibration_score_weights_motion_domains_equally(self):
        truths = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
        domains = ["still"] * 4 + ["walking"] * 2
        probabilities = np.asarray(
            [0.1, 0.9, 0.8, 0.7, 0.1, 0.2],
            dtype=np.float64,
        )

        score = validation.calibration_score(
            truths,
            probabilities,
            domains,
        )

        self.assertAlmostEqual(0.625, score, places=6)

    def test_fixed_fpr_selection_uses_one_global_calibration_threshold(self):
        truths = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
        domains = ["still"] * 4 + ["walking"] * 2
        probabilities = np.asarray(
            [0.10, 0.90, 0.80, 0.70, 0.10, 0.20],
            dtype=np.float64,
        )

        result = validation.calibration_fixed_fpr_score(
            truths,
            probabilities,
            domains,
            target_fpr=0.25,
        )

        self.assertAlmostEqual(0.80, result["threshold"], places=6)
        self.assertAlmostEqual(0.25, result["score"], places=6)
        self.assertEqual({"still", "walking"}, set(result["domainRecall"]))

    def test_deployment_threshold_calibration_can_be_stricter_than_report_fpr(
            self):
        negatives = np.linspace(0.999, 0.001, 400)

        at_one_percent = validation.gate_threshold_metrics.select_threshold_at_fpr(
            negatives,
            0.01,
        )
        at_zero_point_five_percent = (
            validation.gate_threshold_metrics.select_threshold_at_fpr(
                negatives,
                0.005,
            )
        )

        self.assertGreater(at_zero_point_five_percent, at_one_percent)

    def test_threshold_sweep_uses_external_threshold_without_refitting(self):
        rows = [
            {"deviceName": "device:a"},
            {"deviceName": "device:a"},
            {"deviceName": "device:a"},
            {"deviceName": "device:a"},
        ]
        truths = np.asarray([0, 1, 0, 1], dtype=np.int64)
        scores = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float64)

        report = validation.gate_threshold_metrics.fixed_threshold_report(
            rows,
            truths,
            scores,
            0.5,
        )

        self.assertEqual("external_threshold", report["thresholdPolicy"])
        self.assertEqual(0.5, report["overall"]["threshold"])
        self.assertEqual(1, report["overall"]["truePositiveCount"])
        self.assertEqual(1, report["overall"]["falsePositiveCount"])

    def test_threshold_sweep_ranking_enforces_measured_fpr_limit(self):
        runs = [
            {
                "fixedFprMetrics": {
                    "0.01": {
                        "thresholdCalibrationSweep": {
                            "0.005": {
                                "overall": {
                                    "heavyRecall": 0.8,
                                    "lightFalsePositiveRate": 0.009,
                                },
                                "groups": {
                                    "deviceName": {
                                        "device:a": {
                                            "heavyRecall": 0.8,
                                            "lightFalsePositiveRate": 0.004,
                                        },
                                    },
                                },
                            },
                            "0.01": {
                                "overall": {
                                    "heavyRecall": 0.9,
                                    "lightFalsePositiveRate": 0.02,
                                },
                                "groups": {
                                    "deviceName": {
                                        "device:a": {
                                            "heavyRecall": 0.9,
                                            "lightFalsePositiveRate": 0.02,
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        ]

        summary = validation.summarize_threshold_calibration_sweep(
            runs,
            0.01,
        )

        self.assertTrue(summary["0.005"]["falsePositiveLimitMet"])
        self.assertFalse(summary["0.01"]["falsePositiveLimitMet"])
        self.assertAlmostEqual(
            0.8,
            summary["0.005"]["worstDeviceHeavyRecall"],
        )
        self.assertAlmostEqual(
            0.004,
            summary["0.005"]["worstDeviceFalsePositiveRate"],
        )

    def test_threshold_sweep_rejects_single_device_fpr_above_limit(self):
        sweep = {
            "overall": {
                "heavyRecall": 0.85,
                "lightFalsePositiveRate": 0.005,
            },
            "groups": {
                "deviceName": {
                    "device:a": {
                        "heavyRecall": 0.9,
                        "lightFalsePositiveRate": 0.002,
                    },
                    "device:b": {
                        "heavyRecall": 0.8,
                        "lightFalsePositiveRate": 0.012,
                    },
                },
            },
        }
        runs = [{
            "fixedFprMetrics": {
                "0.01": {
                    "thresholdCalibrationSweep": {"0.005": sweep},
                },
            },
        }]

        summary = validation.summarize_threshold_calibration_sweep(
            runs,
            0.01,
        )

        self.assertFalse(summary["0.005"]["falsePositiveLimitMet"])
        self.assertAlmostEqual(
            0.012,
            summary["0.005"]["worstDeviceFalsePositiveRate"],
        )

    def test_fold_runner_exposes_threshold_sweep_contract(self):
        import inspect

        parameters = inspect.signature(
            validation.run_fold_seed
        ).parameters

        self.assertIn("threshold_calibration_fpr", parameters)
        self.assertIn("threshold_calibration_sweep", parameters)
        self.assertIn("false_positive_limit", parameters)

    def test_selection_mode_rejects_unknown_objective(self):
        with self.assertRaisesRegex(
                RuntimeError,
                "unsupported_model_selection_mode"):
            validation.calibration_selection_score(
                "unknown",
                [0, 1],
                [0.2, 0.8],
            )

    def test_pairwise_ranking_loss_prefers_heavy_above_light(self):
        import torch

        targets = torch.tensor(
            [1, 0, 1, 0],
            dtype=torch.long,
        )
        weights = torch.ones(4)
        correct = torch.tensor(
            [[0.0, 2.0], [2.0, 0.0], [0.0, 3.0], [1.0, 0.0]],
        )
        reversed_scores = torch.tensor(
            [[2.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 1.0]],
        )

        correct_loss = validation.pairwise_ranking_loss(
            correct,
            targets,
            weights,
        )
        reversed_loss = validation.pairwise_ranking_loss(
            reversed_scores,
            targets,
            weights,
        )

        self.assertLess(float(correct_loss), float(reversed_loss))
        self.assertTrue(torch.isfinite(correct_loss))

    def test_pairwise_ranking_loss_handles_single_class_batch(self):
        import torch

        logits = torch.tensor(
            [[0.0, 1.0], [0.0, 2.0]],
            requires_grad=True,
        )
        loss = validation.pairwise_ranking_loss(
            logits,
            torch.tensor([1, 1]),
            torch.ones(2),
        )
        loss.backward()

        self.assertEqual(0.0, float(loss.detach()))
        self.assertIsNotNone(logits.grad)

    def test_asymmetric_focal_negative_weight_scales_light_loss(self):
        import torch

        logits = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0]],
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 0], dtype=torch.long)
        weights = torch.ones(2)
        symmetric = validation.asymmetric_focal_loss(
            logits,
            targets,
            weights,
            1.0,
            1.0,
        )
        asymmetric = validation.asymmetric_focal_loss(
            logits,
            targets,
            weights,
            1.0,
            2.0,
        )

        self.assertAlmostEqual(
            float(symmetric) * 2.0,
            float(asymmetric),
            places=6,
        )

    def test_pairwise_loss_mode_rejects_invalid_weight(self):
        import torch

        with self.assertRaisesRegex(
                RuntimeError,
                "invalid_ranking_loss_weight"):
            validation.train_model(
                np.zeros((2, 60), dtype=np.float32),
                [0, 1],
                [0, 1],
                [(-100,) * 4, (-100,) * 4],
                np.asarray([0.5, 0.5], dtype=np.float32),
                np.zeros((2, 60), dtype=np.float32),
                [0, 1],
                ["still", "still"],
                60,
                4,
                1,
                1,
                False,
                0.0,
                0.0,
                1,
                gate_loss_mode="ce_pairwise",
                ranking_loss_weight=11.0,
            )

    def test_group_threshold_diagnostics_skip_one_class_groups(self):
        rows = [
            {"_experimentBlock": "negative-only"},
            {"_experimentBlock": "negative-only"},
            {"_experimentBlock": "mixed"},
            {"_experimentBlock": "mixed"},
        ]
        diagnostics = validation.group_threshold_diagnostics(
            rows,
            [0, 0, 0, 1],
            [0.1, 0.2, 0.3, 0.9],
            "_experimentBlock",
            0.50,
        )

        self.assertIsNone(diagnostics["negative-only"]["threshold"])
        self.assertIsNone(diagnostics["negative-only"]["heavyRecall"])
        self.assertEqual(1.0, diagnostics["mixed"]["heavyRecall"])

    def test_score_group_diagnostics_reports_auc_for_every_group_field(self):
        rows = [
            {
                "groupId": "session-a",
                "_experimentBlock": "block-a",
                "collectorUserName": "collector-a",
                "motionState": "handheld_static",
                "taskId": "tilted_handheld",
                "classLabel": "thumb_light",
            },
            {
                "groupId": "session-a",
                "_experimentBlock": "block-a",
                "collectorUserName": "collector-a",
                "motionState": "handheld_static",
                "taskId": "tilted_handheld",
                "classLabel": "thumb_heavy",
            },
            {
                "groupId": "session-b",
                "_experimentBlock": "block-b",
                "collectorUserName": "collector-a",
                "motionState": "walking",
                "taskId": "walking",
                "classLabel": "index_light",
            },
            {
                "groupId": "session-b",
                "_experimentBlock": "block-b",
                "collectorUserName": "collector-a",
                "motionState": "walking",
                "taskId": "walking",
                "classLabel": "index_heavy",
            },
        ]
        diagnostics = validation.score_group_diagnostics(
            rows,
            [0, 1, 0, 1],
            [0.1, 0.9, 0.2, 0.8],
            0.5,
        )

        self.assertEqual(1.0, diagnostics["session"]["session-a"]["rocAuc"])
        self.assertEqual(
            1.0,
            diagnostics["experimentBlock"]["block-a"]["rocAuc"],
        )
        self.assertEqual(
            1.0,
            diagnostics["posture"]["tilted"]["rocAuc"],
        )
        self.assertEqual(
            1.0,
            diagnostics["finger"]["index"]["rocAuc"],
        )

    def test_scenario_domain_keeps_posture_and_support_separate(self):
        self.assertEqual(
            "walking",
            validation.scenario_domain({"motionState": "walking"}),
        )
        self.assertEqual(
            "placed",
            validation.scenario_domain({"motionState": "placed_static"}),
        )
        self.assertEqual(
            "supine",
            validation.scenario_domain({
                "motionState": "handheld_static",
                "taskId": "supine",
                "bodyPosture": "supine",
            }),
        )
        self.assertEqual(
            "tilted_handheld",
            validation.scenario_domain({
                "motionState": "handheld_static",
                "taskId": "tilted_handheld",
                "bodyPosture": "unspecified",
            }),
        )
        self.assertEqual(
            "tilted",
            validation.posture_domain({
                "motionState": "handheld_static",
                "taskId": "tilted_handheld",
                "bodyPosture": "unspecified",
            }),
        )

    def test_calibration_domain_scores_skip_one_class_domains(self):
        scores = validation.calibration_domain_scores(
            [0, 1, 0, 0],
            [0.1, 0.9, 0.2, 0.3],
            ["still", "still", "walking", "walking"],
        )

        self.assertEqual({"still"}, set(scores))
        self.assertAlmostEqual(1.0, scores["still"], places=6)

    def test_device_fold_never_places_a_device_on_both_sides(self):
        rows = []
        index = 0
        for device_name in ("device-a", "device-b"):
            for label in contract.CLASS_LABELS:
                for session_index in range(2):
                    index += 1
                    rows.append(training_row(
                        index,
                        label,
                        device_name + "-" + label + "-" + str(session_index),
                        device_name,
                    ))

        folds = validation.build_device_folds(rows)

        self.assertEqual(2, len(folds))
        for fold in folds:
            self.assertEqual(
                "_experimentBlock",
                fold["calibrationGroupField"],
            )
            train_devices = {
                validation.physical_device_key(row)
                for row in fold["train"]
            }
            valid_devices = {
                validation.physical_device_key(row)
                for row in fold["valid"]
            }
            self.assertFalse(train_devices & valid_devices)

    def test_external_device_fold_excludes_target_and_other_external_rows(self):
        training_rows = []
        for device_name in ("train-a", "train-b"):
            for index, label in enumerate(contract.CLASS_LABELS):
                training_rows.append(training_row(
                    len(training_rows) + 1,
                    label,
                    device_name + "-" + label,
                    device_name,
                ))
        evaluation_rows = []
        for device_name in ("train-a", "external-c"):
            for index, label in enumerate(contract.CLASS_LABELS):
                evaluation_rows.append(training_row(
                    1000 + len(evaluation_rows),
                    label,
                    "external-" + device_name + "-" + label,
                    device_name,
                ))

        folds = validation.build_external_device_folds(
            training_rows,
            evaluation_rows,
        )

        self.assertEqual(2, len(folds))
        external_hashes = {
            row["sourceSha256"] for row in evaluation_rows
        }
        for fold in folds:
            self.assertEqual(
                "_experimentBlock",
                fold["calibrationGroupField"],
            )
            target_device = validation.physical_device_key(
                fold["valid"][0]
            )
            self.assertTrue(all(
                validation.physical_device_key(row) != target_device
                for row in fold["train"]
            ))
            self.assertFalse(
                external_hashes
                & {row["sourceSha256"] for row in fold["train"]}
            )
            self.assertEqual(
                {target_device},
                {
                    validation.physical_device_key(row)
                    for row in fold["evaluationOnlyRows"]
                },
            )

    def test_duplicate_source_hashes_are_merged_without_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            for dataset_root in (first, second):
                (dataset_root / "training_touch26.jsonl").write_text(
                    '{"sourceSha256":"same","classLabel":"thumb_light",'
                    '"matrix":[],"groupId":"s"}\n',
                    encoding="utf-8",
                )

            rows = validation.load_rows((first, second))

            self.assertEqual(1, len(rows))
            self.assertEqual("same", rows[0]["sourceSha256"])

    def test_summary_uses_same_external_threshold_for_every_group(self):
        runs = [{
            "foldKind": "device",
            "foldId": "device:a",
            "runtimeParameterCount": 100,
            "trainingParameterCount": 100,
            "fixedFprMetrics": {
                "0.01": {
                    "overall": {"heavyRecall": 0.5},
                    "groups": {
                        "deviceName": {
                            "a": {
                                "positiveCount": 25,
                                "heavyRecall": 0.5,
                            },
                        },
                    },
                },
                "0.05": {
                    "overall": {"heavyRecall": 0.7},
                    "groups": {
                        "deviceName": {
                            "a": {
                                "positiveCount": 25,
                                "heavyRecall": 0.7,
                            },
                        },
                    },
                },
            },
        }]

        summary = validation.summarize_runs(runs, "gate")

        self.assertEqual(
            0.5,
            summary["device"]["meanHeavyRecallByTargetFpr"]["0.01"],
        )
        self.assertEqual(
            0.5,
            summary["device"][
                "worstGroupHeavyRecallByTargetFpr"
            ]["0.01"],
        )

    def test_summary_keeps_each_fold_value_distinct(self):
        runs = [
            {
                "foldKind": "device",
                "foldId": fold_id,
                "runtimeParameterCount": 100,
                "trainingParameterCount": 100,
                "fixedFprMetrics": {
                    "0.01": {
                        "overall": {"heavyRecall": recall},
                        "groups": {
                            "deviceName": {
                                fold_id: {
                                    "positiveCount": 25,
                                    "heavyRecall": recall,
                                },
                            },
                        },
                    },
                    "0.05": {
                        "overall": {"heavyRecall": recall},
                        "groups": {
                            "deviceName": {
                                fold_id: {
                                    "positiveCount": 25,
                                    "heavyRecall": recall,
                                },
                            },
                        },
                    },
                },
            }
            for fold_id, recall in (("device:a", 0.2), ("device:b", 0.8))
        ]

        summary = validation.summarize_runs(runs, "gate")
        folds = summary["device"]["folds"]

        self.assertEqual(
            0.2,
            folds["device:a"]["meanHeavyRecallByTargetFpr"]["0.01"],
        )
        self.assertEqual(
            0.8,
            folds["device:b"]["meanHeavyRecallByTargetFpr"]["0.01"],
        )
        self.assertEqual(
            0.2,
            summary["device"]["worstFoldHeavyRecallByTargetFpr"]["0.01"],
        )

    def test_summary_aggregates_target_calibrated_metrics(self):
        runs = []
        for fold_id, recall in (("device:a", 0.4), ("device:b", 0.8)):
            reports = {}
            for target in ("0.01", "0.05"):
                reports[target] = {
                    "targetCalibrationNegativeCount": 16,
                    "overall": {
                        "heavyRecall": recall,
                        "lightFalsePositiveRate": 0.0,
                        "negativeCount": 25,
                    },
                    "groups": {
                        "deviceName": {
                            fold_id: {
                                "positiveCount": 25,
                                "heavyRecall": recall,
                                "negativeCount": 25,
                                "lightFalsePositiveRate": 0.01,
                            },
                        },
                    },
                }
            runs.append({
                "seed": 1,
                "foldKind": "device",
                "foldId": fold_id,
                "runtimeParameterCount": 100,
                "trainingParameterCount": 100,
                "targetCalibratedMetrics": {"16": reports},
            })
        summary = validation.summarize_target_calibration_runs(runs)
        device = summary["16"]["device"]
        self.assertAlmostEqual(
            0.6,
            device["meanHeavyRecallByTargetFpr"]["0.01"],
            places=6,
        )
        self.assertEqual(
            0.4,
            device["worstFoldHeavyRecallByTargetFpr"]["0.01"],
        )
        self.assertEqual(
            0.0,
            device["worstHeldOutFalsePositiveRateByTargetFpr"]["0.01"],
        )
        self.assertFalse(device["proofReadyByTargetFpr"]["0.01"])

    def test_summary_reports_calibration_seed_spread(self):
        runs = []
        for fold_id, recalls in (
                ("device:a", (0.2, 0.4)),
                ("device:b", (0.6, 0.8))):
            for seed, recall in zip((1, 2), recalls):
                reports = {}
                for target in ("0.01", "0.05"):
                    reports[target] = {
                        "targetCalibrationNegativeCount": 16,
                        "overall": {
                            "heavyRecall": recall,
                            "lightFalsePositiveRate": 0.0,
                            "negativeCount": 25,
                        },
                        "groups": {
                            "deviceName": {
                                fold_id: {
                                    "positiveCount": 25,
                                    "heavyRecall": recall,
                                "negativeCount": 25,
                                "lightFalsePositiveRate": 0.0,
                                },
                            },
                        },
                    }
                runs.append({
                    "seed": seed,
                    "foldKind": "device",
                    "foldId": fold_id,
                    "runtimeParameterCount": 100,
                    "trainingParameterCount": 100,
                    "targetCalibratedMetrics": {"16": reports},
                })
        summary = validation.summarize_target_calibration_runs(runs)
        target = summary["16"]["device"][
            "worstSeedHeavyRecallByTargetFpr"
        ]["0.01"]
        self.assertAlmostEqual(0.4, target, places=6)
        spread = summary["16"]["device"][
            "seedRecallRangeByTargetFpr"
        ]["0.01"]
        self.assertAlmostEqual(0.2, spread["maxMinusMin"], places=6)

    def test_external_summary_tracks_worst_run_and_device(self):
        runs = []
        for fold_id, recall in (("device:a", 0.3), ("device:b", 0.7)):
            metrics = {}
            for target in ("0.01", "0.05"):
                metrics[target] = {
                    "overall": {
                        "heavyRecall": recall,
                        "lightFalsePositiveRate": 0.01,
                        "negativeCount": 25,
                    },
                    "groups": {
                        "deviceName": {
                            fold_id: {
                                "positiveCount": 25,
                                "heavyRecall": recall,
                                "negativeCount": 25,
                                "lightFalsePositiveRate": 0.01,
                            },
                        },
                        "bodyPosture": {
                            "supine": {
                                "positiveCount": 25,
                                "heavyRecall": recall,
                                "negativeCount": 25,
                                "lightFalsePositiveRate": 0.01,
                            },
                            "tilted": {
                                "positiveCount": 25,
                                "heavyRecall": 0.9,
                                "negativeCount": 25,
                                "lightFalsePositiveRate": 0.01,
                            },
                        },
                        "motionState": {
                            "walking": {
                                "positiveCount": 25,
                                "heavyRecall": 0.2,
                                "negativeCount": 25,
                                "lightFalsePositiveRate": 0.01,
                            },
                            "placed_static": {
                                "positiveCount": 25,
                                "heavyRecall": 0.9,
                                "negativeCount": 25,
                                "lightFalsePositiveRate": 0.01,
                            },
                        },
                    },
                }
            runs.append({
                "seed": 1,
                "foldKind": "device",
                "foldId": fold_id,
                "evaluationOnlyMetrics": metrics,
            })
        summary = validation.summarize_evaluation_only_runs(runs)
        device = summary["device"]
        self.assertAlmostEqual(
            0.5,
            device["meanHeavyRecallByTargetFpr"]["0.01"],
            places=6,
        )
        self.assertEqual(
            0.3,
            device["worstRunHeavyRecallByTargetFpr"]["0.01"],
        )
        self.assertEqual(
            0.3,
            device["worstDeviceHeavyRecallByTargetFpr"]["0.01"],
        )
        self.assertEqual(
            0.3,
            device["worstPostureHeavyRecallByTargetFpr"]["0.01"],
        )
        self.assertEqual(
            0.2,
            device["worstMotionHeavyRecallByTargetFpr"]["0.01"],
        )
        self.assertEqual(
            0.2,
            device["worstGroupHeavyRecallByTargetFpr"]["0.01"],
        )
        self.assertEqual(
            0.01,
            device["worstFoldFalsePositiveRateByTargetFpr"]["0.01"],
        )
        self.assertFalse(device["proofReadyByTargetFpr"]["0.01"])

    def test_external_summary_ignores_unknown_groups(self):
        metrics = {}
        for target in ("0.01", "0.05"):
            metrics[target] = {
                "overall": {
                    "heavyRecall": 0.9,
                    "lightFalsePositiveRate": 0.0,
                    "negativeCount": 25,
                },
                "groups": {
                    "bodyPosture": {
                        "unknown": {
                            "positiveCount": 100,
                            "heavyRecall": 0.0,
                            "negativeCount": 100,
                            "lightFalsePositiveRate": 0.0,
                        },
                        "supine": {
                            "positiveCount": 25,
                            "heavyRecall": 0.9,
                            "negativeCount": 25,
                            "lightFalsePositiveRate": 0.0,
                        },
                    },
                },
            }
        runs = [{
            "seed": 1,
            "foldKind": "device",
            "foldId": "device:a",
            "evaluationOnlyMetrics": metrics,
        }]

        summary = validation.summarize_evaluation_only_runs(runs)["device"]

        self.assertEqual(
            0.9,
            summary["worstPostureHeavyRecallByTargetFpr"]["0.01"],
        )

    def test_evidence_magnitude_mode_runs_without_axis_channels(self):
        matrix = np.zeros(
            (contract.FRAME_COUNT, len(validation.lightweight.FEATURE_NAMES)),
            dtype=np.float32,
        )
        matrix[:, 12] = 1.0
        matrix[:, 15] = 0.2
        matrix[:, 22:25] = 1.0
        row = {
            "matrix": matrix,
            "classLabel": "thumb_heavy",
            "deviceName": "device-a",
            "groupId": "session-a",
        }
        fit_values, calibration_values, valid_values = (
            validation.prepare_values(
                [row],
                [row],
                [row],
                feature_mode="magnitude_only",
                feature_family="evidence",
            )
        )

        self.assertEqual(
            (len(validation.evidence_features.FEATURE_NAMES),),
            fit_values.shape[1:],
        )
        self.assertEqual(fit_values.shape, calibration_values.shape)
        self.assertEqual(fit_values.shape, valid_values.shape)
        self.assertTrue(np.isfinite(valid_values).all())

    def test_evidence_contract_reaches_validation_preprocessing(self):
        matrix = np.zeros(
            (contract.FRAME_COUNT, len(validation.lightweight.FEATURE_NAMES)),
            dtype=np.float32,
        )
        matrix[:, 8] = 1.0
        matrix[:, 19] = 1.0
        matrix[:, 22:25] = 1.0
        row = {
            "matrix": matrix,
            "classLabel": "thumb_heavy",
            "deviceName": "device-a",
            "groupId": "session-a",
        }

        fit, calibration, valid = validation.prepare_values(
            [row],
            [row],
            [row],
            feature_family="evidence",
            evidence_contract="impact_post5",
        )

        self.assertEqual(fit.shape, calibration.shape)
        self.assertEqual(fit.shape, valid.shape)
        self.assertTrue(np.isfinite(valid).all())

    def test_deployment_gate_requires_worst_group_recall_and_evidence(self):
        def summary():
            return {
                "seedCount": 3,
                "worstDeviceHeavyRecallByTargetFpr": {"0.01": 0.95},
                "worstPostureHeavyRecallByTargetFpr": {"0.01": 0.94},
                "worstMotionHeavyRecallByTargetFpr": {"0.01": 0.93},
                "worstGroupFalsePositiveRateByTargetFpr": {"0.01": 0.01},
                "proofReadyByTargetFpr": {"0.01": True},
                "pairedGroupCoverageByTargetFpr": {
                    "0.01": {
                        "deviceName": {"pairedGroupCount": 3},
                        "bodyPosture": {"pairedGroupCount": 2},
                        "motionState": {"pairedGroupCount": 2},
                    },
                },
            }

        gate = validation.deployment_gate(summary())
        self.assertTrue(gate["eligible"])
        self.assertEqual([], gate["blockers"])

        for field, key in (
            ("deviceName", "worstDeviceHeavyRecallByTargetFpr"),
            ("bodyPosture", "worstPostureHeavyRecallByTargetFpr"),
            ("motionState", "worstMotionHeavyRecallByTargetFpr"),
        ):
            with self.subTest(field=field):
                report = summary()
                report[key]["0.01"] = 0.89
                gate = validation.deployment_gate(report)
                self.assertFalse(gate["eligible"])
                self.assertIn(field + "_recall_below_target", gate["blockers"])

    def test_deployment_gate_rejects_missing_pose_evidence(self):
        report = {
            "seedCount": 3,
            "worstDeviceHeavyRecallByTargetFpr": {"0.01": 0.95},
            "worstPostureHeavyRecallByTargetFpr": {"0.01": None},
            "worstMotionHeavyRecallByTargetFpr": {"0.01": 0.95},
            "worstGroupFalsePositiveRateByTargetFpr": {"0.01": 0.01},
            "proofReadyByTargetFpr": {"0.01": True},
            "pairedGroupCoverageByTargetFpr": {
                "0.01": {
                    "deviceName": {"pairedGroupCount": 3},
                    "bodyPosture": {"pairedGroupCount": 0},
                    "motionState": {"pairedGroupCount": 2},
                },
            },
        }

        gate = validation.deployment_gate(report)

        self.assertFalse(gate["eligible"])
        self.assertIn("bodyPosture_evidence_incomplete", gate["blockers"])

    def test_deployment_gate_does_not_require_ordinary_click_dataset(self):
        report = {
            "seedCount": 3,
            "worstDeviceHeavyRecallByTargetFpr": {"0.01": 0.95},
            "worstPostureHeavyRecallByTargetFpr": {"0.01": 0.95},
            "worstMotionHeavyRecallByTargetFpr": {"0.01": 0.95},
            "worstGroupFalsePositiveRateByTargetFpr": {"0.01": 0.01},
            "proofReadyByTargetFpr": {"0.01": True},
            "pairedGroupCoverageByTargetFpr": {
                "0.01": {
                    "deviceName": {"pairedGroupCount": 3},
                    "bodyPosture": {"pairedGroupCount": 2},
                    "motionState": {"pairedGroupCount": 2},
                },
            },
        }

        gate = validation.deployment_gate(report)

        self.assertTrue(gate["eligible"])
        self.assertNotIn("ordinary_click_evidence_missing", gate["blockers"])

    def test_external_summary_requires_calibration_and_held_out_negatives(self):
        runs = []
        for fold_id, negative_count, calibration_count, status in (
            ("device:a", 1000, 0, "descriptive_only"),
            ("device:b", 25, 1000, "descriptive_only"),
        ):
            metrics = {}
            for target in ("0.01", "0.05"):
                metrics[target] = {
                    "calibrationEvidenceStatus": status,
                    "calibrationNegativeCount": calibration_count,
                    "heldOutNegativeCount": negative_count,
                    "overall": {
                        "heavyRecall": 0.5,
                        "lightFalsePositiveRate": 0.0,
                    "positiveCount": 1000,
                        "negativeCount": negative_count,
                    },
                    "groups": {
                        "deviceName": {
                            fold_id: {
                                "positiveCount": 25,
                                "heavyRecall": 0.5,
                                "negativeCount": negative_count,
                                "lightFalsePositiveRate": 0.0,
                            },
                        },
                    },
                }
            runs.append({
                "seed": 1,
                "foldKind": "device",
                "foldId": fold_id,
                "evaluationOnlyMetrics": metrics,
            })

        summary = validation.summarize_evaluation_only_runs(runs)["device"]
        self.assertFalse(summary["proofReadyByTargetFpr"]["0.01"])
        self.assertEqual(
            0,
            summary["foldMetrics"]["device:a"]["0.01"][
                "calibrationNegativeCount"
            ],
        )
        self.assertEqual(
            25,
            summary["foldMetrics"]["device:b"]["0.01"][
                "heldOutNegativeCount"
            ],
        )

    def test_external_summary_accepts_complete_proof_evidence(self):
        metrics = {}
        for target in ("0.01", "0.05"):
            metrics[target] = {
                "calibrationEvidenceStatus": "proof_ready",
                "calibrationNegativeCount": 1000,
                "heldOutNegativeCount": 1000,
                    "calibrationSessionCount": 5,
                    "heldOutNegativeSessionCount": 5,
                "overall": {
                    "heavyRecall": 0.5,
                    "lightFalsePositiveRate": 0.0,
                    "positiveCount": 1000,
                    "negativeCount": 1000,
                },
                "groups": {
                    "deviceName": {
                        "device:a": {
                            "positiveCount": 25,
                            "heavyRecall": 0.5,
                            "negativeCount": 1000,
                            "lightFalsePositiveRate": 0.0,
                        },
                    },
                },
            }
        runs = [{
            "seed": 1,
            "foldKind": "device",
            "foldId": "device:a",
            "evaluationOnlyMetrics": metrics,
        }]

        summary = validation.summarize_evaluation_only_runs(runs)["device"]
        self.assertTrue(summary["proofReadyByTargetFpr"]["0.01"])

    def test_external_summary_requires_independent_proof_sessions(self):
        metrics = {}
        for target in ("0.01", "0.05"):
            metrics[target] = {
                "calibrationEvidenceStatus": "proof_ready",
                "calibrationNegativeCount": 1000,
                "heldOutNegativeCount": 1000,
                "calibrationSessionCount": 4,
                "heldOutNegativeSessionCount": 5,
                "overall": {
                    "heavyRecall": 0.5,
                    "lightFalsePositiveRate": 0.0,
                    "positiveCount": 1000,
                    "negativeCount": 1000,
                },
                "groups": {
                    "deviceName": {
                        "device:a": {
                            "positiveCount": 25,
                            "heavyRecall": 0.5,
                            "negativeCount": 1000,
                            "lightFalsePositiveRate": 0.0,
                        },
                    },
                },
            }
        runs = [{
            "seed": 1,
            "foldKind": "device",
            "foldId": "device:a",
            "evaluationOnlyMetrics": metrics,
        }]

        summary = validation.summarize_evaluation_only_runs(runs)["device"]

        self.assertFalse(summary["proofReadyByTargetFpr"]["0.01"])
        self.assertEqual(
            4,
            summary["foldMetrics"]["device:a"]["0.01"][
                "calibrationSessionCount"
            ],
        )
        self.assertEqual(
            validation.MIN_PROOF_SESSION_COUNT,
            summary["foldMetrics"]["device:a"]["0.01"][
                "minimumProofSessionCount"
            ],
        )

    def test_target_summary_reads_repeated_calibration_reports(self):
        reports = {}
        for target in ("0.01", "0.05"):
            repetitions = [
                {
                    "overall": {
                        "heavyRecall": recall,
                        "lightFalsePositiveRate": 0.01,
                        "negativeCount": 25,
                    },
                    "groups": {
                        "deviceName": {
                            "device:a": {
                                "positiveCount": 25,
                                "heavyRecall": recall,
                            },
                        },
                    },
                }
                for recall in (0.2, 0.4)
            ]
            reports[target] = {
                "repetitions": 2,
                "overall": {
                    "heavyRecall": 0.3,
                    "lightFalsePositiveRate": 0.01,
                },
                "perRepetition": repetitions,
            }
        runs = [{
            "seed": 1,
            "foldKind": "device",
            "foldId": "device:a",
            "runtimeParameterCount": 100,
            "trainingParameterCount": 100,
            "targetCalibratedMetrics": {"32": reports},
        }]
        summary = validation.summarize_target_calibration_runs(runs)
        device = summary["32"]["device"]
        self.assertAlmostEqual(
            0.3,
            device["meanHeavyRecallByTargetFpr"]["0.01"],
            places=6,
        )
        self.assertEqual(
            0.2,
            device["worstGroupHeavyRecallByTargetFpr"]["0.01"],
        )


if __name__ == "__main__":
    unittest.main()
