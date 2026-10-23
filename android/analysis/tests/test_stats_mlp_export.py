import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from android.analysis import export_stats_mlp_android_model as exporter
from android.analysis import contact_size_proxy
from android.analysis import stats_mlp_contract as contract
from android.analysis import train_stats_mlp_android_candidate as trainer


SHARED_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "aispect-core"
    / "src"
    / "test"
    / "resources"
    / "stats_mlp_v4_golden_fixture.json"
)


class StatsMlpExportTests(unittest.TestCase):
    def valid_payload(self):
        channels = []
        for name in contract.PHYSICAL_FEATURE_NAMES:
            channels.append({
                "name": name,
                "center": 0.0,
                "scale": 1.0,
                "clipMin": -8.0,
                "clipMax": 8.0,
            })
        weights = {}
        for name, shape in exporter.WEIGHT_SHAPES.items():
            count = 1
            for dimension in shape:
                count *= dimension
            weights[name] = {"shape": shape, "values": [0.0] * count}
        model_contract = {
            "runtimeArchitecture": contract.RUNTIME_ARCHITECTURE,
            "featureContract": contract.FEATURE_CONTRACT,
            "sourceFeatureContract": contract.SOURCE_FEATURE_CONTRACT,
            "sourceFeatureNames": list(
                contract.RUNTIME_SOURCE_FEATURE_NAMES
            ),
            "sizeSemantics": contract.CONTACT_SIZE_SEMANTICS,
            "forceMeasurement": False,
            "missingValuePolicy": contract.CONTACT_SIZE_PROXY_POLICY[
                "missingValuePolicy"
            ],
            "sizeBaseline": dict(contract.CONTACT_SIZE_BASELINE_POLICY),
            "featureNames": list(contract.PHYSICAL_FEATURE_NAMES),
            "sourceFeatureIndices": list(contract.SOURCE_FEATURE_INDICES),
            "statsFeatureNames": list(contract.STATS_FEATURE_NAMES),
            "matrixShape": [9, len(contract.PHYSICAL_FEATURE_NAMES)],
            "frameCount": 9,
            "primaryTask": contract.PRIMARY_TASK,
            "gateCount": 2,
            "gateLabelOrder": list(contract.GATE_LABELS),
            "negativeClassLabels": list(contract.NEGATIVE_CLASS_LABELS),
            "positiveClassLabels": list(contract.POSITIVE_CLASS_LABELS),
            "classCount": 2,
            "labelOrder": list(contract.GATE_LABELS),
            "diagnosticClassCount": 4,
            "diagnosticClassLabelOrder": list(contract.CLASS_LABELS),
            "fingerLabelOrder": list(contract.FINGER_LABELS),
            "strengthLabelOrder": list(contract.STRENGTH_LABELS),
            "sequenceChannelCount": len(contract.PHYSICAL_FEATURE_NAMES),
            "statsFeatureCount": contract.STATS_FEATURE_COUNT,
            "inputCount": contract.STATS_FEATURE_COUNT,
            "hiddenCount": 24,
            "dropoutAtInference": False,
            "sequenceChannels": channels,
            "statsScaler": {
                "center": [0.0] * contract.STATS_FEATURE_COUNT,
                "scale": [1.0] * contract.STATS_FEATURE_COUNT,
                "clipMin": -8.0,
                "clipMax": 8.0,
            },
            "evaluationScope": "all_data_fit_only",
            "deploymentAllowed": False,
        }
        return weights, model_contract

    def write_source(self, root, weights, model_contract):
        (root / "weights.json").write_text(
            json.dumps(weights),
            encoding="utf-8",
        )
        (root / "model_contract.json").write_text(
            json.dumps(model_contract),
            encoding="utf-8",
        )

    def test_contract_freezes_ten_physical_channels_and_sixty_stats(self):
        self.assertEqual(10, len(contract.PHYSICAL_FEATURE_NAMES))
        self.assertEqual(10, len(contract.SOURCE_FEATURE_INDICES))
        self.assertEqual(60, len(contract.STATS_FEATURE_NAMES))
        self.assertEqual(
            "causal_time_grid_imu11_touch7_mask5_stats_mlp_v4",
            contract.FEATURE_CONTRACT,
        )
        self.assertEqual(23, len(contract.SOURCE_FEATURE_NAMES))
        self.assertNotIn(
            "orientation_sin",
            contract.SOURCE_FEATURE_NAMES,
        )
        self.assertNotIn(
            "orientation_cos",
            contract.SOURCE_FEATURE_NAMES,
        )
        self.assertNotIn(
            "orientation_available",
            contract.SOURCE_FEATURE_NAMES,
        )
        self.assertEqual(
            "log_axis_ratio",
            contract.PHYSICAL_FEATURE_NAMES[-1],
        )
        self.assertEqual(
            "log_causal_size_ratio",
            contract.RUNTIME_SOURCE_FEATURE_NAMES[12],
        )
        self.assertEqual(
            "size",
            contract.SOURCE_FEATURE_NAMES[12],
        )
        self.assertEqual(("light", "heavy"), contract.GATE_LABELS)
        self.assertEqual(
            ("thumb_light", "index_light"),
            contract.NEGATIVE_CLASS_LABELS,
        )
        self.assertEqual(
            ("thumb_heavy", "index_heavy"),
            contract.POSITIVE_CLASS_LABELS,
        )

    def test_shared_fixture_projection_and_statistics_match_python(self):
        fixture = json.loads(SHARED_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(
            contract.LEGACY_V3_SOURCE_FEATURE_CONTRACT,
            fixture["sourceFeatureContract"],
        )
        self.assertEqual(
            list(contract.SOURCE_FEATURE_NAMES),
            fixture["sourceFeatureNames"],
        )
        self.assertEqual(
            list(contract.PHYSICAL_FEATURE_NAMES),
            fixture["physicalFeatureNames"],
        )
        self.assertEqual(
            list(contract.STATS_FEATURE_NAMES),
            fixture["statsFeatureNames"],
        )
        row = {
            "featureNames": fixture["sourceFeatureNames"],
            "matrix": fixture["sourceMatrix"],
        }
        projection = contact_size_proxy.transform_row(
            row,
            fixture["baselineHistory"],
        )
        expected_projection = np.asarray(
            fixture["expectedProjection"],
            dtype=np.float64,
        )

        np.testing.assert_allclose(
            expected_projection,
            projection,
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            np.asarray(fixture["expectedStats"], dtype=np.float64),
            contact_size_proxy.stats_features(projection[None, ...])[0],
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )
        self.assertTrue(np.isnan(projection[3:6, 8]).all())
        self.assertTrue(np.isnan(projection[6:9, 9]).all())

    def test_training_source_contract_rejects_contract_name_order_and_shape(self):
        valid = self.causal_row("phone-a", 1, 100.0)
        invalid_contract = copy.deepcopy(valid)
        invalid_contract["featureContract"] = contract.SOURCE_FEATURE_CONTRACT
        with self.assertRaisesRegex(
            RuntimeError,
            "invalid_source_feature_contract",
        ):
            trainer.prepare_causal_source_matrices([invalid_contract])

        invalid_names = copy.deepcopy(valid)
        invalid_names["featureNames"][12] = "wrong_size"
        with self.assertRaisesRegex(
            RuntimeError,
            "invalid_source_feature_names",
        ):
            trainer.prepare_causal_source_matrices([invalid_names])

        invalid_shape = copy.deepcopy(valid)
        invalid_shape["matrix"] = invalid_shape["matrix"][:-1]
        with self.assertRaisesRegex(
            RuntimeError,
            "invalid_source_matrix",
        ):
            trainer.prepare_causal_source_matrices([invalid_shape])

    def test_light_touches_are_negative_and_heavy_touches_are_positive(self):
        self.assertEqual(0, contract.class_strength_target("thumb_light"))
        self.assertEqual(0, contract.class_strength_target("index_light"))
        self.assertEqual(1, contract.class_strength_target("thumb_heavy"))
        self.assertEqual(1, contract.class_strength_target("index_heavy"))
        self.assertEqual(0, contract.class_finger_target("thumb_heavy"))
        self.assertEqual(1, contract.class_finger_target("index_light"))
        with self.assertRaises(ValueError):
            contract.class_strength_target("ordinary_touch")

    def test_gate_weights_balance_light_and_heavy_without_finger_split(self):
        samples = [
            {"groupId": "a", "label": 0},
            {"groupId": "a", "label": 0},
            {"groupId": "a", "label": 0},
            {"groupId": "b", "label": 1},
        ]

        weights = trainer.gate_sample_weights(samples)

        self.assertAlmostEqual(
            float(weights[:3].sum()),
            float(weights[3]),
            places=6,
        )
        self.assertTrue((weights > 0).all())

    def test_gate_metrics_ignore_finger_class(self):
        metrics = trainer.gate_metrics(
            [0, 0, 1, 1],
            np.asarray([
                [0.9, 0.1],
                [0.8, 0.2],
                [0.3, 0.7],
                [0.6, 0.4],
            ]),
        )

        self.assertEqual(
            {"light": 1.0, "heavy": 0.5},
            metrics["recallByClass"],
        )
        self.assertAlmostEqual(0.75, metrics["balancedAccuracy"])

    def test_runtime_state_drops_all_context_heads(self):
        state = {
            "shared.0.weight": torch.zeros((24, 60)),
            "shared.0.bias": torch.zeros(24),
            "shared.3.weight": torch.zeros((24, 24)),
            "shared.3.bias": torch.zeros(24),
            "finger.weight": torch.zeros((2, 24)),
            "finger.bias": torch.zeros(2),
            "strength.weight": torch.zeros((2, 24)),
            "strength.bias": torch.zeros(2),
            "context.0.weight": torch.zeros((2, 24)),
            "context.0.bias": torch.zeros(2),
        }

        runtime_state = trainer.runtime_state_dict(state)

        self.assertEqual(
            set(trainer.RUNTIME_STATE_NAMES),
            set(runtime_state),
        )
        self.assertFalse(any(
            name.startswith("context.")
            for name in runtime_state
        ))

    def test_stats_values_follow_channel_major_feature_names(self):
        values = np.zeros((1, contract.FRAME_COUNT, 10), dtype=np.float32)
        values[0, :, 0] = np.arange(contract.FRAME_COUNT, dtype=np.float32)
        values[0, :, 1] = np.arange(
            contract.FRAME_COUNT,
            dtype=np.float32,
        ) + 100.0

        stats = trainer.lightweight.stats_features(values)

        self.assertAlmostEqual(
            float(values[0, :, 0].mean()),
            float(stats[0, 0]),
            places=6,
        )
        self.assertAlmostEqual(
            float(values[0, :, 0].std()),
            float(stats[0, 1]),
            places=6,
        )
        self.assertAlmostEqual(
            float(values[0, :, 1].mean()),
            float(stats[0, 5]),
            places=6,
        )
        self.assertTrue(
            contract.STATS_FEATURE_NAMES[0].startswith(
                contract.PHYSICAL_FEATURE_NAMES[0] + "."
            )
        )
        self.assertTrue(
            contract.STATS_FEATURE_NAMES[5].startswith(
                contract.PHYSICAL_FEATURE_NAMES[1] + "."
            )
        )

    def test_export_writes_runtime_assets_and_disables_deployment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, model_contract = self.valid_payload()
            self.write_source(root, weights, model_contract)

            weights_name, scaler_name = exporter.export_model(
                root,
                root / "assets",
                "candidate",
            )

            self.assertEqual(
                "AispectStatsMlpWeights_candidate.json",
                weights_name,
            )
            self.assertEqual(
                "AispectStatsMlpScaler_candidate.json",
                scaler_name,
            )
            exported = json.loads(
                (root / "assets" / scaler_name).read_text(encoding="utf-8")
            )
            self.assertFalse(exported["deploymentAllowed"])
            self.assertFalse(exported["dropoutAtInference"])
            self.assertEqual(
                "all_data_fit_only",
                exported["evaluationScope"],
            )
            self.assertEqual(
                contract.PRIMARY_TASK,
                exported["primaryTask"],
            )
            self.assertEqual(
                list(contract.GATE_LABELS),
                exported["labelOrder"],
            )

    def test_export_rejects_swapped_gate_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, model_contract = self.valid_payload()
            model_contract["labelOrder"] = ["heavy", "light"]
            self.write_source(root, weights, model_contract)

            with self.assertRaisesRegex(
                RuntimeError,
                "invalid_label_order",
            ):
                exporter.export_model(root, root / "assets", "candidate")

    def test_export_rejects_wrong_stat_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, model_contract = self.valid_payload()
            model_contract = copy.deepcopy(model_contract)
            model_contract["statsFeatureNames"][0] = "wrong.order"
            self.write_source(root, weights, model_contract)

            with self.assertRaisesRegex(
                RuntimeError,
                "invalid_stats_feature_names",
            ):
                exporter.export_model(root, root / "assets", "candidate")

    def test_export_rejects_dropout_or_wrong_weight_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, model_contract = self.valid_payload()
            model_contract["dropoutAtInference"] = True
            self.write_source(root, weights, model_contract)

            with self.assertRaisesRegex(
                RuntimeError,
                "dropout_must_be_disabled_at_inference",
            ):
                exporter.export_model(root, root / "assets", "candidate")

            _, model_contract = self.valid_payload()
            weights["shared.0.weight"]["shape"] = [24, 59]
            self.write_source(root, weights, model_contract)
            with self.assertRaisesRegex(
                RuntimeError,
                "invalid_weight_shape_shared_0_weight",
            ):
                exporter.export_model(root, root / "assets", "candidate")

    def test_export_rejects_wrong_frozen_counts(self):
        mutations = (
            ("frameCount", 8, "invalid_frame_count"),
            (
                "sequenceChannelCount",
                9,
                "invalid_sequence_channel_count",
            ),
            ("statsFeatureCount", 59, "invalid_stats_feature_count"),
            ("inputCount", 61, "invalid_input_count"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for field, value, error in mutations:
                weights, model_contract = self.valid_payload()
                model_contract[field] = value
                self.write_source(root, weights, model_contract)
                with self.subTest(field=field):
                    with self.assertRaisesRegex(RuntimeError, error):
                        exporter.export_model(
                            root,
                            root / "assets",
                            "candidate",
                        )

    def test_export_rejects_non_finite_weight_value(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, model_contract = self.valid_payload()
            weights["shared.0.weight"]["values"][0] = np.nan
            self.write_source(root, weights, model_contract)

            with self.assertRaisesRegex(
                RuntimeError,
                "invalid_weight_value_shared_0_weight",
            ):
                exporter.export_model(root, root / "assets", "candidate")

    def test_export_rejects_invalid_sequence_or_stats_scales(self):
        mutations = (
            (
                "sequence_channel_scale",
                "invalid_sequence_channel_scale",
            ),
            ("stats_scaler_scale", "invalid_stats_scaler_scale"),
        )
        values = (0.0, np.nan)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for location, error in mutations:
                for value in values:
                    weights, model_contract = self.valid_payload()
                    if location == "sequence_channel_scale":
                        model_contract["sequenceChannels"][8]["scale"] = value
                    else:
                        model_contract["statsScaler"]["scale"][0] = value
                    self.write_source(root, weights, model_contract)
                    with self.subTest(location=location, value=value):
                        with self.assertRaisesRegex(RuntimeError, error):
                            exporter.export_model(
                                root,
                                root / "assets",
                                "candidate",
                            )

    def test_export_rejects_v4_with_legacy_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, model_contract = self.valid_payload()
            model_contract["runtimeArchitecture"] = (
                contract.LEGACY_RUNTIME_ARCHITECTURE
            )
            self.write_source(root, weights, model_contract)

            with self.assertRaisesRegex(
                RuntimeError,
                "invalid_causal_size_architecture",
            ):
                exporter.export_model(root, root / "assets", "candidate")

    def test_export_rejects_size_reinterpreted_as_force(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, model_contract = self.valid_payload()
            model_contract["forceMeasurement"] = True
            self.write_source(root, weights, model_contract)

            with self.assertRaisesRegex(
                RuntimeError,
                "size_must_not_be_force_measurement",
            ):
                exporter.export_model(root, root / "assets", "candidate")

    def test_export_rejects_missing_completed_touch_history_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, model_contract = self.valid_payload()
            model_contract["sizeBaseline"].pop("historyScope")
            self.write_source(root, weights, model_contract)

            with self.assertRaisesRegex(
                RuntimeError,
                "invalid_size_baseline_policy",
            ):
                exporter.export_model(root, root / "assets", "candidate")

    def test_export_rejects_wrong_contact_size_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, model_contract = self.valid_payload()
            model_contract["sizeSemantics"] = "force"
            self.write_source(root, weights, model_contract)

            with self.assertRaisesRegex(
                RuntimeError,
                "invalid_size_semantics",
            ):
                exporter.export_model(root, root / "assets", "candidate")

    def test_export_rejects_current_touch_in_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, model_contract = self.valid_payload()
            model_contract["sizeBaseline"]["currentTouchExcluded"] = False
            self.write_source(root, weights, model_contract)

            with self.assertRaisesRegex(
                RuntimeError,
                "invalid_size_baseline_policy",
            ):
                exporter.export_model(root, root / "assets", "candidate")

    def test_export_rejects_session_scoped_size_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, model_contract = self.valid_payload()
            model_contract["sizeBaseline"]["deviceScope"] = "session"
            self.write_source(root, weights, model_contract)

            with self.assertRaisesRegex(
                RuntimeError,
                "invalid_size_baseline_policy",
            ):
                exporter.export_model(root, root / "assets", "candidate")

    def test_training_preparation_accepts_json_rows_and_keeps_contract_shape(self):
        matrix = []
        for frame in range(9):
            row = [0.0] * len(contract.SOURCE_FEATURE_NAMES)
            row[0] = float(frame)
            row[1] = float(frame + 1)
            row[2:8] = [0.1 * frame, 0.2 * frame, 0.3 * frame,
                        0.4 * frame, 0.5 * frame, 0.6 * frame]
            row[contract.SOURCE_FEATURE_NAMES.index("size")] = 0.5
            row[contract.SOURCE_FEATURE_NAMES.index("log_axis_ratio")] = 0.25
            row[contract.SOURCE_FEATURE_NAMES.index("touch_valid")] = 1.0
            row[contract.SOURCE_FEATURE_NAMES.index("size_available")] = 1.0
            matrix.append(row)
        rows = []
        labels = [
            "thumb_light",
            "thumb_heavy",
            "index_light",
            "index_heavy",
        ]
        for index in range(8):
            rows.append(self.causal_row("phone-a", index + 1, 0.5))
        for index, label in enumerate(labels):
            row = self.causal_row("phone-a", index + 9, 0.5)
            row["classLabel"] = label
            row["groupId"] = "session-%d" % index
            rows.append(row)

        prepared = trainer.prepare_all_data(rows)

        self.assertEqual((12, 60), prepared["stats"].shape)
        self.assertEqual(60, prepared["statsCenter"].shape[0])
        self.assertEqual(60, prepared["statsScale"].shape[0])
        samples = [
            {
                "groupId": row["groupId"],
                "label": contract.class_strength_target(row["classLabel"]),
            }
            for row in rows[8:]
        ]
        weights = trainer.gate_sample_weights(samples)
        self.assertEqual(4, weights.shape[0])
        self.assertTrue((weights > 0).all())

    def causal_row(
        self,
        device,
        timestamp,
        size,
        session_id=None,
        size_available=True,
        major=1.0,
        minor=0.5,
        source_file=None,
        include_timestamp=True,
    ):
        matrix = []
        for _ in range(contract.FRAME_COUNT):
            row = [0.0] * len(contract.SOURCE_FEATURE_NAMES)
            row[contract.SOURCE_FEATURE_NAMES.index("size")] = size
            row[contract.SOURCE_FEATURE_NAMES.index("touch_valid")] = 1.0
            row[contract.SOURCE_FEATURE_NAMES.index("size_available")] = (
                1.0 if size_available else 0.0
            )
            row[contract.SOURCE_FEATURE_NAMES.index("major_norm")] = major
            row[contract.SOURCE_FEATURE_NAMES.index("minor_norm")] = minor
            row[contract.SOURCE_FEATURE_NAMES.index("major_available")] = 1.0
            row[contract.SOURCE_FEATURE_NAMES.index("minor_available")] = 1.0
            row[contract.SOURCE_FEATURE_NAMES.index("log_axis_ratio")] = (
                np.log(major / minor)
            )
            matrix.append(row)
        result = {
            "platform": "android",
            "deviceName": device,
            "collectorDeviceId": device,
            "sessionId": session_id or "session-" + str(timestamp),
            "groupId": session_id or "session-" + str(timestamp),
            "sourceFile": source_file or ("touch-%08d.json" % timestamp),
            "cycleIndex": timestamp,
            "stepIndex": 0,
            "sourceSha256": "%064d" % timestamp,
            "sampleId": "sample-%d" % timestamp,
            "classLabel": "thumb_light",
            "featureContract": contract.LEGACY_V3_SOURCE_FEATURE_CONTRACT,
            "featureNames": list(contract.SOURCE_FEATURE_NAMES),
            "matrix": matrix,
        }
        if include_timestamp:
            result["sourceTimestampMs"] = timestamp
        return result

    def test_causal_size_history_spans_sessions_on_same_device(self):
        rows = [
            self.causal_row(
                "phone-a",
                index + 1,
                100.0,
                session_id="session-%d" % index,
            )
            for index in range(8)
        ]
        rows.append(self.causal_row("phone-a", 9, 200.0))

        physical = trainer.prepare_causal_source_matrices(rows)

        self.assertTrue(np.isnan(physical[7, :, 8]).all())
        self.assertAlmostEqual(
            np.log(2.0),
            float(physical[8, 0, 8]),
            places=6,
        )

    def test_causal_size_ratio_is_invariant_to_device_scale(self):
        rows = []
        for device, history_size in (("phone-a", 100.0), ("phone-b", 10000.0)):
            rows.extend(
                self.causal_row(device, index + 1, history_size)
                for index in range(8)
            )
            rows.append(self.causal_row(device, 9, history_size * 2.0))

        physical = trainer.prepare_causal_source_matrices(rows)

        self.assertAlmostEqual(
            float(physical[8, 0, 8]),
            float(physical[17, 0, 8]),
            places=6,
        )

    def test_causal_size_history_separates_same_model_collectors(self):
        rows = []
        for collector, offset, history_size in (
            ("collector-a", 0, 100.0),
            ("collector-b", 100, 1000.0),
        ):
            device_rows = []
            for index in range(8):
                row = self.causal_row(
                    "same-model",
                    offset + index + 1,
                    history_size,
                )
                row["collectorDeviceId"] = collector
                device_rows.append(row)
            current = self.causal_row(
                "same-model",
                offset + 9,
                history_size * 2.0,
            )
            current["collectorDeviceId"] = collector
            device_rows.append(current)
            rows.extend(device_rows)

        physical = trainer.prepare_causal_source_matrices(rows)

        self.assertAlmostEqual(
            np.log(2.0),
            float(physical[8, 0, 8]),
            places=6,
        )
        self.assertAlmostEqual(
            np.log(2.0),
            float(physical[17, 0, 8]),
            places=6,
        )

    def test_causal_size_history_falls_back_to_model_without_collector_id(self):
        rows = [
            self.causal_row("phone-a", index + 1, 100.0)
            for index in range(8)
        ]
        rows.append(self.causal_row("phone-a", 9, 200.0))
        for row in rows:
            row.pop("collectorDeviceId")

        physical = trainer.prepare_causal_source_matrices(rows)

        self.assertAlmostEqual(
            np.log(2.0),
            float(physical[8, 0, 8]),
            places=6,
        )

    def test_causal_size_history_does_not_use_future_touch(self):
        rows = [
            self.causal_row("phone-a", index + 1, 100.0)
            for index in range(8)
        ]
        rows.append(self.causal_row("phone-a", 9, 200.0))

        physical = trainer.prepare_causal_source_matrices(rows)

        self.assertAlmostEqual(
            np.log(2.0),
            float(physical[8, 0, 8]),
            places=6,
        )

    def test_causal_size_history_rejects_nonpositive_and_missing_sizes(self):
        rows = [
            self.causal_row("phone-a", 1, 0.0),
            self.causal_row("phone-a", 2, -1.0),
            self.causal_row("phone-a", 3, np.nan),
            self.causal_row("phone-a", 4, 50.0, size_available=False),
        ]
        rows.extend(
            self.causal_row("phone-a", index + 5, 100.0)
            for index in range(4)
        )
        rows.append(self.causal_row("phone-a", 9, 200.0))

        physical = trainer.prepare_causal_source_matrices(rows)

        self.assertTrue(np.isnan(physical[7, :, 8]).all())
        self.assertTrue(np.isnan(physical[8, :, 8]).all())

    def test_missing_size_keeps_other_statistics_available(self):
        values = np.full(
            (1, contract.FRAME_COUNT, len(contract.PHYSICAL_FEATURE_NAMES)),
            np.nan,
            dtype=np.float32,
        )
        values[0, :, 0] = np.arange(contract.FRAME_COUNT)
        values[0, :, 2:8] = 0.5

        stats = contact_size_proxy.stats_features(values)

        self.assertAlmostEqual(4.0, float(stats[0, 0]), places=6)
        self.assertTrue(np.isfinite(stats[0, 50]))
        self.assertTrue(np.isfinite(stats[0, 55]))

    def test_natural_source_file_order_avoids_lexical_misorder(self):
        rows = []
        for index in range(1, 10):
            rows.append(
                self.causal_row(
                    "phone-a",
                    index,
                    100.0 if index < 9 else 200.0,
                    include_timestamp=False,
                    source_file="capture/frame-%d.json" % index,
                )
            )

        physical = trainer.prepare_causal_source_matrices(rows)

        self.assertAlmostEqual(
            np.log(2.0),
            float(physical[8, 0, 8]),
            places=6,
        )

    def test_causal_order_uses_natural_numeric_file_order(self):
        early = {
            "platform": "android",
            "deviceName": "phone-a",
            "sourceFile": "capture/frame-2.json",
        }
        late = {
            "platform": "android",
            "deviceName": "phone-a",
            "sourceFile": "capture/frame-10.json",
        }

        self.assertLess(
            contact_size_proxy.causal_order_key(early),
            contact_size_proxy.causal_order_key(late),
        )

    def test_shared_fixture_projection_and_statistics(self):
        fixture_path = (
            Path(__file__).resolve().parents[2]
            / "aispect-core"
            / "src"
            / "test"
            / "resources"
            / "stats_mlp_v4_golden_fixture.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        source = np.asarray(fixture["sourceMatrix"], dtype=np.float64)
        expected_projection = np.asarray(
            fixture["expectedProjection"],
            dtype=np.float64,
        )
        expected_stats = np.asarray(
            fixture["expectedStats"],
            dtype=np.float64,
        )

        projection = contact_size_proxy.transform_row(
            {
                "matrix": source,
                "featureNames": fixture["sourceFeatureNames"],
            },
            fixture["baselineHistory"],
        )
        stats = contact_size_proxy.stats_features(projection[None, ...])[0]

        np.testing.assert_allclose(
            projection,
            expected_projection,
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            stats,
            expected_stats,
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertTrue(np.isnan(projection[3:6, 8]).all())
        self.assertTrue(np.isnan(projection[6:9, 9]).all())

    def test_training_source_matrix_rejects_contract_drift(self):
        row = self.causal_row("phone-a", 1, 100.0)

        invalid_contract = copy.deepcopy(row)
        invalid_contract["featureContract"] = contract.SOURCE_FEATURE_CONTRACT
        with self.assertRaisesRegex(
            RuntimeError,
            "invalid_source_feature_contract",
        ):
            trainer.source_matrix(invalid_contract)

        invalid_names = copy.deepcopy(row)
        invalid_names["featureNames"][12] = "wrong_name"
        with self.assertRaisesRegex(
            RuntimeError,
            "invalid_source_feature_names",
        ):
            trainer.source_matrix(invalid_names)

        invalid_shape = copy.deepcopy(row)
        invalid_shape["matrix"] = invalid_shape["matrix"][:-1]
        with self.assertRaisesRegex(
            RuntimeError,
            "invalid_source_matrix",
        ):
            trainer.source_matrix(invalid_shape)


if __name__ == "__main__":
    unittest.main()
