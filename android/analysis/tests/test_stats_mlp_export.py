import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from android.analysis import export_stats_mlp_android_model as exporter
from android.analysis import stats_mlp_contract as contract
from android.analysis import train_stats_mlp_android_candidate as trainer


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
            "sourceFeatureNames": list(contract.SOURCE_FEATURE_NAMES),
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
            "causal_time_grid_imu11_touch7_mask5_stats_mlp_v3",
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
        self.assertEqual(("light", "heavy"), contract.GATE_LABELS)
        self.assertEqual(
            ("thumb_light", "index_light"),
            contract.NEGATIVE_CLASS_LABELS,
        )
        self.assertEqual(
            ("thumb_heavy", "index_heavy"),
            contract.POSITIVE_CLASS_LABELS,
        )

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

    def test_training_preparation_accepts_json_rows_and_keeps_contract_shape(self):
        matrix = []
        for frame in range(9):
            row = [0.0] * 26
            row[0] = float(frame)
            row[1] = float(frame + 1)
            row[2:8] = [0.1 * frame, 0.2 * frame, 0.3 * frame,
                        0.4 * frame, 0.5 * frame, 0.6 * frame]
            row[12] = 0.5
            row[15] = 0.25
            row[20:26] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            matrix.append(row)
        rows = [
            {"matrix": matrix, "classLabel": "thumb_light", "groupId": "session-a"},
            {"matrix": matrix, "classLabel": "thumb_heavy", "groupId": "session-b"},
            {"matrix": matrix, "classLabel": "index_light", "groupId": "session-c"},
            {"matrix": matrix, "classLabel": "index_heavy", "groupId": "session-b"},
        ]

        prepared = trainer.prepare_all_data(rows)

        self.assertEqual((4, 60), prepared["stats"].shape)
        self.assertEqual(60, prepared["statsCenter"].shape[0])
        self.assertEqual(60, prepared["statsScale"].shape[0])
        samples = [
            {
                "groupId": row["groupId"],
                "label": contract.class_strength_target(row["classLabel"]),
            }
            for row in rows
        ]
        weights = trainer.gate_sample_weights(samples)
        self.assertEqual(4, weights.shape[0])
        self.assertTrue((weights > 0).all())


if __name__ == "__main__":
    unittest.main()
