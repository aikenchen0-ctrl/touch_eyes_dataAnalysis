import json
import tempfile
import unittest
from pathlib import Path

from android.analysis import export_time_grid_groupnorm_android_model as exporter
from android.analysis import train_pose_invariant_variants as trainer


class PoseInvariantVariantTests(unittest.TestCase):
    def setUp(self):
        self.matrix = []
        for index in range(9):
            row = [0.0] * 26
            row[0] = index / 10.0
            row[8:11] = [float(index), float(index * 2), float(index * 3)]
            row[19] = 1.0
            self.matrix.append(row)

    def test_no_gravity_variant_removes_absolute_gravity(self):
        result = trainer.build_variant_matrix(self.matrix, "no_gravity")

        self.assertEqual(20, len(result[0]))
        self.assertNotIn("gravity.x", trainer.VARIANTS["no_gravity"]["featureNames"])
        self.assertEqual(0.0, result[4][8])
        self.assertEqual(0.4, result[4][0])

    def test_delta_gravity_is_causal_and_first_slot_is_zero(self):
        result = trainer.build_variant_matrix(self.matrix, "delta_gravity")
        names = trainer.VARIANTS["delta_gravity"]["featureNames"]

        self.assertEqual(23, len(result[0]))
        self.assertEqual("deltaGravity.x", names[8])
        self.assertEqual([0.0, 0.0, 0.0], result[0][8:11])
        self.assertEqual([1.0, 2.0, 3.0], result[1][8:11])

        future_changed = [list(row) for row in self.matrix]
        future_changed[8][8:11] = [900.0, 900.0, 900.0]
        changed_result = trainer.build_variant_matrix(future_changed, "delta_gravity")
        self.assertEqual(result[1][8:11], changed_result[1][8:11])

    def test_invalid_matrix_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "invalid_touch26_matrix_shape"):
            trainer.build_variant_matrix(self.matrix[:8], "no_gravity")

    def test_android_export_preserves_pose_invariant_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            assets = root / "assets"
            source.mkdir()
            feature_names = list(trainer.VARIANTS["no_gravity"]["featureNames"])
            weights = {
                "conv1.weight": {"shape": [32, 20, 3], "values": []},
                "conv1.bias": {"shape": [32], "values": []},
                "norm1.weight": {"shape": [32], "values": []},
                "norm1.bias": {"shape": [32], "values": []},
                "conv2.weight": {"shape": [32, 32, 3], "values": []},
                "conv2.bias": {"shape": [32], "values": []},
                "norm2.weight": {"shape": [32], "values": []},
                "norm2.bias": {"shape": [32], "values": []},
                "output.weight": {"shape": [4, 32], "values": []},
                "output.bias": {"shape": [4], "values": []},
            }
            (source / "weights.json").write_text(json.dumps(weights), encoding="utf-8")
            (source / "scaler.json").write_text(json.dumps({
                "featureNames": feature_names,
                "channels": [{"name": name} for name in feature_names],
            }), encoding="utf-8")
            (source / "model_contract.json").write_text(json.dumps({
                "featureContract": trainer.VARIANTS["no_gravity"]["featureContract"],
                "featureNames": feature_names,
                "matrixShape": [9, 20],
            }), encoding="utf-8")

            _, scaler_name = exporter.export_model(source, assets, "pose_test")
            exported = json.loads((assets / scaler_name).read_text(encoding="utf-8"))

            self.assertEqual([9, 20], exported["matrixShape"])
            self.assertEqual(trainer.VARIANTS["no_gravity"]["featureContract"], exported["featureContract"])


if __name__ == "__main__":
    unittest.main()
