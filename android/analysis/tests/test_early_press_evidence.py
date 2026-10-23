import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from android.analysis import build_early_press_evidence
from android.analysis import early_press_evidence


def sensor_frame(timestamp_ns, acceleration, rotation=1.0):
    return {
        "sensorTimestampElapsedRealtimeNanos": timestamp_ns,
        "receivedElapsedRealtimeNanos": timestamp_ns + 1_000_000,
        "x": acceleration,
        "y": 0.0,
        "z": 0.0,
        "rotationRateX": rotation,
        "rotationRateY": 0.0,
        "rotationRateZ": 0.0,
        "gravityX": 0.0,
        "gravityY": 0.0,
        "gravityZ": 9.8,
    }


def evidence_fixture():
    down_ns = 1_000_000_000
    frames = []
    for offset_ms in range(-100, 41, 5):
        frames.append(sensor_frame(
            down_ns + offset_ms * 1_000_000,
            1.0 + max(offset_ms + 70, 0) / 20.0,
        ))
    return {
        "sensorMetadata": {
            "platform": "android",
            "sensorSource": "linear_acceleration",
            "accelerationUnit": "m/s^2",
            "rotationRateUnit": "rad/s",
            "sensorAxisFrame": "canonical_device",
            "transformVersion": "test_v1",
        },
        "timeAlignment": {
            "downEventElapsedRealtimeNanos": down_ns,
            "liftEventElapsedRealtimeNanos": down_ns + 200_000_000,
        },
        "frames": frames,
    }


class EarlyPressEvidenceTests(unittest.TestCase):
    def test_contract_has_fixed_causal_feature_order(self):
        self.assertEqual(
            (-70, -60, -50, -40, -30, -20, 0, 5, 10),
            early_press_evidence.OFFSETS_MS,
        )
        self.assertEqual(
            43,
            len(early_press_evidence.FEATURE_NAMES),
        )
        self.assertEqual(
            "evidence.available",
            early_press_evidence.FEATURE_NAMES[20],
        )
        self.assertEqual(
            "dense_evidence.available",
            early_press_evidence.FEATURE_NAMES[-1],
        )

    def test_frames_after_training_horizon_do_not_change_evidence(self):
        fixture = evidence_fixture()
        baseline = early_press_evidence.build_evidence(fixture)
        fixture["frames"].extend([
            sensor_frame(
                1_000_000_000 + offset * 1_000_000,
                9999.0,
            )
            for offset in (11, 12, 13)
        ])
        changed = early_press_evidence.build_evidence(fixture)
        self.assertEqual(baseline["values"], changed["values"])

    def test_missing_left_support_returns_masked_row(self):
        fixture = evidence_fixture()
        fixture["frames"] = [
            frame for frame in fixture["frames"]
            if frame["sensorTimestampElapsedRealtimeNanos"] >= 970_000_000
        ]
        result = early_press_evidence.build_evidence(fixture)
        self.assertEqual(
            [0.0] * len(early_press_evidence.FEATURE_NAMES),
            result["values"],
        )

    def test_normalization_fits_only_available_fit_rows(self):
        fit = np.zeros(
            (3, len(early_press_evidence.FEATURE_NAMES)),
            dtype=np.float32,
        )
        fit[0, 0] = 0.0
        fit[1, 0] = 2.0
        fit[0:2, 20] = 1.0
        target = np.zeros(
            (2, len(early_press_evidence.FEATURE_NAMES)),
            dtype=np.float32,
        )
        target[:, 20] = 1.0
        target[0, 0] = 1.0
        target[1, 0] = 1000.0

        first, metadata = early_press_evidence.normalize(fit, target)
        target[1, 0] = -1000.0
        second, _ = early_press_evidence.normalize(fit, target)

        self.assertEqual(2, metadata["fitAvailableCount"])
        self.assertEqual(1, metadata["fitMissingCount"])
        self.assertEqual(2, metadata["fitAvailableCount"])
        self.assertEqual(1, metadata["fitMissingCount"])
        self.assertEqual(2, metadata["availableCount"])
        self.assertEqual(0, metadata["missingCount"])
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.all(np.abs(first) <= early_press_evidence.CLIP_MAX))

    def test_ablation_modes_select_frozen_columns(self):
        values = np.arange(
            len(early_press_evidence.FEATURE_NAMES),
            dtype=np.float32,
        ).reshape(1, -1)
        values[0, 20] = 1.0

        acceleration = early_press_evidence.select_mode(
            values,
            "acceleration_shape",
        )
        onset = early_press_evidence.select_mode(
            values,
            "acceleration_onset",
        )

        self.assertEqual(11, acceleration.shape[1])
        self.assertEqual(7, onset.shape[1])
        self.assertEqual(1.0, float(acceleration[0, -1]))
        self.assertEqual(1.0, float(onset[0, -1]))
        self.assertEqual(
            [
                "acceleration.log_ratio.-40",
                "acceleration.log_ratio.-30",
                "acceleration.log_ratio.-20",
                "acceleration.early_slope",
                "acceleration.rise_fraction",
                "acceleration.early_peak_log_ratio",
                "evidence.available",
            ],
            list(early_press_evidence.selected_feature_names(
                "acceleration_onset",
            )),
        )

    def test_ablation_modes_have_distinct_frozen_widths(self):
        widths = {
            mode: len(early_press_evidence.selected_feature_names(mode))
            for mode in early_press_evidence.EVIDENCE_MODES
        }

        self.assertEqual(21, widths["shape"])
        self.assertEqual(11, widths["acceleration"])
        self.assertEqual(11, widths["rotation"])
        self.assertEqual(7, widths["acceleration_onset"])
        self.assertEqual(10, widths["onset_shape"])
        self.assertEqual(11, widths["rotation_peak_shape"])
        self.assertEqual(10, widths["rotation_peak_shape_no_gain"])
        self.assertEqual(22, widths["dense_shape"])
        self.assertEqual(10, widths["dense_acceleration"])
        self.assertEqual(10, widths["dense_rotation"])
        self.assertEqual(4, widths["dense_cross"])
        self.assertEqual(19, widths["dense_shape_no_gain"])
        self.assertEqual(14, widths["dense_scale_invariant"])
        self.assertEqual(11, widths["dense_scale_invariant_no_cross"])
        self.assertEqual(24, widths["rotation_plus_dense_scale_invariant"])

    def test_evidence_has_exact_frozen_width(self):
        evidence = early_press_evidence.build_evidence(
            evidence_fixture()
        )

        self.assertEqual(
            len(early_press_evidence.FEATURE_NAMES),
            len(evidence["values"]),
        )
        self.assertEqual(3, len(early_press_evidence.cross_shape_features(
            np.asarray((0.1, 0.2, 1.0, 2.0, 3.0, 4.0, 0.2, 0.5, 0.7)),
            np.asarray((0.2, 0.3, 1.0, 2.0, 3.0, 4.0, 0.2, 0.5, 0.7)),
        )))

    def test_dense_shape_is_invariant_to_signal_gain(self):
        first = early_press_evidence.dense_shape_features(np.asarray(
            (
                0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2,
                1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.08, 0.05,
            ),
            dtype=np.float64,
        ))
        second = early_press_evidence.dense_shape_features(np.asarray(
            (
                0.5, 0.5, 0.5, 1.0, 1.5, 2.5, 4.0, 5.0, 6.0,
                5.0, 3.5, 2.5, 1.5, 1.0, 0.5, 0.4, 0.25,
            ),
            dtype=np.float64,
        ))

        # 中文注释：峰值位置、onset 位置、能量中心和归一化 jerk
        # 只描述时序；面积、绝对 jerk 和峰值比仍显式保留在合同内。
        for index in (0, 1, 7):
            self.assertAlmostEqual(first[index], second[index], places=4)

    def test_rotation_peak_shape_separates_gain_from_timing(self):
        values = np.zeros(
            (2, len(early_press_evidence.FEATURE_NAMES)),
            dtype=np.float64,
        )
        values[:, 10:16] = np.asarray((
            (0.2, 0.4, 0.6, 0.8, 0.4, 0.2),
            (0.4, 0.8, 1.2, 1.6, 0.8, 0.4),
        ))
        values[:, 17] = 0.75
        values[:, 18] = (0.8, 1.6)
        values[:, 19] = (0.2, 0.4)
        values[:, -1] = 1.0

        without_gain = early_press_evidence.select_mode(
            values,
            "rotation_peak_shape_no_gain",
        )
        with_gain = early_press_evidence.select_mode(
            values,
            "rotation_peak_shape",
        )

        self.assertTrue(np.allclose(
            without_gain[0, :9],
            without_gain[1, :9],
        ))
        self.assertAlmostEqual(0.6, float(without_gain[0, 7]))
        self.assertNotAlmostEqual(
            float(with_gain[0, 9]),
            float(with_gain[1, 9]),
        )

    def test_ablation_mode_rejects_unknown_name(self):
        with self.assertRaisesRegex(
                RuntimeError,
                "unsupported_early_evidence_mode"):
            early_press_evidence.selected_feature_names("unknown")

    def test_sidecar_builder_rejects_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            source = raw / "sample.json"
            source.write_text(
                json.dumps(evidence_fixture()),
                encoding="utf-8",
            )
            row = {
                "sourceSha256": "0" * 64,
                "sourceFile": "sample.json",
                "sampleId": "sample-a",
            }
            with self.assertRaisesRegex(
                    RuntimeError,
                    "raw_source_hash_mismatch"):
                build_early_press_evidence.build_one(row, raw)

    def test_sidecar_builder_uses_verified_manifest_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "elsewhere" / "sample.json"
            source.parent.mkdir()
            source.write_text(
                json.dumps(evidence_fixture()),
                encoding="utf-8",
            )
            source_sha256 = build_early_press_evidence.sha256_file(source)
            manifest = root / "old.jsonl"
            manifest.write_text(
                json.dumps({
                    "sourceSha256": source_sha256,
                    "sourceFile": str(source),
                }) + "\n",
                encoding="utf-8",
            )
            row = {
                "sourceSha256": source_sha256,
                "sourceFile": "stale-relative-name.json",
                "sampleId": "sample-a",
            }

            with mock.patch.object(
                    build_early_press_evidence.early_press_evidence,
                    "build_evidence",
                    return_value={
                        "contract": early_press_evidence.CONTRACT,
                        "offsetsMs": list(
                            early_press_evidence.OFFSETS_MS
                        ),
                        "featureNames": list(
                            early_press_evidence.FEATURE_NAMES
                        ),
                        "values": [0.0] * (
                            len(early_press_evidence.FEATURE_NAMES) - 1
                        ) + [1.0],
                    }):
                result = build_early_press_evidence.build_one(
                    row,
                    None,
                    source_manifest=(
                        build_early_press_evidence.source_manifest_index(
                            manifest
                        )
                    ),
                )

            self.assertEqual(str(source.resolve()), result["sourceFile"])

    def test_sidecar_builder_rejects_conflicting_manifest_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "old.jsonl"
            rows = [
                {
                    "sourceSha256": "a" * 64,
                    "sourceFile": str(root / "a.json"),
                },
                {
                    "sourceSha256": "a" * 64,
                    "sourceFile": str(root / "b.json"),
                },
            ]
            manifest.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                    RuntimeError,
                    "source_manifest_conflict"):
                build_early_press_evidence.source_manifest_index(
                    manifest
                )


if __name__ == "__main__":
    unittest.main()
