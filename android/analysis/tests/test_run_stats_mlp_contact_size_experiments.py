import unittest
import csv
import json
import tempfile
from pathlib import Path

import numpy as np

try:
    from android.analysis import run_stats_mlp_contact_size_experiments as subject
except ImportError:
    import run_stats_mlp_contact_size_experiments as subject


def make_sample(
    source_file,
    sizes,
    available=None,
    major=None,
    major_available=None,
    group_id="session",
    declared_major_max=None,
    device_name="device",
):
    matrix = np.zeros((9, 26), dtype=np.float32)
    if available is None:
        available = [value > 0.0 for value in sizes]
    matrix[:, subject.SOURCE_SIZE_CHANNEL] = sizes
    matrix[:, subject.SOURCE_SIZE_AVAILABLE_CHANNEL] = np.asarray(
        available,
        dtype=np.float32,
    )
    if major is not None:
        matrix[:, subject.SOURCE_MAJOR_CHANNEL] = major
    if major_available is not None:
        matrix[:, subject.SOURCE_MAJOR_AVAILABLE_CHANNEL] = np.asarray(
            major_available,
            dtype=np.float32,
        )
    sample = {
        "sourceFile": source_file,
        "sourceSha256": source_file,
        "groupId": group_id,
        "deviceName": device_name,
        "matrix": matrix,
    }
    if declared_major_max is not None:
        sample["touchAxisRangeAudit"] = {
            "ranges": {
                "touchMajor": {
                    "declaredToViewDiagonalMedian": declared_major_max,
                }
            }
        }
    return sample


def physical_values(samples):
    values = np.stack([
        sample["matrix"][:, subject.lightweight.PHYSICAL_INDICES]
        for sample in samples
    ])
    return values.astype(np.float32)


class ContactSizeExperimentTests(unittest.TestCase):
    def test_source_and_physical_size_channels_are_name_derived(self):
        self.assertEqual(12, subject.SOURCE_SIZE_CHANNEL)
        self.assertEqual(8, subject.PHYSICAL_SIZE_CHANNEL)
        self.assertEqual(21, subject.SOURCE_SIZE_AVAILABLE_CHANNEL)
        self.assertEqual(22, subject.SOURCE_MAJOR_AVAILABLE_CHANNEL)
        self.assertNotEqual(
            subject.SOURCE_SIZE_CHANNEL,
            subject.PHYSICAL_SIZE_CHANNEL,
        )

    def test_fold_metadata_marks_unseen_device_and_scope(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rows = []
            for device in ("device-a", "device-b"):
                for index in range(4):
                    sample = make_sample(
                        "{}_{}".format(device, index),
                        [1.0] * 9,
                        group_id=device + ":session",
                    )
                    sample["deviceName"] = device
                    sample["classLabel"] = subject.lightweight.CLASS_LABELS[
                        index
                    ]
                    sample["label"] = index
                    rows.append(sample)
            folds = {
                0: (rows[4:], rows[:4]),
                1: (rows[:4], rows[4:]),
            }
            for train, valid in folds.values():
                for sample in train + valid:
                    sample["datasetId"] = str(root)
            (root / "training_cleaning_report.json").write_text(
                json.dumps({
                    "evaluationScope": "cross_device",
                    "validationFoldKind": "held_out_device",
                }),
                encoding="utf-8",
            )

            self.assertEqual(
                "cross_device",
                subject.fold_evaluation_scope(folds),
            )
            metadata = subject.fold_metadata(folds)
            self.assertEqual(
                "held_out_device",
                metadata["0"]["kind"],
            )
            self.assertEqual(["device-a"], metadata["0"]["validDevices"])

    def test_within_touch_median_preserves_relative_change(self):
        sample = make_sample(
            "a",
            [2.0, 4.0, 0.0, 6.0] + [0.0] * 5,
        )
        result = subject.apply_size_mode(
            physical_values([sample]),
            [sample],
            "within_touch_median",
        )
        np.testing.assert_allclose(
            result[0, :4, subject.PHYSICAL_SIZE_CHANNEL],
            [0.5, 1.0, 0.0, 1.5],
        )

    def test_missing_size_stays_zero(self):
        sample = make_sample(
            "a",
            [0.0] * 9,
            available=[False] * 9,
        )
        result = subject.apply_size_mode(
            physical_values([sample]),
            [sample],
            "within_touch_median",
        )
        np.testing.assert_array_equal(
            result[0, :, subject.PHYSICAL_SIZE_CHANNEL],
            np.zeros(9, dtype=np.float32),
        )

    def test_causal_reference_does_not_read_later_touches(self):
        early = make_sample("a", [2.0] * 9, group_id="s")
        middle = make_sample("b", [4.0] * 9, group_id="s")
        late = make_sample("c", [100.0] * 9, group_id="s")
        samples = [late, early, middle]
        result = subject.apply_size_mode(
            physical_values(samples),
            samples,
            "causal_median_8",
        )
        np.testing.assert_allclose(
            result[0, :, subject.PHYSICAL_SIZE_CHANNEL],
            np.ones(9) * 25.0,
        )
        np.testing.assert_allclose(
            result[1, :, subject.PHYSICAL_SIZE_CHANNEL],
            np.ones(9) * 0.5,
        )
        np.testing.assert_allclose(
            result[2, :, subject.PHYSICAL_SIZE_CHANNEL],
            np.ones(9),
        )

    def test_shape_only_keeps_axis_ratio_channel(self):
        sample = make_sample("a", [2.0] * 9)
        ratio_channel = subject.lightweight.FEATURE_NAMES.index(
            "log_axis_ratio"
        )
        sample["matrix"][:, ratio_channel] = 0.75
        result = subject.apply_size_mode(
            physical_values([sample]),
            [sample],
            "shape_only",
        )
        np.testing.assert_array_equal(
            result[0, :, subject.PHYSICAL_SIZE_CHANNEL],
            np.zeros(9, dtype=np.float32),
        )
        physical_ratio = subject.lightweight.PHYSICAL_NAMES.index(
            "log_axis_ratio"
        )
        self.assertAlmostEqual(
            float(result[0, 0, physical_ratio]),
            np.float32(0.75),
        )

    def test_major_ratio_requires_both_availability_flags(self):
        sample = make_sample(
            "a",
            [6.0] * 9,
            available=[True] * 9,
            major=[2.0] * 9,
            major_available=[True] * 4 + [False] * 5,
        )
        result = subject.apply_size_mode(
            physical_values([sample]),
            [sample],
            "major_ratio",
        )
        np.testing.assert_allclose(
            result[0, :, subject.PHYSICAL_SIZE_CHANNEL],
            [3.0] * 4 + [0.0] * 5,
        )

    def test_major_ratio_handles_zero_major_without_warning(self):
        sample = make_sample(
            "a",
            [6.0] * 9,
            available=[True] * 9,
            major=[0.0] * 9,
            major_available=[True] * 9,
        )
        with np.errstate(divide="raise", invalid="raise"):
            result = subject.apply_size_mode(
                physical_values([sample]),
                [sample],
                "major_ratio",
            )
        np.testing.assert_array_equal(
            result[0, :, subject.PHYSICAL_SIZE_CHANNEL],
            np.zeros(9, dtype=np.float32),
        )

    def test_identity_probe_uses_held_out_sessions(self):
        samples = []
        for session_index in range(8):
            for touch_index in range(4):
                sample = make_sample(
                    "{}_{}".format(session_index, touch_index),
                    [float(session_index + 1)] * 9,
                    available=[True] * 9,
                    group_id="session_{}".format(session_index),
                )
                samples.append(sample)
        report = subject.grouped_session_identity_probe(
            samples,
            size_only=True,
            seed=17,
        )
        self.assertGreater(report["heldOutSessionCount"], 0)
        self.assertGreater(report["trainCount"], 0)
        self.assertGreater(report["testCount"], 0)
        self.assertLess(
            report["trainCount"] + report["testCount"],
            len(samples) + 1,
        )
        self.assertGreaterEqual(report["accuracy"], 0.0)
        self.assertLessEqual(report["accuracy"], 1.0)

    def test_declared_major_range_normalizes_each_device_independently(self):
        train = make_sample(
            "train",
            [0.0] * 9,
            major=[0.25, 0.5, 0.75] + [0.0] * 6,
            major_available=[True] * 3 + [False] * 6,
            declared_major_max=0.5,
        )
        valid = make_sample(
            "valid",
            [0.0] * 9,
            major=[0.25, 0.5] + [0.0] * 7,
            major_available=[True] * 2 + [False] * 7,
            declared_major_max=0.25,
        )
        raw = np.stack([
            np.asarray(train["matrix"][:, subject.lightweight.PHYSICAL_INDICES]),
            np.asarray(valid["matrix"][:, subject.lightweight.PHYSICAL_INDICES]),
        ]).astype(np.float32)
        result = subject.apply_size_mode(
            raw,
            [train, valid],
            "declared_major_range",
        )

        np.testing.assert_allclose(
            result[0, :3, subject.PHYSICAL_SIZE_CHANNEL],
            [0.5, 1.0, 1.5],
        )
        np.testing.assert_allclose(
            result[1, :2, subject.PHYSICAL_SIZE_CHANNEL],
            [1.0, 2.0],
        )

    def test_declared_major_range_missing_metadata_is_zeroed(self):
        sample = make_sample("missing", [2.0] * 9)
        result = subject.apply_declared_major_range_mode(
            physical_values([sample]),
            [sample],
        )
        np.testing.assert_array_equal(
            result[0, :, subject.PHYSICAL_SIZE_CHANNEL],
            np.zeros(9, dtype=np.float32),
        )

    def test_declared_major_range_without_dataset_metadata_falls_back(self):
        train = make_sample("train", [2.0] * 9)
        valid = make_sample("valid", [4.0] * 9)
        raw = np.stack([
            np.asarray(train["matrix"][:, subject.lightweight.PHYSICAL_INDICES]),
            np.asarray(valid["matrix"][:, subject.lightweight.PHYSICAL_INDICES]),
        ]).astype(np.float32)
        result = subject.apply_size_mode(
            raw,
            [train, valid],
            "declared_major_range",
        )
        np.testing.assert_array_equal(raw, result)

    def test_major_within_touch_median_removes_device_scale(self):
        sample = make_sample(
            "a",
            [0.0] * 9,
            major=[2.0, 4.0, 6.0] + [0.0] * 6,
            major_available=[True] * 3 + [False] * 6,
        )
        result = subject.apply_size_mode(
            physical_values([sample]),
            [sample],
            "major_within_touch_median",
        )
        np.testing.assert_allclose(
            result[0, :3, subject.PHYSICAL_SIZE_CHANNEL],
            [0.5, 1.0, 1.5],
        )

    def test_gravity_alignment_rotates_device_vectors_into_one_frame(self):
        sample = make_sample("a", [0.0] * 9)
        sample["matrix"][:, 2:5] = np.asarray(
            [1.0, 2.0, 3.0],
            dtype=np.float32,
        )
        sample["matrix"][:, 5:8] = np.asarray(
            [4.0, 5.0, 6.0],
            dtype=np.float32,
        )
        sample["matrix"][:, 8:11] = np.asarray(
            [0.0, 9.80665, 0.0],
            dtype=np.float32,
        )

        aligned = subject.apply_pose_mode(
            physical_values([sample]),
            [sample],
            "gravity_aligned_axes",
        )

        np.testing.assert_allclose(
            aligned[0, :, 2:5],
            np.tile(
                np.asarray([1.0, -3.0, 2.0], dtype=np.float32),
                (9, 1),
            ),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            aligned[0, :, 5:8],
            np.tile(
                np.asarray([4.0, -6.0, 5.0], dtype=np.float32),
                (9, 1),
            ),
            atol=1e-5,
        )

    def test_device_median_removes_device_scale_without_labels(self):
        device_a = make_sample(
            "a",
            [2.0] * 9,
            group_id="a-session",
            device_name="device-a",
        )
        device_b = make_sample(
            "b",
            [6.0] * 9,
            group_id="b-session",
            device_name="device-b",
        )
        raw = physical_values([device_a, device_b])

        result = subject.apply_size_mode(
            raw,
            [device_a, device_b],
            "device_median_oracle",
        )

        np.testing.assert_allclose(
            result[:, :, subject.PHYSICAL_SIZE_CHANNEL],
            np.ones((2, 9), dtype=np.float32),
        )

    def test_causal_device_baseline_uses_only_earlier_samples(self):
        samples = [
            make_sample(
                "a",
                [2.0] * 9,
                group_id="session-a",
                device_name="device-a",
            ),
            make_sample(
                "b",
                [4.0] * 9,
                group_id="session-b",
                device_name="device-a",
            ),
            make_sample(
                "c",
                [100.0] * 9,
                group_id="session-c",
                device_name="device-a",
            ),
        ]
        result = subject.apply_size_mode(
            physical_values(samples),
            samples,
            "causal_device_median_2",
        )
        np.testing.assert_allclose(
            result[0, :, subject.PHYSICAL_SIZE_CHANNEL],
            np.ones(9) * 2.0 / 3.0,
        )
        np.testing.assert_allclose(
            result[1, :, subject.PHYSICAL_SIZE_CHANNEL],
            np.ones(9) * 4.0 / 3.0,
        )
        np.testing.assert_allclose(
            result[2, :, subject.PHYSICAL_SIZE_CHANNEL],
            np.ones(9) * 100.0 / 3.0,
        )

    def test_cross_device_causal_calibration_leaves_unseen_device_size_zero(self):
        train = make_sample(
            "a",
            [2.0] * 9,
            group_id="train-session",
            device_name="device-a",
        )
        valid = make_sample(
            "b",
            [6.0] * 9,
            group_id="valid-session",
            device_name="device-b",
        )

        _, valid_values = subject.prepare_fold_leave_one_device_out(
            [train],
            [valid],
            "causal_device_median_8",
            pose_mode="raw",
        )

        self.assertEqual((1, 60), valid_values.shape)
        size_column = (
            subject.lightweight.PHYSICAL_NAMES.index("size")
            * 5
        )
        valid_size = valid_values[
            :,
            size_column:size_column + 5,
        ]
        raw_valid = physical_values([valid])
        raw_size = raw_valid[:, :, subject.PHYSICAL_SIZE_CHANNEL]
        self.assertGreater(float(np.max(raw_size)), 0.0)
        self.assertEqual(0.0, float(np.max(valid_size)))
        self.assertAlmostEqual(
            0.0,
            float(np.max(valid_size)),
        )

    def test_device_balanced_weights_equalize_device_and_class_mass(self):
        samples = []
        for device, count in (("device-a", 12), ("device-b", 4)):
            for index in range(count):
                sample = make_sample(
                    device + str(index),
                    [1.0] * 9,
                    group_id=device,
                    device_name=device,
                )
                sample["label"] = index % 4
                samples.append(sample)

        weights = subject.device_balanced_weights(samples)

        for device in ("device-a", "device-b"):
            self.assertAlmostEqual(
                8.0,
                float(np.sum([
                    weight
                    for sample, weight in zip(samples, weights)
                    if sample["deviceName"] == device
                ])),
                places=5,
            )

    def test_tail_calibration_excludes_only_unlabeled_warmup(self):
        train = []
        for index in range(2):
            sample = make_sample(
                "train-" + str(index),
                [2.0] * 9,
                group_id="train-" + str(index),
                device_name="device-a",
            )
            sample["classLabel"] = subject.lightweight.CLASS_LABELS[
                index
            ]
            sample["label"] = index
            train.append(sample)
        valid = []
        for index in range(5):
            sample = make_sample(
                str(100 + index) + "-valid",
                [float(index + 1)] * 9,
                group_id="valid-" + str(index),
                device_name="device-b",
            )
            sample["classLabel"] = subject.lightweight.CLASS_LABELS[
                index % 4
            ]
            sample["label"] = index % 4
            valid.append(sample)

        _, values, score_indices = (
            subject.prepare_fold_leave_one_device_out_calibrated_tail(
                train,
                valid,
                "causal_device_median_2_tail",
                pose_mode="raw",
            )
        )

        self.assertEqual((5, 60), values.shape)
        self.assertEqual([2, 3, 4], score_indices)

    def test_tail_calibration_reuses_positive_warmup_for_negatives(self):
        train = [
            make_sample(
                "train-" + str(index),
                [2.0] * 9,
                group_id="train-session",
                device_name="device-a",
            )
            for index in range(2)
        ]
        valid = [
            make_sample(
                str(100 + index) + "-valid",
                [float(index + 1)] * 9,
                group_id="valid-session",
                device_name="device-b",
            )
            for index in range(5)
        ]
        ordinary = [
            make_sample(
                str(index).zfill(3) + "-ordinary",
                [float(index + 1)] * 9,
                group_id="ordinary-session",
                device_name="device-b",
            )
            for index in range(5)
        ]

        retained, values = subject.prepare_ordinary_values(
            train,
            valid,
            ordinary,
            "causal_device_median_2_tail",
            subject.prepare_fold_leave_one_device_out,
            "raw",
            True,
        )

        self.assertEqual(
            [2, 3, 4],
            [int(sample["sourceFile"][:3]) for sample in retained],
        )
        self.assertEqual((3, 60), values.shape)

    def test_ordinary_samples_match_held_out_physical_device(self):
        valid = [make_sample("valid", [1.0] * 9)]
        valid[0].update({
            "platform": "android",
            "collectorDeviceId": "collector-a",
            "deviceName": "shared-name",
        })
        ordinary = [
            make_sample("a", [1.0] * 9),
            make_sample("b", [1.0] * 9),
        ]
        ordinary[0].update({
            "platform": "android",
            "collectorDeviceId": "collector-a",
            "deviceName": "shared-name",
        })
        ordinary[1].update({
            "platform": "android",
            "collectorDeviceId": "collector-b",
            "deviceName": "other-device",
        })

        selected = subject.ordinary_samples_for_fold(
            ordinary,
            valid,
            "held_out_device",
        )

        self.assertEqual(
            ["a"],
            [sample["sourceFile"] for sample in selected],
        )

    def test_ordinary_negative_audit_reports_unmatched_and_fold_coverage(self):
        valid_a = make_sample(
            "valid-a",
            [1.0] * 9,
            group_id="session-a",
            device_name="device-a",
        )
        valid_a["platform"] = "android"
        valid_b = make_sample(
            "valid-b",
            [1.0] * 9,
            group_id="session-b",
            device_name="device-b",
        )
        valid_b["platform"] = "android"
        ordinary = []
        for index, device in enumerate(("device-a", "device-b", "device-c")):
            sample = make_sample(
                "ordinary-{}".format(index),
                [1.0] * 9,
                group_id="ordinary-{}".format(index),
                device_name=device,
            )
            sample["platform"] = "android"
            ordinary.append(sample)

        audit = subject.ordinary_negative_audit(
            ordinary,
            {0: ([], [valid_a]), 1: ([], [valid_b])},
            "held_out_device",
        )

        self.assertEqual(3, audit["totalCount"])
        self.assertEqual(2, audit["matchedCount"])
        self.assertEqual(1, audit["unmatchedCount"])
        self.assertEqual(2, audit["matchedFoldCount"])
        self.assertEqual(1.0, audit["foldCoverage"])
        self.assertEqual(
            1,
            audit["perFold"]["0"]["matchedCount"],
        )

    def test_ordinary_negative_mode_audit_counts_calibrated_negatives(self):
        audit = subject.ordinary_negative_mode_audit(
            {
                "totalCount": 8,
                "positiveFoldCount": 2,
            },
            [
                {
                    "ordinaryDomainSampleCount": 4,
                    "ordinarySampleCount": 3,
                },
                {
                    "ordinaryDomainSampleCount": 4,
                    "ordinarySampleCount": 4,
                },
            ],
        )

        self.assertEqual(8, audit["matchedCount"])
        self.assertEqual(7, audit["scoredCount"])
        self.assertEqual(2, audit["scoredFoldCount"])
        self.assertEqual(1.0, audit["scoredFoldCoverage"])

    def test_loaded_positive_and_ordinary_samples_share_device_identity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            matrix = np.zeros((9, 26), dtype=np.float32).tolist()
            rows = []
            for class_label in subject.lightweight.CLASS_LABELS:
                rows.append({
                    "sourceSha256": "positive-" + class_label,
                    "sampleId": "positive-" + class_label,
                    "sourceFile": "positive-" + class_label,
                    "platform": "android",
                    "collectorDeviceId": "collector-a",
                    "deviceName": "OnePlus PHK110",
                    "classLabel": class_label,
                    "groupId": "session-a",
                    "matrix": matrix,
                })
            with (root / "training_touch26.jsonl").open(
                "w",
                encoding="utf-8",
            ) as output:
                for row in rows:
                    output.write(json.dumps(row) + "\n")
            with (root / "ordinary_touch26.jsonl").open(
                "w",
                encoding="utf-8",
            ) as output:
                output.write(json.dumps({
                    "sourceSha256": "ordinary",
                    "sampleId": "ordinary",
                    "sourceFile": "ordinary",
                    "platform": "android",
                    "collectorDeviceId": "collector-a",
                    "deviceName": "OnePlus PHK110",
                    "classLabel": subject.ORDINARY_CLASS_LABEL,
                    "groupId": "session-a",
                    "matrix": matrix,
                }) + "\n")
            with (root / "strict_group_folds.csv").open(
                "w",
                encoding="utf-8",
                newline="",
            ) as output:
                writer = csv.DictWriter(
                    output,
                    fieldnames=("fold", "sourceSha256", "split"),
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow({
                        "fold": "0",
                        "sourceSha256": row["sourceSha256"],
                        "split": "valid",
                    })

            folds = subject.load_samples(root)
            ordinary = subject.load_ordinary_samples(root)
            train, valid = folds[0]

            self.assertEqual([], train)
            self.assertEqual(
                {
                    "android:device:OnePlus PHK110",
                },
                {
                    subject.physical_sample_device_key(sample)
                    for sample in valid
                },
            )
            self.assertEqual(
                ["ordinary"],
                [
                    sample["sourceFile"]
                    for sample in subject.ordinary_samples_for_fold(
                        ordinary,
                        valid,
                        "held_out_device",
                    )
                ],
            )

    def test_selective_threshold_uses_training_coverage(self):
        train_truths = [0, 1, 2, 3]
        train_probabilities = np.asarray([
            [0.95, 0.02, 0.02, 0.01],
            [0.02, 0.94, 0.02, 0.02],
            [0.02, 0.02, 0.92, 0.04],
            [0.02, 0.02, 0.04, 0.92],
        ])
        valid_truths = [0, 1, 2, 3]
        valid_probabilities = np.asarray([
            [0.90, 0.05, 0.03, 0.02],
            [0.10, 0.80, 0.05, 0.05],
            [0.10, 0.20, 0.60, 0.10],
            [0.50, 0.20, 0.20, 0.10],
        ])

        result = subject.selective_metrics(
            train_truths,
            train_probabilities,
            valid_truths,
            valid_probabilities,
            target_train_coverage=0.75,
        )

        self.assertIsNotNone(result["threshold"])
        self.assertGreaterEqual(result["validationCoverage"], 0.25)
        self.assertGreaterEqual(result["acceptedAccuracy"], 0.5)
        self.assertEqual(19, len(result["confidenceCurve"]))

    def test_expected_calibration_error_detects_overconfidence(self):
        probabilities = np.asarray([
            [0.99, 0.01],
            [0.01, 0.99],
        ])
        result = subject.expected_calibration_error(
            [0, 0],
            probabilities,
        )
        self.assertGreater(result["expectedCalibrationError"], 0.4)
        self.assertGreater(result["brierScore"], 0.9)

    def test_head_selective_reports_per_class_coverage(self):
        result = subject.selective_head_metrics(
            [0, 1, 0, 1],
            np.asarray([
                [0.95, 0.05],
                [0.05, 0.95],
                [0.90, 0.10],
                [0.10, 0.90],
            ]),
            [0, 1, 0, 1],
            np.asarray([
                [0.90, 0.10],
                [0.20, 0.80],
                [0.40, 0.60],
                [0.30, 0.70],
            ]),
            ("thumb", "index"),
            target_train_coverage=0.75,
        )

        self.assertIsNotNone(result["threshold"])
        self.assertEqual(
            {"thumb": 0.5, "index": 0.0},
            result["coverageByClass"],
        )
        self.assertEqual(1.0, result["acceptedMetrics"]["accuracy"])

    def test_layered_decision_emits_only_reliable_head(self):
        result = subject.layered_head_decision_metrics(
            [0, 1, 2, 3],
            np.asarray([
                [0.95, 0.05],
                [0.95, 0.05],
                [0.05, 0.95],
                [0.50, 0.50],
            ]),
            np.asarray([
                [0.95, 0.05],
                [0.05, 0.95],
                [0.50, 0.50],
                [0.50, 0.50],
            ]),
            finger_threshold=0.90,
            strength_threshold=0.90,
        )

        self.assertEqual(
            {
                "both": 2,
                "fingerOnly": 1,
                "strengthOnly": 0,
                "abstain": 1,
                "total": 4,
            },
            result["decisionCounts"],
        )
        self.assertEqual(
            1.0,
            result["bothHeadMetrics"]["accuracy"],
        )
        self.assertEqual(
            1.0,
            result["fingerOnlyMetrics"]["accuracy"],
        )
        self.assertEqual(0.25, result["abstainCoverage"])

    def test_ordinary_touch_false_acceptance_uses_same_confidence_gate(self):
        result = subject.false_acceptance_metrics(
            np.asarray([
                [0.80, 0.10, 0.05, 0.05],
                [0.30, 0.25, 0.25, 0.20],
                [0.45, 0.20, 0.20, 0.15],
            ]),
            0.60,
        )

        self.assertEqual(3, result["ordinaryCount"])
        self.assertEqual(1, result["falseAcceptanceCount"])
        self.assertAlmostEqual(1.0 / 3.0, result["falseAcceptanceRate"])


if __name__ == "__main__":
    unittest.main()
