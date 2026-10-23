import csv
import json
import tempfile
import unittest
import sys
from pathlib import Path
from uuid import uuid4

from android.analysis import collection_annotations as annotations
from android.analysis import build_causal_time_grid_dataset as builder
from android.analysis.tests.test_build_causal_time_grid_dataset import valid_sample

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_android_finger_four_class_cnn as trainer


def write_review(sample_path, sample_id, statuses):
    # 两端都写同一协议；iOS 的 UUID 可为大写，不能错误地强制小写文件名。
    directory = sample_path.parent / "annotations"
    directory.mkdir(exist_ok=True)
    review = {
        "schemaVersion": 1, "contract": "touch_sample_review_v1",
        "sampleId": sample_id, "sampleFileName": sample_path.name,
        "labelStatus": statuses[-1],
        "history": [{"revision": index + 1, "status": status,
                     "timestampMs": 1000 + index, "source": "user_review"}
                    for index, status in enumerate(statuses)],
    }
    destination = directory / (sample_id + ".json")
    destination.write_text(json.dumps(review), encoding="utf-8")
    return destination


class CollectionAnnotationTests(unittest.TestCase):
    def test_legacy_record_keeps_unknown_context(self):
        result = annotations.read_annotation(Path("old.json"), {"id": "old-id"})
        self.assertEqual("legacy_unreviewed", result["labelStatus"])
        self.assertEqual("unknown", result["taskId"])
        self.assertEqual("", annotations.exclusion_reason(result))

    def test_review_does_not_change_original_or_strength(self):
        with tempfile.TemporaryDirectory() as folder:
            sample_id = str(uuid4()).upper()
            path = Path(folder) / "sample.json"
            sample = valid_sample(sample_id)
            raw = json.dumps(sample)
            path.write_text(raw, encoding="utf-8")
            for status in ("mistake", "uncertain"):
                write_review(path, sample_id, [status])
                with self.assertRaisesRegex(RuntimeError, "user_review_" + status):
                    annotations.require_trainable(path, sample)
            write_review(path, sample_id, ["mistake", "confirmed"])
            self.assertEqual("confirmed", annotations.require_trainable(path, sample)["labelStatus"])
            self.assertEqual(raw, path.read_text(encoding="utf-8"))
            self.assertEqual("light", sample["label"]["touchStrength"])

    def test_interleaved_context_v2_is_auditable(self):
        context = {
            "schemaVersion": 2,
            "contract": "touch_collection_context_v2",
            "collectionProtocol": "interleaved_v2",
            "scheduleId": "schedule-v2",
            "cycleIndex": 3,
            "stepIndex": 2,
            "scheduleOrder": [
                "thumb_light", "index_heavy", "index_light", "thumb_heavy"
            ],
            "strengthTruthSource": "user_task",
            "taskId": "handheld",
            "bodyPosture": "standing",
            "requestedDevicePose": "vertical_top_up",
            "grip": "same_hand",
            "support": "handheld",
            "labelStatus": "task_assigned",
        }
        result = annotations.read_annotation(
            Path("sample.json"), {"id": "sample-id", "collectionContext": context})
        self.assertEqual("interleaved_v2", result["collectionProtocol"])
        self.assertEqual("schedule-v2", result["scheduleId"])
        self.assertEqual("2", result["stepIndex"])
        self.assertEqual("user_task", result["strengthTruthSource"])

    def test_interleaved_context_rejects_unprovable_schedule(self):
        valid = {
            "schemaVersion": 2,
            "contract": "touch_collection_context_v2",
            "collectionProtocol": "interleaved_v2",
            "scheduleId": "schedule-v2",
            "cycleIndex": 0,
            "stepIndex": 0,
            "scheduleOrder": [
                "thumb_light", "thumb_heavy", "index_light", "index_heavy"
            ],
            "strengthTruthSource": "user_task",
            "labelStatus": "task_assigned",
        }
        variants = [
            ("scheduleId", ""),
            ("strengthTruthSource", "inferred_from_touch"),
            ("scheduleOrder", [
                "thumb_light", "thumb_light", "index_light", "index_heavy"
            ]),
            ("stepIndex", 4),
            ("cycleIndex", -1),
        ]
        for field, value in variants:
            context = dict(valid)
            context[field] = value
            with self.assertRaisesRegex(RuntimeError, "invalid_interleaved_context"):
                annotations.read_annotation(
                    Path("sample.json"),
                    {"id": "sample-id", "collectionContext": context})

    def test_legacy_context_v1_remains_readable(self):
        context = {
            "schemaVersion": 1,
            "contract": "touch_collection_context_v1",
            "collectionProtocol": "block_v1",
            "taskId": "inverted",
            "strengthTruthSource": "user_task",
            "labelStatus": "task_assigned",
        }
        result = annotations.read_annotation(
            Path("sample.json"), {"id": "sample-id", "collectionContext": context})
        self.assertEqual("block_v1", result["collectionProtocol"])
        self.assertEqual("unknown", result["scheduleId"])

    def test_corrupt_or_mismatched_sidecar_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            sample_id = str(uuid4())
            path = Path(folder) / "sample.json"
            for field, value in (("sampleFileName", "other.json"), ("sampleId", str(uuid4())),
                                 ("labelStatus", "uncertain"), ("history", []), ("schemaVersion", 9)):
                destination = write_review(path, sample_id, ["confirmed"])
                review = json.loads(destination.read_text())
                review[field] = value
                destination.write_text(json.dumps(review))
                with self.assertRaisesRegex(RuntimeError, "invalid_sample_review"):
                    annotations.read_annotation(path, {"id": sample_id})

    def test_path_traversal_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "invalid_sample_identity"):
            annotations.read_annotation("a.json", {"id": "../other"})

    def test_interrupted_atomic_review_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.json"
            sample_id = str(uuid4())
            review = write_review(path, sample_id, ["mistake"])
            review.rename(Path(str(review) + ".bak"))
            with self.assertRaisesRegex(RuntimeError, "sample_review_transaction_incomplete"):
                annotations.read_annotation(path, {"id": sample_id})

    def test_export_excludes_review_files_and_bad_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "raw"
            source.mkdir()
            ids = {}
            for name, status in (("bad", "mistake"), ("unsure", "uncertain"), ("good", "confirmed")):
                sample_id = str(uuid4())
                ids[name] = sample_id
                path = source / (name + ".json")
                record = valid_sample(sample_id)
                record["collectionContext"] = {
                    "schemaVersion": 1, "contract": "touch_collection_context_v1",
                    "taskId": "inverted", "labelStatus": "task_assigned",
                }
                path.write_text(json.dumps(record), encoding="utf-8")
                write_review(path, sample_id, [status])
            summary = builder.build_dataset(source, root / "derived")
            self.assertEqual(3, summary["sourceCount"])
            self.assertEqual(1, summary["acceptedCount"])
            output = json.loads((root / "derived/canonical_imu11.jsonl").read_text())
            self.assertEqual(ids["good"], output["sampleId"])
            self.assertEqual("inverted", output["taskId"])
            self.assertEqual(64, len(output["reviewSha256"]))
            self.assertNotIn("taskId", output["featureNames"])

    def test_conflicting_duplicate_cannot_bypass_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "raw"
            (source / "copy").mkdir(parents=True)
            sample_id = str(uuid4())
            payload = json.dumps(valid_sample(sample_id))
            first = source / "sample.json"
            first.write_text(payload, encoding="utf-8")
            (source / "copy/sample.json").write_text(payload, encoding="utf-8")
            write_review(first, sample_id, ["mistake"])
            result = builder.build_dataset(source, root / "derived")
            self.assertEqual(0, result["acceptedCount"])
            self.assertEqual(2, result["rejectedCount"])
            with (root / "derived/rejected_samples.csv").open(encoding="utf-8") as file:
                self.assertEqual({"duplicate_source_review_conflict"},
                                 {row["rejectionReason"] for row in csv.DictReader(file)})

    def test_raw_training_loader_excludes_reviewed_mistake(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sample_id = str(uuid4())
            path = root / "sample.json"
            path.write_text(json.dumps(valid_sample(sample_id)), encoding="utf-8")
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["path", "id", "class_label"])
                writer.writeheader()
                writer.writerow({"path": str(path), "id": sample_id, "class_label": "thumb_light"})
            write_review(path, sample_id, ["mistake"])
            self.assertEqual([], trainer.load_samples(manifest, ["delta"], [0], "error"))

    def test_restoring_excluded_sample_invalidates_empty_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sample_id = str(uuid4())
            path = root / "sample.json"
            path.write_text(json.dumps(valid_sample(sample_id)), encoding="utf-8")
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["path", "id", "class_label"])
                writer.writeheader()
                writer.writerow({"path": str(path), "id": sample_id, "class_label": "thumb_light"})
            write_review(path, sample_id, ["mistake"])
            cache = root / "cache.pkl"
            self.assertEqual([], trainer.load_or_cache_samples(manifest, ["delta"], [0], "error", cache))
            write_review(path, sample_id, ["mistake", "confirmed"])
            with self.assertRaisesRegex(RuntimeError, "sample_cache_signature_mismatch"):
                trainer.load_or_cache_samples(manifest, ["delta"], [0], "error", cache)


if __name__ == "__main__":
    unittest.main()
