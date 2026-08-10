import io
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

from android.analysis.extract_unique_touch_data import run_extraction


def sample_bytes(sample_id, created_at, collector="tester", device="Device A", value=1):
    payload = {
        "schemaVersion": 3,
        "id": sample_id,
        "createdAt": created_at,
        "collector": {"userName": collector},
        "deviceCapability": {"deviceName": device},
        "value": value,
    }
    return json.dumps(payload, sort_keys=True).encode("utf-8")


class UniqueTouchDataExtractionTests(unittest.TestCase):
    def test_extracts_unique_samples_and_audits_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory) / "data"
            legacy_root = data_root / "lemon" / "lemon"
            collection_root = data_root / "collections" / "batch" / "raw"
            destination = data_root / "deduplicated_raw"
            legacy_root.mkdir(parents=True)
            collection_root.mkdir(parents=True)

            first_bytes = sample_bytes("sample-a", 1_784_779_169.624)
            second_bytes = sample_bytes(
                "sample-b",
                1_784_779_170.624,
                collector="",
                device="Device B",
            )
            (legacy_root / "sample-a.json").write_bytes(first_bytes)
            (collection_root / "sample-b.json").write_bytes(second_bytes)
            (collection_root / "missing-id.json").write_text(
                json.dumps({"createdAt": 1_784_779_171.624}),
                encoding="utf-8",
            )

            archive_path = data_root / "samples.tar"
            with tarfile.open(archive_path, "w") as archive:
                duplicate_info = tarfile.TarInfo("raw/sample-a-copy.json")
                duplicate_info.size = len(first_bytes)
                archive.addfile(duplicate_info, io.BytesIO(first_bytes))

                conflict_bytes = sample_bytes(
                    "sample-b",
                    1_784_779_170.624,
                    collector="tester",
                    device="Device B",
                    value=2,
                )
                conflict_info = tarfile.TarInfo("raw/sample-b-conflict.json")
                conflict_info.size = len(conflict_bytes)
                archive.addfile(conflict_info, io.BytesIO(conflict_bytes))

            summary = run_extraction(data_root, destination)

            self.assertEqual(2, summary["copied"])
            self.assertEqual(1, summary["duplicateIdAndHash"])
            self.assertEqual(1, summary["idConflicts"])
            self.assertEqual(1, summary["invalidSamples"])
            self.assertEqual(2, len(list(destination.rglob("*.json"))) - 1)

            manifest_rows = (destination / "manifest.csv").read_text(
                encoding="utf-8"
            )
            duplicate_rows = (destination / "duplicates.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("sample-a", manifest_rows)
            self.assertIn("sample-b", manifest_rows)
            self.assertIn("duplicate_id_and_hash", duplicate_rows)
            self.assertIn("id_conflict", duplicate_rows)
            self.assertTrue(
                list(destination.glob("tester/device-a/2026-07-23/sample-a.json"))
            )
            self.assertTrue(
                list(destination.glob("batch/device-b/2026-07-23/sample-b.json"))
            )

    def test_rerun_indexes_existing_destination_samples(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory) / "data"
            source_root = data_root / "tao" / "tao"
            destination = data_root / "deduplicated_raw"
            source_root.mkdir(parents=True)
            (source_root / "sample.json").write_bytes(
                sample_bytes("sample-rerun", 1_784_779_169.624)
            )

            first_summary = run_extraction(data_root, destination)
            second_summary = run_extraction(data_root, destination)

            self.assertEqual(1, first_summary["copied"])
            self.assertEqual(0, second_summary["copied"])
            self.assertEqual(1, second_summary["existing"])
            self.assertEqual(1, second_summary["duplicateIdAndHash"])
            self.assertEqual(1, len(list(destination.rglob("sample.json"))))

    def test_rerun_repairs_identical_destination_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_root = Path(temporary_directory) / "data"
            source_root = data_root / "tao" / "tao"
            destination = data_root / "deduplicated_raw"
            source_root.mkdir(parents=True)
            (source_root / "sample.json").write_bytes(
                sample_bytes("sample-repair", 1_784_779_169.624)
            )

            run_extraction(data_root, destination)
            retained = next(destination.rglob("sample.json"))
            duplicate = retained.with_name("sample-copy.json")
            shutil.copyfile(retained, duplicate)

            summary = run_extraction(data_root, destination)

            self.assertEqual(1, summary["repairedDestinationDuplicates"])
            self.assertEqual(1, len(list(destination.rglob("sample*.json"))))


if __name__ == "__main__":
    unittest.main()
