import csv
import json
import tempfile
import unittest
from pathlib import Path

from android.analysis.build_clean_v3_folds import assign_sessions
from android.analysis.build_clean_v3_folds import build_folds
from android.analysis.build_clean_v3_folds import build_collection_folds
from android.analysis.build_clean_v3_folds import build_domain_folds
from android.analysis.build_clean_v3_folds import build_temporal_tail_fold
from android.analysis.build_clean_v3_folds import thin_temporal
from android.analysis.build_clean_v3_folds import validate_no_session_leakage


CLASS_ORDER = ("thumb_light", "thumb_heavy", "index_light", "index_heavy")


def row(
    sample_id,
    created_at,
    domain="tao__oneplus-le2110",
    collection="tao__oneplus-le2110__2026-07-22",
    class_label="thumb_light",
    raw_session_id="",
):
    return {
        "id": sample_id,
        "created_at": str(created_at),
        "domain_group": domain,
        "collection_name": collection,
        "class_label": class_label,
        "motion_state": "handheld_static",
        "raw_session_id": raw_session_id,
        "quality_bucket": "high_quality",
        "sample_weight": "1.0",
    }


class CleanV3FoldTests(unittest.TestCase):
    def test_assigns_raw_sessions_and_infers_legacy_gaps(self):
        rows = [
            row("raw-a", 10, raw_session_id="session-a"),
            row("raw-b", 500, raw_session_id="session-a"),
            row("legacy-a", 0),
            row("legacy-b", 30),
            row("legacy-c", 91),
        ]

        assigned = assign_sessions(rows, gap_seconds=60)
        by_id = {item["id"]: item for item in assigned}

        self.assertEqual(by_id["raw-a"]["session_group"], by_id["raw-b"]["session_group"])
        self.assertEqual(
            by_id["legacy-a"]["session_group"],
            by_id["legacy-b"]["session_group"],
        )
        self.assertNotEqual(
            by_id["legacy-b"]["session_group"],
            by_id["legacy-c"]["session_group"],
        )

    def test_thins_only_within_same_session_class_and_motion(self):
        rows = assign_sessions(
            [
                row("first", 0, raw_session_id="session-a"),
                row("burst", 0.1, raw_session_id="session-a"),
                row(
                    "other-class",
                    0.1,
                    class_label="thumb_heavy",
                    raw_session_id="session-a",
                ),
                row("other-session", 0.1, raw_session_id="session-b"),
            ]
        )

        selected, thinned = thin_temporal(rows, min_gap=0.25)

        self.assertEqual(1, thinned)
        self.assertEqual(
            {"first", "other-class", "other-session"},
            {item["id"] for item in selected},
        )

    def test_domain_and_collection_folds_keep_sessions_isolated(self):
        rows = []
        for domain_index, domain in enumerate(("person-a__device-a", "person-b__device-b")):
            collection = domain + "__2026-07-22"
            for class_index, class_label in enumerate(CLASS_ORDER):
                rows.append(
                    row(
                        f"{domain_index}-{class_index}",
                        domain_index * 100 + class_index,
                        domain=domain,
                        collection=collection,
                        class_label=class_label,
                        raw_session_id=f"session-{domain_index}-{class_index}",
                    )
                )
        assigned = assign_sessions(rows)

        domain_folds = build_domain_folds(assigned, minimum_per_class=1)
        collection_folds = build_collection_folds(assigned, minimum_per_class=1)

        self.assertEqual(2, len(domain_folds))
        self.assertEqual(2, len(collection_folds))
        for fold_rows in list(domain_folds.values()) + list(collection_folds.values()):
            validate_no_session_leakage(fold_rows)
            self.assertEqual({"train", "valid"}, {item["split"] for item in fold_rows})

    def test_temporal_tail_holds_out_complete_latest_sessions(self):
        rows = []
        for session_index in range(4):
            for class_index, class_label in enumerate(CLASS_ORDER):
                rows.append(
                    row(
                        f"{session_index}-{class_index}",
                        session_index * 100 + class_index,
                        class_label=class_label,
                        raw_session_id=f"session-{session_index}",
                    )
                )
        assigned = assign_sessions(rows)

        fold_rows = build_temporal_tail_fold(assigned, validation_fraction=0.25)

        validate_no_session_leakage(fold_rows)
        valid_sessions = {
            item["session_group"] for item in fold_rows if item["split"] == "valid"
        }
        self.assertEqual(1, len(valid_sessions))
        self.assertTrue(
            all(
                item["split"] == "valid"
                for item in fold_rows
                if item["session_group"] in valid_sessions
            )
        )

    def test_build_folds_writes_serializable_reports(self):
        rows = []
        for domain_index, domain in enumerate(("person-a__device-a", "person-b__device-b")):
            for session_index in range(2):
                collection = domain + "__2026-07-22"
                for class_index, class_label in enumerate(CLASS_ORDER):
                    rows.append(
                        row(
                            f"{domain_index}-{session_index}-{class_index}",
                            domain_index * 1000 + session_index * 100 + class_index,
                            domain=domain,
                            collection=collection,
                            class_label=class_label,
                            raw_session_id=f"session-{domain_index}-{session_index}",
                        )
                    )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_manifest = root / "input.csv"
            with input_manifest.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            summary = build_folds(
                input_manifest,
                root / "output",
                min_gap=0,
                minimum_per_class=1,
            )

            report = json.loads((root / "output" / "fold_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["trainingRows"], report["trainingRows"])
            self.assertTrue(list((root / "output" / "folds").glob("domain_*.csv")))


if __name__ == "__main__":
    unittest.main()
