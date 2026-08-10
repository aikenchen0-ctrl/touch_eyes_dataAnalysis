import unittest

from android.analysis import build_grouped_four_class_manifest as builder


class GroupedManifestTests(unittest.TestCase):
    def row(self, sample_id, collection, created_at, class_label="thumb_light"):
        return {
            "id": sample_id,
            "collection_name": collection,
            "created_at": str(created_at),
            "class_label": class_label,
            "motion_state": "handheld_static",
            "path": sample_id + ".json",
        }

    def test_transform_assigns_collection_holdout_and_time_blocks(self):
        master = [
            self.row("a", "train_collection", 1000.0),
            self.row("b", "train_collection", 1310.0),
            self.row("c", "valid_collection", 2000.0),
        ]
        quality = {
            sample_id: {"quality_bucket": "high_quality", "quality_score": "100", "sample_weight": "1.0"}
            for sample_id in ("a", "b", "c")
        }

        rows, _ = builder.transform_rows(
            master,
            quality,
            holdout_collection="valid_collection",
            block_seconds=300,
            min_sample_gap=0,
            unscored_collections=set(),
        )

        by_id = {row["id"]: row for row in rows}
        self.assertEqual(by_id["a"]["split"], "train")
        self.assertEqual(by_id["c"]["split"], "valid")
        self.assertNotEqual(by_id["a"]["session_group"], by_id["b"]["session_group"])

    def test_transform_excludes_unscored_new_rows_but_allows_legacy(self):
        master = [
            self.row("legacy", "old_lemon_tao", 1000.0),
            self.row("new", "new_collection", 1000.0),
        ]

        rows, summary = builder.transform_rows(
            master,
            {},
            holdout_collection="new_collection",
            block_seconds=300,
            min_sample_gap=0,
            unscored_collections={"old_lemon_tao"},
        )

        self.assertEqual([row["id"] for row in rows], ["legacy"])
        self.assertEqual(summary["excludedUnscored"], 1)

    def test_transform_deduplicates_ids_and_thins_temporal_bursts(self):
        master = [
            self.row("a", "collection", 1000.0),
            self.row("a", "collection", 1000.0),
            self.row("b", "collection", 1000.1),
            self.row("c", "collection", 1001.0),
        ]
        quality = {
            sample_id: {"quality_bucket": "high_quality", "quality_score": "100", "sample_weight": "1.0"}
            for sample_id in ("a", "b", "c")
        }

        rows, summary = builder.transform_rows(
            master,
            quality,
            holdout_collection="other",
            block_seconds=300,
            min_sample_gap=0.25,
            unscored_collections=set(),
        )

        self.assertEqual([row["id"] for row in rows], ["a", "c"])
        self.assertEqual(summary["duplicateIds"], 1)
        self.assertEqual(summary["thinnedRows"], 1)


if __name__ == "__main__":
    unittest.main()
