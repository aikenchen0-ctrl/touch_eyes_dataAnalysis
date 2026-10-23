import unittest

import numpy as np

from android.analysis import diagnose_full_touch_separability as diagnostic


class FullTouchDiagnosticTests(unittest.TestCase):
    def test_auc_handles_ties(self):
        self.assertAlmostEqual(
            0.875,
            diagnostic.auc((0, 0, 1, 1), (0.0, 1.0, 1.0, 2.0)),
        )

    def test_recall_at_fpr_uses_strict_threshold(self):
        result = diagnostic.recall_at_fpr(
            (0, 0, 0, 1, 1),
            (0.1, 0.2, 0.3, 0.4, 0.5),
            0.0,
        )

        self.assertEqual(0.0, result["fpr"])
        self.assertEqual(1.0, result["recall"])

    def test_within_group_center_removes_session_offset(self):
        values = np.asarray(((1.0, 4.0), (3.0, 8.0), (10.0, 2.0)))
        centered = diagnostic.within_group_center(
            values,
            np.asarray(("a", "a", "b")),
        )

        np.testing.assert_allclose(
            centered[:, 0],
            (-1.0, 1.0, 0.0),
        )

    def test_session_split_holds_out_whole_single_class_sessions(self):
        samples = []
        for label in (0, 1):
            for posture in ("supine", "side_left"):
                for session_index in range(3):
                    session = (
                        f"{'light' if label == 0 else 'heavy'}-"
                        f"{posture}-{session_index}"
                    )
                    samples.append({
                        "path": session,
                        "label": label,
                        "data": {
                            "collectionSessionID": session,
                            "collectionContext": {
                                "bodyPosture": posture,
                            },
                        },
                    })

        train, valid, validation_sessions = diagnostic.session_split(
            samples,
            seed=7,
        )

        train_sessions = {
            sample["data"]["collectionSessionID"] for sample in train
        }
        valid_sessions = {
            sample["data"]["collectionSessionID"] for sample in valid
        }
        self.assertFalse(train_sessions.intersection(valid_sessions))
        self.assertEqual(set(validation_sessions), valid_sessions)
        self.assertEqual({0, 1}, {sample["label"] for sample in train})
        self.assertEqual({0, 1}, {sample["label"] for sample in valid})
        self.assertEqual(
            {"supine", "side_left"},
            {
                sample["data"]["collectionContext"]["bodyPosture"]
                for sample in valid
            },
        )

    def test_mixed_prefix_groups_interleaved_session_rows(self):
        samples = []
        for index in range(8):
            for session, label, posture in (
                ("light-supine", 0, "supine"),
                ("heavy-supine", 1, "supine"),
                ("light-side", 0, "side_left"),
                ("heavy-side", 1, "side_left"),
            ):
                samples.append({
                    "path": f"{session}-{index}",
                    "label": label,
                    "data": {
                        "collectionSessionID": session,
                        "collectionContext": {
                            "bodyPosture": posture,
                        },
                    },
                })

        selected = diagnostic.choose_mixed_prefix(
            samples,
            minimum_samples=32,
        )

        self.assertEqual(32, len(selected))
        self.assertEqual(
            {
                "light-supine",
                "heavy-supine",
                "light-side",
                "heavy-side",
            },
            {
                sample["data"]["collectionSessionID"]
                for sample in selected
            },
        )


if __name__ == "__main__":
    unittest.main()
