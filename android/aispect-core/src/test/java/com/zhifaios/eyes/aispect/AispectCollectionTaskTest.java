package com.zhifaios.eyes.aispect;

import org.junit.Test;
import java.util.HashSet;
import java.util.Set;
import static org.junit.Assert.*;

public final class AispectCollectionTaskTest {
    @Test public void sixTasksHaveSeparateIdentifiersAndNoStrengthLabel() {
        Set<String> ids = new HashSet<>();
        for (AispectCollectionTask task : AispectCollectionTask.guidedTasks()) {
            assertTrue(ids.add(task.id));
            assertFalse(task.id.contains("heavy"));
            assertFalse(task.id.contains("light"));
        }
        assertEquals(6, ids.size());
        assertEquals(240, ids.size() * 2 * 2 * AispectCollectionTask.TARGET_PER_CELL);
    }

    @Test public void contextRetainsUnknownAndDoesNotChangeLegacyLabel() {
        AispectModels.DatasetLabel old = new AispectModels.DatasetLabel(
                AispectModels.MotionState.HANDHELD_STATIC, AispectModels.TouchStrength.LIGHT,
                AispectModels.FingerType.INDEX);
        assertEquals("handheld_static_light_index", old.datasetKey());
        assertEquals(AispectCollectionTask.UNSPECIFIED, old.collectionContext.task);
        assertEquals("unspecified", old.collectionContext.grip);
    }

    @Test public void progressSeparatesParticipantFingerStrengthAndContext() {
        AispectCollectionContext context = new AispectCollectionContext(
                AispectCollectionTask.SUPINE, "other_hand", "handheld");
        String key = context.progressKey("p1", "index", "light");
        assertNotEquals(key, context.progressKey("p2", "index", "light"));
        assertNotEquals(key, context.progressKey("p1", "thumb", "light"));
        assertNotEquals(key, context.progressKey("p1", "index", "heavy"));
    }

    @Test public void repeatedReviewIsIdempotentAndCanRestoreEligibility() {
        AispectCollectionProgress progress = new AispectCollectionProgress();
        AispectCollectionProgress.Entry first = progress.recordSaved("first", "cell");
        progress.recordSaved("second", "cell");
        progress.review(first, "mistake");
        progress.review(first, "uncertain");
        assertEquals(1, progress.count("cell"));
        progress.review(first, "confirmed");
        progress.review(first, "confirmed");
        assertEquals(2, progress.count("cell"));
    }

    @Test(expected = IllegalArgumentException.class)
    public void invalidReviewDoesNotBecomeAClassLabel() {
        AispectCollectionProgress.validateReviewStatus("heavy");
    }

    @Test public void metadataSeparatesRequestedPoseFromObservedReality() throws Exception {
        org.json.JSONObject context = new AispectCollectionContext(
                AispectCollectionTask.INVERTED, "unspecified", "unspecified").toJson();
        assertEquals("touch_collection_context_v1", context.getString("contract"));
        assertEquals(1, context.getInt("schemaVersion"));
        assertEquals("block_v1", context.getString("collectionProtocol"));
        assertEquals("vertical_top_down", context.getString("requestedDevicePose"));
        assertEquals("task_instruction_not_verified", context.getString("sceneSource"));
        assertEquals("at_contact", context.getString("strengthOnset"));
        assertEquals("tap", context.getString("gestureIntent"));
        assertEquals("task_assigned", context.getString("labelStatus"));
        assertFalse(context.has("pressure"));
        assertFalse(context.has("touchMajor"));
    }

    @Test(expected = IllegalArgumentException.class)
    public void inventedSupportIsRejected() {
        new AispectCollectionContext(AispectCollectionTask.SUPINE, "unspecified", "guessed_from_gravity");
    }

    @Test public void ordinaryNegativeContextIsAuditableAndIndependent() throws Exception {
        AispectCollectionContext context = AispectCollectionContext.negative(
                AispectCollectionTask.TABLETOP,
                "two_hands",
                "hard_surface",
                40
        );
        org.json.JSONObject value = context.toJson();

        assertEquals("negative_tap_v1", value.getString("collectionProtocol"));
        assertEquals("touch_collection_context_v1", value.getString("contract"));
        assertEquals("user_task", value.getString("strengthTruthSource"));
        assertEquals("tabletop", value.getString("taskId"));
        assertEquals(40, value.getInt("targetPerCell"));
    }
}
