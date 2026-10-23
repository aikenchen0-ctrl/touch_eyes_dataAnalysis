package com.zhifaios.eyes.aispect;

import org.junit.Test;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import static org.junit.Assert.*;

public final class AispectCollectionPlanTest {
    private static final String GRIP = "unspecified";
    private static final String SUPPORT = "unspecified";

    @Test public void eightScenesUseTiltedInsteadOfSeatedAndRemoveUprightEntry() {
        Set<String> ids = new HashSet<>();
        for (AispectCollectionTask scene : AispectCollectionTask.collectionScenes()) assertTrue(ids.add(scene.id));
        assertEquals(8, ids.size());
        assertTrue(ids.contains("tilted_handheld"));
        assertFalse(ids.contains("seated_natural"));
        assertFalse(ids.contains("upright"));
        assertEquals("seated_natural", AispectCollectionTask.SEATED_NATURAL.id);
        assertEquals("upright", AispectCollectionTask.UPRIGHT.id);
        assertTrue(ids.contains("walking"));
        assertTrue(ids.contains("tabletop"));
        assertTrue(ids.contains("handheld"));
    }

    @Test public void everyCycleContainsAllFourClassesExactlyOnce() {
        AispectCollectionPlan plan = new AispectCollectionPlan(7L, "schedule-7");
        plan.setTargetSamplesPerCell(100, false);
        int[] counts = new int[AispectCollectionPlan.GROUP_COUNT];
        for (int cycle = 0; cycle < 12; cycle++) {
            Set<Integer> actions = new HashSet<>();
            for (int step = 0; step < AispectCollectionPlan.GROUP_COUNT; step++) {
                int action = plan.currentAction();
                assertTrue(actions.add(action));
                assertEquals(cycle, plan.cycleIndex());
                assertEquals(step, plan.stepIndex());
                counts[action] += 1;
                plan.advance(counts, false);
            }
            assertEquals(4, actions.size());
        }
        assertEquals(12, counts[0]);
        assertEquals(12, counts[1]);
        assertEquals(12, counts[2]);
        assertEquals(12, counts[3]);
    }

    @Test public void sameClassNeverAppearsAtAdjacentCycleBoundary() {
        AispectCollectionPlan plan = new AispectCollectionPlan(19L, "schedule-19");
        plan.setTargetSamplesPerCell(100, false);
        int[] counts = new int[AispectCollectionPlan.GROUP_COUNT];
        int previous = -1;
        for (int sample = 0; sample < 160; sample++) {
            int current = plan.currentAction();
            assertNotEquals(previous, current);
            previous = current;
            counts[current] += 1;
            plan.advance(counts, false);
        }
    }

    @Test public void scheduleIsReproducibleForSameSeedButChangesWithSeed() {
        AispectCollectionPlan first = new AispectCollectionPlan(31L, "schedule-a");
        AispectCollectionPlan second = new AispectCollectionPlan(31L, "schedule-a");
        AispectCollectionPlan other = new AispectCollectionPlan(32L, "schedule-b");
        assertArrayEquals(first.scheduleOrder(), second.scheduleOrder());
        assertNotEquals(first.scheduleId(), other.scheduleId());
        assertEquals(AispectCollectionPlan.GROUP_COUNT, first.scheduleOrder().length);
    }

    @Test public void progressStopsOnlyAfterEveryClassReachesQuota() {
        AispectCollectionPlan plan = new AispectCollectionPlan(5L, "schedule-5");
        assertTrue(plan.setTargetSamplesPerCell(2, false));
        assertFalse(plan.allComplete(new int[]{2, 2, 1, 2}));
        assertTrue(plan.allComplete(new int[]{2, 2, 2, 2}));
        assertFalse(plan.advance(new int[]{2, 2, 2, 2}, false));
    }

    @Test public void rejectedSampleCanBeRefilledInNextCycle() {
        AispectCollectionPlan plan = new AispectCollectionPlan(11L, "schedule-11");
        assertTrue(plan.setTargetSamplesPerCell(1, false));
        int[] counts = new int[]{1, 1, 1, 1};
        assertTrue(plan.allComplete(counts));
        counts[plan.scheduleOrder()[2]] = 0;
        assertFalse(plan.allComplete(counts));
        int initial = plan.currentAction();
        for (int step = 0; step < AispectCollectionPlan.GROUP_COUNT; step++) {
            if (counts[plan.currentAction()] < 1) break;
            plan.advance(counts, false);
        }
        assertTrue(counts[plan.currentAction()] < 1);
    }

    @Test public void switchingSceneRestartsScheduleAndSessionMetadata() {
        AispectCollectionPlan plan = new AispectCollectionPlan(3L, "schedule-3");
        plan.advance(new int[]{1, 0, 0, 0}, false);
        String oldId = plan.scheduleId();
        int[] oldOrder = plan.scheduleOrder();
        assertTrue(plan.selectScene(AispectCollectionTask.INVERTED, false));
        assertEquals(0, plan.cycleIndex());
        assertEquals(0, plan.stepIndex());
        assertNotEquals(oldId, plan.scheduleId());
        assertEquals(AispectCollectionPlan.GROUP_COUNT, plan.scheduleOrder().length);
        assertFalse(Arrays.equals(oldOrder, plan.scheduleOrder()));
        assertEquals(AispectCollectionContext.INTERLEAVED_PROTOCOL,
                plan.label(GRIP, SUPPORT).collectionContext.collectionProtocol);
    }

    @Test public void contextRecordsAuditableInterleavingWithoutChangingTruth() throws Exception {
        AispectCollectionPlan plan = new AispectCollectionPlan(17L, "schedule-17");
        AispectModels.DatasetLabel label = plan.label(GRIP, SUPPORT);
        org.json.JSONObject context = label.collectionContext.toJson();
        assertEquals("touch_collection_context_v2", context.getString("contract"));
        assertEquals(2, context.getInt("schemaVersion"));
        assertEquals(AispectCollectionContext.INTERLEAVED_PROTOCOL,
                context.getString("collectionProtocol"));
        assertEquals("schedule-17", context.getString("scheduleId"));
        assertEquals("user_task", context.getString("strengthTruthSource"));
        assertEquals(0, context.getInt("cycleIndex"));
        assertEquals(0, context.getInt("stepIndex"));
        assertEquals(4, context.getJSONArray("scheduleOrder").length());
        assertEquals(AispectCollectionPlan.actionKey(plan.currentAction()),
                context.getJSONArray("scheduleOrder").getString(0));
    }

    @Test public void configurableQuotaStillAppliesPerClass() {
        for (int target : new int[]{1, 3, 25, 1000}) {
            AispectCollectionPlan plan = new AispectCollectionPlan(13L, "schedule-13");
            assertTrue(plan.setTargetSamplesPerCell(target, false));
            int[] counts = new int[]{target, target, target, target - 1};
            assertFalse(plan.allComplete(counts));
            counts[3] = target;
            assertTrue(plan.allComplete(counts));
        }
    }

    @Test public void completionDependsOnlyOnFourPositiveClasses() {
        AispectCollectionPlan plan = new AispectCollectionPlan(23L, "schedule-23");
        plan.setTargetSamplesPerCell(2, false);
        assertTrue(plan.allComplete(label -> 2, GRIP, SUPPORT));
        assertFalse(plan.advance(label -> 2, GRIP, SUPPORT, false));
    }

    @Test public void ordinaryNegativeUsesDedicatedProtocolWithoutChangingSchedule() throws Exception {
        AispectCollectionPlan plan = new AispectCollectionPlan(29L, "schedule-29");
        int actionBefore = plan.currentAction();
        AispectModels.DatasetLabel label = plan.negativeLabel(GRIP, SUPPORT);

        assertEquals(AispectModels.TouchStrength.ORDINARY, label.touchStrength);
        assertEquals(AispectCollectionContext.NEGATIVE_PROTOCOL,
                label.collectionContext.collectionProtocol);
        assertEquals(actionBefore, plan.currentAction());
        org.json.JSONObject context = label.collectionContext.toJson();
        assertEquals("touch_collection_context_v1", context.getString("contract"));
        assertEquals("negative_tap_v1", context.getString("collectionProtocol"));
        assertEquals("user_task", context.getString("strengthTruthSource"));
        assertFalse(context.has("scheduleOrder"));
    }

    @Test public void interleavedLabelsKeepOneCollectionSessionCondition() {
        AispectCollectionPlan plan = new AispectCollectionPlan(37L, "schedule-37");
        AispectModels.DatasetLabel first = plan.labelForAction(0, GRIP, SUPPORT);
        for (int action = 1; action < AispectCollectionPlan.GROUP_COUNT; action++) {
            AispectModels.DatasetLabel current = plan.labelForAction(action, GRIP, SUPPORT);
            assertTrue(AispectCollectionController.sameCollectionCondition(first, current));
        }
        assertFalse(AispectCollectionController.sameCollectionCondition(
                first,
                plan.negativeLabel(GRIP, SUPPORT)
        ));
    }
}
