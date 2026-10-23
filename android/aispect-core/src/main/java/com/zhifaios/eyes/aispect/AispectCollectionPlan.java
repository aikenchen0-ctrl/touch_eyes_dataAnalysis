package com.zhifaios.eyes.aispect;

import java.util.Random;
import java.util.UUID;

/** 每轮交错四类强度样本；场景只作条件审计。 */
public final class AispectCollectionPlan {
    public static final int GROUP_COUNT = 4;
    public static final int MIN_TARGET_SAMPLES = 1;
    public static final int MAX_TARGET_SAMPLES = 1000;

    /** 由调用方按当前任务条件提供每类已落盘数量。 */
    public interface CountProvider {
        int savedCount(AispectModels.DatasetLabel label);
    }

    private final Random random;
    private final String scheduleIdBase;
    private int scheduleRevision;
    private AispectCollectionTask scene = AispectCollectionTask.TILTED_HANDHELD;
    private int cycleIndex;
    private int stepIndex;
    private int[] scheduleOrder;
    private int targetSamplesPerCell = AispectCollectionTask.TARGET_PER_CELL;

    public AispectCollectionPlan() {
        this(new Random(), UUID.randomUUID().toString());
    }

    public AispectCollectionPlan(long seed, String scheduleId) {
        this(new Random(seed), scheduleId);
    }

    private AispectCollectionPlan(Random random, String scheduleId) {
        if (random == null || scheduleId == null || scheduleId.isEmpty()) {
            throw new IllegalArgumentException("invalid_collection_schedule");
        }
        this.random = random;
        this.scheduleIdBase = scheduleId;
        scheduleOrder = shuffledOrder(null);
    }

    public AispectCollectionTask scene() { return scene; }
    public int cycleIndex() { return cycleIndex; }
    public int cycleNumber() { return cycleIndex + 1; }
    public int stepIndex() { return stepIndex; }
    public String scheduleId() {
        return scheduleRevision == 0 ? scheduleIdBase : scheduleIdBase + "-" + scheduleRevision;
    }
    public int[] scheduleOrder() { return scheduleOrder.clone(); }
    public int targetSamplesPerCell() { return targetSamplesPerCell; }

    public boolean setTargetSamplesPerCell(int target, boolean locked) {
        if (locked || target < MIN_TARGET_SAMPLES || target > MAX_TARGET_SAMPLES) return false;
        if (targetSamplesPerCell != target) {
            targetSamplesPerCell = target;
        }
        return true;
    }

    public boolean selectScene(AispectCollectionTask next, boolean locked) {
        if (locked || next == null || next == AispectCollectionTask.UNSPECIFIED) return false;
        if (scene != next) {
            scene = next;
            restartSchedule();
        }
        return true;
    }

    public boolean advance(int[] counts, boolean locked) {
        validateCounts(counts);
        if (locked || allComplete(counts)) return false;
        int next = nextPendingStep(counts, stepIndex + 1);
        if (next >= GROUP_COUNT) {
            beginNextCycle();
            next = nextPendingStep(counts, 0);
        }
        if (next >= GROUP_COUNT) return false;
        stepIndex = next;
        return true;
    }

    public void restartSchedule() {
        scheduleRevision += 1;
        cycleIndex = 0;
        stepIndex = 0;
        scheduleOrder = shuffledOrder(null);
    }

    public boolean allComplete(int[] counts) {
        validateCounts(counts);
        for (int action : scheduleOrder) {
            if (counts[action] < targetSamplesPerCell) return false;
        }
        return true;
    }

    public boolean advance(CountProvider counts, String grip, String support, boolean locked) {
        if (counts == null) throw new IllegalArgumentException("invalid_collection_counts");
        if (locked || allComplete(counts, grip, support)) return false;
        int[] values = counts(grip, support, counts);
        int next = nextPendingStep(values, stepIndex + 1);
        if (next >= GROUP_COUNT) {
            beginNextCycle();
            next = nextPendingStep(values, 0);
        }
        if (next >= GROUP_COUNT) return false;
        stepIndex = next;
        return true;
    }

    public boolean allComplete(CountProvider counts, String grip, String support) {
        return allComplete(counts(grip, support, counts));
    }

    public int savedCount(int action, CountProvider counts, String grip, String support) {
        if (counts == null) throw new IllegalArgumentException("invalid_collection_counts");
        if (action < 0 || action >= GROUP_COUNT) throw new IllegalArgumentException("invalid_collection_action");
        return counts.savedCount(labelForAction(action, grip, support));
    }

    public int currentAction() {
        return scheduleOrder[stepIndex];
    }

    public AispectModels.DatasetLabel label(String grip, String support) {
        return labelForAction(currentAction(), grip, support);
    }

    /** 普通点击没有手指监督含义，单独使用误触审计协议。 */
    public AispectModels.DatasetLabel negativeLabel(String grip, String support) {
        return new AispectModels.DatasetLabel(
                scene.motionState(),
                AispectModels.TouchStrength.ORDINARY,
                AispectModels.FingerType.THUMB,
                AispectCollectionContext.negative(
                        scene,
                        grip,
                        support,
                        targetSamplesPerCell
                )
        );
    }

    public AispectModels.DatasetLabel labelForAction(int action, String grip, String support) {
        if (action < 0 || action >= GROUP_COUNT) throw new IllegalArgumentException("invalid_collection_action");
        String[] labels = currentScheduleLabels();
        return new AispectModels.DatasetLabel(scene.motionState(),
                action % 2 == 0 ? AispectModels.TouchStrength.LIGHT : AispectModels.TouchStrength.HEAVY,
                action < 2 ? AispectModels.FingerType.THUMB : AispectModels.FingerType.INDEX,
                new AispectCollectionContext(scene, grip, support, targetSamplesPerCell,
                        AispectCollectionContext.INTERLEAVED_PROTOCOL, scheduleId(),
                        cycleIndex, stepIndex, labels));
    }

    public static String actionKey(int action) {
        if (action < 0 || action >= GROUP_COUNT) throw new IllegalArgumentException("invalid_collection_action");
        return (action < 2 ? "thumb" : "index") + "_" + (action % 2 == 0 ? "light" : "heavy");
    }

    private int nextPendingStep(int[] counts, int start) {
        for (int step = start; step < GROUP_COUNT; step++) {
            if (counts[scheduleOrder[step]] < targetSamplesPerCell) return step;
        }
        return GROUP_COUNT;
    }

    private void beginNextCycle() {
        int previousLast = scheduleOrder[GROUP_COUNT - 1];
        cycleIndex += 1;
        stepIndex = 0;
        scheduleOrder = shuffledOrder(previousLast);
    }

    private int[] shuffledOrder(Integer previousLast) {
        int[] order = new int[]{0, 1, 2, 3};
        for (int i = order.length - 1; i > 0; i--) {
            int swap = random.nextInt(i + 1);
            int value = order[i];
            order[i] = order[swap];
            order[swap] = value;
        }
        if (previousLast != null && order[0] == previousLast) {
            int swap = 1 + random.nextInt(order.length - 1);
            int value = order[0];
            order[0] = order[swap];
            order[swap] = value;
        }
        return order;
    }

    private String[] currentScheduleLabels() {
        String[] labels = new String[GROUP_COUNT];
        for (int index = 0; index < GROUP_COUNT; index++) {
            labels[index] = actionKey(scheduleOrder[index]);
        }
        return labels;
    }

    private int[] counts(String grip, String support, CountProvider provider) {
        int[] values = new int[GROUP_COUNT];
        for (int action = 0; action < GROUP_COUNT; action++) {
            values[action] = provider.savedCount(labelForAction(action, grip, support));
        }
        return values;
    }

    private static void validateCounts(int[] counts) {
        if (counts == null || counts.length != GROUP_COUNT) {
            throw new IllegalArgumentException("invalid_collection_counts");
        }
        for (int count : counts) {
            if (count < 0) throw new IllegalArgumentException("invalid_collection_counts");
        }
    }
}
