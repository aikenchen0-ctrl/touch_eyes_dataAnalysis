package com.zhifaios.eyes.aispect;

import org.json.JSONException;
import org.json.JSONArray;
import org.json.JSONObject;

/** 不可变任务上下文随原标签在按下时冻结；未知字段不推测补齐。 */
public final class AispectCollectionContext {
    public static final String CONTRACT_V1 = "touch_collection_context_v1";
    public static final String CONTRACT_V2 = "touch_collection_context_v2";
    public static final String CONTRACT = CONTRACT_V2;
    public static final String BLOCK_PROTOCOL = "block_v1";
    public static final String INTERLEAVED_PROTOCOL = "interleaved_v2";
    public static final String NEGATIVE_PROTOCOL = "negative_tap_v1";
    public static final String[] GRIP_OPTIONS = {
            "unspecified", "same_hand", "other_hand", "two_hands", "not_held"
    };
    public static final String[] SUPPORT_OPTIONS = {
            "unspecified", "handheld", "hard_surface", "soft_surface", "stand", "body"
    };
    public final AispectCollectionTask task;
    public final String grip;
    public final String support;
    public final int targetPerCell;
    public final String collectionProtocol;
    public final String scheduleId;
    public final int cycleIndex;
    public final int stepIndex;
    public final String[] scheduleOrder;

    public AispectCollectionContext(AispectCollectionTask task, String grip, String support) {
        this(task, grip, support, AispectCollectionTask.TARGET_PER_CELL);
    }

    public AispectCollectionContext(AispectCollectionTask task, String grip, String support, int targetPerCell) {
        this(task, grip, support, targetPerCell, BLOCK_PROTOCOL, "legacy_block",
                -1, -1, new String[0]);
    }

    public AispectCollectionContext(AispectCollectionTask task, String grip, String support, int targetPerCell,
                                    String collectionProtocol, String scheduleId, int cycleIndex, int stepIndex,
                                    String[] scheduleOrder) {
        if (task == null || !contains(GRIP_OPTIONS, grip) || !contains(SUPPORT_OPTIONS, support) || targetPerCell < 1) {
            throw new IllegalArgumentException("invalid_collection_context");
        }
        if (collectionProtocol == null || collectionProtocol.isEmpty()
                || scheduleId == null || scheduleId.isEmpty()
                || cycleIndex < -1 || stepIndex < -1 || scheduleOrder == null) {
            throw new IllegalArgumentException("invalid_collection_schedule");
        }
        if (INTERLEAVED_PROTOCOL.equals(collectionProtocol)) {
            if (cycleIndex < 0 || stepIndex < 0 || stepIndex >= AispectCollectionPlan.GROUP_COUNT
                    || scheduleOrder.length != AispectCollectionPlan.GROUP_COUNT) {
                throw new IllegalArgumentException("invalid_interleaved_schedule");
            }
            boolean[] seen = new boolean[AispectCollectionPlan.GROUP_COUNT];
            for (String action : scheduleOrder) {
                int index = actionIndex(action);
                if (index < 0 || seen[index]) {
                    throw new IllegalArgumentException("invalid_interleaved_schedule");
                }
                seen[index] = true;
            }
        }
        this.task = task;
        this.grip = grip;
        this.support = support;
        this.targetPerCell = targetPerCell;
        this.collectionProtocol = collectionProtocol;
        this.scheduleId = scheduleId;
        this.cycleIndex = cycleIndex;
        this.stepIndex = stepIndex;
        this.scheduleOrder = scheduleOrder.clone();
    }

    public static AispectCollectionContext unspecified() {
        return new AispectCollectionContext(AispectCollectionTask.UNSPECIFIED, "unspecified", "unspecified");
    }

    /** 普通点击只用于误触率审计，不进入四类轻重监督采集。 */
    public static AispectCollectionContext negative(
            AispectCollectionTask task,
            String grip,
            String support,
            int targetPerCell
    ) {
        return new AispectCollectionContext(
                task,
                grip,
                support,
                targetPerCell,
                NEGATIVE_PROTOCOL,
                "negative_tap",
                -1,
                -1,
                new String[0]
        );
    }

    public String progressKey(String participant, String finger, String strength) {
        return collectionProtocol + "|" + participant + "|" + task.id + "|" + grip + "|" + support
                + "|" + finger + "|" + strength;
    }

    JSONObject toJson() throws JSONException {
        JSONObject value = new JSONObject();
        boolean interleaved = INTERLEAVED_PROTOCOL.equals(collectionProtocol);
        value.put("schemaVersion", interleaved ? 2 : 1);
        value.put("contract", interleaved ? CONTRACT_V2 : CONTRACT_V1);
        value.put("collectionProtocol", collectionProtocol);
        if (interleaved) {
            value.put("scheduleId", scheduleId);
            value.put("cycleIndex", cycleIndex);
            value.put("stepIndex", stepIndex);
            value.put("scheduleOrder", new JSONArray(scheduleOrder));
        }
        value.put("strengthTruthSource", "user_task");
        value.put("taskId", task.id);
        value.put("bodyPosture", task.bodyPosture);
        value.put("requestedDevicePose", task.requestedDevicePose);
        value.put("sceneSource", "task_instruction_not_verified");
        value.put("grip", grip);
        value.put("support", support);
        value.put("contactMode", "natural");
        value.put("gestureIntent", task == AispectCollectionTask.UNSPECIFIED ? "unknown" : "tap");
        value.put("strengthOnset", task == AispectCollectionTask.UNSPECIFIED ? "unknown" : "at_contact");
        value.put("labelStatus", "task_assigned");
        value.put("targetPerCell", targetPerCell);
        return value;
    }

    private static boolean contains(String[] values, String candidate) {
        for (String value : values) {
            if (value.equals(candidate)) return true;
        }
        return false;
    }

    private static int actionIndex(String action) {
        for (int index = 0; index < AispectCollectionPlan.GROUP_COUNT; index++) {
            if (AispectCollectionPlan.actionKey(index).equals(action)) return index;
        }
        return -1;
    }
}
