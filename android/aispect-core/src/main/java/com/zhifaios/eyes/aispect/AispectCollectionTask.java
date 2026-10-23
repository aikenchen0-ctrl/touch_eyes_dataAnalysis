package com.zhifaios.eyes.aispect;

/** 场景只描述采集指令，不将姿态当作轻重真值；旧任务标识保持不变。 */
public enum AispectCollectionTask {
    UNSPECIFIED("unspecified", "unknown", "unconstrained"),
    SEATED_NATURAL("seated_natural", "seated", "natural"),
    UPRIGHT("upright", "unspecified", "vertical_top_up"),
    INVERTED("inverted", "unspecified", "vertical_top_down"),
    SUPINE("supine", "supine", "screen_down"),
    SIDE_LEFT("side_left", "side_left", "natural"),
    SIDE_RIGHT("side_right", "side_right", "natural"),
    HANDHELD("handheld", "unspecified", "natural"),
    TABLETOP("tabletop", "unspecified", "screen_up"),
    WALKING("walking", "walking", "natural"),
    TILTED_HANDHELD("tilted_handheld", "unspecified", "tilted_toward_user");

    public static final int TARGET_PER_CELL = 10;
    public final String id;
    public final String bodyPosture;
    public final String requestedDevicePose;

    AispectCollectionTask(String id, String bodyPosture, String requestedDevicePose) {
        this.id = id;
        this.bodyPosture = bodyPosture;
        this.requestedDevicePose = requestedDevicePose;
    }

    /** 保留旧版六任务合同；当前采集入口使用 collectionScenes。 */
    public static AispectCollectionTask[] guidedTasks() {
        return new AispectCollectionTask[]{
                SEATED_NATURAL, UPRIGHT, INVERTED, SUPINE, SIDE_LEFT, SIDE_RIGHT
        };
    }

    /** 当前八个场景；倾斜手持不再假定坐姿，正向竖直不再作为采集入口。 */
    public static AispectCollectionTask[] collectionScenes() {
        return new AispectCollectionTask[]{
                TILTED_HANDHELD, INVERTED, SUPINE, SIDE_LEFT, SIDE_RIGHT,
                WALKING, TABLETOP, HANDHELD
        };
    }

    public AispectModels.MotionState motionState() {
        if (this == TABLETOP) return AispectModels.MotionState.PLACED_STATIC;
        if (this == WALKING) return AispectModels.MotionState.WALKING;
        return AispectModels.MotionState.HANDHELD_STATIC;
    }
}
