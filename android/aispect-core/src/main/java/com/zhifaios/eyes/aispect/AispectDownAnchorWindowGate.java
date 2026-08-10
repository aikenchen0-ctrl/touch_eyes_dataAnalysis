package com.zhifaios.eyes.aispect;

/** 记录一轮按下期固定窗口是否已经发布，避免同一触摸重复输出。 */
final class AispectDownAnchorWindowGate {
    private boolean active;
    private boolean published;

    void begin() {
        active = true;
        published = false;
    }

    void reset() {
        active = false;
        published = false;
    }

    boolean tryPublish(AispectSignalWindowBuilder.Window window) {
        if (!active || published || window == null || !window.windowComplete) {
            return false;
        }
        published = true;
        return true;
    }
}
