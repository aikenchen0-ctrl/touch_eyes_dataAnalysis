package com.zhifaios.eyes.aispect;

final class AispectPostLiftCaptureGate {
    private AispectPostLiftCaptureGate() {
    }

    static boolean shouldWait(
            AispectSignalWindowBuilder.Window window,
            long elapsedSinceLiftMs,
            long timeoutMs
    ) {
        return window != null
                && window.preDownComplete
                && !window.postLiftComplete
                && elapsedSinceLiftMs >= 0L
                && elapsedSinceLiftMs < Math.max(0L, timeoutMs);
    }
}
