package com.zhifaios.eyes.aispect;

/** 中文注释：把 MotionEvent 的 uptime 毫秒映射到传感器共用的 elapsedRealtimeNanos 时钟。 */
public final class AispectCaptureClock {
    private static final long NANOS_PER_MILLISECOND = 1_000_000L;

    private AispectCaptureClock() {
    }

    public static long uptimeMillisToElapsedRealtimeNanos(
            long eventUptimeMillis,
            long observedUptimeMillis,
            long observedElapsedRealtimeNanos
    ) {
        return observedElapsedRealtimeNanos
                + (eventUptimeMillis - observedUptimeMillis) * NANOS_PER_MILLISECOND;
    }
}
