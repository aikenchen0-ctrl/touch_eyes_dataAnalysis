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

    /** 中文注释：同一次触摸固定使用按下时建立的时钟原点，避免每个回调的亚毫秒偏移抖动。 */
    public static long touchEventUptimeMillisToElapsedRealtimeNanos(
            long eventUptimeMillis,
            long downEventUptimeMillis,
            long downEventElapsedRealtimeNanos
    ) {
        return downEventElapsedRealtimeNanos
                + (eventUptimeMillis - downEventUptimeMillis) * NANOS_PER_MILLISECOND;
    }
}
