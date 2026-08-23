package com.zhifaios.eyes.aispect;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public final class AispectCaptureClockTest {
    @Test
    public void convertsUptimeMillisUsingObservedClockPair() {
        assertEquals(
                9_995_000_000L,
                AispectCaptureClock.uptimeMillisToElapsedRealtimeNanos(9_995L, 10_000L, 10_000_000_000L)
        );
    }

    @Test
    public void keepsEqualTouchEventMillisStableFromOneDownAnchor() {
        long downUptimeMillis = 10_000L;
        long downElapsedRealtimeNanos = 10_000_800_000L;

        assertEquals(
                10_024_800_000L,
                AispectCaptureClock.touchEventUptimeMillisToElapsedRealtimeNanos(
                        10_024L,
                        downUptimeMillis,
                        downElapsedRealtimeNanos
                )
        );
        assertEquals(
                10_024_800_000L,
                AispectCaptureClock.touchEventUptimeMillisToElapsedRealtimeNanos(
                        10_024L,
                        downUptimeMillis,
                        downElapsedRealtimeNanos
                )
        );
    }
}
