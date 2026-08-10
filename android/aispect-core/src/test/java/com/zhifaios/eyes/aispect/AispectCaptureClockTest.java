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
}
