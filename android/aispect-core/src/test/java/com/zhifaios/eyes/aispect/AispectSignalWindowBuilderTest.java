package com.zhifaios.eyes.aispect;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public final class AispectSignalWindowBuilderTest {
    @Test
    public void dualAnchorWindowSpansDownMinus15ThroughLiftPlus15() {
        AispectSignalWindowBuilder builder = new AispectSignalWindowBuilder();
        WindowFixture fixture = fixture(140, 110);

        AispectSignalWindowBuilder.Window window = builder.buildDualAnchor(
                fixture.frames,
                fixture.downTimestamp,
                fixture.liftTimestamp,
                100.0
        );

        assertEquals(-15, window.firstFrameIndex);
        assertEquals(25, window.lastFrameIndex);
        assertEquals(41, window.frames.length);
        assertEquals(0, window.downReferenceFrameIndex);
        assertEquals(10, window.liftReferenceFrameIndex);
        assertTrue(window.windowComplete);
        assertEquals(-0.15, window.frames[0].timeSinceTouchDown, 1e-9);
        assertEquals(-0.25, window.frames[0].timeSinceAnchor, 1e-9);
    }

    @Test
    public void dualAnchorWindowDoesNotSynthesizeMissingPostLiftFrames() {
        AispectSignalWindowBuilder builder = new AispectSignalWindowBuilder();
        WindowFixture fixture = fixture(125, 110);

        AispectSignalWindowBuilder.Window window = builder.buildDualAnchor(
                fixture.frames,
                fixture.downTimestamp,
                fixture.liftTimestamp,
                100.0
        );

        assertFalse(window.windowComplete);
        assertTrue(window.frames.length < 41);
        assertEquals(125.0, window.frames[window.frames.length - 1].delta, 0.0);
        assertEquals(0.1, window.frames[window.frames.length - 1].x, 0.0);
    }

    @Test
    public void fixedDownWindowIsIncompleteWhenPlusNineIsUnavailable() {
        AispectSignalWindowBuilder builder = new AispectSignalWindowBuilder();
        builder.config().preFrames = 5;
        builder.config().postFrames = 9;
        WindowFixture fixture = fixture(53, 50);

        AispectSignalWindowBuilder.Window window = builder.build(
                fixture.frames,
                fixture.downTimestamp,
                100.0
        );

        assertEquals(-5, window.firstFrameIndex);
        assertEquals(9, window.lastFrameIndex);
        assertEquals(15, window.frames.length);
        assertFalse(window.windowComplete);
    }

    @Test
    public void dualAnchorWindowPreservesRawClockTimestamps() {
        AispectSignalWindowBuilder builder = new AispectSignalWindowBuilder();
        WindowFixture fixture = fixture(140, 110);

        AispectSignalWindowBuilder.Window window = builder.buildDualAnchor(
                fixture.frames,
                fixture.downTimestamp,
                fixture.liftTimestamp,
                100.0
        );

        assertEquals(85_000_000L, window.frames[0].sensorTimestampElapsedRealtimeNanos);
        assertEquals(85_100_000L, window.frames[0].receivedElapsedRealtimeNanos);
    }

    @Test
    public void highRateDualAnchorWindowKeepsAtLeastEightyFiveMilliseconds() {
        AispectSignalWindowBuilder builder = new AispectSignalWindowBuilder();
        WindowFixture fixture = fixture(220, 100, 130, 250.0);

        AispectSignalWindowBuilder.Window window = builder.buildDualAnchor(
                fixture.frames,
                fixture.downTimestamp,
                fixture.liftTimestamp,
                250.0
        );

        assertTrue(window.windowComplete);
        assertTrue(-window.firstFrameIndex > 15);
        assertTrue(window.lastFrameIndex - window.liftReferenceFrameIndex > 15);
        assertTrue(fixture.downTimestamp - window.frames[0].timestampSeconds >= 0.085);
        assertTrue(
                window.frames[window.frames.length - 1].timestampSeconds - fixture.liftTimestamp >= 0.085
        );
    }

    @Test
    public void lowRateDualAnchorWindowStillKeepsAtLeastFifteenFrames() {
        AispectSignalWindowBuilder builder = new AispectSignalWindowBuilder();
        WindowFixture fixture = fixture(60, 20, 25, 50.0);

        AispectSignalWindowBuilder.Window window = builder.buildDualAnchor(
                fixture.frames,
                fixture.downTimestamp,
                fixture.liftTimestamp,
                50.0
        );

        assertEquals(-15, window.firstFrameIndex);
        assertEquals(20, window.lastFrameIndex);
        assertEquals(5, window.liftReferenceFrameIndex);
        assertTrue(window.windowComplete);
    }

    @Test
    public void dualAnchorWindowIsIncompleteWhenPreDownDurationIsTooShort() {
        AispectSignalWindowBuilder builder = new AispectSignalWindowBuilder();
        WindowFixture fixture = fixture(100, 15, 45, 250.0);

        AispectSignalWindowBuilder.Window window = builder.buildDualAnchor(
                fixture.frames,
                fixture.downTimestamp,
                fixture.liftTimestamp,
                250.0
        );

        assertEquals(-15, window.firstFrameIndex);
        assertFalse(window.windowComplete);
    }

    @Test
    public void dualAnchorWindowIsIncompleteWhenPostLiftDurationIsTooShort() {
        AispectSignalWindowBuilder builder = new AispectSignalWindowBuilder();
        WindowFixture fixture = fixture(76, 40, 60, 250.0);

        AispectSignalWindowBuilder.Window window = builder.buildDualAnchor(
                fixture.frames,
                fixture.downTimestamp,
                fixture.liftTimestamp,
                250.0
        );

        assertEquals(15, window.lastFrameIndex - window.liftReferenceFrameIndex);
        assertFalse(window.windowComplete);
    }

    private static WindowFixture fixture(int frameCount, int liftFrameIndex) {
        return fixture(frameCount, 100, liftFrameIndex, 100.0);
    }

    private static WindowFixture fixture(
            int frameCount,
            int downFrameIndex,
            int liftFrameIndex,
            double sampleRateHz
    ) {
        List<AispectModels.ImpactFrame> frames = new ArrayList<>();
        for (int index = 0; index < frameCount; index++) {
            double timestamp = index / sampleRateHz;
            frames.add(new AispectModels.ImpactFrame(
                    index + 1,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    0.25,
                    0.75,
                    timestamp - (downFrameIndex / sampleRateHz),
                    timestamp - (liftFrameIndex / sampleRateHz),
                    timestamp,
                    index * 1_000_000L,
                    index * 1_000_000L + 100_000L
            ));
        }
        return new WindowFixture(
                frames,
                downFrameIndex / sampleRateHz,
                liftFrameIndex / sampleRateHz
        );
    }

    private static final class WindowFixture {
        final List<AispectModels.ImpactFrame> frames;
        final double downTimestamp;
        final double liftTimestamp;

        WindowFixture(List<AispectModels.ImpactFrame> frames, double downTimestamp, double liftTimestamp) {
            this.frames = frames;
            this.downTimestamp = downTimestamp;
            this.liftTimestamp = liftTimestamp;
        }
    }
}
