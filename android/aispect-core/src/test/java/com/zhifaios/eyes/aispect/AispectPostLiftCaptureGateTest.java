package com.zhifaios.eyes.aispect;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public final class AispectPostLiftCaptureGateTest {
    @Test
    public void waitsWhileOnlyPostLiftCoverageIsMissing() {
        AispectSignalWindowBuilder.Window window = window(76, 40, 60, 250.0);

        assertTrue(window.preDownComplete);
        assertFalse(window.postLiftComplete);
        assertTrue(AispectPostLiftCaptureGate.shouldWait(window, 100L, 500L));
    }

    @Test
    public void stopsWaitingAtTimeout() {
        AispectSignalWindowBuilder.Window window = window(76, 40, 60, 250.0);

        assertFalse(AispectPostLiftCaptureGate.shouldWait(window, 500L, 500L));
    }

    @Test
    public void doesNotWaitWhenMissingPreDownCoverageCannotBeRecovered() {
        AispectSignalWindowBuilder.Window window = window(100, 15, 45, 250.0);

        assertFalse(window.preDownComplete);
        assertFalse(AispectPostLiftCaptureGate.shouldWait(window, 100L, 500L));
    }

    @Test
    public void doesNotWaitWhenBothMarginsAreComplete() {
        AispectSignalWindowBuilder.Window window = window(220, 100, 130, 250.0);

        assertTrue(window.windowComplete);
        assertFalse(AispectPostLiftCaptureGate.shouldWait(window, 100L, 500L));
    }

    private static AispectSignalWindowBuilder.Window window(
            int frameCount,
            int downFrameIndex,
            int liftFrameIndex,
            double sampleRateHz
    ) {
        List<AispectModels.ImpactFrame> frames = new ArrayList<>();
        for (int index = 0; index < frameCount; index++) {
            double timestamp = index / sampleRateHz;
            frames.add(new AispectModels.ImpactFrame(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    timestamp - (downFrameIndex / sampleRateHz),
                    timestamp - (liftFrameIndex / sampleRateHz),
                    timestamp,
                    index,
                    index
            ));
        }
        return new AispectSignalWindowBuilder().buildDualAnchor(
                frames,
                downFrameIndex / sampleRateHz,
                liftFrameIndex / sampleRateHz,
                sampleRateHz
        );
    }
}
