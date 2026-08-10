package com.zhifaios.eyes.aispect;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;


import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public final class DownAnchorCaptureContractTest {
    @Test
    public void defaultDownAnchorWindowUsesMinusFiveThroughPlusNine() {
        AispectSignalWindowBuilder builder = new AispectSignalWindowBuilder();
        AispectSignalWindowBuilder.Window window = builder.buildDownAnchor(
                frames(40),
                0.20,
                100.0
        );

        assertEquals(-5, window.firstFrameIndex);
        assertEquals(9, window.lastFrameIndex);
        assertEquals(15, window.frames.length);
        assertTrue(window.windowComplete);
    }

    @Test
    public void downAnchorWindowDoesNotPublishWhenPlusNineIsMissing() {
        AispectSignalWindowBuilder builder = new AispectSignalWindowBuilder();
        AispectSignalWindowBuilder.Window window = builder.buildDownAnchor(
                frames(25),
                0.20,
                100.0
        );

        assertFalse(window.windowComplete);
    }

    private static List<AispectModels.ImpactFrame> frames(int count) {
        ArrayList<AispectModels.ImpactFrame> frames = new ArrayList<>();
        for (int index = 0; index < count; index++) {
            double timestamp = index / 100.0;
            frames.add(new AispectModels.ImpactFrame(
                    0.0,
                    0.1,
                    0.2,
                    0.3,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.5,
                    0.5,
                    timestamp - 0.20,
                    timestamp - 0.20,
                    timestamp
            ));
        }
        return frames;
    }
}
