package com.zhifaios.eyes.aispect;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public final class AispectDownAnchorWindowGateTest {
    @Test
    public void publishesOnlyOnceAfterACompleteWindow() {
        AispectDownAnchorWindowGate gate = new AispectDownAnchorWindowGate();
        gate.begin();

        assertFalse(gate.tryPublish(incompleteWindow()));
        assertTrue(gate.tryPublish(completeWindow()));
        assertFalse(gate.tryPublish(completeWindow()));
    }

    @Test
    public void resetAllowsTheNextTouchToPublish() {
        AispectDownAnchorWindowGate gate = new AispectDownAnchorWindowGate();
        gate.begin();
        assertTrue(gate.tryPublish(completeWindow()));

        gate.reset();
        gate.begin();
        assertTrue(gate.tryPublish(completeWindow()));
    }

    private static AispectSignalWindowBuilder.Window completeWindow() {
        AispectSignalWindowBuilder builder = new AispectSignalWindowBuilder();
        builder.config().preFrames = 5;
        builder.config().postFrames = 9;
        return builder.build(
                frames(30),
                0.10,
                100
        );
    }

    private static AispectSignalWindowBuilder.Window incompleteWindow() {
        AispectSignalWindowBuilder builder = new AispectSignalWindowBuilder();
        builder.config().preFrames = 5;
        builder.config().postFrames = 9;
        return builder.build(
                frames(15),
                0.10,
                100
        );
    }

    private static java.util.List<AispectModels.ImpactFrame> frames(int count) {
        java.util.ArrayList<AispectModels.ImpactFrame> frames = new java.util.ArrayList<>();
        for (int index = 0; index < count; index++) {
            frames.add(new AispectModels.ImpactFrame(
                    0,
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
                    (index / 100.0) - 0.10,
                    (index / 100.0) - 0.10,
                    index / 100.0
            ));
        }
        return frames;
    }
}
