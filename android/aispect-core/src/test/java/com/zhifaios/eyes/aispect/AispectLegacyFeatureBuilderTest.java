package com.zhifaios.eyes.aispect;

import com.zhifaios.eyes.aispect.legacy.AispectLegacyFeatureBuilder;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

public final class AispectLegacyFeatureBuilderTest {
    @Test
    public void legacyBuilderPreservesHistoricalTimeFeatureOnlyForLegacyCaller() {
        AispectModels.ImpactFrame frame = new AispectModels.ImpactFrame(
                0.0,
                1.0,
                2.0,
                3.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.25,
                0.75,
                0.5,
                0.25,
                1.0
        );
        AispectSignalWindowBuilder.Window window = new AispectSignalWindowBuilder.Window(
                new AispectModels.ImpactFrame[]{frame},
                0,
                0.0,
                120.0,
                0.0
        );

        double[][] output = AispectLegacyFeatureBuilder.build(
                window,
                new int[]{0},
                2,
                new String[]{"time_since_anchor", "x_norm"},
                new double[]{0.0, 0.0},
                new double[]{0.5, 0.25},
                null,
                null,
                null
        );

        assertEquals(0.5, output[0][0], 1e-9);
        assertEquals(1.0, output[0][1], 1e-9);
    }

    @Test
    public void legacyBuilderRejectsMismatchedFeatureShape() {
        assertNull(AispectLegacyFeatureBuilder.build(
                null,
                new int[]{0},
                1,
                new String[]{"x"},
                new double[]{0.0, 0.0},
                new double[]{1.0},
                null,
                null,
                null
        ));
    }
}
