package com.zhifaios.eyes.aispect;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

public final class AispectFirstTouchClassifierTest {
    @Test
    public void extractsTheSameTenFeaturesAsTraining() {
        double[] values = AispectFirstTouchClassifier.features(
                0.25,
                0.75,
                0.8,
                0.12,
                0.02,
                0.01,
                0.5
        );

        assertEquals(10, values.length);
        assertEquals(0.25, values[0], 1e-9);
        assertEquals(0.75, values[1], 1e-9);
        assertEquals(Math.log(2.0), values[6], 1e-9);
        assertEquals(Math.log(0.0002), values[7], 1e-9);
        assertEquals(Math.sin(0.5), values[8], 1e-9);
        assertEquals(Math.cos(0.5), values[9], 1e-9);
    }

    @Test
    public void runsReluMlpAndReturnsFourClassPrediction() {
        double[] center = new double[10];
        double[] scale = new double[]{1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        double[][] hiddenWeight = new double[][]{
                {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        };
        double[][] outputWeight = new double[][]{
                {1, 0},
                {0, 1},
                {-1, 0},
                {0, -1},
        };
        AispectFirstTouchClassifier classifier = new AispectFirstTouchClassifier(
                "first_touch_test",
                center,
                scale,
                hiddenWeight,
                new double[]{0, 0},
                outputWeight,
                new double[]{0, 0, 0, 0},
                new String[]{"thumb_light", "thumb_heavy", "index_light", "index_heavy"}
        );

        AispectModels.ImpactPrediction prediction = classifier.predictFeatures(
                new double[]{2, 1, 0, 0, 0, 0, 0, 0, 0, 0}
        );

        assertEquals("first_touch_test", classifier.modelId());
        assertEquals("first_touch_test", prediction.modelId);
        assertEquals("thumb_light", prediction.predictedLabel);
        assertEquals(4, prediction.probabilities.length);
    }

    @Test
    public void returnsNullWhenModelIsUnavailable() {
        AispectFirstTouchClassifier classifier = new AispectFirstTouchClassifier(
                "",
                new double[0],
                new double[0],
                new double[0][0],
                new double[0],
                new double[0][0],
                new double[0],
                new String[0]
        );

        assertEquals("", classifier.modelId());
        assertNull(classifier.predictFeatures(new double[10]));
    }
}
