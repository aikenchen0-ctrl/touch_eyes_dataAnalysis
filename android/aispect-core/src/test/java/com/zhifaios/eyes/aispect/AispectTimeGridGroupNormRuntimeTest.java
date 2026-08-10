package com.zhifaios.eyes.aispect;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.junit.Test;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public final class AispectTimeGridGroupNormRuntimeTest {
    @Test
    public void runtimeRecognizesTheExportedArchitecture() {
        assertTrue(AispectTimeGridGroupNormRuntime.isSupportedArchitecture("time_grid_groupnorm_cnn_v1"));
    }

    @Test
    public void groupNormNormalizesEachChannelGroupAcrossTime() {
        double[][] values = new double[][]{
                new double[]{1.0, 3.0},
                new double[]{5.0, 7.0},
                new double[]{2.0, 4.0},
                new double[]{6.0, 8.0}
        };
        double[][] normalized = AispectTimeGridGroupNormRuntime.groupNorm(
                values,
                new double[]{1.0, 1.0, 1.0, 1.0},
                new double[]{0.0, 0.0, 0.0, 0.0},
                2
        );

        assertEquals(-1.341639, normalized[0][0], 1e-5);
        assertEquals(1.341639, normalized[1][1], 1e-5);
        assertEquals(-1.341639, normalized[2][0], 1e-5);
        assertEquals(1.341639, normalized[3][1], 1e-5);
    }

    @Test
    public void geluUsesTheZeroCenteredActivation() {
        assertEquals(0.0, AispectTimeGridGroupNormRuntime.gelu(0.0), 1e-12);
        assertTrue(AispectTimeGridGroupNormRuntime.gelu(1.0) > 0.8);
        assertTrue(AispectTimeGridGroupNormRuntime.gelu(-1.0) < 0.0);
    }

    @Test
    public void noGravityRuntimeUsesRemappedTouchAvailabilityChannels() throws JSONException {
        String contract = "causal_time_grid_touch7_mask5_no_gravity_v1";
        String[] featureNames = AispectCausalPressFeatureBuilder.featureNames(contract);
        JSONObject scaler = scaler(contract, featureNames);
        JSONObject weights = weights(featureNames.length);
        AispectTimeGridGroupNormRuntime runtime = AispectTimeGridGroupNormRuntime.load(
                "no_gravity_test",
                weights,
                scaler
        );
        assertNotNull(runtime);

        double[][] matrix = new double[9][featureNames.length];
        int majorAvailableIndex = indexOf(featureNames, "major_available");
        for (double[] row : matrix) {
            row[majorAvailableIndex] = 1.0;
        }

        double[] probabilities = runtime.predict(matrix);

        assertNotNull(probabilities);
        assertEquals(4, probabilities.length);
    }

    private static JSONObject scaler(String contract, String[] featureNames) throws JSONException {
        JSONArray names = new JSONArray();
        JSONArray channels = new JSONArray();
        for (String name : featureNames) {
            names.put(name);
            channels.put(new JSONObject()
                    .put("name", name)
                    .put("strategy", "identity")
                    .put("requiresTouchAvailability", "major_norm".equals(name)));
        }
        return new JSONObject()
                .put("runtimeArchitecture", "time_grid_groupnorm_cnn_v1")
                .put("featureContract", contract)
                .put("featureNames", names)
                .put("labelOrder", new JSONArray()
                        .put("thumb_light")
                        .put("thumb_heavy")
                        .put("index_light")
                        .put("index_heavy"))
                .put("matrixShape", new JSONArray().put(9).put(featureNames.length))
                .put("channels", channels);
    }

    private static JSONObject weights(int inputChannels) throws JSONException {
        return new JSONObject()
                .put("conv1.weight", tensor(new int[]{32, inputChannels, 3}))
                .put("conv1.bias", tensor(new int[]{32}))
                .put("norm1.weight", tensor(new int[]{32}))
                .put("norm1.bias", tensor(new int[]{32}))
                .put("conv2.weight", tensor(new int[]{32, 32, 3}))
                .put("conv2.bias", tensor(new int[]{32}))
                .put("norm2.weight", tensor(new int[]{32}))
                .put("norm2.bias", tensor(new int[]{32}))
                .put("output.weight", tensor(new int[]{4, 32}))
                .put("output.bias", tensor(new int[]{4}));
    }

    private static JSONObject tensor(int[] shape) throws JSONException {
        JSONArray shapeJson = new JSONArray();
        JSONArray values = new JSONArray();
        int count = 1;
        for (int dimension : shape) {
            shapeJson.put(dimension);
            count *= dimension;
        }
        for (int index = 0; index < count; index++) {
            values.put(0.0);
        }
        return new JSONObject().put("shape", shapeJson).put("values", values);
    }

    private static int indexOf(String[] values, String target) {
        for (int index = 0; index < values.length; index++) {
            if (target.equals(values[index])) {
                return index;
            }
        }
        throw new IllegalArgumentException("missing_feature_name");
    }
}
