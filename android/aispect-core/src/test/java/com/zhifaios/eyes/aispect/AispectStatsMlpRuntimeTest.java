package com.zhifaios.eyes.aispect;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

public final class AispectStatsMlpRuntimeTest {
    @Test
    public void runtimeRecognizesTheExportedArchitecture() {
        assertTrue(AispectStatsMlpRuntime.isSupportedArchitecture("stats_mlp_strength_gate_v1"));
        assertTrue(AispectStatsMlpRuntime.isSupportedArchitecture("stats_mlp_v1"));
        assertFalse(AispectStatsMlpRuntime.isSupportedArchitecture("time_grid_groupnorm_cnn_v1"));
    }

    @Test
    public void validContractLoadsAndZeroWeightsProduceUniformProbabilities() throws JSONException {
        AispectStatsMlpRuntime runtime = AispectStatsMlpRuntime.load(
                "stats_mlp_test",
                weights(),
                scaler()
        );

        assertNotNull(runtime);
        assertArrayEquals(AispectStatsMlpRuntime.GATE_LABEL_ORDER, runtime.labelOrder());
        assertTrue(runtime.isStrengthGate());
        assertEquals(
                AispectStatsMlpRuntime.SOURCE_FEATURE_CONTRACT,
                scaler().getString("sourceFeatureContract")
        );
        double[] probabilities = runtime.predict(new double[9][10]);

        assertNotNull(probabilities);
        assertEquals(2, probabilities.length);
        for (double probability : probabilities) {
            assertEquals(0.5, probability, 1e-12);
        }

        AispectStatsMlpRuntime.HeadProbabilities heads =
                runtime.predictHeads(new double[9][10]);
        assertNotNull(heads);
        assertArrayEquals(new double[]{0.5, 0.5}, heads.finger, 1e-12);
        assertArrayEquals(new double[]{0.5, 0.5}, heads.strength, 1e-12);
    }

    @Test
    public void legacyFourClassContractStillLoads() throws JSONException {
        JSONObject legacy = scaler(
                AispectStatsMlpRuntime.LEGACY_FEATURE_CONTRACT,
                AispectStatsMlpRuntime.LEGACY_SOURCE_FEATURE_NAMES
        )
                .put("runtimeArchitecture", AispectStatsMlpRuntime.LEGACY_ARCHITECTURE)
                .put("classCount", 4)
                .put("labelOrder", new JSONArray()
                        .put("thumb_light")
                        .put("thumb_heavy")
                        .put("index_light")
                        .put("index_heavy"));
        legacy.remove("primaryTask");
        legacy.remove("gateCount");
        legacy.remove("gateLabelOrder");
        legacy.remove("negativeClassLabels");
        legacy.remove("positiveClassLabels");
        legacy.remove("diagnosticClassCount");
        legacy.remove("diagnosticClassLabelOrder");

        AispectStatsMlpRuntime runtime = AispectStatsMlpRuntime.load(
                "legacy",
                weights(),
                legacy
        );

        assertNotNull(runtime);
        assertFalse(runtime.isStrengthGate());
        assertArrayEquals(AispectStatsMlpRuntime.LEGACY_LABEL_ORDER, runtime.labelOrder());
        assertEquals(4, runtime.predict(new double[9][10]).length);
    }

    @Test
    public void runtimeRejectsWrongSourceOrderAndStatsOrder() throws JSONException {
        JSONObject wrongSource = scaler();
        JSONArray sourceIndices = wrongSource.getJSONArray("sourceFeatureIndices");
        sourceIndices.put(0, 1);
        assertNull(AispectStatsMlpRuntime.load("wrong_source", weights(), wrongSource));

        JSONObject wrongStats = scaler();
        JSONArray statsNames = wrongStats.getJSONArray("statsFeatureNames");
        statsNames.put(0, "wrong.order");
        assertNull(AispectStatsMlpRuntime.load("wrong_stats", weights(), wrongStats));

        JSONObject mismatchedSourceContract = scaler()
                .put(
                        "sourceFeatureContract",
                        "causal_time_grid_imu11_touch9_mask6_v4"
                );
        assertNull(AispectStatsMlpRuntime.load(
                "mismatched_source_contract",
                weights(),
                mismatchedSourceContract
        ));
    }

    @Test
    public void runtimeRejectsWrongWeightShape() throws JSONException {
        JSONObject weights = weights();
        JSONObject firstWeight = weights.getJSONObject("shared.0.weight");
        firstWeight.put("shape", new JSONArray().put(24).put(60).put(1));

        assertNull(AispectStatsMlpRuntime.load("wrong_shape", weights, scaler()));
    }

    @Test
    public void runtimeRejectsDeployableFlagOrUnexpectedTensor() throws JSONException {
        JSONObject deployable = scaler().put("deploymentAllowed", true);
        assertNull(AispectStatsMlpRuntime.load("deployable", weights(), deployable));

        JSONObject extraWeights = weights().put("unexpected.weight", tensor(new int[]{1}));
        assertNull(AispectStatsMlpRuntime.load("extra_tensor", extraWeights, scaler()));
    }

    @Test
    public void causalSizeContractRequiresCompletedTouchHistoryScope() throws JSONException {
        JSONObject valid = scaler(
                AispectStatsMlpRuntime.V4_FEATURE_CONTRACT,
                AispectStatsMlpRuntime.V4_SOURCE_FEATURE_NAMES
        );
        assertNotNull(AispectStatsMlpRuntime.load(
                "causal_size_valid",
                weights(),
                valid
        ));

        JSONObject invalid = scaler(
                AispectStatsMlpRuntime.V4_FEATURE_CONTRACT,
                AispectStatsMlpRuntime.V4_SOURCE_FEATURE_NAMES
        );
        invalid.getJSONObject("sizeBaseline").remove("historyScope");
        assertNull(AispectStatsMlpRuntime.load(
                "causal_size_invalid",
                weights(),
                invalid
        ));

        JSONObject wrongScope = scaler(
                AispectStatsMlpRuntime.V4_FEATURE_CONTRACT,
                AispectStatsMlpRuntime.V4_SOURCE_FEATURE_NAMES
        );
        wrongScope.getJSONObject("sizeBaseline").put(
                "deviceScope",
                "session"
        );
        assertNull(AispectStatsMlpRuntime.load(
                "causal_size_wrong_scope",
                weights(),
                wrongScope
        ));
    }

    @Test
    public void causalSizeStatsNamesExposeProxyAndAxisSemantics() {
        String[] names = AispectStatsMlpRuntime.statsFeatureNames(
                AispectStatsMlpRuntime.V4_FEATURE_NAMES
        );

        assertEquals(60, names.length);
        assertEquals(
                "log_causal_size_ratio.mean",
                names[40]
        );
        assertEquals(
                "log_axis_ratio.mean",
                names[45]
        );
    }

    @Test
    public void causalSizeContractRejectsLegacyArchitecture() throws JSONException {
        JSONObject invalid = scaler(
                AispectStatsMlpRuntime.V4_FEATURE_CONTRACT,
                AispectStatsMlpRuntime.V4_SOURCE_FEATURE_NAMES
        ).put(
                "runtimeArchitecture",
                AispectStatsMlpRuntime.LEGACY_ARCHITECTURE
        );

        assertNull(AispectStatsMlpRuntime.load(
                "invalid_v4_architecture",
                weights(),
                invalid
        ));
    }

    @Test
    public void causalSizeContractRejectsFrozenCountDrift() throws JSONException {
        String[] fields = new String[]{
                "frameCount",
                "sequenceChannelCount",
                "statsFeatureCount",
                "inputCount"
        };
        for (String field : fields) {
            JSONObject invalid = scaler(
                    AispectStatsMlpRuntime.V4_FEATURE_CONTRACT,
                    AispectStatsMlpRuntime.V4_SOURCE_FEATURE_NAMES
            ).put(field, 1);
            assertNull(
                    field,
                    AispectStatsMlpRuntime.load(
                            "invalid_count_" + field,
                            weights(),
                            invalid
                    )
            );
        }
    }

    @Test
    public void causalSizeContractRejectsNonPositiveSequenceScale()
            throws JSONException {
        JSONObject invalid = scaler(
                AispectStatsMlpRuntime.V4_FEATURE_CONTRACT,
                AispectStatsMlpRuntime.V4_SOURCE_FEATURE_NAMES
        );
        invalid.getJSONArray("sequenceChannels")
                .getJSONObject(8)
                .put("scale", 0.0);

        assertNull(AispectStatsMlpRuntime.load(
                "invalid_v4_sequence_scale",
                weights(),
                invalid
        ));
    }

    @Test
    public void causalSizeContractRejectsMissingTouchStatistics()
            throws JSONException {
        JSONObject invalid = scaler(
                AispectStatsMlpRuntime.V4_FEATURE_CONTRACT,
                AispectStatsMlpRuntime.V4_SOURCE_FEATURE_NAMES
        );
        invalid.getJSONObject("sizeBaseline").remove("touchStatistics");

        assertNull(AispectStatsMlpRuntime.load(
                "invalid_v4_touch_statistics",
                weights(),
                invalid
        ));
    }

    @Test
    public void statsFeaturesHaveFrozenCountAndPhysicalOrder() {
        double[][] values = new double[9][10];
        for (int frame = 0; frame < values.length; frame++) {
            for (int channel = 0; channel < values[frame].length; channel++) {
                values[frame][channel] = frame * 0.25 + channel;
            }
        }

        double[] stats = AispectStatsMlpRuntime.statsFeatures(values);

        assertEquals(60, stats.length);
        assertEquals(AispectStatsMlpRuntime.STATS_FEATURE_COUNT, AispectStatsMlpRuntime.statsFeatureNames().length);
        assertEquals("x_norm.mean", AispectStatsMlpRuntime.statsFeatureNames()[0]);
        assertEquals("rotationRate.jerk.max", AispectStatsMlpRuntime.statsFeatureNames()[59]);
    }

    @Test
    public void statsFeaturesInterleaveStatisticsWithinEachChannel() {
        double[][] values = new double[9][10];
        for (int frame = 0; frame < values.length; frame++) {
            values[frame][0] = frame;
            values[frame][1] = frame + 100.0;
        }

        double[] stats = AispectStatsMlpRuntime.statsFeatures(values);

        assertEquals(4.0, stats[0], 1e-12);
        assertEquals(Math.sqrt(20.0 / 3.0), stats[1], 1e-12);
        assertEquals(104.0, stats[5], 1e-12);
    }

    private static JSONObject scaler() throws JSONException {
        return scaler(
                AispectStatsMlpRuntime.FEATURE_CONTRACT,
                AispectStatsMlpRuntime.SOURCE_FEATURE_NAMES
        );
    }

    private static JSONObject scaler(
            String featureContract,
            String[] sourceFeatureNameValues
    ) throws JSONException {
        boolean causalSizeProxy = AispectStatsMlpRuntime.V4_FEATURE_CONTRACT.equals(
                featureContract
        );
        String[] physicalFeatureNames = causalSizeProxy
                ? AispectStatsMlpRuntime.V4_FEATURE_NAMES
                : AispectStatsMlpRuntime.FEATURE_NAMES;
        JSONArray sourceNames = new JSONArray();
        for (String name : sourceFeatureNameValues) {
            sourceNames.put(name);
        }
        JSONArray featureNames = new JSONArray();
        JSONArray channels = new JSONArray();
        for (String name : physicalFeatureNames) {
            featureNames.put(name);
            channels.put(new JSONObject()
                    .put("name", name)
                    .put("center", 0.0)
                    .put("scale", 1.0)
                    .put("clipMin", -8.0)
                    .put("clipMax", 8.0));
        }
        JSONArray sourceIndices = new JSONArray();
        for (int index : AispectStatsMlpRuntime.SOURCE_FEATURE_INDICES) {
            sourceIndices.put(index);
        }
        JSONArray statsNames = new JSONArray();
        for (String name : AispectStatsMlpRuntime.statsFeatureNames(
                physicalFeatureNames
        )) {
            statsNames.put(name);
        }
        JSONObject result = new JSONObject()
                .put("runtimeArchitecture", AispectStatsMlpRuntime.ARCHITECTURE)
                .put("featureContract", featureContract)
                .put(
                        "sourceFeatureContract",
                        causalSizeProxy
                                ? AispectStatsMlpRuntime.V4_SOURCE_FEATURE_CONTRACT
                                : AispectStatsMlpRuntime.SOURCE_FEATURE_CONTRACT
                )
                .put("sourceFeatureNames", sourceNames)
                .put("featureNames", featureNames)
                .put("sourceFeatureIndices", sourceIndices)
                .put("statsFeatureNames", statsNames)
                .put("matrixShape", new JSONArray().put(9).put(10))
                .put("frameCount", 9)
                .put("sequenceChannelCount", 10)
                .put("statsFeatureCount", 60)
                .put("primaryTask", AispectStatsMlpRuntime.PRIMARY_TASK)
                .put("gateCount", 2)
                .put("gateLabelOrder", new JSONArray().put("light").put("heavy"))
                .put("negativeClassLabels", new JSONArray()
                        .put("thumb_light")
                        .put("index_light"))
                .put("positiveClassLabels", new JSONArray()
                        .put("thumb_heavy")
                        .put("index_heavy"))
                .put("classCount", 2)
                .put("inputCount", 60)
                .put("hiddenCount", 24)
                .put("dropoutAtInference", false)
                .put("evaluationScope", "all_data_fit_only")
                .put("deploymentAllowed", false)
                .put("labelOrder", new JSONArray().put("light").put("heavy"))
                .put("diagnosticClassCount", 4)
                .put("diagnosticClassLabelOrder", new JSONArray()
                        .put("thumb_light")
                        .put("thumb_heavy")
                        .put("index_light")
                        .put("index_heavy"))
                .put("fingerLabelOrder", new JSONArray().put("thumb").put("index"))
                .put("strengthLabelOrder", new JSONArray().put("light").put("heavy"))
                .put("sequenceChannels", channels)
                .put("statsScaler", new JSONObject()
                        .put("center", new JSONArray(new double[60]))
                        .put("scale", ones(60))
                        .put("clipMin", -8.0)
                        .put("clipMax", 8.0));
        if (causalSizeProxy) {
            result
                    .put("sizeSemantics", "contact_area_proxy")
                    .put("forceMeasurement", false)
                    .put("missingValuePolicy", "nan_excluded_from_stats")
                    .put(
                            "sizeBaseline",
                            new JSONObject()
                                    .put("causal", true)
                                    .put("currentTouchExcluded", true)
                                    .put(
                                            "historyScope",
                                            "completed_touches_only"
                                    )
                                    .put(
                                            "deviceScope",
                                            "physical_device"
                                    )
                                    .put(
                                            "normalization",
                                            "log_causal_device_median"
                                    )
                                    .put("aggregation", "median")
                                    .put(
                                            "withinTouchAggregation",
                                            "median"
                                    )
                                    .put(
                                            "minimumTouches",
                                            AispectCausalSizeBaseline.MINIMUM_TOUCHES
                                    )
                                    .put(
                                            "maximumTouches",
                                            AispectCausalSizeBaseline.MAXIMUM_TOUCHES
                                    )
                                    .put(
                                            "logRatioClip",
                                            AispectCausalSizeBaseline.LOG_RATIO_CLIP
                                    )
                                    .put(
                                            "touchStatistics",
                                            new JSONArray()
                                                    .put("mean")
                                                    .put("std")
                                                    .put("min")
                                                    .put("max")
                                                    .put("last_delta")
                                    )
                    );
        }
        return result;
    }

    private static JSONObject weights() throws JSONException {
        return new JSONObject()
                .put("shared.0.weight", tensor(new int[]{24, 60}))
                .put("shared.0.bias", tensor(new int[]{24}))
                .put("shared.3.weight", tensor(new int[]{24, 24}))
                .put("shared.3.bias", tensor(new int[]{24}))
                .put("finger.weight", tensor(new int[]{2, 24}))
                .put("finger.bias", tensor(new int[]{2}))
                .put("strength.weight", tensor(new int[]{2, 24}))
                .put("strength.bias", tensor(new int[]{2}));
    }

    private static JSONObject tensor(int[] shape) throws JSONException {
        JSONArray shapeJson = new JSONArray();
        int count = 1;
        for (int dimension : shape) {
            shapeJson.put(dimension);
            count *= dimension;
        }
        return new JSONObject()
                .put("shape", shapeJson)
                .put("values", new JSONArray(new double[count]));
    }

    private static JSONArray ones(int count) throws JSONException {
        JSONArray values = new JSONArray();
        for (int index = 0; index < count; index++) {
            values.put(1.0);
        }
        return values;
    }
}
