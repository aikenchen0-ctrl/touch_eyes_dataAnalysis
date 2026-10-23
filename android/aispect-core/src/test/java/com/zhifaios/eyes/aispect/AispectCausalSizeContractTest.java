package com.zhifaios.eyes.aispect;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

public final class AispectCausalSizeContractTest {
    @Test
    public void v6ProjectionAndStatisticsMatchSharedFixture() throws Exception {
        JSONObject fixture = fixture();
        double[][] source = matrix(fixture.getJSONArray("sourceMatrix"));
        AispectCausalSizeBaseline baseline = new AispectCausalSizeBaseline(
                "golden-device",
                numberList(fixture.getJSONArray("baselineHistory"))
        );

        double[][] projection = AispectCausalPressFeatureBuilder
                .projectStatsMlpV4Source(source, baseline);

        assertNotNull(projection);
        assertRows(
                matrix(fixture.getJSONArray("expectedProjection")),
                projection
        );
        assertArrayEquals(
                numbers(fixture.getJSONArray("expectedStats")),
                AispectStatsMlpRuntime.statsFeatures(projection, true),
                1e-6
        );
        assertArrayEquals(
                AispectStatsMlpRuntime.SOURCE_FEATURE_NAMES,
                strings(fixture.getJSONArray("sourceFeatureNames"))
        );
        assertArrayEquals(
                AispectStatsMlpRuntime.V4_FEATURE_NAMES,
                strings(fixture.getJSONArray("physicalFeatureNames"))
        );
        assertArrayEquals(
                AispectStatsMlpRuntime.statsFeatureNames(
                        AispectStatsMlpRuntime.V4_FEATURE_NAMES
                ),
                strings(fixture.getJSONArray("statsFeatureNames"))
        );
    }

    @Test
    public void v4StatsNamesExposeCausalSizeSemantics() {
        String[] names = AispectStatsMlpRuntime.statsFeatureNames(
                AispectStatsMlpRuntime.V4_FEATURE_NAMES
        );

        assertTrue(names[40].startsWith("log_causal_size_ratio."));
        assertTrue(names[45].startsWith("log_axis_ratio."));
    }

    @Test
    public void baselineWarmsUpAfterEightCompletedTouches() {
        AispectCausalSizeBaseline baseline = new AispectCausalSizeBaseline(
                "phone-a",
                Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        );

        assertTrue(Double.isNaN(baseline.currentMedian()));
        baseline.observeCompletedTouch(touch(800L, 8.0f, false));

        assertEquals(4.5, baseline.currentMedian(), 1e-12);
        assertEquals(8, baseline.completedTouchCount());
    }

    @Test
    public void invalidAndRepeatedTouchesDoNotAdvanceHistory() {
        AispectCausalSizeBaseline baseline = new AispectCausalSizeBaseline(
                "phone-a",
                Collections.emptyList()
        );

        baseline.observeCompletedTouch(touch(100L, 2.0f, true));
        baseline.observeCompletedTouch(touch(100L, 4.0f, true));
        baseline.observeCompletedTouch(touch(200L, Float.NaN, true));
        baseline.observeCompletedTouch(touch(300L, -1.0f, true));
        baseline.observeCompletedTouch(touch(400L, 0.0f, true));

        assertEquals(1, baseline.completedTouchCount());
    }

    @Test
    public void positiveSizeWithoutDeclaredRangeStillWarmsBaseline() {
        AispectCausalSizeBaseline baseline = new AispectCausalSizeBaseline(
                "phone-a",
                Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        );

        baseline.observeCompletedTouch(touch(800L, 8.0f, false));

        assertEquals(8, baseline.completedTouchCount());
        assertEquals(4.5, baseline.currentMedian(), 1e-12);
    }

    @Test
    public void deviceBaselinesStayIsolated() {
        AispectCausalSizeBaseline phoneA = new AispectCausalSizeBaseline(
                "phone-a",
                Collections.emptyList()
        );
        AispectCausalSizeBaseline phoneB = new AispectCausalSizeBaseline(
                "phone-b",
                Collections.emptyList()
        );
        for (int index = 0; index < 8; index++) {
            phoneA.observeCompletedTouch(touch(100L + index, 2.0f, true));
        }

        assertEquals(2.0, phoneA.currentMedian(), 1e-12);
        assertTrue(Double.isNaN(phoneB.currentMedian()));
    }

    @Test
    public void runtimeRejectsV4WithLegacyArchitecture() throws Exception {
        JSONObject scaler = new JSONObject()
                .put("runtimeArchitecture", AispectStatsMlpRuntime.LEGACY_ARCHITECTURE)
                .put("featureContract", AispectStatsMlpRuntime.V4_FEATURE_CONTRACT);

        assertNull(AispectStatsMlpRuntime.load(
                "invalid_v4_architecture",
                new JSONObject(),
                scaler
        ));
    }

    private static JSONObject fixture() throws Exception {
        InputStream input = AispectCausalSizeContractTest.class
                .getClassLoader()
                .getResourceAsStream("stats_mlp_v4_golden_fixture.json");
        if (input == null) {
            throw new IllegalStateException("fixture_not_found");
        }
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[1024];
        int count;
        while ((count = input.read(buffer)) >= 0) {
            output.write(buffer, 0, count);
        }
        input.close();
        return new JSONObject(
                new String(output.toByteArray(), StandardCharsets.UTF_8)
        );
    }

    private static String[] strings(JSONArray values) throws Exception {
        String[] output = new String[values.length()];
        for (int index = 0; index < values.length(); index++) {
            output[index] = values.getString(index);
        }
        return output;
    }

    private static double[] numbers(JSONArray values) throws Exception {
        double[] output = new double[values.length()];
        for (int index = 0; index < values.length(); index++) {
            output[index] = values.isNull(index)
                    ? Double.NaN
                    : values.getDouble(index);
        }
        return output;
    }

    private static List<Double> numberList(JSONArray values) throws Exception {
        return Arrays.stream(numbers(values))
                .boxed()
                .collect(Collectors.toList());
    }

    private static double[][] matrix(JSONArray rows) throws Exception {
        double[][] output = new double[rows.length()][];
        for (int row = 0; row < rows.length(); row++) {
            output[row] = numbers(rows.getJSONArray(row));
        }
        return output;
    }

    private static void assertRows(double[][] expected, double[][] actual) {
        assertEquals(expected.length, actual.length);
        for (int row = 0; row < expected.length; row++) {
            assertEquals(expected[row].length, actual[row].length);
            for (int column = 0; column < expected[row].length; column++) {
                if (Double.isNaN(expected[row][column])) {
                    assertTrue(Double.isNaN(actual[row][column]));
                } else {
                    assertEquals(
                            expected[row][column],
                            actual[row][column],
                            1e-6
                    );
                }
            }
        }
    }

    private static List<AispectTouchFrame> touch(
            long eventNanos,
            float size,
            boolean hasSizeRange
    ) {
        return Collections.singletonList(AispectTouchFrame.forTesting(
                eventNanos,
                eventNanos + 1_000L,
                100f,
                200f,
                1000,
                2000,
                0.5f,
                size,
                20f,
                10f,
                0f,
                1,
                false,
                hasSizeRange
        ));
    }
}
