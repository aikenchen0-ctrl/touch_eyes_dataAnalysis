package com.zhifaios.eyes.aispect;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public final class AispectCausalPressFeatureBuilderTest {
    @Test
    public void touchFrameJsonPreservesViewDimensions() throws Exception {
        AispectTouchFrame frame = touchFrame(0L, 100L, 0.25f, 0.5f, 300, 400, 1f, 0.5f, 50f, 25f, 0f, 2, true);

        JSONObject json = frame.toJson(0);

        assertEquals(300, json.getInt("viewWidthPx"));
        assertEquals(400, json.getInt("viewHeightPx"));
    }

    @Test
    public void builderMatchesSharedFixtureWithReceiptTimeZeroOrderHold() throws Exception {
        JSONObject fixture = fixture();
        JSONArray sourceFrames = fixture.getJSONArray("frames");
        AispectModels.ImpactFrame[] frames = new AispectModels.ImpactFrame[sourceFrames.length()];
        for (int index = 0; index < sourceFrames.length(); index++) {
            frames[index] = impactFrame(sourceFrames.getJSONObject(index));
        }
        List<AispectTouchFrame> touches = new ArrayList<>();
        JSONArray sourceTouches = fixture.getJSONArray("touchFrames");
        for (int index = 0; index < sourceTouches.length(); index++) {
            touches.add(touchFrame(sourceTouches.getJSONObject(index)));
        }

        double[][] actual = AispectCausalPressFeatureBuilder.build(
                frames,
                0,
                new int[]{0, 1},
                touches,
                "causal_imu11_touch9_v1"
        );
        JSONArray expected = fixture.getJSONObject("expected").getJSONArray("causal_imu11_touch9_v1");

        assertEquals(expected.length(), actual.length);
        for (int index = 0; index < expected.length(); index++) {
            assertArrayEquals(expectedRow(expected.getJSONArray(index)), actual[index], 1e-6);
        }
    }

    @Test
    public void contractsHaveFixedCountsAndNoLegacyFeatureNames() {
        assertEquals(12, AispectCausalPressFeatureBuilder.featureNames("causal_imu12_v1").length);
        assertEquals(20, AispectCausalPressFeatureBuilder.featureNames("causal_imu11_touch9_v1").length);
        for (String name : AispectCausalPressFeatureBuilder.featureNames("causal_imu11_touch9_v1")) {
            String lower = name.toLowerCase();
            assertFalse(lower.contains("time"));
            assertFalse(lower.contains("contactpatch"));
            assertFalse(lower.contains("touchnormalization"));
            assertFalse(lower.contains("gaussian"));
            assertFalse(lower.contains("covariance"));
            assertFalse(lower.contains("confidence"));
        }
        assertTrue(AispectCausalPressFeatureBuilder.isCausalFeatureContract("causal_imu12_v1"));
        assertTrue(AispectCausalPressFeatureBuilder.isCausalFeatureContract("causal_imu11_touch9_v1"));
        assertEquals(11, AispectCausalPressFeatureBuilder.featureNames("causal_time_grid_imu11_v2").length);
        assertEquals(20, AispectCausalPressFeatureBuilder.featureNames("causal_time_grid_imu11_touch9_v2").length);
        assertTrue(AispectCausalPressFeatureBuilder.isTimeGridFeatureContract("causal_time_grid_imu11_v2"));
        assertTrue(AispectCausalPressFeatureBuilder.isCausalFeatureContract("causal_time_grid_imu11_touch9_v2"));
    }

    @Test
    public void classifierRecognizesOnlyCausalPressFeatureContracts() {
        assertTrue(AispectImpactCNNClassifier.usesCausalPressFeatureContract("causal_imu12_v1"));
        assertFalse(AispectImpactCNNClassifier.usesCausalPressFeatureContract("causal_imu11_v1"));
        assertTrue(AispectImpactCNNClassifier.usesLegacyFeatureContract("old13_raw_radius"));
        assertTrue(AispectImpactCNNClassifier.usesLegacyFeatureContract(""));
        assertTrue(AispectImpactCNNClassifier.usesCausalPressFeatureContract("causal_imu11_touch9_v1"));
        assertTrue(AispectImpactCNNClassifier.usesCausalPressFeatureContract("causal_time_grid_imu11_touch9_v2"));
        assertFalse(AispectImpactCNNClassifier.usesLegacyFeatureContract("causal_time_grid_imu11_touch9_v2"));
        assertFalse(AispectImpactCNNClassifier.usesCausalPressFeatureContract("old13_raw_radius"));
    }

    @Test
    public void builderRejectsMissingTouchFrameSourceButAllowsEmptyHistory() throws Exception {
        JSONObject fixture = fixture();
        JSONArray sourceFrames = fixture.getJSONArray("frames");
        AispectModels.ImpactFrame[] frames = new AispectModels.ImpactFrame[sourceFrames.length()];
        for (int index = 0; index < sourceFrames.length(); index++) {
            frames[index] = impactFrame(sourceFrames.getJSONObject(index));
        }

        assertTrue(AispectCausalPressFeatureBuilder.build(
                frames, 0, new int[]{0}, new ArrayList<>(), "causal_imu12_v1") != null);
        assertTrue(AispectCausalPressFeatureBuilder.build(
                frames, 0, new int[]{0}, null, "causal_imu12_v1") == null);
    }

    @Test
    public void builderDoesNotUseTouchEventThatIsPhysicallyAfterImuSample() throws Exception {
        AispectModels.ImpactFrame imu = new AispectModels.ImpactFrame(
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                0,
                0,
                0,
                0,
                0,
                250L,
                350L
        );
        List<AispectTouchFrame> touches = new ArrayList<>();
        touches.add(AispectTouchFrame.forTesting(90L, 100L, 0.25f, 0.5f, 300, 400, 1f, 0.5f, 50f, 25f, 0f, 1, true));
        touches.add(AispectTouchFrame.forTesting(290L, 300L, 0.8f, 0.1f, 300, 400, 1f, 0.5f, 50f, 25f, 0f, 1, true));

        double[][] actual = AispectCausalPressFeatureBuilder.build(
                new AispectModels.ImpactFrame[]{imu},
                0,
                new int[]{0},
                touches,
                "causal_imu12_v1"
        );

        assertEquals(0.25, actual[0][0], 1e-6);
        assertEquals(0.50, actual[0][1], 1e-6);
    }

    private static JSONObject fixture() throws Exception {
        InputStream input = AispectCausalPressFeatureBuilderTest.class.getClassLoader().getResourceAsStream("causal_press_features_v1_fixture.json");
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
        return new JSONObject(new String(output.toByteArray(), StandardCharsets.UTF_8));
    }

    private static double[] expectedRow(JSONArray row) throws Exception {
        double[] output = new double[row.length()];
        for (int index = 0; index < row.length(); index++) {
            output[index] = row.getDouble(index);
        }
        return output;
    }

    private static AispectModels.ImpactFrame impactFrame(JSONObject frame) throws Exception {
        return new AispectModels.ImpactFrame(
                0,
                frame.getDouble("x"),
                frame.getDouble("y"),
                frame.getDouble("z"),
                frame.getDouble("rotationRateX"),
                frame.getDouble("rotationRateY"),
                frame.getDouble("rotationRateZ"),
                frame.getDouble("gravityX"),
                frame.getDouble("gravityY"),
                frame.getDouble("gravityZ"),
                0,
                0,
                0,
                0,
                0,
                frame.optLong("sensorTimestampElapsedRealtimeNanos", 0L),
                frame.getLong("receivedElapsedRealtimeNanos")
        );
    }

    private static AispectTouchFrame touchFrame(JSONObject frame) throws Exception {
        JSONObject availability = frame.optJSONObject("axisAvailability");
        boolean hasOrientation = availability != null && availability.optBoolean("hasOrientationRange");
            return touchFrame(
                frame.optLong("eventElapsedRealtimeNanos", 0L),
                frame.getLong("receivedElapsedRealtimeNanos"),
                (float) frame.getDouble("x_norm"),
                (float) frame.getDouble("y_norm"),
                frame.getInt("viewWidthPx"),
                frame.getInt("viewHeightPx"),
                (float) frame.getDouble("pressure"),
                (float) frame.getDouble("size"),
                (float) frame.getDouble("touchMajor"),
                (float) frame.getDouble("touchMinor"),
                (float) frame.getDouble("orientation"),
                frame.getInt("pointerCount"),
                hasOrientation
        );
    }

    private static AispectTouchFrame touchFrame(
            long eventElapsedRealtimeNanos,
            long receivedElapsedRealtimeNanos,
            float xNorm,
            float yNorm,
            int width,
            int height,
            float pressure,
            float size,
            float touchMajor,
            float touchMinor,
            float orientation,
            int pointerCount,
            boolean hasOrientation
    ) {
        return AispectTouchFrame.forTesting(
                eventElapsedRealtimeNanos,
                receivedElapsedRealtimeNanos,
                xNorm * width,
                yNorm * height,
                width,
                height,
                pressure,
                size,
                touchMajor,
                touchMinor,
                orientation,
                pointerCount,
                hasOrientation
        );
    }
}
