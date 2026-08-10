package com.zhifaios.eyes.aispect;

import org.junit.Test;

import java.util.ArrayList;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

public final class RawFrameSchemaTest {
    @Test
    public void newRawFrameSchemaDoesNotDeclareDerivedDelta() {
        for (String field : AispectModels.RAW_FRAME_FIELDS) {
            assertFalse("delta".equals(field));
            assertFalse("maxAbsDelta".equals(field));
        }
        assertTrue(AispectModels.containsRawFrameField("x"));
        assertTrue(AispectModels.containsRawFrameField("timeSinceTouchDown"));
        assertTrue(AispectModels.containsRawFrameField("sensorTimestampElapsedRealtimeNanos"));
        assertTrue(AispectModels.containsRawFrameField("receivedElapsedRealtimeNanos"));
    }

    @Test
    public void rawFrameJsonPreservesSensorAndReceiptTimestamps() throws Exception {
        AispectModels.ImpactFrame frame = new AispectModels.ImpactFrame(
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
                0.25,
                0.75,
                0.01,
                -0.02,
                10.5,
                10_500_000_000L,
                10_500_200_000L
        );

        assertEquals(10_500_000_000L, frame.toJson(4).getLong("sensorTimestampElapsedRealtimeNanos"));
        assertEquals(10_500_200_000L, frame.toJson(4).getLong("receivedElapsedRealtimeNanos"));
    }

    @Test
    public void crossDeviceMetadataDeclaresAndroidUnitsAndTouchDimensions() throws Exception {
        AispectImpactSignalCollector.Snapshot snapshot = new AispectImpactSignalCollector.Snapshot(
                new ArrayList<AispectModels.ImpactFrame>(),
                200.0,
                "linear",
                "linear_acceleration",
                true,
                true
        );

        assertEquals("android", AispectDataLogger.sensorMetadataToJson(snapshot).getString("platform"));
        assertEquals("linear_acceleration", AispectDataLogger.sensorMetadataToJson(snapshot).getString("sensorSource"));
        assertEquals("m/s^2", AispectDataLogger.sensorMetadataToJson(snapshot).getString("accelerationUnit"));
        assertEquals("pixels", AispectDataLogger.touchMetadataToJson().getString("touchUnit"));
    }

    @Test
    public void accelerometerFallbackSubtractsGravityBeforeCanonicalFeatureUse() {
        assertArrayEquals(
                new double[]{9.0, 0.0, -1.0},
                AispectImpactSignalCollector.linearAcceleration(
                        new double[]{10.0, 2.0, 8.8},
                        new double[]{1.0, 2.0, 9.8},
                        "accelerometer_with_gravity"
                ),
                1e-6
        );
        assertNull(AispectImpactSignalCollector.linearAcceleration(
                new double[]{10.0, 2.0, 8.8},
                null,
                "accelerometer_with_gravity"
        ));
        assertTrue(AispectImpactSignalCollector.isGravityTimestampAligned(100_000_000L, 50_000_000L));
        assertFalse(AispectImpactSignalCollector.isGravityTimestampAligned(100_000_000L, 49_999_999L));
    }
}
