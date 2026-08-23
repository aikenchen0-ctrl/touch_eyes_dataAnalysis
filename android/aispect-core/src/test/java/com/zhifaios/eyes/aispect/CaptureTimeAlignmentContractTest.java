package com.zhifaios.eyes.aispect;

import org.json.JSONObject;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public final class CaptureTimeAlignmentContractTest {
    @Test
    public void schemaSixDeclaresCausalReceiptTimeAlignment() throws Exception {
        JSONObject json = AispectDataLogger.timeAlignmentToJson(10L, 20L, 30L, 40L);

        assertEquals(6, AispectDataLogger.RAW_SCHEMA_VERSION);
        assertEquals(2, json.getInt("schemaVersion"));
        assertEquals("elapsedRealtimeNanos", json.getString("clock"));
        assertEquals(
                "touch_event_time_millis_from_down_anchor_v2",
                json.getString("touchTimestampConversion")
        );
        assertEquals(
                "touchReceivedElapsedRealtimeNanos<=imuReceivedElapsedRealtimeNanos",
                json.getString("availabilityRule")
        );
        assertEquals(10L, json.getLong("downEventElapsedRealtimeNanos"));
        assertEquals(20L, json.getLong("downReceivedElapsedRealtimeNanos"));
        assertEquals(30L, json.getLong("liftEventElapsedRealtimeNanos"));
        assertEquals(40L, json.getLong("liftReceivedElapsedRealtimeNanos"));
    }

    @Test
    public void captureWindowDeclaresTimeAndFrameLowerBounds() throws Exception {
        AispectSignalWindowBuilder builder = new AispectSignalWindowBuilder();
        AispectSignalWindowBuilder.Window window = builder.buildDualAnchor(
                frames(240, 200.0),
                0.40,
                0.60,
                200.0
        );

        JSONObject json = AispectDataLogger.captureWindowToJson(window);

        assertEquals(1, json.getInt("schemaVersion"));
        assertEquals(
                AispectSignalWindowBuilder.DUAL_ANCHOR_CAPTURE_CONTRACT,
                json.getString("contract")
        );
        assertEquals(85.0, json.getDouble("requiredPreDownDurationMs"), 1e-9);
        assertEquals(85.0, json.getDouble("requiredPostLiftDurationMs"), 1e-9);
        assertEquals(15, json.getInt("requiredPreDownFrameCount"));
        assertEquals(15, json.getInt("requiredPostLiftFrameCount"));
        assertTrue(json.getInt("actualPreDownFrameCount") >= 15);
        assertTrue(json.getInt("actualPostLiftFrameCount") >= 15);
        assertTrue(json.getDouble("actualPreDownDurationMs") >= 85.0);
        assertTrue(json.getDouble("actualPostLiftDurationMs") >= 85.0);
        assertTrue(json.getBoolean("preDownDurationComplete"));
        assertTrue(json.getBoolean("postLiftDurationComplete"));
        assertTrue(json.getBoolean("preDownFrameCountComplete"));
        assertTrue(json.getBoolean("postLiftFrameCountComplete"));
        assertTrue(json.getBoolean("windowComplete"));
    }

    private static List<AispectModels.ImpactFrame> frames(int count, double sampleRateHz) {
        ArrayList<AispectModels.ImpactFrame> frames = new ArrayList<>();
        for (int index = 0; index < count; index++) {
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
                    timestamp - 0.40,
                    timestamp - 0.60,
                    timestamp,
                    index,
                    index
            ));
        }
        return frames;
    }
}
