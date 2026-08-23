package com.zhifaios.eyes.aispect;

import android.hardware.Sensor;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public final class AispectCrossDeviceCaptureMetadataTest {
    @Test
    public void constantTouchFieldsAreMarkedNotInformative() {
        AispectDeviceCapabilityProfiler profiler = new AispectDeviceCapabilityProfiler();
        profiler.observe(AispectTouchFrame.forTesting(
                1L, 100f, 100f, 1000, 1000, 1f, 0.5f, 40f, 20f, 0f, 1, false
        ));
        profiler.observe(AispectTouchFrame.forTesting(
                2L, 100f, 100f, 1000, 1000, 1f, 0.5f, 40f, 20f, 0f, 1, false
        ));

        AispectDeviceCapabilityProfiler.Snapshot snapshot = profiler.snapshot();
        assertTrue(snapshot.hasMajor);
        assertFalse(snapshot.pressureInformative);
        assertFalse(snapshot.sizeInformative);
        assertFalse(snapshot.touchMajorInformative);
    }

    @Test
    public void sensorSnapshotKeepsRotationObservationOutOfTraining() throws Exception {
        AispectImpactSignalCollector.RotationFrame frame = new AispectImpactSignalCollector.RotationFrame(
                "game_rotation_vector",
                "rotation",
                100L,
                110L,
                3,
                new float[]{0f, 0f, 0f, 1f}
        );
        AispectImpactSignalCollector.Snapshot snapshot = new AispectImpactSignalCollector.Snapshot(
                java.util.Collections.<AispectModels.ImpactFrame>emptyList(),
                200.0,
                "linear",
                "linear_acceleration",
                true,
                true,
                "HEALTHY",
                "HEALTHY",
                "HEALTHY",
                "HEALTHY",
                "game_rotation_vector",
                "rotation",
                java.util.Collections.singletonList(frame),
                2,
                0
        );
        assertEquals("HEALTHY", snapshot.rotationStatus);
        assertFalse(snapshot.rotationFrames.get(0).toJson().getBoolean("trainingInput"));
        assertEquals("game_rotation_vector", snapshot.rotationFrames.get(0).toJson().getString("sourceType"));
    }

    @Test
    public void invertedTouchAxesAndToolAliasesAreMarkedDegraded() throws Exception {
        AispectDeviceCapabilityProfiler profiler = new AispectDeviceCapabilityProfiler();
        profiler.observe(AispectTouchFrame.forTestingAxes(
                1L, 40f, 42f, 40f, 42f, 1f, 0.5f
        ));
        profiler.observe(AispectTouchFrame.forTestingAxes(
                2L, 41f, 43f, 41f, 43f, 1f, 0.5f
        ));

        AispectDeviceCapabilityProfiler.Snapshot snapshot = profiler.snapshot();
        assertFalse(snapshot.touchAxesSemanticallyValid);
        assertFalse(snapshot.toolAxesIndependent);
        assertFalse(snapshot.pressureInformative);
        assertEquals(AispectDeviceCapabilityProfiler.Level.C, snapshot.level);
        assertFalse(snapshot.toJson().getBoolean("touchAxesSemanticallyValid"));
        assertFalse(snapshot.toJson().getBoolean("toolAxesIndependent"));
    }

    @Test
    public void invertedTouchAxesCannotProduceHighNormalizationQuality() {
        AispectTouchFrame first = AispectTouchFrame.forTestingAxes(
                1L, 40f, 42f, 40f, 42f, 1f, 0.5f
        );
        AispectTouchFrame second = AispectTouchFrame.forTestingAxes(
                2L, 41f, 43f, 41f, 43f, 1f, 0.6f
        );
        java.util.List<AispectTouchFrame> frames = Arrays.asList(first, second);
        AispectTouchNormalizationProfile profile = new AispectTouchNormalizationEngine().evaluate(
                frames,
                AispectContactPatch.fromFrames(frames)
        );

        assertFalse(profile.diagnostics.touchAxesSemanticallyValid);
        assertFalse(profile.diagnostics.toolAxesIndependent);
        assertFalse(profile.diagnostics.hasShape);
        assertEquals("medium", profile.diagnostics.fieldQualityLevel);
    }

    @Test
    public void equalTouchAxesAreReliableRadiusButNotIndependentShape() {
        AispectDeviceCapabilityProfiler profiler = new AispectDeviceCapabilityProfiler();
        java.util.List<AispectTouchFrame> frames = new java.util.ArrayList<>();
        for (int index = 0; index < 8; index++) {
            float radius = 40f + index;
            AispectTouchFrame frame = AispectTouchFrame.forTestingAxes(
                    index + 1L,
                    radius,
                    radius,
                    radius,
                    radius,
                    1f,
                    0.4f + index * 0.01f
            );
            frames.add(frame);
            profiler.observe(frame);
        }

        AispectDeviceCapabilityProfiler.Snapshot snapshot = profiler.snapshot();
        assertTrue(snapshot.touchAxesSemanticallyValid);
        assertFalse(snapshot.touchAxesDistinct);
        assertTrue(snapshot.touchMajorInformative);
        assertFalse(snapshot.touchMinorInformative);
        assertEquals(AispectDeviceCapabilityProfiler.Level.B, snapshot.level);

        AispectTouchNormalizationProfile profile = new AispectTouchNormalizationEngine().evaluate(
                frames,
                AispectContactPatch.fromFrames(frames)
        );
        assertFalse(profile.diagnostics.hasShape);
        assertEquals("medium", profile.diagnostics.fieldQualityLevel);
    }

    @Test
    public void currentTouchCapabilityDoesNotInheritPreviousTouchAxisShape() {
        AispectDeviceCapabilityProfiler previousTouchProfiler = new AispectDeviceCapabilityProfiler();
        previousTouchProfiler.observe(AispectTouchFrame.forTestingAxes(
                1L, 44f, 32f, 44f, 32f, 1f, 0.4f
        ));
        previousTouchProfiler.observe(AispectTouchFrame.forTestingAxes(
                2L, 46f, 34f, 46f, 34f, 1f, 0.5f
        ));
        assertTrue(previousTouchProfiler.snapshot().touchAxesDistinct);

        java.util.List<AispectTouchFrame> currentTouchFrames = Arrays.asList(
                AispectTouchFrame.forTestingAxes(
                        3L, 40f, 40f, 40f, 40f, 1f, 0.4f
                ),
                AispectTouchFrame.forTestingAxes(
                        4L, 42f, 42f, 42f, 42f, 1f, 0.5f
                )
        );

        AispectDeviceCapabilityProfiler.Snapshot currentTouch =
                AispectDeviceCapabilityProfiler.snapshotForFrames(currentTouchFrames);

        assertEquals(2, currentTouch.observedFrames);
        assertFalse(currentTouch.touchAxesDistinct);
        assertFalse(currentTouch.touchMinorInformative);
        assertEquals(AispectDeviceCapabilityProfiler.Level.B, currentTouch.level);
    }

    @Test
    public void touchMetadataEpochStaysStableUntilFrameMetadataChanges() throws Exception {
        AispectDataLogger.MetadataEpochTracker tracker =
                new AispectDataLogger.MetadataEpochTracker();
        JSONObject baseMetadata = AispectDataLogger.touchMetadataToJson(0);
        java.util.List<AispectTouchFrame> frames = Arrays.asList(
                touchFrameWithViewSize(1L, 1000, 2000),
                touchFrameWithViewSize(2L, 1000, 2000),
                touchFrameWithViewSize(3L, 1200, 2000)
        );

        AispectDataLogger.TouchMetadataSerialization serialization =
                AispectDataLogger.serializeTouchMetadata(baseMetadata, frames, tracker);
        JSONArray serializedFrames = serialization.touchFrames;

        assertEquals(1, serialization.recordMetadata.getInt("metadataEpoch"));
        assertEquals(1, serializedFrames.getJSONObject(0).getInt("metadataEpoch"));
        assertEquals(1, serializedFrames.getJSONObject(1).getInt("metadataEpoch"));
        assertEquals(2, serializedFrames.getJSONObject(2).getInt("metadataEpoch"));

        AispectDataLogger.TouchMetadataSerialization nextSerialization =
                AispectDataLogger.serializeTouchMetadata(
                        baseMetadata,
                        Collections.singletonList(touchFrameWithViewSize(4L, 1200, 2000)),
                        tracker
                );
        assertEquals(2, nextSerialization.recordMetadata.getInt("metadataEpoch"));
        assertEquals(
                2,
                nextSerialization.touchFrames.getJSONObject(0).getInt("metadataEpoch")
        );
    }

    @Test
    public void rotationFramesAreClippedToImuCaptureWindow() throws Exception {
        AispectModels.ImpactFrame first = impactFrameAt(100L);
        AispectModels.ImpactFrame last = impactFrameAt(300L);
        AispectSignalWindowBuilder.Window window = new AispectSignalWindowBuilder.Window(
                new AispectModels.ImpactFrame[]{first, last},
                -1,
                1,
                0.0,
                0.1,
                200.0,
                0.0,
                0,
                0,
                true
        );
        AispectImpactSignalCollector.Snapshot snapshot = snapshotWithRotationFrames(
                rotationFrameAt(50L),
                rotationFrameAt(100L),
                rotationFrameAt(200L),
                rotationFrameAt(300L),
                rotationFrameAt(350L)
        );

        org.json.JSONObject metadata = AispectDataLogger.sensorMetadataToJson(snapshot, window);
        org.json.JSONArray frames = metadata.getJSONArray("rotationFrames");
        assertEquals(3, frames.length());
        assertEquals(100L, frames.getJSONObject(0).getLong("sensorTimestampElapsedRealtimeNanos"));
        assertEquals(300L, frames.getJSONObject(2).getLong("sensorTimestampElapsedRealtimeNanos"));
    }

    @Test
    public void derivedAccelerationUsesDerivedSourceName() {
        assertEquals(
                "accelerometer_minus_gravity",
                AispectImpactSignalCollector.accelerationFrameSource(
                        Sensor.TYPE_ACCELEROMETER,
                        true
                )
        );
    }

    @Test
    public void accelerationSourceTransitionAdvancesStreamEpoch() {
        assertEquals(
                4,
                AispectImpactSignalCollector.streamEpochAfterSource(
                        4,
                        "",
                        "linear_acceleration"
                )
        );
        assertEquals(
                4,
                AispectImpactSignalCollector.streamEpochAfterSource(
                        4,
                        "linear_acceleration",
                        "linear_acceleration"
                )
        );
        assertEquals(
                5,
                AispectImpactSignalCollector.streamEpochAfterSource(
                        4,
                        "accelerometer_minus_gravity",
                        "accelerometer_raw_no_gravity"
                )
        );
    }

    @Test
    public void impactFrameSerializesTrainingEligibilityFromSource() throws Exception {
        assertTrue(impactFrameAt(100L).toJson(0).getBoolean("trainingInput"));
        assertFalse(impactFrameAt(100L, "accelerometer_raw_no_gravity")
                .toJson(0)
                .getBoolean("trainingInput"));
        assertFalse(impactFrameAt(100L, "unavailable")
                .toJson(0)
                .getBoolean("trainingInput"));
    }

    @Test
    public void unavailableAccelerationIsNotTrainingInput() throws Exception {
        AispectImpactSignalCollector.Snapshot snapshot = new AispectImpactSignalCollector.Snapshot(
                Collections.<AispectModels.ImpactFrame>emptyList(),
                0.0,
                "unavailable",
                "unavailable",
                false,
                false,
                "UNAVAILABLE",
                "UNAVAILABLE",
                "UNAVAILABLE",
                "UNAVAILABLE",
                "unavailable",
                "unavailable",
                Collections.<AispectImpactSignalCollector.RotationFrame>emptyList(),
                1,
                0
        );

        assertFalse(AispectDataLogger.sensorMetadataToJson(snapshot)
                .getBoolean("accelerationTrainingInput"));
    }

    private static AispectModels.ImpactFrame impactFrameAt(long sensorTimestamp) {
        return impactFrameAt(sensorTimestamp, "linear_acceleration");
    }

    private static AispectTouchFrame touchFrameWithViewSize(
            long receivedElapsedRealtimeNanos,
            int width,
            int height
    ) {
        return AispectTouchFrame.forTesting(
                receivedElapsedRealtimeNanos,
                100f,
                100f,
                width,
                height,
                0.5f,
                0.4f,
                40f,
                20f,
                0f,
                1,
                false
        );
    }

    private static AispectModels.ImpactFrame impactFrameAt(
            long sensorTimestamp,
            String sensorSource
    ) {
        return new AispectModels.ImpactFrame(
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                sensorTimestamp / 1_000_000_000.0,
                sensorTimestamp,
                sensorTimestamp + 1,
                sensorSource,
                2
        );
    }

    private static AispectImpactSignalCollector.RotationFrame rotationFrameAt(long sensorTimestamp) {
        return new AispectImpactSignalCollector.RotationFrame(
                "game_rotation_vector",
                "rotation",
                sensorTimestamp,
                sensorTimestamp + 1,
                2,
                new float[]{0f, 0f, 0f, 1f}
        );
    }

    private static AispectImpactSignalCollector.Snapshot snapshotWithRotationFrames(
            AispectImpactSignalCollector.RotationFrame... frames
    ) {
        return new AispectImpactSignalCollector.Snapshot(
                Collections.<AispectModels.ImpactFrame>emptyList(),
                200.0,
                "linear",
                "linear_acceleration",
                true,
                true,
                "HEALTHY",
                "HEALTHY",
                "HEALTHY",
                "HEALTHY",
                "game_rotation_vector",
                "rotation",
                Arrays.asList(frames),
                2,
                0
        );
    }
}
