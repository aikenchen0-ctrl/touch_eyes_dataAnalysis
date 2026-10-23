package com.zhifaios.eyes.aispect;

import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

public final class AispectCrossDeviceTimeGridFeatureBuilderTest {
    @Test
    public void timeGridInterpolatesCanonicalImuValuesAtFixedPhysicalOffsets() {
        AispectModels.ImpactFrame[] frames = new AispectModels.ImpactFrame[]{
                frame(980_000_000L, -2.0),
                frame(990_000_000L, -1.0),
                frame(1_000_000_000L, 0.0),
                frame(1_010_000_000L, 1.0),
                frame(1_020_000_000L, 2.0),
                frame(1_030_000_000L, 3.0)
        };
        List<AispectTouchFrame> touches = new ArrayList<>();
        touches.add(AispectTouchFrame.forTesting(
                990_000_000L, 990_500_000L, 200f, 600f, 1000, 2000,
                0.5f, 0.2f, 20f, 10f, 0f, 1, true
        ));

        double[][] matrix = AispectCausalPressFeatureBuilder.buildTimeGrid(
                frames,
                touches,
                1_000_000_000L,
                "causal_time_grid_imu11_v2"
        );

        assertNotNull(matrix);
        assertEquals(9, matrix.length);
        assertEquals(11, matrix[0].length);
        assertEquals(-1.5, matrix[0][2], 1e-6);
        assertEquals(0.0, matrix[3][2], 1e-6);
        assertEquals(2.5, matrix[8][2], 1e-6);
        assertEquals(0.2, matrix[1][0], 1e-6);
        assertEquals(0.3, matrix[1][1], 1e-6);
    }

    @Test
    public void timeGridRejectsTouchFrameWithoutBothClocks() {
        AispectModels.ImpactFrame[] frames = new AispectModels.ImpactFrame[]{
                frame(980_000_000L, -2.0),
                frame(990_000_000L, -1.0),
                frame(1_000_000_000L, 0.0),
                frame(1_010_000_000L, 1.0),
                frame(1_020_000_000L, 2.0),
                frame(1_030_000_000L, 3.0)
        };
        List<AispectTouchFrame> touches = new ArrayList<>();
        touches.add(AispectTouchFrame.forTesting(
                0L, 990_500_000L, 200f, 600f, 1000, 2000,
                0.5f, 0.2f, 20f, 10f, 0f, 1, true
        ));

        assertNull(AispectCausalPressFeatureBuilder.buildTimeGrid(
                frames,
                touches,
                1_000_000_000L,
                "causal_time_grid_imu11_touch9_v2"
        ));
    }

    @Test
    public void timeGridReadinessRequiresRightSupportBeyondPositiveEndpoint() {
        AispectModels.ImpactFrame[] withoutRightSupport = new AispectModels.ImpactFrame[]{
                frame(980_000_000L, -2.0),
                frame(990_000_000L, -1.0),
                frame(1_000_000_000L, 0.0),
                frame(1_010_000_000L, 1.0),
                frame(1_020_000_000L, 2.0)
        };
        assertFalse(AispectCausalPressFeatureBuilder.hasTimeGridSupport(
                withoutRightSupport,
                1_000_000_000L
        ));

        AispectModels.ImpactFrame[] withRightSupport = new AispectModels.ImpactFrame[]{
                frame(980_000_000L, -2.0),
                frame(990_000_000L, -1.0),
                frame(1_000_000_000L, 0.0),
                frame(1_010_000_000L, 1.0),
                frame(1_020_000_000L, 2.0),
                frame(1_030_000_000L, 3.0)
        };
        assertTrue(AispectCausalPressFeatureBuilder.hasTimeGridSupport(
                withRightSupport,
                1_000_000_000L
        ));
    }

    @Test
    public void timeGridV4ContractUsesPositiveShapeWithoutDeclaredRange() {
        AispectModels.ImpactFrame[] frames = new AispectModels.ImpactFrame[]{
                frame(980_000_000L, -2.0),
                frame(990_000_000L, -1.0),
                frame(1_000_000_000L, 0.0),
                frame(1_010_000_000L, 1.0),
                frame(1_020_000_000L, 2.0),
                frame(1_030_000_000L, 3.0)
        };
        List<AispectTouchFrame> touches = new ArrayList<>();
        touches.add(AispectTouchFrame.forTesting(
                990_000_000L, 990_500_000L, 200f, 600f, 1000, 2000,
                0.5f, 0.2f, 20f, 10f, 0f, 1, true
        ));

        double[][] matrix = AispectCausalPressFeatureBuilder.buildTimeGrid(
                frames,
                touches,
                1_000_000_000L,
                "causal_time_grid_imu11_touch9_mask6_v4"
        );

        assertNotNull(matrix);
        assertEquals(26, matrix[0].length);
        assertTrue(matrix[1][13] > 0.0);
        assertTrue(matrix[1][14] > 0.0);
        assertTrue(matrix[1][15] > 0.0);
    }

    @Test
    public void timeGridTouchContractTreatsNonPositiveOrNonFiniteShapeAsMissing() {
        AispectModels.ImpactFrame[] frames = new AispectModels.ImpactFrame[]{
                frame(980_000_000L, -2.0),
                frame(990_000_000L, -1.0),
                frame(1_000_000_000L, 0.0),
                frame(1_010_000_000L, 1.0),
                frame(1_020_000_000L, 2.0),
                frame(1_030_000_000L, 3.0)
        };
        List<AispectTouchFrame> touches = new ArrayList<>();
        touches.add(AispectTouchFrame.forTesting(
                990_000_000L, 990_500_000L, 200f, 600f, 1000, 2000,
                0.5f, 0.0f, -1f, Float.NaN, 0f, 1, false
        ));

        double[][] matrix = AispectCausalPressFeatureBuilder.buildTimeGrid(
                frames,
                touches,
                1_000_000_000L,
                "causal_time_grid_imu11_touch7_mask5_v3"
        );

        assertNotNull(matrix);
        assertEquals(0.0, matrix[1][12], 1e-6);
        assertEquals(0.0, matrix[1][13], 1e-6);
        assertEquals(0.0, matrix[1][14], 1e-6);
        assertEquals(0.0, matrix[1][15], 1e-6);
        assertEquals(0.0, matrix[1][19], 1e-6);
        assertEquals(0.0, matrix[1][20], 1e-6);
        assertEquals(0.0, matrix[1][21], 1e-6);
    }

    @Test
    public void timeGridTouchMaskContractEmitsTwentyThreeCausalChannels() {
        AispectModels.ImpactFrame[] frames = new AispectModels.ImpactFrame[]{
                frame(980_000_000L, -2.0),
                frame(990_000_000L, -1.0),
                frame(1_000_000_000L, 0.0),
                frame(1_010_000_000L, 1.0),
                frame(1_020_000_000L, 2.0),
                frame(1_030_000_000L, 3.0)
        };
        List<AispectTouchFrame> touches = new ArrayList<>();
        touches.add(AispectTouchFrame.forTesting(
                990_000_000L, 990_500_000L, 200f, 600f, 1000, 2000,
                0.5f, 0.2f, 20f, 10f, 0f, 1, false
        ));

        double[][] matrix = AispectCausalPressFeatureBuilder.buildTimeGrid(
                frames,
                touches,
                1_000_000_000L,
                "causal_time_grid_imu11_touch7_mask5_v3"
        );

        assertNotNull(matrix);
        assertEquals(23, matrix[0].length);
        assertEquals(1.0, matrix[1][16], 1e-6);
        assertEquals(1.0, matrix[1][17], 1e-6);
        assertEquals(0.0, matrix[1][18], 1e-6);
        assertEquals(0.0, matrix[1][19], 1e-6);
        assertEquals(0.0, matrix[1][20], 1e-6);
        assertEquals(0.0, matrix[1][21], 1e-6);
        assertEquals(1.0, matrix[1][22], 1e-6);
    }

    @Test
    public void timeGridV5UsesBatchReceiptAcrossAllSlots() {
        AispectModels.ImpactFrame[] frames = new AispectModels.ImpactFrame[]{
                frameWithReceipt(980_000_000L, 982_000_000L, -2.0),
                frameWithReceipt(990_000_000L, 992_000_000L, -1.0),
                frameWithReceipt(1_000_000_000L, 1_002_000_000L, 0.0),
                frameWithReceipt(1_010_000_000L, 1_012_000_000L, 1.0),
                frameWithReceipt(1_020_000_000L, 1_022_000_000L, 2.0),
                frameWithReceipt(1_030_000_000L, 1_032_000_000L, 3.0)
        };
        List<AispectTouchFrame> touches = new ArrayList<>();
        touches.add(AispectTouchFrame.forTesting(
                985_000_000L, 995_000_000L, 200f, 600f, 1000, 2000,
                0.5f, 0.2f, 20f, 10f, 0f, 1, false
        ));

        double[][] v4 = AispectCausalPressFeatureBuilder.buildTimeGrid(
                frames,
                touches,
                1_000_000_000L,
                "causal_time_grid_imu11_touch9_mask6_v4"
        );
        double[][] v5 = AispectCausalPressFeatureBuilder.buildTimeGrid(
                frames,
                touches,
                1_000_000_000L,
                "causal_time_grid_imu11_touch7_mask5_v5"
        );
        double[][] v6 = AispectCausalPressFeatureBuilder.buildTimeGrid(
                frames,
                touches,
                1_000_000_000L,
                "causal_time_grid_imu11_touch7_mask5_stats_mlp_v4",
                new AispectCausalSizeBaseline(
                        "test-device",
                        java.util.Collections.emptyList()
                )
        );

        assertNotNull(v4);
        assertNotNull(v5);
        assertNotNull(v6);
        assertEquals(0.0, v4[0][0], 1e-6);
        assertEquals(0.2, v5[0][0], 1e-6);
        assertEquals(0.2, v6[0][0], 1e-6);
        assertEquals(10, v6[0].length);
    }

    @Test
    public void timeGridV5RejectsTouchWithLateReceiptOrFutureEvent() {
        AispectModels.ImpactFrame[] frames = new AispectModels.ImpactFrame[]{
                frameWithReceipt(980_000_000L, 982_000_000L, -2.0),
                frameWithReceipt(990_000_000L, 992_000_000L, -1.0),
                frameWithReceipt(1_000_000_000L, 1_002_000_000L, 0.0),
                frameWithReceipt(1_010_000_000L, 1_012_000_000L, 1.0),
                frameWithReceipt(1_020_000_000L, 1_022_000_000L, 2.0),
                frameWithReceipt(1_030_000_000L, 1_032_000_000L, 3.0)
        };
        List<AispectTouchFrame> lateReceipt = new ArrayList<>();
        lateReceipt.add(AispectTouchFrame.forTesting(
                985_000_000L, 1_033_000_000L, 200f, 600f, 1000, 2000,
                0.5f, 0.2f, 20f, 10f, 0f, 1, false
        ));
        List<AispectTouchFrame> futureEvent = new ArrayList<>();
        futureEvent.add(AispectTouchFrame.forTesting(
                986_000_000L, 1_000_000_000L, 200f, 600f, 1000, 2000,
                0.5f, 0.2f, 20f, 10f, 0f, 1, false
        ));

        double[][] late = AispectCausalPressFeatureBuilder.buildTimeGrid(
                frames,
                lateReceipt,
                1_000_000_000L,
                "causal_time_grid_imu11_touch7_mask5_v5"
        );
        double[][] future = AispectCausalPressFeatureBuilder.buildTimeGrid(
                frames,
                futureEvent,
                1_000_000_000L,
                "causal_time_grid_imu11_touch7_mask5_v5"
        );
        double[][] lateV6 = AispectCausalPressFeatureBuilder.buildTimeGrid(
                frames,
                lateReceipt,
                1_000_000_000L,
                "causal_time_grid_imu11_touch7_mask5_stats_mlp_v4",
                new AispectCausalSizeBaseline(
                        "test-device",
                        java.util.Collections.emptyList()
                )
        );
        double[][] futureV6 = AispectCausalPressFeatureBuilder.buildTimeGrid(
                frames,
                futureEvent,
                1_000_000_000L,
                "causal_time_grid_imu11_touch7_mask5_stats_mlp_v4",
                new AispectCausalSizeBaseline(
                        "test-device",
                        java.util.Collections.emptyList()
                )
        );

        assertNotNull(late);
        assertNotNull(future);
        assertNotNull(lateV6);
        assertNotNull(futureV6);
        assertEquals(0.0, late[0][0], 1e-6);
        assertEquals(0.0, future[0][0], 1e-6);
        assertEquals(0.0, lateV6[0][0], 1e-6);
        assertEquals(0.0, futureV6[0][0], 1e-6);
    }

    @Test
    public void timeGridNoGravityContractRemovesThreeAbsoluteGravityChannels() {
        AispectModels.ImpactFrame[] frames = poseFrames();
        List<AispectTouchFrame> touches = touchFrames();

        double[][] matrix = AispectCausalPressFeatureBuilder.buildTimeGrid(
                frames,
                touches,
                1_000_000_000L,
                "causal_time_grid_touch7_mask5_no_gravity_v1"
        );

        assertNotNull(matrix);
        assertEquals(20, matrix[0].length);
        assertEquals(20, AispectCausalPressFeatureBuilder.featureNames(
                "causal_time_grid_touch7_mask5_no_gravity_v1"
        ).length);
        assertEquals("pressure", AispectCausalPressFeatureBuilder.featureNames(
                "causal_time_grid_touch7_mask5_no_gravity_v1"
        )[8]);
    }

    @Test
    public void timeGridDeltaGravityContractUsesCausalAdjacentGridDifferences() {
        AispectModels.ImpactFrame[] frames = poseFrames();
        List<AispectTouchFrame> touches = touchFrames();

        double[][] matrix = AispectCausalPressFeatureBuilder.buildTimeGrid(
                frames,
                touches,
                1_000_000_000L,
                "causal_time_grid_imu11_touch7_mask5_delta_gravity_v1"
        );

        assertNotNull(matrix);
        assertEquals(23, matrix[0].length);
        assertEquals(0.0, matrix[0][8], 1e-6);
        assertEquals(0.0, matrix[0][9], 1e-6);
        assertEquals(0.0, matrix[0][10], 1e-6);
        assertEquals(0.5, matrix[1][8], 1e-6);
        assertEquals(1.0, matrix[1][9], 1e-6);
        assertEquals(1.5, matrix[1][10], 1e-6);
        assertEquals("deltaGravity.x", AispectCausalPressFeatureBuilder.featureNames(
                "causal_time_grid_imu11_touch7_mask5_delta_gravity_v1"
        )[8]);
    }

    @Test
    public void statsMlpContractProjectsPositiveShapeWithoutDeclaredRange() {
        AispectModels.ImpactFrame[] frames = poseFrames();
        List<AispectTouchFrame> touches = new ArrayList<>();
        touches.add(AispectTouchFrame.forTesting(
                990_000_000L, 990_500_000L, 200f, 600f, 1000, 2000,
                0.5f, 0.2f, 20f, 10f, 0f, 1, false
        ));

        double[][] matrix = AispectCausalPressFeatureBuilder.buildTimeGrid(
                frames,
                touches,
                1_000_000_000L,
                "causal_time_grid_imu11_touch7_mask5_stats_mlp_v2"
        );

        assertNotNull(matrix);
        assertEquals(9, matrix.length);
        assertEquals(10, matrix[0].length);
        assertEquals("size", AispectCausalPressFeatureBuilder.featureNames(
                "causal_time_grid_imu11_touch7_mask5_stats_mlp_v2"
        )[8]);
        assertEquals("log_axis_ratio", AispectCausalPressFeatureBuilder.featureNames(
                "causal_time_grid_imu11_touch7_mask5_stats_mlp_v2"
        )[9]);
        assertEquals(0.2, matrix[1][8], 1e-6);
        assertEquals(Math.log(2.0), matrix[1][9], 1e-6);
    }

    private static AispectModels.ImpactFrame[] poseFrames() {
        return new AispectModels.ImpactFrame[]{
                poseFrame(980_000_000L, -2.0),
                poseFrame(990_000_000L, -1.0),
                poseFrame(1_000_000_000L, 0.0),
                poseFrame(1_010_000_000L, 1.0),
                poseFrame(1_020_000_000L, 2.0),
                poseFrame(1_030_000_000L, 3.0)
        };
    }

    private static List<AispectTouchFrame> touchFrames() {
        List<AispectTouchFrame> touches = new ArrayList<>();
        touches.add(AispectTouchFrame.forTesting(
                990_000_000L, 990_500_000L, 200f, 600f, 1000, 2000,
                0.5f, 0.2f, 20f, 10f, 0f, 1, true
        ));
        return touches;
    }

    private static AispectModels.ImpactFrame poseFrame(long timestampNanos, double value) {
        return new AispectModels.ImpactFrame(
                0,
                value,
                value + 1,
                value + 2,
                value + 2,
                value + 4,
                value + 6,
                value,
                value * 2,
                9.8 + value * 3,
                0,
                0,
                0,
                0,
                timestampNanos / 1_000_000_000.0,
                timestampNanos,
                timestampNanos + 2_000_000L
        );
    }

    private static AispectModels.ImpactFrame frame(long timestampNanos, double x) {
        return frameWithReceipt(timestampNanos, timestampNanos + 2_000_000L, x);
    }

    private static AispectModels.ImpactFrame frameWithReceipt(
            long timestampNanos,
            long receiptNanos,
            double x
    ) {
        return new AispectModels.ImpactFrame(
                0,
                x,
                x + 1,
                x + 2,
                x + 2,
                x + 4,
                x + 6,
                0,
                0,
                9.8,
                0,
                0,
                0,
                0,
                timestampNanos / 1_000_000_000.0,
                timestampNanos,
                receiptNanos
        );
    }
}
