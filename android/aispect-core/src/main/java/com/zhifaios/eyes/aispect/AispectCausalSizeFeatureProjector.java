package com.zhifaios.eyes.aispect;

import java.util.Arrays;

/** Project the persisted v5 matrix into the causal size runtime view. */
final class AispectCausalSizeFeatureProjector {
    private static final int SIZE_INDEX = 12;
    private static final int SIZE_AVAILABLE_INDEX = 19;
    private static final int MAJOR_AVAILABLE_INDEX = 20;
    private static final int MINOR_AVAILABLE_INDEX = 21;
    private static final int AXIS_RATIO_INDEX = 15;

    private AispectCausalSizeFeatureProjector() {
    }

    static double[][] project(
            double[][] source,
            String[] sourceFeatureNames,
            double baselineMedian
    ) {
        if (source == null
                || source.length != AispectStatsMlpRuntime.FRAME_COUNT
                || sourceFeatureNames == null
                || !Arrays.equals(
                AispectStatsMlpRuntime.SOURCE_FEATURE_NAMES,
                sourceFeatureNames
        )) {
            return null;
        }
        double[][] output = new double[source.length][
                AispectStatsMlpRuntime.SOURCE_FEATURE_INDICES.length
                ];
        for (int frame = 0; frame < source.length; frame++) {
            if (source[frame] == null
                    || source[frame].length != sourceFeatureNames.length) {
                return null;
            }
            for (int index = 0;
                 index < AispectStatsMlpRuntime.SOURCE_FEATURE_INDICES.length;
                 index++) {
                output[frame][index] = source[frame][
                        AispectStatsMlpRuntime.SOURCE_FEATURE_INDICES[index]
                        ];
            }
            double size = source[frame][SIZE_INDEX];
            output[frame][8] = source[frame][SIZE_AVAILABLE_INDEX] >= 0.5
                    && size > 0.0
                    && Double.isFinite(size)
                    && baselineMedian > 0.0
                    && Double.isFinite(baselineMedian)
                    ? clip(Math.log(size / baselineMedian))
                    : Double.NaN;
            double axisRatio = source[frame][AXIS_RATIO_INDEX];
            output[frame][9] = source[frame][MAJOR_AVAILABLE_INDEX] >= 0.5
                    && source[frame][MINOR_AVAILABLE_INDEX] >= 0.5
                    && Double.isFinite(axisRatio)
                    ? clip(Math.abs(axisRatio))
                    : Double.NaN;
        }
        return output;
    }

    private static double clip(double value) {
        return Math.max(
                -AispectCausalSizeBaseline.LOG_RATIO_CLIP,
                Math.min(
                        AispectCausalSizeBaseline.LOG_RATIO_CLIP,
                        value
                )
        );
    }
}
