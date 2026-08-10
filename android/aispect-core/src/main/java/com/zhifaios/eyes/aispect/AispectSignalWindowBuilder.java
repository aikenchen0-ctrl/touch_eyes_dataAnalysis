package com.zhifaios.eyes.aispect;

import java.util.ArrayList;
import java.util.List;

public final class AispectSignalWindowBuilder {
    public static final String DUAL_ANCHOR_CAPTURE_CONTRACT = "dual_anchor_min_time_and_frames_v1";
    public static final int DEFAULT_DUAL_MARGIN_FRAME_COUNT = 15;
    public static final double DEFAULT_DUAL_MARGIN_DURATION_SECONDS = 0.085;

    public static final class Config {
        public int preFrames = 5;
        public int postFrames = 9;
        public int dualDownPreFrames = DEFAULT_DUAL_MARGIN_FRAME_COUNT;
        public int dualLiftPostFrames = DEFAULT_DUAL_MARGIN_FRAME_COUNT;
        public double dualDownPreDurationSeconds = DEFAULT_DUAL_MARGIN_DURATION_SECONDS;
        public double dualLiftPostDurationSeconds = DEFAULT_DUAL_MARGIN_DURATION_SECONDS;
        public double fallbackSampleRateHz = 120.0;
    }

    public static final class Window {
        public final AispectModels.ImpactFrame[] frames;
        public final int firstFrameIndex;
        public final int lastFrameIndex;
        public final double anchorTimestampSeconds;
        public final double downAnchorTimestampSeconds;
        public final double liftAnchorTimestampSeconds;
        public final double sampleRateHz;
        public final double maxAbsDelta;
        public final int downReferenceFrameIndex;
        public final int liftReferenceFrameIndex;
        public final String captureWindowContract;
        public final boolean preDownComplete;
        public final boolean postLiftComplete;
        public final double actualPreDownDurationSeconds;
        public final double actualPostLiftDurationSeconds;
        public final int requiredPreDownFrameCount;
        public final int requiredPostLiftFrameCount;
        public final double requiredPreDownDurationSeconds;
        public final double requiredPostLiftDurationSeconds;
        public final boolean windowComplete;

        Window(
                AispectModels.ImpactFrame[] frames,
                int firstFrameIndex,
                double anchorTimestampSeconds,
                double sampleRateHz,
                double maxAbsDelta
        ) {
            this(
                    frames,
                    firstFrameIndex,
                    firstFrameIndex + frames.length - 1,
                    anchorTimestampSeconds,
                    Double.NaN,
                    sampleRateHz,
                    maxAbsDelta,
                    0,
                    0,
                    true,
                    true,
                    true,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
            );
        }

        Window(
                AispectModels.ImpactFrame[] frames,
                int firstFrameIndex,
                int lastFrameIndex,
                double downAnchorTimestampSeconds,
                double liftAnchorTimestampSeconds,
                double sampleRateHz,
                double maxAbsDelta,
                int downReferenceFrameIndex,
                int liftReferenceFrameIndex,
                boolean windowComplete
        ) {
            this(
                    frames,
                    firstFrameIndex,
                    lastFrameIndex,
                    downAnchorTimestampSeconds,
                    liftAnchorTimestampSeconds,
                    sampleRateHz,
                    maxAbsDelta,
                    downReferenceFrameIndex,
                    liftReferenceFrameIndex,
                    windowComplete,
                    windowComplete,
                    windowComplete,
                    0,
                    0,
                    Double.isNaN(liftAnchorTimestampSeconds)
                            ? 0
                            : DEFAULT_DUAL_MARGIN_FRAME_COUNT,
                    Double.isNaN(liftAnchorTimestampSeconds)
                            ? 0
                            : DEFAULT_DUAL_MARGIN_FRAME_COUNT,
                    Double.isNaN(liftAnchorTimestampSeconds)
                            ? 0
                            : DEFAULT_DUAL_MARGIN_DURATION_SECONDS,
                    Double.isNaN(liftAnchorTimestampSeconds)
                            ? 0
                            : DEFAULT_DUAL_MARGIN_DURATION_SECONDS
            );
        }

        Window(
                AispectModels.ImpactFrame[] frames,
                int firstFrameIndex,
                int lastFrameIndex,
                double downAnchorTimestampSeconds,
                double liftAnchorTimestampSeconds,
                double sampleRateHz,
                double maxAbsDelta,
                int downReferenceFrameIndex,
                int liftReferenceFrameIndex,
                boolean windowComplete,
                boolean preDownComplete,
                boolean postLiftComplete,
                double actualPreDownDurationSeconds,
                double actualPostLiftDurationSeconds,
                int requiredPreDownFrameCount,
                int requiredPostLiftFrameCount,
                double requiredPreDownDurationSeconds,
                double requiredPostLiftDurationSeconds
        ) {
            this.frames = frames;
            this.firstFrameIndex = firstFrameIndex;
            this.lastFrameIndex = lastFrameIndex;
            this.anchorTimestampSeconds = Double.isNaN(liftAnchorTimestampSeconds)
                    ? downAnchorTimestampSeconds
                    : liftAnchorTimestampSeconds;
            this.downAnchorTimestampSeconds = downAnchorTimestampSeconds;
            this.liftAnchorTimestampSeconds = liftAnchorTimestampSeconds;
            this.sampleRateHz = sampleRateHz;
            this.maxAbsDelta = maxAbsDelta;
            this.downReferenceFrameIndex = downReferenceFrameIndex;
            this.liftReferenceFrameIndex = liftReferenceFrameIndex;
            this.captureWindowContract = Double.isNaN(liftAnchorTimestampSeconds)
                    ? "fixed_frame_window_v1"
                    : DUAL_ANCHOR_CAPTURE_CONTRACT;
            this.preDownComplete = preDownComplete;
            this.postLiftComplete = postLiftComplete;
            this.actualPreDownDurationSeconds = actualPreDownDurationSeconds;
            this.actualPostLiftDurationSeconds = actualPostLiftDurationSeconds;
            this.requiredPreDownFrameCount = requiredPreDownFrameCount;
            this.requiredPostLiftFrameCount = requiredPostLiftFrameCount;
            this.requiredPreDownDurationSeconds = requiredPreDownDurationSeconds;
            this.requiredPostLiftDurationSeconds = requiredPostLiftDurationSeconds;
            this.windowComplete = windowComplete;
        }
    }

    private final Config config = new Config();

    public Config config() {
        return config;
    }

    /** 中文注释：按下锚点固定窗口，只有 windowComplete 为 true 才允许发布。 */
    public Window buildDownAnchor(
            List<AispectModels.ImpactFrame> sourceFrames,
            double downAnchorTimestampSeconds,
            double sampleRateHz
    ) {
        return build(sourceFrames, downAnchorTimestampSeconds, sampleRateHz);
    }

    public Window build(List<AispectModels.ImpactFrame> sourceFrames, double anchorTimestampSeconds, double sampleRateHz) {
        int total = config.preFrames + config.postFrames + 1;
        AispectModels.ImpactFrame[] frames = new AispectModels.ImpactFrame[total];
        for (int i = 0; i < total; i++) {
            frames[i] = AispectModels.ImpactFrame.zero();
        }
        double safeRate = sampleRateHz > 0 ? sampleRateHz : config.fallbackSampleRateHz;
        if (sourceFrames.isEmpty()) {
            return new Window(
                    frames,
                    -config.preFrames,
                    config.postFrames,
                    anchorTimestampSeconds,
                    Double.NaN,
                    safeRate,
                    0,
                    0,
                    0,
                    false
            );
        }

        int referenceIndex = nearestFrameIndex(sourceFrames, anchorTimestampSeconds);
        double referenceTimestamp = sourceFrames.get(referenceIndex).timestampSeconds;
        boolean complete = referenceIndex - config.preFrames >= 0
                && referenceIndex + config.postFrames < sourceFrames.size()
                && hasContinuousRange(
                sourceFrames,
                referenceIndex - config.preFrames,
                referenceIndex + config.postFrames,
                safeRate
        );
        for (int offset = -config.preFrames; offset <= config.postFrames; offset++) {
            int absoluteIndex = referenceIndex + offset;
            int destinationIndex = offset + config.preFrames;
            if (absoluteIndex >= 0 && absoluteIndex < sourceFrames.size()) {
                AispectModels.ImpactFrame source = sourceFrames.get(absoluteIndex);
                frames[destinationIndex] = withAnchorTime(source, source.timestampSeconds - referenceTimestamp);
            } else {
                double relativeTimestamp = offset / safeRate;
                frames[destinationIndex] = new AispectModels.ImpactFrame(
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        relativeTimestamp,
                        relativeTimestamp,
                        referenceTimestamp + relativeTimestamp,
                        0L,
                        0L
                );
            }
        }
        return new Window(
                frames,
                -config.preFrames,
                config.postFrames,
                anchorTimestampSeconds,
                Double.NaN,
                safeRate,
                0,
                0,
                0,
                complete
        );
    }

    public Window buildDualAnchor(
            List<AispectModels.ImpactFrame> sourceFrames,
            double downAnchorTimestampSeconds,
            double liftAnchorTimestampSeconds,
            double sampleRateHz
    ) {
        double safeRate = sampleRateHz > 0 ? sampleRateHz : config.fallbackSampleRateHz;
        if (sourceFrames == null || sourceFrames.isEmpty() || liftAnchorTimestampSeconds < downAnchorTimestampSeconds) {
            return new Window(
                    new AispectModels.ImpactFrame[0],
                    0,
                    -1,
                    downAnchorTimestampSeconds,
                    liftAnchorTimestampSeconds,
                    safeRate,
                    0,
                    0,
                    0,
                    false
            );
        }

        int downIndex = nearestFrameIndex(sourceFrames, downAnchorTimestampSeconds);
        int liftIndex = nearestFrameIndex(sourceFrames, liftAnchorTimestampSeconds);
        if (liftIndex < downIndex) {
            return new Window(
                    new AispectModels.ImpactFrame[0],
                    0,
                    -1,
                    downAnchorTimestampSeconds,
                    liftAnchorTimestampSeconds,
                    safeRate,
                    0,
                    0,
                    0,
                    false
            );
        }

        int requiredPreFrames = Math.max(0, config.dualDownPreFrames);
        int requiredPostFrames = Math.max(0, config.dualLiftPostFrames);
        double requiredPreDuration = Math.max(0.0, config.dualDownPreDurationSeconds);
        double requiredPostDuration = Math.max(0.0, config.dualLiftPostDurationSeconds);
        int frameStartIndex = downIndex - requiredPreFrames;
        int frameEndIndex = liftIndex + requiredPostFrames;
        int durationStartIndex = lastFrameIndexAtOrBefore(
                sourceFrames,
                downAnchorTimestampSeconds - requiredPreDuration
        );
        int durationEndIndex = firstFrameIndexAtOrAfter(
                sourceFrames,
                liftAnchorTimestampSeconds + requiredPostDuration
        );
        int requestedStartIndex = Math.min(frameStartIndex, durationStartIndex);
        int requestedEndIndex = Math.max(frameEndIndex, durationEndIndex);
        int startIndex = clampIndex(requestedStartIndex, 0, sourceFrames.size() - 1);
        int endIndex = clampIndex(requestedEndIndex, 0, sourceFrames.size() - 1);
        double downReferenceTimestamp = sourceFrames.get(downIndex).timestampSeconds;
        double liftReferenceTimestamp = sourceFrames.get(liftIndex).timestampSeconds;
        AispectModels.ImpactFrame[] frames = new AispectModels.ImpactFrame[endIndex - startIndex + 1];
        for (int sourceIndex = startIndex; sourceIndex <= endIndex; sourceIndex++) {
            AispectModels.ImpactFrame source = sourceFrames.get(sourceIndex);
            AispectModels.ImpactFrame frame = withAnchorTimes(
                    source,
                    source.timestampSeconds - downReferenceTimestamp,
                    source.timestampSeconds - liftReferenceTimestamp
            );
            frames[sourceIndex - startIndex] = frame;
        }

        int firstFrameIndex = startIndex - downIndex;
        int lastFrameIndex = endIndex - downIndex;
        double actualPreDownDuration = Math.max(
                0.0,
                downAnchorTimestampSeconds - sourceFrames.get(startIndex).timestampSeconds
        );
        double actualPostLiftDuration = Math.max(
                0.0,
                sourceFrames.get(endIndex).timestampSeconds - liftAnchorTimestampSeconds
        );
        boolean preDownComplete = downIndex - startIndex >= requiredPreFrames
                && actualPreDownDuration + 1e-9 >= requiredPreDuration;
        boolean postLiftComplete = endIndex - liftIndex >= requiredPostFrames
                && actualPostLiftDuration + 1e-9 >= requiredPostDuration;
        boolean complete = preDownComplete
                && postLiftComplete
                && hasContinuousRange(sourceFrames, startIndex, endIndex, safeRate);
        return new Window(
                frames,
                firstFrameIndex,
                lastFrameIndex,
                downAnchorTimestampSeconds,
                liftAnchorTimestampSeconds,
                safeRate,
                0,
                0,
                liftIndex - downIndex,
                complete,
                preDownComplete,
                postLiftComplete,
                actualPreDownDuration,
                actualPostLiftDuration,
                requiredPreFrames,
                requiredPostFrames,
                requiredPreDuration,
                requiredPostDuration
        );
    }

    public List<AispectModels.ImpactFrame> copyWindowFrames(Window window) {
        ArrayList<AispectModels.ImpactFrame> copy = new ArrayList<>();
        for (AispectModels.ImpactFrame frame : window.frames) {
            copy.add(frame);
        }
        return copy;
    }

    private static int nearestFrameIndex(List<AispectModels.ImpactFrame> frames, double anchorTimestampSeconds) {
        int bestIndex = 0;
        double bestDistance = Double.MAX_VALUE;
        for (int i = 0; i < frames.size(); i++) {
            double distance = Math.abs(frames.get(i).timestampSeconds - anchorTimestampSeconds);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    private static int lastFrameIndexAtOrBefore(
            List<AispectModels.ImpactFrame> frames,
            double timestampSeconds
    ) {
        int bestIndex = -1;
        for (int index = 0; index < frames.size(); index++) {
            if (frames.get(index).timestampSeconds <= timestampSeconds + 1e-9) {
                bestIndex = index;
            } else {
                break;
            }
        }
        return bestIndex;
    }

    private static int firstFrameIndexAtOrAfter(
            List<AispectModels.ImpactFrame> frames,
            double timestampSeconds
    ) {
        for (int index = 0; index < frames.size(); index++) {
            if (frames.get(index).timestampSeconds + 1e-9 >= timestampSeconds) {
                return index;
            }
        }
        return frames.size();
    }

    private static int clampIndex(int index, int minimum, int maximum) {
        return Math.max(minimum, Math.min(maximum, index));
    }

    private static boolean hasContinuousRange(
            List<AispectModels.ImpactFrame> frames,
            int startIndex,
            int endIndex,
            double sampleRateHz
    ) {
        if (startIndex < 0 || endIndex >= frames.size() || startIndex >= endIndex) {
            return startIndex == endIndex;
        }
        double expectedInterval = sampleRateHz > 0 ? 1.0 / sampleRateHz : 0;
        double maximumGap = Math.max(0.03, expectedInterval * 4.0);
        for (int index = startIndex + 1; index <= endIndex; index++) {
            double gap = frames.get(index).timestampSeconds - frames.get(index - 1).timestampSeconds;
            if (gap <= 0 || gap > maximumGap) {
                return false;
            }
        }
        return true;
    }

    private static AispectModels.ImpactFrame withAnchorTime(AispectModels.ImpactFrame source, double timeSinceAnchor) {
        return withAnchorTimes(source, source.timeSinceTouchDown, timeSinceAnchor);
    }

    private static AispectModels.ImpactFrame withAnchorTimes(
            AispectModels.ImpactFrame source,
            double timeSinceDownAnchor,
            double timeSinceLiftAnchor
    ) {
        return new AispectModels.ImpactFrame(
                source.delta,
                source.x,
                source.y,
                source.z,
                source.rotationRateX,
                source.rotationRateY,
                source.rotationRateZ,
                source.gravityX,
                source.gravityY,
                source.gravityZ,
                source.xNorm,
                source.yNorm,
                timeSinceDownAnchor,
                timeSinceLiftAnchor,
                source.timestampSeconds,
                source.sensorTimestampElapsedRealtimeNanos,
                source.receivedElapsedRealtimeNanos
        );
    }
}
