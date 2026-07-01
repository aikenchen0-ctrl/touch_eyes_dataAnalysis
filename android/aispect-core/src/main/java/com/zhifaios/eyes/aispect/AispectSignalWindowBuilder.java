package com.zhifaios.eyes.aispect;

import java.util.ArrayList;
import java.util.List;

public final class AispectSignalWindowBuilder {
    public static final class Config {
        public int preFrames = 5;
        public int postFrames = 15;
        public double fallbackSampleRateHz = 120.0;
    }

    public static final class Window {
        public final AispectModels.ImpactFrame[] frames;
        public final int firstFrameIndex;
        public final double anchorTimestampSeconds;
        public final double sampleRateHz;
        public final double maxAbsDelta;

        Window(
                AispectModels.ImpactFrame[] frames,
                int firstFrameIndex,
                double anchorTimestampSeconds,
                double sampleRateHz,
                double maxAbsDelta
        ) {
            this.frames = frames;
            this.firstFrameIndex = firstFrameIndex;
            this.anchorTimestampSeconds = anchorTimestampSeconds;
            this.sampleRateHz = sampleRateHz;
            this.maxAbsDelta = maxAbsDelta;
        }
    }

    private final Config config = new Config();

    public Config config() {
        return config;
    }

    public Window build(List<AispectModels.ImpactFrame> sourceFrames, double anchorTimestampSeconds, double sampleRateHz) {
        int total = config.preFrames + config.postFrames + 1;
        AispectModels.ImpactFrame[] frames = new AispectModels.ImpactFrame[total];
        for (int i = 0; i < total; i++) {
            frames[i] = AispectModels.ImpactFrame.zero();
        }
        double safeRate = sampleRateHz > 0 ? sampleRateHz : config.fallbackSampleRateHz;
        if (sourceFrames.isEmpty()) {
            return new Window(frames, -config.preFrames, anchorTimestampSeconds, safeRate, 0);
        }

        int referenceIndex = nearestFrameIndex(sourceFrames, anchorTimestampSeconds);
        double referenceTimestamp = sourceFrames.get(referenceIndex).timestampSeconds;
        double maxAbsDelta = 0;
        for (int offset = -config.preFrames; offset <= config.postFrames; offset++) {
            int absoluteIndex = referenceIndex + offset;
            int destinationIndex = offset + config.preFrames;
            if (absoluteIndex >= 0 && absoluteIndex < sourceFrames.size()) {
                AispectModels.ImpactFrame source = sourceFrames.get(absoluteIndex);
                frames[destinationIndex] = withAnchorTime(source, source.timestampSeconds - referenceTimestamp);
            } else {
                double relativeTimestamp = offset / safeRate;
                frames[destinationIndex] = new AispectModels.ImpactFrame(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, relativeTimestamp, relativeTimestamp, referenceTimestamp + relativeTimestamp);
            }
            maxAbsDelta = Math.max(maxAbsDelta, Math.abs(frames[destinationIndex].delta));
        }
        return new Window(frames, -config.preFrames, anchorTimestampSeconds, safeRate, maxAbsDelta);
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

    private static AispectModels.ImpactFrame withAnchorTime(AispectModels.ImpactFrame source, double timeSinceAnchor) {
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
                source.timeSinceTouchDown,
                timeSinceAnchor,
                source.timestampSeconds
        );
    }
}
