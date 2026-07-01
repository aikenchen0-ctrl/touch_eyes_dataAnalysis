package com.zhifaios.eyes.aispect;

import java.util.ArrayList;
import java.util.List;

public final class AispectTouchNormalizationEngine {
    public static final class Config {
        public int baselineWarmupCount = 8;
        public int baselineMaxSamples = 64;
    }

    private static final class RunningStats {
        private final int maxSamples;
        private int count;
        private double mean;

        RunningStats(int maxSamples) {
            this.maxSamples = Math.max(1, maxSamples);
        }

        void observe(double value) {
            if (!Double.isFinite(value) || value <= 0) {
                return;
            }
            if (count < maxSamples) {
                count += 1;
                mean += (value - mean) / count;
            } else {
                mean = mean * 0.92 + value * 0.08;
            }
        }

        int count() {
            return count;
        }

        double mean() {
            return mean;
        }
    }

    private static final class TouchStats {
        int frameCount;
        double majorMaxPx;
        double minorMaxPx;
        double orientationRad;
        double screenDiagonalPx;
        double screenAreaPx;
        boolean hasRadius;
        boolean hasShape;
        boolean hasPressureOrSize;
        String source = "unavailable";
    }

    private static final class TouchReference {
        final double major;
        final double minor;
        final double area;

        TouchReference(double major, double minor, double area) {
            this.major = major;
            this.minor = minor;
            this.area = area;
        }
    }

    private final Config config = new Config();
    private AispectTouchNormalizationProfile.Scheme selectedScheme = AispectTouchNormalizationProfile.Scheme.COVARIANCE_MATRIX;
    private RunningStats majorBaseline = new RunningStats(config.baselineMaxSamples);
    private RunningStats minorBaseline = new RunningStats(config.baselineMaxSamples);
    private RunningStats areaBaseline = new RunningStats(config.baselineMaxSamples);
    private final ArrayList<TouchReference> references = new ArrayList<>();

    public Config config() {
        return config;
    }

    public AispectTouchNormalizationProfile.Scheme selectedScheme() {
        return selectedScheme;
    }

    public void setSelectedScheme(AispectTouchNormalizationProfile.Scheme scheme) {
        if (scheme != null) {
            selectedScheme = scheme;
        }
    }

    public void resetBaseline() {
        majorBaseline = new RunningStats(config.baselineMaxSamples);
        minorBaseline = new RunningStats(config.baselineMaxSamples);
        areaBaseline = new RunningStats(config.baselineMaxSamples);
        references.clear();
    }

    public AispectTouchNormalizationProfile evaluate(List<AispectTouchFrame> frames, AispectContactPatch patch) {
        TouchStats stats = collectStats(frames, patch);
        // 三套画像同时生成，保证离线训练可以横向比较，不把采集阶段的选择写死成唯一真相。
        AispectTouchNormalizationProfile.Result screenScale = screenScale(stats);
        AispectTouchNormalizationProfile.Result covarianceMatrix = covarianceMatrix(stats, patch);
        AispectTouchNormalizationProfile.Result sessionBaseline = sessionBaseline(stats, screenScale);
        AispectTouchNormalizationProfile.Diagnostics diagnostics = diagnostics(stats, screenScale);
        return new AispectTouchNormalizationProfile(selectedScheme, screenScale, covarianceMatrix, sessionBaseline, diagnostics, patch);
    }

    public void observe(AispectTouchNormalizationProfile profile) {
        if (profile == null || profile.screenScale == null || !profile.screenScale.valid) {
            return;
        }
        majorBaseline.observe(profile.screenScale.majorValue);
        minorBaseline.observe(profile.screenScale.minorValue);
        areaBaseline.observe(profile.screenScale.areaValue);
        references.add(new TouchReference(profile.screenScale.majorValue, profile.screenScale.minorValue, profile.screenScale.areaValue));
        while (references.size() > Math.max(1, config.baselineMaxSamples)) {
            references.remove(0);
        }
    }

    private TouchStats collectStats(List<AispectTouchFrame> frames, AispectContactPatch patch) {
        TouchStats stats = new TouchStats();
        if (frames != null) {
            for (AispectTouchFrame frame : frames) {
                // 优先使用 Android 原始触摸帧，最大轴长用于降低单帧抖动对接触面积的影响。
                stats.frameCount += 1;
                stats.screenDiagonalPx = Math.max(stats.screenDiagonalPx, diagonal(frame.width, frame.height));
                stats.screenAreaPx = Math.max(stats.screenAreaPx, Math.max(1, frame.width) * (double) Math.max(1, frame.height));
                double major = firstPositive(frame.touchMajor, frame.toolMajor);
                double minor = firstPositive(frame.touchMinor, frame.toolMinor);
                if (major > 0) {
                    stats.majorMaxPx = Math.max(stats.majorMaxPx, major);
                    stats.source = frame.touchMajor > 0 ? "touch_major" : "tool_major";
                    stats.hasRadius = true;
                }
                if (minor > 0) {
                    stats.minorMaxPx = Math.max(stats.minorMaxPx, minor);
                    stats.hasShape = true;
                }
                if (Math.abs(frame.orientation) > 0.0001f) {
                    stats.orientationRad = frame.orientation;
                    stats.hasShape = true;
                }
                if (frame.pressure > 0f || frame.size > 0f) {
                    stats.hasPressureOrSize = true;
                }
            }
        }
        if (stats.majorMaxPx <= 0 && patch != null && patch.rawMajorPx > 0) {
            // 若原始帧缺少半径字段，使用已构建的接触区域兜底，但仍保留来源标记供训练判断。
            stats.majorMaxPx = patch.rawMajorPx;
            stats.minorMaxPx = patch.rawMinorPx;
            stats.orientationRad = patch.rawOrientationRad;
            stats.source = patch.source;
            stats.hasRadius = true;
            stats.hasShape = patch.rawMinorPx > 0;
        }
        if (stats.minorMaxPx <= 0 && stats.majorMaxPx > 0) {
            stats.minorMaxPx = stats.majorMaxPx;
        }
        if (stats.screenDiagonalPx <= 0) {
            stats.screenDiagonalPx = 1;
        }
        if (stats.screenAreaPx <= 0) {
            stats.screenAreaPx = 1;
        }
        return stats;
    }

    private AispectTouchNormalizationProfile.Result screenScale(TouchStats stats) {
        // 屏幕比例画像只消除分辨率差异，不表达接触椭圆方向，适合作为可解释基线。
        boolean valid = stats.majorMaxPx > 0;
        double major = valid ? stats.majorMaxPx / stats.screenDiagonalPx : 0;
        double minor = valid ? stats.minorMaxPx / stats.screenDiagonalPx : 0;
        double area = valid ? ellipseArea(stats.majorMaxPx, stats.minorMaxPx) / stats.screenAreaPx : 0;
        return new AispectTouchNormalizationProfile.Result(
                AispectTouchNormalizationProfile.Scheme.SCREEN_SCALE,
                valid,
                false,
                clamp01(major),
                clamp01(minor),
                clamp01(area),
                normalizeOrientation(stats.orientationRad),
                confidence(valid, stats.frameCount, false),
                stats.majorMaxPx,
                stats.minorMaxPx,
                0,
                stats.source
        );
    }

    private AispectTouchNormalizationProfile.Result covarianceMatrix(TouchStats stats, AispectContactPatch patch) {
        // 协方差画像把接触区域当成概率分布，统一表达圆形半径、椭圆长短轴和方向。
        boolean valid = patch != null && patch.valid;
        boolean usedFallback = patch != null && patch.defaultApplied;
        double major = 0;
        double minor = 0;
        double area = 0;
        double orientation = 0;
        if (patch != null) {
            major = Math.sqrt(Math.max(0, patch.covariance11));
            minor = Math.sqrt(Math.max(0, patch.covariance22));
            area = Math.PI * Math.sqrt(Math.max(0, patch.covariance11 * patch.covariance22 - patch.covariance12 * patch.covariance12));
            orientation = normalizeOrientation(patch.orientationRad);
        }
        return new AispectTouchNormalizationProfile.Result(
                AispectTouchNormalizationProfile.Scheme.COVARIANCE_MATRIX,
                valid,
                usedFallback,
                clamp01(major),
                clamp01(minor),
                clamp01(area),
                orientation,
                confidence(valid, stats.frameCount, usedFallback),
                stats.majorMaxPx,
                stats.minorMaxPx,
                0,
                patch == null ? "unavailable" : patch.source
        );
    }

    private AispectTouchNormalizationProfile.Result sessionBaseline(
            TouchStats stats,
            AispectTouchNormalizationProfile.Result screenScale
    ) {
        // 会话基线只比较当前用户本轮历史均值，适合观察相对变重，但冷启动阶段必须标记为兜底。
        boolean baselineReady = majorBaseline.count() >= config.baselineWarmupCount && areaBaseline.count() >= config.baselineWarmupCount;
        boolean valid = screenScale.valid && baselineReady;
        double major = valid ? screenScale.majorValue / Math.max(majorBaseline.mean(), 0.000001) : 0;
        double minor = valid ? screenScale.minorValue / Math.max(minorBaseline.mean(), 0.000001) : 0;
        double area = valid ? screenScale.areaValue / Math.max(areaBaseline.mean(), 0.000001) : 0;
        return new AispectTouchNormalizationProfile.Result(
                AispectTouchNormalizationProfile.Scheme.SESSION_BASELINE,
                valid,
                !baselineReady,
                clampRatio(major),
                clampRatio(minor),
                clampRatio(area),
                screenScale.orientationValue,
                confidence(valid, stats.frameCount, !baselineReady),
                stats.majorMaxPx,
                stats.minorMaxPx,
                Math.min(majorBaseline.count(), areaBaseline.count()),
                stats.source
        );
    }

    private AispectTouchNormalizationProfile.Diagnostics diagnostics(
            TouchStats stats,
            AispectTouchNormalizationProfile.Result screenScale
    ) {
        // 诊断值只用于分析数据质量和设备差异，不应被当成力度真值直接解释。
        int referenceCount = references.size();
        boolean percentileReady = screenScale.valid && referenceCount >= Math.max(2, config.baselineWarmupCount);
        boolean offsetReady = percentileReady && majorBaseline.count() >= Math.max(2, config.baselineWarmupCount);
        double majorPercentile = percentileReady ? percentile(screenScale.majorValue, 0) : 0;
        double minorPercentile = percentileReady ? percentile(screenScale.minorValue, 1) : 0;
        double areaPercentile = percentileReady ? percentile(screenScale.areaValue, 2) : 0;
        double majorOffset = offsetReady ? safeOffset(screenScale.majorValue, majorBaseline.mean()) : 0;
        double minorOffset = offsetReady ? safeOffset(screenScale.minorValue, minorBaseline.mean()) : 0;
        double areaOffset = offsetReady ? safeOffset(screenScale.areaValue, areaBaseline.mean()) : 0;
        double quality = fieldQuality(stats, screenScale);
        return new AispectTouchNormalizationProfile.Diagnostics(
                percentileReady,
                majorPercentile,
                minorPercentile,
                areaPercentile,
                offsetReady,
                majorOffset,
                minorOffset,
                areaOffset,
                referenceCount,
                quality,
                qualityLevel(quality),
                stats.hasRadius,
                stats.hasShape,
                stats.hasPressureOrSize,
                stats.frameCount,
                stats.source
        );
    }

    private static double ellipseArea(double majorPx, double minorPx) {
        return Math.PI * Math.max(0, majorPx * 0.5) * Math.max(0, minorPx * 0.5);
    }

    private static double diagonal(int width, int height) {
        return Math.sqrt(width * (double) width + height * (double) height);
    }

    private static double firstPositive(float first, float second) {
        if (Float.isFinite(first) && first > 0f) {
            return first;
        }
        if (Float.isFinite(second) && second > 0f) {
            return second;
        }
        return 0;
    }

    private static double normalizeOrientation(double value) {
        if (!Double.isFinite(value)) {
            return 0;
        }
        double normalized = value / Math.PI;
        if (normalized < -1) {
            return -1;
        }
        if (normalized > 1) {
            return 1;
        }
        return normalized;
    }

    private static double confidence(boolean valid, int frameCount, boolean fallback) {
        // 置信度表示接触字段质量：无效为零，真实字段随帧数升高，兜底字段降权，不能等同于按压力度。
        if (!valid) {
            return 0;
        }
        double frameScore = Math.min(1.0, frameCount / 4.0);
        return fallback ? frameScore * 0.35 : frameScore;
    }

    private double percentile(double value, int index) {
        if (!Double.isFinite(value) || references.isEmpty()) {
            return 0;
        }
        int lessOrEqual = 0;
        for (int i = 0; i < references.size(); i++) {
            TouchReference reference = references.get(i);
            double referenceValue;
            if (index == 1) {
                referenceValue = reference.minor;
            } else if (index == 2) {
                referenceValue = reference.area;
            } else {
                referenceValue = reference.major;
            }
            if (value >= referenceValue) {
                lessOrEqual += 1;
            }
        }
        return clamp01(lessOrEqual / (double) references.size());
    }

    private static double safeOffset(double value, double baseline) {
        if (!Double.isFinite(value) || !Double.isFinite(baseline) || baseline <= 0) {
            return 0;
        }
        return clampSigned((value - baseline) / baseline);
    }

    private static double fieldQuality(TouchStats stats, AispectTouchNormalizationProfile.Result screenScale) {
        // 字段质量拆开记录半径、形状、压力和帧数，避免单独依赖 confidence 造成流程偏置。
        if (!screenScale.valid || !stats.hasRadius) {
            return stats.hasPressureOrSize ? 0.2 : 0;
        }
        double quality = 0.55;
        if (stats.hasShape) {
            quality += 0.25;
        }
        if (stats.hasPressureOrSize) {
            quality += 0.1;
        }
        quality += Math.min(0.1, stats.frameCount / 40.0);
        return clamp01(quality);
    }

    private static String qualityLevel(double quality) {
        if (quality >= 0.75) {
            return "high";
        }
        if (quality >= 0.4) {
            return "medium";
        }
        if (quality > 0) {
            return "low";
        }
        return "unavailable";
    }

    private static double clampSigned(double value) {
        if (!Double.isFinite(value)) {
            return 0;
        }
        if (value > 4) {
            return 4;
        }
        if (value < -4) {
            return -4;
        }
        return value;
    }

    private static double clampRatio(double value) {
        if (!Double.isFinite(value) || value < 0) {
            return 0;
        }
        if (value > 4) {
            return 4;
        }
        return value;
    }

    private static double clamp01(double value) {
        if (!Double.isFinite(value) || value < 0) {
            return 0;
        }
        if (value > 1) {
            return 1;
        }
        return value;
    }
}
