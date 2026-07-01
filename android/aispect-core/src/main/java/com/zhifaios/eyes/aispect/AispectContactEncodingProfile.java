package com.zhifaios.eyes.aispect;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.List;

public final class AispectContactEncodingProfile {
    public enum Scheme {
        GAUSSIAN_MAP("gaussian_map"),
        MULTI_CHANNEL("multi_channel"),
        COVARIANCE_MATRIX("covariance_matrix");

        private final String key;

        Scheme(String key) {
            this.key = key;
        }

        public String key() {
            return key;
        }
    }

    public static final int MAP_SIZE = 9;
    public static final double DEFAULT_SIGMA_PX = 12.0;
    public static final double GRID_RADIUS_PX = 48.0;

    public static final class Result {
        public final Scheme scheme;
        public final boolean valid;
        public final boolean usedFallback;
        public final double validMask;
        public final String source;
        public final double centerXNorm;
        public final double centerYNorm;
        public final double rawMajorPx;
        public final double rawMinorPx;
        public final double sigmaXPx;
        public final double sigmaYPx;
        public final double orientationRad;
        public final double covariance11;
        public final double covariance12;
        public final double covariance22;
        public final double[] heatmap;
        public final double[] centerHeatmap;

        Result(
                Scheme scheme,
                boolean valid,
                boolean usedFallback,
                double validMask,
                String source,
                double centerXNorm,
                double centerYNorm,
                double rawMajorPx,
                double rawMinorPx,
                double sigmaXPx,
                double sigmaYPx,
                double orientationRad,
                double covariance11,
                double covariance12,
                double covariance22,
                double[] heatmap,
                double[] centerHeatmap
        ) {
            this.scheme = scheme;
            this.valid = valid;
            this.usedFallback = usedFallback;
            this.validMask = validMask;
            this.source = source;
            this.centerXNorm = centerXNorm;
            this.centerYNorm = centerYNorm;
            this.rawMajorPx = rawMajorPx;
            this.rawMinorPx = rawMinorPx;
            this.sigmaXPx = sigmaXPx;
            this.sigmaYPx = sigmaYPx;
            this.orientationRad = orientationRad;
            this.covariance11 = covariance11;
            this.covariance12 = covariance12;
            this.covariance22 = covariance22;
            this.heatmap = heatmap;
            this.centerHeatmap = centerHeatmap;
        }

        JSONObject toJson() throws JSONException {
            JSONObject json = new JSONObject();
            json.put("scheme", scheme.key());
            json.put("valid", valid);
            json.put("usedFallback", usedFallback);
            json.put("validMask", validMask);
            json.put("source", source);
            json.put("centerXNorm", centerXNorm);
            json.put("centerYNorm", centerYNorm);
            json.put("rawMajorPx", rawMajorPx);
            json.put("rawMinorPx", rawMinorPx);
            json.put("sigmaXPx", sigmaXPx);
            json.put("sigmaYPx", sigmaYPx);
            json.put("orientationRad", orientationRad);
            json.put("orientationDeg", Math.toDegrees(orientationRad));
            if (scheme == Scheme.GAUSSIAN_MAP) {
                json.put("mapSize", MAP_SIZE);
                json.put("gridRadiusPx", GRID_RADIUS_PX);
                json.put("touchContactMap", numberArray(heatmap));
            } else if (scheme == Scheme.MULTI_CHANNEL) {
                json.put("mapSize", MAP_SIZE);
                json.put("gridRadiusPx", GRID_RADIUS_PX);
                json.put("centerHeatmap", numberArray(centerHeatmap));
                JSONObject channels = new JSONObject();
                channels.put("centerXNorm", centerXNorm);
                channels.put("centerYNorm", centerYNorm);
                channels.put("majorPx", rawMajorPx);
                channels.put("minorPx", rawMinorPx);
                channels.put("orientationRad", orientationRad);
                channels.put("orientationDeg", Math.toDegrees(orientationRad));
                channels.put("validMask", validMask);
                json.put("channels", channels);
            } else {
                JSONObject covariance = new JSONObject();
                covariance.put("sigmaXNorm2", covariance11);
                covariance.put("sigmaXYNorm", covariance12);
                covariance.put("sigmaYNorm2", covariance22);
                json.put("covariance", covariance);
                JSONObject features = new JSONObject();
                features.put("xNorm", centerXNorm);
                features.put("yNorm", centerYNorm);
                features.put("cov_11", covariance11);
                features.put("cov_12", covariance12);
                features.put("cov_22", covariance22);
                features.put("validMask", validMask);
                json.put("features", features);
            }
            return json;
        }
    }

    public final Scheme selectedScheme;
    public final Result selectedResult;
    public final Result gaussianMap;
    public final Result multiChannel;
    public final Result covarianceMatrix;

    private AispectContactEncodingProfile(
            Scheme selectedScheme,
            Result gaussianMap,
            Result multiChannel,
            Result covarianceMatrix
    ) {
        this.selectedScheme = selectedScheme;
        this.gaussianMap = gaussianMap;
        this.multiChannel = multiChannel;
        this.covarianceMatrix = covarianceMatrix;
        if (selectedScheme == Scheme.GAUSSIAN_MAP) {
            this.selectedResult = gaussianMap;
        } else if (selectedScheme == Scheme.MULTI_CHANNEL) {
            this.selectedResult = multiChannel;
        } else {
            this.selectedResult = covarianceMatrix;
        }
    }

    public static AispectContactEncodingProfile evaluate(
            List<AispectTouchFrame> frames,
            AispectContactPatch patch,
            Scheme selectedScheme
    ) {
        ContactStats stats = collectStats(frames, patch);
        Result gaussianMap = buildGaussianMap(stats);
        Result multiChannel = buildMultiChannel(stats);
        Result covarianceMatrix = buildCovarianceMatrix(stats);
        Scheme safeScheme = selectedScheme == null ? Scheme.GAUSSIAN_MAP : selectedScheme;
        return new AispectContactEncodingProfile(safeScheme, gaussianMap, multiChannel, covarianceMatrix);
    }

    JSONObject toJson() throws JSONException {
        JSONObject json = new JSONObject();
        json.put("schemaVersion", 1);
        json.put("selectedScheme", selectedScheme.key());
        json.put("selectedResult", selectedResult.toJson());
        json.put("gaussianMap", gaussianMap.toJson());
        json.put("multiChannel", multiChannel.toJson());
        json.put("covarianceMatrix", covarianceMatrix.toJson());
        return json;
    }

    private static Result buildGaussianMap(ContactStats stats) {
        return buildResult(Scheme.GAUSSIAN_MAP, stats, gaussianMap(stats), centerHeatmap());
    }

    private static Result buildMultiChannel(ContactStats stats) {
        return buildResult(Scheme.MULTI_CHANNEL, stats, gaussianMap(stats), centerHeatmap());
    }

    private static Result buildCovarianceMatrix(ContactStats stats) {
        return buildResult(Scheme.COVARIANCE_MATRIX, stats, gaussianMap(stats), centerHeatmap());
    }

    private static Result buildResult(Scheme scheme, ContactStats stats, double[] heatmap, double[] centerHeatmap) {
        double[] covariance = covariance(stats);
        return new Result(
                scheme,
                stats.valid,
                stats.usedFallback,
                stats.valid ? 1.0 : 0.0,
                stats.source,
                clamp01(stats.centerXNorm),
                clamp01(stats.centerYNorm),
                stats.rawMajorPx,
                stats.rawMinorPx,
                stats.sigmaXPx,
                stats.sigmaYPx,
                stats.orientationRad,
                covariance[0],
                covariance[1],
                covariance[2],
                heatmap,
                centerHeatmap
        );
    }

    private static ContactStats collectStats(List<AispectTouchFrame> frames, AispectContactPatch patch) {
        ContactStats stats = new ContactStats();
        if (frames != null) {
            for (AispectTouchFrame frame : frames) {
                stats.count += 1;
                stats.width = Math.max(stats.width, frame.width);
                stats.height = Math.max(stats.height, frame.height);
                stats.centerXNorm += frame.xNorm;
                stats.centerYNorm += frame.yNorm;
                double major = firstPositive(frame.touchMajor, frame.toolMajor);
                double minor = firstPositive(frame.touchMinor, frame.toolMinor);
                if (major > 0) {
                    stats.rawMajorPx = Math.max(stats.rawMajorPx, major);
                    stats.source = frame.touchMajor > 0 ? "ellipse_major_minor" : "tool_major_minor";
                    stats.valid = true;
                }
                if (minor > 0) {
                    stats.rawMinorPx = Math.max(stats.rawMinorPx, minor);
                    stats.hasMinor = true;
                }
                if (Math.abs(frame.orientation) > 0.0001f) {
                    stats.orientationRad = frame.orientation;
                }
            }
        }
        if (stats.count > 0) {
            stats.centerXNorm /= stats.count;
            stats.centerYNorm /= stats.count;
        } else if (patch != null) {
            stats.centerXNorm = patch.xNorm;
            stats.centerYNorm = patch.yNorm;
        }
        if (stats.rawMajorPx <= 0 && patch != null && patch.rawMajorPx > 0) {
            stats.rawMajorPx = patch.rawMajorPx;
            stats.rawMinorPx = patch.rawMinorPx;
            stats.orientationRad = patch.rawOrientationRad;
            stats.source = patch.source;
            stats.valid = patch.valid;
            stats.hasMinor = patch.rawMinorPx > 0;
        }
        if (stats.rawMajorPx <= 0) {
            stats.usedFallback = true;
            stats.source = "unknown_default";
            stats.rawMajorPx = DEFAULT_SIGMA_PX * 2.0;
            stats.rawMinorPx = DEFAULT_SIGMA_PX * 2.0;
        } else if (stats.rawMinorPx <= 0) {
            stats.rawMinorPx = stats.rawMajorPx;
            stats.source = "radius_or_major_circle";
        }
        if (stats.valid && stats.hasMinor) {
            stats.sigmaXPx = Math.max(1.0, stats.rawMajorPx * 0.5);
            stats.sigmaYPx = Math.max(1.0, stats.rawMinorPx * 0.5);
        } else if (stats.valid) {
            stats.sigmaXPx = Math.max(1.0, stats.rawMajorPx * 0.5);
            stats.sigmaYPx = stats.sigmaXPx;
        } else {
            stats.sigmaXPx = DEFAULT_SIGMA_PX;
            stats.sigmaYPx = DEFAULT_SIGMA_PX;
        }
        if (stats.width <= 0) {
            stats.width = 1;
        }
        if (stats.height <= 0) {
            stats.height = 1;
        }
        return stats;
    }

    private static double[] gaussianMap(ContactStats stats) {
        double[] output = new double[MAP_SIZE * MAP_SIZE];
        double cos = Math.cos(stats.orientationRad);
        double sin = Math.sin(stats.orientationRad);
        int center = MAP_SIZE / 2;
        int index = 0;
        for (int y = 0; y < MAP_SIZE; y++) {
            for (int x = 0; x < MAP_SIZE; x++) {
                double dx = ((x - center) / (double) center) * GRID_RADIUS_PX;
                double dy = ((y - center) / (double) center) * GRID_RADIUS_PX;
                double rotatedX = cos * dx + sin * dy;
                double rotatedY = -sin * dx + cos * dy;
                double exponent = -0.5 * (
                        (rotatedX * rotatedX) / Math.max(1.0, stats.sigmaXPx * stats.sigmaXPx)
                                + (rotatedY * rotatedY) / Math.max(1.0, stats.sigmaYPx * stats.sigmaYPx)
                );
                output[index] = Math.exp(exponent);
                index += 1;
            }
        }
        return output;
    }

    private static double[] centerHeatmap() {
        double[] output = new double[MAP_SIZE * MAP_SIZE];
        output[(MAP_SIZE * MAP_SIZE) / 2] = 1.0;
        return output;
    }

    private static double[] covariance(ContactStats stats) {
        double sigmaXNorm = stats.sigmaXPx / Math.max(1, stats.width);
        double sigmaYNorm = stats.sigmaYPx / Math.max(1, stats.height);
        double cos = Math.cos(stats.orientationRad);
        double sin = Math.sin(stats.orientationRad);
        double varianceX = sigmaXNorm * sigmaXNorm;
        double varianceY = sigmaYNorm * sigmaYNorm;
        double cov11 = cos * cos * varianceX + sin * sin * varianceY;
        double cov22 = sin * sin * varianceX + cos * cos * varianceY;
        double cov12 = sin * cos * (varianceX - varianceY);
        return new double[]{cov11, cov12, cov22};
    }

    private static JSONArray numberArray(double[] values) throws JSONException {
        JSONArray array = new JSONArray();
        if (values != null) {
            for (double value : values) {
                array.put(value);
            }
        }
        return array;
    }

    private static double firstPositive(double first, double second) {
        if (Double.isFinite(first) && first > 0) {
            return first;
        }
        if (Double.isFinite(second) && second > 0) {
            return second;
        }
        return 0;
    }

    private static double clamp01(double value) {
        if (value < 0) {
            return 0;
        }
        if (value > 1) {
            return 1;
        }
        return value;
    }

    private static final class ContactStats {
        int count;
        int width;
        int height;
        boolean valid;
        boolean usedFallback;
        boolean hasMinor;
        double centerXNorm;
        double centerYNorm;
        double rawMajorPx;
        double rawMinorPx;
        double sigmaXPx;
        double sigmaYPx;
        double orientationRad;
        String source = "unavailable";
    }
}
