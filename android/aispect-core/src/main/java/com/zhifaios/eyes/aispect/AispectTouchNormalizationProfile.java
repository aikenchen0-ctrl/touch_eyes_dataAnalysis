package com.zhifaios.eyes.aispect;

import org.json.JSONException;
import org.json.JSONObject;

public final class AispectTouchNormalizationProfile {
    public enum Scheme {
        SCREEN_SCALE("screen_scale"),
        COVARIANCE_MATRIX("covariance_matrix"),
        SESSION_BASELINE("session_baseline");

        private final String key;

        Scheme(String key) {
            this.key = key;
        }

        public String key() {
            return key;
        }
    }

    public static final class Result {
        public final Scheme scheme;
        public final boolean valid;
        public final boolean usedFallback;
        public final double majorValue;
        public final double minorValue;
        public final double areaValue;
        public final double orientationValue;
        public final double confidence;
        public final double rawMajorPx;
        public final double rawMinorPx;
        public final int baselineCount;
        public final String source;

        Result(
                Scheme scheme,
                boolean valid,
                boolean usedFallback,
                double majorValue,
                double minorValue,
                double areaValue,
                double orientationValue,
                double confidence,
                double rawMajorPx,
                double rawMinorPx,
                int baselineCount,
                String source
        ) {
            this.scheme = scheme;
            this.valid = valid;
            this.usedFallback = usedFallback;
            this.majorValue = majorValue;
            this.minorValue = minorValue;
            this.areaValue = areaValue;
            this.orientationValue = orientationValue;
            this.confidence = confidence;
            this.rawMajorPx = rawMajorPx;
            this.rawMinorPx = rawMinorPx;
            this.baselineCount = baselineCount;
            this.source = source;
        }

        JSONObject toJson() throws JSONException {
            JSONObject json = new JSONObject();
            json.put("scheme", scheme.key());
            json.put("valid", valid);
            json.put("usedFallback", usedFallback);
            json.put("majorValue", majorValue);
            json.put("minorValue", minorValue);
            json.put("areaValue", areaValue);
            json.put("orientationValue", orientationValue);
            json.put("confidence", confidence);
            json.put("rawMajorPx", rawMajorPx);
            json.put("rawMinorPx", rawMinorPx);
            json.put("baselineCount", baselineCount);
            json.put("source", source);
            return json;
        }
    }

    public static final class Diagnostics {
        public final boolean percentileReady;
        public final double majorPercentile;
        public final double minorPercentile;
        public final double areaPercentile;
        public final boolean offsetReady;
        public final double majorOffset;
        public final double minorOffset;
        public final double areaOffset;
        public final int referenceCount;
        public final double fieldQualityScore;
        public final String fieldQualityLevel;
        public final boolean hasRadius;
        public final boolean hasShape;
        public final boolean hasPressureOrSize;
        public final int frameCount;
        public final String source;

        Diagnostics(
                boolean percentileReady,
                double majorPercentile,
                double minorPercentile,
                double areaPercentile,
                boolean offsetReady,
                double majorOffset,
                double minorOffset,
                double areaOffset,
                int referenceCount,
                double fieldQualityScore,
                String fieldQualityLevel,
                boolean hasRadius,
                boolean hasShape,
                boolean hasPressureOrSize,
                int frameCount,
                String source
        ) {
            this.percentileReady = percentileReady;
            this.majorPercentile = majorPercentile;
            this.minorPercentile = minorPercentile;
            this.areaPercentile = areaPercentile;
            this.offsetReady = offsetReady;
            this.majorOffset = majorOffset;
            this.minorOffset = minorOffset;
            this.areaOffset = areaOffset;
            this.referenceCount = referenceCount;
            this.fieldQualityScore = fieldQualityScore;
            this.fieldQualityLevel = fieldQualityLevel;
            this.hasRadius = hasRadius;
            this.hasShape = hasShape;
            this.hasPressureOrSize = hasPressureOrSize;
            this.frameCount = frameCount;
            this.source = source;
        }

        JSONObject toJson() throws JSONException {
            JSONObject json = new JSONObject();
            JSONObject percentile = new JSONObject();
            percentile.put("ready", percentileReady);
            percentile.put("major", majorPercentile);
            percentile.put("minor", minorPercentile);
            percentile.put("area", areaPercentile);
            percentile.put("referenceCount", referenceCount);
            json.put("devicePercentile", percentile);

            JSONObject offset = new JSONObject();
            offset.put("ready", offsetReady);
            offset.put("major", majorOffset);
            offset.put("minor", minorOffset);
            offset.put("area", areaOffset);
            offset.put("referenceCount", referenceCount);
            json.put("shortTermOffset", offset);

            JSONObject quality = new JSONObject();
            quality.put("score", fieldQualityScore);
            quality.put("level", fieldQualityLevel);
            quality.put("hasRadius", hasRadius);
            quality.put("hasShape", hasShape);
            quality.put("hasPressureOrSize", hasPressureOrSize);
            quality.put("frameCount", frameCount);
            quality.put("source", source);
            json.put("fieldQuality", quality);
            return json;
        }
    }

    public final Scheme selectedScheme;
    public final Result selectedResult;
    public final Result screenScale;
    public final Result covarianceMatrix;
    public final Result sessionBaseline;
    public final Diagnostics diagnostics;
    public final AispectContactPatch contactPatch;

    AispectTouchNormalizationProfile(
            Scheme selectedScheme,
            Result screenScale,
            Result covarianceMatrix,
            Result sessionBaseline,
            Diagnostics diagnostics,
            AispectContactPatch contactPatch
    ) {
        this.selectedScheme = selectedScheme;
        this.screenScale = screenScale;
        this.covarianceMatrix = covarianceMatrix;
        this.sessionBaseline = sessionBaseline;
        this.diagnostics = diagnostics;
        this.contactPatch = contactPatch;
        if (selectedScheme == Scheme.SESSION_BASELINE) {
            this.selectedResult = sessionBaseline;
        } else if (selectedScheme == Scheme.COVARIANCE_MATRIX) {
            this.selectedResult = covarianceMatrix;
        } else {
            this.selectedResult = screenScale;
        }
    }

    JSONObject toJson() throws JSONException {
        JSONObject json = new JSONObject();
        json.put("selectedScheme", selectedScheme.key());
        json.put("selectedResult", selectedResult.toJson());
        json.put("screenScale", screenScale.toJson());
        json.put("covarianceMatrix", covarianceMatrix.toJson());
        json.put("sessionBaseline", sessionBaseline.toJson());
        if (diagnostics != null) {
            json.put("diagnostics", diagnostics.toJson());
        }
        JSONObject schemes = new JSONObject();
        schemes.put("screenScale", screenScale.toJson());
        schemes.put("covarianceMatrix", covarianceMatrix.toJson());
        schemes.put("sessionBaseline", sessionBaseline.toJson());
        json.put("schemes", schemes);
        if (contactPatch != null) {
            json.put("contactPatch", contactPatch.toJson());
        }
        return json;
    }
}
