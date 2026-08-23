package com.zhifaios.eyes.aispect;

import android.os.Build;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.List;
import java.util.Locale;

public final class AispectDeviceCapabilityProfiler {
    public enum Level {
        A,
        B,
        C
    }

    public static final class Snapshot {
        public final Level level;
        public final boolean hasMajor;
        public final boolean hasMinor;
        public final boolean hasPressure;
        public final boolean hasSize;
        public final boolean hasOrientation;
        public final boolean hasToolMajor;
        public final boolean hasToolMinor;
        public final boolean hasTouchMajorRange;
        public final boolean hasTouchMinorRange;
        public final boolean hasToolMajorRange;
        public final boolean hasToolMinorRange;
        public final boolean hasPressureRange;
        public final boolean hasSizeRange;
        public final boolean hasOrientationRange;
        public final boolean touchMajorInformative;
        public final boolean touchMinorInformative;
        public final boolean pressureInformative;
        public final boolean sizeInformative;
        public final boolean orientationInformative;
        public final boolean touchAxesSemanticallyValid;
        public final boolean touchAxesDistinct;
        public final boolean toolAxesIndependent;
        public final int observedFrames;
        public final String deviceName;
        public final String fingerprint;

        Snapshot(
                Level level,
                boolean hasMajor,
                boolean hasMinor,
                boolean hasPressure,
                boolean hasSize,
                boolean hasOrientation,
                boolean hasToolMajor,
                boolean hasToolMinor,
                boolean hasTouchMajorRange,
                boolean hasTouchMinorRange,
                boolean hasToolMajorRange,
                boolean hasToolMinorRange,
                boolean hasPressureRange,
                boolean hasSizeRange,
                boolean hasOrientationRange,
                boolean touchMajorInformative,
                boolean touchMinorInformative,
                boolean pressureInformative,
                boolean sizeInformative,
                boolean orientationInformative,
                boolean touchAxesSemanticallyValid,
                boolean touchAxesDistinct,
                boolean toolAxesIndependent,
                int observedFrames,
                String deviceName,
                String fingerprint
        ) {
            this.level = level;
            this.hasMajor = hasMajor;
            this.hasMinor = hasMinor;
            this.hasPressure = hasPressure;
            this.hasSize = hasSize;
            this.hasOrientation = hasOrientation;
            this.hasToolMajor = hasToolMajor;
            this.hasToolMinor = hasToolMinor;
            this.hasTouchMajorRange = hasTouchMajorRange;
            this.hasTouchMinorRange = hasTouchMinorRange;
            this.hasToolMajorRange = hasToolMajorRange;
            this.hasToolMinorRange = hasToolMinorRange;
            this.hasPressureRange = hasPressureRange;
            this.hasSizeRange = hasSizeRange;
            this.hasOrientationRange = hasOrientationRange;
            this.touchMajorInformative = touchMajorInformative;
            this.touchMinorInformative = touchMinorInformative;
            this.pressureInformative = pressureInformative;
            this.sizeInformative = sizeInformative;
            this.orientationInformative = orientationInformative;
            this.touchAxesSemanticallyValid = touchAxesSemanticallyValid;
            this.touchAxesDistinct = touchAxesDistinct;
            this.toolAxesIndependent = toolAxesIndependent;
            this.observedFrames = observedFrames;
            this.deviceName = deviceName;
            this.fingerprint = fingerprint;
        }

        public String summaryKey() {
            return "level_" + level.name().toLowerCase(Locale.US);
        }

        JSONObject toJson() throws JSONException {
            JSONObject json = new JSONObject();
            json.put("level", level.name());
            json.put("hasMajor", hasMajor);
            json.put("hasMinor", hasMinor);
            json.put("hasPressure", hasPressure);
            json.put("hasSize", hasSize);
            json.put("hasOrientation", hasOrientation);
            json.put("hasToolMajor", hasToolMajor);
            json.put("hasToolMinor", hasToolMinor);
            json.put("hasTouchMajorRange", hasTouchMajorRange);
            json.put("hasTouchMinorRange", hasTouchMinorRange);
            json.put("hasToolMajorRange", hasToolMajorRange);
            json.put("hasToolMinorRange", hasToolMinorRange);
            json.put("hasPressureRange", hasPressureRange);
            json.put("hasSizeRange", hasSizeRange);
            json.put("hasOrientationRange", hasOrientationRange);
            json.put("touchMajorInformative", touchMajorInformative);
            json.put("touchMinorInformative", touchMinorInformative);
            json.put("pressureInformative", pressureInformative);
            json.put("sizeInformative", sizeInformative);
            json.put("orientationInformative", orientationInformative);
            json.put("touchAxesSemanticallyValid", touchAxesSemanticallyValid);
            json.put("touchAxesDistinct", touchAxesDistinct);
            json.put("toolAxesIndependent", toolAxesIndependent);
            json.put("observedFrames", observedFrames);
            json.put("deviceName", deviceName);
            json.put("fingerprint", fingerprint);
            return json;
        }
    }

    private int observedFrames;
    private boolean hasMajor;
    private boolean hasMinor;
    private boolean hasPressure;
    private boolean hasSize;
    private boolean hasOrientation;
    private boolean hasToolMajor;
    private boolean hasToolMinor;
    private boolean hasTouchMajorRange;
    private boolean hasTouchMinorRange;
    private boolean hasToolMajorRange;
    private boolean hasToolMinorRange;
    private boolean hasPressureRange;
    private boolean hasSizeRange;
    private boolean hasOrientationRange;
    private float majorMin = Float.POSITIVE_INFINITY;
    private float majorMax = Float.NEGATIVE_INFINITY;
    private float minorMin = Float.POSITIVE_INFINITY;
    private float minorMax = Float.NEGATIVE_INFINITY;
    private float pressureMin = Float.POSITIVE_INFINITY;
    private float pressureMax = Float.NEGATIVE_INFINITY;
    private float sizeMin = Float.POSITIVE_INFINITY;
    private float sizeMax = Float.NEGATIVE_INFINITY;
    private float orientationMin = Float.POSITIVE_INFINITY;
    private float orientationMax = Float.NEGATIVE_INFINITY;
    private int touchAxisPairCount;
    private int invertedTouchAxisPairCount;
    private int distinctTouchAxisPairCount;
    private int toolAxisComparisonCount;
    private int toolAxisAliasCount;

    public static Snapshot snapshotForFrames(List<AispectTouchFrame> frames) {
        AispectDeviceCapabilityProfiler profiler = new AispectDeviceCapabilityProfiler();
        if (frames != null) {
            for (AispectTouchFrame frame : frames) {
                if (frame != null) {
                    profiler.observe(frame);
                }
            }
        }
        return profiler.snapshot();
    }

    public void observe(AispectTouchFrame frame) {
        observedFrames += 1;
        hasMajor = hasMajor || positive(frame.touchMajor);
        hasMinor = hasMinor || positive(frame.touchMinor);
        hasPressure = hasPressure || frame.pressure > 0f;
        hasSize = hasSize || frame.size > 0f;
        hasOrientation = hasOrientation || frame.orientationObserved;
        hasToolMajor = hasToolMajor || positive(frame.toolMajor);
        hasToolMinor = hasToolMinor || positive(frame.toolMinor);
        hasTouchMajorRange = hasTouchMajorRange || frame.hasTouchMajorRange;
        hasTouchMinorRange = hasTouchMinorRange || frame.hasTouchMinorRange;
        hasToolMajorRange = hasToolMajorRange || frame.hasToolMajorRange;
        hasToolMinorRange = hasToolMinorRange || frame.hasToolMinorRange;
        hasPressureRange = hasPressureRange || frame.hasPressureRange;
        hasSizeRange = hasSizeRange || frame.hasSizeRange;
        hasOrientationRange = hasOrientationRange || frame.hasOrientationRange;
        updateRange(frame.touchMajor, true);
        updateRange(frame.touchMinor, false);
        updatePressure(frame.pressure);
        updateSize(frame.size);
        updateOrientation(frame.orientation);
        observeAxisSemantics(frame);
    }

    public Snapshot snapshot() {
        boolean touchAxesSemanticallyValid = touchAxisPairCount > 0
                && invertedTouchAxisPairCount == 0;
        boolean touchAxesDistinct = distinctTouchAxisPairCount > 0;
        boolean toolAxesIndependent = toolAxisComparisonCount > 0
                && toolAxisAliasCount < toolAxisComparisonCount;
        boolean touchMajorInformative = hasTouchMajorRange
                && informative(majorMin, majorMax)
                && touchAxesSemanticallyValid;
        boolean touchMinorInformative = hasTouchMinorRange
                && informative(minorMin, minorMax)
                && touchAxesSemanticallyValid
                && touchAxesDistinct;
        boolean pressureInformative = hasPressureRange && informative(pressureMin, pressureMax);
        boolean sizeInformative = hasSizeRange && informative(sizeMin, sizeMax);
        boolean orientationInformative = hasOrientationRange && informative(orientationMin, orientationMax);
        Level level;
        if (touchMajorInformative
                && (touchMinorInformative || pressureInformative || orientationInformative)) {
            level = Level.A;
        } else if (touchMajorInformative) {
            level = Level.B;
        } else {
            level = Level.C;
        }
        return new Snapshot(
                level,
                hasMajor,
                hasMinor,
                hasPressure,
                hasSize,
                hasOrientation,
                hasToolMajor,
                hasToolMinor,
                hasTouchMajorRange,
                hasTouchMinorRange,
                hasToolMajorRange,
                hasToolMinorRange,
                hasPressureRange,
                hasSizeRange,
                hasOrientationRange,
                touchMajorInformative,
                touchMinorInformative,
                pressureInformative,
                sizeInformative,
                orientationInformative,
                touchAxesSemanticallyValid,
                touchAxesDistinct,
                toolAxesIndependent,
                observedFrames,
                Build.MANUFACTURER + " " + Build.MODEL,
                Build.FINGERPRINT
        );
    }

    private static boolean positive(float value) {
        return Float.isFinite(value) && value > 0f;
    }

    private void updateRange(float value, boolean major) {
        if (!positive(value)) {
            return;
        }
        if (major) {
            majorMin = Math.min(majorMin, value);
            majorMax = Math.max(majorMax, value);
        } else {
            minorMin = Math.min(minorMin, value);
            minorMax = Math.max(minorMax, value);
        }
    }

    private void updatePressure(float value) {
        if (!Float.isFinite(value)) {
            return;
        }
        pressureMin = Math.min(pressureMin, value);
        pressureMax = Math.max(pressureMax, value);
    }

    private void updateSize(float value) {
        if (!Float.isFinite(value)) {
            return;
        }
        sizeMin = Math.min(sizeMin, value);
        sizeMax = Math.max(sizeMax, value);
    }

    private void updateOrientation(float value) {
        if (!Float.isFinite(value)) {
            return;
        }
        orientationMin = Math.min(orientationMin, value);
        orientationMax = Math.max(orientationMax, value);
    }

    private static boolean informative(float min, float max) {
        return min != Float.POSITIVE_INFINITY
                && max != Float.NEGATIVE_INFINITY
                && max - min > 0.0001f;
    }

    private void observeAxisSemantics(AispectTouchFrame frame) {
        if (positive(frame.touchMajor) && positive(frame.touchMinor)) {
            touchAxisPairCount += 1;
            if (frame.touchMajor + 0.0001f < frame.touchMinor) {
                invertedTouchAxisPairCount += 1;
            }
            if (!same(frame.touchMajor, frame.touchMinor)) {
                distinctTouchAxisPairCount += 1;
            }
        }
        if (positive(frame.touchMajor)
                && positive(frame.touchMinor)
                && positive(frame.toolMajor)
                && positive(frame.toolMinor)) {
            toolAxisComparisonCount += 1;
            if (same(frame.touchMajor, frame.toolMajor)
                    && same(frame.touchMinor, frame.toolMinor)) {
                toolAxisAliasCount += 1;
            }
        }
    }

    private static boolean same(float left, float right) {
        return Math.abs(left - right) <= 0.0001f;
    }
}
