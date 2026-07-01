package com.zhifaios.eyes.aispect;

import android.os.Build;

import org.json.JSONException;
import org.json.JSONObject;

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

    public void observe(AispectTouchFrame frame) {
        observedFrames += 1;
        hasMajor = hasMajor || positive(frame.touchMajor);
        hasMinor = hasMinor || positive(frame.touchMinor);
        hasPressure = hasPressure || frame.pressure > 0f;
        hasSize = hasSize || frame.size > 0f;
        hasOrientation = hasOrientation || Math.abs(frame.orientation) > 0.0001f;
        hasToolMajor = hasToolMajor || positive(frame.toolMajor);
        hasToolMinor = hasToolMinor || positive(frame.toolMinor);
        hasTouchMajorRange = hasTouchMajorRange || frame.hasTouchMajorRange;
        hasTouchMinorRange = hasTouchMinorRange || frame.hasTouchMinorRange;
        hasToolMajorRange = hasToolMajorRange || frame.hasToolMajorRange;
        hasToolMinorRange = hasToolMinorRange || frame.hasToolMinorRange;
        hasPressureRange = hasPressureRange || frame.hasPressureRange;
        hasSizeRange = hasSizeRange || frame.hasSizeRange;
        hasOrientationRange = hasOrientationRange || frame.hasOrientationRange;
    }

    public Snapshot snapshot() {
        Level level;
        if (hasMajor && (hasMinor || hasPressure || hasSize || hasOrientation)) {
            level = Level.A;
        } else if (hasMajor) {
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
                observedFrames,
                Build.MANUFACTURER + " " + Build.MODEL,
                Build.FINGERPRINT
        );
    }

    private static boolean positive(float value) {
        return Float.isFinite(value) && value > 0f;
    }
}
