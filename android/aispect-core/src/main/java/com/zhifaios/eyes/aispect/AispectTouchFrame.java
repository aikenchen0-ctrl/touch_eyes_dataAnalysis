package com.zhifaios.eyes.aispect;

import android.view.MotionEvent;

import org.json.JSONException;
import org.json.JSONObject;

public final class AispectTouchFrame {
    public final double relativeTimestampSeconds;
    public final long eventTimeMillis;
    public final float x;
    public final float y;
    public final int width;
    public final int height;
    public final float xNorm;
    public final float yNorm;
    public final float pressure;
    public final float size;
    public final float touchMajor;
    public final float touchMinor;
    public final float toolMajor;
    public final float toolMinor;
    public final float orientation;
    public final int toolType;
    public final int pointerCount;
    public final boolean hasTouchMajorRange;
    public final boolean hasTouchMinorRange;
    public final boolean hasToolMajorRange;
    public final boolean hasToolMinorRange;
    public final boolean hasPressureRange;
    public final boolean hasSizeRange;
    public final boolean hasOrientationRange;

    private AispectTouchFrame(
            double relativeTimestampSeconds,
            long eventTimeMillis,
            float x,
            float y,
            int width,
            int height,
            float pressure,
            float size,
            float touchMajor,
            float touchMinor,
            float toolMajor,
            float toolMinor,
            float orientation,
            int toolType,
            int pointerCount,
            boolean hasTouchMajorRange,
            boolean hasTouchMinorRange,
            boolean hasToolMajorRange,
            boolean hasToolMinorRange,
            boolean hasPressureRange,
            boolean hasSizeRange,
            boolean hasOrientationRange
    ) {
        this.relativeTimestampSeconds = relativeTimestampSeconds;
        this.eventTimeMillis = eventTimeMillis;
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.xNorm = normalize(x, width);
        this.yNorm = normalize(y, height);
        this.pressure = pressure;
        this.size = size;
        this.touchMajor = touchMajor;
        this.touchMinor = touchMinor;
        this.toolMajor = toolMajor;
        this.toolMinor = toolMinor;
        this.orientation = orientation;
        this.toolType = toolType;
        this.pointerCount = pointerCount;
        this.hasTouchMajorRange = hasTouchMajorRange;
        this.hasTouchMinorRange = hasTouchMinorRange;
        this.hasToolMajorRange = hasToolMajorRange;
        this.hasToolMinorRange = hasToolMinorRange;
        this.hasPressureRange = hasPressureRange;
        this.hasSizeRange = hasSizeRange;
        this.hasOrientationRange = hasOrientationRange;
    }

    public static AispectTouchFrame fromCurrent(MotionEvent event, int pointerIndex, long downEventTime, int width, int height) {
        return makeFrame(event, pointerIndex, -1, downEventTime, width, height);
    }

    public static AispectTouchFrame fromHistorical(MotionEvent event, int pointerIndex, int historyIndex, long downEventTime, int width, int height) {
        return makeFrame(event, pointerIndex, historyIndex, downEventTime, width, height);
    }

    JSONObject toJson(int frameIndex) throws JSONException {
        JSONObject json = new JSONObject();
        json.put("frameIndex", frameIndex);
        json.put("relativeTimestamp", relativeTimestampSeconds);
        json.put("eventTimeMillis", eventTimeMillis);
        json.put("x", x);
        json.put("y", y);
        json.put("x_norm", xNorm);
        json.put("y_norm", yNorm);
        json.put("pressure", pressure);
        json.put("size", size);
        json.put("touchMajor", touchMajor);
        json.put("touchMinor", touchMinor);
        json.put("toolMajor", toolMajor);
        json.put("toolMinor", toolMinor);
        json.put("orientation", orientation);
        json.put("toolType", toolType);
        json.put("pointerCount", pointerCount);
        JSONObject axes = new JSONObject();
        axes.put("hasTouchMajorRange", hasTouchMajorRange);
        axes.put("hasTouchMinorRange", hasTouchMinorRange);
        axes.put("hasToolMajorRange", hasToolMajorRange);
        axes.put("hasToolMinorRange", hasToolMinorRange);
        axes.put("hasPressureRange", hasPressureRange);
        axes.put("hasSizeRange", hasSizeRange);
        axes.put("hasOrientationRange", hasOrientationRange);
        json.put("axisAvailability", axes);
        json.put("contactPatch", AispectContactPatch.fromFrame(this).toJson());
        return json;
    }

    private static AispectTouchFrame makeFrame(MotionEvent event, int pointerIndex, int historyIndex, long downEventTime, int width, int height) {
        boolean isHistorical = historyIndex >= 0;
        long eventTime = isHistorical ? event.getHistoricalEventTime(historyIndex) : event.getEventTime();
        float x = isHistorical ? event.getHistoricalX(pointerIndex, historyIndex) : event.getX(pointerIndex);
        float y = isHistorical ? event.getHistoricalY(pointerIndex, historyIndex) : event.getY(pointerIndex);
        AispectTouchAxisReader.Sample axes = isHistorical
                ? AispectTouchAxisReader.historical(event, pointerIndex, historyIndex)
                : AispectTouchAxisReader.current(event, pointerIndex);
        int safePointerIndex = Math.max(0, Math.min(pointerIndex, event.getPointerCount() - 1));
        return new AispectTouchFrame(
                (eventTime - downEventTime) / 1000.0,
                eventTime,
                x,
                y,
                width,
                height,
                axes.pressure,
                axes.size,
                axes.touchMajor,
                axes.touchMinor,
                axes.toolMajor,
                axes.toolMinor,
                axes.orientation,
                event.getToolType(safePointerIndex),
                event.getPointerCount(),
                axes.hasTouchMajorRange,
                axes.hasTouchMinorRange,
                axes.hasToolMajorRange,
                axes.hasToolMinorRange,
                axes.hasPressureRange,
                axes.hasSizeRange,
                axes.hasOrientationRange
        );
    }

    private static float normalize(float value, int denominator) {
        if (denominator <= 0) {
            return 0f;
        }
        float normalized = value / denominator;
        if (normalized < 0f) {
            return 0f;
        }
        if (normalized > 1f) {
            return 1f;
        }
        return normalized;
    }
}
