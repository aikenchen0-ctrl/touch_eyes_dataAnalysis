package com.zhifaios.eyes.aispect;

import android.view.MotionEvent;

import org.json.JSONException;
import org.json.JSONObject;

public final class AispectTouchFrame {
    public final double relativeTimestampSeconds;
    public final long eventTimeMillis;
    public final long eventElapsedRealtimeNanos;
    public final long receivedElapsedRealtimeNanos;
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
    public final int pointerId;
    public final int actionMasked;
    public final int actionIndex;
    public final boolean hasTouchMajorRange;
    public final boolean hasTouchMinorRange;
    public final boolean hasToolMajorRange;
    public final boolean hasToolMinorRange;
    public final boolean hasPressureRange;
    public final boolean hasSizeRange;
    public final boolean hasOrientationRange;
    public final boolean touchMajorObserved;
    public final boolean touchMinorObserved;
    public final boolean toolMajorObserved;
    public final boolean toolMinorObserved;
    public final boolean pressureObserved;
    public final boolean sizeObserved;
    public final boolean orientationObserved;
    public final boolean touchMajorSynthesizedOrUnknown;
    public final boolean touchMinorSynthesizedOrUnknown;
    public final boolean toolMajorSynthesizedOrUnknown;
    public final boolean toolMinorSynthesizedOrUnknown;
    public final boolean pressureSynthesizedOrUnknown;
    public final boolean sizeSynthesizedOrUnknown;
    public final boolean orientationSynthesizedOrUnknown;
    final AispectTouchInputMetadata inputMetadata;

    private AispectTouchFrame(
            double relativeTimestampSeconds,
            long eventTimeMillis,
            long eventElapsedRealtimeNanos,
            long receivedElapsedRealtimeNanos,
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
            int pointerId,
            int actionMasked,
            int actionIndex,
            boolean hasTouchMajorRange,
            boolean hasTouchMinorRange,
            boolean hasToolMajorRange,
            boolean hasToolMinorRange,
            boolean hasPressureRange,
            boolean hasSizeRange,
            boolean hasOrientationRange,
            boolean touchMajorObserved,
            boolean touchMinorObserved,
            boolean toolMajorObserved,
            boolean toolMinorObserved,
            boolean pressureObserved,
            boolean sizeObserved,
            boolean orientationObserved,
            boolean touchMajorSynthesizedOrUnknown,
            boolean touchMinorSynthesizedOrUnknown,
            boolean toolMajorSynthesizedOrUnknown,
            boolean toolMinorSynthesizedOrUnknown,
            boolean pressureSynthesizedOrUnknown,
            boolean sizeSynthesizedOrUnknown,
            boolean orientationSynthesizedOrUnknown,
            AispectTouchInputMetadata inputMetadata
    ) {
        this.relativeTimestampSeconds = relativeTimestampSeconds;
        this.eventTimeMillis = eventTimeMillis;
        this.eventElapsedRealtimeNanos = eventElapsedRealtimeNanos;
        this.receivedElapsedRealtimeNanos = receivedElapsedRealtimeNanos;
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
        this.pointerId = pointerId;
        this.actionMasked = actionMasked;
        this.actionIndex = actionIndex;
        this.hasTouchMajorRange = hasTouchMajorRange;
        this.hasTouchMinorRange = hasTouchMinorRange;
        this.hasToolMajorRange = hasToolMajorRange;
        this.hasToolMinorRange = hasToolMinorRange;
        this.hasPressureRange = hasPressureRange;
        this.hasSizeRange = hasSizeRange;
        this.hasOrientationRange = hasOrientationRange;
        this.touchMajorObserved = touchMajorObserved;
        this.touchMinorObserved = touchMinorObserved;
        this.toolMajorObserved = toolMajorObserved;
        this.toolMinorObserved = toolMinorObserved;
        this.pressureObserved = pressureObserved;
        this.sizeObserved = sizeObserved;
        this.orientationObserved = orientationObserved;
        this.touchMajorSynthesizedOrUnknown = touchMajorSynthesizedOrUnknown;
        this.touchMinorSynthesizedOrUnknown = touchMinorSynthesizedOrUnknown;
        this.toolMajorSynthesizedOrUnknown = toolMajorSynthesizedOrUnknown;
        this.toolMinorSynthesizedOrUnknown = toolMinorSynthesizedOrUnknown;
        this.pressureSynthesizedOrUnknown = pressureSynthesizedOrUnknown;
        this.sizeSynthesizedOrUnknown = sizeSynthesizedOrUnknown;
        this.orientationSynthesizedOrUnknown = orientationSynthesizedOrUnknown;
        this.inputMetadata = inputMetadata;
    }

    public static AispectTouchFrame fromCurrent(MotionEvent event, int pointerIndex, long downEventTime, int width, int height) {
        return makeFrame(event, pointerIndex, -1, downEventTime, width, height, 0L, 0L, 0L);
    }

    public static AispectTouchFrame fromHistorical(MotionEvent event, int pointerIndex, int historyIndex, long downEventTime, int width, int height) {
        return makeFrame(event, pointerIndex, historyIndex, downEventTime, width, height, 0L, 0L, 0L);
    }

    public static AispectTouchFrame fromCurrent(
            MotionEvent event,
            int pointerIndex,
            long downEventTime,
            int width,
            int height,
            long downEventElapsedRealtimeNanos,
            long receivedUptimeMillis,
            long receivedElapsedRealtimeNanos
    ) {
        return makeFrame(
                event,
                pointerIndex,
                -1,
                downEventTime,
                width,
                height,
                downEventElapsedRealtimeNanos,
                receivedUptimeMillis,
                receivedElapsedRealtimeNanos
        );
    }

    public static AispectTouchFrame fromHistorical(
            MotionEvent event,
            int pointerIndex,
            int historyIndex,
            long downEventTime,
            int width,
            int height,
            long downEventElapsedRealtimeNanos,
            long receivedUptimeMillis,
            long receivedElapsedRealtimeNanos
    ) {
        return makeFrame(
                event,
                pointerIndex,
                historyIndex,
                downEventTime,
                width,
                height,
                downEventElapsedRealtimeNanos,
                receivedUptimeMillis,
                receivedElapsedRealtimeNanos
        );
    }

    JSONObject toJson(int frameIndex) throws JSONException {
        JSONObject json = new JSONObject();
        json.put("frameIndex", frameIndex);
        json.put("relativeTimestamp", relativeTimestampSeconds);
        json.put("eventTimeMillis", eventTimeMillis);
        json.put("eventElapsedRealtimeNanos", eventElapsedRealtimeNanos);
        json.put("receivedElapsedRealtimeNanos", receivedElapsedRealtimeNanos);
        json.put("viewWidthPx", width);
        json.put("viewHeightPx", height);
        json.put("x", x);
        json.put("y", y);
        json.put("viewWidthPx", width);
        json.put("viewHeightPx", height);
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
        json.put("pointerId", pointerId);
        json.put("actionMasked", actionMasked);
        json.put("actionIndex", actionIndex);
        JSONObject axes = new JSONObject();
        axes.put("hasTouchMajorRange", hasTouchMajorRange);
        axes.put("hasTouchMinorRange", hasTouchMinorRange);
        axes.put("hasToolMajorRange", hasToolMajorRange);
        axes.put("hasToolMinorRange", hasToolMinorRange);
        axes.put("hasPressureRange", hasPressureRange);
        axes.put("hasSizeRange", hasSizeRange);
        axes.put("hasOrientationRange", hasOrientationRange);
        axes.put("touchMajorObserved", touchMajorObserved);
        axes.put("touchMinorObserved", touchMinorObserved);
        axes.put("toolMajorObserved", toolMajorObserved);
        axes.put("toolMinorObserved", toolMinorObserved);
        axes.put("pressureObserved", pressureObserved);
        axes.put("sizeObserved", sizeObserved);
        axes.put("orientationObserved", orientationObserved);
        axes.put("touchMajorSynthesizedOrUnknown", touchMajorSynthesizedOrUnknown);
        axes.put("touchMinorSynthesizedOrUnknown", touchMinorSynthesizedOrUnknown);
        axes.put("toolMajorSynthesizedOrUnknown", toolMajorSynthesizedOrUnknown);
        axes.put("toolMinorSynthesizedOrUnknown", toolMinorSynthesizedOrUnknown);
        axes.put("pressureSynthesizedOrUnknown", pressureSynthesizedOrUnknown);
        axes.put("sizeSynthesizedOrUnknown", sizeSynthesizedOrUnknown);
        axes.put("orientationSynthesizedOrUnknown", orientationSynthesizedOrUnknown);
        axes.put("touchMajorSource", axisSource(touchMajorObserved, hasTouchMajorRange, touchMajorSynthesizedOrUnknown));
        axes.put("touchMinorSource", axisSource(touchMinorObserved, hasTouchMinorRange, touchMinorSynthesizedOrUnknown));
        axes.put("toolMajorSource", axisSource(toolMajorObserved, hasToolMajorRange, toolMajorSynthesizedOrUnknown));
        axes.put("toolMinorSource", axisSource(toolMinorObserved, hasToolMinorRange, toolMinorSynthesizedOrUnknown));
        axes.put("pressureSource", axisSource(pressureObserved, hasPressureRange, pressureSynthesizedOrUnknown));
        axes.put("sizeSource", axisSource(sizeObserved, hasSizeRange, sizeSynthesizedOrUnknown));
        axes.put("orientationSource", axisSource(orientationObserved, hasOrientationRange, orientationSynthesizedOrUnknown));
        json.put("axisAvailability", axes);
        if (inputMetadata != null) {
            json.put("inputDeviceMetadata", inputMetadata.toJson());
        }
        json.put("contactPatch", AispectContactPatch.fromFrame(this).toJson());
        return json;
    }

    static AispectTouchFrame forTesting(
            long receivedElapsedRealtimeNanos,
            float x,
            float y,
            int width,
            int height,
            float pressure,
            float size,
            float touchMajor,
            float touchMinor,
            float orientation,
            int pointerCount,
            boolean hasOrientationRange
    ) {
        return forTesting(
                0L,
                receivedElapsedRealtimeNanos,
                x,
                y,
                width,
                height,
                pressure,
                size,
                touchMajor,
                touchMinor,
                orientation,
                pointerCount,
                hasOrientationRange
        );
    }

    static AispectTouchFrame forTestingAxes(
            long receivedElapsedRealtimeNanos,
            float touchMajor,
            float touchMinor,
            float toolMajor,
            float toolMinor,
            float pressure,
            float size
    ) {
        return new AispectTouchFrame(
                0.0,
                0L,
                0L,
                receivedElapsedRealtimeNanos,
                100f,
                100f,
                1000,
                1000,
                pressure,
                size,
                touchMajor,
                touchMinor,
                toolMajor,
                toolMinor,
                0f,
                MotionEvent.TOOL_TYPE_FINGER,
                1,
                0,
                MotionEvent.ACTION_MOVE,
                0,
                true,
                true,
                true,
                true,
                false,
                true,
                false,
                true,
                true,
                true,
                true,
                true,
                true,
                true,
                false,
                false,
                false,
                false,
                true,
                false,
                true,
                null
        );
    }

    static AispectTouchFrame forTesting(
            long eventElapsedRealtimeNanos,
            long receivedElapsedRealtimeNanos,
            float x,
            float y,
            int width,
            int height,
            float pressure,
            float size,
            float touchMajor,
            float touchMinor,
            float orientation,
            int pointerCount,
            boolean hasOrientationRange
    ) {
        return new AispectTouchFrame(
                0.0,
                0L,
                eventElapsedRealtimeNanos,
                receivedElapsedRealtimeNanos,
                x,
                y,
                width,
                height,
                pressure,
                size,
                touchMajor,
                touchMinor,
                0f,
                0f,
                orientation,
                MotionEvent.TOOL_TYPE_FINGER,
                pointerCount,
                0,
                MotionEvent.ACTION_MOVE,
                0,
                false,
                false,
                false,
                false,
                false,
                false,
                hasOrientationRange,
                touchMajor > 0f,
                touchMinor > 0f,
                false,
                false,
                Float.isFinite(pressure),
                Float.isFinite(size),
                Float.isFinite(orientation),
                true,
                true,
                false,
                false,
                true,
                true,
                !hasOrientationRange,
                null
        );
    }

    private static AispectTouchFrame makeFrame(
            MotionEvent event,
            int pointerIndex,
            int historyIndex,
            long downEventTime,
            int width,
            int height,
            long downEventElapsedRealtimeNanos,
            long receivedUptimeMillis,
            long receivedElapsedRealtimeNanos
    ) {
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
                downEventElapsedRealtimeNanos > 0L
                        ? AispectCaptureClock.touchEventUptimeMillisToElapsedRealtimeNanos(
                        eventTime,
                        downEventTime,
                        downEventElapsedRealtimeNanos
                )
                        : receivedElapsedRealtimeNanos > 0L
                        ? AispectCaptureClock.uptimeMillisToElapsedRealtimeNanos(
                        eventTime,
                        receivedUptimeMillis,
                        receivedElapsedRealtimeNanos
                )
                        : 0L,
                receivedElapsedRealtimeNanos,
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
                event.getPointerId(safePointerIndex),
                event.getActionMasked(),
                event.getActionIndex(),
                axes.hasTouchMajorRange,
                axes.hasTouchMinorRange,
                axes.hasToolMajorRange,
                axes.hasToolMinorRange,
                axes.hasPressureRange,
                axes.hasSizeRange,
                axes.hasOrientationRange,
                axes.touchMajorObserved,
                axes.touchMinorObserved,
                axes.toolMajorObserved,
                axes.toolMinorObserved,
                axes.pressureObserved,
                axes.sizeObserved,
                axes.orientationObserved,
                axes.touchMajorSynthesizedOrUnknown,
                axes.touchMinorSynthesizedOrUnknown,
                axes.toolMajorSynthesizedOrUnknown,
                axes.toolMinorSynthesizedOrUnknown,
                axes.pressureSynthesizedOrUnknown,
                axes.sizeSynthesizedOrUnknown,
                axes.orientationSynthesizedOrUnknown,
                axes.inputMetadata
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

    private static String axisSource(boolean observed, boolean declared, boolean synthesizedOrUnknown) {
        if (!observed) {
            return "unavailable";
        }
        if (synthesizedOrUnknown) {
            return "framework_value_undeclared_range";
        }
        return declared ? "android_motion_event_declared_range_unverified" : "android_motion_event";
    }
}
