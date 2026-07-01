package com.zhifaios.eyes.aispect;

import android.view.InputDevice;
import android.view.MotionEvent;

final class AispectTouchAxisReader {
    static final class Sample {
        final float pressure;
        final float size;
        final float touchMajor;
        final float touchMinor;
        final float toolMajor;
        final float toolMinor;
        final float orientation;
        final boolean hasTouchMajorRange;
        final boolean hasTouchMinorRange;
        final boolean hasToolMajorRange;
        final boolean hasToolMinorRange;
        final boolean hasPressureRange;
        final boolean hasSizeRange;
        final boolean hasOrientationRange;

        Sample(
                float pressure,
                float size,
                float touchMajor,
                float touchMinor,
                float toolMajor,
                float toolMinor,
                float orientation,
                boolean hasTouchMajorRange,
                boolean hasTouchMinorRange,
                boolean hasToolMajorRange,
                boolean hasToolMinorRange,
                boolean hasPressureRange,
                boolean hasSizeRange,
                boolean hasOrientationRange
        ) {
            this.pressure = pressure;
            this.size = size;
            this.touchMajor = touchMajor;
            this.touchMinor = touchMinor;
            this.toolMajor = toolMajor;
            this.toolMinor = toolMinor;
            this.orientation = orientation;
            this.hasTouchMajorRange = hasTouchMajorRange;
            this.hasTouchMinorRange = hasTouchMinorRange;
            this.hasToolMajorRange = hasToolMajorRange;
            this.hasToolMinorRange = hasToolMinorRange;
            this.hasPressureRange = hasPressureRange;
            this.hasSizeRange = hasSizeRange;
            this.hasOrientationRange = hasOrientationRange;
        }
    }

    private AispectTouchAxisReader() {
    }

    static Sample current(MotionEvent event, int pointerIndex) {
        return read(event, pointerIndex, -1);
    }

    static Sample historical(MotionEvent event, int pointerIndex, int historyIndex) {
        return read(event, pointerIndex, historyIndex);
    }

    private static Sample read(MotionEvent event, int pointerIndex, int historyIndex) {
        boolean historical = historyIndex >= 0;
        float touchMajor = axisValue(event, pointerIndex, historyIndex, MotionEvent.AXIS_TOUCH_MAJOR);
        float touchMinor = axisValue(event, pointerIndex, historyIndex, MotionEvent.AXIS_TOUCH_MINOR);
        float toolMajor = axisValue(event, pointerIndex, historyIndex, MotionEvent.AXIS_TOOL_MAJOR);
        float toolMinor = axisValue(event, pointerIndex, historyIndex, MotionEvent.AXIS_TOOL_MINOR);
        float pressure = historical ? event.getHistoricalPressure(pointerIndex, historyIndex) : event.getPressure(pointerIndex);
        float size = historical ? event.getHistoricalSize(pointerIndex, historyIndex) : event.getSize(pointerIndex);
        float orientation = axisValue(event, pointerIndex, historyIndex, MotionEvent.AXIS_ORIENTATION);

        if (!positive(touchMajor)) {
            touchMajor = historical ? event.getHistoricalTouchMajor(pointerIndex, historyIndex) : event.getTouchMajor(pointerIndex);
        }
        if (!positive(touchMinor)) {
            touchMinor = historical ? event.getHistoricalTouchMinor(pointerIndex, historyIndex) : event.getTouchMinor(pointerIndex);
        }
        if (!positive(toolMajor)) {
            toolMajor = historical ? event.getHistoricalToolMajor(pointerIndex, historyIndex) : event.getToolMajor(pointerIndex);
        }
        if (!positive(toolMinor)) {
            toolMinor = historical ? event.getHistoricalToolMinor(pointerIndex, historyIndex) : event.getToolMinor(pointerIndex);
        }
        if (!positive(pressure)) {
            pressure = axisValue(event, pointerIndex, historyIndex, MotionEvent.AXIS_PRESSURE);
        }
        if (!positive(size)) {
            size = axisValue(event, pointerIndex, historyIndex, MotionEvent.AXIS_SIZE);
        }
        if (!finiteNonZero(orientation)) {
            orientation = historical ? event.getHistoricalOrientation(pointerIndex, historyIndex) : event.getOrientation(pointerIndex);
        }

        InputDevice device = event.getDevice();
        return new Sample(
                sanitize(pressure),
                sanitize(size),
                sanitize(touchMajor),
                sanitize(touchMinor),
                sanitize(toolMajor),
                sanitize(toolMinor),
                sanitize(orientation),
                hasAxis(device, MotionEvent.AXIS_TOUCH_MAJOR),
                hasAxis(device, MotionEvent.AXIS_TOUCH_MINOR),
                hasAxis(device, MotionEvent.AXIS_TOOL_MAJOR),
                hasAxis(device, MotionEvent.AXIS_TOOL_MINOR),
                hasAxis(device, MotionEvent.AXIS_PRESSURE),
                hasAxis(device, MotionEvent.AXIS_SIZE),
                hasAxis(device, MotionEvent.AXIS_ORIENTATION)
        );
    }

    private static float axisValue(MotionEvent event, int pointerIndex, int historyIndex, int axis) {
        if (historyIndex >= 0) {
            return event.getHistoricalAxisValue(axis, pointerIndex, historyIndex);
        }
        return event.getAxisValue(axis, pointerIndex);
    }

    private static boolean hasAxis(InputDevice device, int axis) {
        if (device == null) {
            return false;
        }
        return device.getMotionRange(axis, InputDevice.SOURCE_TOUCHSCREEN) != null
                || device.getMotionRange(axis) != null;
    }

    private static boolean positive(float value) {
        return Float.isFinite(value) && value > 0f;
    }

    private static boolean finiteNonZero(float value) {
        return Float.isFinite(value) && Math.abs(value) > 0.0001f;
    }

    private static float sanitize(float value) {
        if (!Float.isFinite(value)) {
            return 0f;
        }
        return value;
    }
}
