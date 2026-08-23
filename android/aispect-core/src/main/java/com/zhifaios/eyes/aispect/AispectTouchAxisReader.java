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
        final boolean touchMajorObserved;
        final boolean touchMinorObserved;
        final boolean toolMajorObserved;
        final boolean toolMinorObserved;
        final boolean pressureObserved;
        final boolean sizeObserved;
        final boolean orientationObserved;
        final boolean touchMajorSynthesizedOrUnknown;
        final boolean touchMinorSynthesizedOrUnknown;
        final boolean toolMajorSynthesizedOrUnknown;
        final boolean toolMinorSynthesizedOrUnknown;
        final boolean pressureSynthesizedOrUnknown;
        final boolean sizeSynthesizedOrUnknown;
        final boolean orientationSynthesizedOrUnknown;
        final AispectTouchInputMetadata inputMetadata;

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
        boolean hasTouchMajorRange = hasAxis(device, MotionEvent.AXIS_TOUCH_MAJOR);
        boolean hasTouchMinorRange = hasAxis(device, MotionEvent.AXIS_TOUCH_MINOR);
        boolean hasToolMajorRange = hasAxis(device, MotionEvent.AXIS_TOOL_MAJOR);
        boolean hasToolMinorRange = hasAxis(device, MotionEvent.AXIS_TOOL_MINOR);
        boolean hasPressureRange = hasAxis(device, MotionEvent.AXIS_PRESSURE);
        boolean hasSizeRange = hasAxis(device, MotionEvent.AXIS_SIZE);
        boolean hasOrientationRange = hasAxis(device, MotionEvent.AXIS_ORIENTATION);
        boolean touchMajorObserved = positive(touchMajor);
        boolean touchMinorObserved = positive(touchMinor);
        boolean toolMajorObserved = positive(toolMajor);
        boolean toolMinorObserved = positive(toolMinor);
        boolean pressureObserved = Float.isFinite(pressure);
        boolean sizeObserved = Float.isFinite(size);
        boolean orientationObserved = Float.isFinite(orientation);
        return new Sample(
                sanitize(pressure),
                sanitize(size),
                sanitize(touchMajor),
                sanitize(touchMinor),
                sanitize(toolMajor),
                sanitize(toolMinor),
                sanitize(orientation),
                hasTouchMajorRange,
                hasTouchMinorRange,
                hasToolMajorRange,
                hasToolMinorRange,
                hasPressureRange,
                hasSizeRange,
                hasOrientationRange,
                touchMajorObserved,
                touchMinorObserved,
                toolMajorObserved,
                toolMinorObserved,
                pressureObserved,
                sizeObserved,
                orientationObserved,
                touchMajorObserved && !hasTouchMajorRange,
                touchMinorObserved && (!hasTouchMinorRange || (touchMinor == touchMajor && !hasTouchMinorRange)),
                toolMajorObserved && !hasToolMajorRange,
                toolMinorObserved && !hasToolMinorRange,
                pressureObserved && !hasPressureRange,
                sizeObserved && !hasSizeRange,
                orientationObserved && !hasOrientationRange,
                AispectTouchInputMetadata.from(event)
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
