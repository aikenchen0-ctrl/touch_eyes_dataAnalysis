package com.zhifaios.eyes.aispect;

import android.os.Build;
import android.view.InputDevice;
import android.view.MotionEvent;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

/** 中文注释：保存触摸输入设备的原始 MotionRange 画像，不把声明能力当成实际可用能力。 */
final class AispectTouchInputMetadata {
    static final class AxisRange {
        final int axis;
        final int source;
        final float min;
        final float max;
        final float range;
        final float flat;
        final float fuzz;
        final float resolution;

        AxisRange(InputDevice.MotionRange motionRange) {
            axis = motionRange.getAxis();
            source = motionRange.getSource();
            min = motionRange.getMin();
            max = motionRange.getMax();
            range = motionRange.getRange();
            flat = motionRange.getFlat();
            fuzz = motionRange.getFuzz();
            resolution = motionRange.getResolution();
        }

        JSONObject toJson() throws JSONException {
            JSONObject json = new JSONObject();
            json.put("axis", axis);
            json.put("source", source);
            json.put("sourceName", sourceName(source));
            json.put("min", min);
            json.put("max", max);
            json.put("range", range);
            json.put("flat", flat);
            json.put("fuzz", fuzz);
            json.put("resolution", resolution);
            return json;
        }
    }

    final int deviceId;
    final int source;
    final String sourceName;
    final String descriptor;
    final String name;
    final int vendorId;
    final int productId;
    final List<AxisRange> motionRanges;

    private AispectTouchInputMetadata(
            int deviceId,
            int source,
            String sourceName,
            String descriptor,
            String name,
            int vendorId,
            int productId,
            List<AxisRange> motionRanges
    ) {
        this.deviceId = deviceId;
        this.source = source;
        this.sourceName = sourceName;
        this.descriptor = descriptor;
        this.name = name;
        this.vendorId = vendorId;
        this.productId = productId;
        this.motionRanges = motionRanges;
    }

    static AispectTouchInputMetadata from(MotionEvent event) {
        InputDevice device = event == null ? null : event.getDevice();
        if (device == null) {
            return null;
        }
        ArrayList<AxisRange> ranges = new ArrayList<>();
        List<InputDevice.MotionRange> sourceRanges = device.getMotionRanges();
        if (sourceRanges != null) {
            for (InputDevice.MotionRange range : sourceRanges) {
                if (range != null) {
                    ranges.add(new AxisRange(range));
                }
            }
        }
        String descriptor = "";
        if (Build.VERSION.SDK_INT >= 16) {
            descriptor = safe(device.getDescriptor());
        }
        return new AispectTouchInputMetadata(
                device.getId(),
                event.getSource(),
                sourceName(event.getSource()),
                descriptor,
                safe(device.getName()),
                device.getVendorId(),
                device.getProductId(),
                ranges
        );
    }

    JSONObject toJson() throws JSONException {
        JSONObject json = new JSONObject();
        json.put("deviceId", deviceId);
        json.put("source", source);
        json.put("sourceName", sourceName);
        json.put("descriptor", descriptor);
        json.put("name", name);
        json.put("vendorId", vendorId);
        json.put("productId", productId);
        JSONArray ranges = new JSONArray();
        for (AxisRange range : motionRanges) {
            ranges.put(range.toJson());
        }
        json.put("motionRanges", ranges);
        return json;
    }

    private static String safe(String value) {
        return value == null ? "" : value;
    }

    /** 中文注释：Android SDK 没有公开稳定的 InputDevice source 名称 API，保留无损十六进制值。 */
    private static String sourceName(int source) {
        return String.format(java.util.Locale.US, "0x%08x", source);
    }
}
