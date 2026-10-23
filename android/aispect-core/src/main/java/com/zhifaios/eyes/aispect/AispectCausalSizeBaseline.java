package com.zhifaios.eyes.aispect;

import android.content.Context;
import android.content.SharedPreferences;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 维护单台设备的因果接触尺度基线。
 *
 * 只吸收已经结束的触摸，每次触摸先取有效 size 的中位数，避免长触摸因帧多而
 * 过度影响设备基线。当前触摸和未来触摸永远不会进入分母。
 */
final class AispectCausalSizeBaseline {
    static final int MINIMUM_TOUCHES = 8;
    static final int MAXIMUM_TOUCHES = 64;
    static final double LOG_RATIO_CLIP = 4.0;

    private static final String PREFERENCES = "aispect_causal_size_baseline";
    private static final String DEVICE_KEY = "device_key";
    private static final String VALUES_KEY = "completed_touch_medians";

    private final SharedPreferences preferences;
    private final String deviceKey;
    private final ArrayDeque<Double> completedTouchMedians = new ArrayDeque<>();
    private long lastObservedTouchTimestamp;

    AispectCausalSizeBaseline(Context context, String deviceKey) {
        this.preferences = context == null
                ? null
                : context.getApplicationContext().getSharedPreferences(
                PREFERENCES,
                Context.MODE_PRIVATE
        );
        this.deviceKey = deviceKey == null ? "" : deviceKey.trim();
        load();
    }

    AispectCausalSizeBaseline(String deviceKey, List<Double> values) {
        this.preferences = null;
        this.deviceKey = deviceKey == null ? "" : deviceKey.trim();
        for (Double value : values) {
            append(value);
        }
    }

    double currentMedian() {
        if (completedTouchMedians.size() < MINIMUM_TOUCHES) {
            return Double.NaN;
        }
        List<Double> ordered = new ArrayList<>(completedTouchMedians);
        Collections.sort(ordered);
        int middle = ordered.size() / 2;
        return ordered.size() % 2 == 0
                ? (ordered.get(middle - 1) + ordered.get(middle)) * 0.5
                : ordered.get(middle);
    }

    double logRatio(float size) {
        double median = currentMedian();
        if (!(size > 0.0f) || !Float.isFinite(size) || !(median > 0.0)) {
            return Double.NaN;
        }
        return clip(Math.log(size / median));
    }

    void observeCompletedTouch(List<AispectTouchFrame> frames) {
        if (frames == null || frames.isEmpty()) {
            return;
        }
        AispectTouchFrame first = frames.get(0);
        if (first == null || first.eventElapsedRealtimeNanos <= 0L) {
            return;
        }
        if (first.eventElapsedRealtimeNanos == lastObservedTouchTimestamp) {
            return;
        }
        double median = touchMedian(frames);
        if (!(median > 0.0)) {
            return;
        }
        lastObservedTouchTimestamp = first.eventElapsedRealtimeNanos;
        append(median);
        persist();
    }

    int completedTouchCount() {
        return completedTouchMedians.size();
    }

    void clear() {
        completedTouchMedians.clear();
        lastObservedTouchTimestamp = 0L;
        if (preferences != null) {
            preferences.edit().clear().apply();
        }
    }

    private static double touchMedian(List<AispectTouchFrame> frames) {
        if (frames == null) {
            return Double.NaN;
        }
        ArrayList<Double> values = new ArrayList<>();
        for (AispectTouchFrame frame : frames) {
            if (frame != null
                    && Float.isFinite(frame.size)
                    && frame.size > 0.0f) {
                values.add((double) frame.size);
            }
        }
        if (values.isEmpty()) {
            return Double.NaN;
        }
        Collections.sort(values);
        int middle = values.size() / 2;
        return values.size() % 2 == 0
                ? (values.get(middle - 1) + values.get(middle)) * 0.5
                : values.get(middle);
    }

    private void append(double value) {
        if (!Double.isFinite(value) || value <= 0.0) {
            return;
        }
        completedTouchMedians.addLast(value);
        while (completedTouchMedians.size() > MAXIMUM_TOUCHES) {
            completedTouchMedians.removeFirst();
        }
    }

    private void load() {
        if (preferences == null
                || deviceKey.isEmpty()
                || !deviceKey.equals(preferences.getString(DEVICE_KEY, ""))) {
            return;
        }
        String serialized = preferences.getString(VALUES_KEY, "");
        if (serialized == null || serialized.isEmpty()) {
            return;
        }
        for (String part : serialized.split(",")) {
            try {
                append(Double.parseDouble(part));
            } catch (NumberFormatException ignored) {
                // 中文注释：损坏的持久化项直接忽略，基线会在后续触摸中重建。
            }
        }
    }

    private void persist() {
        if (preferences == null || deviceKey.isEmpty()) {
            return;
        }
        StringBuilder serialized = new StringBuilder();
        for (Double value : completedTouchMedians) {
            if (serialized.length() > 0) {
                serialized.append(',');
            }
            serialized.append(value);
        }
        preferences.edit()
                .putString(DEVICE_KEY, deviceKey)
                .putString(VALUES_KEY, serialized.toString())
                .apply();
    }

    private static double clip(double value) {
        if (value < -LOG_RATIO_CLIP) {
            return -LOG_RATIO_CLIP;
        }
        if (value > LOG_RATIO_CLIP) {
            return LOG_RATIO_CLIP;
        }
        return value;
    }

}
