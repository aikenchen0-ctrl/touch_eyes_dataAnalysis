package com.zhifaios.eyes.aispect;

import org.json.JSONException;
import org.json.JSONObject;

/**
 * 保存触摸轴的真实量程元数据。
 * 不同设备的同名轴可能采用完全不同的单位，必须同时保留来源、量程和分辨率。
 */
final class AispectTouchAxisRange {
    final int source;
    final float min;
    final float max;
    final float flat;
    final float fuzz;
    final float resolution;

    AispectTouchAxisRange(
            int source,
            float min,
            float max,
            float flat,
            float fuzz,
            float resolution
    ) {
        this.source = source;
        this.min = min;
        this.max = max;
        this.flat = flat;
        this.fuzz = fuzz;
        this.resolution = resolution;
    }

    JSONObject toJson() throws JSONException {
        JSONObject json = new JSONObject();
        json.put("source", source);
        json.put("min", min);
        json.put("max", max);
        json.put("flat", flat);
        json.put("fuzz", fuzz);
        json.put("resolution", resolution);
        return json;
    }
}
