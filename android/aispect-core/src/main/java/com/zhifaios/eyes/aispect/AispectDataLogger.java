package com.zhifaios.eyes.aispect;

import android.content.Context;
import android.os.Environment;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Locale;

public final class AispectDataLogger {
    public static final class Config {
        public String datasetDirectoryName = "AispectAndroidDataset";
        public String appVersion = "unknown";
        public String appBuild = "unknown";
    }

    public static final class SavedRecord {
        public final File file;
        public final JSONObject json;

        SavedRecord(File file, JSONObject json) {
            this.file = file;
            this.json = json;
        }
    }

    private final Context context;
    private final Config config = new Config();

    public AispectDataLogger(Context context) {
        this.context = context.getApplicationContext();
    }

    public Config config() {
        return config;
    }

    public File datasetDirectory() {
        File documents = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS);
        if (documents == null) {
            documents = context.getFilesDir();
        }
        return new File(documents, config.datasetDirectoryName);
    }

    public SavedRecord writeSample(
            String id,
            long createdAtMillis,
            String collectionSessionId,
            long collectionSessionStartedAtMillis,
            String recordStatus,
            String rejectionReason,
            AispectModels.DatasetLabel label,
            String sourceTouchKind,
            Float liftOffset,
            List<AispectModels.TouchEvent> touchEvents,
            List<AispectTouchFrame> touchFrames,
            AispectContactPatch contactPatch,
            AispectContactEncodingProfile contactEncodingProfile,
            AispectSignalWindowBuilder.Window impactWindow,
            AispectDeviceCapabilityProfiler.Snapshot capability,
            AispectTouchNormalizationProfile normalizationProfile,
            AispectModels.ImpactPrediction prediction,
            AispectImpactSignalCollector.Snapshot signalSnapshot
    ) throws IOException, JSONException {
        JSONObject record = new JSONObject();
        record.put("schemaVersion", 2);
        record.put("platform", "android");
        record.put("id", id);
        record.put("createdAt", createdAtMillis / 1000.0);
        record.put("collectionSessionID", collectionSessionId);
        record.put("collectionSessionStartedAt", collectionSessionStartedAtMillis / 1000.0);
        record.put("recordStatus", recordStatus);
        if (rejectionReason != null && !rejectionReason.isEmpty()) {
            record.put("rejectionReason", rejectionReason);
        }
        record.put("label", label.toJson());
        record.put("manualLabelSource", "user_selected");
        record.put("sourceTouchKind", sourceTouchKind);
        if (liftOffset != null) {
            record.put("liftOffset", liftOffset);
        }
        record.put("sampleRateHz", impactWindow.sampleRateHz);
        record.put("referenceFrameIndex", 0);
        record.put("anchorTimestamp", impactWindow.anchorTimestampSeconds);
        record.put("maxAbsDelta", impactWindow.maxAbsDelta);
        record.put("appVersion", config.appVersion);
        record.put("appBuild", config.appBuild);
        record.put("deviceCapability", capability.toJson());
        record.put("contactPatch", contactPatch.toJson());
        if (contactEncodingProfile != null) {
            record.put("contactEncoding", contactEncodingProfile.toJson());
        }
        if (normalizationProfile != null) {
            record.put("touchNormalization", normalizationProfile.toJson());
        }
        record.put("motionSignal", motionSignalToJson(signalSnapshot));
        if (prediction != null) {
            record.put("cnnPrediction", prediction.toJson());
        }
        record.put("touchEvents", touchEventsToJson(touchEvents));
        record.put("touchFrames", touchFramesToJson(touchFrames));
        record.put("frames", AispectModels.impactFramesToJson(impactWindow.frames, impactWindow.firstFrameIndex));

        File dir = datasetDirectory();
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IOException("dataset directory create failed");
        }
        String name = String.format(Locale.US, "%d_%s_%s.json", createdAtMillis, label.datasetKey(), id);
        File destination = new File(dir, name);
        try (FileOutputStream output = new FileOutputStream(destination)) {
            output.write(record.toString(2).getBytes(StandardCharsets.UTF_8));
        }
        return new SavedRecord(destination, record);
    }

    private static JSONObject motionSignalToJson(AispectImpactSignalCollector.Snapshot snapshot) throws JSONException {
        JSONObject json = new JSONObject();
        json.put("frameCount", snapshot.frames.size());
        json.put("sampleRateHz", snapshot.sampleRateHz);
        json.put("accelerationSensorName", snapshot.accelerationSensorName);
        json.put("hasGyroscope", snapshot.hasGyroscope);
        json.put("hasGravity", snapshot.hasGravity);
        return json;
    }

    private static JSONArray touchFramesToJson(List<AispectTouchFrame> frames) throws JSONException {
        JSONArray array = new JSONArray();
        for (int i = 0; i < frames.size(); i++) {
            array.put(frames.get(i).toJson(i));
        }
        return array;
    }

    private static JSONArray touchEventsToJson(List<AispectModels.TouchEvent> events) throws JSONException {
        JSONArray array = new JSONArray();
        for (AispectModels.TouchEvent event : events) {
            array.put(event.toJson());
        }
        return array;
    }
}
