package com.zhifaios.eyes.aispect;

import android.content.Context;
import android.os.Environment;
import android.os.Build;
import android.util.DisplayMetrics;
import android.view.Display;
import android.view.WindowManager;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public final class AispectDataLogger {
    static final int RAW_SCHEMA_VERSION = 6;

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
    private final MetadataEpochTracker metadataEpochTracker = new MetadataEpochTracker();

    static final class MetadataEpochTracker {
        private String lastFingerprint = "";
        private int epoch;

        int epochFor(JSONObject metadata) {
            String fingerprint = metadata.toString();
            if (!fingerprint.equals(lastFingerprint)) {
                epoch += 1;
                lastFingerprint = fingerprint;
            }
            return epoch;
        }
    }

    static final class TouchMetadataSerialization {
        final JSONObject recordMetadata;
        final JSONArray touchFrames;

        TouchMetadataSerialization(JSONObject recordMetadata, JSONArray touchFrames) {
            this.recordMetadata = recordMetadata;
            this.touchFrames = touchFrames;
        }
    }

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
            long downEventElapsedRealtimeNanos,
            long downReceivedElapsedRealtimeNanos,
            long liftEventElapsedRealtimeNanos,
            long liftReceivedElapsedRealtimeNanos,
            String recordStatus,
            String rejectionReason,
            AispectModels.CollectorIdentity collectorIdentity,
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
        record.put("schemaVersion", RAW_SCHEMA_VERSION);
        record.put("platform", "android");
        record.put("id", id);
        record.put("createdAt", createdAtMillis / 1000.0);
        record.put("collectionSessionID", collectionSessionId);
        record.put("collectionSessionStartedAt", collectionSessionStartedAtMillis / 1000.0);
        record.put("recordStatus", recordStatus);
        if (rejectionReason != null && !rejectionReason.isEmpty()) {
            record.put("rejectionReason", rejectionReason);
        }
        record.put("collector", collectorIdentity.toJson());
        record.put("label", label.toJson());
        record.put("manualLabelSource", "user_selected");
        record.put("sourceTouchKind", sourceTouchKind);
        record.put("timeAlignment", timeAlignmentToJson(
                downEventElapsedRealtimeNanos,
                downReceivedElapsedRealtimeNanos,
                liftEventElapsedRealtimeNanos,
                liftReceivedElapsedRealtimeNanos
        ));
        if (liftOffset != null) {
            record.put("liftOffset", liftOffset);
        }
        record.put("sampleRateHz", impactWindow.sampleRateHz);
        record.put("referenceFrameIndex", impactWindow.downReferenceFrameIndex);
        record.put("anchorTimestamp", impactWindow.downAnchorTimestampSeconds);
        record.put("downAnchorTimestamp", impactWindow.downAnchorTimestampSeconds);
        record.put("liftAnchorTimestamp", impactWindow.liftAnchorTimestampSeconds);
        record.put("downReferenceFrameIndex", impactWindow.downReferenceFrameIndex);
        record.put("liftReferenceFrameIndex", impactWindow.liftReferenceFrameIndex);
        record.put("startFrameIndex", impactWindow.firstFrameIndex);
        record.put("endFrameIndex", impactWindow.lastFrameIndex);
        record.put("preDownFrameCount", Math.max(0, -impactWindow.firstFrameIndex));
        record.put("postLiftFrameCount", Math.max(0, impactWindow.lastFrameIndex - impactWindow.liftReferenceFrameIndex));
        record.put("windowComplete", impactWindow.windowComplete);
        record.put("captureWindow", captureWindowToJson(impactWindow));
        record.put("appVersion", config.appVersion);
        record.put("appBuild", config.appBuild);
        TouchMetadataSerialization touchSerialization = serializeTouchMetadata(
                touchMetadataBaseToJson(displayRotation()),
                touchFrames,
                metadataEpochTracker
        );
        record.put("deviceCapability", capability.toJson());
        record.put("sensorMetadata", sensorMetadataToJson(signalSnapshot, impactWindow));
        record.put("touchMetadata", touchSerialization.recordMetadata);
        record.put("contactPatch", contactPatch.toJson());
        if (contactEncodingProfile != null) {
            record.put("contactEncoding", contactEncodingProfile.toJson());
        }
        if (normalizationProfile != null) {
            record.put("touchNormalization", normalizationProfile.toJson());
        }
        record.put("motionSignal", motionSignalToJson(signalSnapshot, impactWindow));
        if (prediction != null) {
            record.put("cnnPrediction", prediction.toJson());
        }
        record.put("touchEvents", touchEventsToJson(touchEvents));
        record.put("touchFrames", touchSerialization.touchFrames);
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

    static JSONObject captureWindowToJson(AispectSignalWindowBuilder.Window window) throws JSONException {
        JSONObject json = new JSONObject();
        boolean dualAnchor = AispectSignalWindowBuilder.DUAL_ANCHOR_CAPTURE_CONTRACT
                .equals(window.captureWindowContract);
        int requiredPreDownFrameCount = dualAnchor
                ? window.requiredPreDownFrameCount
                : 0;
        int requiredPostLiftFrameCount = dualAnchor
                ? window.requiredPostLiftFrameCount
                : 0;
        double requiredPreDownDurationMs = dualAnchor
                ? window.requiredPreDownDurationSeconds * 1000.0
                : 0.0;
        double requiredPostLiftDurationMs = dualAnchor
                ? window.requiredPostLiftDurationSeconds * 1000.0
                : 0.0;
        int preFrameCount = Math.max(0, -window.firstFrameIndex);
        int postFrameCount = Math.max(0, window.lastFrameIndex - window.liftReferenceFrameIndex);
        boolean preDownFrameComplete = !dualAnchor || preFrameCount >= requiredPreDownFrameCount;
        boolean postLiftFrameComplete = !dualAnchor || postFrameCount >= requiredPostLiftFrameCount;
        boolean preDownDurationComplete = !dualAnchor
                || window.actualPreDownDurationSeconds * 1000.0 + 1e-6 >= requiredPreDownDurationMs;
        boolean postLiftDurationComplete = !dualAnchor
                || window.actualPostLiftDurationSeconds * 1000.0 + 1e-6 >= requiredPostLiftDurationMs;
        json.put("schemaVersion", 1);
        json.put("contract", window.captureWindowContract);
        json.put("requiredPreDownDurationMs", requiredPreDownDurationMs);
        json.put("requiredPostLiftDurationMs", requiredPostLiftDurationMs);
        json.put("requiredPreDownFrameCount", requiredPreDownFrameCount);
        json.put("requiredPostLiftFrameCount", requiredPostLiftFrameCount);
        json.put("actualPreDownFrameCount", preFrameCount);
        json.put("actualPostLiftFrameCount", postFrameCount);
        json.put("actualPreDownDurationMs", window.actualPreDownDurationSeconds * 1000.0);
        json.put("actualPostLiftDurationMs", window.actualPostLiftDurationSeconds * 1000.0);
        json.put("preDownDurationComplete", preDownDurationComplete);
        json.put("postLiftDurationComplete", postLiftDurationComplete);
        json.put("preDownFrameCountComplete", preDownFrameComplete);
        json.put("postLiftFrameCountComplete", postLiftFrameComplete);
        json.put("preDownComplete", window.preDownComplete);
        json.put("postLiftComplete", window.postLiftComplete);
        json.put("windowComplete", window.windowComplete);
        return json;
    }

    private static JSONObject motionSignalToJson(
            AispectImpactSignalCollector.Snapshot snapshot,
            AispectSignalWindowBuilder.Window impactWindow
    ) throws JSONException {
        JSONObject json = new JSONObject();
        json.put("frameCount", snapshot.frames.size());
        json.put("sampleRateHz", snapshot.sampleRateHz);
        json.put("accelerationSensorName", snapshot.accelerationSensorName);
        json.put("accelerationSensorSource", snapshot.accelerationSensorSource);
        json.put("accelerationTrainingInput", AispectModels.ImpactFrame.isTrainingAccelerationSource(
                snapshot.accelerationSensorSource
        ));
        json.put("hasGyroscope", snapshot.hasGyroscope);
        json.put("hasGravity", snapshot.hasGravity);
        json.put("gyroscopeDeclared", snapshot.hasGyroscope);
        json.put("gravityDeclared", snapshot.hasGravity);
        json.put("gyroscopeAvailable", "HEALTHY".equals(snapshot.gyroscopeStatus));
        json.put("gravityAvailable", "HEALTHY".equals(snapshot.gravityStatus));
        json.put("accelerationStatus", snapshot.accelerationStatus);
        json.put("gyroscopeStatus", snapshot.gyroscopeStatus);
        json.put("gravityStatus", snapshot.gravityStatus);
        json.put("rotationStatus", snapshot.rotationStatus);
        json.put("rotationSensorSource", snapshot.rotationSensorSource);
        json.put("rotationFrameCount", rotationFramesForWindow(snapshot, impactWindow).size());
        json.put("streamEpoch", snapshot.streamEpoch);
        json.put("timestampRegressionCount", snapshot.timestampRegressionCount);
        return json;
    }

    static JSONObject sensorMetadataToJson(AispectImpactSignalCollector.Snapshot snapshot) throws JSONException {
        return sensorMetadataToJson(snapshot, null);
    }

    static JSONObject sensorMetadataToJson(
            AispectImpactSignalCollector.Snapshot snapshot,
            AispectSignalWindowBuilder.Window impactWindow
    ) throws JSONException {
        JSONObject json = new JSONObject();
        json.put("platform", "android");
        json.put("sensorSource", snapshot.accelerationSensorSource);
        json.put("accelerationTrainingInput", AispectModels.ImpactFrame.isTrainingAccelerationSource(
                snapshot.accelerationSensorSource
        ));
        json.put("accelerationUnit", "m/s^2");
        json.put("rotationRateUnit", "rad/s");
        json.put("sensorAxisFrame", "canonical_device");
        json.put("axisTransformMatrix", identityAxisTransformMatrix());
        json.put("transformVersion", "cross_device_v2");
        json.put("apiLevel", Build.VERSION.SDK_INT);
        json.put("manufacturer", Build.MANUFACTURER);
        json.put("brand", Build.BRAND);
        json.put("model", Build.MODEL);
        json.put("osVersion", Build.VERSION.RELEASE);
        json.put("buildFingerprint", Build.FINGERPRINT);
        json.put("accelerationStatus", snapshot.accelerationStatus);
        json.put("gyroscopeStatus", snapshot.gyroscopeStatus);
        json.put("gravityStatus", snapshot.gravityStatus);
        json.put("rotationStatus", snapshot.rotationStatus);
        json.put("rotationSensorSource", snapshot.rotationSensorSource);
        json.put("rotationSensorName", snapshot.rotationSensorName);
        json.put("streamEpoch", snapshot.streamEpoch);
        json.put("timestampRegressionCount", snapshot.timestampRegressionCount);
        json.put("rotationTrainingInput", false);
        json.put("gyroscopeDeclared", snapshot.hasGyroscope);
        json.put("gravityDeclared", snapshot.hasGravity);
        json.put("gyroscopeAvailable", "HEALTHY".equals(snapshot.gyroscopeStatus));
        json.put("gravityAvailable", "HEALTHY".equals(snapshot.gravityStatus));
        JSONArray rotationFrames = new JSONArray();
        for (AispectImpactSignalCollector.RotationFrame frame : rotationFramesForWindow(snapshot, impactWindow)) {
            rotationFrames.put(frame.toJson());
        }
        json.put("rotationFrames", rotationFrames);
        return json;
    }

    private static List<AispectImpactSignalCollector.RotationFrame> rotationFramesForWindow(
            AispectImpactSignalCollector.Snapshot snapshot,
            AispectSignalWindowBuilder.Window impactWindow
    ) {
        if (impactWindow == null || impactWindow.frames == null || impactWindow.frames.length == 0) {
            return snapshot.rotationFrames;
        }
        AispectModels.ImpactFrame first = impactWindow.frames[0];
        AispectModels.ImpactFrame last = impactWindow.frames[impactWindow.frames.length - 1];
        long start = first.sensorTimestampElapsedRealtimeNanos;
        long end = last.sensorTimestampElapsedRealtimeNanos;
        boolean useReceivedClock = start <= 0L || end < start;
        if (useReceivedClock) {
            start = first.receivedElapsedRealtimeNanos;
            end = last.receivedElapsedRealtimeNanos;
        }
        if (start <= 0L || end < start) {
            return snapshot.rotationFrames;
        }
        ArrayList<AispectImpactSignalCollector.RotationFrame> result = new ArrayList<>();
        for (AispectImpactSignalCollector.RotationFrame frame : snapshot.rotationFrames) {
            long timestamp = useReceivedClock
                    ? frame.receivedElapsedRealtimeNanos
                    : frame.sensorTimestampElapsedRealtimeNanos;
            if (timestamp >= start && timestamp <= end) {
                result.add(frame);
            }
        }
        return result;
    }

    static JSONObject touchMetadataToJson() throws JSONException {
        return touchMetadataToJson(0);
    }

    static JSONObject touchMetadataToJson(int displayRotation) throws JSONException {
        JSONObject json = new JSONObject();
        json.put("schemaVersion", 2);
        json.put("platform", "android");
        json.put("touchUnit", "pixels");
        json.put("touchTransformVersion", "cross_device_v2");
        json.put("touchCoordinateFrame", "android_view");
        json.put("displayRotation", displayRotation);
        json.put("touchCoordinateTransformMatrix", touchCoordinateTransformMatrix(displayRotation));
        return json;
    }

    private JSONObject touchMetadataBaseToJson(int displayRotation) throws JSONException {
        JSONObject json = touchMetadataToJson(displayRotation);
        DisplayMetrics metrics = context.getResources().getDisplayMetrics();
        json.put("apiLevel", Build.VERSION.SDK_INT);
        json.put("manufacturer", Build.MANUFACTURER);
        json.put("brand", Build.BRAND);
        json.put("model", Build.MODEL);
        json.put("osVersion", Build.VERSION.RELEASE);
        json.put("buildFingerprint", Build.FINGERPRINT);
        json.put("density", metrics.density);
        json.put("densityDpi", metrics.densityDpi);
        json.put("xdpi", metrics.xdpi);
        json.put("ydpi", metrics.ydpi);
        json.put("displayWidthPx", metrics.widthPixels);
        json.put("displayHeightPx", metrics.heightPixels);
        Display display = defaultDisplay();
        if (display != null) {
            json.put("displayId", display.getDisplayId());
            if (Build.VERSION.SDK_INT >= 23) {
                Display.Mode mode = display.getMode();
                if (mode != null) {
                    json.put("displayModeId", mode.getModeId());
                    json.put("displayModeWidthPx", mode.getPhysicalWidth());
                    json.put("displayModeHeightPx", mode.getPhysicalHeight());
                    json.put("displayModeRefreshRateHz", mode.getRefreshRate());
                }
            }
        }
        // 日志器没有 Activity/View 引用时不能猜测系统 Insets，显式记录不可用而不是伪造像素。
        json.put("windowInsets", new JSONObject()
                .put("available", false)
                .put("source", "logger_context_without_activity")
                .put("leftPx", 0)
                .put("topPx", 0)
                .put("rightPx", 0)
                .put("bottomPx", 0));
        return json;
    }

    static TouchMetadataSerialization serializeTouchMetadata(
            JSONObject baseMetadata,
            List<AispectTouchFrame> frames,
            MetadataEpochTracker tracker
    ) throws JSONException {
        JSONArray serializedFrames = new JSONArray();
        JSONObject recordMetadata = null;
        for (int index = 0; index < frames.size(); index++) {
            AispectTouchFrame frame = frames.get(index);
            JSONObject frameMetadata = new JSONObject(baseMetadata.toString());
            frameMetadata.put("viewWidthPx", frame.width);
            frameMetadata.put("viewHeightPx", frame.height);
            if (frame.inputMetadata != null) {
                frameMetadata.put("inputDevice", frame.inputMetadata.toJson());
            }
            int frameMetadataEpoch = tracker.epochFor(frameMetadata);
            JSONObject frameJson = frame.toJson(index);
            frameJson.put("metadataEpoch", frameMetadataEpoch);
            serializedFrames.put(frameJson);
            if (recordMetadata == null) {
                recordMetadata = frameMetadata;
                recordMetadata.put("metadataEpoch", frameMetadataEpoch);
            }
        }
        if (recordMetadata == null) {
            recordMetadata = new JSONObject(baseMetadata.toString());
            recordMetadata.put("metadataEpoch", tracker.epochFor(recordMetadata));
        }
        return new TouchMetadataSerialization(recordMetadata, serializedFrames);
    }

    private int displayRotation() {
        Display display = defaultDisplay();
        return display == null ? 0 : display.getRotation();
    }

    private Display defaultDisplay() {
        WindowManager manager = (WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
        return manager == null ? null : manager.getDefaultDisplay();
    }

    private static JSONArray touchCoordinateTransformMatrix(int displayRotation) {
        JSONArray matrix = new JSONArray();
        switch (displayRotation) {
            case 1:
                matrix.put(new JSONArray().put(0).put(1).put(0));
                matrix.put(new JSONArray().put(-1).put(0).put(1));
                break;
            case 2:
                matrix.put(new JSONArray().put(-1).put(0).put(1));
                matrix.put(new JSONArray().put(0).put(-1).put(1));
                break;
            case 3:
                matrix.put(new JSONArray().put(0).put(-1).put(1));
                matrix.put(new JSONArray().put(1).put(0).put(0));
                break;
            default:
                matrix.put(new JSONArray().put(1).put(0).put(0));
                matrix.put(new JSONArray().put(0).put(1).put(0));
                break;
        }
        return matrix;
    }

    private static JSONArray identityAxisTransformMatrix() {
        JSONArray matrix = new JSONArray();
        matrix.put(new JSONArray().put(1).put(0).put(0));
        matrix.put(new JSONArray().put(0).put(1).put(0));
        matrix.put(new JSONArray().put(0).put(0).put(1));
        return matrix;
    }

    static JSONObject timeAlignmentToJson(
            long downEventElapsedRealtimeNanos,
            long downReceivedElapsedRealtimeNanos,
            long liftEventElapsedRealtimeNanos,
            long liftReceivedElapsedRealtimeNanos
    ) throws JSONException {
        JSONObject json = new JSONObject();
        json.put("schemaVersion", 2);
        json.put("clock", "elapsedRealtimeNanos");
        json.put("imuEventTimestampSource", "SensorEvent.timestamp");
        json.put("touchEventTimestampSource", "MotionEvent.eventTime");
        json.put("touchTimestampConversion", "touch_event_time_millis_from_down_anchor_v2");
        json.put("availabilityRule", "touchReceivedElapsedRealtimeNanos<=imuReceivedElapsedRealtimeNanos");
        json.put("downEventElapsedRealtimeNanos", downEventElapsedRealtimeNanos);
        json.put("downReceivedElapsedRealtimeNanos", downReceivedElapsedRealtimeNanos);
        json.put("liftEventElapsedRealtimeNanos", liftEventElapsedRealtimeNanos);
        json.put("liftReceivedElapsedRealtimeNanos", liftReceivedElapsedRealtimeNanos);
        return json;
    }

    private static JSONArray touchEventsToJson(List<AispectModels.TouchEvent> events) throws JSONException {
        JSONArray array = new JSONArray();
        for (AispectModels.TouchEvent event : events) {
            array.put(event.toJson());
        }
        return array;
    }
}
