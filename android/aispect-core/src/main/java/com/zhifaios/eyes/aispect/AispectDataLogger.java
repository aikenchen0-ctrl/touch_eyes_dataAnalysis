package com.zhifaios.eyes.aispect;

import android.content.Context;
import android.os.Environment;
import android.view.Display;
import android.view.WindowManager;

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
        record.put("collectionContext", label.collectionContext.toJson());
        record.put("automaticContext", automaticContext(impactWindow, signalSnapshot, downEventElapsedRealtimeNanos));
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
        record.put("deviceCapability", capability.toJson());
        record.put("sensorMetadata", sensorMetadataToJson(signalSnapshot));
        record.put("touchMetadata", touchMetadataToJson(displayRotation()));
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

    /** 原始重力用于分组，不从它猜测身体姿势；显示信息注明实际观测阶段。 */
    private JSONObject automaticContext(AispectSignalWindowBuilder.Window window,
                                       AispectImpactSignalCollector.Snapshot signal,
                                       long downNanos) throws JSONException {
        JSONObject result = new JSONObject();
        result.put("schemaVersion", 1);
        result.put("captureSource", "native_view");
        result.put("displayRotation", displayRotation());
        result.put("displayObservationPhase", "record_write");
        result.put("screenRole", "unknown");
        result.put("foldState", "unknown");
        result.put("gravityUnit", "m/s^2");
        AispectModels.ImpactFrame before = null;
        for (AispectModels.ImpactFrame frame : window.frames) {
            if (frame.sensorTimestampElapsedRealtimeNanos > 0
                    && frame.sensorTimestampElapsedRealtimeNanos <= downNanos
                    && (before == null || frame.sensorTimestampElapsedRealtimeNanos
                    > before.sensorTimestampElapsedRealtimeNanos)) before = frame;
        }
        boolean available = signal.hasGravity && before != null
                && Double.isFinite(before.gravityX) && Double.isFinite(before.gravityY)
                && Double.isFinite(before.gravityZ)
                && Math.sqrt(before.gravityX * before.gravityX + before.gravityY * before.gravityY
                + before.gravityZ * before.gravityZ) > 1e-6;
        result.put("gravityAvailable", available);
        result.put("gravityAtPreDown", available ? new JSONArray().put(before.gravityX)
                .put(before.gravityY).put(before.gravityZ) : JSONObject.NULL);
        result.put("gravityTimestampKnown", false);
        result.put("imuCarrierTimestampNanos", before == null
                ? JSONObject.NULL : before.sensorTimestampElapsedRealtimeNanos);
        return result;
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

    private static JSONObject motionSignalToJson(AispectImpactSignalCollector.Snapshot snapshot) throws JSONException {
        JSONObject json = new JSONObject();
        json.put("frameCount", snapshot.frames.size());
        json.put("sampleRateHz", snapshot.sampleRateHz);
        json.put("accelerationSensorName", snapshot.accelerationSensorName);
        json.put("accelerationSensorSource", snapshot.accelerationSensorSource);
        json.put("hasGyroscope", snapshot.hasGyroscope);
        json.put("hasGravity", snapshot.hasGravity);
        return json;
    }

    static JSONObject sensorMetadataToJson(AispectImpactSignalCollector.Snapshot snapshot) throws JSONException {
        JSONObject json = new JSONObject();
        json.put("platform", "android");
        json.put("sensorSource", snapshot.accelerationSensorSource);
        json.put("accelerationUnit", "m/s^2");
        json.put("rotationRateUnit", "rad/s");
        json.put("sensorAxisFrame", "canonical_device");
        json.put("axisTransformMatrix", identityAxisTransformMatrix());
        json.put("transformVersion", "cross_device_v1");
        json.put("gravityAvailable", snapshot.hasGravity);
        return json;
    }

    static JSONObject touchMetadataToJson() throws JSONException {
        return touchMetadataToJson(0);
    }

    static JSONObject touchMetadataToJson(int displayRotation) throws JSONException {
        JSONObject json = new JSONObject();
        json.put("touchUnit", "pixels");
        json.put("touchTransformVersion", "cross_device_v1");
        json.put("touchCoordinateFrame", "android_view");
        json.put("displayRotation", displayRotation);
        json.put("touchCoordinateTransformMatrix", touchCoordinateTransformMatrix(displayRotation));
        return json;
    }

    private int displayRotation() {
        WindowManager manager = (WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
        Display display = manager == null ? null : manager.getDefaultDisplay();
        return display == null ? 0 : display.getRotation();
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
        json.put("schemaVersion", 1);
        json.put("clock", "elapsedRealtimeNanos");
        json.put("imuEventTimestampSource", "SensorEvent.timestamp");
        json.put("touchEventTimestampSource", "MotionEvent.eventTime");
        json.put("touchTimestampConversion", "uptimeMillis_to_elapsedRealtimeNanos_at_callback");
        json.put("availabilityRule", "touchReceivedElapsedRealtimeNanos<=imuReceivedElapsedRealtimeNanos");
        json.put("downEventElapsedRealtimeNanos", downEventElapsedRealtimeNanos);
        json.put("downReceivedElapsedRealtimeNanos", downReceivedElapsedRealtimeNanos);
        json.put("liftEventElapsedRealtimeNanos", liftEventElapsedRealtimeNanos);
        json.put("liftReceivedElapsedRealtimeNanos", liftReceivedElapsedRealtimeNanos);
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
