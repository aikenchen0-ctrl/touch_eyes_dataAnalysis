package com.zhifaios.eyes.aispect;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public final class AispectModels {
    static final String[] RAW_FRAME_FIELDS = new String[]{
            "frameIndex",
            "relativeTimestamp",
            "x",
            "y",
            "z",
            "x_norm",
            "y_norm",
            "rotationRateX",
            "rotationRateY",
            "rotationRateZ",
            "gravityX",
            "gravityY",
            "gravityZ",
            "timeSinceTouchDown",
            "timeSinceAnchor",
            "sensorTimestampElapsedRealtimeNanos",
            "receivedElapsedRealtimeNanos"
    };

    static boolean containsRawFrameField(String field) {
        if (field == null) {
            return false;
        }
        for (String candidate : RAW_FRAME_FIELDS) {
            if (field.equals(candidate)) {
                return true;
            }
        }
        return false;
    }

    public static final class CollectorIdentity {
        public final String userName;
        public final String deviceId;

        public CollectorIdentity(String userName, String deviceId) {
            this.userName = normalizeUserName(userName);
            this.deviceId = deviceId == null ? "" : deviceId.trim();
        }

        public static String normalizeUserName(String value) {
            return value == null ? "" : value.trim();
        }

        public static boolean isValidUserName(String value) {
            return !normalizeUserName(value).isEmpty();
        }

        JSONObject toJson() throws JSONException {
            JSONObject json = new JSONObject();
            json.put("userName", userName);
            json.put("deviceId", deviceId);
            return json;
        }
    }

    private AispectModels() {
    }

    public enum TouchKind {
        SAMPLE("sample"),
        LIGHT_TAP("lightTap"),
        HEAVY_TAP("heavyTap"),
        LIGHT_HOLD("lightHold"),
        HEAVY_HOLD("heavyHold"),
        HEAVY_PRESS("heavyPress"),
        DRAG_START("dragStart"),
        DRAG("drag"),
        DRAG_END("dragEnd"),
        CANCEL("cancel");

        private final String key;

        TouchKind(String key) {
            this.key = key;
        }

        public String key() {
            return key;
        }
    }

    public enum MotionState {
        HANDHELD_STATIC("handheld_static"),
        PLACED_STATIC("placed_static"),
        WALKING("walking");

        private final String key;

        MotionState(String key) {
            this.key = key;
        }

        public String key() {
            return key;
        }
    }

    public enum TouchStrength {
        LIGHT("light"),
        HEAVY("heavy");

        private final String key;

        TouchStrength(String key) {
            this.key = key;
        }

        public String key() {
            return key;
        }
    }

    public enum FingerType {
        THUMB("thumb"),
        INDEX("index");

        private final String key;

        FingerType(String key) {
            this.key = key;
        }

        public String key() {
            return key;
        }
    }

    public static final class DatasetLabel {
        public final MotionState motionState;
        public final TouchStrength touchStrength;
        public final FingerType fingerType;

        public DatasetLabel(MotionState motionState, TouchStrength touchStrength) {
            this(motionState, touchStrength, FingerType.THUMB);
        }

        public DatasetLabel(MotionState motionState, TouchStrength touchStrength, FingerType fingerType) {
            this.motionState = motionState;
            this.touchStrength = touchStrength;
            this.fingerType = fingerType == null ? FingerType.THUMB : fingerType;
        }

        public String datasetKey() {
            return motionState.key() + "_" + touchStrength.key() + "_" + fingerType.key();
        }

        JSONObject toJson() throws JSONException {
            JSONObject json = new JSONObject();
            json.put("motionState", motionState.key());
            json.put("touchStrength", touchStrength.key());
            json.put("fingerType", fingerType.key());
            return json;
        }
    }

    public static final class TouchEvent {
        public final TouchKind kind;
        public final float x;
        public final float y;
        public final double timestampSeconds;
        public final Float liftOffset;

        public TouchEvent(TouchKind kind, float x, float y, double timestampSeconds, Float liftOffset) {
            this.kind = kind;
            this.x = x;
            this.y = y;
            this.timestampSeconds = timestampSeconds;
            this.liftOffset = liftOffset;
        }

        JSONObject toJson() throws JSONException {
            JSONObject json = new JSONObject();
            json.put("kind", kind.key());
            json.put("x", x);
            json.put("y", y);
            json.put("timestamp", timestampSeconds);
            if (liftOffset != null) {
                json.put("liftOffset", liftOffset);
            }
            return json;
        }
    }

    public static final class ImpactFrame {
        public final double delta;
        public final double x;
        public final double y;
        public final double z;
        public final double rotationRateX;
        public final double rotationRateY;
        public final double rotationRateZ;
        public final double gravityX;
        public final double gravityY;
        public final double gravityZ;
        public final double xNorm;
        public final double yNorm;
        public final double timeSinceTouchDown;
        public final double timeSinceAnchor;
        public final double timestampSeconds;
        public final long sensorTimestampElapsedRealtimeNanos;
        public final long receivedElapsedRealtimeNanos;

        public ImpactFrame(
                double delta,
                double x,
                double y,
                double z,
                double rotationRateX,
                double rotationRateY,
                double rotationRateZ,
                double gravityX,
                double gravityY,
                double gravityZ,
                double xNorm,
                double yNorm,
                double timeSinceTouchDown,
                double timeSinceAnchor,
                double timestampSeconds
        ) {
            this(
                    delta,
                    x,
                    y,
                    z,
                    rotationRateX,
                    rotationRateY,
                    rotationRateZ,
                    gravityX,
                    gravityY,
                    gravityZ,
                    xNorm,
                    yNorm,
                    timeSinceTouchDown,
                    timeSinceAnchor,
                    timestampSeconds,
                    0L,
                    0L
            );
        }

        public ImpactFrame(
                double delta,
                double x,
                double y,
                double z,
                double rotationRateX,
                double rotationRateY,
                double rotationRateZ,
                double gravityX,
                double gravityY,
                double gravityZ,
                double xNorm,
                double yNorm,
                double timeSinceTouchDown,
                double timeSinceAnchor,
                double timestampSeconds,
                long sensorTimestampElapsedRealtimeNanos,
                long receivedElapsedRealtimeNanos
        ) {
            this.delta = delta;
            this.x = x;
            this.y = y;
            this.z = z;
            this.rotationRateX = rotationRateX;
            this.rotationRateY = rotationRateY;
            this.rotationRateZ = rotationRateZ;
            this.gravityX = gravityX;
            this.gravityY = gravityY;
            this.gravityZ = gravityZ;
            this.xNorm = xNorm;
            this.yNorm = yNorm;
            this.timeSinceTouchDown = timeSinceTouchDown;
            this.timeSinceAnchor = timeSinceAnchor;
            this.timestampSeconds = timestampSeconds;
            this.sensorTimestampElapsedRealtimeNanos = sensorTimestampElapsedRealtimeNanos;
            this.receivedElapsedRealtimeNanos = receivedElapsedRealtimeNanos;
        }

        public static ImpactFrame zero() {
            return new ImpactFrame(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        }

        JSONObject toJson(int frameIndex) throws JSONException {
            JSONObject json = new JSONObject();
            json.put("frameIndex", frameIndex);
            json.put("relativeTimestamp", timeSinceTouchDown);
            json.put("x", x);
            json.put("y", y);
            json.put("z", z);
            json.put("x_norm", xNorm);
            json.put("y_norm", yNorm);
            json.put("rotationRateX", rotationRateX);
            json.put("rotationRateY", rotationRateY);
            json.put("rotationRateZ", rotationRateZ);
            json.put("gravityX", gravityX);
            json.put("gravityY", gravityY);
            json.put("gravityZ", gravityZ);
            json.put("timeSinceTouchDown", timeSinceTouchDown);
            json.put("timeSinceAnchor", timeSinceAnchor);
            json.put("sensorTimestampElapsedRealtimeNanos", sensorTimestampElapsedRealtimeNanos);
            json.put("receivedElapsedRealtimeNanos", receivedElapsedRealtimeNanos);
            return json;
        }
    }

    public static final class ImpactPrediction {
        public final double heavyProbability;
        public final double lightProbability;
        public final String modelId;
        public final double[] probabilities;
        public final String[] labelOrder;
        public final String predictedLabel;
        public final double predictedProbability;

        public ImpactPrediction(double heavyProbability, double lightProbability, String modelId) {
            this(new double[]{heavyProbability, lightProbability}, new String[]{"heavy", "light"}, modelId);
        }

        public ImpactPrediction(double[] probabilities, String[] labelOrder, String modelId) {
            this.probabilities = probabilities == null ? new double[0] : probabilities.clone();
            this.labelOrder = labelOrder == null ? new String[0] : labelOrder.clone();
            this.modelId = modelId;
            this.heavyProbability = probabilityForLabel("heavy");
            this.lightProbability = probabilityForLabel("light");
            int bestIndex = bestIndex(this.probabilities);
            this.predictedLabel = bestIndex >= 0 && bestIndex < this.labelOrder.length ? this.labelOrder[bestIndex] : "";
            this.predictedProbability = bestIndex >= 0 && bestIndex < this.probabilities.length ? this.probabilities[bestIndex] : 0;
        }

        public boolean isHeavy() {
            return isHeavy(null, null, null);
        }

        public boolean isHeavy(AispectSignalWindowBuilder.Window window, AispectContactPatch contactPatch, AispectTouchNormalizationProfile normalizationProfile) {
            boolean modelHeavy = predictedHeavyByModel();
            if (!modelHeavy) {
                return false;
            }
            if (!hasHeavyEvidence(window, contactPatch, normalizationProfile)) {
                return false;
            }
            return true;
        }

        JSONObject toJson() throws JSONException {
            JSONObject json = new JSONObject();
            json.put("modelId", modelId);
            json.put("heavyProbability", heavyProbability);
            json.put("lightProbability", lightProbability);
            json.put("decision", predictedLabel == null || predictedLabel.isEmpty() ? (isHeavy() ? "heavy" : "light") : predictedLabel);
            json.put("predictedLabel", predictedLabel);
            json.put("predictedProbability", predictedProbability);
            JSONArray labels = new JSONArray();
            for (String label : labelOrder) {
                labels.put(label);
            }
            JSONArray values = new JSONArray();
            for (double probability : probabilities) {
                values.put(probability);
            }
            json.put("labelOrder", labels);
            json.put("probabilities", values);
            return json;
        }

        public double probabilityForLabel(String label) {
            if (label == null) {
                return 0;
            }
            for (int i = 0; i < labelOrder.length && i < probabilities.length; i++) {
                if (label.equals(labelOrder[i])) {
                    return probabilities[i];
                }
            }
            return 0;
        }

        private boolean predictedHeavyByModel() {
            if ("heavy".equals(predictedLabel) || predictedLabel.endsWith("_heavy")) {
                return true;
            }
            if ("light".equals(predictedLabel) || predictedLabel.endsWith("_light")) {
                return false;
            }
            return heavyProbability >= lightProbability;
        }

        private boolean hasHeavyEvidence(AispectSignalWindowBuilder.Window window, AispectContactPatch contactPatch, AispectTouchNormalizationProfile normalizationProfile) {
            if (window == null) {
                return true;
            }
            double maxDelta = window.maxAbsDelta;
            double dynamicScore = maxDelta;
            double shapeScore = 0.0;
            double timeScore = 0.0;
            boolean hasPatchShape = false;
            if (contactPatch != null) {
                double patchShape = Math.abs(contactPatch.covariance11 - contactPatch.covariance22) + Math.abs(contactPatch.covariance12);
                hasPatchShape = patchShape > 0.00001;
                shapeScore = Math.max(shapeScore, patchShape);
            }
            if (normalizationProfile != null && normalizationProfile.selectedResult != null) {
                // 归一化画像只提供质量上下文，不作为重触证据本身。
            }
            int nonZeroFrames = 0;
            int shapeChangeFrames = 0;
            double previousMagnitude = Double.NaN;
            double previousDelta = Double.NaN;
            double previousMajor = Double.NaN;
            double previousMinor = Double.NaN;
            for (AispectModels.ImpactFrame frame : window.frames) {
                double magnitude = Math.abs(frame.delta) + Math.abs(frame.x) + Math.abs(frame.y) + Math.abs(frame.z);
                if (magnitude > 1e-6) {
                    nonZeroFrames += 1;
                }
                if (!Double.isNaN(previousMagnitude)) {
                    double deltaChange = Math.abs(frame.delta - previousDelta);
                    double magnitudeChange = Math.abs(magnitude - previousMagnitude);
                    if (deltaChange >= 0.01 || magnitudeChange >= 0.018) {
                        shapeChangeFrames += 1;
                    }
                }
                if (!Double.isNaN(previousMajor) && !Double.isNaN(previousMinor)) {
                    double normalizedShape = Math.abs(frame.xNorm - previousMajor) + Math.abs(frame.yNorm - previousMinor);
                    if (normalizedShape >= 0.05) {
                        shapeChangeFrames += 1;
                    }
                }
                previousMagnitude = magnitude;
                previousDelta = frame.delta;
                previousMajor = frame.xNorm;
                previousMinor = frame.yNorm;
                timeScore = Math.max(timeScore, frame.timeSinceAnchor);
            }
            boolean enoughDelta = dynamicScore >= 0.28;
            boolean enoughShape = (shapeScore >= 0.09 && shapeChangeFrames >= 1)
                    || (hasPatchShape && shapeChangeFrames >= 1)
                    || shapeChangeFrames >= 2;
            boolean enoughTime = timeScore >= 0.035 && nonZeroFrames >= 3;
            return enoughDelta && (enoughShape || enoughTime);
        }

        private static int bestIndex(double[] values) {
            if (values == null || values.length == 0) {
                return -1;
            }
            int best = 0;
            for (int i = 1; i < values.length; i++) {
                if (values[i] > values[best]) {
                    best = i;
                }
            }
            return best;
        }
    }

    static JSONArray impactFramesToJson(ImpactFrame[] frames, int firstFrameIndex) throws JSONException {
        JSONArray array = new JSONArray();
        for (int i = 0; i < frames.length; i++) {
            array.put(frames[i].toJson(firstFrameIndex + i));
        }
        return array;
    }
}
