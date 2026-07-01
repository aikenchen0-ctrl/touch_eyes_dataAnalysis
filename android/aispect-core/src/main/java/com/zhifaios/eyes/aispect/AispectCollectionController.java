package com.zhifaios.eyes.aispect;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.view.MotionEvent;

import org.json.JSONException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.UUID;

public final class AispectCollectionController {
    public interface Listener {
        void onStatusChanged(Status status);

        void onRecordSaved(AispectDataLogger.SavedRecord record, Status status);

        void onRecordFailed(Exception error, Status status);
    }

    public static final class Config {
        public long postTouchCaptureDelayMs = 140L;
        public boolean enableCnnDiagnostics = true;
    }

    public static final class Status {
        public final boolean enabled;
        public final AispectModels.DatasetLabel selectedLabel;
        public final int acceptedCount;
        public final int rejectedCount;
        public final String lastEventKey;
        public final String lastEventDetail;
        public final String datasetDirectoryPath;
        public final AispectDeviceCapabilityProfiler.Snapshot capability;
        public final AispectModels.ImpactPrediction latestPrediction;
        public final AispectImpactCNNClassifier.ModelInfo selectedModelInfo;
        public final boolean isLabelLocked;
        public final double latestSampleRateHz;
        public final double latestImpactMaxAbsDelta;
        public final String latestImpactLog;
        public final AispectModels.TouchEvent latestTouchEvent;
        public final AispectTouchNormalizationProfile.Scheme selectedNormalizationScheme;
        public final AispectTouchNormalizationProfile latestNormalizationProfile;
        public final AispectContactEncodingProfile.Scheme selectedContactEncodingScheme;
        public final AispectContactEncodingProfile latestContactEncodingProfile;
        public final int latestPredictionSequence;

        Status(
                boolean enabled,
                AispectModels.DatasetLabel selectedLabel,
                int acceptedCount,
                int rejectedCount,
                String lastEventKey,
                String lastEventDetail,
                String datasetDirectoryPath,
                AispectDeviceCapabilityProfiler.Snapshot capability,
                AispectModels.ImpactPrediction latestPrediction,
                AispectImpactCNNClassifier.ModelInfo selectedModelInfo,
                boolean isLabelLocked,
                double latestSampleRateHz,
                double latestImpactMaxAbsDelta,
                String latestImpactLog,
                AispectModels.TouchEvent latestTouchEvent,
                AispectTouchNormalizationProfile.Scheme selectedNormalizationScheme,
                AispectTouchNormalizationProfile latestNormalizationProfile,
                AispectContactEncodingProfile.Scheme selectedContactEncodingScheme,
                AispectContactEncodingProfile latestContactEncodingProfile,
                int latestPredictionSequence
        ) {
            this.enabled = enabled;
            this.selectedLabel = selectedLabel;
            this.acceptedCount = acceptedCount;
            this.rejectedCount = rejectedCount;
            this.lastEventKey = lastEventKey;
            this.lastEventDetail = lastEventDetail;
            this.datasetDirectoryPath = datasetDirectoryPath;
            this.capability = capability;
            this.latestPrediction = latestPrediction;
            this.selectedModelInfo = selectedModelInfo;
            this.isLabelLocked = isLabelLocked;
            this.latestSampleRateHz = latestSampleRateHz;
            this.latestImpactMaxAbsDelta = latestImpactMaxAbsDelta;
            this.latestImpactLog = latestImpactLog;
            this.latestTouchEvent = latestTouchEvent;
            this.selectedNormalizationScheme = selectedNormalizationScheme;
            this.latestNormalizationProfile = latestNormalizationProfile;
            this.selectedContactEncodingScheme = selectedContactEncodingScheme;
            this.latestContactEncodingProfile = latestContactEncodingProfile;
            this.latestPredictionSequence = latestPredictionSequence;
        }
    }

    private final Handler handler = new Handler(Looper.getMainLooper());
    private final Config config = new Config();
    private final AispectInteractionEngine interactionEngine;
    private final AispectImpactSignalCollector signalCollector;
    private final AispectSignalWindowBuilder windowBuilder = new AispectSignalWindowBuilder();
    private final AispectDataLogger dataLogger;
    private final AispectImpactCNNClassifier classifier;
    private final AispectDeviceCapabilityProfiler capabilityProfiler = new AispectDeviceCapabilityProfiler();
    private final AispectTouchNormalizationEngine normalizationEngine = new AispectTouchNormalizationEngine();
    private final ArrayList<AispectTouchFrame> activeTouchFrames = new ArrayList<>();
    private final ArrayList<AispectModels.TouchEvent> activeEvents = new ArrayList<>();
    private AispectModels.DatasetLabel selectedLabel = new AispectModels.DatasetLabel(AispectModels.MotionState.HANDHELD_STATIC, AispectModels.TouchStrength.LIGHT);
    private AispectModels.DatasetLabel activeLabel = selectedLabel;
    private Listener listener;
    private boolean enabled;
    private boolean activeTouch;
    private boolean activeSaveScheduled;
    private boolean activeShouldSaveSample = true;
    private String selectedModelId;
    private String activeModelId;
    private AispectTouchNormalizationProfile.Scheme activeNormalizationScheme = AispectTouchNormalizationProfile.Scheme.COVARIANCE_MATRIX;
    private AispectContactEncodingProfile.Scheme selectedContactEncodingScheme = AispectContactEncodingProfile.Scheme.GAUSSIAN_MAP;
    private AispectContactEncodingProfile.Scheme activeContactEncodingScheme = AispectContactEncodingProfile.Scheme.GAUSSIAN_MAP;
    private int acceptedCount;
    private int rejectedCount;
    private long collectionSessionStartedAtMillis;
    private String collectionSessionId = UUID.randomUUID().toString();
    private long downEventTimeMillis;
    private double downTimestampSeconds;
    private int activeTouchSequence;
    private AispectModels.TouchEvent endingEvent;
    private AispectModels.ImpactPrediction latestPrediction;
    private double latestSampleRateHz;
    private double latestImpactMaxAbsDelta;
    private String latestImpactLog;
    private AispectModels.TouchEvent latestTouchEvent;
    private AispectTouchNormalizationProfile latestNormalizationProfile;
    private AispectContactEncodingProfile latestContactEncodingProfile;
    private int latestPredictionSequence;
    private String lastEventKey = "collector_idle";
    private String lastEventDetail;

    public AispectCollectionController(Context context) {
        dataLogger = new AispectDataLogger(context);
        classifier = new AispectImpactCNNClassifier(context);
        AispectImpactCNNClassifier.ModelInfo defaultModel = classifier.defaultModelInfo();
        if (defaultModel != null) {
            selectedModelId = defaultModel.id;
            activeModelId = defaultModel.id;
        }
        signalCollector = new AispectImpactSignalCollector(context);
        interactionEngine = new AispectInteractionEngine(event -> {
            activeEvents.add(event);
            handleAispectEvent(event);
        });
    }

    public Config config() {
        return config;
    }

    public AispectDataLogger dataLogger() {
        return dataLogger;
    }

    public AispectInteractionEngine interactionEngine() {
        return interactionEngine;
    }

    public List<AispectImpactCNNClassifier.ModelInfo> availableModels() {
        return classifier.availableModels();
    }

    public AispectImpactCNNClassifier.ModelInfo selectedModelInfo() {
        return classifier.modelInfoForId(selectedModelId);
    }

    public AispectTouchNormalizationProfile.Scheme selectedNormalizationScheme() {
        return normalizationEngine.selectedScheme();
    }

    public AispectContactEncodingProfile.Scheme selectedContactEncodingScheme() {
        return selectedContactEncodingScheme;
    }

    public void setSelectedContactEncodingScheme(AispectContactEncodingProfile.Scheme scheme) {
        if (enabled || activeTouch || scheme == null) {
            return;
        }
        selectedContactEncodingScheme = scheme;
        clearLatestPredictionSnapshot();
        lastEventKey = "collector_task_updated";
        lastEventDetail = scheme.key();
        emitStatus();
    }

    public void setSelectedNormalizationScheme(AispectTouchNormalizationProfile.Scheme scheme) {
        if (enabled || activeTouch || scheme == null) {
            return;
        }
        normalizationEngine.setSelectedScheme(scheme);
        clearLatestPredictionSnapshot();
        lastEventKey = "collector_task_updated";
        lastEventDetail = scheme.key();
        emitStatus();
    }

    public void setListener(Listener listener) {
        this.listener = listener;
    }

    public void setSelectedLabel(AispectModels.DatasetLabel label) {
        if (enabled || activeTouch) {
            return;
        }
        selectedLabel = label;
        clearLatestPredictionSnapshot();
        lastEventKey = "collector_task_updated";
        lastEventDetail = label.datasetKey();
        emitStatus();
    }

    public void setSelectedModelId(String modelId) {
        if (activeTouch || modelId == null) {
            return;
        }
        AispectImpactCNNClassifier.ModelInfo info = classifier.modelInfoForId(modelId);
        if (info == null) {
            return;
        }
        selectedModelId = info.id;
        clearLatestPredictionSnapshot();
        lastEventKey = "collector_task_updated";
        lastEventDetail = info.id;
        emitStatus();
    }

    public AispectModels.DatasetLabel selectedLabel() {
        return selectedLabel;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
        clearLatestPredictionSnapshot();
        if (enabled) {
            acceptedCount = 0;
            rejectedCount = 0;
            collectionSessionStartedAtMillis = System.currentTimeMillis();
            collectionSessionId = UUID.randomUUID().toString();
            normalizationEngine.resetBaseline();
            signalCollector.start();
            lastEventKey = "collector_session_ready";
            lastEventDetail = selectedLabel.datasetKey();
        } else {
            activeTouch = false;
            activeSaveScheduled = false;
            activeTouchFrames.clear();
            activeEvents.clear();
            endingEvent = null;
            signalCollector.stop();
            lastEventKey = "collector_session_stopped";
            lastEventDetail = null;
        }
        emitStatus();
    }

    public boolean isEnabled() {
        return enabled;
    }

    public boolean isTouchActive() {
        return activeTouch;
    }

    public boolean handleMotionEvent(MotionEvent event, int width, int height) {
        if (!enabled) {
            return false;
        }
        collectTouchFrames(event, width, height);
        return interactionEngine.handleMotionEvent(event);
    }

    public boolean handleDiagnosticMotionEvent(MotionEvent event, int width, int height) {
        if (enabled) {
            return false;
        }
        if (event.getActionMasked() == MotionEvent.ACTION_DOWN && !signalCollector.isRunning()) {
            signalCollector.start();
        }
        collectTouchFrames(event, width, height);
        return interactionEngine.handleMotionEvent(event);
    }

    public Status status() {
        return makeStatus();
    }

    private void collectTouchFrames(MotionEvent event, int width, int height) {
        int action = event.getActionMasked();
        int pointerIndex = Math.max(0, event.getActionIndex());
        if (action == MotionEvent.ACTION_DOWN) {
            activeTouch = true;
            activeSaveScheduled = false;
            activeShouldSaveSample = enabled;
            activeTouchSequence += 1;
            activeLabel = selectedLabel;
            activeTouchFrames.clear();
            activeEvents.clear();
            endingEvent = null;
            clearLatestPredictionSnapshot();
            downEventTimeMillis = event.getDownTime();
            downTimestampSeconds = downEventTimeMillis / 1000.0;
            activeModelId = selectedModelId;
            activeNormalizationScheme = normalizationEngine.selectedScheme();
            activeContactEncodingScheme = selectedContactEncodingScheme;
            lastEventKey = activeShouldSaveSample ? "collector_waiting_for_touch_end" : "collector_diagnostic_waiting";
            lastEventDetail = activeLabel.datasetKey();
            signalCollector.updateTouchContext(unit(event.getX(), width), unit(event.getY(), height), downTimestampSeconds);
            emitStatus();
        }
        if (!activeTouch) {
            return;
        }
        int safePointerIndex = Math.min(pointerIndex, event.getPointerCount() - 1);
        for (int i = 0; i < event.getHistorySize(); i++) {
            AispectTouchFrame frame = AispectTouchFrame.fromHistorical(event, safePointerIndex, i, downEventTimeMillis, width, height);
            activeTouchFrames.add(frame);
            capabilityProfiler.observe(frame);
            signalCollector.updateTouchContext(frame.xNorm, frame.yNorm, downTimestampSeconds);
        }
        AispectTouchFrame current = AispectTouchFrame.fromCurrent(event, safePointerIndex, downEventTimeMillis, width, height);
        activeTouchFrames.add(current);
        capabilityProfiler.observe(current);
        signalCollector.updateTouchContext(current.xNorm, current.yNorm, downTimestampSeconds);
    }

    private void handleAispectEvent(AispectModels.TouchEvent event) {
        latestTouchEvent = event;
        switch (event.kind) {
            case LIGHT_TAP:
            case HEAVY_TAP:
                if (activeSaveScheduled) {
                    break;
                }
                activeSaveScheduled = true;
                endingEvent = new AispectModels.TouchEvent(
                        touchKindForLabel(activeLabel),
                        event.x,
                        event.y,
                        event.timestampSeconds,
                        event.liftOffset
                );
                if (!activeEvents.isEmpty()) {
                    activeEvents.set(activeEvents.size() - 1, endingEvent);
                }
                latestTouchEvent = endingEvent;
                int sequence = activeTouchSequence;
                lastEventKey = "collector_collecting_postroll";
                lastEventDetail = endingEvent.kind.key();
                emitStatus();
                handler.postDelayed(() -> finalizeCompletedTouch(sequence), Math.max(0L, config.postTouchCaptureDelayMs));
                break;
            case DRAG_END:
            case CANCEL:
                if (activeSaveScheduled) {
                    break;
                }
                activeSaveScheduled = true;
                endingEvent = new AispectModels.TouchEvent(
                        touchKindForLabel(activeLabel),
                        event.x,
                        event.y,
                        event.timestampSeconds,
                        event.liftOffset
                );
                if (!activeEvents.isEmpty()) {
                    activeEvents.set(activeEvents.size() - 1, endingEvent);
                }
                latestTouchEvent = endingEvent;
                int dragSequence = activeTouchSequence;
                lastEventKey = "collector_collecting_postroll";
                lastEventDetail = endingEvent.kind.key();
                emitStatus();
                handler.postDelayed(() -> finalizeCompletedTouch(dragSequence), Math.max(0L, config.postTouchCaptureDelayMs));
                break;
            default:
                break;
        }
    }

    private static AispectModels.TouchKind touchKindForLabel(AispectModels.DatasetLabel label) {
        if (label != null && label.touchStrength == AispectModels.TouchStrength.HEAVY) {
            return AispectModels.TouchKind.HEAVY_TAP;
        }
        return AispectModels.TouchKind.LIGHT_TAP;
    }

    private void finalizeCompletedTouch(int sequence) {
        if (!activeTouch || endingEvent == null || activeTouchSequence != sequence) {
            return;
        }
        activeTouch = false;
        AispectImpactSignalCollector.Snapshot signalSnapshot = signalCollector.snapshot();
        AispectSignalWindowBuilder.Window impactWindow = windowBuilder.build(signalSnapshot.frames, endingEvent.timestampSeconds, signalSnapshot.sampleRateHz);
        AispectDeviceCapabilityProfiler.Snapshot capability = capabilityProfiler.snapshot();
        AispectContactPatch patch = AispectContactPatch.fromFrames(new ArrayList<>(activeTouchFrames));
        latestNormalizationProfile = normalizationEngine.evaluate(new ArrayList<>(activeTouchFrames), patch);
        latestContactEncodingProfile = AispectContactEncodingProfile.evaluate(new ArrayList<>(activeTouchFrames), patch, activeContactEncodingScheme);
        AispectModels.ImpactPrediction prediction = null;
        if (config.enableCnnDiagnostics) {
            prediction = classifier.predict(impactWindow, activeModelId, patch, latestNormalizationProfile, latestContactEncodingProfile);
            if (prediction == null) {
                prediction = classifier.predictBest(impactWindow, capability, patch, latestNormalizationProfile, latestContactEncodingProfile);
            }
            if (prediction != null && prediction.predictedProbability >= 0.5 && "heavy".equals(prediction.predictedLabel) && !prediction.isHeavy(impactWindow, patch, latestNormalizationProfile)) {
                prediction = downgradePrediction(prediction);
            }
        }
        latestPrediction = prediction;
        latestPredictionSequence = sequence;
        latestSampleRateHz = signalSnapshot.sampleRateHz;
        latestImpactMaxAbsDelta = impactWindow.maxAbsDelta;
        latestImpactLog = impactWindowLog(impactWindow, prediction, latestNormalizationProfile);
        if (!activeShouldSaveSample) {
            lastEventKey = "collector_diagnostic_ready";
            lastEventDetail = activeLabel.datasetKey();
            activeSaveScheduled = false;
            activeTouchFrames.clear();
            activeEvents.clear();
            endingEvent = null;
            signalCollector.stop();
            emitStatus();
            return;
        }
        try {
            AispectDataLogger.SavedRecord record = dataLogger.writeSample(
                    UUID.randomUUID().toString(),
                    System.currentTimeMillis(),
                    collectionSessionId,
                    collectionSessionStartedAtMillis,
                    "accepted",
                    "",
                    activeLabel,
                    endingEvent.kind.key(),
                    endingEvent.liftOffset,
                    new ArrayList<>(activeEvents),
                    new ArrayList<>(activeTouchFrames),
                    patch,
                    latestContactEncodingProfile,
                    impactWindow,
                    capability,
                    latestNormalizationProfile,
                    null,
                    signalSnapshot
            );
            normalizationEngine.observe(latestNormalizationProfile);
            acceptedCount += 1;
            lastEventKey = "collector_event_saved";
            lastEventDetail = activeLabel.datasetKey();
            Status status = makeStatus();
            if (listener != null) {
                listener.onRecordSaved(record, status);
                listener.onStatusChanged(status);
            }
        } catch (IOException | JSONException error) {
            rejectedCount += 1;
            lastEventKey = "collector_event_write_failed";
            lastEventDetail = error.getClass().getSimpleName();
            Status status = makeStatus();
            if (listener != null) {
                listener.onRecordFailed(error, status);
                listener.onStatusChanged(status);
            }
        } finally {
            activeSaveScheduled = false;
            activeShouldSaveSample = true;
            activeTouchFrames.clear();
            activeEvents.clear();
            endingEvent = null;
        }
    }

    private void emitStatus() {
        if (listener != null) {
            listener.onStatusChanged(makeStatus());
        }
    }

    private void clearLatestPredictionSnapshot() {
        latestPrediction = null;
        latestPredictionSequence = 0;
        latestTouchEvent = null;
        latestImpactMaxAbsDelta = 0;
        latestImpactLog = null;
        latestNormalizationProfile = null;
        latestContactEncodingProfile = null;
    }

    private Status makeStatus() {
        return new Status(
                enabled,
                selectedLabel,
                acceptedCount,
                rejectedCount,
                lastEventKey,
                lastEventDetail,
                dataLogger.datasetDirectory().getAbsolutePath(),
                capabilityProfiler.snapshot(),
                latestPrediction,
                classifier.modelInfoForId(selectedModelId),
                enabled || activeTouch,
                latestSampleRateHz,
                latestImpactMaxAbsDelta,
                latestImpactLog,
                latestTouchEvent,
                normalizationEngine.selectedScheme(),
                latestNormalizationProfile,
                selectedContactEncodingScheme,
                latestContactEncodingProfile,
                latestPredictionSequence
        );
    }

    private static String impactWindowLog(
            AispectSignalWindowBuilder.Window window,
            AispectModels.ImpactPrediction prediction,
            AispectTouchNormalizationProfile normalizationProfile
    ) {
        StringBuilder builder = new StringBuilder();
        if (prediction == null) {
            builder.append(String.format(Locale.US, "impact delta window max %.4f cnn unavailable", window.maxAbsDelta));
        } else {
            builder.append(String.format(
                    Locale.US,
                    "impact delta window max %.4f decision %s %.4f heavy %.4f light %.4f",
                    window.maxAbsDelta,
                    prediction.predictedLabel,
                    prediction.predictedProbability,
                    prediction.heavyProbability,
                    prediction.lightProbability
            ));
        }
        if (normalizationProfile != null) {
            builder.append('\n');
            builder.append(String.format(
                    Locale.US,
                    "touch norm %s major %.5f minor %.5f area %.5f conf %.2f",
                    normalizationProfile.selectedScheme.key(),
                    normalizationProfile.selectedResult.majorValue,
                    normalizationProfile.selectedResult.minorValue,
                    normalizationProfile.selectedResult.areaValue,
                    normalizationProfile.selectedResult.confidence
            ));
        }
        for (int i = 0; i < window.frames.length; i++) {
            AispectModels.ImpactFrame frame = window.frames[i];
            int frameIndex = window.firstFrameIndex + i;
            builder.append('\n');
            builder.append(String.format(
                    Locale.US,
                    "f%+d d%+.4f x%+.4f y%+.4f z%+.4f m%+.4f",
                    frameIndex,
                    frame.delta,
                    frame.x,
                    frame.y,
                    frame.z,
                    Math.abs(frame.delta)
            ));
        }
        return builder.toString();
    }

    private static AispectModels.ImpactPrediction downgradePrediction(AispectModels.ImpactPrediction prediction) {
        if (prediction == null || prediction.probabilities.length != 2 || prediction.labelOrder.length != 2) {
            return prediction;
        }
        double heavy = Math.min(prediction.heavyProbability, prediction.lightProbability * 0.7);
        double light = Math.max(prediction.lightProbability, 1e-6);
        double total = heavy + light;
        if (total <= 0) {
            return prediction;
        }
        heavy /= total;
        light /= total;
        return new AispectModels.ImpactPrediction(new double[]{heavy, light}, new String[]{"heavy", "light"}, prediction.modelId);
    }

    private static double unit(float value, int denominator) {
        if (denominator <= 0) {
            return 0;
        }
        double output = value / denominator;
        if (output < 0) {
            return 0;
        }
        if (output > 1) {
            return 1;
        }
        return output;
    }
}
