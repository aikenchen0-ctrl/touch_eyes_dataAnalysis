package com.zhifaios.eyes.aispect;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.view.MotionEvent;

import org.json.JSONException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class AispectCollectionController {
    private static final String IDENTITY_PREFERENCES = "aispect_collector_identity";
    private static final String DEVICE_ID_KEY = "device_id";
    public interface Listener {
        void onStatusChanged(Status status);

        void onRecordSaved(AispectDataLogger.SavedRecord record, Status status);

        void onRecordFailed(Exception error, Status status);
    }

    public static final class Config {
        public long postTouchCaptureDelayMs = 85L;
        public long postTouchCapturePollIntervalMs = 8L;
        public long postTouchCaptureTimeoutMs = 500L;
        public boolean enableCnnDiagnostics = false;
        // 引导采集可按单元格配额暂停；默认关闭以保留其他 SDK 调用方的连续采集行为。
        public int targetSamplesPerCell = 0;
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
    private final AispectDownAnchorWindowGate downWindowGate = new AispectDownAnchorWindowGate();
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
    private String selectedUserName = "";
    private AispectModels.CollectorIdentity activeCollectorIdentity;
    private final String deviceId;
    private AispectTouchNormalizationProfile.Scheme activeNormalizationScheme = AispectTouchNormalizationProfile.Scheme.COVARIANCE_MATRIX;
    private AispectContactEncodingProfile.Scheme selectedContactEncodingScheme = AispectContactEncodingProfile.Scheme.GAUSSIAN_MAP;
    private AispectContactEncodingProfile.Scheme activeContactEncodingScheme = AispectContactEncodingProfile.Scheme.GAUSSIAN_MAP;
    private int acceptedCount;
    private int rejectedCount;
    private long collectionSessionStartedAtMillis;
    private String collectionSessionId = "";
    private long downEventTimeMillis;
    private long downEventElapsedRealtimeNanos;
    private long downReceivedElapsedRealtimeNanos;
    private long liftEventElapsedRealtimeNanos;
    private long liftReceivedElapsedRealtimeNanos;
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
    private Runnable downWindowPoll;
    private String lastEventKey = "collector_idle";
    private String lastEventDetail;
    private final AispectCollectionProgress taskProgress = new AispectCollectionProgress();
    private final ExecutorService reviewExecutor = Executors.newSingleThreadExecutor();
    private AispectDataLogger.SavedRecord latestSavedRecord;
    private AispectCollectionProgress.Entry latestProgressEntry;
    private boolean reviewPending;
    private boolean closed;

    public AispectCollectionController(Context context) {
        SharedPreferences preferences = context.getSharedPreferences(IDENTITY_PREFERENCES, Context.MODE_PRIVATE);
        String storedDeviceId = preferences.getString(DEVICE_ID_KEY, "");
        if (storedDeviceId == null || storedDeviceId.isEmpty()) {
            storedDeviceId = UUID.randomUUID().toString();
            preferences.edit().putString(DEVICE_ID_KEY, storedDeviceId).apply();
        }
        deviceId = storedDeviceId;
        dataLogger = new AispectDataLogger(context);
        classifier = new AispectImpactCNNClassifier(context);
        classifier.setDeviceIdentity(deviceId);
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
        // 中文注释：验证模式需要按下前的滚动缓存，因此控制器创建后即预热 IMU。
        signalCollector.start();
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

    public String firstTouchModelId() {
        return "down_anchor_pre5_post9_v1_pending";
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
        if (label == null || activeTouch || activeSaveScheduled || reviewPending) {
            return;
        }
        if (!sameCollectionCondition(selectedLabel, label)) {
            // 条件变化才开启新会话；同一条件内暂停恢复继续使用原会话身份。
            invalidateCollectionSession();
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

    public void setCollectorUserName(String userName) {
        if (enabled || activeTouch || activeSaveScheduled || reviewPending) {
            return;
        }
        String normalized = AispectModels.CollectorIdentity.normalizeUserName(userName);
        if (!normalized.equals(selectedUserName)) invalidateCollectionSession();
        selectedUserName = normalized;
    }

    public String collectorUserName() {
        return selectedUserName;
    }

    public boolean canStartCollection() {
        return AispectModels.CollectorIdentity.isValidUserName(selectedUserName);
    }

    public boolean canSelectLabel() {
        return !activeTouch && !activeSaveScheduled && !reviewPending;
    }

    public int selectedTaskCount() {
        return taskCount(selectedLabel);
    }

    public int taskCount(AispectModels.DatasetLabel label) {
        return taskProgress.count(label.collectionContext.progressKey(selectedUserName,
                label.fingerType.key(), label.touchStrength.key()));
    }

    public boolean canReviewSelectedTask() {
        return canReviewLatest() && latestProgressEntry != null
                && latestProgressEntry.key.equals(selectedLabel.collectionContext.progressKey(selectedUserName,
                selectedLabel.fingerType.key(), selectedLabel.touchStrength.key()));
    }

    public boolean canReviewLatest() {
        return !closed && !reviewPending && !activeTouch && !activeSaveScheduled && latestSavedRecord != null;
    }

    public void reviewLatest(String reviewStatus) {
        AispectCollectionProgress.validateReviewStatus(reviewStatus);
        if (!canReviewLatest()) return;
        final AispectDataLogger.SavedRecord record = latestSavedRecord;
        final AispectCollectionProgress.Entry entry = latestProgressEntry;
        reviewPending = true;
        emitStatus();
        reviewExecutor.execute(() -> {
            Exception failure = null;
            try {
                AispectSampleReviewStore.write(record.file, entry.sampleId, reviewStatus);
            } catch (Exception error) {
                failure = error;
            }
            final Exception result = failure;
            handler.post(() -> {
                reviewPending = false;
                if (closed) return;
                if (result == null) {
                    taskProgress.review(entry, reviewStatus);
                    lastEventKey = "collector_review_" + reviewStatus;
                } else {
                    lastEventKey = "collector_review_failed";
                }
                lastEventDetail = null;
                emitStatus();
            });
        });
    }

    public void setEnabled(boolean enabled) {
        if (enabled && (activeTouch || activeSaveScheduled || reviewPending)) return;
        if (enabled && !canStartCollection()) {
            return;
        }
        if (enabled && config.targetSamplesPerCell > 0
                && selectedTaskCount() >= config.targetSamplesPerCell) return;
        this.enabled = enabled;
        clearLatestPredictionSnapshot();
        if (enabled) {
            acceptedCount = 0;
            rejectedCount = 0;
            cancelDownWindowPoll();
            downWindowGate.reset();
            ensureCollectionSession();
            activeCollectorIdentity = new AispectModels.CollectorIdentity(selectedUserName, deviceId);
            classifier.setDeviceIdentity(deviceId);
            normalizationEngine.resetBaseline();
            signalCollector.start();
            lastEventKey = "collector_session_ready";
            lastEventDetail = selectedLabel.datasetKey();
        } else {
            activeCollectorIdentity = null;
            activeTouch = false;
            activeSaveScheduled = false;
            cancelDownWindowPoll();
            downWindowGate.reset();
            activeTouchFrames.clear();
            activeEvents.clear();
            endingEvent = null;
            lastEventKey = "collector_session_stopped";
            lastEventDetail = null;
        }
        emitStatus();
    }

    public void shutdown() {
        closed = true;
        reviewExecutor.shutdown();
        cancelDownWindowPoll();
        downWindowGate.reset();
        activeTouch = false;
        signalCollector.stop();
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
        long receivedUptimeMillis = SystemClock.uptimeMillis();
        long receivedElapsedRealtimeNanos = SystemClock.elapsedRealtimeNanos();
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
            downEventElapsedRealtimeNanos = AispectCaptureClock.uptimeMillisToElapsedRealtimeNanos(
                    downEventTimeMillis,
                    receivedUptimeMillis,
                    receivedElapsedRealtimeNanos
            );
            downReceivedElapsedRealtimeNanos = receivedElapsedRealtimeNanos;
            liftEventElapsedRealtimeNanos = 0L;
            liftReceivedElapsedRealtimeNanos = 0L;
            downTimestampSeconds = downEventElapsedRealtimeNanos / 1_000_000_000.0;
            activeModelId = selectedModelId;
            activeNormalizationScheme = normalizationEngine.selectedScheme();
            activeContactEncodingScheme = selectedContactEncodingScheme;
            downWindowGate.begin();
            lastEventKey = activeShouldSaveSample ? "collector_down_window_waiting" : "collector_diagnostic_down_window_waiting";
            lastEventDetail = activeLabel.datasetKey();
            signalCollector.updateTouchContext(unit(event.getX(), width), unit(event.getY(), height), downTimestampSeconds);
            scheduleDownWindowCheck(activeTouchSequence);
            emitStatus();
        }
        if (!activeTouch) {
            return;
        }
        int safePointerIndex = Math.min(pointerIndex, event.getPointerCount() - 1);
        for (int i = 0; i < event.getHistorySize(); i++) {
            AispectTouchFrame frame = AispectTouchFrame.fromHistorical(
                    event,
                    safePointerIndex,
                    i,
                    downEventTimeMillis,
                    width,
                    height,
                    receivedUptimeMillis,
                    receivedElapsedRealtimeNanos
            );
            activeTouchFrames.add(frame);
            capabilityProfiler.observe(frame);
            signalCollector.updateTouchContext(frame.xNorm, frame.yNorm, downTimestampSeconds);
        }
        AispectTouchFrame current = AispectTouchFrame.fromCurrent(
                event,
                safePointerIndex,
                downEventTimeMillis,
                width,
                height,
                receivedUptimeMillis,
                receivedElapsedRealtimeNanos
        );
        activeTouchFrames.add(current);
        capabilityProfiler.observe(current);
        signalCollector.updateTouchContext(current.xNorm, current.yNorm, downTimestampSeconds);
        if (action == MotionEvent.ACTION_UP
                || action == MotionEvent.ACTION_CANCEL
                || action == MotionEvent.ACTION_POINTER_UP) {
            liftEventElapsedRealtimeNanos = current.eventElapsedRealtimeNanos;
            liftReceivedElapsedRealtimeNanos = current.receivedElapsedRealtimeNanos;
        }
    }

    /** 中文注释：按下后轮询传感器滚动缓存，等待真实 +9 帧后只发布一次窗口就绪状态。 */
    private void scheduleDownWindowCheck(final int sequence) {
        cancelDownWindowPoll();
        Runnable poll = new Runnable() {
            @Override
            public void run() {
                if (!activeTouch || activeTouchSequence != sequence) {
                    return;
                }
                AispectImpactSignalCollector.Snapshot snapshot = signalCollector.snapshot();
                AispectImpactCNNClassifier.ModelInfo activeModel = classifier.modelInfoForId(activeModelId);
                if (activeModel != null
                        && AispectCausalPressFeatureBuilder.isTimeGridFeatureContract(activeModel.featureContract)
                        && liftEventElapsedRealtimeNanos > 0L
                        && liftEventElapsedRealtimeNanos
                        < downEventElapsedRealtimeNanos + 25_000_000L) {
                    downWindowGate.reset();
                    downWindowPoll = null;
                    latestPrediction = null;
                    latestPredictionSequence = sequence;
                    lastEventKey = "collector_down_window_rejected_lift_before_endpoint";
                    lastEventDetail = activeLabel.datasetKey();
                    emitStatus();
                    return;
                }
                boolean timeGridModel = activeModel != null
                        && AispectCausalPressFeatureBuilder.isTimeGridFeatureContract(activeModel.featureContract);
                boolean hasTimeGridSupport = !timeGridModel || AispectCausalPressFeatureBuilder.hasTimeGridSupport(
                        snapshot.frames.toArray(new AispectModels.ImpactFrame[0]),
                        downEventElapsedRealtimeNanos
                );
                if (!hasTimeGridSupport) {
                    handler.postDelayed(this, 8L);
                    return;
                }
                AispectSignalWindowBuilder.Window window = windowBuilder.buildDownAnchor(
                        snapshot.frames,
                        downTimestampSeconds,
                        snapshot.sampleRateHz
                );
                if (downWindowGate.tryPublish(window)) {
                    if (activeModel != null && AispectImpactCNNClassifier.usesCausalPressFeatureContract(activeModel.featureContract)) {
                        AispectSignalWindowBuilder.Window predictionWindow = timeGridModel
                                ? new AispectSignalWindowBuilder.Window(
                                snapshot.frames.toArray(new AispectModels.ImpactFrame[0]),
                                0,
                                downTimestampSeconds,
                                snapshot.sampleRateHz,
                                0.0
                        )
                                : window;
                        latestPrediction = classifier.predict(predictionWindow, activeModelId, new ArrayList<>(activeTouchFrames));
                    } else {
                        latestPrediction = null;
                    }
                    latestPredictionSequence = sequence;
                    latestSampleRateHz = snapshot.sampleRateHz;
                    latestImpactMaxAbsDelta = 0;
                    latestImpactLog = downWindowReadyLog(window);
                    lastEventKey = latestPrediction == null
                            ? "collector_down_window_ready_model_pending"
                            : "collector_down_window_ready_causal_prediction";
                    lastEventDetail = activeLabel.datasetKey();
                    emitStatus();
                    downWindowPoll = null;
                    return;
                }
                handler.postDelayed(this, 8L);
            }
        };
        downWindowPoll = poll;
        handler.post(poll);
    }

    private void cancelDownWindowPoll() {
        if (downWindowPoll != null) {
            handler.removeCallbacks(downWindowPoll);
            downWindowPoll = null;
        }
    }

    private static String downWindowReadyLog(AispectSignalWindowBuilder.Window window) {
        return String.format(
                Locale.US,
                "down anchor window ready [%+d...%+d] frames %d; new model pending",
                window.firstFrameIndex,
                window.lastFrameIndex,
                window.frames.length
        );
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
                schedulePostLiftFinalize(sequence, config.postTouchCaptureDelayMs);
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
                schedulePostLiftFinalize(dragSequence, config.postTouchCaptureDelayMs);
                break;
            default:
                break;
        }
    }

    private static AispectModels.TouchKind touchKindForLabel(AispectModels.DatasetLabel label) {
        if (label != null && label.touchStrength == AispectModels.TouchStrength.HEAVY) {
            return AispectModels.TouchKind.HEAVY_TAP;
        }
        if (label != null && label.touchStrength == AispectModels.TouchStrength.ORDINARY) {
            return AispectModels.TouchKind.SAMPLE;
        }
        return AispectModels.TouchKind.LIGHT_TAP;
    }

    private void schedulePostLiftFinalize(int sequence, long delayMs) {
        handler.postDelayed(
                () -> finalizeCompletedTouch(sequence),
                Math.max(0L, delayMs)
        );
    }

    private void finalizeCompletedTouch(int sequence) {
        if (!activeTouch || endingEvent == null || activeTouchSequence != sequence) {
            return;
        }
        AispectImpactSignalCollector.Snapshot signalSnapshot = signalCollector.snapshot();
        AispectSignalWindowBuilder.Window captureWindow = windowBuilder.buildDualAnchor(
                signalSnapshot.frames,
                downTimestampSeconds,
                liftEventElapsedRealtimeNanos > 0L
                        ? liftEventElapsedRealtimeNanos / 1_000_000_000.0
                        : endingEvent.timestampSeconds,
                signalSnapshot.sampleRateHz
        );
        long elapsedSinceLiftMs = liftReceivedElapsedRealtimeNanos > 0L
                ? Math.max(
                0L,
                (SystemClock.elapsedRealtimeNanos() - liftReceivedElapsedRealtimeNanos) / 1_000_000L
        )
                : config.postTouchCaptureTimeoutMs;
        if (AispectPostLiftCaptureGate.shouldWait(
                captureWindow,
                elapsedSinceLiftMs,
                config.postTouchCaptureTimeoutMs
        )) {
            lastEventKey = "collector_waiting_post_lift_margin";
            lastEventDetail = String.format(
                    Locale.US,
                    "%.1fms/%d frames",
                    captureWindow.actualPostLiftDurationSeconds * 1000.0,
                    Math.max(0, captureWindow.lastFrameIndex - captureWindow.liftReferenceFrameIndex)
            );
            emitStatus();
            schedulePostLiftFinalize(sequence, config.postTouchCapturePollIntervalMs);
            return;
        }
        activeTouch = false;
        cancelDownWindowPoll();
        // 中文注释：只有完整窗口通过抬起余量门后才进入设备尺寸历史；
        // 诊断样本和成功保存样本都观察，但当前触摸的预测早已完成。
        classifier.observeCompletedCausalSizeTouch(
                new ArrayList<>(activeTouchFrames)
        );
        AispectDeviceCapabilityProfiler.Snapshot capability = capabilityProfiler.snapshot();
        AispectContactPatch patch = AispectContactPatch.fromFrames(new ArrayList<>(activeTouchFrames));
        latestNormalizationProfile = normalizationEngine.evaluate(new ArrayList<>(activeTouchFrames), patch);
        latestContactEncodingProfile = AispectContactEncodingProfile.evaluate(new ArrayList<>(activeTouchFrames), patch, activeContactEncodingScheme);
        latestSampleRateHz = signalSnapshot.sampleRateHz;
        latestImpactMaxAbsDelta = 0;
        latestImpactLog = String.format(
                Locale.US,
                "raw down/lift window [%+d...%+d] frames %d; down-anchor model pending",
                captureWindow.firstFrameIndex,
                captureWindow.lastFrameIndex,
                captureWindow.frames.length
        );
        if (!activeShouldSaveSample) {
            lastEventKey = "collector_diagnostic_ready_model_pending";
            lastEventDetail = activeLabel.datasetKey();
            activeSaveScheduled = false;
            activeTouchFrames.clear();
            activeEvents.clear();
            endingEvent = null;
            emitStatus();
            return;
        }
        try {
            AispectDataLogger.SavedRecord record = dataLogger.writeSample(
                    UUID.randomUUID().toString(),
                    System.currentTimeMillis(),
                    collectionSessionId,
                    collectionSessionStartedAtMillis,
                    downEventElapsedRealtimeNanos,
                    downReceivedElapsedRealtimeNanos,
                    liftEventElapsedRealtimeNanos,
                    liftReceivedElapsedRealtimeNanos,
                    "accepted",
                    "",
                    activeCollectorIdentity,
                    activeLabel,
                    endingEvent.kind.key(),
                    endingEvent.liftOffset,
                    new ArrayList<>(activeEvents),
                    new ArrayList<>(activeTouchFrames),
                    patch,
                    latestContactEncodingProfile,
                    captureWindow,
                    capability,
                    latestNormalizationProfile,
                    null,
                    signalSnapshot
            );
            normalizationEngine.observe(latestNormalizationProfile);
            latestSavedRecord = record;
            latestProgressEntry = taskProgress.recordSaved(record.json.getString("id"),
                    activeLabel.collectionContext.progressKey(activeCollectorIdentity.userName,
                            activeLabel.fingerType.key(), activeLabel.touchStrength.key()));
            acceptedCount += 1;
            lastEventKey = "collector_event_saved";
            lastEventDetail = activeLabel.datasetKey();
            if (config.targetSamplesPerCell > 0 && selectedTaskCount() >= config.targetSamplesPerCell) {
                // 最后一条已落盘才暂停；保留当前标签和复核入口，不能自动开始下一组。
                enabled = false;
                lastEventKey = "collector_group_completed";
            }
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
            downWindowGate.reset();
            activeTouchFrames.clear();
            activeEvents.clear();
            endingEvent = null;
            emitStatus();
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

    String collectionSessionIdForTesting() {
        return collectionSessionId;
    }

    long collectionSessionStartedAtMillisForTesting() {
        return collectionSessionStartedAtMillis;
    }

    private void ensureCollectionSession() {
        if (collectionSessionStartedAtMillis <= 0L || collectionSessionId == null
                || collectionSessionId.isEmpty()) {
            collectionSessionStartedAtMillis = System.currentTimeMillis();
            collectionSessionId = UUID.randomUUID().toString();
        }
    }

    private void invalidateCollectionSession() {
        collectionSessionStartedAtMillis = 0L;
        collectionSessionId = "";
    }

    static boolean sameCollectionCondition(AispectModels.DatasetLabel first,
                                           AispectModels.DatasetLabel second) {
        if (first == null || second == null) return false;
        AispectCollectionContext left = first.collectionContext;
        AispectCollectionContext right = second.collectionContext;
        return left.task == right.task
                && left.grip.equals(right.grip)
                && left.support.equals(right.support)
                && left.collectionProtocol.equals(right.collectionProtocol)
                && left.scheduleId.equals(right.scheduleId);
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
                enabled || activeTouch || activeSaveScheduled || reviewPending,
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
