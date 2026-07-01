package com.zhifaios.eyes.aispect;

import android.os.Handler;
import android.os.Looper;
import android.view.MotionEvent;

public final class AispectInteractionEngine {
    public interface Listener {
        void onTouchEvent(AispectModels.TouchEvent event);
    }

    public static final class Config {
        public long holdDurationMs = 600L;
        public long minimumSampleIntervalMs = 11L;
        public float dragActivationRadiusPx = 8f;
        public float heavyLiftOffsetThresholdPx = 6f;
    }

    private final Handler handler = new Handler(Looper.getMainLooper());
    private final Config config = new Config();
    private Listener listener;
    private float initialX;
    private float initialY;
    private float lastX;
    private float lastY;
    private long initialEventTime;
    private long lastSampleEventTime;
    private long lastEventTime;
    private boolean isDragging;
    private boolean isTracking;
    private Runnable holdRunnable;

    public AispectInteractionEngine(Listener listener) {
        this.listener = listener;
    }

    public Config config() {
        return config;
    }

    public void setListener(Listener listener) {
        this.listener = listener;
    }

    public boolean handleMotionEvent(MotionEvent event) {
        int action = event.getActionMasked();
        if (action == MotionEvent.ACTION_DOWN) {
            begin(event);
            return true;
        }
        if (!isTracking) {
            return true;
        }
        if (action == MotionEvent.ACTION_MOVE) {
            move(event);
            return true;
        }
        if (action == MotionEvent.ACTION_UP) {
            end(event);
            return true;
        }
        if (action == MotionEvent.ACTION_CANCEL) {
            cancel(event);
            return true;
        }
        return true;
    }

    public boolean isTracking() {
        return isTracking;
    }

    private void begin(MotionEvent event) {
        cancelHold();
        isTracking = true;
        isDragging = false;
        initialX = event.getX();
        initialY = event.getY();
        lastX = initialX;
        lastY = initialY;
        initialEventTime = event.getEventTime();
        lastSampleEventTime = initialEventTime;
        lastEventTime = initialEventTime;
        emit(AispectModels.TouchKind.SAMPLE, initialX, initialY, initialEventTime, null);
        scheduleHold();
    }

    private void move(MotionEvent event) {
        int pointerIndex = 0;
        int historySize = event.getHistorySize();
        boolean didEmitSample = false;
        float latestX = event.getX(pointerIndex);
        float latestY = event.getY(pointerIndex);
        long latestEventTime = event.getEventTime();
        for (int i = 0; i < historySize; i++) {
            float x = event.getHistoricalX(pointerIndex, i);
            float y = event.getHistoricalY(pointerIndex, i);
            long eventTime = event.getHistoricalEventTime(i);
            latestX = x;
            latestY = y;
            latestEventTime = eventTime;
            if (eventTime - lastSampleEventTime >= config.minimumSampleIntervalMs) {
                lastSampleEventTime = eventTime;
                lastX = x;
                lastY = y;
                lastEventTime = eventTime;
                emit(AispectModels.TouchKind.SAMPLE, x, y, eventTime, null);
                didEmitSample = true;
            }
        }

        latestX = event.getX(pointerIndex);
        latestY = event.getY(pointerIndex);
        latestEventTime = event.getEventTime();
        if (latestEventTime - lastSampleEventTime >= config.minimumSampleIntervalMs) {
            lastSampleEventTime = latestEventTime;
            lastX = latestX;
            lastY = latestY;
            lastEventTime = latestEventTime;
            emit(AispectModels.TouchKind.SAMPLE, latestX, latestY, latestEventTime, null);
            didEmitSample = true;
        }

        float distance = distance(latestX - initialX, latestY - initialY);
        if (distance > Math.max(1f, config.dragActivationRadiusPx)) {
            if (!isDragging) {
                isDragging = true;
                cancelHold();
                emit(AispectModels.TouchKind.DRAG_START, latestX, latestY, latestEventTime, null);
            }
            emit(AispectModels.TouchKind.DRAG, latestX, latestY, latestEventTime, null);
        } else if (!didEmitSample && latestEventTime != lastEventTime) {
            lastX = latestX;
            lastY = latestY;
            lastEventTime = latestEventTime;
        }
    }

    private void end(MotionEvent event) {
        cancelHold();
        float x = event.getX();
        float y = event.getY();
        long eventTime = event.getEventTime();
        if (isDragging) {
            emit(AispectModels.TouchKind.DRAG_END, x, y, eventTime, null);
        } else {
            float liftOffset = distance(x - lastX, y - lastY);
            AispectModels.TouchKind kind = liftOffset >= Math.max(1f, config.heavyLiftOffsetThresholdPx)
                    ? AispectModels.TouchKind.HEAVY_TAP
                    : AispectModels.TouchKind.LIGHT_TAP;
            emit(kind, x, y, eventTime, liftOffset);
        }
        resetState();
    }

    private void cancel(MotionEvent event) {
        cancelHold();
        emit(AispectModels.TouchKind.CANCEL, event.getX(), event.getY(), event.getEventTime(), null);
        resetState();
    }

    private void scheduleHold() {
        holdRunnable = () -> {
            if (!isTracking || isDragging) {
                return;
            }
            emit(AispectModels.TouchKind.LIGHT_HOLD, initialX, initialY, initialEventTime, null);
        };
        handler.postDelayed(holdRunnable, Math.max(100L, config.holdDurationMs));
    }

    private void cancelHold() {
        if (holdRunnable != null) {
            handler.removeCallbacks(holdRunnable);
            holdRunnable = null;
        }
    }

    private void emit(AispectModels.TouchKind kind, float x, float y, long eventTimeMillis, Float liftOffset) {
        if (listener == null) {
            return;
        }
        listener.onTouchEvent(new AispectModels.TouchEvent(kind, x, y, eventTimeMillis / 1000.0, liftOffset));
    }

    private void resetState() {
        isTracking = false;
        isDragging = false;
        holdRunnable = null;
    }

    private static float distance(float x, float y) {
        return (float) Math.sqrt(x * x + y * y);
    }
}
