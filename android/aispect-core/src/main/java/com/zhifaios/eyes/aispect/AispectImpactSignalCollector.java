package com.zhifaios.eyes.aispect;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;

import java.util.ArrayList;
import java.util.List;

public final class AispectImpactSignalCollector {
    public static final class Config {
        public int samplePeriodUs = 4167;
        public int fallbackSamplePeriodUs = 10000;
        public int rollingFrameLimit = 1200;
        public double axisWeightX = 0.01;
        public double axisWeightY = 1.0;
        public double axisWeightZ = 6.5;
    }

    public static final class Snapshot {
        public final List<AispectModels.ImpactFrame> frames;
        public final double sampleRateHz;
        public final String accelerationSensorName;
        public final String accelerationSensorSource;
        public final boolean hasGyroscope;
        public final boolean hasGravity;

        Snapshot(
                List<AispectModels.ImpactFrame> frames,
                double sampleRateHz,
                String accelerationSensorName,
                String accelerationSensorSource,
                boolean hasGyroscope,
                boolean hasGravity
        ) {
            this.frames = frames;
            this.sampleRateHz = sampleRateHz;
            this.accelerationSensorName = accelerationSensorName;
            this.accelerationSensorSource = accelerationSensorSource;
            this.hasGyroscope = hasGyroscope;
            this.hasGravity = hasGravity;
        }
    }

    private final Object lock = new Object();
    private final SensorManager sensorManager;
    private final Config config = new Config();
    private final ArrayList<AispectModels.ImpactFrame> rollingFrames = new ArrayList<>();
    private final SensorEventListener listener = new SensorEventListener() {
        @Override
        public void onSensorChanged(SensorEvent event) {
            handleSensorEvent(event);
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {
        }
    };

    private Sensor accelerationSensor;
    private Sensor gyroscopeSensor;
    private Sensor gravitySensor;
    private HandlerThread handlerThread;
    private Handler handler;
    private boolean isRunning;
    private int activeSamplePeriodUs;
    private long lastSampleSensorNanos;
    private double averageSampleIntervalSeconds;
    private double gyroX;
    private double gyroY;
    private double gyroZ;
    private double gravityX;
    private double gravityY;
    private double gravityZ;
    private long gravitySensorTimestampNanos;
    private double touchXNorm;
    private double touchYNorm;
    private double touchDownTimestampSeconds;

    public AispectImpactSignalCollector(Context context) {
        sensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        if (sensorManager != null) {
            Sensor linearAcceleration = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
            Sensor accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
            accelerationSensor = linearAcceleration != null ? linearAcceleration : accelerometer;
            gyroscopeSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
            gravitySensor = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY);
        }
    }

    public Config config() {
        return config;
    }

    public void start() {
        stop();
        synchronized (lock) {
            rollingFrames.clear();
            isRunning = sensorManager != null && accelerationSensor != null;
            activeSamplePeriodUs = 0;
            lastSampleSensorNanos = 0L;
            averageSampleIntervalSeconds = 0;
            gyroX = 0;
            gyroY = 0;
            gyroZ = 0;
            gravityX = 0;
            gravityY = 0;
            gravityZ = 0;
            gravitySensorTimestampNanos = 0L;
            touchXNorm = 0;
            touchYNorm = 0;
            touchDownTimestampSeconds = 0;
        }
        if (!isRunning) {
            return;
        }
        handlerThread = new HandlerThread("AispectImpactSignalCollector");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
        int periodUs = registerSensorsWithFallback();
        if (periodUs <= 0) {
            stop();
        }
    }

    public void stop() {
        if (sensorManager != null) {
            sensorManager.unregisterListener(listener);
        }
        if (handlerThread != null) {
            handlerThread.quitSafely();
            handlerThread = null;
            handler = null;
        }
        synchronized (lock) {
            isRunning = false;
        }
    }

    public boolean isRunning() {
        synchronized (lock) {
            return isRunning;
        }
    }

    public void updateTouchContext(double xNorm, double yNorm, double touchDownTimestampSeconds) {
        synchronized (lock) {
            this.touchXNorm = clampUnit(xNorm);
            this.touchYNorm = clampUnit(yNorm);
            if (touchDownTimestampSeconds > 0) {
                this.touchDownTimestampSeconds = touchDownTimestampSeconds;
            }
        }
    }

    public Snapshot snapshot() {
        synchronized (lock) {
            double rate = averageSampleIntervalSeconds > 0 ? 1.0 / averageSampleIntervalSeconds : 0;
            String sensorName = accelerationSensor != null ? accelerationSensor.getName() : "unavailable";
            String sensorSource = accelerationSensor != null && accelerationSensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION
                    ? "linear_acceleration"
                    : "accelerometer_with_gravity";
            return new Snapshot(
                    new ArrayList<>(rollingFrames),
                    rate,
                    sensorName,
                    sensorSource,
                    gyroscopeSensor != null,
                    gravitySensor != null
            );
        }
    }

    private void handleSensorEvent(SensorEvent event) {
        long receivedElapsedRealtimeNanos = SystemClock.elapsedRealtimeNanos();
        int type = event.sensor.getType();
        synchronized (lock) {
            if (type == Sensor.TYPE_GYROSCOPE) {
                gyroX = event.values[0];
                gyroY = event.values[1];
                gyroZ = event.values[2];
                return;
            }
            if (type == Sensor.TYPE_GRAVITY) {
                gravityX = event.values[0];
                gravityY = event.values[1];
                gravityZ = event.values[2];
                gravitySensorTimestampNanos = event.timestamp;
                return;
            }
            if (!isRunning || type != Sensor.TYPE_LINEAR_ACCELERATION && type != Sensor.TYPE_ACCELEROMETER) {
                return;
            }
            double x = event.values[0];
            double y = event.values[1];
            double z = event.values[2];
            String source = type == Sensor.TYPE_LINEAR_ACCELERATION
                    ? "linear_acceleration"
                    : "accelerometer_with_gravity";
            if ("accelerometer_with_gravity".equals(source)
                    && !isGravityTimestampAligned(event.timestamp, gravitySensorTimestampNanos)) {
                return;
            }
            double[] linearAcceleration = linearAcceleration(
                    new double[]{x, y, z},
                    new double[]{gravityX, gravityY, gravityZ},
                    source
            );
            if (linearAcceleration == null) {
                return;
            }
            x = linearAcceleration[0];
            y = linearAcceleration[1];
            z = linearAcceleration[2];
            if (lastSampleSensorNanos > 0L) {
                double interval = (event.timestamp - lastSampleSensorNanos) / 1_000_000_000.0;
                if (interval > 0) {
                    averageSampleIntervalSeconds = averageSampleIntervalSeconds == 0
                            ? interval
                            : averageSampleIntervalSeconds * 0.9 + interval * 0.1;
                }
            }
            lastSampleSensorNanos = event.timestamp;
            double timestampSeconds = event.timestamp / 1_000_000_000.0;
            double timeSinceTouchDown = touchDownTimestampSeconds > 0 ? timestampSeconds - touchDownTimestampSeconds : 0;
            rollingFrames.add(new AispectModels.ImpactFrame(
                    0,
                    x,
                    y,
                    z,
                    gyroX,
                    gyroY,
                    gyroZ,
                    gravityX,
                    gravityY,
                    gravityZ,
                    touchXNorm,
                    touchYNorm,
                    timeSinceTouchDown,
                    0,
                    timestampSeconds,
                    event.timestamp,
                    receivedElapsedRealtimeNanos
            ));
            if (rollingFrames.size() > Math.max(80, config.rollingFrameLimit)) {
                rollingFrames.remove(0);
            }
        }
    }

    private int registerSensorsWithFallback() {
        int[] candidates = new int[]{
                Math.max(1, config.samplePeriodUs),
                Math.max(1, config.fallbackSamplePeriodUs),
                16667,
                SensorManager.SENSOR_DELAY_GAME,
                SensorManager.SENSOR_DELAY_UI
        };
        for (int periodUs : candidates) {
            if (tryRegisterSensors(periodUs)) {
                synchronized (lock) {
                    activeSamplePeriodUs = periodUs;
                }
                return periodUs;
            }
            sensorManager.unregisterListener(listener);
        }
        return 0;
    }

    private boolean tryRegisterSensors(int periodUs) {
        try {
            sensorManager.registerListener(listener, accelerationSensor, periodUs, handler);
            if (gyroscopeSensor != null) {
                sensorManager.registerListener(listener, gyroscopeSensor, periodUs, handler);
            }
            if (gravitySensor != null) {
                sensorManager.registerListener(listener, gravitySensor, periodUs, handler);
            }
            return true;
        } catch (RuntimeException error) {
            return false;
        }
    }

    private static double clampUnit(double value) {
        if (value < 0) {
            return 0;
        }
        if (value > 1) {
            return 1;
        }
        return value;
    }

    static double[] linearAcceleration(double[] acceleration, double[] gravity, String source) {
        if (acceleration == null || acceleration.length != 3 || source == null) {
            return null;
        }
        if ("linear_acceleration".equals(source) || "user_acceleration".equals(source)) {
            return acceleration.clone();
        }
        if (!"accelerometer_with_gravity".equals(source)
                || gravity == null
                || gravity.length != 3) {
            return null;
        }
        return new double[]{
                acceleration[0] - gravity[0],
                acceleration[1] - gravity[1],
                acceleration[2] - gravity[2]
        };
    }

    static boolean isGravityTimestampAligned(long accelerationTimestampNanos, long gravityTimestampNanos) {
        return accelerationTimestampNanos > 0L
                && gravityTimestampNanos > 0L
                && Math.abs(accelerationTimestampNanos - gravityTimestampNanos) <= 50_000_000L;
    }
}
