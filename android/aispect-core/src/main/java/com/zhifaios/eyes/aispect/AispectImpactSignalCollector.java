package com.zhifaios.eyes.aispect;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

public final class AispectImpactSignalCollector {
    public static final class RotationFrame {
        public final String sourceType;
        public final String sensorName;
        public final long sensorTimestampElapsedRealtimeNanos;
        public final long receivedElapsedRealtimeNanos;
        public final int accuracy;
        public final float[] values;
        public final int streamEpoch;

        RotationFrame(
                String sourceType,
                String sensorName,
                long sensorTimestampElapsedRealtimeNanos,
                long receivedElapsedRealtimeNanos,
                int accuracy,
                float[] values
        ) {
            this(
                    sourceType,
                    sensorName,
                    sensorTimestampElapsedRealtimeNanos,
                    receivedElapsedRealtimeNanos,
                    accuracy,
                    values,
                    0
            );
        }

        RotationFrame(
                String sourceType,
                String sensorName,
                long sensorTimestampElapsedRealtimeNanos,
                long receivedElapsedRealtimeNanos,
                int accuracy,
                float[] values,
                int streamEpoch
        ) {
            this.sourceType = sourceType;
            this.sensorName = sensorName;
            this.sensorTimestampElapsedRealtimeNanos = sensorTimestampElapsedRealtimeNanos;
            this.receivedElapsedRealtimeNanos = receivedElapsedRealtimeNanos;
            this.accuracy = accuracy;
            this.values = values == null ? new float[0] : values.clone();
            this.streamEpoch = streamEpoch;
        }

        JSONObject toJson() throws JSONException {
            JSONObject json = new JSONObject();
            json.put("sourceType", sourceType);
            json.put("sensorName", sensorName);
            json.put("sensorTimestampElapsedRealtimeNanos", sensorTimestampElapsedRealtimeNanos);
            json.put("receivedElapsedRealtimeNanos", receivedElapsedRealtimeNanos);
            json.put("accuracy", accuracy);
            json.put("valueCount", values.length);
            double norm = 0.0;
            for (float value : values) {
                norm += value * value;
            }
            json.put("norm", Math.sqrt(norm));
            org.json.JSONArray array = new org.json.JSONArray();
            for (float value : values) {
                array.put(value);
            }
            json.put("values", array);
            json.put("trainingInput", false);
            json.put("streamEpoch", streamEpoch);
            return json;
        }
    }

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
        public final String accelerationStatus;
        public final String gyroscopeStatus;
        public final String gravityStatus;
        public final String rotationStatus;
        public final String rotationSensorSource;
        public final String rotationSensorName;
        public final List<RotationFrame> rotationFrames;
        public final int streamEpoch;
        public final int timestampRegressionCount;

        Snapshot(
                List<AispectModels.ImpactFrame> frames,
                double sampleRateHz,
                String accelerationSensorName,
                String accelerationSensorSource,
                boolean hasGyroscope,
                boolean hasGravity
        ) {
            this(
                    frames,
                    sampleRateHz,
                    accelerationSensorName,
                    accelerationSensorSource,
                    hasGyroscope,
                    hasGravity,
                    "UNKNOWN",
                    hasGyroscope ? "WAITING_FIRST_EVENT" : "UNAVAILABLE",
                    hasGravity ? "WAITING_FIRST_EVENT" : "UNAVAILABLE",
                    "UNAVAILABLE",
                    "unavailable",
                    "unavailable",
                    new ArrayList<RotationFrame>(),
                    0,
                    0
            );
        }

        Snapshot(
                List<AispectModels.ImpactFrame> frames,
                double sampleRateHz,
                String accelerationSensorName,
                String accelerationSensorSource,
                boolean hasGyroscope,
                boolean hasGravity,
                String accelerationStatus,
                String gyroscopeStatus,
                String gravityStatus,
                String rotationStatus,
                String rotationSensorSource,
                String rotationSensorName,
                List<RotationFrame> rotationFrames,
                int streamEpoch,
                int timestampRegressionCount
        ) {
            this.frames = frames;
            this.sampleRateHz = sampleRateHz;
            this.accelerationSensorName = accelerationSensorName;
            this.accelerationSensorSource = accelerationSensorSource;
            this.hasGyroscope = hasGyroscope;
            this.hasGravity = hasGravity;
            this.accelerationStatus = accelerationStatus;
            this.gyroscopeStatus = gyroscopeStatus;
            this.gravityStatus = gravityStatus;
            this.rotationStatus = rotationStatus;
            this.rotationSensorSource = rotationSensorSource;
            this.rotationSensorName = rotationSensorName;
            this.rotationFrames = rotationFrames == null
                    ? new ArrayList<RotationFrame>()
                    : new ArrayList<>(rotationFrames);
            this.streamEpoch = streamEpoch;
            this.timestampRegressionCount = timestampRegressionCount;
        }
    }

    private final Object lock = new Object();
    private final SensorManager sensorManager;
    private final Config config = new Config();
    private final ArrayList<AispectModels.ImpactFrame> rollingFrames = new ArrayList<>();
    private final ArrayList<RotationFrame> rotationFrames = new ArrayList<>();
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
    private Sensor linearAccelerationSensor;
    private Sensor accelerometerSensor;
    private Sensor gyroscopeSensor;
    private Sensor gravitySensor;
    private Sensor rotationSensor;
    private Sensor gameRotationSensor;
    private Sensor rotationVectorSensor;
    private Sensor geomagneticRotationSensor;
    private String rotationSensorSource = "unavailable";
    private int rotationCandidateIndex;
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
    private long gyroscopeSensorTimestampNanos;
    private long rotationSensorTimestampNanos;
    private long lastAccelerationReceivedNanos;
    private long lastGyroscopeReceivedNanos;
    private long lastGravityReceivedNanos;
    private long lastRotationReceivedNanos;
    private long streamStartElapsedNanos;
    private int streamEpoch;
    private int timestampRegressionCount;
    private int accelerationTimestampRegressionCount;
    private int gyroscopeTimestampRegressionCount;
    private int gravityTimestampRegressionCount;
    private int rotationTimestampRegressionCount;
    private String lastAccelerationFrameSource = "";
    private boolean accelerationRegistered;
    private boolean gyroscopeRegistered;
    private boolean gravityRegistered;
    private boolean rotationRegistered;
    private double touchXNorm;
    private double touchYNorm;
    private double touchDownTimestampSeconds;

    public AispectImpactSignalCollector(Context context) {
        sensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        if (sensorManager != null) {
            linearAccelerationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
            accelerometerSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
            accelerationSensor = linearAccelerationSensor != null ? linearAccelerationSensor : accelerometerSensor;
            gyroscopeSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
            gravitySensor = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY);
            gameRotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR);
            rotationVectorSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);
            geomagneticRotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GEOMAGNETIC_ROTATION_VECTOR);
            selectRotationCandidate(0);
        }
    }

    public Config config() {
        return config;
    }

    public void start() {
        stop();
        synchronized (lock) {
            rollingFrames.clear();
            rotationFrames.clear();
            isRunning = sensorManager != null && (accelerationSensor != null || rotationSensor != null);
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
            gyroscopeSensorTimestampNanos = 0L;
            rotationSensorTimestampNanos = 0L;
            lastAccelerationReceivedNanos = 0L;
            lastGyroscopeReceivedNanos = 0L;
            lastGravityReceivedNanos = 0L;
            lastRotationReceivedNanos = 0L;
            streamStartElapsedNanos = SystemClock.elapsedRealtimeNanos();
            timestampRegressionCount = 0;
            accelerationTimestampRegressionCount = 0;
            gyroscopeTimestampRegressionCount = 0;
            gravityTimestampRegressionCount = 0;
            rotationTimestampRegressionCount = 0;
            lastAccelerationFrameSource = "";
            accelerationRegistered = false;
            gyroscopeRegistered = false;
            gravityRegistered = false;
            rotationRegistered = false;
            rotationCandidateIndex = 0;
            selectRotationCandidate(0);
            streamEpoch += 1;
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
            switchStalledSensorsIfNeeded();
            updateAccelerationStreamSource(accelerationSource());
            double rate = averageSampleIntervalSeconds > 0 ? 1.0 / averageSampleIntervalSeconds : 0;
            String sensorName = accelerationSensor != null ? accelerationSensor.getName() : "unavailable";
            String sensorSource = accelerationSource();
            return new Snapshot(
                    new ArrayList<>(rollingFrames),
                    rate,
                    sensorName,
                    sensorSource,
                    gyroscopeSensor != null,
                    gravitySensor != null,
                    accelerationStatus(),
                    sensorStatus(gyroscopeSensor, gyroscopeSensorTimestampNanos, lastGyroscopeReceivedNanos, gyroscopeTimestampRegressionCount),
                    sensorStatus(gravitySensor, gravitySensorTimestampNanos, lastGravityReceivedNanos, gravityTimestampRegressionCount),
                    sensorStatus(rotationSensor, rotationSensorTimestampNanos, lastRotationReceivedNanos, rotationTimestampRegressionCount),
                    rotationSensorSource,
                    rotationSensor != null ? rotationSensor.getName() : "unavailable",
                    new ArrayList<>(rotationFrames),
                    streamEpoch,
                    timestampRegressionCount
            );
        }
    }

    private String accelerationSource() {
        if (accelerationSensor == null || !accelerationRegistered) {
            return "unavailable";
        }
        if (accelerationSensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
            return "linear_acceleration";
        }
        boolean gravityHealthy = gravityRegistered
                && gravitySensorTimestampNanos > 0L
                && !isStalled(lastGravityReceivedNanos);
        return !gravityHealthy
                ? "accelerometer_raw_no_gravity"
                : "accelerometer_minus_gravity";
    }

    private String accelerationStatus() {
        if (accelerationSensor == null || !accelerationRegistered) {
            return "UNAVAILABLE";
        }
        if (lastSampleSensorNanos <= 0L) {
            return "WAITING_FIRST_EVENT";
        }
        if (isStalled(lastAccelerationReceivedNanos)) {
            return "STALLED";
        }
        if ("accelerometer_raw_no_gravity".equals(accelerationSource())) {
            return "DEGRADED";
        }
        return accelerationTimestampRegressionCount > 0 ? "DEGRADED" : "HEALTHY";
    }

    private String sensorStatus(Sensor sensor, long timestampNanos, long receivedNanos, int regressionCount) {
        boolean registered = sensor == gyroscopeSensor
                ? gyroscopeRegistered
                : sensor == gravitySensor
                ? gravityRegistered
                : sensor == rotationSensor && rotationRegistered;
        if (sensor == null || !registered) {
            return "UNAVAILABLE";
        }
        if (timestampNanos <= 0L) {
            return "WAITING_FIRST_EVENT";
        }
        if (regressionCount > 0) {
            return "DEGRADED";
        }
        return isStalled(receivedNanos) ? "STALLED" : "HEALTHY";
    }

    private boolean isStalled(long receivedNanos) {
        if (receivedNanos <= 0L) {
            return false;
        }
        long timeoutNanos = Math.max(500_000_000L, Math.max(1, activeSamplePeriodUs) * 1_000L * 8L);
        return SystemClock.elapsedRealtimeNanos() - receivedNanos > timeoutNanos;
    }

    private void selectRotationCandidate(int index) {
        Sensor[] sensors = new Sensor[]{gameRotationSensor, rotationVectorSensor, geomagneticRotationSensor};
        String[] sources = new String[]{"game_rotation_vector", "rotation_vector", "geomagnetic_rotation_vector"};
        rotationSensor = null;
        rotationSensorSource = "unavailable";
        for (int i = Math.max(0, index); i < sensors.length; i++) {
            if (sensors[i] != null) {
                rotationSensor = sensors[i];
                rotationSensorSource = sources[i];
                rotationCandidateIndex = i;
                return;
            }
        }
        rotationCandidateIndex = sensors.length;
    }

    private void switchStalledSensorsIfNeeded() {
        if (!isRunning || handler == null) {
            return;
        }
        long now = SystemClock.elapsedRealtimeNanos();
        boolean rotationNeedsSwitch = rotationSensor != null && rotationRegistered
                && ((rotationSensorTimestampNanos <= 0L && now - streamStartElapsedNanos > 500_000_000L)
                || isStalled(lastRotationReceivedNanos));
        if (rotationNeedsSwitch && registerNextRotationCandidate(activeSamplePeriodUs)) {
            streamEpoch += 1;
            streamStartElapsedNanos = now;
        }
        boolean accelerationNeedsSwitch = accelerationRegistered
                && accelerationSensor != null
                && accelerationSensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION
                && ((lastSampleSensorNanos <= 0L && now - streamStartElapsedNanos > 500_000_000L)
                || isStalled(lastAccelerationReceivedNanos));
        if (accelerationNeedsSwitch && accelerometerSensor != null) {
            sensorManager.unregisterListener(listener, accelerationSensor);
            accelerationSensor = accelerometerSensor;
            accelerationRegistered = registerSensor(accelerationSensor, activeSamplePeriodUs);
            if (accelerationRegistered) {
                streamEpoch += 1;
                streamStartElapsedNanos = now;
                lastSampleSensorNanos = 0L;
                averageSampleIntervalSeconds = 0.0;
                lastAccelerationFrameSource = "";
            }
        }
    }

    private boolean registerNextRotationCandidate(int periodUs) {
        sensorManager.unregisterListener(listener, rotationSensor);
        rotationRegistered = false;
        return registerRotationCandidate(rotationCandidateIndex + 1, periodUs);
    }

    private boolean registerRotationCandidate(int startIndex, int periodUs) {
        Sensor[] sensors = new Sensor[]{gameRotationSensor, rotationVectorSensor, geomagneticRotationSensor};
        String[] sources = new String[]{"game_rotation_vector", "rotation_vector", "geomagnetic_rotation_vector"};
        for (int i = Math.max(0, startIndex); i < sensors.length; i++) {
            if (sensors[i] != null && registerSensor(sensors[i], periodUs)) {
                rotationSensor = sensors[i];
                rotationSensorSource = sources[i];
                rotationCandidateIndex = i;
                rotationRegistered = true;
                rotationSensorTimestampNanos = 0L;
                lastRotationReceivedNanos = 0L;
                return true;
            }
        }
        rotationSensor = null;
        rotationSensorSource = "unavailable";
        rotationCandidateIndex = sensors.length;
        return false;
    }

    private void handleSensorEvent(SensorEvent event) {
        long receivedElapsedRealtimeNanos = SystemClock.elapsedRealtimeNanos();
        int type = event.sensor.getType();
        synchronized (lock) {
            if (type == Sensor.TYPE_GAME_ROTATION_VECTOR
                    || type == Sensor.TYPE_ROTATION_VECTOR
                    || type == Sensor.TYPE_GEOMAGNETIC_ROTATION_VECTOR) {
                if (!isRunning || rotationSensor == null || type != rotationSensor.getType()) {
                    return;
                }
                if (rotationSensorTimestampNanos > 0L && event.timestamp <= rotationSensorTimestampNanos) {
                    timestampRegressionCount += 1;
                    rotationTimestampRegressionCount += 1;
                    return;
                }
                rotationSensorTimestampNanos = event.timestamp;
                lastRotationReceivedNanos = receivedElapsedRealtimeNanos;
                rotationFrames.add(new RotationFrame(
                        rotationSensorSource,
                        event.sensor.getName(),
                        event.timestamp,
                        receivedElapsedRealtimeNanos,
                        event.accuracy,
                        event.values,
                        streamEpoch
                ));
                if (rotationFrames.size() > Math.max(80, config.rollingFrameLimit)) {
                    rotationFrames.remove(0);
                }
                return;
            }
            if (type == Sensor.TYPE_GYROSCOPE) {
                if (!isRunning) {
                    return;
                }
                if (gyroscopeSensorTimestampNanos > 0L && event.timestamp <= gyroscopeSensorTimestampNanos) {
                    timestampRegressionCount += 1;
                    gyroscopeTimestampRegressionCount += 1;
                    return;
                }
                gyroX = event.values[0];
                gyroY = event.values[1];
                gyroZ = event.values[2];
                gyroscopeSensorTimestampNanos = event.timestamp;
                lastGyroscopeReceivedNanos = receivedElapsedRealtimeNanos;
                return;
            }
            if (type == Sensor.TYPE_GRAVITY) {
                if (!isRunning) {
                    return;
                }
                if (gravitySensorTimestampNanos > 0L && event.timestamp <= gravitySensorTimestampNanos) {
                    timestampRegressionCount += 1;
                    gravityTimestampRegressionCount += 1;
                    return;
                }
                gravityX = event.values[0];
                gravityY = event.values[1];
                gravityZ = event.values[2];
                gravitySensorTimestampNanos = event.timestamp;
                lastGravityReceivedNanos = receivedElapsedRealtimeNanos;
                return;
            }
            if (!isRunning || type != Sensor.TYPE_LINEAR_ACCELERATION && type != Sensor.TYPE_ACCELEROMETER) {
                return;
            }
            double x = event.values[0];
            double y = event.values[1];
            double z = event.values[2];
            boolean gravityHealthy = gravityRegistered
                    && gravitySensorTimestampNanos > 0L
                    && !isStalled(lastGravityReceivedNanos);
            String source = accelerationFrameSource(type, gravityHealthy);
            if ("accelerometer_minus_gravity".equals(source)
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
            if (lastSampleSensorNanos > 0L && event.timestamp <= lastSampleSensorNanos) {
                timestampRegressionCount += 1;
                accelerationTimestampRegressionCount += 1;
                return;
            }
            if (lastSampleSensorNanos > 0L) {
                double interval = (event.timestamp - lastSampleSensorNanos) / 1_000_000_000.0;
                if (interval > 0) {
                    averageSampleIntervalSeconds = averageSampleIntervalSeconds == 0
                            ? interval
                            : averageSampleIntervalSeconds * 0.9 + interval * 0.1;
                }
            }
            lastSampleSensorNanos = event.timestamp;
            lastAccelerationReceivedNanos = receivedElapsedRealtimeNanos;
            updateAccelerationStreamSource(source);
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
                    receivedElapsedRealtimeNanos,
                    source,
                    streamEpoch
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
        sensorManager.unregisterListener(listener);
        accelerationSensor = linearAccelerationSensor != null ? linearAccelerationSensor : accelerometerSensor;
        selectRotationCandidate(0);
        accelerationRegistered = false;
        gyroscopeRegistered = false;
        gravityRegistered = false;
        rotationRegistered = false;
        try {
            if (accelerationSensor != null) {
                accelerationRegistered = registerSensor(accelerationSensor, periodUs);
                if (!accelerationRegistered && accelerationSensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION
                        && accelerometerSensor != null) {
                    accelerationSensor = accelerometerSensor;
                    accelerationRegistered = registerSensor(accelerationSensor, periodUs);
                }
            }
            if (gyroscopeSensor != null) {
                gyroscopeRegistered = registerSensor(gyroscopeSensor, periodUs);
            }
            if (gravitySensor != null) {
                gravityRegistered = registerSensor(gravitySensor, periodUs);
            }
            boolean rotationOk = registerRotationCandidate(rotationCandidateIndex, periodUs);
            return accelerationRegistered || rotationOk;
        } catch (RuntimeException error) {
            return false;
        }
    }

    private boolean registerSensor(Sensor sensor, int periodUs) {
        return sensor != null && sensorManager.registerListener(listener, sensor, periodUs, handler);
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
        if ("linear_acceleration".equals(source)
                || "user_acceleration".equals(source)
                || "accelerometer_raw_no_gravity".equals(source)) {
            return acceleration.clone();
        }
        if (!("accelerometer_with_gravity".equals(source)
                || "accelerometer_minus_gravity".equals(source))
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

    static String accelerationFrameSource(int sensorType, boolean gravityHealthy) {
        if (sensorType == Sensor.TYPE_LINEAR_ACCELERATION) {
            return "linear_acceleration";
        }
        if (sensorType == Sensor.TYPE_ACCELEROMETER) {
            return gravityHealthy
                    ? "accelerometer_minus_gravity"
                    : "accelerometer_raw_no_gravity";
        }
        return "unavailable";
    }

    static int streamEpochAfterSource(
            int currentEpoch,
            String previousSource,
            String currentSource
    ) {
        if (previousSource == null
                || previousSource.isEmpty()
                || previousSource.equals(currentSource)) {
            return currentEpoch;
        }
        return currentEpoch + 1;
    }

    private void updateAccelerationStreamSource(String source) {
        if (source == null || source.isEmpty() || "unavailable".equals(source)) {
            return;
        }
        streamEpoch = streamEpochAfterSource(streamEpoch, lastAccelerationFrameSource, source);
        lastAccelerationFrameSource = source;
    }
}
