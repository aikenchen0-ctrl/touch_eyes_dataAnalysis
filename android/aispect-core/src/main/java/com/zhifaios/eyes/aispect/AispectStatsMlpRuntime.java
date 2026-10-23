package com.zhifaios.eyes.aispect;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

/**
 * 统计特征多任务 MLP 的端侧推理实现。
 *
 * 输入合同固定为九个时间槽和十个物理通道。力度门控模型以轻触为负类、
 * 重触为正类；手指头仅作辅助诊断，旧模型继续保留四分类兼容。
 */
final class AispectStatsMlpRuntime {
    static final String ARCHITECTURE = "stats_mlp_strength_gate_v1";
    static final String LEGACY_ARCHITECTURE = "stats_mlp_v1";
    static final String PRIMARY_TASK = "strength_gate";
    static final String FEATURE_CONTRACT =
            "causal_time_grid_imu11_touch7_mask5_stats_mlp_v3";
    static final String V4_FEATURE_CONTRACT =
            "causal_time_grid_imu11_touch7_mask5_stats_mlp_v4";
    static final String V2_FEATURE_CONTRACT =
            "causal_time_grid_imu11_touch7_mask5_stats_mlp_v2";
    static final String LEGACY_FEATURE_CONTRACT =
            "causal_time_grid_imu11_touch7_mask5_stats_mlp_v1";
    static final String SOURCE_FEATURE_CONTRACT =
            "causal_time_grid_imu11_touch7_mask5_v5";
    static final String V4_SOURCE_FEATURE_CONTRACT =
            "causal_time_grid_imu11_touch7_mask5_v6";
    static final String[] FEATURE_NAMES = new String[]{
            "x_norm",
            "y_norm",
            "userAcceleration.x",
            "userAcceleration.y",
            "userAcceleration.z",
            "rotationRate.x",
            "rotationRate.y",
            "rotationRate.z",
            "size",
            "log_axis_ratio"
    };
    static final String[] V4_FEATURE_NAMES = new String[]{
            "x_norm",
            "y_norm",
            "userAcceleration.x",
            "userAcceleration.y",
            "userAcceleration.z",
            "rotationRate.x",
            "rotationRate.y",
            "rotationRate.z",
            "log_causal_size_ratio",
            "log_axis_ratio"
    };
    static final int[] SOURCE_FEATURE_INDICES = new int[]{0, 1, 2, 3, 4, 5, 6, 7, 12, 15};
    static final String[] LEGACY_SOURCE_FEATURE_NAMES = new String[]{
            "x_norm",
            "y_norm",
            "userAcceleration.x",
            "userAcceleration.y",
            "userAcceleration.z",
            "rotationRate.x",
            "rotationRate.y",
            "rotationRate.z",
            "gravity.x",
            "gravity.y",
            "gravity.z",
            "pressure",
            "size",
            "major_norm",
            "minor_norm",
            "log_axis_ratio",
            "pointer_count",
            "touch_valid",
            "pressure_available",
            "size_available",
            "major_available",
            "minor_available",
            "pointer_count_available"
    };
    static final String[] V2_SOURCE_FEATURE_NAMES = new String[]{
            "x_norm",
            "y_norm",
            "userAcceleration.x",
            "userAcceleration.y",
            "userAcceleration.z",
            "rotationRate.x",
            "rotationRate.y",
            "rotationRate.z",
            "gravity.x",
            "gravity.y",
            "gravity.z",
            "pressure",
            "size",
            "major_norm",
            "minor_norm",
            "log_axis_ratio",
            "orientation_sin",
            "orientation_cos",
            "pointer_count",
            "touch_valid",
            "pressure_available",
            "size_available",
            "major_available",
            "minor_available",
            "orientation_available",
            "pointer_count_available"
    };
    static final String[] SOURCE_FEATURE_NAMES = new String[]{
            "x_norm",
            "y_norm",
            "userAcceleration.x",
            "userAcceleration.y",
            "userAcceleration.z",
            "rotationRate.x",
            "rotationRate.y",
            "rotationRate.z",
            "gravity.x",
            "gravity.y",
            "gravity.z",
            "pressure",
            "size",
            "major_norm",
            "minor_norm",
            "log_axis_ratio",
            "pointer_count",
            "touch_valid",
            "pressure_available",
            "size_available",
            "major_available",
            "minor_available",
            "pointer_count_available"
    };
    static final String[] V4_SOURCE_FEATURE_NAMES = new String[]{
            "x_norm",
            "y_norm",
            "userAcceleration.x",
            "userAcceleration.y",
            "userAcceleration.z",
            "rotationRate.x",
            "rotationRate.y",
            "rotationRate.z",
            "gravity.x",
            "gravity.y",
            "gravity.z",
            "pressure",
            "log_causal_size_ratio",
            "major_norm",
            "minor_norm",
            "log_axis_ratio",
            "pointer_count",
            "touch_valid",
            "pressure_available",
            "size_available",
            "major_available",
            "minor_available",
            "pointer_count_available"
    };
    static final String[] LEGACY_LABEL_ORDER = new String[]{
            "thumb_light",
            "thumb_heavy",
            "index_light",
            "index_heavy"
    };
    static final String[] FINGER_LABEL_ORDER = new String[]{"thumb", "index"};
    static final String[] STRENGTH_LABEL_ORDER = new String[]{"light", "heavy"};
    static final String[] GATE_LABEL_ORDER = new String[]{"light", "heavy"};
    static final String[] NEGATIVE_CLASS_LABELS = new String[]{
            "thumb_light",
            "index_light"
    };
    static final String[] POSITIVE_CLASS_LABELS = new String[]{
            "thumb_heavy",
            "index_heavy"
    };
    static final int FRAME_COUNT = 9;
    static final int STATS_FEATURE_COUNT = 60;
    static final int HIDDEN_COUNT = 24;
    private static final double EPSILON = 1e-9;
    private static final Set<String> WEIGHT_KEYS = new HashSet<>(Arrays.asList(
            "shared.0.weight",
            "shared.0.bias",
            "shared.3.weight",
            "shared.3.bias",
            "finger.weight",
            "finger.bias",
            "strength.weight",
            "strength.bias"
    ));
    private static final Set<String> SIZE_BASELINE_KEYS = new HashSet<>(Arrays.asList(
            "causal",
            "currentTouchExcluded",
            "deviceScope",
            "historyScope",
            "normalization",
            "aggregation",
            "withinTouchAggregation",
            "minimumTouches",
            "maximumTouches",
            "logRatioClip",
            "touchStatistics"
    ));

    private final String featureContract;
    private final String[] featureNames;
    private final String[] labelOrder;
    private final boolean strengthGate;
    private final boolean nanExcludedFromStats;
    private final SequenceChannel[] sequenceChannels;
    private final String[] statsFeatureNames;
    private final double[] statsCenter;
    private final double[] statsScale;
    private final double statsClipMin;
    private final double statsClipMax;
    private final Tensor shared0Weight;
    private final Tensor shared0Bias;
    private final Tensor shared3Weight;
    private final Tensor shared3Bias;
    private final Tensor fingerWeight;
    private final Tensor fingerBias;
    private final Tensor strengthWeight;
    private final Tensor strengthBias;

    private AispectStatsMlpRuntime(
            String featureContract,
            String[] featureNames,
            String[] labelOrder,
            boolean strengthGate,
            boolean nanExcludedFromStats,
            SequenceChannel[] sequenceChannels,
            String[] statsFeatureNames,
            double[] statsCenter,
            double[] statsScale,
            double statsClipMin,
            double statsClipMax,
            Tensor shared0Weight,
            Tensor shared0Bias,
            Tensor shared3Weight,
            Tensor shared3Bias,
            Tensor fingerWeight,
            Tensor fingerBias,
            Tensor strengthWeight,
            Tensor strengthBias
    ) {
        this.featureContract = featureContract;
        this.featureNames = featureNames;
        this.labelOrder = labelOrder;
        this.strengthGate = strengthGate;
        this.nanExcludedFromStats = nanExcludedFromStats;
        this.sequenceChannels = sequenceChannels;
        this.statsFeatureNames = statsFeatureNames;
        this.statsCenter = statsCenter;
        this.statsScale = statsScale;
        this.statsClipMin = statsClipMin;
        this.statsClipMax = statsClipMax;
        this.shared0Weight = shared0Weight;
        this.shared0Bias = shared0Bias;
        this.shared3Weight = shared3Weight;
        this.shared3Bias = shared3Bias;
        this.fingerWeight = fingerWeight;
        this.fingerBias = fingerBias;
        this.strengthWeight = strengthWeight;
        this.strengthBias = strengthBias;
    }

    static boolean isSupportedArchitecture(String value) {
        return ARCHITECTURE.equals(value) || LEGACY_ARCHITECTURE.equals(value);
    }

    static AispectStatsMlpRuntime load(
            String modelId,
            JSONObject weights,
            JSONObject scaler
    ) throws JSONException {
        String runtimeArchitecture = scaler.optString("runtimeArchitecture", "");
        if (!isSupportedArchitecture(runtimeArchitecture)) {
            return null;
        }
        boolean strengthGate = ARCHITECTURE.equals(runtimeArchitecture);
        String featureContract = scaler.optString("featureContract", "");
        if (!FEATURE_CONTRACT.equals(featureContract)
                && !V4_FEATURE_CONTRACT.equals(featureContract)
                && !V2_FEATURE_CONTRACT.equals(featureContract)
                && !LEGACY_FEATURE_CONTRACT.equals(featureContract)) {
            return null;
        }
        // 中文注释：新合同必须同时锁定因果接触尺度代理和统计缺失策略，
        // 避免把设备原始尺寸误标成压力或真实牛顿力。
        boolean causalSizeProxy = V4_FEATURE_CONTRACT.equals(featureContract);
        if (causalSizeProxy) {
            if (!strengthGate) {
                return null;
            }
            JSONObject sizeBaseline = scaler.optJSONObject("sizeBaseline");
            if (!V4_SOURCE_FEATURE_CONTRACT.equals(
                    scaler.optString("sourceFeatureContract", "")
            )
                    || !"contact_area_proxy".equals(
                    scaler.optString("sizeSemantics", "")
            )
                    || scaler.optBoolean("forceMeasurement", true)
                    || !"nan_excluded_from_stats".equals(
                    scaler.optString("missingValuePolicy", "")
            )
                    || sizeBaseline == null
                    || !hasExactKeys(sizeBaseline, SIZE_BASELINE_KEYS)
                    || !sizeBaseline.optBoolean("causal", false)
                    || !sizeBaseline.optBoolean(
                    "currentTouchExcluded",
                    false
            )
                    || !"physical_device".equals(
                    sizeBaseline.optString("deviceScope", "")
            )
                    || !"completed_touches_only".equals(
                    sizeBaseline.optString("historyScope", "")
            )
                    || !"log_causal_device_median".equals(
                    sizeBaseline.optString("normalization", "")
            )
                    || !"median".equals(
                    sizeBaseline.optString("aggregation", "")
            )
                    || !"median".equals(
                    sizeBaseline.optString("withinTouchAggregation", "")
            )
                    || sizeBaseline.optInt("minimumTouches", -1)
                    != AispectCausalSizeBaseline.MINIMUM_TOUCHES
                    || sizeBaseline.optInt("maximumTouches", -1)
                    != AispectCausalSizeBaseline.MAXIMUM_TOUCHES
                    || sizeBaseline.optDouble("logRatioClip", -1.0)
                    != AispectCausalSizeBaseline.LOG_RATIO_CLIP
                    || !matches(
                    sizeBaseline.optJSONArray("touchStatistics"),
                    new String[]{
                            "mean",
                            "std",
                            "min",
                            "max",
                            "last_delta"
                    }
            )) {
                return null;
            }
        }
        // 中文注释：v3 必须同时锁定无方向字段的 v5 源合同，避免物理视图相同但源时序不同。
        if (FEATURE_CONTRACT.equals(featureContract)
                && !SOURCE_FEATURE_CONTRACT.equals(
                scaler.optString("sourceFeatureContract", "")
        )) {
            return null;
        }
        String[] expectedSourceFeatureNames;
        if (V4_FEATURE_CONTRACT.equals(featureContract)) {
            expectedSourceFeatureNames = V4_SOURCE_FEATURE_NAMES;
        } else if (FEATURE_CONTRACT.equals(featureContract)) {
            expectedSourceFeatureNames = SOURCE_FEATURE_NAMES;
        } else if (V2_FEATURE_CONTRACT.equals(featureContract)) {
            expectedSourceFeatureNames = V2_SOURCE_FEATURE_NAMES;
        } else {
            expectedSourceFeatureNames = LEGACY_SOURCE_FEATURE_NAMES;
        }
        if (!"all_data_fit_only".equals(scaler.optString("evaluationScope", ""))
                || scaler.optBoolean("deploymentAllowed", true)
                || !hasExactWeightKeys(weights)) {
            return null;
        }
        String[] featureNames = stringArray(scaler.getJSONArray("featureNames"));
        String[] sourceFeatureNames = stringArray(scaler.getJSONArray("sourceFeatureNames"));
        String[] labelOrder = stringArray(scaler.getJSONArray("labelOrder"));
        String[] fingerLabelOrder = stringArray(scaler.getJSONArray("fingerLabelOrder"));
        String[] strengthLabelOrder = stringArray(scaler.getJSONArray("strengthLabelOrder"));
        String[] gateLabelOrder = strengthGate
                ? stringArray(scaler.getJSONArray("gateLabelOrder"))
                : new String[0];
        String[] negativeClassLabels = strengthGate
                ? stringArray(scaler.getJSONArray("negativeClassLabels"))
                : new String[0];
        String[] positiveClassLabels = strengthGate
                ? stringArray(scaler.getJSONArray("positiveClassLabels"))
                : new String[0];
        String[] diagnosticClassLabelOrder = strengthGate
                ? stringArray(scaler.getJSONArray("diagnosticClassLabelOrder"))
                : LEGACY_LABEL_ORDER;
        String[] statsFeatureNames = stringArray(scaler.getJSONArray("statsFeatureNames"));
        int[] sourceFeatureIndices = intArray(scaler.getJSONArray("sourceFeatureIndices"));
        int[] matrixShape = intArray(scaler.getJSONArray("matrixShape"));
        String[] expectedFeatureNames = causalSizeProxy
                ? V4_FEATURE_NAMES
                : FEATURE_NAMES;
        String[] expectedStatsFeatureNames = statsFeatureNames(
                expectedFeatureNames
        );
        if (!Arrays.equals(expectedFeatureNames, featureNames)
                || !Arrays.equals(expectedSourceFeatureNames, sourceFeatureNames)
                || (strengthGate
                ? !Arrays.equals(GATE_LABEL_ORDER, labelOrder)
                : !Arrays.equals(LEGACY_LABEL_ORDER, labelOrder))
                || !Arrays.equals(FINGER_LABEL_ORDER, fingerLabelOrder)
                || !Arrays.equals(STRENGTH_LABEL_ORDER, strengthLabelOrder)
                || strengthGate
                && (!PRIMARY_TASK.equals(scaler.optString("primaryTask", ""))
                || scaler.optInt("gateCount", -1) != GATE_LABEL_ORDER.length
                || !Arrays.equals(GATE_LABEL_ORDER, gateLabelOrder)
                || !Arrays.equals(NEGATIVE_CLASS_LABELS, negativeClassLabels)
                || !Arrays.equals(POSITIVE_CLASS_LABELS, positiveClassLabels)
                || scaler.optInt(
                        "diagnosticClassCount",
                        -1
                ) != LEGACY_LABEL_ORDER.length
                || !Arrays.equals(
                        LEGACY_LABEL_ORDER,
                        diagnosticClassLabelOrder
                ))
                || !Arrays.equals(expectedStatsFeatureNames, statsFeatureNames)
                || !Arrays.equals(SOURCE_FEATURE_INDICES, sourceFeatureIndices)
                || matrixShape.length != 2
                || matrixShape[0] != FRAME_COUNT
                || matrixShape[1] != expectedFeatureNames.length
                || scaler.optInt("frameCount", -1) != FRAME_COUNT
                || scaler.optInt("sequenceChannelCount", -1) != expectedFeatureNames.length
                || scaler.optInt("statsFeatureCount", -1) != STATS_FEATURE_COUNT
                || scaler.optInt(
                        "classCount",
                        -1
                ) != (
                        strengthGate
                                ? GATE_LABEL_ORDER.length
                                : LEGACY_LABEL_ORDER.length
                )
                || scaler.optInt("inputCount", -1) != STATS_FEATURE_COUNT
                || scaler.optInt("hiddenCount", -1) != HIDDEN_COUNT
                || scaler.has("dropoutAtInference")
                && scaler.getBoolean("dropoutAtInference")) {
            return null;
        }
        SequenceChannel[] channels = channels(
                scaler.getJSONArray("sequenceChannels"),
                featureNames
        );
        if (channels == null) {
            return null;
        }
        JSONObject statsScaler = scaler.getJSONObject("statsScaler");
        double[] statsCenter = numberArray(statsScaler.getJSONArray("center"));
        double[] statsScale = numberArray(statsScaler.getJSONArray("scale"));
        double statsClipMin = statsScaler.optDouble("clipMin", -8.0);
        double statsClipMax = statsScaler.optDouble("clipMax", 8.0);
        if (statsCenter.length != STATS_FEATURE_COUNT
                || statsScale.length != STATS_FEATURE_COUNT
                || !allFinite(statsCenter)
                || !allPositiveFinite(statsScale)
                || !Double.isFinite(statsClipMin)
                || !Double.isFinite(statsClipMax)
                || !(statsClipMin < statsClipMax)) {
            return null;
        }
        Tensor shared0Weight = tensor(weights, "shared.0.weight");
        Tensor shared0Bias = tensor(weights, "shared.0.bias");
        Tensor shared3Weight = tensor(weights, "shared.3.weight");
        Tensor shared3Bias = tensor(weights, "shared.3.bias");
        Tensor fingerWeight = tensor(weights, "finger.weight");
        Tensor fingerBias = tensor(weights, "finger.bias");
        Tensor strengthWeight = tensor(weights, "strength.weight");
        Tensor strengthBias = tensor(weights, "strength.bias");
        if (!matches(shared0Weight, HIDDEN_COUNT, STATS_FEATURE_COUNT)
                || !matches(shared0Bias, HIDDEN_COUNT)
                || !matches(shared3Weight, HIDDEN_COUNT, HIDDEN_COUNT)
                || !matches(shared3Bias, HIDDEN_COUNT)
                || !matches(fingerWeight, 2, HIDDEN_COUNT)
                || !matches(fingerBias, 2)
                || !matches(strengthWeight, 2, HIDDEN_COUNT)
                || !matches(strengthBias, 2)) {
            return null;
        }
        return new AispectStatsMlpRuntime(
                featureContract,
                featureNames,
                labelOrder,
                strengthGate,
                causalSizeProxy,
                channels,
                statsFeatureNames,
                statsCenter,
                sanitizeScale(statsScale),
                statsClipMin,
                statsClipMax,
                shared0Weight,
                shared0Bias,
                shared3Weight,
                shared3Bias,
                fingerWeight,
                fingerBias,
                strengthWeight,
                strengthBias
        );
    }

    String featureContract() {
        return featureContract;
    }

    String[] labelOrder() {
        return labelOrder.clone();
    }

    String[] featureNames() {
        return featureNames.clone();
    }

    boolean isStrengthGate() {
        return strengthGate;
    }

    double[] predict(double[][] raw) {
        HeadProbabilities heads = predictHeads(raw);
        if (heads == null) {
            return null;
        }
        if (strengthGate) {
            return heads.strength.clone();
        }
        double[] probabilities = new double[]{
                heads.finger[0] * heads.strength[0],
                heads.finger[0] * heads.strength[1],
                heads.finger[1] * heads.strength[0],
                heads.finger[1] * heads.strength[1]
        };
        double total = probabilities[0] + probabilities[1] + probabilities[2] + probabilities[3];
        if (!(total > 0.0) || !Double.isFinite(total)) {
            return null;
        }
        for (int index = 0; index < probabilities.length; index++) {
            probabilities[index] /= total;
        }
        return probabilities;
    }

    HeadProbabilities predictHeads(double[][] raw) {
        if (raw == null || raw.length != FRAME_COUNT) {
            return null;
        }
        double[][] normalized = new double[FRAME_COUNT][featureNames.length];
        for (int frame = 0; frame < FRAME_COUNT; frame++) {
            if (raw[frame] == null || raw[frame].length != featureNames.length) {
                return null;
            }
            for (int channel = 0; channel < featureNames.length; channel++) {
                double value = raw[frame][channel];
                normalized[frame][channel] = nanExcludedFromStats
                        && !Double.isFinite(value)
                        ? Double.NaN
                        : sequenceChannels[channel].normalize(value);
            }
        }
        double[] stats = statsFeatures(normalized, nanExcludedFromStats);
        for (int index = 0; index < stats.length; index++) {
            stats[index] = clip(
                    (stats[index] - statsCenter[index]) / statsScale[index],
                    statsClipMin,
                    statsClipMax
            );
        }
        double[] firstHidden = gelu(linear(stats, shared0Weight, shared0Bias));
        double[] secondHidden = gelu(linear(firstHidden, shared3Weight, shared3Bias));
        double[] finger = softmax(linear(secondHidden, fingerWeight, fingerBias));
        double[] strength = softmax(linear(secondHidden, strengthWeight, strengthBias));
        if (finger == null || strength == null) {
            return null;
        }
        return new HeadProbabilities(finger, strength);
    }

    static String[] statsFeatureNames() {
        return statsFeatureNames(FEATURE_NAMES);
    }

    static String[] statsFeatureNames(String[] physicalFeatureNames) {
        String[] names = new String[STATS_FEATURE_COUNT];
        int index = 0;
        for (String featureName : physicalFeatureNames) {
            names[index++] = featureName + ".mean";
            names[index++] = featureName + ".std";
            names[index++] = featureName + ".min";
            names[index++] = featureName + ".max";
            names[index++] = featureName + ".last_delta";
        }
        names[index++] = "userAcceleration.magnitude.mean";
        names[index++] = "userAcceleration.magnitude.std";
        names[index++] = "userAcceleration.magnitude.max";
        names[index++] = "userAcceleration.jerk.mean";
        names[index++] = "userAcceleration.jerk.max";
        names[index++] = "rotationRate.magnitude.mean";
        names[index++] = "rotationRate.magnitude.std";
        names[index++] = "rotationRate.magnitude.max";
        names[index++] = "rotationRate.jerk.mean";
        names[index] = "rotationRate.jerk.max";
        return names;
    }

    static double[] statsFeatures(double[][] values) {
        return statsFeatures(values, false);
    }

    static double[] statsFeatures(
            double[][] values,
            boolean excludeNonFinite
    ) {
        if (values == null || values.length == 0 || values[0] == null) {
            throw new IllegalArgumentException("invalid_stats_input");
        }
        int frameCount = values.length;
        int channelCount = values[0].length;
        for (double[] row : values) {
            if (row == null || row.length != channelCount) {
                throw new IllegalArgumentException("invalid_stats_input");
            }
        }
        double[] output = new double[STATS_FEATURE_COUNT];
        int outputIndex = 0;
        for (int channel = 0; channel < channelCount; channel++) {
            double sum = 0.0;
            double minimum = Double.POSITIVE_INFINITY;
            double maximum = Double.NEGATIVE_INFINITY;
            int validCount = 0;
            double firstValid = Double.NaN;
            double lastValid = Double.NaN;
            for (double[] row : values) {
                double value = row[channel];
                if (excludeNonFinite && !Double.isFinite(value)) {
                    continue;
                }
                sum += value;
                minimum = Math.min(minimum, value);
                maximum = Math.max(maximum, value);
                if (validCount == 0) {
                    firstValid = value;
                }
                lastValid = value;
                validCount += 1;
            }
            if (validCount == 0) {
                output[outputIndex++] = 0.0;
                output[outputIndex++] = 0.0;
                output[outputIndex++] = 0.0;
                output[outputIndex++] = 0.0;
                output[outputIndex++] = 0.0;
                continue;
            }
            double mean = sum / validCount;
            double squared = 0.0;
            for (double[] row : values) {
                double value = row[channel];
                if (excludeNonFinite && !Double.isFinite(value)) {
                    continue;
                }
                double delta = value - mean;
                squared += delta * delta;
            }
            output[outputIndex++] = mean;
            output[outputIndex++] = Math.sqrt(squared / validCount);
            output[outputIndex++] = minimum;
            output[outputIndex++] = maximum;
            output[outputIndex++] = lastValid - firstValid;
        }
        double[] accelerationMagnitude = magnitude(values, 2, 5);
        double[] accelerationJerk = jerk(values, 2, 5);
        double[] rotationMagnitude = magnitude(values, 5, 8);
        double[] rotationJerk = jerk(values, 5, 8);
        output[outputIndex++] = mean(accelerationMagnitude);
        output[outputIndex++] = standardDeviation(accelerationMagnitude);
        output[outputIndex++] = maximum(accelerationMagnitude);
        output[outputIndex++] = mean(accelerationJerk);
        output[outputIndex++] = maximum(accelerationJerk);
        output[outputIndex++] = mean(rotationMagnitude);
        output[outputIndex++] = standardDeviation(rotationMagnitude);
        output[outputIndex++] = maximum(rotationMagnitude);
        output[outputIndex++] = mean(rotationJerk);
        output[outputIndex] = maximum(rotationJerk);
        return output;
    }

    static double gelu(double value) {
        return 0.5 * value * (1.0 + erf(value / Math.sqrt(2.0)));
    }

    private static double[] magnitude(double[][] values, int start, int end) {
        double[] output = new double[values.length];
        for (int frame = 0; frame < values.length; frame++) {
            double squared = 0.0;
            for (int channel = start; channel < end; channel++) {
                squared += values[frame][channel] * values[frame][channel];
            }
            output[frame] = Math.sqrt(squared);
        }
        return output;
    }

    private static double[] jerk(double[][] values, int start, int end) {
        double[] output = new double[Math.max(0, values.length - 1)];
        for (int frame = 0; frame < output.length; frame++) {
            double squared = 0.0;
            for (int channel = start; channel < end; channel++) {
                double delta = values[frame + 1][channel] - values[frame][channel];
                squared += delta * delta;
            }
            output[frame] = Math.sqrt(squared);
        }
        return output;
    }

    private static double mean(double[] values) {
        if (values.length == 0) {
            return 0.0;
        }
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }

    private static double standardDeviation(double[] values) {
        if (values.length == 0) {
            return 0.0;
        }
        double mean = mean(values);
        double squared = 0.0;
        for (double value : values) {
            double delta = value - mean;
            squared += delta * delta;
        }
        return Math.sqrt(squared / values.length);
    }

    private static double maximum(double[] values) {
        if (values.length == 0) {
            return 0.0;
        }
        double maximum = values[0];
        for (double value : values) {
            maximum = Math.max(maximum, value);
        }
        return maximum;
    }

    private static double[] linear(double[] input, Tensor weights, Tensor bias) {
        int outputCount = weights.shape[0];
        double[] output = new double[outputCount];
        for (int outputIndex = 0; outputIndex < outputCount; outputIndex++) {
            double sum = bias.values[outputIndex];
            int base = outputIndex * input.length;
            for (int inputIndex = 0; inputIndex < input.length; inputIndex++) {
                sum += input[inputIndex] * weights.values[base + inputIndex];
            }
            output[outputIndex] = sum;
        }
        return output;
    }

    private static double[] gelu(double[] values) {
        double[] output = new double[values.length];
        for (int index = 0; index < values.length; index++) {
            output[index] = gelu(values[index]);
        }
        return output;
    }

    private static double[] softmax(double[] values) {
        double maximum = values[0];
        for (double value : values) {
            maximum = Math.max(maximum, value);
        }
        double total = 0.0;
        double[] output = new double[values.length];
        for (int index = 0; index < values.length; index++) {
            output[index] = Math.exp(values[index] - maximum);
            total += output[index];
        }
        if (!(total > 0.0) || !Double.isFinite(total)) {
            return null;
        }
        for (int index = 0; index < output.length; index++) {
            output[index] /= total;
        }
        return output;
    }

    private static double erf(double value) {
        double sign = value < 0.0 ? -1.0 : 1.0;
        double absolute = Math.abs(value);
        double t = 1.0 / (1.0 + 0.3275911 * absolute);
        double polynomial = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t);
        return sign * (1.0 - polynomial * Math.exp(-absolute * absolute));
    }

    private static SequenceChannel[] channels(JSONArray values, String[] featureNames) {
        try {
            if (values.length() != featureNames.length) {
                return null;
            }
            SequenceChannel[] output = new SequenceChannel[values.length()];
            for (int index = 0; index < values.length(); index++) {
                JSONObject value = values.getJSONObject(index);
                if (!featureNames[index].equals(value.getString("name"))) {
                    return null;
                }
                double center = value.getDouble("center");
                double scale = value.getDouble("scale");
                double clipMin = value.optDouble("clipMin", -8.0);
                double clipMax = value.optDouble("clipMax", 8.0);
                if (!Double.isFinite(center)
                        || !Double.isFinite(scale)
                        || !(scale > 0.0)
                        || !Double.isFinite(clipMin)
                        || !Double.isFinite(clipMax)
                        || !(clipMin < clipMax)) {
                    return null;
                }
                output[index] = new SequenceChannel(
                        center,
                        scale,
                        clipMin,
                        clipMax
                );
            }
            return output;
        } catch (JSONException error) {
            // 中文注释：损坏的模型资产统一按不可加载处理，避免启动期异常穿透。
            return null;
        }
    }

    private static Tensor tensor(JSONObject root, String name) throws JSONException {
        JSONObject object = root.getJSONObject(name);
        int[] shape = intArray(object.getJSONArray("shape"));
        double[] values = numberArray(object.getJSONArray("values"));
        int count = 1;
        for (int dimension : shape) {
            count *= dimension;
        }
        if (values.length != count) {
            throw new JSONException("invalid_tensor_value_count");
        }
        return new Tensor(shape, values);
    }

    private static boolean hasExactWeightKeys(JSONObject weights) {
        return hasExactKeys(weights, WEIGHT_KEYS);
    }

    private static boolean hasExactKeys(
            JSONObject values,
            Set<String> expected
    ) {
        HashSet<String> actual = new HashSet<>();
        Iterator<String> keys = values.keys();
        while (keys.hasNext()) {
            actual.add(keys.next());
        }
        return expected.equals(actual);
    }

    private static boolean matches(Tensor tensor, int... shape) {
        if (tensor.shape.length != shape.length) {
            return false;
        }
        int count = 1;
        for (int index = 0; index < shape.length; index++) {
            if (tensor.shape[index] != shape[index]) {
                return false;
            }
            count *= shape[index];
        }
        return tensor.values.length == count;
    }

    private static double[] sanitizeScale(double[] scale) {
        double[] output = new double[scale.length];
        for (int index = 0; index < scale.length; index++) {
            output[index] = nonZero(scale[index]);
        }
        return output;
    }

    private static double nonZero(double value) {
        return Math.abs(value) < EPSILON ? 1.0 : value;
    }

    private static double clip(double value, double minimum, double maximum) {
        double finite = Double.isFinite(value) ? value : 0.0;
        return Math.max(minimum, Math.min(maximum, finite));
    }

    private static int[] intArray(JSONArray values) throws JSONException {
        int[] output = new int[values.length()];
        for (int index = 0; index < output.length; index++) {
            output[index] = values.getInt(index);
        }
        return output;
    }

    private static String[] stringArray(JSONArray values) throws JSONException {
        String[] output = new String[values.length()];
        for (int index = 0; index < output.length; index++) {
            output[index] = values.getString(index);
        }
        return output;
    }

    private static boolean matches(JSONArray actual, String[] expected)
            throws JSONException {
        return actual != null && Arrays.equals(stringArray(actual), expected);
    }

    private static double[] numberArray(JSONArray values) throws JSONException {
        double[] output = new double[values.length()];
        for (int index = 0; index < output.length; index++) {
            output[index] = values.getDouble(index);
            if (!Double.isFinite(output[index])) {
                throw new JSONException("non_finite_number");
            }
        }
        return output;
    }

    private static boolean allFinite(double[] values) {
        for (double value : values) {
            if (!Double.isFinite(value)) {
                return false;
            }
        }
        return true;
    }

    private static boolean allPositiveFinite(double[] values) {
        for (double value : values) {
            if (!Double.isFinite(value) || !(value > 0.0)) {
                return false;
            }
        }
        return true;
    }

    private static final class SequenceChannel {
        private final double center;
        private final double scale;
        private final double clipMin;
        private final double clipMax;

        private SequenceChannel(double center, double scale, double clipMin, double clipMax) {
            this.center = center;
            this.scale = scale;
            this.clipMin = clipMin;
            this.clipMax = clipMax;
        }

        private double normalize(double value) {
            return clip((value - center) / scale, clipMin, clipMax);
        }
    }

    static final class HeadProbabilities {
        final double[] finger;
        final double[] strength;

        HeadProbabilities(double[] finger, double[] strength) {
            this.finger = finger;
            this.strength = strength;
        }
    }

    private static final class Tensor {
        private final int[] shape;
        private final double[] values;

        private Tensor(int[] shape, double[] values) {
            this.shape = shape;
            this.values = values;
        }
    }
}
