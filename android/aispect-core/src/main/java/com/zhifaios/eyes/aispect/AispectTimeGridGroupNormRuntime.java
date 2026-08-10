package com.zhifaios.eyes.aispect;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

final class AispectTimeGridGroupNormRuntime {
    static final String ARCHITECTURE = "time_grid_groupnorm_cnn_v1";
    private static final double EPSILON = 1e-5;
    private static final int GROUP_COUNT = 4;

    private final String modelId;
    private final String featureContract;
    private final String[] featureNames;
    private final String[] labelOrder;
    private final Channel[] channels;
    private final Tensor conv1;
    private final Tensor conv1Bias;
    private final Tensor norm1Weight;
    private final Tensor norm1Bias;
    private final Tensor conv2;
    private final Tensor conv2Bias;
    private final Tensor norm2Weight;
    private final Tensor norm2Bias;
    private final Tensor output;
    private final Tensor outputBias;

    private AispectTimeGridGroupNormRuntime(
            String modelId,
            String featureContract,
            String[] featureNames,
            String[] labelOrder,
            Channel[] channels,
            Tensor conv1,
            Tensor conv1Bias,
            Tensor norm1Weight,
            Tensor norm1Bias,
            Tensor conv2,
            Tensor conv2Bias,
            Tensor norm2Weight,
            Tensor norm2Bias,
            Tensor output,
            Tensor outputBias
    ) {
        this.modelId = modelId;
        this.featureContract = featureContract;
        this.featureNames = featureNames;
        this.labelOrder = labelOrder;
        this.channels = channels;
        this.conv1 = conv1;
        this.conv1Bias = conv1Bias;
        this.norm1Weight = norm1Weight;
        this.norm1Bias = norm1Bias;
        this.conv2 = conv2;
        this.conv2Bias = conv2Bias;
        this.norm2Weight = norm2Weight;
        this.norm2Bias = norm2Bias;
        this.output = output;
        this.outputBias = outputBias;
    }

    static boolean isSupportedArchitecture(String value) {
        return ARCHITECTURE.equals(value);
    }

    static AispectTimeGridGroupNormRuntime load(String modelId, JSONObject weights, JSONObject scaler) throws JSONException {
        if (!isSupportedArchitecture(scaler.optString("runtimeArchitecture", ""))) {
            return null;
        }
        String featureContract = scaler.getString("featureContract");
        String[] featureNames = stringArray(scaler.getJSONArray("featureNames"));
        String[] labelOrder = stringArray(scaler.getJSONArray("labelOrder"));
        int[] matrixShape = intArray(scaler.getJSONArray("matrixShape"));
        if (matrixShape.length != 2 || matrixShape[0] != 9 || matrixShape[1] != featureNames.length
                || labelOrder.length == 0
                || !AispectCausalPressFeatureBuilder.isTimeGridFeatureContract(featureContract)
                || !matchesFeatureNames(featureContract, featureNames)) {
            return null;
        }
        Channel[] channels = channels(scaler.getJSONArray("channels"), featureNames);
        Tensor conv1 = tensor(weights, "conv1.weight");
        Tensor conv1Bias = tensor(weights, "conv1.bias");
        Tensor norm1Weight = tensor(weights, "norm1.weight");
        Tensor norm1Bias = tensor(weights, "norm1.bias");
        Tensor conv2 = tensor(weights, "conv2.weight");
        Tensor conv2Bias = tensor(weights, "conv2.bias");
        Tensor norm2Weight = tensor(weights, "norm2.weight");
        Tensor norm2Bias = tensor(weights, "norm2.bias");
        Tensor output = tensor(weights, "output.weight");
        Tensor outputBias = tensor(weights, "output.bias");
        if (!validNetwork(featureNames.length, labelOrder.length, conv1, conv1Bias, norm1Weight, norm1Bias,
                conv2, conv2Bias, norm2Weight, norm2Bias, output, outputBias)) {
            return null;
        }
        return new AispectTimeGridGroupNormRuntime(
                modelId,
                featureContract,
                featureNames,
                labelOrder,
                channels,
                conv1,
                conv1Bias,
                norm1Weight,
                norm1Bias,
                conv2,
                conv2Bias,
                norm2Weight,
                norm2Bias,
                output,
                outputBias
        );
    }

    String featureContract() {
        return featureContract;
    }

    int inputChannels() {
        return featureNames.length;
    }

    String[] labelOrder() {
        return labelOrder.clone();
    }

    double[] predict(double[][] raw) {
        if (raw == null || raw.length != 9) {
            return null;
        }
        double[][] input = new double[channels.length][raw.length];
        for (int frame = 0; frame < raw.length; frame++) {
            if (raw[frame] == null || raw[frame].length != channels.length) {
                return null;
            }
            for (int channel = 0; channel < channels.length; channel++) {
                input[channel][frame] = channels[channel].normalize(raw[frame]);
            }
        }
        double[][] hidden = groupNorm(conv1d(input, conv1, conv1Bias), norm1Weight.values, norm1Bias.values, GROUP_COUNT);
        hidden = gelu(hidden);
        hidden = groupNorm(conv1d(hidden, conv2, conv2Bias), norm2Weight.values, norm2Bias.values, GROUP_COUNT);
        hidden = gelu(hidden);
        double[] pooled = meanPool(hidden);
        return softmax(linear(pooled, output, outputBias));
    }

    static double[][] groupNorm(double[][] values, double[] gamma, double[] beta, int groupCount) {
        if (values == null || values.length == 0 || values.length % groupCount != 0
                || gamma == null || beta == null || gamma.length != values.length || beta.length != values.length) {
            throw new IllegalArgumentException("invalid_group_norm_shape");
        }
        int length = values[0].length;
        if (length == 0) {
            throw new IllegalArgumentException("invalid_group_norm_length");
        }
        double[][] output = new double[values.length][length];
        int groupSize = values.length / groupCount;
        for (int group = 0; group < groupCount; group++) {
            int firstChannel = group * groupSize;
            int count = groupSize * length;
            double mean = 0.0;
            for (int channel = firstChannel; channel < firstChannel + groupSize; channel++) {
                if (values[channel] == null || values[channel].length != length) {
                    throw new IllegalArgumentException("invalid_group_norm_values");
                }
                for (int position = 0; position < length; position++) {
                    mean += values[channel][position];
                }
            }
            mean /= count;
            double variance = 0.0;
            for (int channel = firstChannel; channel < firstChannel + groupSize; channel++) {
                for (int position = 0; position < length; position++) {
                    double delta = values[channel][position] - mean;
                    variance += delta * delta;
                }
            }
            double denominator = Math.sqrt(variance / count + EPSILON);
            for (int channel = firstChannel; channel < firstChannel + groupSize; channel++) {
                for (int position = 0; position < length; position++) {
                    output[channel][position] = ((values[channel][position] - mean) / denominator) * gamma[channel] + beta[channel];
                }
            }
        }
        return output;
    }

    static double gelu(double value) {
        return 0.5 * value * (1.0 + erf(value / Math.sqrt(2.0)));
    }

    private static boolean matchesFeatureNames(String featureContract, String[] names) {
        String[] expected = AispectCausalPressFeatureBuilder.featureNames(featureContract);
        if (expected.length != names.length) {
            return false;
        }
        for (int index = 0; index < expected.length; index++) {
            if (!expected[index].equals(names[index])) {
                return false;
            }
        }
        return true;
    }

    private static Channel[] channels(JSONArray array, String[] featureNames) throws JSONException {
        if (array.length() != featureNames.length) {
            throw new JSONException("invalid_channel_count");
        }
        Channel[] output = new Channel[array.length()];
        for (int index = 0; index < array.length(); index++) {
            JSONObject value = array.getJSONObject(index);
            if (!featureNames[index].equals(value.getString("name"))) {
                throw new JSONException("invalid_channel_order");
            }
            output[index] = new Channel(
                    index,
                    value.optString("strategy", "identity"),
                    value.optDouble("center", 0.0),
                    nonZero(value.optDouble("scale", 1.0)),
                    value.isNull("clipMin") ? Double.NaN : value.optDouble("clipMin"),
                    value.isNull("clipMax") ? Double.NaN : value.optDouble("clipMax"),
                    availabilityIndex(featureNames, index, value.optBoolean("requiresTouchAvailability", false))
            );
        }
        return output;
    }

    private static int availabilityIndex(String[] names, int index, boolean required) throws JSONException {
        if (!required) {
            return -1;
        }
        String name = names[index];
        String availabilityName = null;
        if ("pressure".equals(name)) {
            availabilityName = "pressure_available";
        } else if ("size".equals(name)) {
            availabilityName = "size_available";
        } else if ("major_norm".equals(name)) {
            availabilityName = "major_available";
        } else if ("minor_norm".equals(name) || "log_axis_ratio".equals(name)) {
            availabilityName = "minor_available";
        } else if ("pointer_count".equals(name)) {
            availabilityName = "pointer_count_available";
        }
        if (availabilityName == null) {
            throw new JSONException("unknown_touch_availability_feature");
        }
        for (int candidate = 0; candidate < names.length; candidate++) {
            if (availabilityName.equals(names[candidate])) {
                return candidate;
            }
        }
        throw new JSONException("missing_touch_availability_channel");
    }

    private static boolean validNetwork(
            int inputChannels,
            int classCount,
            Tensor conv1,
            Tensor conv1Bias,
            Tensor norm1Weight,
            Tensor norm1Bias,
            Tensor conv2,
            Tensor conv2Bias,
            Tensor norm2Weight,
            Tensor norm2Bias,
            Tensor output,
            Tensor outputBias
    ) {
        return matches(conv1, 32, inputChannels, 3)
                && matches(conv1Bias, 32)
                && matches(norm1Weight, 32)
                && matches(norm1Bias, 32)
                && matches(conv2, 32, 32, 3)
                && matches(conv2Bias, 32)
                && matches(norm2Weight, 32)
                && matches(norm2Bias, 32)
                && matches(output, classCount, 32)
                && matches(outputBias, classCount);
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

    private static double[][] conv1d(double[][] input, Tensor weights, Tensor bias) {
        int outputChannels = weights.shape[0];
        int inputChannels = weights.shape[1];
        int length = input[0].length;
        double[][] output = new double[outputChannels][length];
        for (int outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
            for (int position = 0; position < length; position++) {
                double sum = bias.values[outputChannel];
                for (int inputChannel = 0; inputChannel < inputChannels; inputChannel++) {
                    for (int kernel = 0; kernel < 3; kernel++) {
                        int source = position + kernel - 1;
                        if (source >= 0 && source < length) {
                            int weightIndex = (outputChannel * inputChannels + inputChannel) * 3 + kernel;
                            sum += input[inputChannel][source] * weights.values[weightIndex];
                        }
                    }
                }
                output[outputChannel][position] = sum;
            }
        }
        return output;
    }

    private static double[][] gelu(double[][] values) {
        double[][] output = new double[values.length][values[0].length];
        for (int channel = 0; channel < values.length; channel++) {
            for (int position = 0; position < values[channel].length; position++) {
                output[channel][position] = gelu(values[channel][position]);
            }
        }
        return output;
    }

    private static double[] meanPool(double[][] values) {
        double[] output = new double[values.length];
        for (int channel = 0; channel < values.length; channel++) {
            for (double value : values[channel]) {
                output[channel] += value;
            }
            output[channel] /= values[channel].length;
        }
        return output;
    }

    private static double[] linear(double[] input, Tensor weights, Tensor bias) {
        double[] output = new double[weights.shape[0]];
        for (int outputIndex = 0; outputIndex < output.length; outputIndex++) {
            double sum = bias.values[outputIndex];
            for (int inputIndex = 0; inputIndex < input.length; inputIndex++) {
                sum += input[inputIndex] * weights.values[outputIndex * input.length + inputIndex];
            }
            output[outputIndex] = sum;
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

    private static Tensor tensor(JSONObject root, String key) throws JSONException {
        JSONObject value = root.getJSONObject(key);
        return new Tensor(intArray(value.getJSONArray("shape")), numberArray(value.getJSONArray("values")));
    }

    private static int[] intArray(JSONArray array) throws JSONException {
        int[] output = new int[array.length()];
        for (int index = 0; index < array.length(); index++) {
            output[index] = array.getInt(index);
        }
        return output;
    }

    private static String[] stringArray(JSONArray array) throws JSONException {
        String[] output = new String[array.length()];
        for (int index = 0; index < array.length(); index++) {
            output[index] = array.getString(index);
        }
        return output;
    }

    private static double[] numberArray(JSONArray array) throws JSONException {
        double[] output = new double[array.length()];
        for (int index = 0; index < array.length(); index++) {
            output[index] = array.getDouble(index);
        }
        return output;
    }

    private static double nonZero(double value) {
        return Math.abs(value) < 1e-9 ? 1.0 : value;
    }

    private static final class Channel {
        final int index;
        final String strategy;
        final double center;
        final double scale;
        final double clipMin;
        final double clipMax;
        final int availabilityIndex;

        Channel(int index, String strategy, double center, double scale, double clipMin, double clipMax, int availabilityIndex) {
            this.index = index;
            this.strategy = strategy;
            this.center = center;
            this.scale = scale;
            this.clipMin = clipMin;
            this.clipMax = clipMax;
            this.availabilityIndex = availabilityIndex;
        }

        double normalize(double[] raw) {
            if (availabilityIndex >= 0 && raw[availabilityIndex] < 0.5) {
                return 0.0;
            }
            double value = raw[index];
            if ("robust_median_iqr".equals(strategy)) {
                value = (value - center) / scale;
                if (!Double.isNaN(clipMin)) {
                    value = Math.max(clipMin, value);
                }
                if (!Double.isNaN(clipMax)) {
                    value = Math.min(clipMax, value);
                }
            }
            return Double.isFinite(value) ? value : 0.0;
        }

    }

    private static final class Tensor {
        final int[] shape;
        final double[] values;

        Tensor(int[] shape, double[] values) {
            this.shape = shape;
            this.values = values;
        }
    }
}
