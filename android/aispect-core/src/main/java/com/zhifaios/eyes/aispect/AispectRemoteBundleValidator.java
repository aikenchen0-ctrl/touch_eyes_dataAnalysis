package com.zhifaios.eyes.aispect;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Pattern;

public final class AispectRemoteBundleValidator {
    public static final String FEATURE_SCHEMA_ID = "aispect-time-grid-groupnorm-v1";
    public static final String MODEL_FORMAT = "aispect-time-grid-json-v1";
    private static final String[] LABELS = new String[]{
            "thumb_light", "thumb_heavy", "index_light", "index_heavy"
    };
    private static final Pattern SAFE_SEGMENT = Pattern.compile(
            "^[A-Za-z0-9._+\\-]+$"
    );

    public static final class BundleException extends Exception {
        BundleException(String code) {
            super(code);
        }

        BundleException(String code, Throwable cause) {
            super(code, cause);
        }
    }

    public static final class ValidatedBundle {
        public final String modelId;
        public final String version;
        public final String featureContract;
        public final String[] featureNames;
        public final String[] labelOrder;
        public final int frameCount;
        public final int channelCount;
        public final JSONObject weights;
        public final JSONObject scaler;

        ValidatedBundle(
                String modelId,
                String version,
                String featureContract,
                String[] featureNames,
                String[] labelOrder,
                int frameCount,
                int channelCount,
                JSONObject weights,
                JSONObject scaler
        ) {
            this.modelId = modelId;
            this.version = version;
            this.featureContract = featureContract;
            this.featureNames = featureNames.clone();
            this.labelOrder = labelOrder.clone();
            this.frameCount = frameCount;
            this.channelCount = channelCount;
            this.weights = weights;
            this.scaler = scaler;
        }
    }

    public ValidatedBundle validate(File directory) throws BundleException {
        try {
            byte[] manifestBytes = read(directory, "manifest.json");
            byte[] weightsBytes = read(directory, "weights.json");
            byte[] scalerBytes = read(directory, "scaler.json");
            JSONObject manifest = object(manifestBytes);
            JSONObject weights = object(weightsBytes);
            JSONObject scaler = object(scalerBytes);
            validateManifest(manifest, weightsBytes, scalerBytes);

            String modelId = text(manifest, "modelId");
            String version = text(manifest, "version");
            requireSafeSegment(modelId);
            requireSafeSegment(version);
            String featureContract = text(scaler, "featureContract");
            String[] featureNames = strings(scaler.getJSONArray("featureNames"));
            String[] expectedFeatures = AispectCausalPressFeatureBuilder.featureNames(
                    featureContract
            );
            require(expectedFeatures.length == 20 || expectedFeatures.length == 23);
            require(Arrays.equals(expectedFeatures, featureNames));
            requireStrings(scaler.getJSONArray("labelOrder"), LABELS);
            require(scaler.getInt("classCount") == 4);
            require(scaler.getInt("frameCount") == 9);
            require(AispectTimeGridGroupNormRuntime.ARCHITECTURE.equals(
                    scaler.getString("runtimeArchitecture")
            ));
            JSONArray shape = scaler.getJSONArray("matrixShape");
            require(shape.length() == 2);
            require(shape.getInt(0) == 9 && shape.getInt(1) == featureNames.length);
            validateChannels(scaler.getJSONArray("channels"), featureNames);
            validateInputContract(manifest.getJSONObject("inputContract"), featureContract, featureNames);
            validateWeights(weights, featureNames.length);
            AispectTimeGridGroupNormRuntime runtime = AispectTimeGridGroupNormRuntime.load(
                    modelId, weights, scaler
            );
            require(runtime != null);
            return new ValidatedBundle(
                    modelId,
                    version,
                    featureContract,
                    featureNames,
                    LABELS,
                    9,
                    featureNames.length,
                    weights,
                    scaler
            );
        } catch (BundleException error) {
            throw error;
        } catch (IOException error) {
            throw new BundleException("bundle_read_failed", error);
        } catch (JSONException | RuntimeException error) {
            throw new BundleException("bundle_contract_mismatch", error);
        }
    }

    private static void validateManifest(
            JSONObject manifest,
            byte[] weightsBytes,
            byte[] scalerBytes
    ) throws JSONException, BundleException {
        require(manifest.getInt("schemaVersion") == 1);
        require("runtime".equals(manifest.getString("bundleRole")));
        require("android".equals(manifest.getString("platform")));
        require(MODEL_FORMAT.equals(manifest.getString("modelFormat")));
        require(FEATURE_SCHEMA_ID.equals(manifest.getString("featureSchemaId")));
        require(AispectTimeGridGroupNormRuntime.ARCHITECTURE.equals(
                manifest.getString("runtimeArchitecture")
        ));
        require(manifest.getBoolean("deploymentAllowed"));
        require(manifest.getInt("classCount") == 4);
        requireStrings(manifest.getJSONArray("labelOrder"), LABELS);
        JSONObject files = manifest.getJSONObject("files");
        require(files.length() == 2);
        validateIdentity(files.getJSONObject("weights"), "weights.json", weightsBytes);
        validateIdentity(files.getJSONObject("scaler"), "scaler.json", scalerBytes);
    }

    private static void validateIdentity(
            JSONObject identity,
            String expectedPath,
            byte[] content
    ) throws JSONException, BundleException {
        require(expectedPath.equals(identity.getString("path")));
        require(identity.getLong("sizeBytes") == content.length);
        require(sha256(content).equals(identity.getString("sha256")));
    }

    private static void validateInputContract(
            JSONObject input,
            String featureContract,
            String[] featureNames
    ) throws JSONException, BundleException {
        require(input.getInt("frameCount") == 9);
        require(featureContract.equals(input.getString("featureContract")));
        requireStrings(input.getJSONArray("featureNames"), featureNames);
        JSONArray shape = input.getJSONArray("matrixShape");
        require(shape.length() == 2);
        require(shape.getInt(0) == 9 && shape.getInt(1) == featureNames.length);
    }

    private static void validateChannels(JSONArray channels, String[] names)
            throws JSONException, BundleException {
        require(channels.length() == names.length);
        for (int index = 0; index < names.length; index++) {
            JSONObject channel = channels.getJSONObject(index);
            require(names[index].equals(channel.getString("name")));
            require(finite(channel.getDouble("center")));
            double scale = channel.getDouble("scale");
            require(finite(scale) && scale > 0.0);
            require(channel.has("strategy"));
            require(channel.has("requiresTouchAvailability"));
        }
    }

    private static void validateWeights(JSONObject weights, int channels)
            throws JSONException, BundleException {
        LinkedHashMap<String, int[]> expected = new LinkedHashMap<>();
        expected.put("conv1.weight", new int[]{32, channels, 3});
        expected.put("conv1.bias", new int[]{32});
        expected.put("norm1.weight", new int[]{32});
        expected.put("norm1.bias", new int[]{32});
        expected.put("conv2.weight", new int[]{32, 32, 3});
        expected.put("conv2.bias", new int[]{32});
        expected.put("norm2.weight", new int[]{32});
        expected.put("norm2.bias", new int[]{32});
        expected.put("output.weight", new int[]{4, 32});
        expected.put("output.bias", new int[]{4});
        require(weights.length() == expected.size());
        for (Map.Entry<String, int[]> entry : expected.entrySet()) {
            JSONObject tensor = weights.getJSONObject(entry.getKey());
            JSONArray shape = tensor.getJSONArray("shape");
            int count = 1;
            int[] expectedShape = entry.getValue();
            require(shape.length() == expectedShape.length);
            for (int index = 0; index < expectedShape.length; index++) {
                require(shape.getInt(index) == expectedShape[index]);
                count *= expectedShape[index];
            }
            JSONArray values = tensor.getJSONArray("values");
            require(values.length() == count);
            for (int index = 0; index < values.length(); index++) {
                require(finite(values.getDouble(index)));
            }
        }
    }

    private static byte[] read(File directory, String name) throws IOException {
        if (directory == null || !directory.isDirectory()) {
            throw new IOException("bundle_directory_missing");
        }
        File file = new File(directory, name);
        if (!file.isFile()) {
            throw new IOException("bundle_file_missing");
        }
        return AispectFileIO.readBytes(file);
    }

    private static JSONObject object(byte[] bytes) throws JSONException {
        return new JSONObject(new String(bytes, StandardCharsets.UTF_8));
    }

    private static String text(JSONObject object, String key) throws JSONException, BundleException {
        String value = object.getString(key);
        require(!value.trim().isEmpty());
        return value;
    }

    private static String[] strings(JSONArray values) throws JSONException {
        String[] output = new String[values.length()];
        for (int index = 0; index < output.length; index++) {
            output[index] = values.getString(index);
        }
        return output;
    }

    private static void requireStrings(JSONArray values, String[] expected)
            throws JSONException, BundleException {
        require(values.length() == expected.length);
        for (int index = 0; index < expected.length; index++) {
            require(expected[index].equals(values.getString(index)));
        }
    }

    private static void requireSafeSegment(String value) throws BundleException {
        require(SAFE_SEGMENT.matcher(value).matches());
        require(!".".equals(value) && !"..".equals(value));
    }

    private static String sha256(byte[] content) throws BundleException {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(content);
            StringBuilder output = new StringBuilder();
            for (byte value : digest) {
                output.append(String.format("%02x", value & 0xff));
            }
            return output.toString();
        } catch (NoSuchAlgorithmException error) {
            throw new BundleException("sha256_unavailable", error);
        }
    }

    private static boolean finite(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }

    private static void require(boolean value) throws BundleException {
        if (!value) {
            throw new BundleException("bundle_contract_mismatch");
        }
    }
}
