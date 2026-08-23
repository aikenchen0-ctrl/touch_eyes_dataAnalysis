package com.zhifaios.eyes.aispect;

import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

public final class AispectRemoteBundleValidatorTest {
    @Test
    public void acceptsNineByTwentyThreeDeltaGravityBundle() throws Exception {
        Path directory = TestBundle.create(
                "causal_time_grid_imu11_touch7_mask5_delta_gravity_v1",
                23,
                "remote_delta_gravity",
                "2026.08.13"
        );

        AispectRemoteBundleValidator.ValidatedBundle bundle =
                new AispectRemoteBundleValidator().validate(directory.toFile());

        assertEquals("remote_delta_gravity", bundle.modelId);
        assertEquals("2026.08.13", bundle.version);
        assertEquals(9, bundle.frameCount);
        assertEquals(23, bundle.featureNames.length);
    }

    @Test
    public void acceptsNineByTwentyNoGravityBundle() throws Exception {
        Path directory = TestBundle.create(
                "causal_time_grid_touch7_mask5_no_gravity_v1",
                20,
                "remote_no_gravity",
                "2026.08.13"
        );

        AispectRemoteBundleValidator.ValidatedBundle bundle =
                new AispectRemoteBundleValidator().validate(directory.toFile());

        assertEquals(20, bundle.featureNames.length);
    }

    @Test
    public void rejectsChannelCountThatDoesNotMatchFeatureContract() throws Exception {
        Path directory = TestBundle.create(
                "causal_time_grid_imu11_touch7_mask5_delta_gravity_v1",
                20,
                "invalid_channel_count",
                "2026.08.13"
        );

        assertThrows(
                AispectRemoteBundleValidator.BundleException.class,
                () -> new AispectRemoteBundleValidator().validate(directory.toFile())
        );
    }

    @Test
    public void rejectsManifestChecksumMismatch() throws Exception {
        Path directory = TestBundle.create(
                "causal_time_grid_imu11_touch7_mask5_delta_gravity_v1",
                23,
                "invalid_checksum",
                "2026.08.13"
        );
        JSONObject manifest = new JSONObject(new String(Files.readAllBytes(
                directory.resolve("manifest.json")
        ), StandardCharsets.UTF_8));
        manifest.getJSONObject("files").getJSONObject("weights")
                .put("sha256", "0".repeat(64));
        Files.write(
                directory.resolve("manifest.json"),
                manifest.toString().getBytes(StandardCharsets.UTF_8)
        );

        assertThrows(
                AispectRemoteBundleValidator.BundleException.class,
                () -> new AispectRemoteBundleValidator().validate(directory.toFile())
        );
    }

    static final class TestBundle {
        private static final String[] LABELS = new String[]{
                "thumb_light", "thumb_heavy", "index_light", "index_heavy"
        };

        private TestBundle() {
        }

        static Path create(
                String featureContract,
                int channelCount,
                String modelId,
                String version
        ) throws Exception {
            Path directory = Files.createTempDirectory("aispect-remote-bundle-");
            JSONObject scaler = scaler(featureContract, channelCount);
            JSONObject weights = weights(channelCount);
            byte[] scalerBytes = scaler.toString().getBytes(StandardCharsets.UTF_8);
            byte[] weightsBytes = weights.toString().getBytes(StandardCharsets.UTF_8);
            Files.write(directory.resolve("scaler.json"), scalerBytes);
            Files.write(directory.resolve("weights.json"), weightsBytes);
            JSONObject manifest = new JSONObject()
                    .put("schemaVersion", 1)
                    .put("bundleRole", "runtime")
                    .put("platform", "android")
                    .put("modelId", modelId)
                    .put("version", version)
                    .put("modelFormat", "aispect-time-grid-json-v1")
                    .put("featureSchemaId", "aispect-time-grid-groupnorm-v1")
                    .put("runtimeArchitecture", "time_grid_groupnorm_cnn_v1")
                    .put("deploymentAllowed", true)
                    .put("classCount", 4)
                    .put("labelOrder", new JSONArray(LABELS))
                    .put("inputContract", new JSONObject()
                            .put("frameCount", 9)
                            .put("matrixShape", new JSONArray().put(9).put(channelCount))
                            .put("featureContract", featureContract)
                            .put("featureNames", scaler.getJSONArray("featureNames")))
                    .put("files", new JSONObject()
                            .put("weights", identity("weights.json", weightsBytes))
                            .put("scaler", identity("scaler.json", scalerBytes)));
            Files.write(
                    directory.resolve("manifest.json"), manifest.toString().getBytes(StandardCharsets.UTF_8)
            );
            return directory;
        }

        private static JSONObject scaler(String featureContract, int channelCount)
                throws Exception {
            String[] names = AispectCausalPressFeatureBuilder.featureNames(featureContract);
            JSONArray features = new JSONArray();
            JSONArray channels = new JSONArray();
            for (String name : names) {
                features.put(name);
                channels.put(new JSONObject()
                        .put("name", name)
                        .put("center", 0.0)
                        .put("scale", 1.0)
                        .put("strategy", "identity")
                        .put("requiresTouchAvailability", false));
            }
            while (features.length() < channelCount) {
                String name = "unexpected_" + features.length();
                features.put(name);
                channels.put(new JSONObject()
                        .put("name", name)
                        .put("center", 0.0)
                        .put("scale", 1.0)
                        .put("strategy", "identity")
                        .put("requiresTouchAvailability", false));
            }
            while (features.length() > channelCount) {
                features.remove(features.length() - 1);
                channels.remove(channels.length() - 1);
            }
            return new JSONObject()
                    .put("runtimeArchitecture", "time_grid_groupnorm_cnn_v1")
                    .put("featureContract", featureContract)
                    .put("featureNames", features)
                    .put("labelOrder", new JSONArray(LABELS))
                    .put("classCount", 4)
                    .put("frameCount", 9)
                    .put("matrixShape", new JSONArray().put(9).put(channelCount))
                    .put("channels", channels);
        }

        private static JSONObject weights(int channelCount) throws Exception {
            return new JSONObject()
                    .put("conv1.weight", tensor(new int[]{32, channelCount, 3}))
                    .put("conv1.bias", tensor(new int[]{32}))
                    .put("norm1.weight", tensor(new int[]{32}))
                    .put("norm1.bias", tensor(new int[]{32}))
                    .put("conv2.weight", tensor(new int[]{32, 32, 3}))
                    .put("conv2.bias", tensor(new int[]{32}))
                    .put("norm2.weight", tensor(new int[]{32}))
                    .put("norm2.bias", tensor(new int[]{32}))
                    .put("output.weight", tensor(new int[]{4, 32}))
                    .put("output.bias", tensor(new int[]{4}));
        }

        private static JSONObject tensor(int[] shape) throws Exception {
            JSONArray dimensions = new JSONArray();
            JSONArray values = new JSONArray();
            int count = 1;
            for (int value : shape) {
                dimensions.put(value);
                count *= value;
            }
            for (int index = 0; index < count; index++) {
                values.put(0.0);
            }
            return new JSONObject().put("shape", dimensions).put("values", values);
        }

        private static JSONObject identity(String path, byte[] content) throws Exception {
            return new JSONObject()
                    .put("path", path)
                    .put("sizeBytes", content.length)
                    .put("sha256", sha256(content));
        }

        private static String sha256(byte[] content) throws Exception {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(content);
            StringBuilder output = new StringBuilder();
            for (byte value : digest) {
                output.append(String.format("%02x", value & 0xff));
            }
            return output.toString();
        }
    }
}
