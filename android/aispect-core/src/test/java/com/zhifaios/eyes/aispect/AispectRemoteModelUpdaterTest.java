package com.zhifaios.eyes.aispect;

import org.json.JSONObject;
import org.junit.Test;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.util.HashMap;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

public final class AispectRemoteModelUpdaterTest {
    @Test
    public void shaMismatchDoesNotActivateCandidate() throws Exception {
        File bundle = AispectRemoteBundleValidatorTest.TestBundle.create(
                "causal_time_grid_imu11_touch7_mask5_delta_gravity_v1",
                23,
                "remote_update_model",
                "1.0.0"
        ).toFile();
        HashMap<String, byte[]> content = new HashMap<>();
        for (String name : new String[]{"manifest", "weights", "scaler"}) {
            content.put(name, Files.readAllBytes(new File(bundle, name + ".json").toPath()));
        }
        HashMap<String, AispectModelAssignment.Artifact> artifacts = new HashMap<>();
        for (String name : content.keySet()) {
            String checksum = sha256(content.get(name));
            if ("weights".equals(name)) {
                checksum = "0" + checksum.substring(1);
            }
            artifacts.put(name, new AispectModelAssignment.Artifact(
                    name,
                    "https://example.invalid/" + name,
                    checksum,
                    content.get(name).length
            ));
        }
        AispectModelAssignment assignment = assignment(artifacts);
        File root = Files.createTempDirectory("aispect-remote-updater-").toFile();
        AispectRemoteModelStore store = new AispectRemoteModelStore(root);
        AispectRemoteModelUpdater updater = new AispectRemoteModelUpdater(
                new FakeApi(assignment, content),
                store,
                new File(root, "downloads"),
                "com.zhifaios.eyes.android",
                "device-1",
                "1.0.0",
                "1.0.0",
                "manual"
        );

        AispectRemoteModelUpdater.Result result = updater.update();

        assertEquals(AispectRemoteModelUpdater.Status.FAILED, result.status);
        assertEquals("artifact_sha256_mismatch", result.error);
        assertNull(store.loadCurrent());
    }

    private static AispectModelAssignment assignment(
            HashMap<String, AispectModelAssignment.Artifact> artifacts
    ) throws Exception {
        JSONObject model = new JSONObject()
                .put("id", "remote_update_model")
                .put("version", "1.0.0")
                .put("featureSchemaId", AispectRemoteBundleValidator.FEATURE_SCHEMA_ID)
                .put("modelType", AispectRemoteBundleValidator.MODEL_FORMAT);
        for (String name : new String[]{"manifest", "weights", "scaler"}) {
            AispectModelAssignment.Artifact artifact = artifacts.get(name);
            String title = name.substring(0, 1).toUpperCase() + name.substring(1);
            model.put(name + "URL", artifact.url)
                    .put(name + "SHA256", artifact.sha256)
                    .put(name + "SizeBytes", artifact.sizeBytes);
        }
        return AispectModelAssignment.parse(new JSONObject().put("models", new org.json.JSONArray().put(model)));
    }

    private static String sha256(byte[] values) throws Exception {
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(values);
        StringBuilder output = new StringBuilder();
        for (byte value : digest) {
            output.append(String.format("%02x", value & 0xff));
        }
        return output.toString();
    }

    private static final class FakeApi implements AispectRemoteModelUpdater.RemoteApi {
        private final AispectModelAssignment assignment;
        private final HashMap<String, byte[]> content;

        FakeApi(AispectModelAssignment assignment, HashMap<String, byte[]> content) {
            this.assignment = assignment;
            this.content = content;
        }

        @Override
        public AispectModelAssignment fetchAssignment(
                String appId,
                String deviceId,
                String sdkVersion,
                String appVersion,
                String currentModelId,
                String currentModelVersion
        ) {
            return assignment;
        }

        @Override
        public byte[] download(AispectModelAssignment.Artifact artifact) {
            return content.get(artifact.logicalName);
        }

        @Override
        public void report(AispectRemoteModelUpdater.UpdateEvent event) {
        }
    }
}
