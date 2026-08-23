package com.zhifaios.eyes.aispect;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

public final class AispectModelAssignment {
    private static final Pattern SHA256 = Pattern.compile("^[0-9a-f]{64}$");
    public final String modelId;
    public final String version;
    public final String featureSchemaId;
    public final String modelFormat;
    public final Map<String, Artifact> artifacts;

    public static final class Artifact {
        public final String logicalName;
        public final String url;
        public final String sha256;
        public final long sizeBytes;

        Artifact(String logicalName, String url, String sha256, long sizeBytes) {
            this.logicalName = logicalName;
            this.url = url;
            this.sha256 = sha256;
            this.sizeBytes = sizeBytes;
        }
    }

    private AispectModelAssignment(
            String modelId,
            String version,
            String featureSchemaId,
            String modelFormat,
            Map<String, Artifact> artifacts
    ) {
        this.modelId = modelId;
        this.version = version;
        this.featureSchemaId = featureSchemaId;
        this.modelFormat = modelFormat;
        this.artifacts = artifacts;
    }

    public static AispectModelAssignment parse(JSONObject payload)
            throws JSONException {
        JSONArray models = payload.getJSONArray("models");
        if (models.length() != 1) {
            throw new JSONException("assignment_model_count_invalid");
        }
        JSONObject model = models.getJSONObject(0);
        HashMap<String, Artifact> artifacts = new HashMap<>();
        for (String name : new String[]{"manifest", "weights", "scaler"}) {
            String url = model.getString(name + "URL");
            String digest = model.getString(name + "SHA256");
            long size = model.getLong(name + "SizeBytes");
            if (url.isEmpty() || !SHA256.matcher(digest).matches() || size <= 0) {
                throw new JSONException("assignment_artifact_invalid");
            }
            artifacts.put(name, new Artifact(name, url, digest, size));
        }
        return new AispectModelAssignment(
                model.getString("id"),
                model.getString("version"),
                model.getString("featureSchemaId"),
                model.getString("modelType"),
                artifacts
        );
    }
}
