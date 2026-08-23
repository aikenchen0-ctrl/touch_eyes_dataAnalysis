package com.zhifaios.eyes.aispect;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

public final class AispectModelHttpClient implements AispectRemoteModelUpdater.RemoteApi {
    private static final int CONNECT_TIMEOUT_MS = 10_000;
    private static final int READ_TIMEOUT_MS = 30_000;
    private static final int MAX_ASSIGNMENT_BYTES = 1024 * 1024;
    private final String baseUrl;

    public AispectModelHttpClient(String baseUrl) {
        this.baseUrl = AispectModelApiUrl.normalize(baseUrl);
    }

    @Override
    public AispectModelAssignment fetchAssignment(
            String appId,
            String deviceId,
            String sdkVersion,
            String appVersion,
            String currentModelId,
            String currentModelVersion
    ) throws Exception {
        StringBuilder url = new StringBuilder(baseUrl)
                .append("/api/v1/model-assignment?appId=").append(encoded(appId))
                .append("&deviceId=").append(encoded(deviceId))
                .append("&platform=android")
                .append("&sdkVersion=").append(encoded(sdkVersion))
                .append("&appVersion=").append(encoded(appVersion));
        if (currentModelId != null && currentModelVersion != null) {
            url.append("&currentModelId=").append(encoded(currentModelId))
                    .append("&currentModelVersion=").append(encoded(currentModelVersion));
        }
        HttpURLConnection connection = open(url.toString(), "GET");
        try {
            int status = connection.getResponseCode();
            if (status == 404) {
                return null;
            }
            if (status != 200) {
                throw new IOException("assignment_http_" + status);
            }
            return AispectModelAssignment.parse(new JSONObject(new String(
                    read(connection.getInputStream(), MAX_ASSIGNMENT_BYTES),
                    StandardCharsets.UTF_8
            )));
        } finally {
            connection.disconnect();
        }
    }

    @Override
    public byte[] download(AispectModelAssignment.Artifact artifact) throws Exception {
        if (artifact == null || artifact.sizeBytes > Integer.MAX_VALUE - 1) {
            throw new IOException("artifact_size_invalid");
        }
        HttpURLConnection connection = open(artifact.url, "GET");
        try {
            int status = connection.getResponseCode();
            if (status != 200) {
                throw new IOException("artifact_http_" + status);
            }
            return read(connection.getInputStream(), (int) artifact.sizeBytes + 1);
        } finally {
            connection.disconnect();
        }
    }

    @Override
    public void report(AispectRemoteModelUpdater.UpdateEvent event) throws Exception {
        postAccepted("/api/v1/model-update-events", new JSONObject()
                .put("appId", event.appId)
                .put("platform", "android")
                .put("deviceId", event.deviceId)
                .put("fromModelId", value(event.fromModelId))
                .put("fromModelVersion", value(event.fromModelVersion))
                .put("toModelId", value(event.toModelId))
                .put("toModelVersion", value(event.toModelVersion))
                .put("status", event.status)
                .put("error", value(event.error))
                .put("sdkVersion", event.sdkVersion)
                .put("appVersion", event.appVersion)
                .put("triggerSource", event.triggerSource), "update_report_failed");
    }

    public void registerDevice(
            String appId,
            String deviceId,
            String deviceName,
            String sdkVersion,
            String appVersion,
            String currentModelId,
            String currentModelVersion
    ) throws Exception {
        JSONObject payload = new JSONObject()
                .put("platform", "android")
                .put("appId", appId)
                .put("deviceId", deviceId)
                .put("deviceName", value(deviceName))
                .put("sdkVersion", sdkVersion)
                .put("appVersion", appVersion)
                .put("currentModelId", value(currentModelId))
                .put("currentModelVersion", value(currentModelVersion));
        postAccepted("/api/v1/devices/register", payload, "device_registration_failed");
    }

    private void postAccepted(String path, JSONObject payload, String errorCode)
            throws Exception {
        byte[] content = payload.toString().getBytes(StandardCharsets.UTF_8);
        HttpURLConnection connection = open(baseUrl + path, "POST");
        connection.setDoOutput(true);
        connection.setRequestProperty("Content-Type", "application/json; charset=utf-8");
        connection.setFixedLengthStreamingMode(content.length);
        try {
            connection.getOutputStream().write(content);
            if (connection.getResponseCode() != 202) {
                throw new IOException(errorCode);
            }
        } finally {
            connection.disconnect();
        }
    }

    private static HttpURLConnection open(String url, String method) throws IOException {
        HttpURLConnection connection = (HttpURLConnection) URI.create(url).toURL().openConnection();
        connection.setRequestMethod(method);
        connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
        connection.setReadTimeout(READ_TIMEOUT_MS);
        connection.setUseCaches(false);
        connection.setInstanceFollowRedirects(false);
        connection.setRequestProperty("Accept", "application/json");
        return connection;
    }

    private static byte[] read(InputStream input, int limit) throws IOException {
        try (InputStream stream = input;
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[16 * 1024];
            int total = 0;
            int count;
            while ((count = stream.read(buffer)) >= 0) {
                total += count;
                if (total > limit) {
                    throw new IOException("response_size_exceeded");
                }
                output.write(buffer, 0, count);
            }
            return output.toByteArray();
        }
    }

    private static String encoded(String value) throws Exception {
        return URLEncoder.encode(value, "UTF-8");
    }

    private static Object value(String value) {
        return value == null ? JSONObject.NULL : value;
    }
}
