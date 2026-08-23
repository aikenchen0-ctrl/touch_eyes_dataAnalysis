package com.zhifaios.eyes.aispect;

import org.json.JSONObject;

import java.io.IOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

/** Sends one complete v6 collection sample to the management API. */
public final class AispectCollectionUploadClient {
    public enum Result {
        SUCCESS,
        RETRYABLE_FAILURE,
        PERMANENT_FAILURE
    }

    interface ConnectionFactory {
        HttpURLConnection open(URL url) throws IOException;
    }

    private final String endpoint;
    private final String appId;
    private final String deviceId;
    private final String deviceName;
    private final String appVersion;
    private final String sdkVersion;
    private final int connectTimeoutMs;
    private final int readTimeoutMs;
    private final ConnectionFactory connectionFactory;

    public AispectCollectionUploadClient(
            String baseUrl,
            String appId,
            String deviceId,
            String deviceName,
            String appVersion,
            String sdkVersion,
            int connectTimeoutMs,
            int readTimeoutMs
    ) {
        this(baseUrl, appId, deviceId, deviceName, appVersion, sdkVersion,
                connectTimeoutMs, readTimeoutMs, url -> (HttpURLConnection) url.openConnection());
    }

    AispectCollectionUploadClient(
            String baseUrl,
            String appId,
            String deviceId,
            String deviceName,
            String appVersion,
            String sdkVersion,
            int connectTimeoutMs,
            int readTimeoutMs,
            ConnectionFactory connectionFactory
    ) {
        if (baseUrl == null || baseUrl.trim().isEmpty()) {
            throw new IllegalArgumentException("collection upload base URL is required");
        }
        this.endpoint = trimTrailingSlash(baseUrl.trim()) + "/api/v1/collection-samples";
        this.appId = required(appId, "appId");
        this.deviceId = required(deviceId, "deviceId");
        this.deviceName = deviceName == null ? "" : deviceName;
        this.appVersion = appVersion == null ? "unknown" : appVersion;
        this.sdkVersion = sdkVersion == null ? "unknown" : sdkVersion;
        this.connectTimeoutMs = Math.max(1, connectTimeoutMs);
        this.readTimeoutMs = Math.max(1, readTimeoutMs);
        this.connectionFactory = connectionFactory;
    }

    public Result upload(JSONObject sample) {
        HttpURLConnection connection = null;
        try {
            connection = connectionFactory.open(new URL(endpoint));
            connection.setRequestMethod("POST");
            connection.setDoOutput(true);
            connection.setConnectTimeout(connectTimeoutMs);
            connection.setReadTimeout(readTimeoutMs);
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setRequestProperty("Accept", "application/json");
            connection.setRequestProperty("X-App-Id", appId);
            connection.setRequestProperty("X-Device-Id", deviceId);
            if (!deviceName.isEmpty()) {
                connection.setRequestProperty("X-Device-Name", deviceName);
            }
            connection.setRequestProperty("X-App-Version", appVersion);
            connection.setRequestProperty("X-Sdk-Version", sdkVersion);
            byte[] payload = sample.toString().getBytes(StandardCharsets.UTF_8);
            try (OutputStream output = connection.getOutputStream()) {
                output.write(payload);
            }
            int responseCode = connection.getResponseCode();
            if (responseCode == HttpURLConnection.HTTP_OK
                    || responseCode == HttpURLConnection.HTTP_CREATED) {
                return Result.SUCCESS;
            }
            if (responseCode == 422
                    || (responseCode >= 400 && responseCode < 500
                    && responseCode != 408
                    && responseCode != 429)) {
                return Result.PERMANENT_FAILURE;
            }
            return Result.RETRYABLE_FAILURE;
        } catch (IOException | RuntimeException error) {
            return Result.RETRYABLE_FAILURE;
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }

    String endpoint() {
        return endpoint;
    }

    private static String trimTrailingSlash(String value) {
        int end = value.length();
        while (end > 0 && value.charAt(end - 1) == '/') {
            end -= 1;
        }
        return value.substring(0, end);
    }

    private static String required(String value, String name) {
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalArgumentException(name + " is required");
        }
        return value;
    }
}
