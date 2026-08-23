package com.zhifaios.eyes.aispect;

import org.json.JSONObject;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class AispectCollectionUploadClientTest {
    @Test
    public void buildsCollectionEndpointAndRequiredHeaders() throws Exception {
        JSONObject sample = new JSONObject()
                .put("schemaVersion", 6)
                .put("platform", "android")
                .put("id", "sample-1");
        RecordingConnection connection = new RecordingConnection();
        AispectCollectionUploadClient client = new AispectCollectionUploadClient(
                "http://154.217.234.113:8765/",
                "com.zhifaios.eyes.android",
                "device-1",
                "PHK110",
                "1.0",
                "v6",
                5000,
                5000,
                url -> {
                    connection.url = url;
                    return connection;
                }
        );

        AispectCollectionUploadClient.Result result = client.upload(sample);

        assertEquals(AispectCollectionUploadClient.Result.SUCCESS, result);
        assertEquals("http://154.217.234.113:8765/api/v1/collection-samples", connection.url.toString());
        assertEquals("com.zhifaios.eyes.android", connection.requestProperties.get("X-App-Id"));
        assertEquals("device-1", connection.requestProperties.get("X-Device-Id"));
        assertEquals("PHK110", connection.requestProperties.get("X-Device-Name"));
        assertEquals("application/json", connection.requestProperties.get("Content-Type"));
        assertEquals("android", new JSONObject(connection.body.toString("UTF-8")).getString("platform"));
    }

    @Test
    public void acceptsCreatedAndAlreadyProcessedResponses() throws Exception {
        JSONObject sample = new JSONObject().put("id", "sample-1");
        for (int code : new int[]{200, 201}) {
            RecordingConnection connection = new RecordingConnection();
            connection.responseCode = code;
            AispectCollectionUploadClient client = client(connection);
            assertEquals(AispectCollectionUploadClient.Result.SUCCESS, client.upload(sample));
        }
    }

    @Test
    public void classifiesValidationAsPermanentAndTransportAsRetryable() throws Exception {
        JSONObject sample = new JSONObject().put("id", "sample-1");
        RecordingConnection invalid = new RecordingConnection();
        invalid.responseCode = 422;
        assertEquals(AispectCollectionUploadClient.Result.PERMANENT_FAILURE, client(invalid).upload(sample));

        RecordingConnection unavailable = new RecordingConnection();
        unavailable.responseCode = 503;
        assertEquals(AispectCollectionUploadClient.Result.RETRYABLE_FAILURE, client(unavailable).upload(sample));
    }

    private static AispectCollectionUploadClient client(RecordingConnection connection) {
        return new AispectCollectionUploadClient(
                "http://127.0.0.1:8765",
                "app",
                "device",
                "name",
                "1",
                "sdk",
                100,
                100,
                url -> {
                    connection.url = url;
                    return connection;
                }
        );
    }

    private static final class RecordingConnection extends HttpURLConnection {
        final ByteArrayOutputStream body = new ByteArrayOutputStream();
        final java.util.Map<String, String> requestProperties = new java.util.HashMap<>();
        URL url;
        int responseCode = 201;

        RecordingConnection() throws Exception {
            super(new URL("http://127.0.0.1"));
        }

        @Override public void disconnect() { }
        @Override public boolean usingProxy() { return false; }
        @Override public void connect() { }
        @Override public void setRequestProperty(String key, String value) { requestProperties.put(key, value); }
        @Override public void setDoOutput(boolean dooutput) { }
        @Override public void setConnectTimeout(int timeout) { }
        @Override public void setReadTimeout(int timeout) { }
        @Override public java.io.OutputStream getOutputStream() { return body; }
        @Override public int getResponseCode() { return responseCode; }
        @Override public java.io.InputStream getInputStream() { return new ByteArrayInputStream(new byte[0]); }
        @Override public java.io.InputStream getErrorStream() { return new ByteArrayInputStream(new byte[0]); }
    }
}
