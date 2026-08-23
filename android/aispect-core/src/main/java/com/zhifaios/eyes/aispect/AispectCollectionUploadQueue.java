package com.zhifaios.eyes.aispect;

import android.content.Context;

import androidx.annotation.NonNull;
import androidx.work.Constraints;
import androidx.work.ExistingWorkPolicy;
import androidx.work.NetworkType;
import androidx.work.OneTimeWorkRequest;
import androidx.work.WorkManager;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

/** Durable local queue for samples awaiting management-server acknowledgement. */
public final class AispectCollectionUploadQueue {
    private static final String PREFERENCES = "aispect_collection_upload";
    private static final String WORK_NAME = "aispect_collection_sample_upload";
    private static final String BASE_URL = "base_url";
    private static final String APP_ID = "app_id";
    private static final String DEVICE_ID = "device_id";
    private static final String DEVICE_NAME = "device_name";
    private static final String APP_VERSION = "app_version";
    private static final String SDK_VERSION = "sdk_version";

    public static final class Config {
        public String baseUrl;
        public String appId;
        public String deviceId;
        public String deviceName;
        public String appVersion;
        public String sdkVersion = "schema6";
    }

    private final Context context;
    private final File pendingDirectory;

    public AispectCollectionUploadQueue(Context context, Config config) {
        this.context = context.getApplicationContext();
        this.pendingDirectory = new File(this.context.getFilesDir(), "collection-upload-pending");
        if (!pendingDirectory.exists()) {
            //noinspection ResultOfMethodCallIgnored
            pendingDirectory.mkdirs();
        }
        configure(config);
    }

    public void enqueue(JSONObject sample) throws IOException {
        if (sample == null || !sample.has("id")) {
            throw new IllegalArgumentException("sample id is required");
        }
        String id;
        try {
            id = sample.getString("id");
        } catch (org.json.JSONException error) {
            throw new IOException("sample id is invalid", error);
        }
        String safeId = id.replaceAll("[^A-Za-z0-9._-]", "_");
        if (safeId.isEmpty()) {
            safeId = UUID.randomUUID().toString();
        }
        File destination = new File(pendingDirectory, safeId + ".json");
        File temporary = new File(pendingDirectory, "." + safeId + ".tmp");
        try (FileOutputStream output = new FileOutputStream(temporary)) {
            output.write(sample.toString().getBytes(StandardCharsets.UTF_8));
        }
        if (!temporary.renameTo(destination)) {
            throw new IOException("failed to atomically queue collection sample");
        }
        schedule();
    }

    public void configure(Config config) {
        if (config == null || config.baseUrl == null || config.baseUrl.trim().isEmpty()) {
            return;
        }
        context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
                .edit()
                .putString(BASE_URL, config.baseUrl)
                .putString(APP_ID, config.appId)
                .putString(DEVICE_ID, config.deviceId)
                .putString(DEVICE_NAME, config.deviceName)
                .putString(APP_VERSION, config.appVersion)
                .putString(SDK_VERSION, config.sdkVersion)
                .apply();
    }

    private void schedule() {
        Constraints constraints = new Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .build();
        OneTimeWorkRequest request = new OneTimeWorkRequest.Builder(UploadWorker.class)
                .setConstraints(constraints)
                .build();
        WorkManager.getInstance(context).enqueueUniqueWork(WORK_NAME, ExistingWorkPolicy.KEEP, request);
    }

    public static final class UploadWorker extends Worker {
        public UploadWorker(@NonNull Context context, @NonNull WorkerParameters parameters) {
            super(context, parameters);
        }

        @NonNull
        @Override
        public Result doWork() {
            Context context = getApplicationContext();
            android.content.SharedPreferences preferences = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE);
            String baseUrl = preferences.getString(BASE_URL, "");
            if (baseUrl == null || baseUrl.trim().isEmpty()) {
                return Result.success();
            }
            AispectCollectionUploadClient client = new AispectCollectionUploadClient(
                    baseUrl,
                    preferences.getString(APP_ID, context.getPackageName()),
                    preferences.getString(DEVICE_ID, ""),
                    preferences.getString(DEVICE_NAME, ""),
                    preferences.getString(APP_VERSION, "unknown"),
                    preferences.getString(SDK_VERSION, "schema6"),
                    10_000,
                    20_000
            );
            File pendingDirectory = new File(context.getFilesDir(), "collection-upload-pending");
            File[] files = pendingDirectory.listFiles((directory, name) -> name.endsWith(".json"));
            if (files == null || files.length == 0) {
                return Result.success();
            }
            boolean retry = false;
            for (File file : files) {
                try {
                    JSONObject sample = new JSONObject(read(file));
                    AispectCollectionUploadClient.Result result = client.upload(sample);
                    if (result == AispectCollectionUploadClient.Result.SUCCESS) {
                        //noinspection ResultOfMethodCallIgnored
                        file.delete();
                    } else if (result == AispectCollectionUploadClient.Result.RETRYABLE_FAILURE) {
                        retry = true;
                    } else {
                        File failed = new File(pendingDirectory, file.getName() + ".failed");
                        //noinspection ResultOfMethodCallIgnored
                        file.renameTo(failed);
                    }
                } catch (Exception error) {
                    retry = true;
                }
            }
            return retry ? Result.retry() : Result.success();
        }

        private static String read(File file) throws IOException {
            try (FileInputStream input = new FileInputStream(file)) {
                byte[] bytes = new byte[(int) file.length()];
                int offset = 0;
                while (offset < bytes.length) {
                    int count = input.read(bytes, offset, bytes.length - offset);
                    if (count < 0) {
                        break;
                    }
                    offset += count;
                }
                return new String(bytes, 0, offset, StandardCharsets.UTF_8);
            }
        }
    }
}
