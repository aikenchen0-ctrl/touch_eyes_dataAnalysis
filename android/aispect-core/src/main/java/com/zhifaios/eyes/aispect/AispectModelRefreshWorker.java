package com.zhifaios.eyes.aispect;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Build;

import androidx.annotation.NonNull;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

import java.io.File;
import java.util.UUID;

public final class AispectModelRefreshWorker extends Worker {
    private static final String PREFERENCES = "aispect_remote_model_state";
    private static final String INSTALL_ID = "install_id";

    public AispectModelRefreshWorker(
            @NonNull Context context,
            @NonNull WorkerParameters parameters
    ) {
        super(context, parameters);
    }

    @NonNull
    @Override
    public Result doWork() {
        String baseUrl = getInputData().getString(AispectModelUpdateScheduler.INPUT_BASE_URL);
        String appId = getInputData().getString(AispectModelUpdateScheduler.INPUT_APP_ID);
        String sdkVersion = getInputData().getString(AispectModelUpdateScheduler.INPUT_SDK_VERSION);
        String appVersion = getInputData().getString(AispectModelUpdateScheduler.INPUT_APP_VERSION);
        if (empty(baseUrl) || empty(appId) || empty(sdkVersion) || empty(appVersion)) {
            return Result.success();
        }
        Context context = getApplicationContext();
        AispectModelHttpClient api;
        AispectRemoteModelStore store;
        try {
            api = new AispectModelHttpClient(baseUrl);
            store = new AispectRemoteModelStore(
                    new File(context.getFilesDir(), "aispect_runtime_models")
            );
        } catch (RuntimeException | AispectRemoteModelStore.StoreException error) {
            return Result.success();
        }
        String deviceId = installId(context);
        register(api, store, appId, deviceId, sdkVersion, appVersion);
        AispectRemoteModelUpdater updater = new AispectRemoteModelUpdater(
                api,
                store,
                new File(context.getCacheDir(), "aispect_model_downloads"),
                appId,
                deviceId,
                sdkVersion,
                appVersion,
                AispectModelUpdateTrigger.fromWireName(
                        getInputData().getString(AispectModelUpdateScheduler.INPUT_TRIGGER)
                ).wireName()
        );
        AispectRemoteModelUpdater.Result result = updater.update();
        if (result.status == AispectRemoteModelUpdater.Status.UPDATED) {
            register(api, store, appId, deviceId, sdkVersion, appVersion);
            AispectModelRuntimeController.shared().requestReload();
        }
        return result.status == AispectRemoteModelUpdater.Status.FAILED && result.retryable
                ? Result.retry() : Result.success();
    }

    private static void register(
            AispectModelHttpClient api,
            AispectRemoteModelStore store,
            String appId,
            String deviceId,
            String sdkVersion,
            String appVersion
    ) {
        AispectRemoteBundleValidator.ValidatedBundle current = store.loadCurrent();
        try {
            api.registerDevice(
                    appId,
                    deviceId,
                    Build.MANUFACTURER + " " + Build.MODEL,
                    sdkVersion,
                    appVersion,
                    current == null ? null : current.modelId,
                    current == null ? null : current.version
            );
        } catch (Exception ignored) {
            // 设备登记失败不阻塞模型拉取，下一次联网工作会重试登记。
        }
    }

    private static String installId(Context context) {
        SharedPreferences preferences = context.getSharedPreferences(
                PREFERENCES, Context.MODE_PRIVATE
        );
        String existing = preferences.getString(INSTALL_ID, "");
        if (existing != null && !existing.isEmpty()) {
            return existing;
        }
        String created = UUID.randomUUID().toString();
        preferences.edit().putString(INSTALL_ID, created).commit();
        return created;
    }

    private static boolean empty(String value) {
        return value == null || value.trim().isEmpty();
    }
}
