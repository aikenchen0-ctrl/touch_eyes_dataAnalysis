package com.zhifaios.eyes.aispect;

import android.content.Context;

import androidx.work.BackoffPolicy;
import androidx.work.Constraints;
import androidx.work.Data;
import androidx.work.ExistingWorkPolicy;
import androidx.work.NetworkType;
import androidx.work.OneTimeWorkRequest;
import androidx.work.WorkManager;

import java.util.concurrent.TimeUnit;

public final class AispectModelUpdateScheduler {
    static final String WORK_NAME = "aispect-model-assignment-refresh";
    static final String INPUT_BASE_URL = "baseUrl";
    static final String INPUT_APP_ID = "appId";
    static final String INPUT_SDK_VERSION = "sdkVersion";
    static final String INPUT_APP_VERSION = "appVersion";
    static final String INPUT_TRIGGER = "triggerSource";

    private AispectModelUpdateScheduler() {
    }

    public static void refreshNow(
            Context context,
            String baseUrl,
            String appId,
            String sdkVersion,
            String appVersion,
            AispectModelUpdateTrigger trigger
    ) {
        if (context == null || baseUrl == null || baseUrl.trim().isEmpty()) {
            return;
        }
        try {
            AispectModelApiUrl.normalize(baseUrl);
        } catch (IllegalArgumentException error) {
            return;
        }
        Data input = new Data.Builder()
                .putString(INPUT_BASE_URL, baseUrl)
                .putString(INPUT_APP_ID, appId)
                .putString(INPUT_SDK_VERSION, sdkVersion)
                .putString(INPUT_APP_VERSION, appVersion)
                .putString(INPUT_TRIGGER, trigger == null
                        ? AispectModelUpdateTrigger.FOREGROUND.wireName()
                        : trigger.wireName())
                .build();
        OneTimeWorkRequest request = new OneTimeWorkRequest.Builder(
                AispectModelRefreshWorker.class
        ).setConstraints(new Constraints.Builder()
                .setRequiredNetworkType(NetworkType.CONNECTED)
                .build())
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30L, TimeUnit.SECONDS)
                .setInputData(input)
                .build();
        WorkManager.getInstance(context.getApplicationContext()).enqueueUniqueWork(
                WORK_NAME,
                ExistingWorkPolicy.REPLACE,
                request
        );
    }
}
