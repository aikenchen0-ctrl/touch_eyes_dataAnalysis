package com.zhifaios.eyes.aispect;

import java.io.File;
import java.io.IOException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.UUID;

public final class AispectRemoteModelUpdater {
    public enum Status { NO_ASSIGNMENT, NO_CHANGE, UPDATED, FAILED }

    public static final class Result {
        public final Status status;
        public final String error;
        public final boolean retryable;

        Result(Status status, String error, boolean retryable) {
            this.status = status;
            this.error = error;
            this.retryable = retryable;
        }
    }

    public static final class UpdateEvent {
        public final String eventId;
        public final String appId;
        public final String deviceId;
        public final String fromModelId;
        public final String fromModelVersion;
        public final String toModelId;
        public final String toModelVersion;
        public final String status;
        public final String error;
        public final String sdkVersion;
        public final String appVersion;
        public final String triggerSource;

        public UpdateEvent(
                String eventId,
                String appId,
                String deviceId,
                String fromModelId,
                String fromModelVersion,
                String toModelId,
                String toModelVersion,
                String status,
                String error,
                String sdkVersion,
                String appVersion,
                String triggerSource
        ) {
            this.eventId = eventId == null ? UUID.randomUUID().toString() : eventId;
            this.appId = appId;
            this.deviceId = deviceId;
            this.fromModelId = fromModelId;
            this.fromModelVersion = fromModelVersion;
            this.toModelId = toModelId;
            this.toModelVersion = toModelVersion;
            this.status = status;
            this.error = error;
            this.sdkVersion = sdkVersion;
            this.appVersion = appVersion;
            this.triggerSource = triggerSource;
        }
    }

    public interface RemoteApi {
        AispectModelAssignment fetchAssignment(
                String appId,
                String deviceId,
                String sdkVersion,
                String appVersion,
                String currentModelId,
                String currentModelVersion
        ) throws Exception;

        byte[] download(AispectModelAssignment.Artifact artifact) throws Exception;

        void report(UpdateEvent event) throws Exception;
    }

    private final RemoteApi api;
    private final AispectRemoteModelStore store;
    private final File downloadRoot;
    private final String appId;
    private final String deviceId;
    private final String sdkVersion;
    private final String appVersion;
    private final String triggerSource;

    public AispectRemoteModelUpdater(
            RemoteApi api,
            AispectRemoteModelStore store,
            File downloadRoot,
            String appId,
            String deviceId,
            String sdkVersion,
            String appVersion,
            String triggerSource
    ) {
        this.api = api;
        this.store = store;
        this.downloadRoot = downloadRoot;
        this.appId = appId;
        this.deviceId = deviceId;
        this.sdkVersion = sdkVersion;
        this.appVersion = appVersion;
        this.triggerSource = triggerSource;
    }

    public Result update() {
        AispectRemoteBundleValidator.ValidatedBundle current = store.loadCurrent();
        String currentId = current == null ? null : current.modelId;
        String currentVersion = current == null ? null : current.version;
        AispectModelAssignment assignment = null;
        File download = null;
        try {
            assignment = api.fetchAssignment(
                    appId, deviceId, sdkVersion, appVersion, currentId, currentVersion
            );
            if (assignment == null) {
                return new Result(Status.NO_ASSIGNMENT, null, false);
            }
            if (assignment.modelId.equals(currentId)
                    && assignment.version.equals(currentVersion)) {
                return new Result(Status.NO_CHANGE, null, false);
            }
            if (!AispectRemoteBundleValidator.FEATURE_SCHEMA_ID.equals(
                    assignment.featureSchemaId
            ) || !AispectRemoteBundleValidator.MODEL_FORMAT.equals(assignment.modelFormat)) {
                throw new IOException("assignment_contract_mismatch");
            }
            if (!downloadRoot.isDirectory() && !downloadRoot.mkdirs()) {
                throw new IOException("download_directory_unavailable");
            }
            download = new File(downloadRoot, UUID.randomUUID().toString());
            if (!download.mkdir()) {
                throw new IOException("download_directory_unavailable");
            }
            for (String name : new String[]{"manifest", "weights", "scaler"}) {
                AispectModelAssignment.Artifact artifact = assignment.artifacts.get(name);
                if (artifact == null) {
                    throw new IOException("assignment_artifact_missing");
                }
                byte[] content = api.download(artifact);
                if (content == null || content.length != artifact.sizeBytes) {
                    throw new IOException("artifact_size_mismatch");
                }
                if (!sha256(content).equals(artifact.sha256)) {
                    throw new IOException("artifact_sha256_mismatch");
                }
                AispectFileIO.writeBytes(new File(download, name + ".json"), content);
            }
            AispectRemoteBundleValidator.ValidatedBundle candidate =
                    new AispectRemoteBundleValidator().validate(download);
            if (!assignment.modelId.equals(candidate.modelId)
                    || !assignment.version.equals(candidate.version)) {
                throw new IOException("assignment_manifest_mismatch");
            }
            AispectRemoteModelStore.StagedModel staged = store.stage(download);
            store.activate(staged);
            report(new UpdateEvent(null, appId, deviceId, currentId, currentVersion,
                    assignment.modelId, assignment.version, "success", null,
                    sdkVersion, appVersion, triggerSource));
            return new Result(Status.UPDATED, null, false);
        } catch (Exception error) {
            String code = errorCode(error);
            report(new UpdateEvent(null, appId, deviceId, currentId, currentVersion,
                    assignment == null ? null : assignment.modelId,
                    assignment == null ? null : assignment.version,
                    "failed", code, sdkVersion, appVersion, triggerSource));
            return new Result(Status.FAILED, code, isRetryable(error));
        } finally {
            deleteRecursively(download);
        }
    }

    private void report(UpdateEvent event) {
        try {
            api.report(event);
        } catch (Exception ignored) {
            // 更新回执失败不影响已完成的本地原子激活。
        }
    }

    private static boolean isRetryable(Exception error) {
        if (error instanceof java.net.SocketTimeoutException
                || error instanceof java.net.ConnectException
                || error instanceof java.net.UnknownHostException) {
            return true;
        }
        String message = error.getMessage();
        return message != null && message.matches("^(assignment|artifact)_http_5\\d\\d$");
    }

    private static String errorCode(Exception error) {
        String message = error.getMessage();
        return message != null && message.matches("^[a-z0-9_]+$")
                ? message : "model_update_failed";
    }

    private static String sha256(byte[] content) throws IOException {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(content);
            StringBuilder output = new StringBuilder();
            for (byte value : digest) {
                output.append(String.format("%02x", value & 0xff));
            }
            return output.toString();
        } catch (NoSuchAlgorithmException error) {
            throw new IOException("sha256_unavailable", error);
        }
    }

    private static void deleteRecursively(File file) {
        if (file == null || !file.exists()) {
            return;
        }
        File[] children = file.listFiles();
        if (children != null) {
            for (File child : children) {
                deleteRecursively(child);
            }
        }
        file.delete();
    }
}
