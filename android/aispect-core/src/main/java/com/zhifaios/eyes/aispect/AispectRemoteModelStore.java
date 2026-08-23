package com.zhifaios.eyes.aispect;

import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

public final class AispectRemoteModelStore {
    private final File root;
    private final File stagingRoot;
    private final File versionsRoot;
    private final File stateFile;
    private final AispectRemoteBundleValidator validator;

    public static final class StoreException extends Exception {
        StoreException(String code) {
            super(code);
        }

        StoreException(String code, Throwable cause) {
            super(code, cause);
        }
    }

    public static final class StagedModel {
        private final File directory;

        private StagedModel(File directory) {
            this.directory = directory;
        }
    }

    public AispectRemoteModelStore(File root) throws StoreException {
        this(root, new AispectRemoteBundleValidator());
    }

    public AispectRemoteModelStore(
            File root,
            AispectRemoteBundleValidator validator
    ) throws StoreException {
        if (root == null || validator == null) {
            throw new StoreException("model_store_invalid");
        }
        this.root = root;
        this.validator = validator;
        this.stagingRoot = new File(root, "staging");
        this.versionsRoot = new File(root, "versions");
        this.stateFile = new File(root, "state.json");
        if ((!root.isDirectory() && !root.mkdirs())
                || (!stagingRoot.isDirectory() && !stagingRoot.mkdirs())
                || (!versionsRoot.isDirectory() && !versionsRoot.mkdirs())) {
            throw new StoreException("model_store_unavailable");
        }
    }

    public StagedModel stage(File sourceDirectory) throws StoreException {
        File staged = new File(stagingRoot, UUID.randomUUID().toString());
        try {
            validator.validate(sourceDirectory);
            if (!staged.mkdir()) {
                throw new IOException("staging_create_failed");
            }
            for (String name : new String[]{"manifest.json", "weights.json", "scaler.json"}) {
                AispectFileIO.copy(
                        new File(sourceDirectory, name),
                        new File(staged, name)
                );
            }
            validator.validate(staged);
            return new StagedModel(staged);
        } catch (AispectRemoteBundleValidator.BundleException | IOException error) {
            deleteRecursively(staged);
            throw new StoreException("model_stage_failed", error);
        }
    }

    public synchronized void activate(StagedModel staged) throws StoreException {
        if (staged == null || !inside(staged.directory, stagingRoot)) {
            throw new StoreException("model_stage_invalid");
        }
        try {
            AispectRemoteBundleValidator.ValidatedBundle bundle = validator.validate(staged.directory);
            File modelRoot = new File(versionsRoot, bundle.modelId);
            File target = new File(modelRoot, bundle.version);
            if (!modelRoot.isDirectory() && !modelRoot.mkdirs()) {
                throw new IOException("model_directory_create_failed");
            }
            JSONObject state = readState();
            String current = state.optString("current", null);
            if (target.exists()) {
                validator.validate(target);
                deleteRecursively(staged.directory);
            } else {
                move(staged.directory, target);
            }
            String relative = relative(target);
            if (!relative.equals(current)) {
                writeState(new JSONObject()
                        .put("current", relative)
                        .put("previous", current == null ? JSONObject.NULL : current));
            }
        } catch (Exception error) {
            throw new StoreException("model_activate_failed", error);
        }
    }

    public synchronized AispectRemoteBundleValidator.ValidatedBundle loadCurrent() {
        File current = stateDirectory("current");
        if (current == null) {
            return null;
        }
        try {
            return validator.validate(current);
        } catch (AispectRemoteBundleValidator.BundleException error) {
            return null;
        }
    }

    public synchronized boolean restorePrevious() throws StoreException {
        File previous = stateDirectory("previous");
        if (previous == null) {
            return false;
        }
        try {
            validator.validate(previous);
            writeState(new JSONObject()
                    .put("current", relative(previous))
                    .put("previous", JSONObject.NULL));
            return true;
        } catch (AispectRemoteBundleValidator.BundleException error) {
            return false;
        } catch (IOException | org.json.JSONException error) {
            throw new StoreException("model_restore_failed", error);
        }
    }

    public synchronized File currentDirectory() {
        return stateDirectory("current");
    }

    private File stateDirectory(String key) {
        try {
            String value = readState().optString(key, null);
            if (value == null || value.isEmpty()) {
                return null;
            }
            File directory = new File(root, value).getCanonicalFile();
            return inside(directory, versionsRoot) && directory.isDirectory()
                    ? directory : null;
        } catch (Exception error) {
            return null;
        }
    }

    private JSONObject readState() throws IOException {
        if (!stateFile.isFile()) {
            return new JSONObject();
        }
        try {
            return new JSONObject(new String(
                    AispectFileIO.readBytes(stateFile), StandardCharsets.UTF_8
            ));
        } catch (Exception error) {
            throw new IOException("model_state_invalid", error);
        }
    }

    private void writeState(JSONObject state) throws IOException {
        File temporary = new File(root, ".state." + UUID.randomUUID() + ".tmp");
        AispectFileIO.writeBytes(temporary,
                (state.toString() + "\n").getBytes(StandardCharsets.UTF_8));
        move(temporary, stateFile);
    }

    private String relative(File file) throws IOException {
        String rootPath = root.getCanonicalPath();
        String filePath = file.getCanonicalPath();
        if (!filePath.startsWith(rootPath + File.separator)) {
            throw new IOException("model_path_invalid");
        }
        return filePath.substring(rootPath.length() + 1)
                .replace(File.separatorChar, '/');
    }

    private static void move(File source, File target) throws IOException {
        AispectFileIO.moveReplace(source, target);
    }

    private static boolean inside(File child, File parent) {
        try {
            String parentPath = parent.getCanonicalPath();
            String childPath = child.getCanonicalPath();
            return childPath.startsWith(parentPath + File.separator);
        } catch (IOException error) {
            return false;
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
