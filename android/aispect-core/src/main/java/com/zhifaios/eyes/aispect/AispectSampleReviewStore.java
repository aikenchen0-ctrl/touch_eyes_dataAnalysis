package com.zhifaios.eyes.aispect;

import android.util.AtomicFile;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

/** 复核以同目录旁车文件原子写入；历史和原始样本均保留。 */
final class AispectSampleReviewStore {
    static void write(File sample, String sampleId, String status) throws IOException, JSONException {
        AispectCollectionProgress.validateReviewStatus(status);
        if (!sample.isFile()) throw new IOException("review_sample_missing");
        File directory = new File(sample.getParentFile(), "annotations");
        if (!directory.isDirectory() && !directory.mkdirs()) throw new IOException("review_directory_failed");
        // UUID 校验阻止样本标识被当成相对路径使用。
        UUID.fromString(sampleId);
        AtomicFile file = new AtomicFile(new File(directory, sampleId + ".json"));
        JSONObject document;
        if (file.getBaseFile().exists() || new File(file.getBaseFile() + ".bak").exists()) {
            document = new JSONObject(new String(file.readFully(), StandardCharsets.UTF_8));
            if (!sampleId.equals(document.getString("sampleId"))
                    || document.getInt("schemaVersion") != 1
                    || !"touch_sample_review_v1".equals(document.getString("contract"))
                    || !sample.getName().equals(document.getString("sampleFileName"))) {
                throw new IOException("review_identity_mismatch");
            }
            JSONArray previous = document.getJSONArray("history");
            if (previous.length() == 0) throw new IOException("review_history_missing");
            for (int index = 0; index < previous.length(); index++) {
                JSONObject event = previous.getJSONObject(index);
                AispectCollectionProgress.validateReviewStatus(event.getString("status"));
                if (event.getInt("revision") != index + 1) throw new IOException("review_revision_invalid");
            }
            if (!previous.getJSONObject(previous.length() - 1).getString("status")
                    .equals(document.getString("labelStatus"))) throw new IOException("review_status_mismatch");
        } else {
            document = new JSONObject();
            document.put("schemaVersion", 1);
            document.put("contract", "touch_sample_review_v1");
            document.put("sampleId", sampleId);
            document.put("sampleFileName", sample.getName());
            document.put("history", new JSONArray());
        }
        JSONArray history = document.getJSONArray("history");
        JSONObject event = new JSONObject();
        event.put("revision", history.length() + 1);
        event.put("status", status);
        event.put("timestampMs", System.currentTimeMillis());
        event.put("source", "user_review");
        history.put(event);
        document.put("labelStatus", status);
        FileOutputStream stream = null;
        try {
            stream = file.startWrite();
            stream.write(document.toString(2).getBytes(StandardCharsets.UTF_8));
            file.finishWrite(stream);
        } catch (IOException error) {
            file.failWrite(stream);
            throw error;
        }
    }
}
