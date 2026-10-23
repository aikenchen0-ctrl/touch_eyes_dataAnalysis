package com.zhifaios.eyes.aispect;

import java.util.HashMap;
import java.util.Map;

/** 本进程进度只在落盘成功后增加；复核可撤回计数但不删除样本。 */
public final class AispectCollectionProgress {
    public static final class Entry {
        public final String sampleId;
        public final String key;
        private String reviewStatus = "task_assigned";

        private Entry(String sampleId, String key) {
            this.sampleId = sampleId;
            this.key = key;
        }
    }

    private final Map<String, Integer> counts = new HashMap<>();

    public Entry recordSaved(String sampleId, String key) {
        Entry entry = new Entry(sampleId, key);
        counts.put(key, count(key) + 1);
        return entry;
    }

    public void review(Entry entry, String status) {
        validateReviewStatus(status);
        int delta = (eligible(status) ? 1 : 0) - (eligible(entry.reviewStatus) ? 1 : 0);
        counts.put(entry.key, Math.max(0, count(entry.key) + delta));
        entry.reviewStatus = status;
    }

    public int count(String key) {
        Integer value = counts.get(key);
        return value == null ? 0 : value;
    }

    public static void validateReviewStatus(String status) {
        if (!"confirmed".equals(status) && !"mistake".equals(status) && !"uncertain".equals(status)) {
            throw new IllegalArgumentException("invalid_review_status");
        }
    }

    private static boolean eligible(String status) {
        return "task_assigned".equals(status) || "confirmed".equals(status);
    }
}
