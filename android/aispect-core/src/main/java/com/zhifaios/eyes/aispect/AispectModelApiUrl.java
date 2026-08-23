package com.zhifaios.eyes.aispect;

import java.net.URI;

final class AispectModelApiUrl {
    private AispectModelApiUrl() {
    }

    static String normalize(String value) {
        try {
            String normalized = value == null ? "" : value.trim().replaceAll("/+$", "");
            URI uri = URI.create(normalized);
            if (!("http".equals(uri.getScheme()) || "https".equals(uri.getScheme()))
                    || uri.getHost() == null) {
                throw new IllegalArgumentException("model_api_url_invalid");
            }
            return normalized;
        } catch (RuntimeException error) {
            if ("model_api_url_invalid".equals(error.getMessage())) {
                throw error;
            }
            throw new IllegalArgumentException("model_api_url_invalid", error);
        }
    }
}
