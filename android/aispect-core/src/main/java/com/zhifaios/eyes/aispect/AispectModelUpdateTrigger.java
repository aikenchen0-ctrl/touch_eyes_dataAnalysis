package com.zhifaios.eyes.aispect;

import java.util.Locale;

public enum AispectModelUpdateTrigger {
    FOREGROUND,
    MANUAL,
    PERIODIC;

    String wireName() {
        return name().toLowerCase(Locale.US);
    }

    static AispectModelUpdateTrigger fromWireName(String value) {
        if (value == null) {
            return FOREGROUND;
        }
        try {
            return valueOf(value.toUpperCase(Locale.US));
        } catch (IllegalArgumentException error) {
            return FOREGROUND;
        }
    }
}
