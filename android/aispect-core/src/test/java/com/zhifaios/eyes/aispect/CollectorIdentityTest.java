package com.zhifaios.eyes.aispect;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public final class CollectorIdentityTest {
    @Test
    public void trimsUserNameAndRejectsBlankValues() {
        assertEquals("  Alice  ", AispectModels.CollectorIdentity.normalizeUserName("  Alice  "));
        assertTrue(AispectModels.CollectorIdentity.isValidUserName("Alice"));
        assertFalse(AispectModels.CollectorIdentity.isValidUserName("   "));
        assertFalse(AispectModels.CollectorIdentity.isValidUserName(null));
    }
}
