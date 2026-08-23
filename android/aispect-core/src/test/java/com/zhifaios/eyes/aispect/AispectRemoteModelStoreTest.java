package com.zhifaios.eyes.aispect;

import org.junit.Test;

import java.io.File;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public final class AispectRemoteModelStoreTest {
    @Test
    public void activationKeepsPreviousValidatedVersionForRollback() throws Exception {
        Path root = java.nio.file.Files.createTempDirectory("aispect-remote-store-");
        AispectRemoteModelStore store = new AispectRemoteModelStore(root.toFile());
        File first = AispectRemoteBundleValidatorTest.TestBundle.create(
                "causal_time_grid_imu11_touch7_mask5_delta_gravity_v1",
                23,
                "remote_model",
                "1.0.0"
        ).toFile();
        File second = AispectRemoteBundleValidatorTest.TestBundle.create(
                "causal_time_grid_imu11_touch7_mask5_delta_gravity_v1",
                23,
                "remote_model",
                "2.0.0"
        ).toFile();

        store.activate(store.stage(first));
        store.activate(store.stage(second));

        assertEquals("2.0.0", store.loadCurrent().version);
        assertTrue(store.restorePrevious());
        assertEquals("1.0.0", store.loadCurrent().version);
    }
}
