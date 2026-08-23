package com.zhifaios.eyes.aispect;

public final class AispectModelRuntimeController {
    private static final AispectModelRuntimeController SHARED =
            new AispectModelRuntimeController();

    private Runnable reloader;
    private boolean touchActive;
    private boolean reloadPending;

    private AispectModelRuntimeController() {
    }

    public static AispectModelRuntimeController shared() {
        return SHARED;
    }

    public void bind(Runnable nextReloader) {
        Runnable run = null;
        synchronized (this) {
            reloader = nextReloader;
            if (reloadPending && !touchActive && reloader != null) {
                reloadPending = false;
                run = reloader;
            }
        }
        run(run);
    }

    public synchronized void unbind() {
        reloader = null;
    }

    public synchronized void onTouchStarted() {
        touchActive = true;
    }

    public void onTouchFinished() {
        Runnable run = null;
        synchronized (this) {
            touchActive = false;
            if (reloadPending && reloader != null) {
                reloadPending = false;
                run = reloader;
            }
        }
        run(run);
    }

    public void requestReload() {
        Runnable run = null;
        synchronized (this) {
            if (touchActive || reloader == null) {
                reloadPending = true;
            } else {
                run = reloader;
            }
        }
        run(run);
    }

    private static void run(Runnable runnable) {
        if (runnable != null) {
            runnable.run();
        }
    }
}
