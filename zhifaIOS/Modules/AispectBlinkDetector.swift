import ARKit
import QuartzCore

final class AispectBlinkDetector: NSObject {
    struct Config {
        // 中文注释：闭眼阈值，越大越严格。
        var blinkThreshold: Float = 0.29
        // 中文注释：闭眼持续时间阈值。
        var blinkHoldDuration: TimeInterval = 0.18
        // 中文注释：单次检测超时时间。
        var sessionTimeout: TimeInterval = 1.8
    }

    var config = Config()
    var onLatency: ((TimeInterval) -> Void)?
    var onBlink: ((Bool) -> Void)?
    var onStatus: ((String) -> Void)?

    private let session = ARSession()
    private var startTime: TimeInterval = 0
    private var firstSampleTime: TimeInterval?
    private var blinkStartTime: TimeInterval?
    private var isRunning = false
    private var timeoutWork: DispatchWorkItem?

    override init() {
        super.init()
        session.delegate = self
    }

    func start() {
        guard !isRunning else { return }
        guard ARFaceTrackingConfiguration.isSupported else {
            onStatus?("blink_unsupported")
            return
        }
        // 中文注释：启动人脸跟踪并记录起始时间。
        isRunning = true
        startTime = CACurrentMediaTime()
        firstSampleTime = nil
        blinkStartTime = nil
        onStatus?("blink_starting")
        let configuration = ARFaceTrackingConfiguration()
        configuration.isLightEstimationEnabled = true
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        scheduleTimeout()
    }

    func stop() {
        guard isRunning else { return }
        // 中文注释：停止会话并清理超时任务。
        isRunning = false
        timeoutWork?.cancel()
        timeoutWork = nil
        session.pause()
        onStatus?("blink_stopped")
    }

    private func scheduleTimeout() {
        timeoutWork?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            self.onBlink?(false)
            self.stop()
        }
        timeoutWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + config.sessionTimeout, execute: work)
    }

    private func handleFrame(_ frame: ARFrame) {
        let now = CACurrentMediaTime()
        if firstSampleTime == nil {
            // 中文注释：首帧回调用于计算启动延迟。
            firstSampleTime = now
            onLatency?(now - startTime)
            onStatus?("blink_tracking")
        }
        for anchor in frame.anchors {
            guard let faceAnchor = anchor as? ARFaceAnchor else { continue }
            let left = (faceAnchor.blendShapes[.eyeBlinkLeft] as? NSNumber)?.floatValue ?? 0
            let right = (faceAnchor.blendShapes[.eyeBlinkRight] as? NSNumber)?.floatValue ?? 0
            let isClosed = left >= config.blinkThreshold && right >= config.blinkThreshold
            if isClosed {
                // 中文注释：进入闭眼状态后开始计时。
                if blinkStartTime == nil {
                    blinkStartTime = now
                }
                if let start = blinkStartTime, now - start >= config.blinkHoldDuration {
                    onBlink?(true)
                    stop()
                    break
                }
            } else {
                blinkStartTime = nil
            }
        }
    }
}

extension AispectBlinkDetector: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        handleFrame(frame)
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        onStatus?("blink_error")
        stop()
    }
}
