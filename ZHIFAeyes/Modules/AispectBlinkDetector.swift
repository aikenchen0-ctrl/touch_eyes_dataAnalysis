import ARKit
import AVFoundation
import ImageIO
import QuartzCore
import UIKit
import Vision

final class AispectBlinkDetector: NSObject {
    struct Config {
        // 中文注释：手动调参时建议优先只改下面 5 个 Vision 参数：
        // 中文注释：`visionClosedEyeRatioThreshold`、`visionStrongClosedEyeRatioThreshold`、
        // 中文注释：`visionReopenEyeRatioThreshold`、`blinkHoldDuration`、
        // 中文注释：`visionStrongBlinkHoldDuration`。
        // 中文注释：如果“几乎总是判定为是”，优先把闭眼阈值调小、把保持时长调大。
        // 中文注释：如果“明明闭眼了却经常判定为否”，优先把闭眼阈值调大、把保持时长调小。
        // 中文注释：ARKit 兜底分支沿用 blendShapes 的闭眼阈值。
        // 中文注释：仅在 Vision 不可用时生效；调大更严格，调小更敏感。
        var blinkThreshold: Float = 0.2
        // 中文注释：常规闭眼持续时间阈值，优先保证“可用准确率”下的较低判定延迟。
        // 中文注释：如果判定阶段偏久，先把它调小；如果误判“是”太多，先把它调大。
        var blinkHoldDuration: TimeInterval = 0.06
        // 中文注释：单次检测超时时间。
        // 中文注释：只影响最长等待时间，不直接决定“是/否”阈值。
        var sessionTimeout: TimeInterval = 1.8
        // 中文注释：Vision 分支的归一化眼睑开合阈值，越小越接近闭眼。
        // 中文注释：这是最核心的常规闭眼阈值；调小会更严格、更不容易报“是”，调大会更敏感。
        var visionClosedEyeRatioThreshold: CGFloat = 0.60
        // 中文注释：Vision 分支的强闭眼阈值，命中后走更短保持窗口。
        // 中文注释：如果“明显没闭眼也很容易报是”，把它调小；如果想让重闭眼更快命中，可以略微调大。
        var visionStrongClosedEyeRatioThreshold: CGFloat = 0.46
        // 中文注释：Vision 分支的开眼回滞阈值，用于避免闭眼边缘抖动频繁重置。
        // 中文注释：它必须大于闭眼阈值；调低会更快从闭眼态恢复，调高会更稳定但可能让“是”更黏。
        var visionReopenEyeRatioThreshold: CGFloat = 0.74
        // 中文注释：Vision 分支建立开眼基线前需要的最少有效样本数。
        // 中文注释：调大基线更稳但启动更慢，调小启动更快但更容易受瞬时表情影响。
        var visionMinimumOpenSampleCount: Int = 2
        // 中文注释：是否启用强闭眼快速通道；排查误报时建议先关闭。
        var usesVisionStrongBlinkFastPath: Bool = false
        // 中文注释：明显闭眼时允许使用更短保持窗口，减少额外等待帧数。
        // 中文注释：如果“强闭眼通道”太激进导致误判，把它调大；如果想让重闭眼更快，再调小一点。
        var visionStrongBlinkHoldDuration: TimeInterval = 0.03
        // 中文注释：原始比值与平滑比值的融合权重，越接近 1 越信任原始值。
        // 中文注释：排查“总是报是”时，保持它偏向原始值通常更稳。
        var visionRawRatioWeight: CGFloat = 0.72
        // 中文注释：Vision 分支在检测窗口升到 30fps，优先压缩首个稳定命中时间。
        // 中文注释：调高会更灵敏更耗电，调低会更省但更容易漏掉短眨眼。
        var preferredVisionFrameRate: Int32 = 30
        // 中文注释：页面常驻时使用更低待机帧率，降低持续待机功耗。
        // 中文注释：只影响待机预热态，不影响真正检测窗口的判定阈值。
        var idleVisionFrameRate: Int32 = 5
        // 中文注释：Vision 跟踪置信度过低时回退为整帧重新检测。
        // 中文注释：调高会更容易放弃跟踪改走整帧检测，调低会更依赖跟踪框连续性。
        var visionTrackingConfidenceThreshold: VNConfidence = 0.35
    }

    private enum Backend {
        case idle
        case vision
        case arkit
    }

    private enum VisionSessionMode {
        case idle
        case detecting
    }

    struct DebugSnapshot {
        let leftBaseline: CGFloat
        let rightBaseline: CGFloat
        let leftRawRatio: CGFloat
        let rightRawRatio: CGFloat
        let leftEffectiveRatio: CGFloat
        let rightEffectiveRatio: CGFloat
        let averageRawRatio: CGFloat
        let effectiveAverageRatio: CGFloat
        let effectiveMaxRatio: CGFloat
        let isClosed: Bool
        let isStronglyClosed: Bool
    }

    nonisolated(unsafe) var config = Config()
    var onPrepareLatency: ((TimeInterval) -> Void)?
    var onLatency: ((TimeInterval) -> Void)?
    var onFaceLockLatency: ((TimeInterval) -> Void)?
    var onDecisionLatency: ((TimeInterval) -> Void)?
    var onBlink: ((Bool) -> Void)?
    var onStatus: ((String) -> Void)?
    var onDebugSnapshot: ((DebugSnapshot) -> Void)?

    nonisolated(unsafe) private let arSession = ARSession()
    nonisolated(unsafe) private let captureSession = AVCaptureSession()
    nonisolated(unsafe) private let captureOutput = AVCaptureVideoDataOutput()
    private let visionQueue = DispatchQueue(label: "com.zhifa.blink.vision")
    nonisolated(unsafe) private let trackingRequestHandler = VNSequenceRequestHandler()

    nonisolated(unsafe) private var startTime: TimeInterval = 0
    nonisolated(unsafe) private var prepareStartTime: TimeInterval?
    nonisolated(unsafe) private var firstSampleTime: TimeInterval?
    nonisolated(unsafe) private var faceLockTime: TimeInterval?
    nonisolated(unsafe) private var blinkStartTime: TimeInterval?
    nonisolated(unsafe) private var activeBackend: Backend = .idle
    nonisolated(unsafe) private var isRunning = false
    nonisolated(unsafe) private var timeoutWork: DispatchWorkItem?
    nonisolated(unsafe) private var isVisionConfigured = false
    nonisolated(unsafe) private var visionDevice: AVCaptureDevice?
    nonisolated(unsafe) private var trackedObservation: VNDetectedObjectObservation?
    nonisolated(unsafe) private var leftEyeOpenBaseline: CGFloat?
    nonisolated(unsafe) private var rightEyeOpenBaseline: CGFloat?
    nonisolated(unsafe) private var smoothedLeftEyeRatio: CGFloat?
    nonisolated(unsafe) private var smoothedRightEyeRatio: CGFloat?
    nonisolated(unsafe) private var stableOpenSampleCount = 0
    nonisolated(unsafe) private var visionSessionMode: VisionSessionMode = .idle
    nonisolated(unsafe) private var cachedVisionOrientation: CGImagePropertyOrientation = .leftMirrored

    override init() {
        super.init()
        arSession.delegate = self
    }

    func start() {
        refreshVisionOrientation()
        if isVisionBackendActive(), isRunning, isVisionDetecting() {
            return
        }
        if isARKitBackendActive(), isRunning {
            return
        }
        startTime = CACurrentMediaTime()
        resetBlinkWindowState()
        publishStatus("blink_starting")
        if isVisionBackendActive(), isRunning {
            // 中文注释：预热阶段只保活相机会话，不再复用 idle 期的人脸基线。
            // 中文注释：点击开始检测时强制清空基线，避免路过人脸或坏 landmarks 污染正式判定。
            resetDetectionState()
            scheduleTimeout()
            visionSessionMode = .detecting
            visionQueue.async { [weak self] in
                self?.applyVisionFrameRate(self?.config.preferredVisionFrameRate ?? 15)
            }
            return
        }
        isRunning = true
        activeBackend = .idle
        visionSessionMode = .detecting
        resetDetectionState()
        ensureVideoAccess { [weak self] granted in
            guard let self else { return }
            guard self.isRunning else { return }
            guard granted else {
                self.publishStatus("blink_camera_denied")
                self.stop(statusKey: nil)
                return
            }
            self.startPreferredBackend()
        }
    }

    func prepare() {
        refreshVisionOrientation()
        guard !isRunning else {
            if isVisionBackendActive() {
                enterIdleModeIfNeeded(reportStatus: false)
            }
            return
        }
        guard AVCaptureDevice.authorizationStatus(for: .video) == .authorized else { return }
        guard frontVideoDevice() != nil else { return }
        prepareStartTime = CACurrentMediaTime()
        isRunning = true
        activeBackend = .vision
        visionSessionMode = .idle
        resetDetectionState()
        startVisionBackend()
    }

    func stop() {
        if isVisionBackendActive(), isRunning {
            enterIdleModeIfNeeded(reportStatus: true)
            return
        }
        stop(statusKey: "blink_stopped")
    }

    func suspend() {
        stop(statusKey: nil)
    }

    private func startPreferredBackend() {
        scheduleTimeout()
        if frontVideoDevice() != nil {
            startVisionBackend()
            return
        }
        guard ARFaceTrackingConfiguration.isSupported else {
            publishStatus("blink_unsupported")
            stop(statusKey: nil)
            return
        }
        startARKitBackend()
    }

    private func startARKitBackend() {
        activeBackend = .arkit
        let configuration = ARFaceTrackingConfiguration()
        configuration.isLightEstimationEnabled = false
        arSession.run(configuration)
    }

    private func stop(statusKey: String?) {
        timeoutWork?.cancel()
        timeoutWork = nil
        guard isRunning || activeBackend != .idle else {
            if let statusKey {
                publishStatus(statusKey)
            }
            return
        }
        isRunning = false
        activeBackend = .idle
        visionSessionMode = .idle
        blinkStartTime = nil
        trackedObservation = nil
        arSession.pause()
        visionQueue.async { [weak self] in
            guard let self else { return }
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
            }
        }
        if let statusKey {
            publishStatus(statusKey)
        }
    }

    private func scheduleTimeout() {
        timeoutWork?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            self.publishBlinkResult(false)
        }
        timeoutWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + config.sessionTimeout, execute: work)
    }

    private func ensureVideoAccess(completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    completion(granted)
                }
            }
        case .denied, .restricted:
            completion(false)
        @unknown default:
            completion(false)
        }
    }

    private func publishLatency(_ latency: TimeInterval) {
        onLatency?(latency)
    }

    private func publishPrepareLatency(_ latency: TimeInterval) {
        onPrepareLatency?(latency)
    }

    private func publishFaceLockLatency(_ latency: TimeInterval) {
        onFaceLockLatency?(latency)
    }

    private func publishDecisionLatency(_ latency: TimeInterval) {
        onDecisionLatency?(latency)
    }

    private func publishBlinkResult(_ result: Bool) {
        guard isRunning else { return }
        if startTime > 0 {
            publishDecisionLatency(CACurrentMediaTime() - startTime)
        }
        onBlink?(result)
        stop()
    }

    private func publishStatus(_ key: String) {
        onStatus?(key)
    }

    nonisolated private func resetDetectionState() {
        resetBlinkWindowState()
        trackedObservation = nil
        leftEyeOpenBaseline = nil
        rightEyeOpenBaseline = nil
        smoothedLeftEyeRatio = nil
        smoothedRightEyeRatio = nil
        stableOpenSampleCount = 0
    }

    nonisolated private func resetBlinkWindowState() {
        firstSampleTime = nil
        faceLockTime = nil
        blinkStartTime = nil
    }

    nonisolated private func startVisionBackend() {
        refreshVisionOrientation()
        activeBackend = .vision
        visionQueue.async { [weak self] in
            guard let self else { return }
            do {
                try self.configureVisionSessionIfNeeded()
                guard self.isRunning, self.isVisionBackendActive() else { return }
                self.applyVisionFrameRate(
                    self.isVisionDetecting()
                    ? self.config.preferredVisionFrameRate
                    : self.config.idleVisionFrameRate
                )
                if !self.captureSession.isRunning {
                    self.captureSession.startRunning()
                }
            } catch {
                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    guard self.isRunning else { return }
                    if self.isVisionDetecting(), ARFaceTrackingConfiguration.isSupported {
                        self.startARKitBackend()
                    } else {
                        if self.isVisionDetecting() {
                            self.publishStatus("blink_error")
                        }
                        self.stop(statusKey: nil)
                    }
                }
            }
        }
    }

    nonisolated private func configureVisionSessionIfNeeded() throws {
        guard !isVisionConfigured else { return }
        guard let device = frontVideoDevice() else {
            throw BlinkError.missingFrontCamera
        }
        visionDevice = device
        let input = try AVCaptureDeviceInput(device: device)
        captureSession.beginConfiguration()
        for sessionInput in captureSession.inputs {
            captureSession.removeInput(sessionInput)
        }
        for sessionOutput in captureSession.outputs {
            captureSession.removeOutput(sessionOutput)
        }
        if captureSession.canSetSessionPreset(.vga640x480) {
            captureSession.sessionPreset = .vga640x480
        } else {
            captureSession.sessionPreset = .low
        }
        guard captureSession.canAddInput(input),
              captureSession.canAddOutput(captureOutput) else {
            captureSession.commitConfiguration()
            throw BlinkError.unableToConfigureCapture
        }
        captureSession.addInput(input)

        captureOutput.alwaysDiscardsLateVideoFrames = true
        captureOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
        ]
        captureOutput.setSampleBufferDelegate(self, queue: visionQueue)
        captureSession.addOutput(captureOutput)
        captureSession.commitConfiguration()
        isVisionConfigured = true
    }

    nonisolated private func applyVisionFrameRate(_ frameRate: Int32) {
        guard let visionDevice else { return }
        guard let resolvedFrameRate = resolveSupportedVisionFrameRate(for: Double(frameRate), device: visionDevice) else {
            return
        }
        do {
            try visionDevice.lockForConfiguration()
            let frameDuration = CMTime(
                seconds: 1.0 / resolvedFrameRate,
                preferredTimescale: 600
            )
            visionDevice.activeVideoMinFrameDuration = frameDuration
            visionDevice.activeVideoMaxFrameDuration = frameDuration
            visionDevice.unlockForConfiguration()
        } catch {
            return
        }
    }

    nonisolated private func resolveSupportedVisionFrameRate(
        for requestedFrameRate: Double,
        device: AVCaptureDevice
    ) -> Double? {
        let ranges = device.activeFormat.videoSupportedFrameRateRanges
        guard !ranges.isEmpty else { return nil }
        let tolerance = 0.05

        if ranges.contains(where: { range in
            range.minFrameRate - tolerance <= requestedFrameRate
                && requestedFrameRate <= range.maxFrameRate + tolerance
        }) {
            return requestedFrameRate
        }

        var bestCandidate: Double?
        var bestDistance = Double.greatestFiniteMagnitude

        for range in ranges {
            let candidate: Double
            if requestedFrameRate < range.minFrameRate {
                candidate = range.minFrameRate
            } else {
                candidate = range.maxFrameRate
            }

            let distance = abs(candidate - requestedFrameRate)
            if distance < bestDistance || (abs(distance - bestDistance) <= tolerance && candidate > (bestCandidate ?? 0)) {
                bestCandidate = candidate
                bestDistance = distance
            }
        }

        return bestCandidate
    }

    nonisolated private func processVisionFrame(_ sampleBuffer: CMSampleBuffer) {
        guard isRunning, isVisionBackendActive() else { return }
        markPrepareLatencyIfNeeded()
        markFirstSampleIfNeeded()
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let orientation = currentVisionOrientation()
        if let trackedObservation,
           let updatedObservation = trackFaceObservation(
            trackedObservation,
            pixelBuffer: pixelBuffer,
            orientation: orientation
           ) {
            self.trackedObservation = updatedObservation
            if let faceObservation = detectFaceLandmarks(
                in: pixelBuffer,
                orientation: orientation,
                regionOfInterest: expandedBoundingBox(updatedObservation.boundingBox)
            ) {
                self.trackedObservation = faceObservation
                handleVisionObservation(faceObservation)
                return
            }
        }

        guard let faceObservation = detectFaceLandmarks(
            in: pixelBuffer,
            orientation: orientation,
            regionOfInterest: nil
        ) else {
            blinkStartTime = nil
            trackedObservation = nil
            return
        }
        trackedObservation = faceObservation
        handleVisionObservation(faceObservation)
    }

    nonisolated private func markFirstSampleIfNeeded() {
        guard isVisionDetecting() else { return }
        guard firstSampleTime == nil else { return }
        let now = CACurrentMediaTime()
        firstSampleTime = now
        let latency = now - startTime
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            guard self.isRunning else { return }
            self.publishLatency(latency)
            self.publishStatus("blink_tracking")
        }
    }

    nonisolated private func markPrepareLatencyIfNeeded() {
        guard let startTime = prepareStartTime else { return }
        prepareStartTime = nil
        let latency = CACurrentMediaTime() - startTime
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.publishPrepareLatency(latency)
        }
    }

    nonisolated private func handleVisionObservation(_ faceObservation: VNFaceObservation) {
        guard let eyeMetrics = extractEyeMetrics(from: faceObservation) else {
            blinkStartTime = nil
            return
        }
        guard isPlausibleEyeMetrics(eyeMetrics) else {
            blinkStartTime = nil
            return
        }
        guard isVisionDetecting() else {
            // 中文注释：预热态只热启动相机和人脸链路，不在 idle 期累计闭眼基线。
            // 中文注释：这样正式检测只依赖用户当前这轮的开眼样本，不吃路过人脸的脏数据。
            return
        }
        markFaceLockLatencyIfNeeded()

        if !hasUsableVisionBaseline() {
            resetVisionBaseline(with: eyeMetrics)
            return
        }

        guard stableOpenSampleCount >= config.visionMinimumOpenSampleCount,
              let leftBaseline = leftEyeOpenBaseline,
              let rightBaseline = rightEyeOpenBaseline,
              leftBaseline > 0.0001,
              rightBaseline > 0.0001 else {
            updateVisionBaseline(with: eyeMetrics)
            return
        }

        let leftRatio = min(1.1, eyeMetrics.left / leftBaseline)
        let rightRatio = min(1.1, eyeMetrics.right / rightBaseline)
        let averageRawRatio = (leftRatio + rightRatio) * 0.5
        if averageRawRatio >= config.visionReopenEyeRatioThreshold {
            updateVisionBaseline(with: eyeMetrics)
        }

        let smoothedLeft = smoothEyeRatio(leftRatio, previous: smoothedLeftEyeRatio)
        let smoothedRight = smoothEyeRatio(rightRatio, previous: smoothedRightEyeRatio)
        smoothedLeftEyeRatio = smoothedLeft
        smoothedRightEyeRatio = smoothedRight

        let effectiveLeft = blendedEyeRatio(raw: leftRatio, smoothed: smoothedLeft)
        let effectiveRight = blendedEyeRatio(raw: rightRatio, smoothed: smoothedRight)
        let effectiveAverage = (effectiveLeft + effectiveRight) * 0.5
        let effectiveMax = max(effectiveLeft, effectiveRight)
        // 中文注释：这里把“常规闭眼阈值”和“开眼回滞阈值”组合成主判定。
        // 中文注释：如果绝大多数时候都报“是”，优先先调小 `visionClosedEyeRatioThreshold`。
        let isClosed = effectiveAverage <= config.visionClosedEyeRatioThreshold
            && effectiveMax <= config.visionReopenEyeRatioThreshold
        // 中文注释：这里是“强闭眼快速通道”；太容易误报时优先调小 `visionStrongClosedEyeRatioThreshold`
        // 中文注释：或者调大 `visionStrongBlinkHoldDuration`。
        let isStronglyClosed = config.usesVisionStrongBlinkFastPath
            && effectiveAverage <= config.visionStrongClosedEyeRatioThreshold
            && effectiveMax <= config.visionClosedEyeRatioThreshold
        publishDebugSnapshot(
            DebugSnapshot(
                leftBaseline: leftBaseline,
                rightBaseline: rightBaseline,
                leftRawRatio: leftRatio,
                rightRawRatio: rightRatio,
                leftEffectiveRatio: effectiveLeft,
                rightEffectiveRatio: effectiveRight,
                averageRawRatio: averageRawRatio,
                effectiveAverageRatio: effectiveAverage,
                effectiveMaxRatio: effectiveMax,
                isClosed: isClosed,
                isStronglyClosed: isStronglyClosed
            )
        )
        let now = CACurrentMediaTime()
        if isClosed {
            if blinkStartTime == nil {
                blinkStartTime = now
            }
            // 中文注释：普通闭眼走 `blinkHoldDuration`，强闭眼走更短的 `visionStrongBlinkHoldDuration`。
            // 中文注释：如果你感觉“闭眼判定太久”，改这两个时间；如果你感觉“总是误报是”，把它们调大。
            let requiredHoldDuration = isStronglyClosed
                ? config.visionStrongBlinkHoldDuration
                : config.blinkHoldDuration
            if let blinkStartTime, now - blinkStartTime >= requiredHoldDuration {
                DispatchQueue.main.async { [weak self] in
                    self?.publishBlinkResult(true)
                }
            }
        } else if effectiveAverage >= config.visionReopenEyeRatioThreshold {
            blinkStartTime = nil
        }
    }

    nonisolated private func markFaceLockLatencyIfNeeded() {
        guard isVisionDetecting() else { return }
        guard faceLockTime == nil else { return }
        let now = CACurrentMediaTime()
        faceLockTime = now
        let latency = now - startTime
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            guard self.isRunning else { return }
            self.publishFaceLockLatency(latency)
        }
    }

    nonisolated private func updateVisionBaseline(with metrics: EyeMetrics) {
        guard isPlausibleEyeMetrics(metrics) else { return }
        updateEyeBaseline(&leftEyeOpenBaseline, current: metrics.left)
        updateEyeBaseline(&rightEyeOpenBaseline, current: metrics.right)
        stableOpenSampleCount = min(
            stableOpenSampleCount + 1,
            max(config.visionMinimumOpenSampleCount, 6)
        )
    }

    nonisolated private func resetVisionBaseline(with metrics: EyeMetrics) {
        leftEyeOpenBaseline = metrics.left
        rightEyeOpenBaseline = metrics.right
        smoothedLeftEyeRatio = nil
        smoothedRightEyeRatio = nil
        stableOpenSampleCount = 1
    }

    nonisolated private func updateEyeBaseline(_ baseline: inout CGFloat?, current: CGFloat) {
        guard current > 0.0001 else { return }
        guard let currentBaseline = baseline else {
            baseline = current
            return
        }
        if !isPlausibleSingleEyeMetric(currentBaseline) {
            baseline = current
            return
        }
        let upwardSpikeLimit = currentBaseline * visionMaximumBaselineUpwardJumpRatio
        if current > upwardSpikeLimit {
            return
        }
        if current >= currentBaseline {
            baseline = currentBaseline * 0.84 + current * 0.16
            return
        }
        if current >= currentBaseline * 0.75 {
            baseline = currentBaseline * 0.70 + current * 0.30
            return
        }
        baseline = currentBaseline * 0.42 + current * 0.58
    }

    nonisolated private func blendedEyeRatio(raw: CGFloat, smoothed: CGFloat) -> CGFloat {
        let rawWeight = min(max(config.visionRawRatioWeight, 0), 1)
        let smoothedWeight = 1 - rawWeight
        return raw * rawWeight + smoothed * smoothedWeight
    }

    nonisolated private func smoothEyeRatio(_ current: CGFloat, previous: CGFloat?) -> CGFloat {
        guard let previous else { return current }
        return previous * 0.2 + current * 0.8
    }

    nonisolated private func hasUsableVisionBaseline() -> Bool {
        guard let leftBaseline = leftEyeOpenBaseline,
              let rightBaseline = rightEyeOpenBaseline else {
            return false
        }
        guard isPlausibleSingleEyeMetric(leftBaseline),
              isPlausibleSingleEyeMetric(rightBaseline) else {
            return false
        }
        return symmetryRatio(leftBaseline, rightBaseline) >= visionMinimumEyeSymmetryRatio
    }

    nonisolated private func isPlausibleEyeMetrics(_ metrics: EyeMetrics) -> Bool {
        guard isPlausibleSingleEyeMetric(metrics.left),
              isPlausibleSingleEyeMetric(metrics.right) else {
            return false
        }
        return symmetryRatio(metrics.left, metrics.right) >= visionMinimumEyeSymmetryRatio
    }

    nonisolated private func isPlausibleSingleEyeMetric(_ value: CGFloat) -> Bool {
        value >= visionMinimumEyeAspectRatio
            && value <= visionMaximumEyeAspectRatio
    }

    nonisolated private func symmetryRatio(_ left: CGFloat, _ right: CGFloat) -> CGFloat {
        let larger = max(left, right)
        let smaller = min(left, right)
        guard larger > 0.0001 else { return 0 }
        return smaller / larger
    }

    nonisolated private func publishDebugSnapshot(_ snapshot: DebugSnapshot) {
        DispatchQueue.main.async { [weak self] in
            self?.onDebugSnapshot?(snapshot)
        }
    }

    nonisolated private func detectFaceLandmarks(
        in pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation,
        regionOfInterest: CGRect?
    ) -> VNFaceObservation? {
        let request = VNDetectFaceLandmarksRequest()
        request.preferBackgroundProcessing = false
        if let regionOfInterest {
            request.regionOfInterest = regionOfInterest
        }
        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: orientation,
            options: [:]
        )
        do {
            try handler.perform([request])
            return bestFaceObservation(from: request.results)
        } catch {
            return nil
        }
    }

    nonisolated private func enterIdleModeIfNeeded(reportStatus: Bool) {
        guard isRunning, isVisionBackendActive() else { return }
        timeoutWork?.cancel()
        timeoutWork = nil
        visionSessionMode = .idle
        resetBlinkWindowState()
        visionQueue.async { [weak self] in
            self?.applyVisionFrameRate(self?.config.idleVisionFrameRate ?? 5)
        }
        if reportStatus {
            DispatchQueue.main.async { [weak self] in
                self?.publishStatus("blink_stopped")
            }
        }
    }

    nonisolated private func isVisionBackendActive() -> Bool {
        switch activeBackend {
        case .vision:
            return true
        case .idle, .arkit:
            return false
        }
    }

    nonisolated private func isARKitBackendActive() -> Bool {
        switch activeBackend {
        case .arkit:
            return true
        case .idle, .vision:
            return false
        }
    }

    nonisolated private func isVisionDetecting() -> Bool {
        switch visionSessionMode {
        case .detecting:
            return true
        case .idle:
            return false
        }
    }

    nonisolated private func trackFaceObservation(
        _ observation: VNDetectedObjectObservation,
        pixelBuffer: CVPixelBuffer,
        orientation: CGImagePropertyOrientation
    ) -> VNDetectedObjectObservation? {
        let request = VNTrackObjectRequest(detectedObjectObservation: observation)
        request.trackingLevel = .fast
        do {
            try trackingRequestHandler.perform([request], on: pixelBuffer, orientation: orientation)
            guard let trackedObservation = request.results?.first as? VNDetectedObjectObservation,
                  trackedObservation.confidence >= config.visionTrackingConfidenceThreshold else {
                return nil
            }
            return trackedObservation
        } catch {
            return nil
        }
    }

    nonisolated private func bestFaceObservation(from results: [VNFaceObservation]?) -> VNFaceObservation? {
        results?.max(by: { left, right in
            let leftArea = left.boundingBox.width * left.boundingBox.height
            let rightArea = right.boundingBox.width * right.boundingBox.height
            return leftArea < rightArea
        })
    }

    nonisolated private func extractEyeMetrics(from faceObservation: VNFaceObservation) -> EyeMetrics? {
        guard let landmarks = faceObservation.landmarks,
              let leftEye = eyeAspectRatio(from: landmarks.leftEye),
              let rightEye = eyeAspectRatio(from: landmarks.rightEye) else {
            return nil
        }
        return EyeMetrics(left: leftEye, right: rightEye)
    }

    nonisolated private func eyeAspectRatio(from region: VNFaceLandmarkRegion2D?) -> CGFloat? {
        guard let region else { return nil }
        let contour = region.normalizedPoints.map {
            CGPoint(x: CGFloat($0.x), y: CGFloat($0.y))
        }
        guard let geometry = makeEyeGeometry(from: contour) else { return nil }
        let width = distanceBetween(geometry.leftCorner, geometry.rightCorner)
        guard width > 0.0001 else { return nil }

        let sampleFractions: [CGFloat] = [1.0 / 3.0, 2.0 / 3.0]
        guard let upperSamples = sampleEyeArc(geometry.upperArc, fractions: sampleFractions),
              let lowerSamples = sampleEyeArc(geometry.lowerArc, fractions: sampleFractions),
              upperSamples.count == lowerSamples.count,
              upperSamples.count == sampleFractions.count else {
            return nil
        }

        let verticalA = distanceBetween(upperSamples[0], lowerSamples[0])
        let verticalB = distanceBetween(upperSamples[1], lowerSamples[1])
        return (verticalA + verticalB) / (2 * width)
    }

    nonisolated private func makeEyeGeometry(from contour: [CGPoint]) -> EyeGeometry? {
        guard contour.count >= 6 else { return nil }
        guard let leftIndex = contour.enumerated().min(by: { lhs, rhs in
            lhs.element.x < rhs.element.x
        })?.offset,
              let rightIndex = contour.enumerated().max(by: { lhs, rhs in
                  lhs.element.x < rhs.element.x
              })?.offset,
              leftIndex != rightIndex else {
            return nil
        }

        let forwardArc = contourArc(contour, from: leftIndex, to: rightIndex, step: 1)
        let backwardArc = contourArc(contour, from: leftIndex, to: rightIndex, step: -1)
        guard forwardArc.count >= 3, backwardArc.count >= 3 else { return nil }

        let forwardAverageY = averageY(for: forwardArc)
        let backwardAverageY = averageY(for: backwardArc)
        let upperArc = forwardAverageY >= backwardAverageY ? forwardArc : backwardArc
        let lowerArc = forwardAverageY >= backwardAverageY ? backwardArc : forwardArc

        return EyeGeometry(
            leftCorner: contour[leftIndex],
            rightCorner: contour[rightIndex],
            upperArc: upperArc,
            lowerArc: lowerArc
        )
    }

    nonisolated private func contourArc(
        _ contour: [CGPoint],
        from startIndex: Int,
        to endIndex: Int,
        step: Int
    ) -> [CGPoint] {
        guard !contour.isEmpty else { return [] }
        var index = startIndex
        var points: [CGPoint] = [contour[startIndex]]
        while index != endIndex {
            index = (index + step + contour.count) % contour.count
            points.append(contour[index])
        }
        return points
    }

    nonisolated private func averageY(for points: [CGPoint]) -> CGFloat {
        guard !points.isEmpty else { return 0 }
        let interior = points.count > 2 ? Array(points.dropFirst().dropLast()) : points
        guard !interior.isEmpty else { return 0 }
        let sum = interior.reduce(CGFloat.zero) { partial, point in
            partial + point.y
        }
        return sum / CGFloat(interior.count)
    }

    nonisolated private func sampleEyeArc(
        _ points: [CGPoint],
        fractions: [CGFloat]
    ) -> [CGPoint]? {
        guard points.count >= 2 else { return nil }
        var cumulativeDistances: [CGFloat] = [0]
        cumulativeDistances.reserveCapacity(points.count)
        for index in 1..<points.count {
            let segment = distanceBetween(points[index - 1], points[index])
            cumulativeDistances.append(cumulativeDistances[index - 1] + segment)
        }
        guard let totalLength = cumulativeDistances.last, totalLength > 0.0001 else {
            return nil
        }

        return fractions.map { fraction in
            let clampedFraction = min(max(fraction, 0), 1)
            let targetDistance = totalLength * clampedFraction
            for index in 1..<cumulativeDistances.count {
                let upperBound = cumulativeDistances[index]
                guard targetDistance <= upperBound || index == cumulativeDistances.count - 1 else {
                    continue
                }
                let lowerBound = cumulativeDistances[index - 1]
                let segmentLength = max(upperBound - lowerBound, 0.0001)
                let progress = (targetDistance - lowerBound) / segmentLength
                return interpolatePoint(
                    from: points[index - 1],
                    to: points[index],
                    progress: progress
                )
            }
            return points[points.count - 1]
        }
    }

    nonisolated private func interpolatePoint(
        from start: CGPoint,
        to end: CGPoint,
        progress: CGFloat
    ) -> CGPoint {
        CGPoint(
            x: start.x + (end.x - start.x) * progress,
            y: start.y + (end.y - start.y) * progress
        )
    }

    nonisolated private func distanceBetween(_ lhs: CGPoint, _ rhs: CGPoint) -> CGFloat {
        hypot(lhs.x - rhs.x, lhs.y - rhs.y)
    }

    nonisolated private func expandedBoundingBox(_ boundingBox: CGRect) -> CGRect {
        let horizontalInset = boundingBox.width * 0.18
        let verticalInset = boundingBox.height * 0.24
        let expandedBoundingBox = boundingBox.insetBy(dx: -horizontalInset, dy: -verticalInset)
        return expandedBoundingBox.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
    }

    nonisolated private func frontVideoDevice() -> AVCaptureDevice? {
        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .builtInTrueDepthCamera],
            mediaType: .video,
            position: .front
        )
        return discoverySession.devices.first
    }

    nonisolated private func refreshVisionOrientation() {
        cachedVisionOrientation = resolveVisionOrientation()
    }

    nonisolated private func currentVisionOrientation() -> CGImagePropertyOrientation {
        cachedVisionOrientation
    }

    nonisolated private func resolveVisionOrientation() -> CGImagePropertyOrientation {
        let interfaceOrientation = readInterfaceOrientation()
        switch interfaceOrientation {
        case .portrait:
            return .leftMirrored
        case .portraitUpsideDown:
            return .rightMirrored
        case .landscapeLeft:
            return .downMirrored
        case .landscapeRight:
            return .upMirrored
        default:
            return .leftMirrored
        }
    }

    nonisolated private func readInterfaceOrientation() -> UIInterfaceOrientation {
        if Thread.isMainThread {
            return MainActor.assumeIsolated {
                UIApplication.shared.connectedScenes
                    .compactMap { $0 as? UIWindowScene }
                    .first?.interfaceOrientation ?? .portrait
            }
        }
        return DispatchQueue.main.sync { @MainActor in
            UIApplication.shared.connectedScenes
                .compactMap { $0 as? UIWindowScene }
                .first?.interfaceOrientation ?? .portrait
        }
    }

    private func handleARFrame(_ frame: ARFrame) {
        guard isRunning, isARKitBackendActive() else { return }
        let now = CACurrentMediaTime()
        if firstSampleTime == nil {
            firstSampleTime = now
            publishLatency(now - startTime)
            publishStatus("blink_tracking")
        }
        for anchor in frame.anchors {
            guard let faceAnchor = anchor as? ARFaceAnchor else { continue }
            if faceLockTime == nil {
                faceLockTime = now
                publishFaceLockLatency(now - startTime)
            }
            let leftBlink = faceAnchor.blendShapes[.eyeBlinkLeft]?.floatValue ?? 0
            let rightBlink = faceAnchor.blendShapes[.eyeBlinkRight]?.floatValue ?? 0
            let isClosed = leftBlink >= config.blinkThreshold && rightBlink >= config.blinkThreshold
            if isClosed {
                if blinkStartTime == nil {
                    blinkStartTime = now
                }
                if let blinkStartTime, now - blinkStartTime >= config.blinkHoldDuration {
                    publishBlinkResult(true)
                    break
                }
            } else {
                blinkStartTime = nil
            }
        }
    }
}

extension AispectBlinkDetector: AVCaptureVideoDataOutputSampleBufferDelegate {
    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        autoreleasepool {
            processVisionFrame(sampleBuffer)
        }
    }
}

extension AispectBlinkDetector: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        handleARFrame(frame)
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        publishStatus("blink_error")
        stop(statusKey: nil)
    }
}

private extension AispectBlinkDetector {
    nonisolated var visionMinimumEyeAspectRatio: CGFloat { 0.08 }
    nonisolated var visionMaximumEyeAspectRatio: CGFloat { 0.55 }
    nonisolated var visionMinimumEyeSymmetryRatio: CGFloat { 0.42 }
    nonisolated var visionMaximumBaselineUpwardJumpRatio: CGFloat { 1.28 }

    struct EyeGeometry {
        let leftCorner: CGPoint
        let rightCorner: CGPoint
        let upperArc: [CGPoint]
        let lowerArc: [CGPoint]
    }

    struct EyeMetrics {
        let left: CGFloat
        let right: CGFloat
    }

    enum BlinkError: Error {
        case missingFrontCamera
        case unableToConfigureCapture
    }
}
