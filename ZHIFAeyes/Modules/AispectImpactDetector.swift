import CoreMotion
import QuartzCore

final class AispectImpactDetector {
    struct Config {
        // 中文注释：目标采样间隔，实际频率受硬件与系统调度影响。
        var updateInterval: TimeInterval = 1.0 / 240.0
        // 中文注释：冲击判定阈值，基于相邻帧加权模长的变化量。
        var impactDeltaThreshold: Double = 0.25
        // 中文注释：触发冷却时间，避免连续抖动造成多次触发。
        var cooldown: TimeInterval = 0.18
        // 中文注释：优先使用 userAcceleration 去掉重力分量。
        var usesUserAcceleration: Bool = true
        // 中文注释：三轴权重，用于强化按压方向特征。
        var axisWeightX: Double = 0.01
        var axisWeightY: Double = 1.0
        var axisWeightZ: Double = 6.5
        // 中文注释：采样率回调更新的最小间隔。
        var sampleRateUpdateInterval: TimeInterval = 0.5
        // 中文注释：是否启用 CNN 作为候选冲击的二次分类器。
        var usesCNNClassifier: Bool = true
        // 中文注释：CNN 判定为重触时需要达到的最小概率。
        var cnnHeavyProbabilityThreshold: Double = 0.90
    }

    var config = Config()
    var onImpact: ((Double) -> Void)?
    var onSampleRate: ((Double) -> Void)?
    var onImpactLog: ((String) -> Void)?
    var onDatasetStatus: ((AispectImpactDatasetCollector.Status) -> Void)?
    private(set) var isRunning = false

    private let motionManager = CMMotionManager()
    private let classifier = AispectImpactCNNClassifier.shared
    private let datasetCollector = AispectImpactDatasetCollector()
    private let queue = OperationQueue()
    private var lastImpactTime: TimeInterval = 0
    private var lastMagnitude: Double = 0
    private var lastSampleTime: TimeInterval = 0
    private var averageSampleInterval: TimeInterval = 0
    private var lastSampleRateNotify: TimeInterval = 0
    // 中文注释：用于冲击窗口前后帧记录。
    private var deltaWindowSigned: [Double] = []
    private var sampleWindow: [ImpactSample] = []
    private var pendingImpact: ImpactLogBuffer?
    private var hasBaseline = false

    // 中文注释：单帧采样快照，用于日志显示。
    private struct ImpactSample {
        let signedDelta: Double
        let x: Double
        let y: Double
        let z: Double
        let magnitude: Double

        var frame: AispectImpactFrame {
            AispectImpactFrame(delta: signedDelta, x: x, y: y, z: z)
        }
    }

    // 中文注释：冲击窗口缓冲，包含前后帧与剩余补采样数量。
    private struct ImpactLogBuffer {
        var pre: [ImpactSample]
        var current: ImpactSample
        var post: [ImpactSample]
        var remaining: Int
        var triggerDelta: Double
    }

    init() {
        datasetCollector.onStatusChanged = { [weak self] status in
            self?.onDatasetStatus?(status)
        }
    }

    func start() {
        guard !isRunning else { return }
        isRunning = true
        lastMagnitude = 0
        lastImpactTime = 0
        lastSampleTime = 0
        averageSampleInterval = 0
        lastSampleRateNotify = 0
        deltaWindowSigned.removeAll()
        pendingImpact = nil
        hasBaseline = false
        // 中文注释：优先使用 DeviceMotion 的 userAcceleration。
        if config.usesUserAcceleration, motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = config.updateInterval
            motionManager.startDeviceMotionUpdates(to: queue) { [weak self] motion, _ in
                guard let self, let motion else { return }
                let ua = motion.userAcceleration
                self.handleSample(x: ua.x, y: ua.y, z: ua.z)
            }
        } else if motionManager.isAccelerometerAvailable {
            // 中文注释：退化为原始加速度数据。
            motionManager.accelerometerUpdateInterval = config.updateInterval
            motionManager.startAccelerometerUpdates(to: queue) { [weak self] data, _ in
                guard let self, let data else { return }
                let a = data.acceleration
                self.handleSample(x: a.x, y: a.y, z: a.z)
            }
        } else {
            isRunning = false
        }
    }

    func stop() {
        guard isRunning else { return }
        datasetCollector.finalizeCaptureIfNeeded()
        if let pending = pendingImpact {
            finalizePendingImpact(buffer: pending, padMissingPostSamples: true)
        }
        motionManager.stopDeviceMotionUpdates()
        motionManager.stopAccelerometerUpdates()
        isRunning = false
    }

    private func handleSample(x: Double, y: Double, z: Double) {
        let now = CACurrentMediaTime()
        // 中文注释：加权合成让 Z 轴占主导，同时保留横向噪声信息。
        let magnitude = weightedMagnitude(x: x, y: y, z: z)
        // 中文注释：首帧建立基线，避免无效 delta。
        if !hasBaseline {
            hasBaseline = true
            lastSampleTime = now
            lastMagnitude = magnitude
            deltaWindowSigned.append(0)
            sampleWindow.append(
                ImpactSample(
                    signedDelta: 0,
                    x: x,
                    y: y,
                    z: z,
                    magnitude: magnitude
                )
            )
            return
        }
        // 中文注释：估算实际采样间隔并回调显示。
        if lastSampleTime > 0 {
            let interval = now - lastSampleTime
            if interval > 0 {
                if averageSampleInterval == 0 {
                    averageSampleInterval = interval
                } else {
                    averageSampleInterval = averageSampleInterval * 0.9 + interval * 0.1
                }
                if now - lastSampleRateNotify >= config.sampleRateUpdateInterval {
                    lastSampleRateNotify = now
                    let hz = averageSampleInterval > 0 ? 1.0 / averageSampleInterval : 0
                    DispatchQueue.main.async { [weak self] in
                        self?.onSampleRate?(hz)
                    }
                }
            }
        }
        lastSampleTime = now
        let signedDelta = magnitude - lastMagnitude
        let delta = abs(signedDelta)
        let sampleRateHz = averageSampleInterval > 0 ? 1.0 / averageSampleInterval : 0
        lastMagnitude = magnitude
        deltaWindowSigned.append(signedDelta)
        // 中文注释：保留滚动窗口用于冲击日志的前后文。
        let sample = ImpactSample(
            signedDelta: signedDelta,
            x: x,
            y: y,
            z: z,
            magnitude: magnitude
        )
        sampleWindow.append(sample)
        datasetCollector.appendSample(sample.frame, timestamp: now, sampleRateHz: sampleRateHz)
        // 中文注释：保持固定长度窗口，避免内存增长。
        if deltaWindowSigned.count > 40 {
            deltaWindowSigned.removeFirst(deltaWindowSigned.count - 40)
        }
        if sampleWindow.count > 40 {
            sampleWindow.removeFirst(sampleWindow.count - 40)
        }
        // 中文注释：冲击触发后继续补齐后续帧。
        if var pending = pendingImpact {
            pending.post.append(sample)
            pending.remaining -= 1
            if pending.remaining <= 0 {
                finalizePendingImpact(buffer: pending, padMissingPostSamples: false)
            } else {
                pendingImpact = pending
                emitImpactLog(buffer: pending, prediction: nil)
            }
        }
        // 中文注释：冲击触发采用 delta 峰值 + 冷却时间，降低误触。
        if delta >= config.impactDeltaThreshold, now - lastImpactTime >= config.cooldown {
            lastImpactTime = now
            let pre = Array(sampleWindow.dropLast().suffix(5))
            let buffer = ImpactLogBuffer(
                pre: pre,
                current: sample,
                post: [],
                remaining: 15,
                triggerDelta: delta
            )
            guard config.usesCNNClassifier, classifier.isReady else {
                emitImpactLog(buffer: buffer, prediction: nil)
                DispatchQueue.main.async { [weak self] in
                    self?.onImpact?(delta)
                }
                return
            }
            pendingImpact = buffer
            emitImpactLog(buffer: buffer, prediction: nil)
        }
    }

    // 中文注释：加权模长用于突出按压方向与衰减横向噪声。
    private func weightedMagnitude(x: Double, y: Double, z: Double) -> Double {
        let wx = max(config.axisWeightX, 0.0)
        let wy = max(config.axisWeightY, 0.0)
        let wz = max(config.axisWeightZ, 0.0)
        return (wx * x * x + wy * y * y + wz * z * z).squareRoot()
    }

    // 中文注释：输出冲击窗口日志，包含前后帧与当前帧。
    private func finalizePendingImpact(buffer: ImpactLogBuffer, padMissingPostSamples: Bool) {
        var completed = buffer
        if padMissingPostSamples, completed.remaining > 0 {
            let zero = ImpactSample(signedDelta: 0, x: 0, y: 0, z: 0, magnitude: 0)
            for _ in 0..<completed.remaining {
                completed.post.append(zero)
            }
            completed.remaining = 0
        }
        pendingImpact = nil
        let prediction = classifyImpact(buffer: completed)
        emitImpactLog(buffer: completed, prediction: prediction)
        guard let prediction else { return }
        guard prediction.heavyProbability >= config.cnnHeavyProbabilityThreshold else { return }
        DispatchQueue.main.async { [weak self] in
            self?.onImpact?(prediction.heavyProbability)
        }
    }

    private func classifyImpact(buffer: ImpactLogBuffer) -> AispectImpactPrediction? {
        let frames = makeFrames(buffer: buffer)
        guard frames.count == 21 else { return nil }
        return classifier.predict(frames: frames)
    }

    private func makeFrames(buffer: ImpactLogBuffer) -> [AispectImpactFrame] {
        let preCount = 5
        let postCount = 15
        var values = Array(repeating: AispectImpactFrame(delta: 0, x: 0, y: 0, z: 0), count: preCount + postCount + 1)
        let preStart = preCount - min(preCount, buffer.pre.count)
        for (offset, value) in buffer.pre.suffix(preCount).enumerated() {
            values[preStart + offset] = value.frame
        }
        values[preCount] = buffer.current.frame
        for (offset, value) in buffer.post.prefix(postCount).enumerated() {
            values[preCount + 1 + offset] = value.frame
        }
        return values
    }

    private func emitImpactLog(buffer: ImpactLogBuffer, prediction: AispectImpactPrediction?) {
        let preCount = 5
        let postCount = 15
        let total = preCount + postCount + 1
        var values: [ImpactSample] = Array(
            repeating: ImpactSample(signedDelta: 0, x: 0, y: 0, z: 0, magnitude: 0),
            count: total
        )
        let preStart = preCount - min(preCount, buffer.pre.count)
        for (offset, value) in buffer.pre.suffix(preCount).enumerated() {
            values[preStart + offset] = value
        }
        values[preCount] = buffer.current
        for (offset, value) in buffer.post.prefix(postCount).enumerated() {
            values[preCount + 1 + offset] = value
        }
        var lines: [String] = []
        if let prediction {
            lines.append(
                String(
                    format: "impact delta window (post remaining %d) heavy %.4f light %.4f decision %@",
                    buffer.remaining,
                    prediction.heavyProbability,
                    prediction.lightProbability,
                    prediction.isHeavy && prediction.heavyProbability >= config.cnnHeavyProbabilityThreshold ? "heavy" : "light"
                )
            )
        } else if config.usesCNNClassifier, classifier.isReady {
            lines.append(String(format: "impact delta window (post remaining %d) cnn pending", buffer.remaining))
        } else {
            lines.append(String(format: "impact delta window (post remaining %d) cnn unavailable", buffer.remaining))
        }
        for i in 0..<total {
            let index = i - preCount
            let marker = index == 0 ? ">" : " "
            let sample = values[i]
            // 中文注释：使用带符号的 delta 表示变化方向。
            lines.append(
                String(
                    format: "%@%+03d d %+0.5f x %+0.4f y %+0.4f z %+0.4f m %.5f",
                    marker,
                    index,
                    sample.signedDelta,
                    sample.x,
                    sample.y,
                    sample.z,
                    sample.magnitude
                )
            )
        }
        let logText = lines.joined(separator: "\n")
        DispatchQueue.main.async { [weak self] in
            self?.onImpactLog?(logText)
        }
    }

    func setDatasetCollectionEnabled(_ enabled: Bool) {
        datasetCollector.setEnabled(enabled)
    }

    func setDatasetCollectionLabel(_ label: AispectImpactDatasetLabel) {
        datasetCollector.setSelectedLabel(label)
    }

    func beginDatasetCapture(at timestamp: TimeInterval) {
        datasetCollector.beginCapture(at: timestamp)
    }

    func acceptDatasetCapture(anchorTimestamp: TimeInterval, touchKind: String, liftOffset: Double?) -> TimeInterval {
        datasetCollector.acceptCapture(anchorTimestamp: anchorTimestamp, touchKind: touchKind, liftOffset: liftOffset)
    }

    func rejectDatasetCapture(reasonKey: String) {
        datasetCollector.rejectCapture(reasonKey: reasonKey)
    }

    var datasetDirectoryPath: String {
        datasetCollector.datasetDirectoryPath
    }
}
