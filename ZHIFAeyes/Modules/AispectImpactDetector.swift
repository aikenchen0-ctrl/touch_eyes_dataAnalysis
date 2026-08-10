import CoreMotion
import QuartzCore

final class AispectImpactDetector {
    struct Config {
        // 中文注释：目标采样间隔，实际频率受硬件与系统调度影响。
        var updateInterval: TimeInterval = 1.0 / 240.0
        // 中文注释：按下期固定窗口的前后帧数量。
        var downPreFrames: Int = 5
        var downPostFrames: Int = 9
        // 中文注释：采样率回调更新的最小间隔。
        var sampleRateUpdateInterval: TimeInterval = 0.5
        // 中文注释：优先使用 userAcceleration 去掉重力分量。
        var usesUserAcceleration: Bool = true
    }

    var config = Config()
    var onSampleRate: ((Double) -> Void)?
    var onImpactLog: ((String) -> Void)?
    var onDatasetStatus: ((AispectImpactDatasetCollector.Status) -> Void)?
    private(set) var isRunning = false
    private(set) var isDatasetCollectionEnabled = false

    private let motionManager = CMMotionManager()
    private let datasetCollector = AispectImpactDatasetCollector()
    private let queue = OperationQueue()
    private var lastSampleTime: TimeInterval = 0
    private var averageSampleInterval: TimeInterval = 0
    private var lastSampleRateNotify: TimeInterval = 0
    private var sampleWindow: [ImpactSample] = []
    private var downAnchorTimestamp: TimeInterval?
    private var downWindowPublished = false

    private struct ImpactSample {
        let timestamp: TimeInterval
        let x: Double
        let y: Double
        let z: Double
        let rotationRateX: Double
        let rotationRateY: Double
        let rotationRateZ: Double
        let gravityX: Double
        let gravityY: Double
        let gravityZ: Double

        var frame: AispectImpactFrame {
            AispectImpactFrame(
                x: x,
                y: y,
                z: z,
                rotationRateX: rotationRateX,
                rotationRateY: rotationRateY,
                rotationRateZ: rotationRateZ,
                gravityX: gravityX,
                gravityY: gravityY,
                gravityZ: gravityZ
            )
        }
    }

    init() {
        queue.name = "paifa.aispect.impact.sensor"
        datasetCollector.onStatusChanged = { [weak self] status in
            self?.onDatasetStatus?(status)
        }
    }

    func start() {
        guard !isRunning else { return }
        isRunning = true
        lastSampleTime = 0
        averageSampleInterval = 0
        lastSampleRateNotify = 0
        sampleWindow.removeAll(keepingCapacity: true)
        downWindowPublished = false

        if config.usesUserAcceleration, motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = config.updateInterval
            motionManager.startDeviceMotionUpdates(to: queue) { [weak self] motion, _ in
                guard let self, let motion else { return }
                let acceleration = motion.userAcceleration
                let rotation = motion.rotationRate
                let gravity = motion.gravity
                self.handleSample(
                    x: acceleration.x,
                    y: acceleration.y,
                    z: acceleration.z,
                    rotationRateX: rotation.x,
                    rotationRateY: rotation.y,
                    rotationRateZ: rotation.z,
                    gravityX: gravity.x,
                    gravityY: gravity.y,
                    gravityZ: gravity.z
                )
            }
        } else if motionManager.isAccelerometerAvailable {
            motionManager.accelerometerUpdateInterval = config.updateInterval
            motionManager.startAccelerometerUpdates(to: queue) { [weak self] data, _ in
                guard let self, let data else { return }
                let acceleration = data.acceleration
                self.handleSample(
                    x: acceleration.x,
                    y: acceleration.y,
                    z: acceleration.z,
                    rotationRateX: 0,
                    rotationRateY: 0,
                    rotationRateZ: 0,
                    gravityX: 0,
                    gravityY: 0,
                    gravityZ: 0
                )
            }
        } else {
            isRunning = false
        }
    }

    func stop() {
        datasetCollector.finalizeCaptureIfNeeded()
        guard isRunning else {
            downAnchorTimestamp = nil
            downWindowPublished = false
            return
        }
        motionManager.stopDeviceMotionUpdates()
        motionManager.stopAccelerometerUpdates()
        isRunning = false
        downAnchorTimestamp = nil
        downWindowPublished = false
    }

    private func handleSample(
        x: Double,
        y: Double,
        z: Double,
        rotationRateX: Double,
        rotationRateY: Double,
        rotationRateZ: Double,
        gravityX: Double,
        gravityY: Double,
        gravityZ: Double
    ) {
        let now = CACurrentMediaTime()
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

        let sample = ImpactSample(
            timestamp: now,
            x: x,
            y: y,
            z: z,
            rotationRateX: rotationRateX,
            rotationRateY: rotationRateY,
            rotationRateZ: rotationRateZ,
            gravityX: gravityX,
            gravityY: gravityY,
            gravityZ: gravityZ
        )
        sampleWindow.append(sample)
        let maximumWindow = max(config.downPreFrames + config.downPostFrames + 1 + 16, 48)
        if sampleWindow.count > maximumWindow {
            sampleWindow.removeFirst(sampleWindow.count - maximumWindow)
        }
        let sampleRateHz = averageSampleInterval > 0 ? 1.0 / averageSampleInterval : 0
        datasetCollector.appendSample(sample.frame, timestamp: now, sampleRateHz: sampleRateHz)
        publishDownWindowIfReady()
    }

    // 中文注释：只接受缓存中的真实连续帧，不用补零帧伪造按下期窗口。
    private func publishDownWindowIfReady() {
        guard !downWindowPublished,
              let anchor = downAnchorTimestamp,
              let referenceIndex = nearestFrameIndex(to: anchor) else {
            return
        }
        let start = referenceIndex - config.downPreFrames
        let end = referenceIndex + config.downPostFrames
        guard start >= 0, end < sampleWindow.count else { return }
        downWindowPublished = true
        let window = Array(sampleWindow[start...end])
        datasetCollector.markDownWindowReady()
        let lines = window.enumerated().map { offset, sample in
            let frameIndex = offset - config.downPreFrames
            return String(
                format: "f%+03d x %+0.4f y %+0.4f z %+0.4f",
                frameIndex,
                sample.x,
                sample.y,
                sample.z
            )
        }
        let text = ([
            String(format: "down anchor window ready [%+d...%+d] frames %d; new model pending", -config.downPreFrames, config.downPostFrames, window.count)
        ] + lines).joined(separator: "\n")
        DispatchQueue.main.async { [weak self] in
            self?.onImpactLog?(text)
        }
    }

    private func nearestFrameIndex(to timestamp: TimeInterval) -> Int? {
        guard !sampleWindow.isEmpty else { return nil }
        return sampleWindow.enumerated().min {
            abs($0.element.timestamp - timestamp) < abs($1.element.timestamp - timestamp)
        }?.offset
    }

    func setDatasetCollectionEnabled(_ enabled: Bool) {
        if enabled {
            isDatasetCollectionEnabled = true
            datasetCollector.setEnabled(true)
            start()
            return
        }
        stop()
        isDatasetCollectionEnabled = false
        datasetCollector.setEnabled(false)
    }

    func setDatasetCollectionLabel(_ label: AispectImpactDatasetLabel) {
        datasetCollector.setSelectedLabel(label)
    }

    func beginDatasetCapture(at timestamp: TimeInterval) {
        downAnchorTimestamp = timestamp
        downWindowPublished = false
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
