import CoreMotion
import Foundation

enum AispectImpactMotionState: String, CaseIterable, Codable {
    case handheldStatic = "handheld_static"
    case placedStatic = "placed_static"
    case walking

    var displayName: String {
        switch self {
        case .handheldStatic:
            return "handheld_static"
        case .placedStatic:
            return "placed_static"
        case .walking:
            return "walking"
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let rawValue = try container.decode(String.self)
        switch rawValue {
        case "static", "handheld_static":
            self = .handheldStatic
        case "placed_static":
            self = .placedStatic
        case "walking":
            self = .walking
        default:
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unsupported motion state: \(rawValue)")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

enum AispectImpactTouchStrength: String, CaseIterable, Codable {
    case light
    case heavy

    var displayName: String {
        rawValue
    }
}

struct AispectImpactDatasetLabel: Codable, Equatable {
    var motionState: AispectImpactMotionState
    var touchStrength: AispectImpactTouchStrength

    var datasetKey: String {
        "\(motionState.rawValue)_\(touchStrength.rawValue)"
    }
}

struct AispectImpactSystemActivityEvidence: Codable, Equatable {
    let timestamp: Double
    let stationary: Bool
    let walking: Bool
    let running: Bool
    let automotive: Bool
    let cycling: Bool
    let unknown: Bool
    let confidence: String
    let authorizationStatus: String

    var inferredMotionState: String {
        if walking || running {
            return "walking"
        }
        if stationary {
            return "static"
        }
        return "unknown"
    }
}

final class AispectImpactDatasetCollector {
    struct Config {
        // 中文注释：按下锚点原始窗口在按下前保留的真实帧数量。
        var preFrames: Int = 15
        // 中文注释：按下锚点原始窗口在抬起后保留的真实帧数量。
        var postFrames: Int = 15
        // 中文注释：滚动缓存长度，便于保留前置样本。
        var rollingFrameLimit: Int = 600
        // 中文注释：采集任务默认目标样本数。
        var targetAcceptedSamples: Int = 30
        // 中文注释：数据文件写入目录名称。
        var datasetDirectoryName: String = "AispectImpactDataset"
        // 中文注释：计算抬起后等待时长时使用的兜底采样率。
        var fallbackSampleRate: Double = 120
    }

    struct Status {
        let isEnabled: Bool
        let selectedLabel: AispectImpactDatasetLabel
        let isLabelLocked: Bool
        let totalCount: Int
        let countsByLabel: [String: Int]
        let acceptedCount: Int
        let rejectedCount: Int
        let targetCount: Int
        let lastEventKey: String
        let lastEventDetail: String?
        let collectionSessionID: String?
        let latestActivity: AispectImpactSystemActivityEvidence?
        let datasetDirectoryPath: String
    }

    private struct CapturedFrame {
        let timestamp: TimeInterval
        let frame: AispectImpactFrame
    }

    private struct CaptureSession {
        let id: String
        let label: AispectImpactDatasetLabel
        let startedAt: TimeInterval
        let preFrames: [CapturedFrame]
        var recordedFrames: [CapturedFrame]
        var remainingPostFrames: Int?
        var anchorTimestamp: TimeInterval?
        var endedTouchKind: String?
        var endedLiftOffset: Double?
        var lastSampleRateHz: Double
    }

    private struct SampleFrameRecord: Codable {
        let frameIndex: Int
        let relativeTimestamp: Double
        let timeSinceTouchDown: Double
        let timeSinceAnchor: Double
        let x: Double
        let y: Double
        let z: Double
        let x_norm: Double
        let y_norm: Double
        let rotationRateX: Double
        let rotationRateY: Double
        let rotationRateZ: Double
        let gravityX: Double
        let gravityY: Double
        let gravityZ: Double
    }

    private struct SampleRecord: Codable {
        let schemaVersion: Int
        let id: String
        let createdAt: Double
        let collectionSessionID: String
        let collectionSessionStartedAt: Double
        let label: AispectImpactDatasetLabel
        let manualLabelSource: String
        let sourceTouchKind: String
        let liftOffset: Double?
        let sampleRateHz: Double
        let referenceFrameIndex: Int
        let anchorTimestamp: Double
        let downAnchorTimestamp: Double
        let liftAnchorTimestamp: Double
        let downReferenceFrameIndex: Int
        let liftReferenceFrameIndex: Int
        let startFrameIndex: Int
        let endFrameIndex: Int
        let preDownFrameCount: Int
        let postLiftFrameCount: Int
        let windowComplete: Bool
        let activityEvidence: AispectImpactSystemActivityEvidence?
        let appVersion: String
        let appBuild: String
        let frames: [SampleFrameRecord]
    }

    var config = Config()
    var onStatusChanged: ((Status) -> Void)?

    private let stateQueue = DispatchQueue(label: "paifa.aispect.impact.dataset.collector")
    private let activityManager = CMMotionActivityManager()
    private let activityQueue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "paifa.aispect.impact.dataset.activity"
        queue.maxConcurrentOperationCount = 1
        return queue
    }()
    private let fileManager = FileManager.default

    private var isEnabled = false
    private var selectedLabel = AispectImpactDatasetLabel(
        motionState: .handheldStatic,
        touchStrength: .light
    )
    private var rollingFrames: [CapturedFrame] = []
    private var activeSession: CaptureSession?
    private var countsByLabel: [String: Int] = [:]
    private var acceptedCount = 0
    private var rejectedCount = 0
    private var lastEventKey = "collector_idle"
    private var lastEventDetail: String?
    private var collectionSessionID: String?
    private var collectionSessionStartedAt: TimeInterval = 0
    private var latestActivity: AispectImpactSystemActivityEvidence?

    init() {
        emitStatus()
        stateQueue.async { [weak self] in
            guard let self else { return }
            self.loadExistingSummary()
            let status = self.makeStatusLocked()
            DispatchQueue.main.async { [weak self] in
                self?.onStatusChanged?(status)
            }
        }
    }

    var datasetDirectoryPath: String {
        stateQueue.sync {
            datasetDirectoryURL.path
        }
    }

    func setEnabled(_ enabled: Bool) {
        stateQueue.sync {
            isEnabled = enabled
            if !enabled {
                activeSession = nil
                collectionSessionID = nil
                collectionSessionStartedAt = 0
                lastEventKey = "collector_session_stopped"
                lastEventDetail = nil
            } else {
                acceptedCount = 0
                rejectedCount = 0
                rollingFrames.removeAll()
                collectionSessionID = UUID().uuidString
                collectionSessionStartedAt = Date().timeIntervalSince1970
                lastEventKey = "collector_session_ready"
                lastEventDetail = selectedLabel.datasetKey
            }
        }
        if enabled {
            startActivityUpdates()
        } else {
            stopActivityUpdates()
        }
        emitStatus()
    }

    func setSelectedLabel(_ label: AispectImpactDatasetLabel) {
        stateQueue.sync {
            guard !isEnabled else { return }
            selectedLabel = label
            lastEventKey = "collector_task_updated"
            lastEventDetail = label.datasetKey
        }
        emitStatus()
    }

    func beginCapture(at timestamp: TimeInterval) {
        stateQueue.sync {
            guard isEnabled else { return }
            guard activeSession == nil else { return }
            let pre = Array(rollingFrames.suffix(config.preFrames))
            activeSession = CaptureSession(
                id: UUID().uuidString,
                label: selectedLabel,
                startedAt: timestamp,
                preFrames: pre,
                recordedFrames: [],
                remainingPostFrames: nil,
                anchorTimestamp: nil,
                endedTouchKind: nil,
                endedLiftOffset: nil,
                lastSampleRateHz: config.fallbackSampleRate
            )
            lastEventKey = "collector_down_window_waiting"
            lastEventDetail = selectedLabel.datasetKey
        }
        emitStatus()
    }

    func appendSample(_ frame: AispectImpactFrame, timestamp: TimeInterval, sampleRateHz: Double) {
        var shouldEmitStatus = false
        stateQueue.sync {
            let capturedFrame = CapturedFrame(timestamp: timestamp, frame: frame)
            rollingFrames.append(capturedFrame)
            if rollingFrames.count > config.rollingFrameLimit {
                rollingFrames.removeFirst(rollingFrames.count - config.rollingFrameLimit)
            }

            guard var session = activeSession else { return }
            session.recordedFrames.append(capturedFrame)
            if sampleRateHz > 0 {
                session.lastSampleRateHz = sampleRateHz
            }

            if let remaining = session.remainingPostFrames {
                let next = remaining - 1
                session.remainingPostFrames = next
                activeSession = session
                if next <= 0 {
                    finalizeCaptureLocked()
                    shouldEmitStatus = true
                }
            } else {
                activeSession = session
            }
        }
        if shouldEmitStatus {
            emitStatus()
        }
    }

    func markDownWindowReady() {
        stateQueue.sync {
            guard activeSession != nil else { return }
            lastEventKey = "collector_down_window_ready_model_pending"
            lastEventDetail = selectedLabel.datasetKey
        }
        emitStatus()
    }

    func acceptCapture(anchorTimestamp: TimeInterval, touchKind: String, liftOffset: Double?) -> TimeInterval {
        let delay: TimeInterval = stateQueue.sync {
            guard var session = activeSession else { return 0 }
            session.anchorTimestamp = anchorTimestamp
            session.endedTouchKind = touchKind
            session.endedLiftOffset = liftOffset
            session.remainingPostFrames = config.postFrames
            activeSession = session
            lastEventKey = "collector_collecting_postroll"
            lastEventDetail = touchKind

            let sampleRate = max(session.lastSampleRateHz, config.fallbackSampleRate)
            return Double(config.postFrames) / sampleRate
        }
        emitStatus()
        return delay
    }

    func rejectCapture(reasonKey: String) {
        stateQueue.sync {
            guard activeSession != nil else { return }
            activeSession = nil
            rejectedCount += 1
            lastEventKey = reasonKey
            lastEventDetail = nil
        }
        emitStatus()
    }

    func finalizeCaptureIfNeeded() {
        var shouldEmitStatus = false
        stateQueue.sync {
            guard activeSession != nil else { return }
            finalizeCaptureLocked()
            shouldEmitStatus = true
        }
        if shouldEmitStatus {
            emitStatus()
        }
    }

    private func finalizeCaptureLocked() {
        guard var session = activeSession else { return }
        activeSession = nil

        guard let anchorTimestamp = session.anchorTimestamp else {
            rejectedCount += 1
            lastEventKey = "collector_event_missing_anchor"
            lastEventDetail = nil
            return
        }

        let allFrames = session.preFrames + session.recordedFrames
        guard !allFrames.isEmpty else {
            rejectedCount += 1
            lastEventKey = "collector_event_empty"
            lastEventDetail = nil
            return
        }

        let downReferenceSourceIndex = nearestFrameIndex(for: session.startedAt, in: allFrames)
        let liftReferenceSourceIndex = nearestFrameIndex(for: anchorTimestamp, in: allFrames)
        guard liftReferenceSourceIndex >= downReferenceSourceIndex else {
            rejectedCount += 1
            lastEventKey = "collector_event_invalid_anchor_order"
            lastEventDetail = nil
            return
        }

        let startSourceIndex = max(0, downReferenceSourceIndex - config.preFrames)
        let endSourceIndex = min(allFrames.count - 1, liftReferenceSourceIndex + config.postFrames)
        let sampleRate = max(session.lastSampleRateHz, config.fallbackSampleRate)
        let downReferenceTimestamp = allFrames[downReferenceSourceIndex].timestamp
        let downReferenceFrameIndex = 0
        let liftReferenceFrameIndex = liftReferenceSourceIndex - downReferenceSourceIndex
        let startFrameIndex = startSourceIndex - downReferenceSourceIndex
        let endFrameIndex = endSourceIndex - downReferenceSourceIndex
        let windowComplete = startSourceIndex == downReferenceSourceIndex - config.preFrames
            && endSourceIndex == liftReferenceSourceIndex + config.postFrames
        var frameRecords: [SampleFrameRecord] = []

        for absoluteIndex in startSourceIndex...endSourceIndex {
            let source = allFrames[absoluteIndex]
            frameRecords.append(
                SampleFrameRecord(
                    frameIndex: absoluteIndex - downReferenceSourceIndex,
                    relativeTimestamp: source.timestamp - downReferenceTimestamp,
                    timeSinceTouchDown: source.timestamp - session.startedAt,
                    timeSinceAnchor: source.timestamp - anchorTimestamp,
                    x: source.frame.x,
                    y: source.frame.y,
                    z: source.frame.z,
                    x_norm: source.frame.xNorm,
                    y_norm: source.frame.yNorm,
                    rotationRateX: source.frame.rotationRateX,
                    rotationRateY: source.frame.rotationRateY,
                    rotationRateZ: source.frame.rotationRateZ,
                    gravityX: source.frame.gravityX,
                    gravityY: source.frame.gravityY,
                    gravityZ: source.frame.gravityZ
                )
            )
        }

        let record = SampleRecord(
            schemaVersion: 5,
            id: session.id,
            createdAt: Date().timeIntervalSince1970,
            collectionSessionID: collectionSessionID ?? "unknown",
            collectionSessionStartedAt: collectionSessionStartedAt,
            label: session.label,
            manualLabelSource: "user_selected",
            sourceTouchKind: session.endedTouchKind ?? "unknown",
            liftOffset: session.endedLiftOffset,
            sampleRateHz: sampleRate,
            referenceFrameIndex: downReferenceFrameIndex,
            anchorTimestamp: session.startedAt,
            downAnchorTimestamp: session.startedAt,
            liftAnchorTimestamp: anchorTimestamp,
            downReferenceFrameIndex: downReferenceFrameIndex,
            liftReferenceFrameIndex: liftReferenceFrameIndex,
            startFrameIndex: startFrameIndex,
            endFrameIndex: endFrameIndex,
            preDownFrameCount: max(0, -startFrameIndex),
            postLiftFrameCount: max(0, endFrameIndex - liftReferenceFrameIndex),
            windowComplete: windowComplete,
            activityEvidence: latestActivity,
            appVersion: bundleShortVersion,
            appBuild: bundleBuildVersion,
            frames: frameRecords
        )

        saveLocked(record: record, label: session.label)
        acceptedCount += 1
        lastEventKey = "collector_event_saved"
        lastEventDetail = session.label.datasetKey
    }

    private func nearestFrameIndex(for anchorTimestamp: TimeInterval, in frames: [CapturedFrame]) -> Int {
        guard let index = frames.enumerated().min(by: { lhs, rhs in
            abs(lhs.element.timestamp - anchorTimestamp) < abs(rhs.element.timestamp - anchorTimestamp)
        })?.offset else {
            return max(frames.count - 1, 0)
        }
        return index
    }

    private func saveLocked(record: SampleRecord, label: AispectImpactDatasetLabel) {
        do {
            try ensureDatasetDirectoryExistsLocked()
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(record)
            let fileName = "\(Int(Date().timeIntervalSince1970 * 1000))_\(label.datasetKey)_\(record.id).json"
            let destination = datasetDirectoryURL.appendingPathComponent(fileName)
            try data.write(to: destination, options: .atomic)
            countsByLabel[label.datasetKey, default: 0] += 1
        } catch {
            rejectedCount += 1
            lastEventKey = "collector_event_write_failed"
            lastEventDetail = nil
        }
    }

    private func loadExistingSummary() {
        countsByLabel = [:]
        guard let enumerator = fileManager.enumerator(at: datasetDirectoryURL, includingPropertiesForKeys: nil) else {
            return
        }

        for case let fileURL as URL in enumerator {
            guard fileURL.pathExtension == "json" else { continue }
            let fileName = fileURL.deletingPathExtension().lastPathComponent
            let components = fileName.split(separator: "_")
            guard components.count >= 3 else { continue }
            let key = "\(components[1])_\(components[2])"
            countsByLabel[key, default: 0] += 1
        }
    }

    private func ensureDatasetDirectoryExistsLocked() throws {
        try fileManager.createDirectory(
            at: datasetDirectoryURL,
            withIntermediateDirectories: true
        )
    }

    private var datasetDirectoryURL: URL {
        let documentDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        return documentDirectory.appendingPathComponent(config.datasetDirectoryName, isDirectory: true)
    }

    private func emitStatus() {
        let status: Status = stateQueue.sync {
            makeStatusLocked()
        }

        DispatchQueue.main.async { [weak self] in
            self?.onStatusChanged?(status)
        }
    }

    private func startActivityUpdates() {
        guard CMMotionActivityManager.isActivityAvailable() else { return }
        activityManager.startActivityUpdates(to: activityQueue) { [weak self] activity in
            guard let self, let activity else { return }
            self.stateQueue.async {
                self.latestActivity = self.makeActivityEvidence(from: activity)
                let status = self.makeStatusLocked()
                DispatchQueue.main.async { [weak self] in
                    self?.onStatusChanged?(status)
                }
            }
        }
    }

    private func stopActivityUpdates() {
        guard CMMotionActivityManager.isActivityAvailable() else { return }
        activityManager.stopActivityUpdates()
    }

    private func makeActivityEvidence(from activity: CMMotionActivity) -> AispectImpactSystemActivityEvidence {
        AispectImpactSystemActivityEvidence(
            timestamp: activity.startDate.timeIntervalSince1970,
            stationary: activity.stationary,
            walking: activity.walking,
            running: activity.running,
            automotive: activity.automotive,
            cycling: activity.cycling,
            unknown: activity.unknown,
            confidence: confidenceString(activity.confidence),
            authorizationStatus: motionAuthorizationStatusString()
        )
    }

    private func confidenceString(_ confidence: CMMotionActivityConfidence) -> String {
        switch confidence {
        case .low:
            return "low"
        case .medium:
            return "medium"
        case .high:
            return "high"
        @unknown default:
            return "unknown"
        }
    }

    private func motionAuthorizationStatusString() -> String {
        if #available(iOS 11.0, *) {
            switch CMMotionActivityManager.authorizationStatus() {
            case .authorized:
                return "authorized"
            case .denied:
                return "denied"
            case .restricted:
                return "restricted"
            case .notDetermined:
                return "not_determined"
            @unknown default:
                return "unknown"
            }
        }
        return "unknown"
    }

    private var bundleShortVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown"
    }

    private var bundleBuildVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown"
    }

    private func makeStatusLocked() -> Status {
        Status(
            isEnabled: isEnabled,
            selectedLabel: selectedLabel,
            isLabelLocked: isEnabled,
            totalCount: countsByLabel.values.reduce(0, +),
            countsByLabel: countsByLabel,
            acceptedCount: acceptedCount,
            rejectedCount: rejectedCount,
            targetCount: config.targetAcceptedSamples,
            lastEventKey: lastEventKey,
            lastEventDetail: lastEventDetail,
            collectionSessionID: collectionSessionID,
            latestActivity: latestActivity,
            datasetDirectoryPath: datasetDirectoryURL.path
        )
    }
}
