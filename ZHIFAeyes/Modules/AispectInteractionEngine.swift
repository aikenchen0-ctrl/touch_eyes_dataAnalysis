import UIKit

enum AispectTouchKind {
    case sample
    case lightTap
    case heavyTap
    case lightHold
    case heavyHold
    case heavyPress
    case dragStart
    case drag
    case dragEnd
    case cancel
}

struct AispectTouchEvent {
    let kind: AispectTouchKind
    let location: CGPoint
    let timestamp: TimeInterval
    let liftOffset: CGFloat?

    init(kind: AispectTouchKind, location: CGPoint, timestamp: TimeInterval, liftOffset: CGFloat? = nil) {
        self.kind = kind
        self.location = location
        self.timestamp = timestamp
        self.liftOffset = liftOffset
    }
}

protocol AispectGestureDelegate: AnyObject {
    func aispectRecognizer(_ recognizer: AispectGestureRecognizer, didEmit event: AispectTouchEvent)
}

final class AispectInteractionEngine {
    // 中文注释：长按触发时长。
    var holdDuration: TimeInterval = 0.6
    // 中文注释：拖拽激活的最小位移。
    var dragActivationRadius: CGFloat = 8
    // 中文注释：采样节流间隔。
    var minimumSampleInterval: TimeInterval = 1.0 / 90.0
    // 中文注释：抬起偏移触发 heavyTap 的阈值。
    var heavyLiftOffsetThreshold: CGFloat = 6
}

final class AispectGestureRecognizer: UIGestureRecognizer {
    weak var eventDelegate: AispectGestureDelegate?
    let engine: AispectInteractionEngine

    private var initialLocation: CGPoint = .zero
    private var initialTimestamp: TimeInterval = 0
    private var lastSampleTimestamp: TimeInterval = 0
    private var holdWorkItem: DispatchWorkItem?
    private var isDragging = false
    private var lastLocation: CGPoint = .zero
    private var lastEventTimestamp: TimeInterval = 0

    init(engine: AispectInteractionEngine, target: Any?, action: Selector?) {
        self.engine = engine
        super.init(target: target, action: action)
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent) {
        guard let touch = touches.first, let view = view else { return }
        state = .began
        initialLocation = touch.location(in: view)
        initialTimestamp = touch.timestamp
        lastSampleTimestamp = touch.timestamp
        lastLocation = initialLocation
        lastEventTimestamp = touch.timestamp
        isDragging = false

        // 中文注释：起始帧先发 sample，用于驱动动效。
        emitEvent(
            kind: .sample,
            location: initialLocation,
            timestamp: initialTimestamp
        )
        scheduleHoldEvent()
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent) {
        guard let touch = touches.first, let view = view else { return }
        // 中文注释：coalesced touches 仅用于提高采样密度。
        let coalescedTouches = event.coalescedTouches(for: touch) ?? [touch]
        var latestLocation = touch.location(in: view)
        var latestTimestamp = touch.timestamp
        var didEmitSample = false

        for sample in coalescedTouches {
            let location = sample.location(in: view)
            let interval = sample.timestamp - lastSampleTimestamp
            latestLocation = location
            latestTimestamp = sample.timestamp
            if interval >= engine.minimumSampleInterval {
                lastSampleTimestamp = sample.timestamp
                lastLocation = location
                lastEventTimestamp = sample.timestamp
                emitEvent(
                    kind: .sample,
                    location: location,
                    timestamp: sample.timestamp
                )
                didEmitSample = true
            }
        }

        let distance = hypot(latestLocation.x - initialLocation.x, latestLocation.y - initialLocation.y)

        // 中文注释：位移超过阈值时进入拖拽态。
        if distance > engine.dragActivationRadius {
            if !isDragging {
                isDragging = true
                holdWorkItem?.cancel()
                emitEvent(
                    kind: .dragStart,
                    location: latestLocation,
                    timestamp: latestTimestamp
                )
            }
            emitEvent(
                kind: .drag,
                location: latestLocation,
                timestamp: latestTimestamp
            )
        } else if !didEmitSample, latestTimestamp != lastEventTimestamp {
            lastLocation = latestLocation
            lastEventTimestamp = latestTimestamp
        }
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent) {
        defer { resetState() }
        guard let touch = touches.first, let view = view else { return }
        let location = touch.location(in: view)
        if isDragging {
            emitEvent(
                kind: .dragEnd,
                location: location,
                timestamp: touch.timestamp
            )
        } else {
            // 中文注释：抬起偏移用于捕捉手指离开瞬间的位移峰值。
            let offset = hypot(location.x - lastLocation.x, location.y - lastLocation.y)
            let isHeavy = offset >= engine.heavyLiftOffsetThreshold
            let kind: AispectTouchKind = isHeavy ? .heavyTap : .lightTap
            emitEvent(
                kind: kind,
                location: location,
                timestamp: touch.timestamp,
                liftOffset: offset
            )
        }
        state = .ended
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent) {
        if let touch = touches.first, let view = view {
            let location = touch.location(in: view)
            emitEvent(
                kind: .cancel,
                location: location,
                timestamp: touch.timestamp
            )
        } else if lastEventTimestamp > 0 {
            emitEvent(
                kind: .cancel,
                location: lastLocation,
                timestamp: lastEventTimestamp
            )
        }
        resetState()
        state = .cancelled
    }

    override func reset() {
        resetState()
        super.reset()
    }

    private func scheduleHoldEvent() {
        holdWorkItem?.cancel()
        // 中文注释：长按计时器，仅在未拖拽时触发。
        let workItem = DispatchWorkItem { [weak self] in
            guard let self = self else { return }
            guard self.state == .began || self.state == .changed else { return }
            guard self.view != nil else { return }
            let event = AispectTouchEvent(
                kind: .lightHold,
                location: self.initialLocation,
                timestamp: self.initialTimestamp
            )
            self.eventDelegate?.aispectRecognizer(self, didEmit: event)
            self.state = .changed
        }
        holdWorkItem = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + engine.holdDuration, execute: workItem)
    }

    private func emitEvent(kind: AispectTouchKind, location: CGPoint, timestamp: TimeInterval, liftOffset: CGFloat? = nil) {
        let event = AispectTouchEvent(
            kind: kind,
            location: location,
            timestamp: timestamp,
            liftOffset: liftOffset
        )
        eventDelegate?.aispectRecognizer(self, didEmit: event)
        state = .changed
    }

    private func resetState() {
        holdWorkItem?.cancel()
        holdWorkItem = nil
        isDragging = false
    }
}
