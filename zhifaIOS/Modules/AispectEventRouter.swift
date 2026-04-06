import UIKit

enum AispectHotZone {
    case full
    case rect(CGRect)
    case leftEdge(thickness: CGFloat, verticalRange: ClosedRange<CGFloat>?)
    case rightEdge(thickness: CGFloat, verticalRange: ClosedRange<CGFloat>?)
    case topEdge(thickness: CGFloat, horizontalRange: ClosedRange<CGFloat>?)
    case bottomEdge(thickness: CGFloat, horizontalRange: ClosedRange<CGFloat>?)

    func contains(point: CGPoint, in bounds: CGRect) -> Bool {
        // 中文注释：统一入口，按区域类型判断是否命中。
        switch self {
        case .full:
            return bounds.contains(point)
        case .rect(let rect):
            return rect.contains(point)
        case .leftEdge(let thickness, let range):
            guard point.x <= bounds.minX + thickness else { return false }
            return containsVerticalRange(point: point, bounds: bounds, range: range)
        case .rightEdge(let thickness, let range):
            guard point.x >= bounds.maxX - thickness else { return false }
            return containsVerticalRange(point: point, bounds: bounds, range: range)
        case .topEdge(let thickness, let range):
            guard point.y <= bounds.minY + thickness else { return false }
            return containsHorizontalRange(point: point, bounds: bounds, range: range)
        case .bottomEdge(let thickness, let range):
            guard point.y >= bounds.maxY - thickness else { return false }
            return containsHorizontalRange(point: point, bounds: bounds, range: range)
        }
    }

    private func containsVerticalRange(point: CGPoint, bounds: CGRect, range: ClosedRange<CGFloat>?) -> Bool {
        // 中文注释：纵向区间命中以高度比例计算。
        guard let range = range else { return true }
        let ratio = (point.y - bounds.minY) / max(bounds.height, 1)
        return range.contains(ratio)
    }

    private func containsHorizontalRange(point: CGPoint, bounds: CGRect, range: ClosedRange<CGFloat>?) -> Bool {
        // 中文注释：横向区间命中以宽度比例计算。
        guard let range = range else { return true }
        let ratio = (point.x - bounds.minX) / max(bounds.width, 1)
        return range.contains(ratio)
    }
}

enum AispectEdgeAction {
    case leftEdgeUp
    case leftEdgeDown
    case leftEdgePullIn
    case rightEdgeUp
    case rightEdgeDown
    case rightEdgePullIn
}

struct AispectEdgeGestureConfig {
    var edgeThickness: CGFloat
    var triggerDistance: CGFloat

    init(edgeThickness: CGFloat = 24, triggerDistance: CGFloat = 24) {
        self.edgeThickness = edgeThickness
        self.triggerDistance = triggerDistance
    }
}

struct AispectHoldProgress {
    let progress: CGFloat
    let location: CGPoint
    let isHeavyCandidate: Bool
}

struct AispectHoldTrigger {
    let location: CGPoint
    let isHeavyCandidate: Bool
}

struct AispectEventBinding {
    let kinds: Set<AispectTouchKind>
    let zone: AispectHotZone
    let handler: (AispectTouchEvent) -> Void
    let predicate: ((AispectTouchEvent) -> Bool)?

    init(kinds: Set<AispectTouchKind>, zone: AispectHotZone, predicate: ((AispectTouchEvent) -> Bool)? = nil, handler: @escaping (AispectTouchEvent) -> Void) {
        self.kinds = kinds
        self.zone = zone
        self.predicate = predicate
        self.handler = handler
    }
}

final class AispectEventRouter {
    private let engine: AispectInteractionEngine
    private var bindings: [AispectEventBinding] = []

    private var holdStartTimestamp: TimeInterval?
    private var holdStartLocation: CGPoint = .zero
    private var isDragging = false
    private var didTriggerHold = false
    private var edgeDragContext: EdgeDragContext?

    var holdMoveTolerance: CGFloat = 8
    var onHoldProgress: ((AispectHoldProgress) -> Void)?
    var onHoldProgressEnd: (() -> Void)?
    var onHoldTriggered: ((AispectHoldTrigger) -> Void)?
    var onEdgeAction: ((AispectEdgeAction) -> Void)?
    var edgeConfig = AispectEdgeGestureConfig()

    init(engine: AispectInteractionEngine) {
        self.engine = engine
    }

    func register(_ binding: AispectEventBinding) {
        bindings.append(binding)
    }

    func reset() {
        holdStartTimestamp = nil
        holdStartLocation = .zero
        isDragging = false
        didTriggerHold = false
        edgeDragContext = nil
        onHoldProgressEnd?()
    }

    func handle(event: AispectTouchEvent, in view: UIView) {
        // 中文注释：先更新全局状态，再分发绑定回调。
        updateHoldProgress(with: event)
        updateEdgeGesture(with: event, in: view.bounds)

        let bounds = view.bounds
        for binding in bindings {
            guard binding.kinds.contains(event.kind) else { continue }
            guard binding.zone.contains(point: event.location, in: bounds) else { continue }
            if let predicate = binding.predicate, !predicate(event) {
                continue
            }
            binding.handler(event)
        }
    }

    private func updateHoldProgress(with event: AispectTouchEvent) {
        switch event.kind {
        case .sample:
            if holdStartTimestamp == nil {
                holdStartTimestamp = event.timestamp
                holdStartLocation = event.location
                isDragging = false
            }

            guard let start = holdStartTimestamp, !isDragging else { return }

            let distance = hypot(event.location.x - holdStartLocation.x, event.location.y - holdStartLocation.y)
            if distance > holdMoveTolerance {
                // 中文注释：移动过大直接取消长按进度。
                endHoldProgress()
                return
            }

            let duration = max(engine.holdDuration, 0.1)
            let progress = min(max((event.timestamp - start) / duration, 0), 1)
            let isHeavyCandidate = false
            onHoldProgress?(AispectHoldProgress(progress: CGFloat(progress), location: event.location, isHeavyCandidate: isHeavyCandidate))
            if progress >= 1, !isHeavyCandidate, !didTriggerHold {
                didTriggerHold = true
                onHoldTriggered?(AispectHoldTrigger(location: event.location, isHeavyCandidate: isHeavyCandidate))
            }
        case .dragStart:
            isDragging = true
            endHoldProgress()
        case .dragEnd, .cancel, .lightTap, .heavyTap, .lightHold, .heavyHold, .heavyPress:
            isDragging = false
            endHoldProgress()
        default:
            break
        }
    }

    private func endHoldProgress() {
        guard holdStartTimestamp != nil else { return }
        holdStartTimestamp = nil
        didTriggerHold = false
        onHoldProgressEnd?()
    }

    private func updateEdgeGesture(with event: AispectTouchEvent, in bounds: CGRect) {
        switch event.kind {
        case .dragStart:
            // 中文注释：仅在拖拽开始时记录边缘上下文。
            edgeDragContext = makeEdgeContextIfNeeded(for: event.location, in: bounds)
        case .drag:
            guard var context = edgeDragContext, !context.triggered else { return }
            let action = resolveEdgeAction(from: context, to: event.location)
            if let action {
                context.triggered = true
                edgeDragContext = context
                onEdgeAction?(action)
            }
        case .dragEnd, .cancel, .lightTap, .heavyTap:
            edgeDragContext = nil
        default:
            break
        }
    }

    private func makeEdgeContextIfNeeded(for point: CGPoint, in bounds: CGRect) -> EdgeDragContext? {
        let thickness = edgeConfig.edgeThickness
        if point.x <= bounds.minX + thickness {
            return EdgeDragContext(side: .left, start: point, triggered: false)
        }
        if point.x >= bounds.maxX - thickness {
            return EdgeDragContext(side: .right, start: point, triggered: false)
        }
        return nil
    }

    private func resolveEdgeAction(from context: EdgeDragContext, to point: CGPoint) -> AispectEdgeAction? {
        let dx = point.x - context.start.x
        let dy = point.y - context.start.y
        let threshold = edgeConfig.triggerDistance
        if abs(dx) < threshold, abs(dy) < threshold {
            return nil
        }
        switch context.side {
        case .left:
            if dx > threshold { return .leftEdgePullIn }
            if dy < -threshold { return .leftEdgeUp }
            if dy > threshold { return .leftEdgeDown }
        case .right:
            if dx < -threshold { return .rightEdgePullIn }
            if dy < -threshold { return .rightEdgeUp }
            if dy > threshold { return .rightEdgeDown }
        }
        return nil
    }

    private struct EdgeDragContext {
        let side: EdgeSide
        let start: CGPoint
        var triggered: Bool
    }

    private enum EdgeSide {
        case left
        case right
    }
}
