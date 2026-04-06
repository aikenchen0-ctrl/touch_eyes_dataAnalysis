import UIKit

struct ConnectionSegment {
    let start: CGPoint
    let end: CGPoint
    let bendsOutward: Bool
}

protocol ConnectionLineProvider: AnyObject {
    func connectionSegments() -> [ConnectionSegment]
}

final class ConnectionLineView: UIView {
    weak var provider: ConnectionLineProvider?

    private let shapeLayer = CAShapeLayer()
    private var displayLink: CADisplayLink?

    override init(frame: CGRect) {
        super.init(frame: frame)
        setupView()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupView()
    }

    private func setupView() {
        isUserInteractionEnabled = false
        backgroundColor = .clear
        shapeLayer.strokeColor = UIColor.systemBlue.withAlphaComponent(0.7).cgColor
        shapeLayer.fillColor = UIColor.clear.cgColor
        shapeLayer.lineWidth = 1.5
        shapeLayer.lineDashPattern = [6, 4]
        shapeLayer.shadowColor = UIColor.systemBlue.cgColor
        shapeLayer.shadowOpacity = 0.35
        shapeLayer.shadowRadius = 4
        shapeLayer.shadowOffset = CGSize(width: 0, height: 0)
        layer.addSublayer(shapeLayer)

        // 中文注释：高刷新率驱动连线更新，避免逐元素监听。
        let link = CADisplayLink(target: self, selector: #selector(step))
        if #available(iOS 15.0, *) {
            link.preferredFrameRateRange = CAFrameRateRange(minimum: 60, maximum: 120, preferred: 120)
        }
        link.add(to: .main, forMode: .common)
        displayLink = link
    }

    @objc private func step() {
        updatePath()
    }

    private func updatePath() {
        let segments = provider?.connectionSegments() ?? []
        let path = UIBezierPath()
        for segment in segments {
            let start = segment.start
            let end = segment.end
            let horizontalOffset: CGFloat = segment.bendsOutward ? 4 : -4
            let midX = start.x + horizontalOffset

            path.move(to: start)
            path.addLine(to: CGPoint(x: midX, y: start.y))
            path.addLine(to: CGPoint(x: midX, y: end.y))
            path.addLine(to: end)
        }
        shapeLayer.path = path.cgPath
    }

    deinit {
        displayLink?.invalidate()
    }
}
