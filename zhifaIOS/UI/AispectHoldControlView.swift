import UIKit

final class AispectHoldControlView: UIView {
    private let ringLayer = CAShapeLayer()
    private let knobView = UIView()
    private var ringRadius: CGFloat = 0

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

        ringLayer.strokeColor = UIColor.white.withAlphaComponent(0.35).cgColor
        ringLayer.fillColor = UIColor.white.withAlphaComponent(0.06).cgColor
        ringLayer.lineWidth = 1
        layer.addSublayer(ringLayer)

        knobView.backgroundColor = UIColor.white.withAlphaComponent(0.6)
        knobView.layer.cornerRadius = 6
        addSubview(knobView)
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        let radius = min(bounds.width, bounds.height) * 0.5
        ringRadius = max(radius - 10, 1)
        let path = UIBezierPath(ovalIn: CGRect(x: bounds.midX - radius, y: bounds.midY - radius, width: radius * 2, height: radius * 2))
        ringLayer.path = path.cgPath
        ringLayer.frame = bounds

        resetKnob(animated: false)
    }

    func updateKnob(angle: CGFloat, normalizedDistance: CGFloat) {
        let clamped = min(max(normalizedDistance, 0), 1)
        let distance = ringRadius * clamped
        let center = CGPoint(x: bounds.midX, y: bounds.midY)
        let knobCenter = CGPoint(
            x: center.x + cos(angle) * distance,
            y: center.y + sin(angle) * distance
        )
        setKnobCenter(knobCenter)
    }

    func resetKnob(animated: Bool) {
        let center = CGPoint(x: bounds.midX, y: bounds.midY)
        if animated {
            UIView.animate(withDuration: 0.2, delay: 0, options: [.curveEaseOut, .beginFromCurrentState]) {
                self.setKnobCenter(center)
            }
        } else {
            setKnobCenter(center)
        }
    }

    private func setKnobCenter(_ center: CGPoint) {
        let knobSize: CGFloat = 12
        knobView.frame = CGRect(
            x: center.x - knobSize * 0.5,
            y: center.y - knobSize * 0.5,
            width: knobSize,
            height: knobSize
        )
    }
}
