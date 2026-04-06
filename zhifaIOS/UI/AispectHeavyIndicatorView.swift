import UIKit

final class AispectHeavyIndicatorView: UIView {
    private let containerView = UIView()
    private let ringLayer = CAShapeLayer()
    private let glowLayer = CAShapeLayer()
    private let label = UILabel()
    private let size: CGFloat = 120
    private let padding: CGFloat = 16

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

        containerView.bounds = CGRect(x: 0, y: 0, width: size, height: size)
        containerView.backgroundColor = UIColor.clear
        addSubview(containerView)

        glowLayer.strokeColor = UIColor.systemYellow.withAlphaComponent(0.5).cgColor
        glowLayer.fillColor = UIColor.clear.cgColor
        glowLayer.lineWidth = 6
        glowLayer.shadowColor = UIColor.systemYellow.cgColor
        glowLayer.shadowOpacity = 0.7
        glowLayer.shadowRadius = 8
        glowLayer.shadowOffset = .zero
        containerView.layer.addSublayer(glowLayer)

        ringLayer.strokeColor = UIColor.systemYellow.withAlphaComponent(0.95).cgColor
        ringLayer.fillColor = UIColor.white.withAlphaComponent(0.08).cgColor
        ringLayer.lineWidth = 2
        containerView.layer.addSublayer(ringLayer)

        label.text = "HEAVY"
        label.font = UIFont.systemFont(ofSize: 16, weight: .bold)
        label.textColor = UIColor.systemYellow
        label.textAlignment = .center
        label.frame = containerView.bounds
        containerView.addSubview(label)
    }

    func show(at point: CGPoint, in bounds: CGRect) {
        let insetBounds = bounds.insetBy(dx: padding + size * 0.5, dy: padding + size * 0.5)
        let clampedX = min(max(point.x, insetBounds.minX), insetBounds.maxX)
        let clampedY = min(max(point.y, insetBounds.minY), insetBounds.maxY)
        containerView.center = CGPoint(x: clampedX, y: clampedY)

        let ringRect = containerView.bounds.insetBy(dx: 10, dy: 10)
        ringLayer.path = UIBezierPath(ovalIn: ringRect).cgPath
        glowLayer.path = UIBezierPath(ovalIn: containerView.bounds.insetBy(dx: 4, dy: 4)).cgPath

        containerView.alpha = 0
        containerView.transform = CGAffineTransform(scaleX: 0.85, y: 0.85)
        UIView.animate(withDuration: 0.12, delay: 0, options: [.curveEaseOut], animations: {
            self.containerView.alpha = 1
            self.containerView.transform = .identity
        }, completion: nil)

        let pulse = CABasicAnimation(keyPath: "transform.scale")
        pulse.fromValue = 0.95
        pulse.toValue = 1.05
        pulse.duration = 0.18
        pulse.autoreverses = true
        glowLayer.add(pulse, forKey: "pulse")
    }

    func hide() {
        UIView.animate(withDuration: 0.12, delay: 0, options: [.curveEaseIn], animations: {
            self.containerView.alpha = 0
            self.containerView.transform = CGAffineTransform(scaleX: 0.9, y: 0.9)
        }, completion: nil)
    }
}
