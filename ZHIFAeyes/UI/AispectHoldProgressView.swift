import UIKit

final class AispectHoldProgressView: UIView {
    private let trackLayer = CALayer()
    private let fillLayer = CALayer()

    var progress: CGFloat = 0 {
        didSet { updateFill() }
    }

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
        layer.cornerRadius = 2
        layer.masksToBounds = true

        trackLayer.backgroundColor = UIColor.white.withAlphaComponent(0.18).cgColor
        fillLayer.backgroundColor = UIColor.white.withAlphaComponent(0.6).cgColor
        layer.addSublayer(trackLayer)
        layer.addSublayer(fillLayer)
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        trackLayer.frame = bounds
        updateFill()
    }

    private func updateFill() {
        let clamped = min(max(progress, 0), 1)
        let width = bounds.width * clamped
        fillLayer.frame = CGRect(x: 0, y: 0, width: width, height: bounds.height)
    }
}
