import UIKit

struct FloatingHomeStyle {
    var cornerRadius: CGFloat
    var borderWidth: CGFloat
    var borderColor: UIColor
    var blurStyle: UIBlurEffect.Style
    var shadowColor: UIColor
    var shadowOpacity: Float
    var shadowRadius: CGFloat
    var shadowOffset: CGSize

    static var systemDefault: FloatingHomeStyle {
        FloatingHomeStyle(
            cornerRadius: 20,
            borderWidth: 0.5,
            borderColor: UIColor.white.withAlphaComponent(0.35),
            blurStyle: .systemUltraThinMaterial,
            shadowColor: UIColor.black,
            shadowOpacity: 0.08,
            shadowRadius: 18,
            shadowOffset: CGSize(width: 0, height: 10)
        )
    }

    static var edgeToEdge: FloatingHomeStyle {
        FloatingHomeStyle(
            cornerRadius: 0,
            borderWidth: 0,
            borderColor: .clear,
            blurStyle: .systemUltraThinMaterial,
            shadowColor: .clear,
            shadowOpacity: 0,
            shadowRadius: 0,
            shadowOffset: .zero
        )
    }
}

final class FloatingHomeContainerView: UIView {
    let contentView = UIView()
    private let blurView = UIVisualEffectView(effect: UIBlurEffect(style: .systemUltraThinMaterial))

    private var style: FloatingHomeStyle = .systemDefault
    private var resetAnimator: UIViewPropertyAnimator?
    private var currentPressure: CGFloat = 0

    override init(frame: CGRect) {
        super.init(frame: frame)
        setupView()
        applyStyle(style)
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupView()
        applyStyle(style)
    }

    func applyStyle(_ style: FloatingHomeStyle) {
        self.style = style
        layer.cornerRadius = style.cornerRadius
        layer.cornerCurve = .continuous
        layer.borderWidth = style.borderWidth
        layer.borderColor = style.borderColor.cgColor
        layer.shadowColor = style.shadowColor.cgColor
        layer.shadowOpacity = style.shadowOpacity
        layer.shadowRadius = style.shadowRadius
        layer.shadowOffset = style.shadowOffset
        layer.masksToBounds = false

        blurView.effect = UIBlurEffect(style: style.blurStyle)
        blurView.layer.cornerRadius = style.cornerRadius
        blurView.layer.cornerCurve = .continuous
        blurView.clipsToBounds = true

        contentView.layer.cornerRadius = style.cornerRadius
        contentView.layer.cornerCurve = .continuous
        contentView.clipsToBounds = true
    }

    func updatePressure(_ progress: CGFloat) {
        resetAnimator?.stopAnimation(true)
        currentPressure = min(max(progress, 0), 1)
        applyPressure(currentPressure)
    }

    func releasePressure(animated: Bool = true) {
        currentPressure = 0
        if !animated {
            applyPressure(0)
            return
        }
        resetAnimator?.stopAnimation(true)
        let timing = UISpringTimingParameters(mass: 1.0, stiffness: 240, damping: 26, initialVelocity: .zero)
        let animator = UIViewPropertyAnimator(duration: 0, timingParameters: timing)
        animator.addAnimations { [weak self] in
            self?.applyPressure(0)
        }
        animator.startAnimation()
        resetAnimator = animator
    }

    private func setupView() {
        backgroundColor = .clear

        blurView.translatesAutoresizingMaskIntoConstraints = false
        contentView.translatesAutoresizingMaskIntoConstraints = false
        contentView.backgroundColor = .clear

        addSubview(blurView)
        addSubview(contentView)

        NSLayoutConstraint.activate([
            blurView.leadingAnchor.constraint(equalTo: leadingAnchor),
            blurView.trailingAnchor.constraint(equalTo: trailingAnchor),
            blurView.topAnchor.constraint(equalTo: topAnchor),
            blurView.bottomAnchor.constraint(equalTo: bottomAnchor),

            contentView.leadingAnchor.constraint(equalTo: leadingAnchor),
            contentView.trailingAnchor.constraint(equalTo: trailingAnchor),
            contentView.topAnchor.constraint(equalTo: topAnchor),
            contentView.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])
    }

    private func applyPressure(_ progress: CGFloat) {
        let clamped = min(max(progress, 0), 1)
        let scale = 1 - clamped * 0.015
        transform = CGAffineTransform(scaleX: scale, y: scale)
        layer.shadowOpacity = style.shadowOpacity + Float(clamped) * 0.1
        layer.shadowRadius = style.shadowRadius + clamped * 8
        let baseAlpha = style.borderColor.cgColor.alpha
        let borderAlpha = min(baseAlpha + clamped * 0.2, 1)
        layer.borderColor = style.borderColor.withAlphaComponent(borderAlpha).cgColor
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        layer.shadowPath = UIBezierPath(roundedRect: bounds, cornerRadius: style.cornerRadius).cgPath
    }
}
