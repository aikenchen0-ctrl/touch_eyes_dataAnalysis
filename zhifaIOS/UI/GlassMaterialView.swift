import UIKit

struct GlassMaterialStyle {
    var cornerRadius: CGFloat
    var borderWidth: CGFloat
    var borderColor: UIColor
    var blurStyle: UIBlurEffect.Style
    var shadowColor: UIColor
    var shadowOpacity: Float
    var shadowRadius: CGFloat
    var shadowOffset: CGSize

    static var barDefault: GlassMaterialStyle {
        GlassMaterialStyle(
            cornerRadius: 12,
            borderWidth: 0.5,
            borderColor: UIColor.white.withAlphaComponent(0.35),
            blurStyle: .systemUltraThinMaterial,
            shadowColor: UIColor.black,
            shadowOpacity: 0.08,
            shadowRadius: 18,
            shadowOffset: CGSize(width: 0, height: 10)
        )
    }

    static var bubbleDefault: GlassMaterialStyle {
        GlassMaterialStyle(
            cornerRadius: 16,
            borderWidth: 0.5,
            borderColor: UIColor.white.withAlphaComponent(0.3),
            blurStyle: .systemUltraThinMaterial,
            shadowColor: UIColor.black,
            shadowOpacity: 0.12,
            shadowRadius: 6,
            shadowOffset: CGSize(width: 0, height: 2)
        )
    }

    static var inputDefault: GlassMaterialStyle {
        GlassMaterialStyle(
            cornerRadius: 12,
            borderWidth: 0.5,
            borderColor: UIColor.white.withAlphaComponent(0.3),
            blurStyle: .systemUltraThinMaterial,
            shadowColor: UIColor.black,
            shadowOpacity: 0.08,
            shadowRadius: 10,
            shadowOffset: CGSize(width: 0, height: 4)
        )
    }
}

final class GlassMaterialView: UIView {
    let contentView = UIView()
    private let blurView = UIVisualEffectView(effect: UIBlurEffect(style: .systemUltraThinMaterial))

    private var style: GlassMaterialStyle = .barDefault

    init(style: GlassMaterialStyle = .barDefault) {
        super.init(frame: .zero)
        setupView()
        applyStyle(style)
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupView()
        applyStyle(style)
    }

    func applyStyle(_ style: GlassMaterialStyle) {
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

    func setBlurStyle(_ blurStyle: UIBlurEffect.Style) {
        blurView.effect = UIBlurEffect(style: blurStyle)
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
}
