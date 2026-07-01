import UIKit

struct MessageBubbleStyle {
    var cornerRadius: CGFloat
    var borderWidth: CGFloat
    var borderColor: UIColor
    var outgoingBlurStyle: UIBlurEffect.Style
    var incomingBlurStyle: UIBlurEffect.Style
    var shadowColor: UIColor
    var shadowOpacity: Float
    var shadowRadius: CGFloat
    var shadowOffset: CGSize
    var nameFont: UIFont
    var textFont: UIFont
    var nameBadgeTopOffset: CGFloat
    var nameBadgeLeadingInset: CGFloat

    static var systemDefault: MessageBubbleStyle {
        MessageBubbleStyle(
            cornerRadius: 16,
            borderWidth: 0.5,
            borderColor: UIColor.white.withAlphaComponent(0.3),
            outgoingBlurStyle: .systemUltraThinMaterial,
            incomingBlurStyle: .systemUltraThinMaterial,
            shadowColor: UIColor.black,
            shadowOpacity: 0.12,
            shadowRadius: 6,
            shadowOffset: CGSize(width: 0, height: 2),
            nameFont: UIFont.systemFont(ofSize: 12, weight: .medium),
            textFont: UIFont.systemFont(ofSize: 14),
            nameBadgeTopOffset: -10,
            nameBadgeLeadingInset: 8
        )
    }

    static var detailedDefault: MessageBubbleStyle {
        MessageBubbleStyle(
            cornerRadius: 16,
            borderWidth: 0.5,
            borderColor: UIColor.white.withAlphaComponent(0.3),
            outgoingBlurStyle: .systemUltraThinMaterial,
            incomingBlurStyle: .systemUltraThinMaterial,
            shadowColor: UIColor.black,
            shadowOpacity: 0.14,
            shadowRadius: 6,
            shadowOffset: CGSize(width: 0, height: 2),
            nameFont: UIFont.systemFont(ofSize: 12, weight: .semibold),
            textFont: UIFont.systemFont(ofSize: 14),
            nameBadgeTopOffset: -8,
            nameBadgeLeadingInset: 8
        )
    }
}

final class MessageBubbleView: UIView {
    private let glassView = GlassMaterialView(style: .bubbleDefault)
    private let nameBadgeView = UIVisualEffectView(effect: UIBlurEffect(style: .systemThinMaterial))
    private let nameLabel = UILabel()
    private let messageLabel = UILabel()
    private var nameBadgeTopConstraint: NSLayoutConstraint?
    private var nameBadgeLeadingConstraint: NSLayoutConstraint?

    private var style: MessageBubbleStyle = .systemDefault

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

    func applyStyle(_ style: MessageBubbleStyle) {
        self.style = style
        glassView.applyStyle(GlassMaterialStyle(
            cornerRadius: style.cornerRadius,
            borderWidth: style.borderWidth,
            borderColor: style.borderColor,
            blurStyle: style.outgoingBlurStyle,
            shadowColor: style.shadowColor,
            shadowOpacity: style.shadowOpacity,
            shadowRadius: style.shadowRadius,
            shadowOffset: style.shadowOffset
        ))

        nameBadgeView.layer.cornerRadius = style.cornerRadius / 2
        nameBadgeView.layer.cornerCurve = .continuous
        nameBadgeView.clipsToBounds = true

        messageLabel.font = style.textFont
        messageLabel.textColor = UIColor.label
        messageLabel.layer.shadowColor = style.shadowColor.cgColor
        messageLabel.layer.shadowOpacity = style.shadowOpacity
        messageLabel.layer.shadowRadius = style.shadowRadius
        messageLabel.layer.shadowOffset = style.shadowOffset

        nameLabel.font = style.nameFont
        nameLabel.textColor = UIColor.label.withAlphaComponent(0.8)
        nameBadgeTopConstraint?.constant = style.nameBadgeTopOffset
        nameBadgeLeadingConstraint?.constant = style.nameBadgeLeadingInset
    }

    func configure(text: String, senderName: String?, isOutgoing: Bool, showsName: Bool) {
        messageLabel.text = text
        nameLabel.text = senderName
        nameBadgeView.isHidden = !showsName

        let blurStyle = isOutgoing ? style.outgoingBlurStyle : style.incomingBlurStyle
        glassView.setBlurStyle(blurStyle)
    }

    private func setupView() {
        backgroundColor = .clear

        glassView.translatesAutoresizingMaskIntoConstraints = false
        nameBadgeView.translatesAutoresizingMaskIntoConstraints = false
        nameLabel.translatesAutoresizingMaskIntoConstraints = false
        messageLabel.translatesAutoresizingMaskIntoConstraints = false

        messageLabel.numberOfLines = 0

        addSubview(glassView)
        addSubview(nameBadgeView)
        nameBadgeView.contentView.addSubview(nameLabel)
        addSubview(messageLabel)

        nameBadgeLeadingConstraint = nameBadgeView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 8)
        nameBadgeTopConstraint = nameBadgeView.topAnchor.constraint(equalTo: topAnchor, constant: -10)

        NSLayoutConstraint.activate([
            glassView.leadingAnchor.constraint(equalTo: leadingAnchor),
            glassView.trailingAnchor.constraint(equalTo: trailingAnchor),
            glassView.topAnchor.constraint(equalTo: topAnchor),
            glassView.bottomAnchor.constraint(equalTo: bottomAnchor),

            nameBadgeLeadingConstraint!,
            nameBadgeTopConstraint!,

            nameLabel.leadingAnchor.constraint(equalTo: nameBadgeView.contentView.leadingAnchor, constant: 6),
            nameLabel.trailingAnchor.constraint(equalTo: nameBadgeView.contentView.trailingAnchor, constant: -6),
            nameLabel.topAnchor.constraint(equalTo: nameBadgeView.contentView.topAnchor, constant: 2),
            nameLabel.bottomAnchor.constraint(equalTo: nameBadgeView.contentView.bottomAnchor, constant: -2),

            messageLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 12),
            messageLabel.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -12),
            messageLabel.topAnchor.constraint(equalTo: topAnchor, constant: 10),
            messageLabel.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -10)
        ])
    }
}

final class BubbleMenuBarView: UIView {
    private let glassView = GlassMaterialView(style: .barDefault)
    private let stackView = UIStackView()

    override init(frame: CGRect) {
        super.init(frame: frame)
        setupView()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupView()
    }

    private func setupView() {
        backgroundColor = .clear
        glassView.translatesAutoresizingMaskIntoConstraints = false
        glassView.applyStyle(GlassMaterialStyle(
            cornerRadius: 12,
            borderWidth: 0.5,
            borderColor: UIColor.white.withAlphaComponent(0.3),
            blurStyle: .systemUltraThinMaterial,
            shadowColor: UIColor.black,
            shadowOpacity: 0.12,
            shadowRadius: 10,
            shadowOffset: CGSize(width: 0, height: 4)
        ))
        addSubview(glassView)

        stackView.axis = .horizontal
        stackView.spacing = 8
        stackView.translatesAutoresizingMaskIntoConstraints = false
        glassView.contentView.addSubview(stackView)

        NSLayoutConstraint.activate([
            glassView.leadingAnchor.constraint(equalTo: leadingAnchor),
            glassView.trailingAnchor.constraint(equalTo: trailingAnchor),
            glassView.topAnchor.constraint(equalTo: topAnchor),
            glassView.bottomAnchor.constraint(equalTo: bottomAnchor),

            stackView.leadingAnchor.constraint(equalTo: glassView.contentView.leadingAnchor, constant: 8),
            stackView.trailingAnchor.constraint(equalTo: glassView.contentView.trailingAnchor, constant: -8),
            stackView.topAnchor.constraint(equalTo: glassView.contentView.topAnchor, constant: 6),
            stackView.bottomAnchor.constraint(equalTo: glassView.contentView.bottomAnchor, constant: -6)
        ])

        let items: [(String, String)] = [
            ("Voice", "waveform"),
            ("Copy", "doc.on.doc"),
            ("Forward", "arrowshape.turn.up.right"),
            ("Save", "bookmark"),
            ("Select", "checkmark.circle"),
            ("Quote", "quote.bubble"),
            ("Zoom", "arrow.up.left.and.down.right.magnifyingglass"),
            ("Delete", "trash")
        ]
        for (title, systemName) in items {
            let button = MenuIconButton(title: title, systemName: systemName)
            stackView.addArrangedSubview(button)
        }
    }
}

final class MenuIconButton: UIButton {
    init(title: String, systemName: String) {
        super.init(frame: .zero)
        let image = UIImage(systemName: systemName)
        setImage(image, for: .normal)
        setTitle(title, for: .normal)
        setTitleColor(UIColor.label, for: .normal)
        titleLabel?.font = UIFont.systemFont(ofSize: 10, weight: .medium)
        tintColor = UIColor.label
        contentEdgeInsets = UIEdgeInsets(top: 2, left: 4, bottom: 2, right: 4)
        imageEdgeInsets = UIEdgeInsets(top: 0, left: -2, bottom: 0, right: 2)
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
    }
}
