import UIKit

final class BottomBarView: UIView {
    private let glassView = GlassMaterialView(style: .barDefault)
    private let quickActionsView = QuickActionsView()
    private let inputViewContainer = AITextInputView()

    private let voiceButton = IconButton(systemName: "waveform")
    private let homeButton = IconButton(systemName: "house")
    private let giftButton = IconButton(systemName: "gift")
    private let addButton = IconButton(systemName: "plus")

    override init(frame: CGRect) {
        super.init(frame: frame)
        setupView()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupView()
    }

    private func setupView() {
        let leftStack = UIStackView(arrangedSubviews: [voiceButton, homeButton])
        leftStack.axis = .horizontal
        leftStack.spacing = 6
        leftStack.alignment = .center

        let rightStack = UIStackView(arrangedSubviews: [giftButton, addButton])
        rightStack.axis = .horizontal
        rightStack.spacing = 6
        rightStack.alignment = .center

        let inputRow = UIStackView(arrangedSubviews: [leftStack, inputViewContainer, rightStack])
        inputRow.axis = .horizontal
        inputRow.spacing = 8
        inputRow.alignment = .center

        let container = UIStackView(arrangedSubviews: [quickActionsView, inputRow])
        container.axis = .vertical
        container.spacing = 6
        container.translatesAutoresizingMaskIntoConstraints = false

        glassView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(glassView)
        glassView.contentView.addSubview(container)
        NSLayoutConstraint.activate([
            glassView.leadingAnchor.constraint(equalTo: leadingAnchor),
            glassView.trailingAnchor.constraint(equalTo: trailingAnchor),
            glassView.topAnchor.constraint(equalTo: topAnchor),
            glassView.bottomAnchor.constraint(equalTo: bottomAnchor),

            container.leadingAnchor.constraint(equalTo: glassView.contentView.leadingAnchor, constant: 12),
            container.trailingAnchor.constraint(equalTo: glassView.contentView.trailingAnchor, constant: -12),
            container.topAnchor.constraint(equalTo: glassView.contentView.topAnchor, constant: 6),
            container.bottomAnchor.constraint(equalTo: glassView.contentView.bottomAnchor, constant: -6)
        ])

        inputViewContainer.heightAnchor.constraint(equalToConstant: 44).isActive = true
        inputViewContainer.onEditingChanged = { [weak self] isEditing in
            self?.updateHomeIcon(isEditing: isEditing)
        }
    }

    func setInputFloatingBarVisible(_ visible: Bool, mode: AIFloatingActionMode) {
        inputViewContainer.setFloatingBarVisible(visible, mode: mode)
    }

    func setInputSuggestion(text: String?, style: AISuggestionDisplayStyle) {
        inputViewContainer.setSuggestion(text: text, style: style)
    }

    func inputViewFrame(in view: UIView) -> CGRect {
        inputViewContainer.convert(inputViewContainer.bounds, to: view)
    }

    private func updateHomeIcon(isEditing: Bool) {
        let name = isEditing ? "face.smiling" : "house"
        homeButton.setImage(UIImage(systemName: name), for: .normal)
    }
}

final class QuickActionsView: UIView {
    private let scrollView = UIScrollView()
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
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        stackView.axis = .horizontal
        stackView.spacing = 8
        stackView.translatesAutoresizingMaskIntoConstraints = false

        addSubview(scrollView)
        scrollView.addSubview(stackView)

        NSLayoutConstraint.activate([
            scrollView.leadingAnchor.constraint(equalTo: leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: trailingAnchor),
            scrollView.topAnchor.constraint(equalTo: topAnchor),
            scrollView.bottomAnchor.constraint(equalTo: bottomAnchor),

            stackView.leadingAnchor.constraint(equalTo: scrollView.contentLayoutGuide.leadingAnchor),
            stackView.trailingAnchor.constraint(equalTo: scrollView.contentLayoutGuide.trailingAnchor),
            stackView.topAnchor.constraint(equalTo: scrollView.contentLayoutGuide.topAnchor),
            stackView.bottomAnchor.constraint(equalTo: scrollView.contentLayoutGuide.bottomAnchor),
            stackView.heightAnchor.constraint(equalTo: scrollView.frameLayoutGuide.heightAnchor)
        ])

        let titles = ["Gallery", "Video", "Call", "Location", "RedPacket", "Transfer", "Files", "Cards", "Favorites", "Contracts"]
        for title in titles {
            let button = TextButton(title: title)
            button.titleLabel?.font = UIFont.systemFont(ofSize: 12, weight: .medium)
            stackView.addArrangedSubview(button)
        }
    }
}

final class AITextInputView: UIView, UITextViewDelegate {
    private let glassView = GlassMaterialView(style: .inputDefault)
    private let textView = UITextView()
    private let placeholderLabel = UILabel()
    private let floatingBar = AIFloatingActionBarView()
    private let suggestionLabel = UILabel()
    private var wasEmptyOnBeginEditing = true

    var onEditingChanged: ((Bool) -> Void)?

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
        textView.translatesAutoresizingMaskIntoConstraints = false
        textView.backgroundColor = .clear
        textView.delegate = self
        textView.font = UIFont.systemFont(ofSize: 14)

        placeholderLabel.translatesAutoresizingMaskIntoConstraints = false
        placeholderLabel.text = "Type a message"
        placeholderLabel.textColor = UIColor.secondaryLabel
        placeholderLabel.font = UIFont.systemFont(ofSize: 14)

        floatingBar.translatesAutoresizingMaskIntoConstraints = false
        floatingBar.isHidden = true

        suggestionLabel.translatesAutoresizingMaskIntoConstraints = false
        suggestionLabel.font = UIFont.systemFont(ofSize: 12)
        suggestionLabel.textColor = UIColor.systemBlue
        suggestionLabel.isHidden = true

        addSubview(glassView)
        glassView.contentView.addSubview(textView)
        glassView.contentView.addSubview(placeholderLabel)
        addSubview(floatingBar)
        addSubview(suggestionLabel)

        NSLayoutConstraint.activate([
            glassView.leadingAnchor.constraint(equalTo: leadingAnchor),
            glassView.trailingAnchor.constraint(equalTo: trailingAnchor),
            glassView.topAnchor.constraint(equalTo: topAnchor),
            glassView.bottomAnchor.constraint(equalTo: bottomAnchor),

            textView.leadingAnchor.constraint(equalTo: glassView.contentView.leadingAnchor, constant: 8),
            textView.trailingAnchor.constraint(equalTo: glassView.contentView.trailingAnchor, constant: -8),
            textView.topAnchor.constraint(equalTo: glassView.contentView.topAnchor, constant: 6),
            textView.bottomAnchor.constraint(equalTo: glassView.contentView.bottomAnchor, constant: -6),

            placeholderLabel.leadingAnchor.constraint(equalTo: textView.leadingAnchor, constant: 2),
            placeholderLabel.centerYAnchor.constraint(equalTo: textView.centerYAnchor),

            floatingBar.centerXAnchor.constraint(equalTo: centerXAnchor),
            floatingBar.bottomAnchor.constraint(equalTo: topAnchor, constant: -6),

            suggestionLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 8),
            suggestionLabel.bottomAnchor.constraint(equalTo: topAnchor, constant: -4)
        ])
    }

    func textViewDidChange(_ textView: UITextView) {
        placeholderLabel.isHidden = !textView.text.isEmpty
        updateModeForCurrentText()
    }

    func textViewDidBeginEditing(_ textView: UITextView) {
        wasEmptyOnBeginEditing = textView.text.isEmpty
        setFloatingBarVisible(true, mode: wasEmptyOnBeginEditing ? .compose : .revise)
        onEditingChanged?(true)
    }

    func textViewDidEndEditing(_ textView: UITextView) {
        setFloatingBarVisible(false, mode: .compose)
        onEditingChanged?(false)
    }

    func setFloatingBarVisible(_ visible: Bool, mode: AIFloatingActionMode) {
        floatingBar.isHidden = !visible
        if visible {
            floatingBar.updateMode(mode)
        }
    }

    func setSuggestion(text: String?, style: AISuggestionDisplayStyle) {
        suggestionLabel.text = text
        suggestionLabel.isHidden = (text == nil)
        switch style {
        case .badge:
            suggestionLabel.textAlignment = .left
        case .floating:
            suggestionLabel.textAlignment = .center
        }
    }

    private func updateModeForCurrentText() {
        if textView.text.isEmpty {
            floatingBar.updateMode(.compose)
        } else if !wasEmptyOnBeginEditing {
            floatingBar.updateMode(.revise)
        }
    }
}

enum AIFloatingActionMode {
    case compose
    case revise
}

enum AISuggestionDisplayStyle {
    case badge
    case floating
}

final class AIFloatingActionBarView: UIView {
    private let glassView = GlassMaterialView(style: .inputDefault)
    private let stackView = UIStackView()

    private var buttons: [UIButton] = []

    override init(frame: CGRect) {
        super.init(frame: frame)
        setupView()
        updateMode(.compose)
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupView()
        updateMode(.compose)
    }

    private func setupView() {
        backgroundColor = .clear
        glassView.translatesAutoresizingMaskIntoConstraints = false
        glassView.applyStyle(GlassMaterialStyle(
            cornerRadius: 10,
            borderWidth: 0.5,
            borderColor: UIColor.white.withAlphaComponent(0.3),
            blurStyle: .systemUltraThinMaterial,
            shadowColor: UIColor.black,
            shadowOpacity: 0.12,
            shadowRadius: 8,
            shadowOffset: CGSize(width: 0, height: 4)
        ))
        stackView.axis = .horizontal
        stackView.spacing = 8
        stackView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(glassView)
        glassView.contentView.addSubview(stackView)

        NSLayoutConstraint.activate([
            glassView.leadingAnchor.constraint(equalTo: leadingAnchor),
            glassView.trailingAnchor.constraint(equalTo: trailingAnchor),
            glassView.topAnchor.constraint(equalTo: topAnchor),
            glassView.bottomAnchor.constraint(equalTo: bottomAnchor),

            stackView.leadingAnchor.constraint(equalTo: glassView.contentView.leadingAnchor, constant: 8),
            stackView.trailingAnchor.constraint(equalTo: glassView.contentView.trailingAnchor, constant: -8),
            stackView.topAnchor.constraint(equalTo: glassView.contentView.topAnchor, constant: 4),
            stackView.bottomAnchor.constraint(equalTo: glassView.contentView.bottomAnchor, constant: -4)
        ])
    }

    func updateMode(_ mode: AIFloatingActionMode) {
        buttons.forEach { $0.removeFromSuperview() }
        buttons.removeAll()

        let titles: [String]
        switch mode {
        case .compose:
            titles = ["Rewrite", "Polish", "Continue"]
        case .revise:
            titles = ["Select", "Speak", "Delete", "Edit", "Rewrite", "Extend", "Polish"]
        }

        for title in titles {
            let button = TextButton(title: title)
            button.titleLabel?.font = UIFont.systemFont(ofSize: 12, weight: .medium)
            buttons.append(button)
            stackView.addArrangedSubview(button)
        }
    }
}
