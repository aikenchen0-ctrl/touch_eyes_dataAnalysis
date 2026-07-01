import UIKit

final class TopBarView: UIView {
    private let glassView = GlassMaterialView(style: .barDefault)
    private let leftStatusButton = IconButton(systemName: "exclamationmark.circle")
    private let titleButton = TextButton(title: "Session")
    private let editButton = IconButton(systemName: "pencil")

    private let addUserButton = IconButton(systemName: "person.badge.plus")
    private let accountButton = TextButton(title: "Account")
    private let searchButton = IconButton(systemName: "magnifyingglass")

    var onStatusTapped: (() -> Void)?

    override init(frame: CGRect) {
        super.init(frame: frame)
        setupView()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupView()
    }

    private func setupView() {
        let leftStack = UIStackView(arrangedSubviews: [leftStatusButton, titleButton, editButton])
        leftStack.axis = .horizontal
        leftStack.spacing = 6
        leftStack.alignment = .center

        let rightStack = UIStackView(arrangedSubviews: [addUserButton, accountButton, searchButton])
        rightStack.axis = .horizontal
        rightStack.spacing = 6
        rightStack.alignment = .center

        let spacer = UIView()
        let container = UIStackView(arrangedSubviews: [leftStack, spacer, rightStack])
        container.axis = .horizontal
        container.alignment = .center
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
            container.topAnchor.constraint(equalTo: glassView.contentView.topAnchor),
            container.bottomAnchor.constraint(equalTo: glassView.contentView.bottomAnchor)
        ])

        leftStatusButton.addTarget(self, action: #selector(handleStatusTap), for: .touchUpInside)
    }

    @objc private func handleStatusTap() {
        onStatusTapped?()
    }
}

final class IconButton: UIButton {
    init(systemName: String) {
        super.init(frame: .zero)
        let image = UIImage(systemName: systemName)
        setImage(image, for: .normal)
        tintColor = UIColor.label
        widthAnchor.constraint(equalToConstant: 20).isActive = true
        heightAnchor.constraint(equalToConstant: 20).isActive = true
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
    }
}

final class TextButton: UIButton {
    init(title: String) {
        super.init(frame: .zero)
        setTitle(title, for: .normal)
        setTitleColor(UIColor.label, for: .normal)
        titleLabel?.font = UIFont.systemFont(ofSize: 14, weight: .medium)
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
    }
}
