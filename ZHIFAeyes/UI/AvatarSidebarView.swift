import UIKit

final class AvatarSidebarView: UIView {
    private let glassView = GlassMaterialView(style: .barDefault)
    private let listView = VirtualListView()
    private let pinnedView = AvatarItemView()
    private var pinnedTopConstraint: NSLayoutConstraint?
    private var pinnedBottomConstraint: NSLayoutConstraint?
    private let insertTopButton = SidebarInsertButton(title: "Forward")
    private let insertBottomButton = SidebarInsertButton(title: "Moments")
    private var insertWorkItem: DispatchWorkItem?
    private var isInsertVisible = false
    private let insertButtonSize = CGSize(width: 52, height: 20)
    private let insertButtonGap: CGFloat = 8
    private var items: [String] = []
    var selectedIndex: Int = 0 {
        didSet {
            updatePinnedState()
            scheduleInsertButtons()
        }
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
        backgroundColor = .clear

        glassView.translatesAutoresizingMaskIntoConstraints = false
        listView.translatesAutoresizingMaskIntoConstraints = false
        listView.metrics = VirtualListMetrics(
            itemSize: CGSize(width: 48, height: 48),
            spacing: 8,
            contentInsets: UIEdgeInsets(top: 8, left: 8, bottom: 8, right: 8)
        )
        listView.dataSource = self
        listView.onScroll = { [weak self] _ in
            self?.updatePinnedState()
            self?.scheduleInsertButtons()
        }

        addSubview(glassView)
        glassView.contentView.addSubview(listView)
        glassView.contentView.addSubview(insertTopButton)
        glassView.contentView.addSubview(insertBottomButton)
        glassView.contentView.addSubview(pinnedView)
        pinnedView.translatesAutoresizingMaskIntoConstraints = false
        pinnedView.isHidden = true
        insertTopButton.isHidden = true
        insertBottomButton.isHidden = true
        NSLayoutConstraint.activate([
            glassView.leadingAnchor.constraint(equalTo: leadingAnchor),
            glassView.trailingAnchor.constraint(equalTo: trailingAnchor),
            glassView.topAnchor.constraint(equalTo: topAnchor),
            glassView.bottomAnchor.constraint(equalTo: bottomAnchor),

            listView.leadingAnchor.constraint(equalTo: glassView.contentView.leadingAnchor),
            listView.trailingAnchor.constraint(equalTo: glassView.contentView.trailingAnchor),
            listView.topAnchor.constraint(equalTo: glassView.contentView.topAnchor),
            listView.bottomAnchor.constraint(equalTo: glassView.contentView.bottomAnchor)
        ])

        pinnedTopConstraint = pinnedView.topAnchor.constraint(equalTo: glassView.contentView.topAnchor, constant: 8)
        pinnedBottomConstraint = pinnedView.bottomAnchor.constraint(equalTo: glassView.contentView.bottomAnchor, constant: -8)
        NSLayoutConstraint.activate([
            pinnedView.leadingAnchor.constraint(equalTo: glassView.contentView.leadingAnchor, constant: 8),
            pinnedView.widthAnchor.constraint(equalToConstant: 48),
            pinnedView.heightAnchor.constraint(equalToConstant: 48),
            pinnedTopConstraint!
        ])

        items = (0..<200).map { index in "S\(index)" }
        listView.reloadData()
        updatePinnedState()
    }

    func layoutSnapshot() -> VirtualListLayoutSnapshot {
        listView.layoutSnapshot()
    }

    func anchorPointForItem(at index: Int) -> CGPoint? {
        guard index >= 0, index < items.count else { return nil }
        let snapshot = listView.layoutSnapshot()
        let frame = snapshot.frameForItem(at: index)
        let visibleX = frame.maxX - snapshot.contentOffset.x
        let visibleY = frame.midY - snapshot.contentOffset.y
        return CGPoint(x: visibleX, y: visibleY)
    }

    private func updatePinnedState() {
        guard selectedIndex >= 0, selectedIndex < items.count else {
            hideInsertButtons()
            return
        }
        let snapshot = listView.layoutSnapshot()
        let frame = snapshot.frameForItem(at: selectedIndex)
        let visibleTop = frame.minY - snapshot.contentOffset.y
        let visibleBottom = frame.maxY - snapshot.contentOffset.y
        let topLimit = snapshot.metrics.contentInsets.top
        let bottomLimit = snapshot.bounds.height - snapshot.metrics.contentInsets.bottom

        if visibleTop < topLimit {
            pinnedTopConstraint?.isActive = true
            pinnedBottomConstraint?.isActive = false
            pinnedView.isHidden = false
            pinnedView.configure(title: items[selectedIndex])
            listView.setHiddenIndices([selectedIndex])
            hideInsertButtons()
        } else if visibleBottom > bottomLimit {
            pinnedTopConstraint?.isActive = false
            pinnedBottomConstraint?.isActive = true
            pinnedView.isHidden = false
            pinnedView.configure(title: items[selectedIndex])
            listView.setHiddenIndices([selectedIndex])
            hideInsertButtons()
        } else {
            pinnedTopConstraint?.isActive = true
            pinnedBottomConstraint?.isActive = false
            pinnedView.isHidden = true
            listView.setHiddenIndices([])
        }
    }

    private func scheduleInsertButtons() {
        insertWorkItem?.cancel()
        insertWorkItem = nil
        hideInsertButtons()

        guard pinnedView.isHidden else { return }
        guard selectedVisibleFrame() != nil else { return }

        // 中文注释：停留 5 秒后再显示插入按钮，滚动/选中变化会重置
        let workItem = DispatchWorkItem { [weak self] in
            self?.showInsertButtonsIfNeeded()
        }
        insertWorkItem = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + 5.0, execute: workItem)
    }

    private func showInsertButtonsIfNeeded() {
        guard pinnedView.isHidden else { return }
        guard selectedVisibleFrame() != nil else { return }
        guard !isInsertVisible else {
            layoutInsertButtons()
            return
        }

        let topGap = insertButtonSize.height + insertButtonGap
        let bottomGap = insertButtonSize.height + insertButtonGap
        listView.setExtraSpacing(VirtualListExtraSpacing(index: selectedIndex, topGap: topGap, bottomGap: bottomGap))
        isInsertVisible = true
        insertTopButton.isHidden = false
        insertBottomButton.isHidden = false
        layoutInsertButtons()
    }

    private func hideInsertButtons() {
        insertTopButton.isHidden = true
        insertBottomButton.isHidden = true
        if !isInsertVisible { return }
        isInsertVisible = false
        listView.setExtraSpacing(nil)
    }

    private func selectedVisibleFrame() -> CGRect? {
        guard selectedIndex >= 0, selectedIndex < items.count else { return nil }
        let snapshot = listView.layoutSnapshot()
        let frame = snapshot.frameForItem(at: selectedIndex)
        let visibleFrame = CGRect(
            x: frame.minX - snapshot.contentOffset.x,
            y: frame.minY - snapshot.contentOffset.y,
            width: frame.width,
            height: frame.height
        )
        if visibleFrame.maxY < 0 || visibleFrame.minY > snapshot.bounds.height {
            return nil
        }
        return visibleFrame
    }

    private func layoutInsertButtons() {
        guard isInsertVisible, let frame = selectedVisibleFrame() else { return }
        let topGap = insertButtonSize.height + insertButtonGap
        let bottomGap = insertButtonSize.height + insertButtonGap
        let centerX = frame.midX
        let topCenterY = frame.minY - topGap * 0.5
        let bottomCenterY = frame.maxY + bottomGap * 0.5

        insertTopButton.frame = CGRect(
            x: centerX - insertButtonSize.width * 0.5,
            y: topCenterY - insertButtonSize.height * 0.5,
            width: insertButtonSize.width,
            height: insertButtonSize.height
        )
        insertBottomButton.frame = CGRect(
            x: centerX - insertButtonSize.width * 0.5,
            y: bottomCenterY - insertButtonSize.height * 0.5,
            width: insertButtonSize.width,
            height: insertButtonSize.height
        )
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        layoutInsertButtons()
    }
}

extension AvatarSidebarView: VirtualListDataSource {
    func listView(_ listView: VirtualListView, itemCountInSection section: Int) -> Int {
        items.count
    }

    func listView(_ listView: VirtualListView, viewForItemAt index: Int, reusableView: UIView?) -> UIView {
        let view = (reusableView as? AvatarItemView) ?? AvatarItemView()
        view.configure(title: items[index])
        return view
    }
}

final class AvatarItemView: UIView {
    private let titleLabel = UILabel()

    override init(frame: CGRect) {
        super.init(frame: frame)
        setupView()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupView()
    }

    private func setupView() {
        backgroundColor = UIColor.systemBlue.withAlphaComponent(0.2)
        layer.cornerRadius = 6
        layer.borderWidth = 1
        layer.borderColor = UIColor.systemBlue.withAlphaComponent(0.3).cgColor

        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        titleLabel.font = UIFont.systemFont(ofSize: 10, weight: .medium)
        titleLabel.textAlignment = .center
        titleLabel.textColor = UIColor.label

        addSubview(titleLabel)
        NSLayoutConstraint.activate([
            titleLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 2),
            titleLabel.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -2),
            titleLabel.centerYAnchor.constraint(equalTo: centerYAnchor)
        ])
    }

    func configure(title: String) {
        titleLabel.text = title
    }
}

final class SidebarInsertButton: UIButton {
    init(title: String) {
        super.init(frame: .zero)
        setupView(title: title)
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupView(title: "")
    }

    private func setupView(title: String) {
        setTitle(title, for: .normal)
        setTitleColor(UIColor.label, for: .normal)
        titleLabel?.font = UIFont.systemFont(ofSize: 10, weight: .semibold)
        backgroundColor = UIColor.systemRed.withAlphaComponent(0.2)
        layer.cornerRadius = 6
        layer.borderWidth = 1
        layer.borderColor = UIColor.systemRed.withAlphaComponent(0.4).cgColor
    }
}
