import UIKit

final class RightSidebarView: UIView {
    private struct HeightDefaults {
        static let account: CGFloat = 0.45
        static let workflow: CGFloat = 0.3
        static let tool: CGFloat = account * 0.35
        static let expanded: CGFloat = 0.85
        static let compressed: CGFloat = 0.07
    }

    private let glassView = GlassMaterialView(style: .barDefault)
    private let accountList = IconListView(title: "Accounts", itemSize: CGSize(width: 48, height: 48))
    private let dividerLabel = UILabel()
    private let workflowList = IconListView(title: "Aiff", itemSize: CGSize(width: 48, height: 48))
    private let toolList = IconListView(title: "Tools", itemSize: CGSize(width: 48, height: 48))

    private var accountHeightConstraint: NSLayoutConstraint?
    private var toolHeightConstraint: NSLayoutConstraint?
    private var workflowHeightConstraint: NSLayoutConstraint?
    var selectedAccountIndex: Int = 0
    private var accountCollapseWorkItem: DispatchWorkItem?
    private var toolCollapseWorkItem: DispatchWorkItem?
    private var workflowCollapseWorkItem: DispatchWorkItem?
    private var accountHeightMultiplier: CGFloat = HeightDefaults.account
    private var toolHeightMultiplier: CGFloat = HeightDefaults.tool
    private var workflowHeightMultiplier: CGFloat = HeightDefaults.workflow

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

        dividerLabel.text = "Advisors"
        dividerLabel.font = UIFont.systemFont(ofSize: 10, weight: .semibold)
        dividerLabel.textColor = UIColor.secondaryLabel
        dividerLabel.textAlignment = .center
        dividerLabel.translatesAutoresizingMaskIntoConstraints = false

        let stack = UIStackView(arrangedSubviews: [accountList, dividerLabel, workflowList, toolList])
        stack.axis = .vertical
        stack.spacing = 6
        stack.translatesAutoresizingMaskIntoConstraints = false

        glassView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(glassView)
        glassView.contentView.addSubview(stack)
        NSLayoutConstraint.activate([
            glassView.leadingAnchor.constraint(equalTo: leadingAnchor),
            glassView.trailingAnchor.constraint(equalTo: trailingAnchor),
            glassView.topAnchor.constraint(equalTo: topAnchor),
            glassView.bottomAnchor.constraint(equalTo: bottomAnchor),

            stack.leadingAnchor.constraint(equalTo: glassView.contentView.leadingAnchor, constant: 8),
            stack.trailingAnchor.constraint(equalTo: glassView.contentView.trailingAnchor, constant: -8),
            stack.topAnchor.constraint(equalTo: glassView.contentView.topAnchor, constant: 8),
            stack.bottomAnchor.constraint(equalTo: glassView.contentView.bottomAnchor, constant: -8)
        ])

        accountHeightConstraint = accountList.heightAnchor.constraint(equalTo: heightAnchor, multiplier: HeightDefaults.account)
        accountHeightConstraint?.isActive = true
        workflowHeightConstraint = workflowList.heightAnchor.constraint(equalTo: heightAnchor, multiplier: HeightDefaults.workflow)
        workflowHeightConstraint?.isActive = true
        toolHeightConstraint = toolList.heightAnchor.constraint(equalTo: heightAnchor, multiplier: HeightDefaults.tool)
        toolHeightConstraint?.isActive = true

        accountList.updateItems((0..<20).map { index in "A\(index)" })
        workflowList.updateItems((0..<24).map { index in "W\(index)" })
        toolList.updateItems((0..<12).map { index in "T\(index)" })

        accountList.onScroll = { [weak self] _ in
            self?.handleAccountScroll()
        }
        toolList.onScroll = { [weak self] _ in
            self?.handleToolScroll()
        }
        workflowList.onScroll = { [weak self] _ in
            self?.handleWorkflowScroll()
        }
    }

    func expandToolList() {
        applyLayout(
            account: HeightDefaults.compressed,
            workflow: HeightDefaults.compressed,
            tool: HeightDefaults.expanded,
            animated: true
        )
    }

    func collapseToolList() {
        applyLayout(
            account: HeightDefaults.account,
            workflow: HeightDefaults.workflow,
            tool: HeightDefaults.tool,
            animated: true
        )
    }

    func expandWorkflowList() {
        applyLayout(
            account: HeightDefaults.compressed,
            workflow: HeightDefaults.expanded,
            tool: HeightDefaults.compressed,
            animated: true
        )
    }

    func collapseWorkflowList() {
        applyLayout(
            account: HeightDefaults.account,
            workflow: HeightDefaults.workflow,
            tool: HeightDefaults.tool,
            animated: true
        )
    }

    func accountAnchorPoint(at index: Int) -> CGPoint? {
        accountList.anchorPointForItem(at: index)
    }

    private func handleToolScroll() {
        // 中文注释：工具列滚动时扩大高度，并在停止滚动后自动收回
        cancelAllCollapse()
        expandToolList()
        scheduleToolCollapse()
    }

    private func handleAccountScroll() {
        // 中文注释：账号列滚动时优先收回工具列
        cancelAllCollapse()
        applyLayout(
            account: HeightDefaults.expanded,
            workflow: HeightDefaults.compressed,
            tool: HeightDefaults.compressed,
            animated: true
        )
        scheduleAccountCollapse()
    }

    private func handleWorkflowScroll() {
        // 中文注释：流程列滚动时扩大高度，并在停止滚动后自动收回
        cancelAllCollapse()
        expandWorkflowList()
        scheduleWorkflowCollapse()
    }

    private func scheduleAccountCollapse() {
        accountCollapseWorkItem?.cancel()
        let workItem = DispatchWorkItem { [weak self] in
            self?.applyLayout(
                account: HeightDefaults.account,
                workflow: HeightDefaults.workflow,
                tool: HeightDefaults.tool,
                animated: true
            )
        }
        accountCollapseWorkItem = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + 2, execute: workItem)
    }

    private func scheduleToolCollapse() {
        toolCollapseWorkItem?.cancel()
        let workItem = DispatchWorkItem { [weak self] in
            self?.collapseToolList()
        }
        toolCollapseWorkItem = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + 2, execute: workItem)
    }

    private func scheduleWorkflowCollapse() {
        workflowCollapseWorkItem?.cancel()
        let workItem = DispatchWorkItem { [weak self] in
            self?.collapseWorkflowList()
        }
        workflowCollapseWorkItem = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + 2, execute: workItem)
    }

    private func cancelAllCollapse() {
        accountCollapseWorkItem?.cancel()
        workflowCollapseWorkItem?.cancel()
        toolCollapseWorkItem?.cancel()
    }

    private func applyLayout(account: CGFloat, workflow: CGFloat, tool: CGFloat, animated: Bool) {
        setAccountHeightMultiplier(account, animated: animated)
        setWorkflowHeightMultiplier(workflow, animated: animated)
        setToolHeightMultiplier(tool, animated: animated)
    }

    private func setAccountHeightMultiplier(_ multiplier: CGFloat, animated: Bool) {
        if abs(accountHeightMultiplier - multiplier) < 0.001 {
            return
        }
        accountHeightMultiplier = multiplier
        applyAccountHeightMultiplier(multiplier, animated: animated)
    }

    private func setToolHeightMultiplier(_ multiplier: CGFloat, animated: Bool) {
        if abs(toolHeightMultiplier - multiplier) < 0.001 {
            return
        }
        toolHeightMultiplier = multiplier
        applyToolHeightMultiplier(multiplier, animated: animated)
    }

    private func setWorkflowHeightMultiplier(_ multiplier: CGFloat, animated: Bool) {
        if abs(workflowHeightMultiplier - multiplier) < 0.001 {
            return
        }
        workflowHeightMultiplier = multiplier
        applyWorkflowHeightMultiplier(multiplier, animated: animated)
    }

    private func applyAccountHeightMultiplier(_ multiplier: CGFloat, animated: Bool) {
        accountHeightConstraint?.isActive = false
        accountHeightConstraint = accountList.heightAnchor.constraint(equalTo: heightAnchor, multiplier: multiplier)
        accountHeightConstraint?.isActive = true
        if animated {
            UIView.animate(withDuration: 0.25, delay: 0, options: [.curveEaseOut]) {
                self.layoutIfNeeded()
            }
        } else {
            setNeedsLayout()
        }
    }

    private func applyToolHeightMultiplier(_ multiplier: CGFloat, animated: Bool) {
        toolHeightConstraint?.isActive = false
        toolHeightConstraint = toolList.heightAnchor.constraint(equalTo: heightAnchor, multiplier: multiplier)
        toolHeightConstraint?.isActive = true
        if animated {
            UIView.animate(withDuration: 0.25, delay: 0, options: [.curveEaseOut]) {
                self.layoutIfNeeded()
            }
        } else {
            setNeedsLayout()
        }
    }

    private func applyWorkflowHeightMultiplier(_ multiplier: CGFloat, animated: Bool) {
        workflowHeightConstraint?.isActive = false
        workflowHeightConstraint = workflowList.heightAnchor.constraint(equalTo: heightAnchor, multiplier: multiplier)
        workflowHeightConstraint?.isActive = true
        if animated {
            UIView.animate(withDuration: 0.25, delay: 0, options: [.curveEaseOut]) {
                self.layoutIfNeeded()
            }
        } else {
            setNeedsLayout()
        }
    }
}

final class IconListView: UIView {
    private let titleLabel = UILabel()
    private let listView = VirtualListView()
    private var items: [String] = []
    var onScroll: ((CGPoint) -> Void)?

    init(title: String, itemSize: CGSize) {
        super.init(frame: .zero)
        titleLabel.text = title
        listView.metrics = VirtualListMetrics(itemSize: itemSize, spacing: 8, contentInsets: UIEdgeInsets(top: 4, left: 0, bottom: 4, right: 0))
        setupView()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        listView.metrics = VirtualListMetrics(itemSize: CGSize(width: 48, height: 48), spacing: 8, contentInsets: UIEdgeInsets(top: 4, left: 0, bottom: 4, right: 0))
        setupView()
    }

    private func setupView() {
        titleLabel.font = UIFont.systemFont(ofSize: 10, weight: .semibold)
        titleLabel.textColor = UIColor.secondaryLabel
        titleLabel.translatesAutoresizingMaskIntoConstraints = false

        listView.translatesAutoresizingMaskIntoConstraints = false
        listView.dataSource = self
        listView.onScroll = { [weak self] offset in
            self?.onScroll?(offset)
        }

        addSubview(titleLabel)
        addSubview(listView)

        NSLayoutConstraint.activate([
            titleLabel.leadingAnchor.constraint(equalTo: leadingAnchor),
            titleLabel.trailingAnchor.constraint(equalTo: trailingAnchor),
            titleLabel.topAnchor.constraint(equalTo: topAnchor),

            listView.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 4),
            listView.leadingAnchor.constraint(equalTo: leadingAnchor),
            listView.trailingAnchor.constraint(equalTo: trailingAnchor),
            listView.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])
    }

    func updateItems(_ items: [String]) {
        self.items = items
        listView.reloadData()
    }

    func layoutSnapshot() -> VirtualListLayoutSnapshot {
        listView.layoutSnapshot()
    }

    func anchorPointForItem(at index: Int) -> CGPoint? {
        guard index >= 0, index < items.count else { return nil }
        let snapshot = listView.layoutSnapshot()
        let frame = snapshot.frameForItem(at: index)
        let visibleX = frame.minX - snapshot.contentOffset.x
        let visibleY = frame.midY - snapshot.contentOffset.y
        return CGPoint(x: visibleX, y: visibleY)
    }
}

extension IconListView: VirtualListDataSource {
    func listView(_ listView: VirtualListView, itemCountInSection section: Int) -> Int {
        items.count
    }

    func listView(_ listView: VirtualListView, viewForItemAt index: Int, reusableView: UIView?) -> UIView {
        let view = (reusableView as? IconTextItemView) ?? IconTextItemView()
        view.configure(title: items[index])
        return view
    }
}

final class IconTextItemView: UIView {
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
        backgroundColor = UIColor.systemGreen.withAlphaComponent(0.2)
        layer.cornerRadius = 6
        layer.borderWidth = 1
        layer.borderColor = UIColor.systemGreen.withAlphaComponent(0.3).cgColor

        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        titleLabel.font = UIFont.systemFont(ofSize: 10, weight: .medium)
        titleLabel.textAlignment = .center
        titleLabel.textColor = UIColor.label
        titleLabel.numberOfLines = 2

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
