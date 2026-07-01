import UIKit

struct UnreadMessageItem {
    let id: String
    let senderName: String
    let text: String
}

final class UnreadMessagesView: UIView {
    private let tableView = UITableView()
    private var items: [UnreadMessageItem] = []
    private let layoutMetrics = UnreadListLayoutMetrics(rowHeight: 88, spacing: 0, contentInsets: UIEdgeInsets(top: 8, left: 0, bottom: 8, right: 0))

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
        tableView.translatesAutoresizingMaskIntoConstraints = false
        tableView.backgroundColor = .clear
        tableView.separatorStyle = .none
        tableView.rowHeight = layoutMetrics.rowHeight
        tableView.dataSource = self
        tableView.register(UnreadMessageCell.self, forCellReuseIdentifier: UnreadMessageCell.reuseID)

        addSubview(tableView)
        NSLayoutConstraint.activate([
            tableView.leadingAnchor.constraint(equalTo: leadingAnchor),
            tableView.trailingAnchor.constraint(equalTo: trailingAnchor),
            tableView.topAnchor.constraint(equalTo: topAnchor),
            tableView.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])

        items = DemoUnreadFactory.makeItems()
        tableView.reloadData()
    }

    func layoutSnapshot() -> UnreadListLayoutSnapshot {
        UnreadListLayoutSnapshot(
            metrics: layoutMetrics,
            itemCount: items.count,
            contentOffset: tableView.contentOffset,
            bounds: tableView.bounds,
            width: tableView.bounds.width
        )
    }

    func visibleIndices(limit: Int) -> [Int] {
        let snapshot = layoutSnapshot()
        guard snapshot.itemCount > 0 else { return [] }
        let stride = snapshot.metrics.rowHeight + snapshot.metrics.spacing
        let minY = max(0, snapshot.contentOffset.y)
        let maxY = min(snapshot.contentOffset.y + snapshot.bounds.height, snapshot.metrics.totalHeight(itemCount: snapshot.itemCount))
        let startIndex = max(0, Int(floor((minY - snapshot.metrics.contentInsets.top) / stride)))
        let endIndex = min(snapshot.itemCount - 1, Int(ceil((maxY - snapshot.metrics.contentInsets.top) / stride)))
        if startIndex > endIndex { return [] }
        let indices = Array(startIndex...endIndex)
        return Array(indices.prefix(limit))
    }

    func avatarAnchorPoint(at index: Int) -> CGPoint? {
        guard index >= 0, index < items.count else { return nil }
        let snapshot = layoutSnapshot()
        let frame = snapshot.frameForItem(at: index)
        let avatarX = 12 + 48
        let visibleY = frame.midY - snapshot.contentOffset.y
        return CGPoint(x: CGFloat(avatarX), y: visibleY)
    }

    func bubbleAnchorPoint(at index: Int) -> CGPoint? {
        guard index >= 0, index < items.count else { return nil }
        let snapshot = layoutSnapshot()
        let frame = snapshot.frameForItem(at: index)
        let bubbleX = 12 + 48 + 12
        let visibleY = frame.midY - snapshot.contentOffset.y
        return CGPoint(x: CGFloat(bubbleX), y: visibleY)
    }
}

struct UnreadListLayoutMetrics {
    let rowHeight: CGFloat
    let spacing: CGFloat
    let contentInsets: UIEdgeInsets

    func totalHeight(itemCount: Int) -> CGFloat {
        contentInsets.top
            + contentInsets.bottom
            + CGFloat(itemCount) * rowHeight
            + max(0, CGFloat(itemCount - 1)) * spacing
    }
}

struct UnreadListLayoutSnapshot {
    let metrics: UnreadListLayoutMetrics
    let itemCount: Int
    let contentOffset: CGPoint
    let bounds: CGRect
    let width: CGFloat

    func frameForItem(at index: Int) -> CGRect {
        let stride = metrics.rowHeight + metrics.spacing
        let y = metrics.contentInsets.top + CGFloat(index) * stride
        return CGRect(x: 0, y: y, width: width, height: metrics.rowHeight)
    }
}

extension UnreadMessagesView: UITableViewDataSource {
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        items.count
    }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        guard let cell = tableView.dequeueReusableCell(withIdentifier: UnreadMessageCell.reuseID, for: indexPath) as? UnreadMessageCell else {
            return UITableViewCell()
        }
        cell.configure(item: items[indexPath.row])
        return cell
    }
}

final class UnreadMessageCell: UITableViewCell {
    static let reuseID = "UnreadMessageCell"

    private let avatarView = AvatarItemView()
    private let bubbleView = MessageBubbleView()

    override init(style: UITableViewCell.CellStyle, reuseIdentifier: String?) {
        super.init(style: style, reuseIdentifier: reuseIdentifier)
        setupView()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupView()
    }

    private func setupView() {
        selectionStyle = .none
        backgroundColor = .clear
        contentView.backgroundColor = .clear

        avatarView.translatesAutoresizingMaskIntoConstraints = false
        bubbleView.translatesAutoresizingMaskIntoConstraints = false
        bubbleView.applyStyle(.detailedDefault)

        contentView.addSubview(avatarView)
        contentView.addSubview(bubbleView)

        NSLayoutConstraint.activate([
            avatarView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 12),
            avatarView.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),
            avatarView.widthAnchor.constraint(equalToConstant: 48),
            avatarView.heightAnchor.constraint(equalToConstant: 48),

            bubbleView.leadingAnchor.constraint(equalTo: avatarView.trailingAnchor, constant: 12),
            bubbleView.trailingAnchor.constraint(lessThanOrEqualTo: contentView.trailingAnchor, constant: -12),
            bubbleView.centerYAnchor.constraint(equalTo: contentView.centerYAnchor),
            bubbleView.heightAnchor.constraint(equalToConstant: 64),
            bubbleView.widthAnchor.constraint(lessThanOrEqualTo: contentView.widthAnchor, multiplier: 0.7)
        ])
    }

    func configure(item: UnreadMessageItem) {
        avatarView.configure(title: item.senderName)
        bubbleView.configure(text: item.text, senderName: item.senderName, isOutgoing: false, showsName: true)
    }
}

private enum DemoUnreadFactory {
    static func makeItems() -> [UnreadMessageItem] {
        [
            UnreadMessageItem(id: "u1", senderName: "Alex", text: "Unread item one"),
            UnreadMessageItem(id: "u2", senderName: "Casey", text: "Unread item two"),
            UnreadMessageItem(id: "u3", senderName: "Jordan", text: "Unread item three")
        ]
    }
}
