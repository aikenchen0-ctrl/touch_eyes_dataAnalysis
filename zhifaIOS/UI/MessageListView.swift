import UIKit

final class MessageListView: UIView {
    private let tableView = UITableView()
    private var items: [IMMessage] = []
    private let layoutMetrics = MessageListLayoutMetrics(itemHeight: 110, spacing: 4, contentInsets: UIEdgeInsets(top: 8, left: 0, bottom: 8, right: 0))
    private var activeMenuIndex: Int?

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
        tableView.rowHeight = layoutMetrics.itemHeight
        tableView.dataSource = self
        tableView.delegate = self
        tableView.register(MessageBubbleCell.self, forCellReuseIdentifier: MessageBubbleCell.reuseID)

        addSubview(tableView)
        NSLayoutConstraint.activate([
            tableView.leadingAnchor.constraint(equalTo: leadingAnchor),
            tableView.trailingAnchor.constraint(equalTo: trailingAnchor),
            tableView.topAnchor.constraint(equalTo: topAnchor),
            tableView.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])

        items = DemoMessageFactory.makeMessages()
        tableView.reloadData()
    }

    func layoutSnapshot() -> MessageListLayoutSnapshot {
        MessageListLayoutSnapshot(
            metrics: layoutMetrics,
            itemCount: items.count,
            contentOffset: tableView.contentOffset,
            bounds: tableView.bounds,
            width: tableView.bounds.width
        )
    }

    func anchorPointForMessage(at index: Int, isOutgoing: Bool) -> CGPoint? {
        guard index >= 0, index < items.count else { return nil }
        let snapshot = layoutSnapshot()
        let frame = snapshot.frameForItem(at: index)
        // 中文注释：用固定比例估算气泡宽度，避免依赖具体 Cell 布局
        let bubbleWidth = min(frame.width * 0.7, frame.width - 24)
        let bubbleX = isOutgoing ? frame.maxX - bubbleWidth - 12 : frame.minX + 12
        let anchorX = isOutgoing ? bubbleX + bubbleWidth : bubbleX
        let visibleX = anchorX - snapshot.contentOffset.x
        let visibleY = frame.midY - snapshot.contentOffset.y
        return CGPoint(x: visibleX, y: visibleY)
    }
}

struct MessageListLayoutMetrics {
    let itemHeight: CGFloat
    let spacing: CGFloat
    let contentInsets: UIEdgeInsets
}

struct MessageListLayoutSnapshot {
    let metrics: MessageListLayoutMetrics
    let itemCount: Int
    let contentOffset: CGPoint
    let bounds: CGRect
    let width: CGFloat

    func frameForItem(at index: Int) -> CGRect {
        let stride = metrics.itemHeight + metrics.spacing
        let y = metrics.contentInsets.top + CGFloat(index) * stride
        let height = metrics.itemHeight
        return CGRect(x: 0, y: y, width: width, height: height)
    }
}

extension MessageListView: UITableViewDataSource {
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        items.count
    }

    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        guard let cell = tableView.dequeueReusableCell(withIdentifier: MessageBubbleCell.reuseID, for: indexPath) as? MessageBubbleCell else {
            return UITableViewCell()
        }
        let message = items[indexPath.row]
        let showsMenu = activeMenuIndex == indexPath.row
        cell.configure(message: message, showsMenu: showsMenu)
        return cell
    }
}

extension MessageListView: UITableViewDelegate {
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        let previousIndex = activeMenuIndex
        activeMenuIndex = (activeMenuIndex == indexPath.row) ? nil : indexPath.row
        var indices: [IndexPath] = [indexPath]
        if let previousIndex = previousIndex, previousIndex != indexPath.row {
            indices.append(IndexPath(row: previousIndex, section: 0))
        }
        tableView.reloadRows(at: indices, with: .none)
    }
}

final class MessageBubbleCell: UITableViewCell {
    static let reuseID = "MessageBubbleCell"

    private let bubbleView = MessageBubbleView()
    private let menuView = BubbleMenuBarView()
    private var leadingConstraint: NSLayoutConstraint?
    private var trailingConstraint: NSLayoutConstraint?
    private var menuLeadingConstraint: NSLayoutConstraint?
    private var menuTrailingConstraint: NSLayoutConstraint?
    private var menuVisible = false
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
        clipsToBounds = false
        backgroundColor = .clear
        contentView.backgroundColor = .clear
        contentView.clipsToBounds = false

        bubbleView.translatesAutoresizingMaskIntoConstraints = false
        menuView.translatesAutoresizingMaskIntoConstraints = false
        menuView.isHidden = true
        menuView.alpha = 0

        contentView.addSubview(bubbleView)
        contentView.addSubview(menuView)

        leadingConstraint = bubbleView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 12)
        trailingConstraint = bubbleView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -12)
        menuLeadingConstraint = menuView.leadingAnchor.constraint(equalTo: bubbleView.leadingAnchor)
        menuTrailingConstraint = menuView.trailingAnchor.constraint(equalTo: bubbleView.trailingAnchor)

        NSLayoutConstraint.activate([
            bubbleView.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 6),
            bubbleView.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -6),
            bubbleView.widthAnchor.constraint(lessThanOrEqualTo: contentView.widthAnchor, multiplier: 0.7),

            menuView.bottomAnchor.constraint(equalTo: bubbleView.topAnchor, constant: -6),
            menuLeadingConstraint!,
            menuTrailingConstraint!
        ])

        leadingConstraint?.isActive = true
        menuLeadingConstraint?.isActive = true
    }

    func configure(message: IMMessage, showsMenu: Bool) {
        let senderName = message.isOutgoing ? "Me" : "Sender"
        bubbleView.configure(text: message.previewText, senderName: senderName, isOutgoing: message.isOutgoing, showsName: !message.isOutgoing)
        updateMenuLayout(isOutgoing: message.isOutgoing)
        setMenuVisible(showsMenu, animated: menuVisible != showsMenu)
        let isOutgoing = message.isOutgoing

        if isOutgoing {
            leadingConstraint?.isActive = false
            trailingConstraint?.isActive = true
        } else {
            trailingConstraint?.isActive = false
            leadingConstraint?.isActive = true
        }
    }

    private func updateMenuLayout(isOutgoing: Bool) {
        menuLeadingConstraint?.isActive = !isOutgoing
        menuTrailingConstraint?.isActive = isOutgoing
    }

    private func setMenuVisible(_ visible: Bool, animated: Bool) {
        menuVisible = visible
        if visible {
            menuView.isHidden = false
        }
        if visible {
            menuView.transform = CGAffineTransform(translationX: 0, y: 6)
        }

        let animations = {
            self.menuView.alpha = visible ? 1 : 0
            let offset: CGFloat = visible ? 0 : 6
            self.menuView.transform = CGAffineTransform(translationX: 0, y: offset)
        }

        if animated {
            UIView.animate(withDuration: 0.18, delay: 0, options: [.curveEaseOut], animations: animations) { _ in
                if !visible {
                    self.menuView.isHidden = true
                }
            }
        } else {
            animations()
            if !visible {
                menuView.isHidden = true
            }
        }
    }
}

private enum DemoMessageFactory {
    static func makeMessages() -> [IMMessage] {
        let now = Date()
        return [
            IMMessage(id: "m1", conversationID: "c1", senderID: "u1", createdAt: now, isOutgoing: false, kind: .text("Sample incoming"), replyToMessageID: nil),
            IMMessage(id: "m2", conversationID: "c1", senderID: "u2", createdAt: now, isOutgoing: true, kind: .text("Sample outgoing"), replyToMessageID: nil)
        ]
    }
}

private extension IMMessage {
    var previewText: String {
        switch kind {
        case .text(let value):
            return value
        case .system(let value):
            return value
        case .file(_, let name):
            return name
        case .card(let value):
            return value
        case .custom(let type, _):
            return type
        default:
            return "Message"
        }
    }
}
