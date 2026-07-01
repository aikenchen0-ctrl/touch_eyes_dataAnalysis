import UIKit

struct VirtualListMetrics {
    var itemSize: CGSize
    var spacing: CGFloat
    var contentInsets: UIEdgeInsets

    init(itemSize: CGSize, spacing: CGFloat = 8, contentInsets: UIEdgeInsets = .zero) {
        self.itemSize = itemSize
        self.spacing = spacing
        self.contentInsets = contentInsets
    }
}

struct VirtualListExtraSpacing {
    let index: Int
    let topGap: CGFloat
    let bottomGap: CGFloat

    func applies(to itemIndex: Int) -> Bool {
        itemIndex >= index
    }

    func offset(for itemIndex: Int) -> CGFloat {
        if itemIndex < index { return 0 }
        if itemIndex == index { return topGap }
        return topGap + bottomGap
    }

    var totalGap: CGFloat {
        topGap + bottomGap
    }
}

struct VirtualListLayoutSnapshot {
    let metrics: VirtualListMetrics
    let itemCount: Int
    let contentOffset: CGPoint
    let bounds: CGRect
    let extraSpacing: VirtualListExtraSpacing?

    func frameForItem(at index: Int) -> CGRect {
        var y = metrics.contentInsets.top + CGFloat(index) * (metrics.itemSize.height + metrics.spacing)
        if let extra = extraSpacing, extra.applies(to: index) {
            y += extra.offset(for: index)
        }
        return CGRect(x: metrics.contentInsets.left, y: y, width: metrics.itemSize.width, height: metrics.itemSize.height)
    }
}

protocol VirtualListDataSource: AnyObject {
    func listView(_ listView: VirtualListView, itemCountInSection section: Int) -> Int
    func listView(_ listView: VirtualListView, viewForItemAt index: Int, reusableView: UIView?) -> UIView
}

final class VirtualListView: UIView {
    weak var dataSource: VirtualListDataSource?
    var metrics: VirtualListMetrics = VirtualListMetrics(itemSize: CGSize(width: 44, height: 44)) {
        didSet { reloadData() }
    }
    var onScroll: ((CGPoint) -> Void)?
    private var hiddenIndices: Set<Int> = []
    private var extraSpacing: VirtualListExtraSpacing?

    private let scrollView = UIScrollView()
    private var reusableViews: [String: [UIView]] = [:]
    private var visibleViews: [Int: UIView] = [:]
    private var itemCount: Int = 0

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
        scrollView.showsVerticalScrollIndicator = false
        scrollView.delegate = self
        addSubview(scrollView)

        NSLayoutConstraint.activate([
            scrollView.leadingAnchor.constraint(equalTo: leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: trailingAnchor),
            scrollView.topAnchor.constraint(equalTo: topAnchor),
            scrollView.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])
    }

    func reloadData() {
        clearVisibleViews()
        itemCount = dataSource?.listView(self, itemCountInSection: 0) ?? 0
        updateContentSize()
        updateVisibleViews()
    }

    func setHiddenIndices(_ indices: Set<Int>) {
        hiddenIndices = indices
        updateVisibleViews()
    }

    func setExtraSpacing(_ spacing: VirtualListExtraSpacing?) {
        extraSpacing = spacing
        updateContentSize()
        updateVisibleViews()
    }

    func layoutSnapshot() -> VirtualListLayoutSnapshot {
        VirtualListLayoutSnapshot(
            metrics: metrics,
            itemCount: itemCount,
            contentOffset: scrollView.contentOffset,
            bounds: scrollView.bounds,
            extraSpacing: extraSpacing
        )
    }

    private func updateContentSize() {
        let height = metrics.contentInsets.top
            + metrics.contentInsets.bottom
            + CGFloat(itemCount) * metrics.itemSize.height
            + max(0, CGFloat(itemCount - 1)) * metrics.spacing
            + (extraSpacing?.totalGap ?? 0)
        scrollView.contentSize = CGSize(width: bounds.width, height: height)
    }

    private func updateVisibleViews() {
        guard itemCount > 0 else { return }
        let visibleBounds = scrollView.bounds
        let offsetY = scrollView.contentOffset.y
        let minY = max(0, offsetY)
        let maxY = min(scrollView.contentSize.height, offsetY + visibleBounds.height)

        // 中文注释：仅根据滚动区可视范围计算需要渲染的索引
        let stride = metrics.itemSize.height + metrics.spacing
        let baseStart = max(0, Int(floor((minY - metrics.contentInsets.top) / stride)))
        let baseEnd = min(itemCount - 1, Int(ceil((maxY - metrics.contentInsets.top) / stride)))
        let startIndex = max(0, baseStart - 2)
        let endIndex = min(itemCount - 1, baseEnd + 2)

        var nextVisible: [Int: UIView] = [:]
        for index in startIndex...endIndex {
            if let view = visibleViews[index] {
                view.frame = frameForItem(at: index)
                view.isHidden = hiddenIndices.contains(index)
                nextVisible[index] = view
            } else {
                let reusable = dequeueReusableView()
                let view = dataSource?.listView(self, viewForItemAt: index, reusableView: reusable) ?? UIView()
                view.frame = frameForItem(at: index)
                view.isHidden = hiddenIndices.contains(index)
                scrollView.addSubview(view)
                nextVisible[index] = view
            }
        }

        for (index, view) in visibleViews where nextVisible[index] == nil {
            enqueueReusableView(view)
            view.removeFromSuperview()
        }

        visibleViews = nextVisible
    }

    private func frameForItem(at index: Int) -> CGRect {
        var y = metrics.contentInsets.top + CGFloat(index) * (metrics.itemSize.height + metrics.spacing)
        if let extra = extraSpacing, extra.applies(to: index) {
            y += extra.offset(for: index)
        }
        return CGRect(x: metrics.contentInsets.left, y: y, width: metrics.itemSize.width, height: metrics.itemSize.height)
    }

    private func enqueueReusableView(_ view: UIView) {
        let key = String(describing: type(of: view))
        var bucket = reusableViews[key, default: []]
        bucket.append(view)
        reusableViews[key] = bucket
    }

    private func dequeueReusableView() -> UIView? {
        guard let key = reusableViews.keys.first, var bucket = reusableViews[key], !bucket.isEmpty else {
            return nil
        }
        let view = bucket.removeLast()
        reusableViews[key] = bucket
        return view
    }

    private func clearVisibleViews() {
        for (_, view) in visibleViews {
            view.removeFromSuperview()
        }
        visibleViews.removeAll()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        updateContentSize()
        updateVisibleViews()
    }
}

extension VirtualListView: UIScrollViewDelegate {
    func scrollViewDidScroll(_ scrollView: UIScrollView) {
        updateVisibleViews()
        onScroll?(scrollView.contentOffset)
    }
}
