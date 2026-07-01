import UIKit

final class RootViewController: UIViewController {
    private let meshBackgroundView = MeshGradientBackgroundView()
    private let floatingHomeView = FloatingHomeContainerView()
    private let safeContentGuide = UILayoutGuide()
    private let heavyIndicatorView = AispectHeavyIndicatorView()
    private let impactTestView = AispectImpactTestView()
    private let holdProgressView = AispectHoldProgressView()
    private let holdControlView = AispectHoldControlView()
    private let topBar = TopBarView()
    private let bottomBar = BottomBarView()
    private let leftSidebar = AvatarSidebarView()
    private let rightSidebar = RightSidebarView()
    private let messageList = MessageListView()
    private let unreadMessagesView = UnreadMessagesView()
    private let connectionLines = ConnectionLineView()
    private var isUnreadMode = false

    private let aispectEngine = AispectInteractionEngine()
    private lazy var aispectRouter = AispectEventRouter(engine: aispectEngine)
    private let impactDetector = AispectImpactDetector()
    private let blinkDetector = AispectBlinkDetector()
    private var aispectRecognizer: AispectGestureRecognizer?
    private let holdFeedback = UIImpactFeedbackGenerator(style: .medium)
    private let edgeFeedback = UIImpactFeedbackGenerator(style: .light)
    private var isHoldControlActive = false
    private var heavyIndicatorHideWork: DispatchWorkItem?
    private let isImpactTestEnabled = true
    private var isTouchActive = false
    private var didImpactHeavyPress = false
    private var impactWindowStart: TimeInterval = 0
    private let impactWindowDuration: TimeInterval = 0.25
    private var lastAispectEvent: AispectTouchEvent?
    private var hapticOption: AispectHapticOption = .impactRigid
    private var impactHaptic = UIImpactFeedbackGenerator(style: .rigid)
    private let notificationHaptic = UINotificationFeedbackGenerator()
    private let selectionHaptic = UISelectionFeedbackGenerator()
    private var datasetLabel = AispectImpactDatasetLabel(
        motionState: .handheldStatic,
        touchStrength: .light
    )
    private var pendingImpactStopWork: DispatchWorkItem?

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        floatingHomeView.applyStyle(.edgeToEdge)
        setupLayout()
        setupAispect()
        setupAispectRouter()
        setupTopBar()
        setupImpactDetector()
        setupBlinkDetector()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        let safeFrame = view.safeAreaLayoutGuide.layoutFrame
        let availableHeight = max(safeFrame.height - 24, 120)
        impactTestView.maximumExpandedHeight = availableHeight
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        // 中文注释：页面可见时预热闭眼检测链路，避免轻点时再冷启动相机。
        impactTestView.setBlinkPrepareLatency(nil)
        blinkDetector.prepare()
    }

    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        // 中文注释：页面离开后立即挂起相机会话，避免待机功耗持续存在。
        blinkDetector.suspend()
    }

    private func setupLayout() {
        meshBackgroundView.translatesAutoresizingMaskIntoConstraints = false
        floatingHomeView.translatesAutoresizingMaskIntoConstraints = false
        heavyIndicatorView.translatesAutoresizingMaskIntoConstraints = false
        impactTestView.translatesAutoresizingMaskIntoConstraints = false
        topBar.translatesAutoresizingMaskIntoConstraints = false
        bottomBar.translatesAutoresizingMaskIntoConstraints = false
        leftSidebar.translatesAutoresizingMaskIntoConstraints = false
        rightSidebar.translatesAutoresizingMaskIntoConstraints = false
        messageList.translatesAutoresizingMaskIntoConstraints = false
        unreadMessagesView.translatesAutoresizingMaskIntoConstraints = false
        connectionLines.translatesAutoresizingMaskIntoConstraints = false

        view.addSubview(meshBackgroundView)
        view.addSubview(floatingHomeView)
        view.addSubview(holdProgressView)
        view.addSubview(holdControlView)
        view.addSubview(heavyIndicatorView)
        view.addSubview(impactTestView)
        view.addLayoutGuide(safeContentGuide)
        let contentView = floatingHomeView.contentView
        contentView.addSubview(topBar)
        contentView.addSubview(bottomBar)
        contentView.addSubview(leftSidebar)
        contentView.addSubview(rightSidebar)
        contentView.addSubview(messageList)
        contentView.addSubview(unreadMessagesView)
        contentView.addSubview(connectionLines)
        // 中文注释：连线数据由主控制器提供。
        connectionLines.provider = self

        NSLayoutConstraint.activate([
            meshBackgroundView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            meshBackgroundView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            meshBackgroundView.topAnchor.constraint(equalTo: view.topAnchor),
            meshBackgroundView.bottomAnchor.constraint(equalTo: view.bottomAnchor),

            floatingHomeView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            floatingHomeView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            floatingHomeView.topAnchor.constraint(equalTo: view.topAnchor),
            floatingHomeView.bottomAnchor.constraint(equalTo: view.bottomAnchor),

            heavyIndicatorView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            heavyIndicatorView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            heavyIndicatorView.topAnchor.constraint(equalTo: view.topAnchor),
            heavyIndicatorView.bottomAnchor.constraint(equalTo: view.bottomAnchor),

            impactTestView.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 12),
            impactTestView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            impactTestView.bottomAnchor.constraint(lessThanOrEqualTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -12),

            safeContentGuide.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor),
            safeContentGuide.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor),
            safeContentGuide.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            safeContentGuide.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor),

            topBar.topAnchor.constraint(equalTo: safeContentGuide.topAnchor),
            topBar.leadingAnchor.constraint(equalTo: safeContentGuide.leadingAnchor),
            topBar.trailingAnchor.constraint(equalTo: safeContentGuide.trailingAnchor),
            topBar.heightAnchor.constraint(equalToConstant: 44),

            bottomBar.bottomAnchor.constraint(equalTo: safeContentGuide.bottomAnchor),
            bottomBar.leadingAnchor.constraint(equalTo: safeContentGuide.leadingAnchor),
            bottomBar.trailingAnchor.constraint(equalTo: safeContentGuide.trailingAnchor),
            bottomBar.heightAnchor.constraint(equalToConstant: 96),

            leftSidebar.topAnchor.constraint(equalTo: topBar.bottomAnchor),
            leftSidebar.leadingAnchor.constraint(equalTo: safeContentGuide.leadingAnchor),
            leftSidebar.bottomAnchor.constraint(equalTo: bottomBar.topAnchor),
            leftSidebar.widthAnchor.constraint(equalToConstant: 72),

            rightSidebar.topAnchor.constraint(equalTo: topBar.bottomAnchor),
            rightSidebar.trailingAnchor.constraint(equalTo: safeContentGuide.trailingAnchor),
            rightSidebar.bottomAnchor.constraint(equalTo: bottomBar.topAnchor),
            rightSidebar.widthAnchor.constraint(equalToConstant: 96),

            messageList.topAnchor.constraint(equalTo: topBar.bottomAnchor),
            messageList.leadingAnchor.constraint(equalTo: leftSidebar.trailingAnchor),
            messageList.trailingAnchor.constraint(equalTo: rightSidebar.leadingAnchor),
            messageList.bottomAnchor.constraint(equalTo: bottomBar.topAnchor),

            unreadMessagesView.topAnchor.constraint(equalTo: topBar.bottomAnchor),
            unreadMessagesView.leadingAnchor.constraint(equalTo: leftSidebar.trailingAnchor),
            unreadMessagesView.trailingAnchor.constraint(equalTo: rightSidebar.leadingAnchor),
            unreadMessagesView.bottomAnchor.constraint(equalTo: bottomBar.topAnchor),

            connectionLines.topAnchor.constraint(equalTo: topBar.bottomAnchor),
            connectionLines.leadingAnchor.constraint(equalTo: safeContentGuide.leadingAnchor),
            connectionLines.trailingAnchor.constraint(equalTo: safeContentGuide.trailingAnchor),
            connectionLines.bottomAnchor.constraint(equalTo: bottomBar.topAnchor)
        ])

        unreadMessagesView.isHidden = true
        holdProgressView.isHidden = true
        holdControlView.isHidden = true
        heavyIndicatorView.isHidden = true
        impactTestView.isHidden = !isImpactTestEnabled
    }

    private func setupTopBar() {
        topBar.onStatusTapped = { [weak self] in
            self?.toggleUnreadMode()
        }
    }

    private func toggleUnreadMode() {
        let shouldShowUnread = unreadMessagesView.isHidden
        isUnreadMode = shouldShowUnread
        unreadMessagesView.isHidden = !shouldShowUnread
        messageList.isHidden = shouldShowUnread
        connectionLines.isHidden = false
    }

    private func setupAispect() {
        // 中文注释：采样节流间隔用于样本更新频率控制。
        aispectEngine.minimumSampleInterval = 1.0 / 240.0
        // 中文注释：冲击阈值调节与测试面板联动。
        impactTestView.setImpactThreshold(CGFloat(impactDetector.config.impactDeltaThreshold))
        impactTestView.onImpactThresholdChanged = { [weak self] value in
            self?.impactDetector.config.impactDeltaThreshold = Double(value)
        }
        impactTestView.setHapticOption(hapticOption)
        impactTestView.onHapticOptionChanged = { [weak self] option in
            self?.hapticOption = option
            self?.triggerHaptic(option)
        }
        impactDetector.setDatasetCollectionLabel(datasetLabel)
        impactTestView.setCollectionMotionState(datasetLabel.motionState)
        impactTestView.setCollectionTouchStrength(datasetLabel.touchStrength)
        impactTestView.onCollectionEnabledChanged = { [weak self] isEnabled in
            self?.impactDetector.setDatasetCollectionEnabled(isEnabled)
            self?.impactTestView.setCollectionEnabled(isEnabled)
        }
        impactTestView.onCollectionMotionChanged = { [weak self] motionState in
            guard let self else { return }
            self.datasetLabel.motionState = motionState
            self.impactDetector.setDatasetCollectionLabel(self.datasetLabel)
        }
        impactTestView.onCollectionStrengthChanged = { [weak self] touchStrength in
            guard let self else { return }
            self.datasetLabel.touchStrength = touchStrength
            self.impactDetector.setDatasetCollectionLabel(self.datasetLabel)
        }
        impactTestView.onCopyDatasetPath = { [weak self] in
            self?.copyDatasetPath()
        }
        let recognizer = AispectGestureRecognizer(engine: aispectEngine, target: self, action: #selector(handleAispectGesture))
        recognizer.eventDelegate = self
        recognizer.cancelsTouchesInView = false
        recognizer.delaysTouchesBegan = false
        recognizer.delaysTouchesEnded = false
        recognizer.delegate = self
        view.addGestureRecognizer(recognizer)
        aispectRecognizer = recognizer
    }

    private func setupAispectRouter() {
        aispectRouter.onHoldProgress = { [weak self] progress in
            self?.showHoldProgress(progress)
        }
        aispectRouter.onHoldProgressEnd = { [weak self] in
            self?.hideHoldProgress()
        }
        aispectRouter.onHoldTriggered = { [weak self] _ in
            self?.holdFeedback.impactOccurred()
        }
        aispectRouter.onEdgeAction = { [weak self] _ in
            self?.edgeFeedback.impactOccurred()
        }
        aispectRouter.register(
            AispectEventBinding(kinds: [.heavyHold, .heavyPress], zone: .full) { [weak self] event in
                // 中文注释：重触长按时展示旋钮控件。
                self?.showHoldControl(at: event.location)
            }
        )
    }

    @objc private func handleAispectGesture() {
        // 中文注释：事件由 delegate 接管。
    }

    private func setupImpactDetector() {
        // 中文注释：冲击事件与面板显示绑定。
        impactDetector.onImpact = { [weak self] delta in
            self?.handleImpactDetected(delta: delta)
        }
        impactDetector.onSampleRate = { [weak self] hz in
            self?.impactTestView.setImpactSampleRate(hz)
        }
        impactDetector.onImpactLog = { [weak self] text in
            self?.impactTestView.setImpactLog(text)
        }
        impactDetector.onDatasetStatus = { [weak self] status in
            self?.impactTestView.setCollectionStatus(status)
        }
    }

    private func setupBlinkDetector() {
        // 中文注释：闭眼检测回调与面板联动。
        blinkDetector.onPrepareLatency = { [weak self] latency in
            self?.impactTestView.setBlinkPrepareLatency(latency)
        }
        blinkDetector.onLatency = { [weak self] latency in
            self?.impactTestView.setBlinkLatency(latency)
        }
        blinkDetector.onFaceLockLatency = { [weak self] latency in
            self?.impactTestView.setBlinkFaceLockLatency(latency)
        }
        blinkDetector.onDecisionLatency = { [weak self] latency in
            self?.impactTestView.setBlinkDecisionLatency(latency)
        }
        blinkDetector.onBlink = { [weak self] success in
            let key = success ? "blink_detected" : "blink_not_detected"
            self?.impactTestView.setBlinkStatusKey(key)
            let resultKey = success ? "blink_result_yes" : "blink_result_no"
            self?.impactTestView.setBlinkResultKey(resultKey)
        }
        blinkDetector.onDebugSnapshot = { [weak self] snapshot in
            let debugText = String(
                format: "base L%.2f R%.2f raw L%.2f R%.2f\navg %.2f eff %.2f max %.2f C%d S%d",
                snapshot.leftBaseline,
                snapshot.rightBaseline,
                snapshot.leftRawRatio,
                snapshot.rightRawRatio,
                snapshot.averageRawRatio,
                snapshot.effectiveAverageRatio,
                snapshot.effectiveMaxRatio,
                snapshot.isClosed ? 1 : 0,
                snapshot.isStronglyClosed ? 1 : 0
            )
            DispatchQueue.main.async {
                self?.impactTestView.setBlinkDebugText(debugText)
            }
        }
        blinkDetector.onStatus = { [weak self] text in
            self?.impactTestView.setBlinkStatusKey(text)
        }
    }

    private func handleImpactDetected(delta: Double) {
        guard isTouchActive else { return }
        guard !didImpactHeavyPress else { return }
        guard let lastEvent = lastAispectEvent else { return }
        let timeSinceStart = lastEvent.timestamp - impactWindowStart
        // 中文注释：仅在短触摸窗口内接受冲击判定。
        if timeSinceStart > impactWindowDuration { return }
        if lastEvent.kind == .heavyPress || lastEvent.kind == .heavyHold { return }
        didImpactHeavyPress = true
        triggerHaptic(hapticOption)
        let synthetic = AispectTouchEvent(
            kind: .heavyPress,
            location: lastEvent.location,
            timestamp: lastEvent.timestamp
        )
        aispectRouter.handle(event: synthetic, in: view)
        updateVisualEffects(for: synthetic)
        showHoldControl(at: synthetic.location)
        showHeavyIndicator(at: synthetic.location)
    }

    private func triggerHaptic(_ option: AispectHapticOption) {
        if let style = option.impactStyle {
            impactHaptic = UIImpactFeedbackGenerator(style: style)
            impactHaptic.prepare()
            impactHaptic.impactOccurred(intensity: 1.0)
            return
        }
        if let type = option.notificationType {
            notificationHaptic.prepare()
            notificationHaptic.notificationOccurred(type)
            return
        }
        if option.isSelection {
            selectionHaptic.prepare()
            selectionHaptic.selectionChanged()
        }
    }

}

extension RootViewController: UIGestureRecognizerDelegate {
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        true
    }
}

extension RootViewController: AispectGestureDelegate {
    func aispectRecognizer(_ recognizer: AispectGestureRecognizer, didEmit event: AispectTouchEvent) {
        lastAispectEvent = event
        if event.kind == .sample, !isTouchActive {
            // 中文注释：首次触摸进入冲击监听窗口。
            pendingImpactStopWork?.cancel()
            isTouchActive = true
            didImpactHeavyPress = false
            impactWindowStart = event.timestamp
            impactDetector.beginDatasetCapture(at: event.timestamp)
            impactDetector.start()
        }
        if event.kind == .heavyPress || event.kind == .heavyHold {
            didImpactHeavyPress = true
        }
        if event.kind == .lightTap || event.kind == .heavyTap {
            // 中文注释：触摸结束即停止冲击监听。
            isTouchActive = false
            let postCaptureDuration = impactDetector.acceptDatasetCapture(
                anchorTimestamp: event.timestamp,
                touchKind: String(describing: event.kind),
                liftOffset: event.liftOffset.map(Double.init)
            )
            scheduleImpactDetectorStop(after: max(postCaptureDuration, 0.08))
        }
        if event.kind == .dragEnd || event.kind == .cancel {
            isTouchActive = false
            let reasonKey = event.kind == .dragEnd ? "collector_event_rejected_drag" : "collector_event_rejected_cancel"
            impactDetector.rejectDatasetCapture(reasonKey: reasonKey)
            scheduleImpactDetectorStop(after: 0.02)
        }
        if isImpactTestEnabled {
            impactTestView.update(event: event)
        }
        let inputFrame = bottomBar.inputViewFrame(in: view)
        let isInInput = inputFrame.contains(event.location)

        switch event.kind {
        case .sample, .dragStart, .drag:
            aispectRouter.handle(event: event, in: view)
            updateVisualEffects(for: event)
            if isHoldControlActive, event.kind != .sample {
                updateHoldControl(with: event)
            }
        case .lightHold, .heavyHold, .heavyPress:
            aispectRouter.handle(event: event, in: view)
            if isInInput {
                bottomBar.setInputFloatingBarVisible(true, mode: .compose)
                bottomBar.setInputSuggestion(text: "Try rewrite or polish", style: .badge)
            }
            updateVisualEffects(for: event)
            showHoldControl(at: event.location)
            if event.kind != .lightHold {
                triggerHaptic(hapticOption)
                showHeavyIndicator(at: event.location)
            }
        case .lightTap:
            aispectRouter.handle(event: event, in: view)
            bottomBar.setInputSuggestion(text: nil, style: .badge)
            resetVisualEffects()
            // 中文注释：轻点触发闭眼检测并测量首帧延迟。
            impactTestView.setBlinkLatency(nil)
            impactTestView.setBlinkFaceLockLatency(nil)
            impactTestView.setBlinkDecisionLatency(nil)
            impactTestView.setBlinkStatusKey("blink_starting")
            impactTestView.setBlinkResultKey("blink_result_unknown")
            impactTestView.setBlinkDebugText(nil)
            blinkDetector.start()
        case .heavyTap, .dragEnd, .cancel:
            aispectRouter.handle(event: event, in: view)
            resetVisualEffects()
            hideHoldControl()
            // 中文注释：若已触发重触则避免重复触觉。
            if event.kind == .heavyTap, !didImpactHeavyPress {
                triggerHaptic(hapticOption)
                showHeavyIndicator(at: event.location)
            }
        default:
            aispectRouter.handle(event: event, in: view)
            break
        }
    }
}

extension RootViewController {
    private func scheduleImpactDetectorStop(after delay: TimeInterval) {
        pendingImpactStopWork?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            guard !self.isTouchActive else { return }
            self.impactDetector.stop()
        }
        pendingImpactStopWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + delay, execute: work)
    }

    private func copyDatasetPath() {
        // 中文注释：复制采集目录路径，方便外部导出训练数据。
        UIPasteboard.general.string = impactDetector.datasetDirectoryPath
    }

    private func updateVisualEffects(for event: AispectTouchEvent) {
        let progress = pressureValue(for: event)
        let localPoint = meshBackgroundView.convert(event.location, from: view)
        meshBackgroundView.updateTouch(point: localPoint, pressure: progress)
        floatingHomeView.updatePressure(progress)
    }

    private func pressureValue(for event: AispectTouchEvent) -> CGFloat {
        // 中文注释：按事件类型映射为视觉压力值。
        switch event.kind {
        case .heavyPress, .heavyHold:
            return 1.0
        case .heavyTap:
            return 0.9
        case .lightHold:
            return 0.6
        case .sample, .dragStart, .drag:
            return 0.25
        default:
            return 0.0
        }
    }

    private func resetVisualEffects() {
        meshBackgroundView.resetPressure()
        floatingHomeView.releasePressure()
    }

    private func showHeavyIndicator(at location: CGPoint) {
        heavyIndicatorHideWork?.cancel()
        heavyIndicatorView.isHidden = false
        heavyIndicatorView.show(at: location, in: view.bounds)
        scheduleHideHeavyIndicator()
    }

    private func scheduleHideHeavyIndicator() {
        heavyIndicatorHideWork?.cancel()
        let work = DispatchWorkItem { [weak self] in
            self?.heavyIndicatorView.hide()
            self?.heavyIndicatorView.isHidden = true
        }
        heavyIndicatorHideWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.55, execute: work)
    }

    private func showHoldProgress(_ progress: AispectHoldProgress) {
        if progress.isHeavyCandidate { return }
        let size = CGSize(width: 120, height: 4)
        let offset = CGPoint(x: 0, y: -28)
        // 中文注释：进度条位置跟随手指并上移避免遮挡。
        holdProgressView.frame = CGRect(
            x: progress.location.x - size.width * 0.5 + offset.x,
            y: progress.location.y - size.height * 0.5 + offset.y,
            width: size.width,
            height: size.height
        )
        holdProgressView.progress = progress.progress
        holdProgressView.isHidden = false
        if progress.progress < 0.1 {
            holdFeedback.prepare()
        }
    }

    private func hideHoldProgress() {
        holdProgressView.isHidden = true
    }

    private func showHoldControl(at location: CGPoint) {
        let size: CGFloat = 96
        holdControlView.frame = CGRect(
            x: location.x - size * 0.5,
            y: location.y - size * 0.5,
            width: size,
            height: size
        )
        holdControlView.isHidden = false
        holdControlView.resetKnob(animated: false)
        isHoldControlActive = true
    }

    private func hideHoldControl() {
        holdControlView.isHidden = true
        holdControlView.resetKnob(animated: true)
        isHoldControlActive = false
    }

    private func updateHoldControl(with event: AispectTouchEvent) {
        let localPoint = view.convert(event.location, to: holdControlView)
        let center = CGPoint(x: holdControlView.bounds.midX, y: holdControlView.bounds.midY)
        let dx = localPoint.x - center.x
        let dy = localPoint.y - center.y
        let angle = atan2(dy, dx)
        let radius = max(min(holdControlView.bounds.width, holdControlView.bounds.height) * 0.5 - 10, 1)
        let distance = hypot(dx, dy)
        let normalized = min(distance / radius, 1)
        holdControlView.updateKnob(angle: angle, normalizedDistance: normalized)
    }
}

extension RootViewController: ConnectionLineProvider {
    func connectionSegments() -> [ConnectionSegment] {
        var segments: [ConnectionSegment] = []

        if isUnreadMode {
            let indices = unreadMessagesView.visibleIndices(limit: 6)
            for index in indices {
                guard let avatarAnchor = unreadMessagesView.avatarAnchorPoint(at: index),
                      let bubbleAnchor = unreadMessagesView.bubbleAnchorPoint(at: index) else { continue }
                let start = unreadMessagesView.convert(avatarAnchor, to: view)
                let end = unreadMessagesView.convert(bubbleAnchor, to: view)
                segments.append(ConnectionSegment(start: start, end: end, bendsOutward: false))
            }
            return segments
        }

        if let messageAnchor = messageList.anchorPointForMessage(at: 0, isOutgoing: false),
           let avatarAnchor = leftSidebar.anchorPointForItem(at: leftSidebar.selectedIndex) {
            let start = messageList.convert(messageAnchor, to: view)
            let end = leftSidebar.convert(avatarAnchor, to: view)
            segments.append(ConnectionSegment(start: start, end: end, bendsOutward: false))
        }

        if let messageAnchor = messageList.anchorPointForMessage(at: 1, isOutgoing: true),
           let accountAnchor = rightSidebar.accountAnchorPoint(at: rightSidebar.selectedAccountIndex) {
            let start = messageList.convert(messageAnchor, to: view)
            let end = rightSidebar.convert(accountAnchor, to: view)
            segments.append(ConnectionSegment(start: start, end: end, bendsOutward: true))
        }

        return segments
    }
}
