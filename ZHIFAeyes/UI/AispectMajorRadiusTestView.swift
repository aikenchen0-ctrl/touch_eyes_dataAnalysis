import UIKit

final class AispectImpactTestView: UIView {
    private let scrollView = UIScrollView()
    private let contentView = UIView()
    private let panelToggleButton = UIButton(type: .system)
    private let titleLabel = UILabel()
    private let valueLabel = UILabel()
    private let impactLabel = UILabel()
    private let impactSlider = UISlider()
    private let impactRateLabel = UILabel()
    private let impactLogLabel = UILabel()
    private let copyLogButton = UIButton(type: .system)
    private let blinkPrepareLatencyLabel = UILabel()
    private let blinkLatencyLabel = UILabel()
    private let blinkFaceLockLatencyLabel = UILabel()
    private let blinkDecisionLatencyLabel = UILabel()
    private let blinkStatusLabel = UILabel()
    private let blinkResultLabel = UILabel()
    private let blinkDebugLabel = UILabel()
    private let hapticLabel = UILabel()
    private let hapticPicker = UIPickerView()
    private let collectorTitleLabel = UILabel()
    private let collectorSwitch = UISwitch()
    private let collectorMotionControl = UISegmentedControl(items: ["handheld", "placed", "walking"])
    private let collectorStrengthControl = UISegmentedControl(items: ["light", "heavy"])
    private let collectorGuideLabel = UILabel()
    private let collectorStatusLabel = UILabel()
    private let copyDatasetPathButton = UIButton(type: .system)

    private let padding = UIEdgeInsets(top: 10, left: 12, bottom: 10, right: 12)
    private let collapsedSize = CGSize(width: 60, height: 34)
    private let titleHeight: CGFloat = 16
    private let valueHeight: CGFloat = 44
    private let impactLabelHeight: CGFloat = 16
    private let sliderHeight: CGFloat = 24
    private let impactRateHeight: CGFloat = 14
    private let impactLogHeight: CGFloat = 240
    private let copyButtonHeight: CGFloat = 24
    private let blinkLabelHeight: CGFloat = 14
    private let blinkDebugHeight: CGFloat = 34
    private let hapticLabelHeight: CGFloat = 16
    private let pickerHeight: CGFloat = 80
    private let collectorTitleHeight: CGFloat = 18
    private let collectorSwitchHeight: CGFloat = 28
    private let collectorSegmentHeight: CGFloat = 28
    private let collectorGuideHeight: CGFloat = 54
    private let collectorStatusHeight: CGFloat = 108
    private var isCollapsed = false
    var maximumExpandedHeight: CGFloat = .greatestFiniteMagnitude {
        didSet {
            guard oldValue != maximumExpandedHeight else { return }
            invalidateIntrinsicContentSize()
            setNeedsLayout()
        }
    }

    var onImpactThresholdChanged: ((CGFloat) -> Void)?
    var onHapticOptionChanged: ((AispectHapticOption) -> Void)?
    var onCollectionEnabledChanged: ((Bool) -> Void)?
    var onCollectionMotionChanged: ((AispectImpactMotionState) -> Void)?
    var onCollectionStrengthChanged: ((AispectImpactTouchStrength) -> Void)?
    var onCopyDatasetPath: (() -> Void)?
    private var hapticOptions: [AispectHapticOption] = AispectHapticOption.allCases

    override init(frame: CGRect) {
        super.init(frame: frame)
        setupView()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupView()
    }

    private func setupView() {
        isUserInteractionEnabled = true
        backgroundColor = UIColor.black.withAlphaComponent(0.6)
        layer.cornerRadius = 10
        layer.masksToBounds = true

        scrollView.alwaysBounceVertical = true
        scrollView.showsVerticalScrollIndicator = true
        scrollView.delaysContentTouches = false
        scrollView.canCancelContentTouches = true

        contentView.backgroundColor = .clear

        panelToggleButton.setTitle(localized("impact_panel_collapse"), for: .normal)
        panelToggleButton.setTitleColor(.white, for: .normal)
        panelToggleButton.backgroundColor = UIColor.systemBlue.withAlphaComponent(0.85)
        panelToggleButton.layer.cornerRadius = 8
        panelToggleButton.titleLabel?.font = UIFont.systemFont(ofSize: 11, weight: .semibold)
        panelToggleButton.addTarget(self, action: #selector(handlePanelToggle), for: .touchUpInside)

        titleLabel.font = UIFont.systemFont(ofSize: 12, weight: .semibold)
        titleLabel.textColor = UIColor.white.withAlphaComponent(0.9)
        titleLabel.text = localized("impact_panel_title")

        valueLabel.font = UIFont.monospacedDigitSystemFont(ofSize: 12, weight: .regular)
        valueLabel.textColor = UIColor.white
        valueLabel.numberOfLines = 3

        impactLabel.font = UIFont.systemFont(ofSize: 11, weight: .semibold)
        impactLabel.textColor = UIColor.white.withAlphaComponent(0.9)
        impactLabel.text = localized("impact_threshold_title")

        impactSlider.minimumValue = 0.05
        impactSlider.maximumValue = 1.0
        impactSlider.value = 0.25
        impactSlider.isEnabled = false
        impactSlider.isContinuous = true
        // 中文注释：滑动即刷新阈值，便于现场调参。
        impactSlider.addTarget(self, action: #selector(handleImpactSlider(_:)), for: .valueChanged)

        impactRateLabel.font = UIFont.systemFont(ofSize: 10, weight: .regular)
        impactRateLabel.textColor = UIColor.white.withAlphaComponent(0.7)
        impactRateLabel.text = localized("impact_rate_idle")

        impactLogLabel.font = UIFont.monospacedDigitSystemFont(ofSize: 9, weight: .regular)
        impactLogLabel.textColor = UIColor.white.withAlphaComponent(0.75)
        impactLogLabel.numberOfLines = 0
        impactLogLabel.text = localized("impact_log_idle")

        copyLogButton.setTitle(localized("impact_copy_log"), for: .normal)
        copyLogButton.setTitleColor(UIColor.white, for: .normal)
        copyLogButton.backgroundColor = UIColor.white.withAlphaComponent(0.12)
        copyLogButton.layer.cornerRadius = 6
        // 中文注释：复制当前冲击日志文本。
        copyLogButton.addTarget(self, action: #selector(handleCopyLog), for: .touchUpInside)

        // 中文注释：闭眼检测首帧延迟与状态显示。
        blinkPrepareLatencyLabel.font = UIFont.systemFont(ofSize: 10, weight: .regular)
        blinkPrepareLatencyLabel.textColor = UIColor.white.withAlphaComponent(0.7)
        blinkPrepareLatencyLabel.text = blinkLocalized("blink_prepare_latency_idle")

        blinkLatencyLabel.font = UIFont.systemFont(ofSize: 10, weight: .regular)
        blinkLatencyLabel.textColor = UIColor.white.withAlphaComponent(0.7)
        blinkLatencyLabel.text = blinkLocalized("blink_first_frame_latency_idle")

        blinkFaceLockLatencyLabel.font = UIFont.systemFont(ofSize: 10, weight: .regular)
        blinkFaceLockLatencyLabel.textColor = UIColor.white.withAlphaComponent(0.7)
        blinkFaceLockLatencyLabel.text = blinkLocalized("blink_face_lock_latency_idle")

        blinkDecisionLatencyLabel.font = UIFont.systemFont(ofSize: 10, weight: .regular)
        blinkDecisionLatencyLabel.textColor = UIColor.white.withAlphaComponent(0.7)
        blinkDecisionLatencyLabel.text = blinkLocalized("blink_decision_latency_idle")

        blinkStatusLabel.font = UIFont.systemFont(ofSize: 10, weight: .regular)
        blinkStatusLabel.textColor = UIColor.white.withAlphaComponent(0.7)
        blinkStatusLabel.text = blinkLocalized("blink_state_idle")

        blinkResultLabel.font = UIFont.systemFont(ofSize: 10, weight: .regular)
        blinkResultLabel.textColor = UIColor.white.withAlphaComponent(0.7)
        blinkResultLabel.text = blinkLocalized("blink_result_unknown")

        blinkDebugLabel.font = UIFont.monospacedDigitSystemFont(ofSize: 9, weight: .regular)
        blinkDebugLabel.textColor = UIColor.white.withAlphaComponent(0.68)
        blinkDebugLabel.numberOfLines = 2
        blinkDebugLabel.text = "blink debug --"

        hapticLabel.font = UIFont.systemFont(ofSize: 11, weight: .semibold)
        hapticLabel.textColor = UIColor.white.withAlphaComponent(0.9)
        hapticLabel.text = localized("impact_haptic_title")

        hapticPicker.dataSource = self
        hapticPicker.delegate = self
        hapticPicker.backgroundColor = UIColor.white.withAlphaComponent(0.05)

        collectorTitleLabel.font = UIFont.systemFont(ofSize: 11, weight: .semibold)
        collectorTitleLabel.textColor = UIColor.white.withAlphaComponent(0.9)
        collectorTitleLabel.text = localized("collector_title")

        collectorSwitch.onTintColor = UIColor.systemGreen
        collectorSwitch.addTarget(self, action: #selector(handleCollectorSwitch(_:)), for: .valueChanged)

        collectorMotionControl.selectedSegmentIndex = 0
        collectorMotionControl.setTitle(localized("collector_motion_handheld_static"), forSegmentAt: 0)
        collectorMotionControl.setTitle(localized("collector_motion_placed_static"), forSegmentAt: 1)
        collectorMotionControl.setTitle(localized("collector_motion_walking"), forSegmentAt: 2)
        collectorMotionControl.addTarget(self, action: #selector(handleCollectorMotionChanged(_:)), for: .valueChanged)

        collectorStrengthControl.selectedSegmentIndex = 0
        collectorStrengthControl.setTitle(localized("collector_strength_light"), forSegmentAt: 0)
        collectorStrengthControl.setTitle(localized("collector_strength_heavy"), forSegmentAt: 1)
        collectorStrengthControl.addTarget(self, action: #selector(handleCollectorStrengthChanged(_:)), for: .valueChanged)

        collectorGuideLabel.font = UIFont.systemFont(ofSize: 10, weight: .regular)
        collectorGuideLabel.textColor = UIColor.white.withAlphaComponent(0.78)
        collectorGuideLabel.numberOfLines = 0
        collectorGuideLabel.text = localized("collector_guide_text")

        collectorStatusLabel.font = UIFont.monospacedDigitSystemFont(ofSize: 9, weight: .regular)
        collectorStatusLabel.textColor = UIColor.white.withAlphaComponent(0.75)
        collectorStatusLabel.numberOfLines = 0
        collectorStatusLabel.text = localized("collector_status_idle")

        copyDatasetPathButton.setTitle(localized("collector_copy_path"), for: .normal)
        copyDatasetPathButton.setTitleColor(UIColor.white, for: .normal)
        copyDatasetPathButton.backgroundColor = UIColor.white.withAlphaComponent(0.12)
        copyDatasetPathButton.layer.cornerRadius = 6
        copyDatasetPathButton.addTarget(self, action: #selector(handleCopyDatasetPath), for: .touchUpInside)

        addSubview(scrollView)
        addSubview(panelToggleButton)
        scrollView.addSubview(contentView)
        contentView.addSubview(titleLabel)
        contentView.addSubview(valueLabel)
        contentView.addSubview(impactLabel)
        contentView.addSubview(impactSlider)
        contentView.addSubview(impactRateLabel)
        contentView.addSubview(impactLogLabel)
        contentView.addSubview(copyLogButton)
        contentView.addSubview(blinkPrepareLatencyLabel)
        contentView.addSubview(blinkLatencyLabel)
        contentView.addSubview(blinkFaceLockLatencyLabel)
        contentView.addSubview(blinkDecisionLatencyLabel)
        contentView.addSubview(blinkStatusLabel)
        contentView.addSubview(blinkResultLabel)
        contentView.addSubview(blinkDebugLabel)
        contentView.addSubview(hapticLabel)
        contentView.addSubview(hapticPicker)
        contentView.addSubview(collectorTitleLabel)
        contentView.addSubview(collectorSwitch)
        contentView.addSubview(collectorMotionControl)
        contentView.addSubview(collectorStrengthControl)
        contentView.addSubview(collectorGuideLabel)
        contentView.addSubview(collectorStatusLabel)
        contentView.addSubview(copyDatasetPathButton)
    }

    override var intrinsicContentSize: CGSize {
        if isCollapsed {
            return collapsedSize
        }
        let contentWidth = padding.left + 180 + padding.right
        let totalHeight = padding.top
            + titleHeight + 6
            + valueHeight + 8
            + impactLabelHeight + 6
            + sliderHeight + 4
            + impactRateHeight + 6
            + impactLogHeight + 6
            + copyButtonHeight + 8
            + blinkLabelHeight + 4
            + blinkLabelHeight + 4
            + blinkLabelHeight + 4
            + blinkLabelHeight + 4
            + blinkLabelHeight + 4
            + blinkLabelHeight + 8
            + blinkDebugHeight + 8
            + hapticLabelHeight + 6
            + pickerHeight + 10
            + collectorTitleHeight + 6
            + collectorSwitchHeight + 6
            + collectorSegmentHeight + 6
            + collectorSegmentHeight + 6
            + collectorGuideHeight + 6
            + collectorStatusHeight + 6
            + copyButtonHeight
            + padding.bottom
        return CGSize(width: contentWidth, height: min(totalHeight, maximumExpandedHeight))
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        if isCollapsed {
            scrollView.isHidden = true
            panelToggleButton.frame = bounds
            return
        }

        scrollView.isHidden = false
        let toggleWidth: CGFloat = 56
        let toggleHeight: CGFloat = 28
        panelToggleButton.frame = CGRect(
            x: bounds.width - padding.right - toggleWidth,
            y: padding.top,
            width: toggleWidth,
            height: toggleHeight
        )

        let topInset = padding.top + toggleHeight + 6
        scrollView.frame = bounds.inset(by: UIEdgeInsets(top: topInset, left: 0, bottom: 0, right: 0))
        let contentWidth = bounds.width
        var y = padding.top
        let width = contentWidth - padding.left - padding.right
        titleLabel.frame = CGRect(x: padding.left, y: y, width: width, height: titleHeight)
        y += titleHeight + 6
        valueLabel.frame = CGRect(x: padding.left, y: y, width: width, height: valueHeight)
        y += valueHeight + 8
        impactLabel.frame = CGRect(x: padding.left, y: y, width: width, height: impactLabelHeight)
        y += impactLabelHeight + 6
        impactSlider.frame = CGRect(x: padding.left, y: y, width: width, height: sliderHeight)
        y += sliderHeight + 4
        impactRateLabel.frame = CGRect(x: padding.left, y: y, width: width, height: impactRateHeight)
        y += impactRateHeight + 6
        impactLogLabel.frame = CGRect(x: padding.left, y: y, width: width, height: impactLogHeight)
        y += impactLogHeight + 6
        copyLogButton.frame = CGRect(x: padding.left, y: y, width: width, height: copyButtonHeight)
        y += copyButtonHeight + 8
        blinkPrepareLatencyLabel.frame = CGRect(x: padding.left, y: y, width: width, height: blinkLabelHeight)
        y += blinkLabelHeight + 4
        blinkLatencyLabel.frame = CGRect(x: padding.left, y: y, width: width, height: blinkLabelHeight)
        y += blinkLabelHeight + 4
        blinkFaceLockLatencyLabel.frame = CGRect(x: padding.left, y: y, width: width, height: blinkLabelHeight)
        y += blinkLabelHeight + 4
        blinkDecisionLatencyLabel.frame = CGRect(x: padding.left, y: y, width: width, height: blinkLabelHeight)
        y += blinkLabelHeight + 4
        blinkStatusLabel.frame = CGRect(x: padding.left, y: y, width: width, height: blinkLabelHeight)
        y += blinkLabelHeight + 4
        blinkResultLabel.frame = CGRect(x: padding.left, y: y, width: width, height: blinkLabelHeight)
        y += blinkLabelHeight + 8
        blinkDebugLabel.frame = CGRect(x: padding.left, y: y, width: width, height: blinkDebugHeight)
        y += blinkDebugHeight + 8
        hapticLabel.frame = CGRect(x: padding.left, y: y, width: width, height: hapticLabelHeight)
        y += hapticLabelHeight + 6
        hapticPicker.frame = CGRect(x: padding.left, y: y, width: width, height: pickerHeight)
        y += pickerHeight + 10
        collectorTitleLabel.frame = CGRect(x: padding.left, y: y, width: width, height: collectorTitleHeight)
        y += collectorTitleHeight + 6
        collectorSwitch.frame = CGRect(x: padding.left, y: y, width: 52, height: collectorSwitchHeight)
        y += collectorSwitchHeight + 6
        collectorMotionControl.frame = CGRect(x: padding.left, y: y, width: width, height: collectorSegmentHeight)
        y += collectorSegmentHeight + 6
        collectorStrengthControl.frame = CGRect(x: padding.left, y: y, width: width, height: collectorSegmentHeight)
        y += collectorSegmentHeight + 6
        collectorGuideLabel.frame = CGRect(x: padding.left, y: y, width: width, height: collectorGuideHeight)
        y += collectorGuideHeight + 6
        collectorStatusLabel.frame = CGRect(x: padding.left, y: y, width: width, height: collectorStatusHeight)
        y += collectorStatusHeight + 6
        copyDatasetPathButton.frame = CGRect(x: padding.left, y: y, width: width, height: copyButtonHeight)
        y += copyButtonHeight + padding.bottom

        contentView.frame = CGRect(x: 0, y: 0, width: contentWidth, height: y)
        scrollView.contentSize = CGSize(width: contentWidth, height: y)
    }

    func update(event: AispectTouchEvent) {
        let liftText: String
        if let lift = event.liftOffset {
            liftText = String(format: "%.2f", lift)
        } else {
            liftText = "--"
        }
        let format = localized("impact_event_format")
        valueLabel.text = String(
            format: format,
            localized(eventLocalizationKey(for: event.kind)),
            event.location.x,
            event.location.y,
            liftText,
            event.timestamp
        )
        impactLabel.text = localized("impact_threshold_format")
    }

    func setImpactThreshold(_ value: CGFloat) {
        impactLabel.text = localized("impact_threshold_format")
    }

    func setImpactSampleRate(_ hz: Double) {
        if hz <= 0 {
            impactRateLabel.text = localized("impact_rate_idle")
        } else {
            let format = localized("impact_rate_format")
            impactRateLabel.text = String(format: format, hz)
        }
    }

    func setImpactLog(_ text: String) {
        impactLogLabel.text = text
    }

    func setBlinkPrepareLatency(_ value: TimeInterval?) {
        blinkPrepareLatencyLabel.text = blinkMetricText(
            value,
            idleKey: "blink_prepare_latency_idle",
            formatKey: "blink_prepare_latency_format"
        )
    }

    func setBlinkLatency(_ value: TimeInterval?) {
        blinkLatencyLabel.text = blinkMetricText(
            value,
            idleKey: "blink_first_frame_latency_idle",
            formatKey: "blink_first_frame_latency_format"
        )
    }

    func setBlinkFaceLockLatency(_ value: TimeInterval?) {
        blinkFaceLockLatencyLabel.text = blinkMetricText(
            value,
            idleKey: "blink_face_lock_latency_idle",
            formatKey: "blink_face_lock_latency_format"
        )
    }

    func setBlinkDecisionLatency(_ value: TimeInterval?) {
        blinkDecisionLatencyLabel.text = blinkMetricText(
            value,
            idleKey: "blink_decision_latency_idle",
            formatKey: "blink_decision_latency_format"
        )
    }

    func setBlinkStatusKey(_ key: String) {
        blinkStatusLabel.text = blinkLocalized(key)
    }

    func setBlinkResultKey(_ key: String) {
        blinkResultLabel.text = blinkLocalized(key)
    }

    func setBlinkDebugText(_ text: String?) {
        blinkDebugLabel.text = text ?? "blink debug --"
    }

    func setHapticOption(_ option: AispectHapticOption) {
        if let index = hapticOptions.firstIndex(of: option) {
            hapticPicker.selectRow(index, inComponent: 0, animated: false)
        }
    }

    func setCollectionEnabled(_ isEnabled: Bool) {
        collectorSwitch.isOn = isEnabled
        collectorMotionControl.isEnabled = !isEnabled
        collectorStrengthControl.isEnabled = !isEnabled
    }

    func setCollectionMotionState(_ motionState: AispectImpactMotionState) {
        switch motionState {
        case .handheldStatic:
            collectorMotionControl.selectedSegmentIndex = 0
        case .placedStatic:
            collectorMotionControl.selectedSegmentIndex = 1
        case .walking:
            collectorMotionControl.selectedSegmentIndex = 2
        }
    }

    func setCollectionTouchStrength(_ touchStrength: AispectImpactTouchStrength) {
        collectorStrengthControl.selectedSegmentIndex = touchStrength == .light ? 0 : 1
    }

    func setCollectionStatus(_ status: AispectImpactDatasetCollector.Status) {
        collectorStatusLabel.text = collectionStatusText(status)
        collectorGuideLabel.text = status.isLabelLocked
            ? localized("collector_guide_locked")
            : localized("collector_guide_text")
    }

    @objc private func handleImpactSlider(_ sender: UISlider) {
        impactLabel.text = localized("impact_threshold_format")
    }

    @objc private func handleCopyLog() {
        // 中文注释：复制完整冲击日志用于外部分析。
        UIPasteboard.general.string = impactLogLabel.text ?? ""
    }

    @objc private func handleCollectorSwitch(_ sender: UISwitch) {
        onCollectionEnabledChanged?(sender.isOn)
    }

    @objc private func handleCollectorMotionChanged(_ sender: UISegmentedControl) {
        let motionState: AispectImpactMotionState
        switch sender.selectedSegmentIndex {
        case 0:
            motionState = .handheldStatic
        case 1:
            motionState = .placedStatic
        default:
            motionState = .walking
        }
        onCollectionMotionChanged?(motionState)
    }

    @objc private func handleCollectorStrengthChanged(_ sender: UISegmentedControl) {
        let touchStrength: AispectImpactTouchStrength = sender.selectedSegmentIndex == 0 ? .light : .heavy
        onCollectionStrengthChanged?(touchStrength)
    }

    @objc private func handleCopyDatasetPath() {
        onCopyDatasetPath?()
    }

    @objc private func handlePanelToggle() {
        isCollapsed.toggle()
        backgroundColor = isCollapsed ? UIColor.black.withAlphaComponent(0.2) : UIColor.black.withAlphaComponent(0.6)
        layer.cornerRadius = isCollapsed ? 17 : 10
        panelToggleButton.setTitle(
            localized(isCollapsed ? "impact_panel_expand" : "impact_panel_collapse"),
            for: .normal
        )
        invalidateIntrinsicContentSize()
        setNeedsLayout()
        superview?.setNeedsLayout()
        superview?.layoutIfNeeded()
    }

    // 中文注释：统一从中文资源包读取面板文案。
    private func localized(_ key: String) -> String {
        if let path = Bundle.main.path(forResource: "zh-Hans", ofType: "lproj"),
           let bundle = Bundle(path: path) {
            let text = bundle.localizedString(forKey: key, value: key, table: nil)
            if text != key {
                return text
            }
        }
        return NSLocalizedString(key, comment: "")
    }

    private func blinkLocalized(_ key: String) -> String {
        localized(key)
    }

    private func blinkMetricText(_ value: TimeInterval?, idleKey: String, formatKey: String) -> String {
        if let value {
            let format = blinkLocalized(formatKey)
            return String(format: format, value * 1000)
        }
        return blinkLocalized(idleKey)
    }

    private func collectionStatusText(_ status: AispectImpactDatasetCollector.Status) -> String {
        let stateKey = status.isEnabled ? "collector_state_on" : "collector_state_off"
        let stateText = localized(stateKey)
        let taskFormat = localized("collector_task_format")
        let taskText = String(
            format: taskFormat,
            localized(motionStateLocalizationKey(status.selectedLabel.motionState)),
            localized(status.selectedLabel.touchStrength == .light ? "collector_strength_light" : "collector_strength_heavy")
        )
        let progressFormat = localized("collector_progress_format")
        let progressText = String(format: progressFormat, status.acceptedCount, status.targetCount, status.rejectedCount)
        let totalFormat = localized("collector_total_format")
        let totalText = String(format: totalFormat, status.totalCount)
        let activityText = collectorActivityText(status.latestActivity)
        let eventText = collectorEventText(status)
        return [stateText, taskText, progressText, totalText, activityText, eventText].joined(separator: "\n")
    }

    private func collectorEventText(_ status: AispectImpactDatasetCollector.Status) -> String {
        let base = localized(status.lastEventKey)
        guard let detail = status.lastEventDetail, !detail.isEmpty else {
            return base
        }
        return "\(base)：\(detail)"
    }

    private func collectorActivityText(_ activity: AispectImpactSystemActivityEvidence?) -> String {
        guard let activity else {
            return localized("collector_activity_unavailable")
        }
        let format = localized("collector_activity_format")
        return String(
            format: format,
            localized(activityStateKey(activity)),
            localized(activityConfidenceKey(activity.confidence))
        )
    }

    private func activityStateKey(_ activity: AispectImpactSystemActivityEvidence) -> String {
        switch activity.inferredMotionState {
        case "walking":
            return "collector_motion_walking"
        case "static":
            return "collector_motion_handheld_static"
        default:
            return "collector_activity_unknown"
        }
    }

    private func motionStateLocalizationKey(_ motionState: AispectImpactMotionState) -> String {
        switch motionState {
        case .handheldStatic:
            return "collector_motion_handheld_static"
        case .placedStatic:
            return "collector_motion_placed_static"
        case .walking:
            return "collector_motion_walking"
        }
    }

    private func activityConfidenceKey(_ confidence: String) -> String {
        switch confidence {
        case "low":
            return "collector_confidence_low"
        case "medium":
            return "collector_confidence_medium"
        case "high":
            return "collector_confidence_high"
        default:
            return "collector_confidence_unknown"
        }
    }

    private func eventLocalizationKey(for kind: AispectTouchKind) -> String {
        switch kind {
        case .sample:
            return "impact_event_sample"
        case .lightTap:
            return "impact_event_light_tap"
        case .heavyTap:
            return "impact_event_heavy_tap"
        case .lightHold:
            return "impact_event_light_hold"
        case .heavyHold:
            return "impact_event_heavy_hold"
        case .heavyPress:
            return "impact_event_heavy_press"
        case .dragStart:
            return "impact_event_drag_start"
        case .drag:
            return "impact_event_drag"
        case .dragEnd:
            return "impact_event_drag_end"
        case .cancel:
            return "impact_event_cancel"
        }
    }
}

extension AispectImpactTestView: UIPickerViewDataSource, UIPickerViewDelegate {
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        1
    }

    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        hapticOptions.count
    }

    func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
        hapticOptions[row].displayName
    }

    func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        onHapticOptionChanged?(hapticOptions[row])
    }
}
