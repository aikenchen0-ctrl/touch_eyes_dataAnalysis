# 技术说明渐进式生效披露关闭手电筒轻量虚拟列表，仅监听滚动并按需复用可视项，用于侧栏大规模头像/流程列表。

## 闭眼检测补充
- `AispectBlinkDetector`：闭眼检测改为 `Vision + AVCaptureSession` 主链，优先使用前置 RGB 相机在 `640x480` 低分辨率下做实时人脸 landmarks 与眼睑开合比计算，仅在 Vision 会话不可用时退回 `ARKit`，显著降低启动负担与持续功耗。
- `AispectBlinkDetector`：Vision 分支使用前置串行视频输出、丢弃积压帧、ROI 扩张后的人脸 landmarks 检测与轻量目标跟踪，首帧仍输出延迟指标，便于继续对比启动耗时。
- `AispectBlinkDetector`：眨眼判定改为“眼睑开合比基线 + 平滑 + 持续时长阈值”的有限状态机，不再依赖 TrueDepth 才能得到的 `eyeBlinkLeft/right`，同时保留 ARKit `blendShapes` 作为兜底路径。
- `AispectBlinkDetector.Config`：新增 Vision 归一化闭眼阈值、最少开眼样本数、目标帧率与跟踪置信度阈值，允许在不同 App 中按场景低成本复用同一检测器。
- `AispectBlinkDetector`：新增 `prepare()` 与 `suspend()` 生命周期接口，页面可见时提前预热前置相机会话并进入低帧率待机，用户真正触发检测时仅切换到高帧率检测窗口，检测结束后自动回到待机态，而不是每次重新冷启动整条链路。
- `AispectBlinkDetector.Config`：新增 `idleVisionFrameRate`，用于把页面常驻阶段的 Vision 待机帧率压到低功耗档位。
- `AispectBlinkDetector`：新增 `onPrepareLatency`、`onFaceLockLatency`、`onDecisionLatency` 指标回调，分别记录页面预热完成、第一次拿到有效眼部特征、以及最终给出眨眼判定的时延，便于区分瓶颈到底在相机预热、锁脸还是判定窗口。
- `AispectBlinkDetector`：根据真机观测结果，将 Vision 检测窗口从过于保守的“重平滑 + 180ms 保持”改为“20fps 检测帧率 + 80ms 闭眼保持 + 回滞阈值”的更敏捷判定策略，避免正常或轻微刻意闭眼还未穿过阈值就直接超时失败。
- `AispectBlinkDetector`：Vision 判定改为优先使用“原始归一化开眼比与轻平滑结果的较小值”作为闭眼依据，并仅在重新明显睁眼时更新开眼基线，减少平滑滞后和闭眼期间基线被拖低的问题。
- `AispectBlinkDetector.Config`：继续把检测窗口往低延迟方向推进，默认改为 `30fps` 检测帧率、`2` 个开眼样本即可建立基线、`60ms` 常规闭眼保持，并新增 `visionStrongClosedEyeRatioThreshold` 与 `visionStrongBlinkHoldDuration`，让“明显双眼闭合”可以走更短的快速命中路径。
- `AispectBlinkDetector`：缓存 Vision 图像方向并在 `start/prepare` 等低频入口刷新，移除后台视频队列每帧同步到主线程读取界面方向的开销，减少高频帧处理阻塞。
- `AispectBlinkDetector`：检测态帧率切换改为自动协商到设备当前格式最接近的受支持值，不再要求“精确等于 30fps”才允许升频，避免部分机型因只暴露 `29.97fps` 等区间而停留在低帧率待机档。
- `AispectBlinkDetector`：`face lock` 延迟改为在第一次拿到有效眼部特征时立刻记账，不再等开眼基线完全稳定后才上报，便于更准确地区分“锁脸”和“判定窗口”两类瓶颈。
- `AispectBlinkDetector`：视频帧处理改为包在 `autoreleasepool` 内执行，降低高频 Vision 请求带来的临时对象堆积风险，尽量维持检测窗口尾延迟稳定。
- `AispectBlinkDetector.Config` / `handleVisionObservation`：关键阈值与保持时长旁边补充了中文调参注释，直接说明“调大/调小会怎样”，方便在 Xcode 里围绕误判偏高或判定偏慢做手工微调。
- `AispectBlinkDetector.Config`：新增 `usesVisionStrongBlinkFastPath` 与 `visionRawRatioWeight`，排查阶段默认关闭强闭眼快速通道，并把原始比值与平滑比值改为中性加权融合，避免 `min(raw, smoothed)` 系统性偏向“更闭眼”。
- `AispectBlinkDetector.DebugSnapshot`：新增闭眼调试快照回调，实时暴露基线、原始比值、融合比值以及 `isClosed/isStronglyClosed` 状态，便于区分“阈值太松”和“融合/基线逻辑有偏”。
- `AispectImpactTestView` / `RootViewController`：测试面板新增 `blink debug` 两行调试文本，真机上可直接看到 `base/raw/eff/max` 关键值，而不必每次盲调后只看“是/否”。
- `AispectBlinkDetector`：预热期基线新增异常值过滤、左右眼对称性校验、坏基线自恢复与更快的向下回收策略，避免一次异常 landmarks 帧把 `left/right eye baseline` 抬到明显失真的高值，进而导致后续“不闭眼也判成是”。
- `AispectBlinkDetector`：眼部开合度计算从“眼轮廓包围盒高宽比”替换为基于眼轮廓点顺序重建上下眼睑后的点距 `EAR`，用两组上下点距对左右眼角宽度归一化，降低单个离群 landmarks 点、ROI 漂移和包围盒异常扩张对闭眼判定的污染。
- `AispectBlinkDetector`：修复预热态的重大逻辑缺陷，`prepare()` 期间只保活前置相机与人脸链路，不再在 `idle` 态累计可复用闭眼基线；`start()` 切入正式检测时会清空旧基线并按当前用户这轮样本重建，避免“只要预热阶段有人脸路过，后续就整轮判定漂移”。
- `AispectImpactTestView`：测试面板新增闭眼检测延迟、状态与结果显示。
- `AispectImpactTestView`：闭眼调试区扩展为“预热延迟 / 首帧延迟 / 锁脸延迟 / 判定延迟 / 状态 / 结果”六行诊断信息，方便真机上直接对比预热前后收益。
- `Localizable.strings`：补充相机权限被拒绝时的闭眼检测状态文案。
- `Localizable.strings`：新增预热、首帧、锁脸、判定四类延迟的中文格式化文案。
- `ZHIFAeyes.xcodeproj`：相机权限说明文案更新为“前置相机用于闭眼检测”，与当前 Vision 主链实现保持一致。
- `RootViewController`：在页面 `viewDidAppear` 时预热闭眼检测器，在 `viewDidDisappear` 时挂起相机会话，避免离页后仍持续占用功耗。

## 触摸结束偏移补充
- `AispectTouchEvent.liftOffset`：记录触摸结束时的抬起偏移距离，用于在未命中冲击时辅助判定 heavyTap，并在测试面板展示。
- `AispectImpactDetector`：impact delta window 记录每帧 delta 与 x/y/z/magnitude，便于观察冲击细节。
- `AispectImpactTestView`：测试面板新增 copy log 按钮，可直接复制 delta window 文本。
- `MessageListLayoutSnapshot` / `VirtualListLayoutSnapshot`：用于根据滚动偏移与固定度量计算元素位置，供连线系统使用。
- `FloatingHomeContainerView`：悬浮式主页容器，支持 iOS 风格圆角、边框与毛玻璃材质；当作为全屏根容器使用时可切换到 `edgeToEdge` 样式，关闭圆角、边框与阴影，避免根层视觉缩边。
- `RootViewController` / `safeContentGuide`：主页主容器铺满整个屏幕，根控制器底色统一为黑色；主内容依赖独立的安全区布局 guide，而不是依赖装饰性容器自身的安全区，符合 iPhone 全屏背景与内容避让分层的常见做法。
- `ZHIFAeyesApp`：SwiftUI 入口使用全屏黑色底板包裹 `ContentView`，用于兜住 UIKit 宿主层尚未完成布局时的背景色，避免安全区先露出白底。
- `GlassMaterialView`：统一玻璃材质背景，使用系统语义化模糊、细边框与轻阴影，避免手动透明度渲染。
- `MeshGradientBackgroundView`：SwiftUI 承载 MeshGradient 的背景视图；当前背景层已统一改为纯黑底，低版本兜底层与 iOS 18+ 的 MeshGradient 颜色均收敛为黑色，便于作为全屏暗色背景基底。
- `MeshGradientDriver`：MeshGradient 驱动对象，接收触摸位置与压力并联动网格顶点。
- `AispectHoldProgressView`：长按进度条视图，轻量渲染用于提示长按触发进度。
- `AispectHoldControlView`：重触长按的控制视图，支持旋钮位置更新与回弹重置。
- `AispectHeavyIndicatorView`：重触命中提示视图，用高亮环与文字提示当前重触状态。

## 历史 CNN 冲击分类补充（已停用）
- `AispectImpactCNNClassifier`：历史版本曾读取 App Bundle 权重，对抬起中心的 `-5...+15` 21 帧窗口执行 heavy/light 二分类；旧权重和 scaler 仍保留用于历史复现，不参与 schema v5 新标准结果。
- `AispectImpactFrame`：历史版本曾包含 `delta/x/y/z` 特征；新标准只写真实传感器通道，不再计算或落盘 `delta`。
- `AispectImpactPrediction`：历史输出结构保留用于兼容旧资源，新按下锚点模型尚未训练时不产生预测。
- `AispectImpactDetector`：历史 delta 阈值候选触发和旧 21 帧 CNN 已移出运行路径，当前只等待按下锚点 `-5...+9` 真实窗口并报告模型待训练。
- `AispectImpactCNNWeights.json` / `AispectImpactCNNScaler.json`：旧模型资源继续保留，不依赖 PyTorch 或 Core ML 运行时加载新标准模型；新模型待 schema v5 数据重新训练后再接入。

## 训练数据采集补充
- `AispectImpactDatasetCollector`：日常采集模块，支持手动选择 `handheld_static/placed_static/walking` 与 `light/heavy` 组合标签，并在按下/抬起双锚点原始窗口结束后保存可变长度真实序列；固定训练窗口暂定为按下 `-5...+9`，待新数据训练。
- `AispectImpactMotionState` / `AispectImpactTouchStrength` / `AispectImpactDatasetLabel`：用于表达训练样本的运动状态与触摸强度标签，当前将“手持静止”和“放置静止”显式拆分，避免旧 `static` 标签混杂两类场景；同时保留对历史 `static` 样本的兼容解码并映射到 `handheld_static`。
- `AispectImpactSystemActivityEvidence`：封装 `CMMotionActivityManager` 给出的系统侧活动证据，记录 `walking/stationary/running` 等状态布尔值、置信度与授权状态，用于和人工标签一起落盘，便于后续清洗错标样本。
- `AispectImpactDatasetCollector.Status`：新增结构化状态回调，包含任务标签、有效/无效计数、任务目标数与最近一次采集结果，便于 UI 做任务式反馈。
- `AispectImpactDatasetCollector`：内部改为串行状态队列，避免主线程标签切换与 Core Motion 采样并发写入导致错标签。
- `AispectImpactDatasetCollector`：在采集开关开启时同步启动 `CMMotionActivityManager`，将最新系统活动证据、采集 session 标识、App 版本号与 build 号写入每条样本 JSON。
- `AispectImpactDatasetCollector`：历史样本统计改为启动后在后台串行队列异步扫描，避免当本地 JSON 数量较大时阻塞主线程并造成 App 打开卡死。
- `AispectImpactDetector`：新增采集器透传接口，在现有加速度采样链路中直接写入训练数据，避免单独再开一套传感器监听；同时将有效采集与拖动/取消样本的丢弃显式分流。
- `RootViewController`：仅在 `lightTap` / `heavyTap` 时接受样本，`dragEnd` / `cancel` 直接记为无效并丢弃；触摸结束后延迟约 80ms 再停止冲击采样，以便补到真实后置帧。
- `AispectImpactTestView`：测试面板升级为任务式采集交互，采集中锁定标签，显示中文引导、任务进度、最近一次采样结果、系统活动证据摘要与采集目录复制按钮；运动状态标签扩展为“手持静止 / 放置静止 / 走路”三档。
- `AispectImpactTestView`：测试面板新增滚动容器与折叠按钮，内容过长时可纵向滚动查看，并可收起为右上角小按钮，避免在 iPhone 上遮挡主交互区域。
- `RootViewController` / `AispectImpactTestView`：测试面板展开态最大高度会按当前安全区动态限制，防止整块面板超出屏幕导致回弹异常，保证滚动只发生在面板内部。
- `Localizable.strings`：补充冲击测试面板与采集器的中文资源，完成手机端展示汉化。
- `ZHIFAeyes.xcodeproj`：补充 `NSMotionUsageDescription`，允许真机上请求系统运动与健身权限，采集 walking/static 的系统侧证据；当前工程名、Target、产品名与入口文件已统一重命名为 `ZHIFAeyes`。
- 采集数据目录：默认写入 App 沙盒 `Documents/AispectImpactDataset`，每条样本单独保存为 JSON，便于后续批量导出到训练脚本。

## 按下/抬起双锚点原始采集补充
- `AispectSignalWindowBuilder.buildDualAnchor`：Android 原始窗口使用时间与帧数双下限。开始位置取 `down-85ms` 与按下前 `15` 个真实 IMU 帧中更早者；结束位置取 `lift+85ms` 与抬起后 `15` 个真实 IMU 帧中更晚者。窗口长度随采样率和按压时长变化，不补零伪造原始帧。
- Android `AispectCollectionController`：抬起后首次等待 `85ms` 检查窗口；若按下侧已完整而抬起侧尚未同时满足 `85ms` 和 `15` 帧，则每 `8ms` 继续检查，最长等待 `500ms`。超时或按下前冗余无法补救时仍保存真实帧，并以 `windowComplete=false` 供清洗层筛除。
- Android schema `6` 保持兼容并新增版本化 `captureWindow` 审计对象，声明 `dual_anchor_min_time_and_frames_v1`、前后要求时长/帧数、实际覆盖时长和四个分项完成状态；旧根级 `preDownFrameCount`、`postLiftFrameCount` 与 `windowComplete` 继续保留。
- iOS `AispectImpactDetector` / `AispectImpactDatasetCollector` 当前仍为 schema `5` 和纯前后 `15` 帧规则，尚未同步 Android 的 `85ms + 15帧` 双下限；跨平台采集合同在 iOS 完成同构改造和真机验收前不能视为已统一。
- 原始采集冗余与模型输入严格分离。当前 v2 模型仍只使用 `-15...+25ms` 九个物理时间槽，扩大原始保存范围不会增加模型通道、训练窗口或按下期推理延迟。
- 新标准原始 IMU 帧不写入 `delta` 或 `maxAbsDelta`，只保留真实加速度、陀螺仪、重力、触摸归一化坐标和时间对齐字段。旧模型资源和历史数据保留，但在新标准采集路径中不作为分类结果；新模型需在 schema v6 数据重新采集后训练。

## Android schema v6 时间对齐补充
- `AispectCaptureClock`：`ACTION_DOWN` 时固定一次 `SystemClock.uptimeMillis()` 与 `SystemClock.elapsedRealtimeNanos()` 的时钟原点，之后所有 `MotionEvent.getEventTime()` 均使用该固定偏移换算到 `elapsedRealtimeNanos`。这样同一毫秒内的历史点不会因不同回调的亚毫秒接收偏移发生倒序；原始 `eventTimeMillis` 仍完整保存。
- `AispectModels.ImpactFrame`：每个 IMU 帧新增 `sensorTimestampElapsedRealtimeNanos`（硬件 `SensorEvent.timestamp`）和 `receivedElapsedRealtimeNanos`（应用收到该回调的单调时钟）。窗口复制与锚点重算会保留这两个字段，补位帧统一为 `0`，不伪造原始时间戳。
- `AispectTouchFrame`：每个 `touchFrames[]` 元素新增 `eventElapsedRealtimeNanos` 与 `receivedElapsedRealtimeNanos`。事件时间使用按下锚点固定偏移还原；同一 `eventTimeMillis` 的点允许相同纳秒时间，并按 `frameIndex` 保持稳定顺序。`timeAlignment.schemaVersion=2` 且 `touchTimestampConversion=touch_event_time_millis_from_down_anchor_v2`。
- 旧 schema v6 的 `timeAlignment.schemaVersion=1` 数据可在清洗端按 `downEventElapsedRealtimeNanos + (eventTimeMillis-downEventTimeMillis)*1e6` 重建触摸事件时间；不需要丢弃原始样本。
- `AispectTouchInputMetadata`：触摸输入源名称不调用设备上不存在的隐藏/非稳定 SDK API，保留 `source` 原始整数并以 `0x%08x` 形式写出 `sourceName`，保证 compileSdk 34 及运行端兼容。
- `AispectCollectionController`：`ACTION_DOWN` 保存按下事件/接收单调时间；按下和抬起锚点、IMU 窗口选择均使用同一 `elapsedRealtime` 时间轴。采集窗口的旧相对时间字段继续写出，供历史工具兼容读取。
- `AispectDataLogger`：新记录使用 `schemaVersion: 6`，新增 `timeAlignment` 对象，保存按下/抬起的事件与接收时间、时钟来源和双时间门。训练阶段必须同时满足 `touchReceivedElapsedRealtimeNanos <= imuReceivedElapsedRealtimeNanos` 与 `touchEventElapsedRealtimeNanos <= imuSensorTimestampElapsedRealtimeNanos`，再选择最新触摸帧执行零阶保持（ZOH），不得按名义采样率配对。
- `android/analysis/causal_press_features.py` / `AispectCausalPressFeatureBuilder`：schema v6 按下期活动合同为 `causal_imu12_v1`（9 个 IMU + 归一化坐标 + `touch_valid`）和 `causal_imu11_touch9_v1`（20 通道）。二者只读取 IMU 与逐帧 `touchFrames[]`，不读取根级 Contact Patch、整次归一化画像、抬起相对时间或其他结束后汇总。20 通道版本追加 pressure、size、按当前视图对角线归一化的 major/minor、对数轴比、方向 sin/cos、指针数和触摸有效位。缺少 IMU 硬件/接收时间、触摸事件/接收时间或已对齐触摸帧的视图尺寸时，因果输入拒绝构建。
- `AispectTouchFrame`：schema v6 的 `touchFrames[]` 每帧新增 `viewWidthPx` 与 `viewHeightPx`，使训练端可以使用产生该触摸帧时的真实视图对角线归一化接触轴，而不使用整次触摸最大值或结束后聚合值。
- `train_android_finger_four_class_cnn.py` / `AispectImpactCNNClassifier`：新的 down-anchor CLI 默认并强制使用 causal preset，拒绝 legacy preset、手工特征名单和矩阵后处理变换；训练 scaler 与报告写入 `featureContract`。训练分割出口强制检查 train/valid 的原始 session 不交集且不得缺失 session ID。Android 仅在 scaler 的合同为 causal v1 时把已就绪按下窗口和 `activeTouchFrames` 传入因果构建器，然后使用导出的 center/scale 标准化；旧模型保持显式 legacy 兼容路径，不会被解释为新因果模型。
- `android/analysis/training_common.py` / `android/analysis/legacy/README.zh-CN.md`：新训练器的通用 PyTorch、张量、解析和 JSON 工具已与旧 schema v5 训练脚本解耦；旧 `train_android_touch_cnn.py` 及其汇总特征路径只允许历史复现，禁止处理 schema v6 因果训练。
- `AispectLegacyFeatureBuilder`：旧 Contact Patch、`touchNormalization`、高斯图、多通道、协方差和 `time_since_anchor` 读取已集中到 `aispect/legacy` 包，仅供非 causal 合同历史模型复现。主分类器先判断 `causal_*` 合同，causal 分支不会调用该类，也不会读取根级聚合字段；新增因果模型禁止复用 legacy 入口。输入守卫分别校验时间槽数和通道数，不再错误要求两者相等，允许合法的 `时间槽数 x 特征通道数` 矩阵。

## Android 移植与采集补充
- `android/`：新增独立 Android 多模块工程，包含可复用库模块 `aispect-core` 与采集示例 App；当前迁移范围排除声学与闭眼相机链路，优先落地触摸采集、IMU 窗口、设备能力画像、CNN 诊断与 JSON 落盘。
- `AispectCollectionController`：Android 采集主入口，宿主 View 只需把 `MotionEvent` 与控件宽高传入 `handleMotionEvent`，即可完成单次触摸的原始触摸帧、事件流、按下期窗口就绪状态、按下/抬起双锚点 IMU 原始窗口、设备能力与训练样本保存；旧 CNN 诊断路径默认关闭。
- `AispectTouchFrame` / `AispectTouchAxisReader`：采集 Android 系统公开触摸底层字段，保留每帧 `pointerId/actionMasked/actionIndex`、事件/接收双时钟，以及每个轴的 `declared/observed/synthesizedOrUnknown` 和来源字符串；便捷 getter 只能作为同一 MotionEvent 的读取路径，不能改变轴语义。
- `AispectDeviceCapabilityProfiler`：基于 `InputDevice.MotionRange` 与实际采样值生成 Level A/B/C 能力画像；范围声明、值存在和信息性分开记录，恒定压力等 Framework 默认值不能仅凭存在或声明范围升级能力。画像新增 `touchAxesSemanticallyValid`、`touchAxesDistinct` 与 `toolAxesIndependent`：出现 `touchMajor < touchMinor` 时触摸轴语义无效，major/minor 全程相等时只证明半径而不证明独立形状，`toolMajor/toolMinor` 与触摸轴全程同值时只能视为别名而非独立证据。Level A 要求可靠 major 且 minor、pressure 或 orientation 至少一个独立增强字段真实有变化；仅有可靠半径或只有同源归一化 `size` 时为 Level B，其他情况为 Level C。单条 JSON 的画像通过 `snapshotForFrames(activeTouchFrames)` 仅从本次触摸重建，避免 observed、variable、informative 和轴别名状态继承上一条触摸；累计画像仅可用于界面诊断。所有原始字段仍原样保存供离线重判。
- `AispectContactPatch`：把 Android 触摸椭圆归一化为跨设备接触 patch；只使用 `touchMajor/touchMinor`，当只有触摸长轴时使用明确命名的 `touch_major_only_synthesized_minor`，绝不把 `toolMajor/toolMinor` 静默覆盖为触摸轴；所有半径字段不可用时使用默认小 patch 并记录 `defaultApplied`。
- `AispectContactEncodingProfile`：新增真正面向 CNN 输入设计的 Contact Patch 三方案编码，包含 `gaussian_map`、`multi_channel`、`covariance_matrix`。其中二维高斯方案输出固定 `9x9` 接触热图，缺失半径时使用默认 `sigma = 12px` 且 `validMask = 0`；多通道方案输出中心、major、minor、orientation 与 valid mask；协方差方案输出 `x/y/cov_11/cov_12/cov_22/validMask`。每条新 Android 样本会同时写入三套编码，面板按钮只标记本轮主方案，便于后续离线横向比对。
- `AispectTouchNormalizationEngine` / `AispectTouchNormalizationProfile`：Android 触摸归一化新增三套可切换方案，分别为屏幕比例归一化、接触椭圆协方差矩阵归一化、本轮历史基线相对归一化；接触长短轴只读取逐帧 `touchMajor/touchMinor`，`toolMajor/toolMinor` 仅保留为原始观察字段，不能进入接触归一化。诊断同步输出 `touchAxesSemanticallyValid/toolAxesIndependent`；轴语义倒置或 major/minor 不可区分时不能声明 `hasShape=true`，字段质量最高降为 `medium`，防止设备默认值、单半径或轴别名被误评为 high。采集面板允许在采集关闭时切换当前方案，训练样本 JSON 会同时写入 `screenScale`、`covarianceMatrix`、`sessionBaseline`、设备内分位诊断、短期历史偏移诊断、字段质量标记与 `contactPatch`，便于后续离线对比哪套方案更优。
- `AispectImpactSignalCollector` / `AispectSignalWindowBuilder`：使用 `Sensor.TYPE_LINEAR_ACCELERATION` 优先、同一时间轴的普通加速度减重力次之；每次注册都检查 `SensorManager.registerListener()` 的布尔结果，线性加速度或姿态候选注册失败、首帧超时或停流时才切换下一候选并递增 `streamEpoch`。重力只有在注册成功、已有首帧且未停流时才允许声明 `accelerometer_minus_gravity`；普通加速度完成减重力后，记录级与逐帧均使用该派生来源名，不再把输出误标成带重力原始值。重力停流或恢复造成 `accelerometer_minus_gravity` 与 `accelerometer_raw_no_gravity` 切换时也必须建立新 `streamEpoch`。无有效重力时普通加速度仍可作为原始证据保存，但逐帧与记录级均固定 `trainingInput=false`；只有 `linear_acceleration` 与 `accelerometer_minus_gravity` 可标记为训练候选。窗口重建只改锚点相对时间，必须逐帧透传原始 `sensorSource/streamEpoch/trainingInput` 和双时钟。陀螺仪、重力和姿态候选分别记录 `UNAVAILABLE/WAITING_FIRST_EVENT/HEALTHY/DEGRADED/STALLED` 健康状态和时间戳倒退计数；姿态优先采集 `GAME_ROTATION_VECTOR`，再独立回退到其他姿态流，写入 `sensorMetadata.rotationFrames[]` 且固定 `trainingInput=false`。姿态来源允许随回退跨 epoch 改变，但同一 epoch 内禁止混源，最后一个健康姿态帧来源必须与当前活动来源一致。Android 目标采样周期调整为 4167us；`ACTION_DOWN` 后的端侧模型窗口与原始采集窗口分离，原始窗口开始于按下前同时满足至少 `85ms` 和 `15` 个真实 IMU 帧的位置，结束于抬起后同时满足至少 `85ms` 和 `15` 帧的位置，不计算 delta。
- `AispectImpactSignalCollector` / `AndroidManifest.xml`：补充 Android 12+ 高频传感器权限 `HIGH_SAMPLING_RATE_SENSORS`，并在注册 4167us 采样失败时按 10000us、16667us、`SENSOR_DELAY_GAME`、`SENSOR_DELAY_UI` 逐级降级，避免小米 Android 15 因高频 IMU 注册抛出 `SecurityException` 导致点击采集开关闪退。
- `AispectImpactCNNClassifier`：Android 侧内置纯 Java 1D CNN 推理器，支持当前 iOS 4 通道模型与旧触控包内多组 `g11/g13_time` 模型；新增 `ModelInfo`、`availableModels()`、`modelInfoForId()` 与 catalog 加载缓存，宿主可以显示真实模型列表并指定模型执行诊断推理。历史 Contact Patch 与归一化字段统一由 `AispectLegacyFeatureBuilder` 提供，不能从 causal 分支进入。
- `AispectLegacyFeatureBuilder`：补齐旧 Contact Patch 三方案 CNN 输入的手机端同名特征，支持 `gaussian.map.00...80`、`gaussian.center/sum/energy/spread/sigma_x_px/sigma_y_px/valid_mask`、`multi.center_x_norm/center_y_norm/major_px/minor_px/orientation_rad/valid_mask`、`covariance.x_norm/y_norm/cov_11/cov_12/cov_22/valid_mask`。缺失 IMU 帧与历史训练脚本保持一致按零帧补齐；该路径仅用于旧模型复现。
- `AispectImpactCNNClassifier`：关键输入链路已补充中文注释，明确手机端必须按训练导出的 `featureNames` 顺序取值、缺失时间帧补零、高斯热图缺失时重建、默认 `sigma` 只作占位且由 `validMask` 表达可信度，避免后续复用模块时误把 `confidence` 当成力度标签。
- `AispectLegacyFeatureBuilder`：仅为旧模型复现保留 `contactPatch.*`、`touchNormalization.*`、`gaussian.*`、`multi.*` 和 `covariance.*` 的特征映射，并保持缺失历史字段的旧占位策略。该类不参与任何 `causal_imu12_v1` 或 `causal_imu11_touch9_v1` 模型。
- `AispectImpactCNNClassifier`：新增 Android 全量原始数据前五窗口 22 通道 CNN 模型，包含 `android_patch22_pre5_post9`、`android_patch22_pre5_post6`、`android_patch22_pre5_post10`、`android_patch22_pre5_post0` 与 `android_patch22_pre5_post7`；旧 `g13_time` 模型继续作为 catalog 后续可选项保留。
- `AispectImpactCNNClassifier`：二分类默认回装模型切到 `android_patch21_no_confidence_pre5_post9`，输入为 `enhanced_patch21_no_confidence` 的 21 通道 `-5...+9` 窗口，不再把 `touchNormalization.confidence` 作为训练或推理通道；当前权重使用全量 high_quality 清单训练，并将大面积轻触 hard negative 加权为 3.0。旧 `android_patch22_pre5_post9` 仍留在 catalog 中便于手机端手动切换对照。
- `AispectImpactCNNClassifier` / `AispectModels.ImpactPrediction`：端侧 CNN 推理从固定二分类输出扩展为读取 scaler 中的 `classCount` 与 `labelOrder`，支持二分类 heavy/light 与四分类 `thumb_light/thumb_heavy/index_light/index_heavy` 共用同一套模型切换、推理和 JSON 诊断落盘；旧 `heavyProbability/lightProbability/isHeavy()` 字段继续保留作为二分类兼容接口。
- `AispectImpactCNNModelCatalog.json`：新增并置顶拇指/食指轻重点四分类消融前五模型，依次为 `old13_raw_radius`、`old13_multichannel`、`enhanced_patch22`、`old13_gaussian_summary` 与 `enhanced_patch21_no_confidence`，手机端默认选中第一名并可在面板中继续热切换。
- `AispectImpactCNNModelCatalog.json`：新增并置顶基于 `lemon + tao` 两人全量 30214 条样本重新训练的四分类消融前五模型，依次为 `enhanced_patch21_no_confidence`、`enhanced_patch22`、`old13_multichannel`、`old13_gaussian_summary` 与 `old13_raw_radius`；旧四分类与二分类模型仍保留在 catalog 后面，可继续在手机端手动切换对照。
- `data/` / `models/` / `.gitattributes`：GitHub 上传结构按数据、模型、报告和代码分层整理；`lemon` 与 `tao` 展开 JSON 目录保留为本地工作副本，上传时使用 `data/lemon` 与 `data/tao` 下的压缩包，CNN 权重、scaler、数据压缩包与 manifest CSV 通过 Git LFS 跟踪。`android/local.properties`、展开原始目录和 `android/analysis/results` 完整消融中间结果不进入 Git；可复查的精简结果以 `models/reports`、`models/lemon_tao_full4_top5` 与 `data/manifests` 为准。
- `AispectImpactCNNModelCatalog.json`：新增并置顶 OnePlus PHK110 设备专属四分类时间留出优先模型 `oneplus_four_class_hard_core_old13`，基于当前手机 `handheld_static` 四类 high_quality + review hard 样本平衡训练，并按每类最新 25% 做时间留出复核。随机切分第一名 `old13_raw_radius` 在时间留出下暴露食指/拇指混淆，`old13` 时间留出平衡准确率约 `91.67%` 且 `index_light` 召回约 `93.9%`，因此端侧四分类默认模型切到 `old13`，`old13_raw_radius` 保留为 catalog 第二项用于对照。
- `AispectImpactCNNModelCatalog.json`：新增并置顶补采后 OnePlus 四分类装机模型 `oneplus_four_class_after_recollect_hard_nearfull_old13_raw_radius`，并保留 `oneplus_four_class_after_recollect_hard_temporal10_old13_raw_radius` 作为第二项对照。模型选择基于手机当前 `4641` 条原生 JSON 重新清洗后的 hard 平衡集，每类 `321` 条；严格每类最新 `25%` 时间留出显示分布漂移仍很强，最佳仅约 `71.56%` 平衡准确率，因此只作为自我批判指标。用于选择 preset 的 `10%` 最新尾段留出中，`old13_raw_radius` 平衡准确率约 `88.28%`，四类召回约 `thumb_light 75.0% / thumb_heavy 96.9% / index_light 90.6% / index_heavy 90.6%`，同强度轻触互混 `7` 次、重触互混 `1` 次；当前端侧默认四分类模型使用同一 preset 的 nearfull 训练版，训练集中每类 `320` 条，剩余每类 `1` 条仅用于导出健康检查。
- `AispectImpactCNNModelCatalog.json`：修复四分类训练脚本未真正使用 `sample_weight` 的问题后，新增并置顶 weighted 装机模型 `oneplus_four_class_after_recollect_hard_nearfull_weighted_old13_multichannel`，并保留 `oneplus_four_class_after_recollect_hard_temporal10_weighted_old13_multichannel` 作为第二项评估对照。weighted `10%` 最新尾段核心消融中，`old13_multichannel` 平衡准确率约 `87.50%`、最小召回约 `81.25%`，四类召回约 `thumb_light 84.4% / thumb_heavy 90.6% / index_light 93.8% / index_heavy 81.2%`，同强度轻触互混 `6` 次、重触互混 `0` 次；相较 raw radius，牺牲少量总分但降低半径单点依赖并改善最弱类，因此端侧默认四分类模型切到 weighted multichannel nearfull 版。
- `AispectCollectionController`：旧模型选择状态仍保留用于历史对照，但新标准不调用 CNN。`Status` 输出按下窗口等待/就绪、最近采样率与原始窗口摘要；触摸结束后生成 `contactPatch` 与三套触摸归一化画像并写入 schema v6，训练 JSON 不写入 CNN 预测或 delta 派生值。
- `AispectModels.ImpactPrediction` / `AispectCollectionController`：二分类 heavy 输出新增端侧证据约束，CNN 判为 `heavy` 后仍需满足足够冲击 `maxAbsDelta`，并同时具备持续形态变化或时间动态证据；接触面积、半径、字段质量和归一化置信度不能单独把一次触摸推出重触。证据不足时二分类结果会降级为 light，四分类模型输出保持原有标签分布用于验证和切换。
- `AispectCollectionController` / `MainActivity`：CNN 模型切换改为手机端热切换，训练采集开关打开后仍可在两次触摸之间切换“下一次触摸”使用的模型；单次触摸开始时会锁定当时的 `activeModelId`，避免一条样本中途换模型导致诊断结果不可追溯。标签、运动场景与接触编码仍按采集任务锁定，保证训练真值稳定。
- `MainActivity`：CNN 诊断显示改为固定高度方框，稳定展示模型、最高预测、置信度和人工标签，不再因为概率列表或行数变化撑开布局；人工标签占位符与传参顺序同步修正为手指、轻重点、场景，四分类模型继续展示预测类别，旧二分类模型继续显示轻点/重点输出，模型切换面板仍沿用现有上一项/下一项交互。
- `MainActivity`：采集教程区域固定为五行高度并顶部对齐，采集开关切换导致教程文案从 `1...5` 变成锁定说明时，不再改变后续采集状态、原生样本、CNN 诊断和四分类验证框的位置。
- `MainActivity`：采集面板新增“四分类验证”显示框，验证模式开启后持续显示并累计 `thumb_light/thumb_heavy/index_light/index_heavy` 四类 `correct/total/recall`、macro recall 与总体 accuracy；训练采集开关打开时仍保存原生训练 JSON，同时也会把完成预测计入验证统计。
- `MainActivity`：新增“验证模式”开关，默认开启；面板外触摸会直接显示人工标签、CNN 输出、对错、置信度、当前组合统计、四类召回、macro recall 与总体 accuracy，用于实时观察当前模型准确性。
- `AispectCollectionController` / `MainActivity`：四分类召回统计新增完成预测序号 `latestPredictionSequence` 作为去重键，并接受 `collector_diagnostic_ready` 与 `collector_event_saved` 两类完成状态进入计数；新一轮 `ACTION_DOWN`、采集开关切换、标签切换、模型切换或编码方案切换只清空上一轮预测快照，不清空累计统计，避免同一条预测重复计数又保留跨开关/跨模型的连续验证总量。训练样本落盘口径不变，采集开关开启时仍把每次面板外触摸按当前人工标签保存为 accepted 样本。
- `MainActivity`：面板常驻区域隐藏冲击窗口日志、复制日志按钮和接触编码明细数据，避免长窗口文本挤占四分类验证数据显示空间。
- `AispectCollectionController` / `MainActivity`：训练采集改为无触觉反馈、无红色冲击闪屏、无 CNN 预测落盘；`cnnPrediction` 只用于屏幕验证显示，不写入训练 JSON，确保导出的训练数据只包含人工标签、原始触摸帧、IMU 窗口、设备能力、Contact Patch 与接触编码等原生字段。面板新增“最近原生样本”摘要，显示刚保存样本的人工标签、触摸帧数、窗口帧数、采样率、`maxDelta`、major/minor/pressure 和文件名。
- `zhifaIOS_old.app` 适配：Android assets 已接入 `/Users/xueyin/Desktop/zhifaIOS_old.app` 内的旧触控 CNN 模型 catalog 与多组 `g11/g13_time` 权重；示例 App 默认选中旧包中的 `g13_time_f14_pre4_post9` 增强模型，并允许手动切换 catalog 内 8 个模型。旧包中的 iOS Mach-O 可执行文件不能直接移植到 Android，683MB 的训练数据导出文件也不打入 APK。
- `AispectDataLogger`：默认把 Android 训练样本写入 App 外部私有 Documents 下的 `AispectAndroidDataset`，样本 JSON 包含 `touchFrames` 原始触摸字段、`deviceCapability`、`contactPatch`、`contactEncoding`、对齐后的 `frames` 与 `motionSignal`；训练采集路径不写入 `cnnPrediction`，避免模型输出反向污染训练集。`sensorMetadata.rotationFrames[]` 只保存最终 IMU 原始窗口首尾硬件时间戳之间的姿态观察帧，`motionSignal.rotationFrameCount` 与实际落盘数量一致，不再复制最长约 6 秒的整段滚动缓存。
- `AispectTouchInputMetadata` / `AispectDataLogger`：记录 `InputDevice.MotionRange` 的轴、来源、范围、flat、fuzz、resolution，以及 API、品牌/型号、构建指纹、显示尺寸/模式、density/xdpi/ydpi、View 尺寸和变换版本。`touchMetadata.metadataEpoch` 表示首个触摸帧所属版本，每个 `touchFrames[]` 元素也写入正整数 `metadataEpoch`；记录级显示快照、逐帧 View 尺寸或逐帧输入设备元数据相对前一帧变化时递增，指纹不变时保持不变，跨样本也不回退。当前没有逐事件保存 `displayRotation/windowInsets`，因此只能定位已有逐帧证据中的 View/输入设备切换，不能伪称可还原触摸内部的旋转或 Insets 历史。`windowInsets` 在无 Activity/View 注入时明确标为不可用，不用零值伪装已测量 Insets；`physical_mm` 尚未生成，当前元数据只作为原始证据和后续标定输入，不能被解释为统一物理单位。
- `AispectDataLogger` / `AispectCollectionController`：Android 采集落盘口径改为“采集阶段不判断有效/无效”，所有面板外触摸结束路径都会按当前人工选择的运动状态、轻重点与手指类型保存为 `recordStatus = accepted` 的训练样本；拖动、取消、偏移等行为不再由手机端丢弃或标成 rejected。控制器同时增加单次触摸保存闸门，同一轮 `ACTION_DOWN` 后即使系统连续派发多个结束类事件，也只会安排一次落盘，避免一次按压生成多条样本。
- `MainActivity`：Android 示例 App 改为复刻旧 iOS 的全屏主界面 + 左上悬浮 `AispectImpactTestView` 面板结构；面板展开宽度对齐 iOS 的 204pt 结构，收起态为 132x64 可点击小面板，采集控件从大按钮改为 switch，冲击阈值区域补齐 slider 与格式化阈值文本。面板负责“冲击测试”“CNN 模型切换”“训练数据采集”、上一项/下一项模型切换、采集开关、三档运动场景、轻点/重点、引导文案、状态日志、CNN 推理结果与数据目录复制。训练采集开启后，`dispatchTouchEvent` 以 ACTION_DOWN 是否在面板外为准锁定本轮全屏采集，MOVE/UP/CANCEL 即使经过面板上方也继续送入采集链路，不再限制在底部采集框；收起/展开切换在 `dispatchTouchEvent` 最前面使用独立热区优先响应，收起态按整块小面板命中，展开态按面板顶部 58dp 命中，热区额外外扩 36dp，避免被全屏采集或按钮自身命中区影响。
- `MainActivity` / `DemoHomeView`：触摸反馈收敛为中性触摸环，只按原始触摸半径/面积改变大小，不触发振动、红色重触提示或全屏冲击闪烁，避免采集时把 App 反馈混进用户动作和 IMU 数据。
- `MainActivity`：训练采集期间将人工选择的 `label.touchStrength` 与 CNN 验证显示彻底区分。样本真值始终来自采集面板锁定的轻点/重点标签；CNN 输出仅用于屏幕验证，不写入训练 JSON，也不作为最终识别文案。
- `AispectCollectionController`：Android 训练采集在触摸结束时把 `LIGHT_TAP/HEAVY_TAP` 统一重写为本轮锁定的人工标签，`sourceTouchKind` 与 `touchEvents` 最后一条结束事件都会同步为 `lightTap` 或 `heavyTap`，避免轻点样本因底层抬起偏移偶发被记录成重点。
- `MainActivity` / `DemoHomeView`：训练阶段默认关闭旧 CNN 实时诊断，结果区显示“训练阶段旧 CNN 已关闭”和当前人工训练标签；轻点采集开启时同时压低触摸环视觉强度，避免半径波动在界面上产生红色重触反馈错觉。
- `MainActivity` / `DemoHomeView`：Android 背景从假消息列表清理为训练采集底板，保留暗色渐变、采集区域边界、轻量网格、中心准星、当前采集状态与触摸环，去掉与训练无关的会话和输入框装饰。
- `MainActivity`：悬浮面板切换改为只响应明确的收起/测试按钮命中，收起态缩小为 `88x42dp` 并放在右上侧，展开态不再让整条标题栏或整块面板触发收起，减少采集区测试时误触面板。
- `MainActivity`：修复展开面板滚动后收起按钮可能被单独提升到面板底部的问题，切换时只提升整个面板层级，不再改变按钮在 `LinearLayout` 内的顺序，保证收起/测试按钮位置稳定。
- `AispectModels.DatasetLabel` / `MainActivity`：Android 训练采集标签从“运动状态 + 轻重触控”扩展为“运动状态 + 轻重触控 + 手指类型”，当前支持 `thumb` 与 `index`。采集面板新增“拇指 / 食指”分段按钮，采集开关开启后与运动状态、轻重点一同锁定；新样本 JSON 的 `label.fingerType` 与文件名 `datasetKey` 会保留该维度，方便后续按拇指/食指分别训练或评估。
- `AispectCollectionController` / `MainActivity`：新增采集关闭状态下的验证诊断入口。验证模式默认开启，用户不打开训练采集开关也能在全屏触摸区触发一次 CNN 推理；控制器会临时启动 IMU、收集触摸窗口、生成 Contact Patch 与归一化画像、调用当前选中模型输出四分类结果，但不会调用 `AispectDataLogger.writeSample()`，因此不会写入训练 JSON。
- `MainActivity`：验证面板显示当前四分类模型输出、置信度、四类概率、人工标签对比和四类召回统计。统计口径使用面板当前选择的 `thumb/index + light/heavy` 作为人工答案，只在训练采集关闭时生效，适合快速观察“当前模型在手机上是否按预期识别”，不替代离线留出集评估。
- `MainActivity`：扩大测试面板收起/展开按钮和收起态小面板尺寸，按钮热区外扩到 22dp，降低全屏采集时误触或点不中面板切换的概率。
- `android/analysis/clean_android_touch_dataset.py`：新增 Android 采集数据清洗工具，读取真机导出的 JSON 样本后生成 `manifest.csv`、`train_manifest.csv`、`balanced_train_manifest.csv` 与 `summary.json`，并可按 `high_quality/review/rejected` 三层复制样本。清洗真值只使用面板里人工选择的 `label.touchStrength`，不再把 `sourceTouchKind` 或旧 CNN 输出当训练标签；旧 CNN 几乎全判 heavy 的现象会写入 `quality_warnings`，只作旧模型失效诊断。manifest 额外输出 `sample_weight`，把近期大面积轻触和 `light_label_heavy_shape` 样本作为 hard negative 加权进入训练，其中大面积轻触同时支持固定阈值和轻触分布 p90 阈值。
- `android/analysis/clean_android_touch_dataset.py`：manifest 增加 `finger_type` 列，新样本按 `thumb/index` 进入 `label_id`；历史样本缺少 `label.fingerType` 时默认按 `thumb` 读取，避免旧数据在训练视图中落入 `unknown` 类。
- `android/analysis/mark_android_history_thumb.py`：历史 Android 样本手指标签补标脚本，按结构化 JSON/CSV 解析补齐 `label.fingerType = thumb`、`finger_type` 列与带 `_thumb` 的标签名；当前已用于本地 `9185` 条唯一原始样本及其展开整理副本，保持旧采集数据明确归类为拇指。
- `android/analysis/build_android_finger_four_class_dataset.py`：从手机全量 raw 与旧拇指基准目录构建手指四分类数据集，按文件差集将旧样本视为 `thumb`、新增样本视为 `index`，并默认限定同一运动场景后按 `thumb_light/thumb_heavy/index_light/index_heavy` 四类抽取均衡 manifest 与可独立训练的 JSON 副本。当前脚本也支持直接读取清洗后的 `source manifest` 与 `quality_buckets`，优先使用 manifest/JSON 中的 `finger_type`，只有旧样本缺失手指字段时才回退到文件名或参考目录推断，避免补采样本被二次误标。
- `android/analysis/train_android_finger_four_class_cnn.py`：新增四分类 CNN 训练入口，复用现有 `enhanced_patch22` 特征提取和 `-5...+9` 时间窗口，输出四类顺序、权重、scaler 与训练报告；当前导出结果标明需要手机端推理器支持四类输出后才能直接部署到端侧。训练入口新增 `split_strategy`，支持随机分层切分与按每类 `created_at` 最新样本留出的 temporal 切分，后续默认用 temporal 结果审计连续采集造成的验证泄漏。四分类训练现在会读取 manifest 的 `sample_weight` 并使用 `CrossEntropyLoss(reduction="none")`，确保大面积轻触 hard negative 的权重真正进入损失函数，而不是只存在于 CSV 中。
- `android/analysis/clean_android_touch_dataset.py`：跨设备清洗规则改为按“设备名称 + 运动状态”计算 `max_abs_delta` 与触摸位移的稳健上限，保留绝对安全下限同时避免 PHK110 walking 的正常量纲被固定阈值整体误删；新增 `weighted_train_manifest.csv`，高质量样本权重为 `1.0`、非致命 review 样本权重为 `0.35`，疑似人工标签与形态冲突样本继续降权，不再把冲突样本反向放大。
- `android/analysis/build_grouped_four_class_manifest.py`：读取全量主 manifest 与多批清洗质量 manifest，按 ID 合并质量分层，未清洗的新 collection 默认拒绝进入训练；脚本按 collection 起点生成五分钟 `session_group`，支持按最小时间间隔压缩连续冗余样本，并把指定 collection 整体标记为 predefined validation，输出行数、质量层、去重、降采样和会话组统计。
- `android/analysis/train_android_finger_four_class_cnn.py`：新增 `sample_cache` 特征缓存和 `early_stopping_patience` 控制；缓存记录特征 preset、时间窗口和缺帧策略签名，后续 collection 留出折只重新应用当前 manifest 的 split，签名不一致、缓存缺少 manifest 样本或 manifest 新增缓存外样本时都会拒绝复用。早停仍按验证集平衡准确率保存最佳权重，减少跨批次指标停止改善后继续训练的成本。
- `models/all_collections_grouped_clean_v2_20260724`：保存四个新增 collection 的严格留出模型、一个随机切分对照、统一比较表与诊断报告。清洗后随机切分平衡准确率约 `83.77%`，四个整批留出折分别约为 `57.83% / 28.67% / 67.28% / 27.41%`；LE2110 接触长轴中位数约 `5~15px`、触摸帧约 `2~5`，PHK110 分别约 `1.2~1.5` 与 `7~9`，证明当前 patch21 仍主要学习设备和采集会话特征，暂不具备替换端侧模型的跨批次泛化条件。
- `android/analysis/train_android_finger_four_class_cnn.py`：现有数据离线去设备偏差实验新增 `imu11_no_time`、`invariant_shape` 与 `relative_touch` 三种矩阵变换。前者只保留坐标与九个 IMU 通道；无量纲形状方案追加长短轴对数比、协方差相关系数和偏心率；相对接触方案再追加当前 major/minor/面积相对同 collection/source group 最多 256 条历史无标签样本滚动中位数的对数比和基线就绪度，严格按时间先计算再写入当前样本，不读取未来样本或标签。
- `android/analysis/train_android_finger_four_class_cnn.py`：新增训练折内 `collectionName + classLabel` 逆频率权重，先按组样本数计算、归一化到均值 `1.0`、裁剪至 `0.25~4.0`，再乘 `cleaned_v2` 质量权重；该步骤只接收 split 后的训练样本，验证样本不参与统计或加权。
- `android/analysis/run_group_invariant_four_class_experiments.py`：基于同一 patch21 特征缓存串行执行三种派生输入乘四个完整 collection 留出折，固定 batch `128`、seed `22`、最多 `120` epoch 与 patience `20`，已完成报告自动跳过并支持中断恢复；结束后聚合每折平衡准确率、最弱召回和混淆矩阵到统一 CSV/JSON。
- `android/analysis/train_android_finger_four_class_cnn.py`：训练与验证张量改为 CPU 常驻并按 batch 搬入 CUDA，训练继续使用 CUDA `randperm` 保持随机顺序，验证按最多 2048 条分批推理；模型、优化器和损失函数仍驻留 GPU，避免 Windows WDDM 在整折张量一次映射时出现显存充足但 `resource already mapped` 或小额分配失败。
- `android/analysis/train_android_finger_four_class_cnn.py`：历史计划准备模式仍只读扫描旧 schema manifest；因果训练实际要求 schema v6、双时间字段、逐帧视图宽高和 session-disjoint split。缺失这些字段时必须拒绝样本，不能用设备常量或整批数据反推后静默补齐。
- `android/analysis/train_android_finger_four_class_cnn.py`：新增 `--anchor-mode down|lift`。`down` 使用 `downReferenceFrameIndex` 构造可部署的固定窗口；`lift` 仅用于离线抬起期上限对照。训练报告新增最佳 epoch、锚点模式、按 `motionState` 和 `durationBucket` 分组的训练/验证指标，避免只凭整体随机指标选择模型。
- `android/analysis/datasets/all_collections_invariant_v3_20260724` / `models/all_collections_invariant_v3_20260724`：保存仅使用现有数据的离线去设备偏差定义、12 个严格 collection 留出模型及候选比较。`relative_touch` 四折平均平衡准确率约 `44.72%`、最差折约 `32.11%`、最坏类别召回约 `9.10%`；相较原始 patch21，最差折和最坏类别有所改善，但平均值从约 `45.30%` 略降，未达到平均 `70%`、最差折 `60%` 的门槛，因此只保留为诊断结果，不导入 Android 模型目录。
- `models/all_collections_old13_seen_devices_v2_20260724`：完全复刻 `full_old13_seen_devices_120` 的 `old13`、`-5...+9`、120 epoch、batch `256`、学习率 `0.001`、weight decay `0.0001`、随机 `25%`、seed `22` 与 zero 缺帧方式，只把旧 `49051` 条数据替换为当前 grouped clean v2 的 `79878` 条。新模型训练/验证为 `59908/19970`，验证平衡准确率约 `79.15%`，低于旧模型约 `81.20%`；新增数据扩大了用户、设备和 collection 分布，随机验证仍可用但没有提升，模型保留为比较产物而不自动替换端侧默认模型。
- `AispectImpactCNNModelCatalog.json` / `crossDevice` 产品变体：将最终全量训练模型 `full_old13_seen_devices_current_preprocessed_all_118` 的 scaler 与 weights 导入 Android assets 并置顶模型目录，`crossDevice` 变体的 `DEFAULT_MODEL_ID` 同步指向该模型；带随机验证的 `full_old13_seen_devices_current_preprocessed_120` 与旧 `full_old13_seen_devices_120` 继续保留为可切换对照，标准变体默认模型不变。
- `android/analysis/train_android_finger_four_class_cnn.py`：新增 `split_strategy=all` 最终训练模式，全部样本进入训练且验证集为空；无验证样本时每个 epoch 都更新待导出状态，因此固定 epoch 训练会明确导出最后一轮权重，不使用训练准确率模拟模型选择，也不会触发验证早停。训练报告中的 valid count 和指标保持为零，避免把全量训练指标误解为验证结果。
- `models/all_collections_old13_seen_devices_v2_20260724/full_old13_seen_devices_current_preprocessed_all_118`：在随机验证模型确定最佳 epoch `118` 后，使用相同 old13、`-5...+9`、batch `256`、学习率 `0.001`、weight decay `0.0001` 与 seed `22`，把 grouped clean v2 全部 `79878` 条按四分类合并训练，不保留验证集。最终训练平衡准确率约 `85.34%`，四类训练召回约为 `thumb_light 79.40% / thumb_heavy 81.12% / index_light 92.44% / index_heavy 88.39%`；这些只表示全量拟合状态，模型选择依据仍是前一轮随机验证，不把训练指标当成新验证结论。
- `android/analysis/run_android_finger_four_class_cnn_ablation.py`：新增拇指/食指轻重点四分类正式 CNN 全量消融脚本，固定同一四分类 manifest、`-5...+9` 时间窗口、训练参数与随机种子，逐个调用 `train_android_finger_four_class_cnn.py` 训练 `old13`、raw 半径、多通道、协方差、高斯热图和 22 通道等 12 个 preset，并汇总通道数、四类召回、验证准确率、验证平衡准确率、最小召回与混淆矩阵到 `finger_four_class_cnn_ablation_summary.csv` 和 `finger_four_class_cnn_ablation_report.json`。消融入口同步透传 `split_strategy`，使 preset 排名可以直接基于时间留出，而不是只看随机验证。
- `android/analysis/run_schema5_full_factorial_ablation.py`：新增 schema v5 按下锚点四分类完整因子消融调度器，固定 12 个特征预设、7 个真实按下窗口和 3 个运动场景严格留出折。脚本复用既有 `feature_temporal`、`window_temporal` 和 `motion_holdout` 输出路径，只有同一 model id 的报告、权重和 scaler 全部存在时才跳过，支持中断恢复与 dry-run；每完成一项均重写状态、时间筛选和严格留出 CSV/JSON 汇总。每个 down-anchor 命令显式传入 `--require-press-active`，严格排序只作为离线候选评估，不导入 Android assets 或替换 App 默认模型。
- `android/analysis/train_android_finger_four_class_cnn.py`：按下锚点训练现在强制 `--require-press-active`，只保留 `liftReferenceFrameIndex >=` 请求最大正帧索引的样本，禁止把抬起后的原始 IMU 帧伪装成按压期输入；直接将该条件设为 `False` 会报错。缓存签名增加 `causalPressActiveV1`，旧缓存不能复用；按下期缓存允许在同一完整 manifest 的不同严格折之间复用。内部 `lift` 对齐接口仅保留给完整触摸历史兼容，CLI 训练仍限定 `down`。新增 `--deterministic`，训练时固定 CUDA/cuDNN 的确定性配置并写入完整参数报告。
- `android/analysis/run_android_down_window_252_ablation.py`：新增按下期时间窗口消融调度器。它生成 84 个连续 `down-pre ... down+post` 窗口规格，共覆盖 4 至 31 帧总长度；每个规格在 `handheld_static`、`placed_static`、`walking` 三个完整运动场景留出折上训练，合计 252 次。运行器固定 `old13_raw_radius`、按下锚点、因果过滤、确定性训练和既有 CNN 结构；每份完成报告的训练和验证计数都必须分别匹配 `liftReferenceFrameIndex >= postFrame` 的样本数。输出训练/验证/总可输出覆盖率、等待帧和约 210Hz 下的估算等待毫秒，并按验证准确率、最弱类召回率、总可输出覆盖率和更短等待构成的四目标 Pareto 前沿选择离线候选，不导入 Android assets。
- `android/analysis/run_android_down_window_252_ablation.py`：新增 `dense_4_15` 窗口计划，在 4 至 15 帧总长度内遍历所有连续的 `preFrames + anchor + postFrames` 分配，共 114 个窗口。它与原 `representative_84` 计划共用训练和汇总逻辑，通过 `--window-plan` 选择；在三个运动状态留出折上共生成 342 次严格训练任务，用于取代稀疏抽样后查找局部稳定窗口区间。
- `android/analysis/run_android_down_window_252_ablation.py`：新增 `--workers`，并发时按窗口分组调度，同一窗口的三个运动状态留出折依然串行，不同窗口可并发训练。该约束避免同一窗口共享 sample cache 的写入竞争；只有主调度进程写入全局汇总与进度文件，从而避免并发写文件冲突。子训练进程默认限制 `OMP_NUM_THREADS` 与 `MKL_NUM_THREADS` 为 `4`，也允许以环境变量覆盖；本次在 RTX 2080 上以 `6` 个窗口工作进程、每个 `2` 个 CPU 线程完成密集窗口训练。
- `models/bbsiak_schema5_down_window_252_20260807/`：完成 252/252 次严格训练，252 份报告、权重和 scaler 均存在，84 个候选全部完成三折。最佳严格候选为 `pre1_post2`，即 `frameIndex=-1,0,1,2` 的四帧窗口：平均 balanced accuracy `69.63%`、最差折 `68.04%`、最差类别召回 `47.96%`、按压期覆盖率 `100%`，中位最佳 epoch 为 `31`。原 `pre5_post9` 在因果口径下排名第 40，平均 balanced accuracy `63.59%`、最差类别召回 `6.96%`、最差覆盖率 `97.50%`。原始 `timeSinceTouchDown` 显示 `+2` 的等待为 `10ms`，`+9` 为 `45ms`，`+15` 为 `75ms`；后两者的全量可用率分别为 `98.45%` 和 `72.23%`。该结果仅说明当前数据和严格场景留出口径下的最佳按下期窗口，尚未部署到 App。
- `models/bbsiak_schema5_down_window_252_20260807/window_84_overview.csv`：按综合排名展开 84 个按下期窗口，使用“中文 (English)”对照表头并显式标记前十候选，并并列 `handheld_static`、`placed_static`、`walking` 三个留出折的最小类别召回率、平衡准确率与覆盖率；表尾保留三折平均、最差值和中位最佳 epoch，用于在 Excel/WPS 中直接筛选和对比离散窗口。
- `android/analysis/legacy/train_android_touch_cnn_legacy.py`：历史 schema v5 CNN 训练导出脚本，保留旧 `old13`、Contact Patch 三方案与时间特征，仅用于历史模型复现、差异审计和旧资产迁移；根级 `train_android_touch_cnn.py` 只保留兼容转发，不得作为 schema v6 因果训练入口。
- `android/analysis/legacy/train_android_touch_cnn_legacy.py`：历史 JSON 若没有 `contactEncoding` 字段，会从 `touchFrames` 重建二维高斯、多通道与协方差输入；半径缺失时使用旧默认接触区域并通过 `validMask` 标记未知。该逻辑与新的 `causal_press_features.py` 分离，不能混用。
- `android/analysis/export_android_touch_feature_report.py`：新增不依赖 PyTorch 的增强特征报告脚本，读取 `balanced_train_manifest.csv` 后输出 `enhanced_patch_feature_report.json`、`enhanced_patch_feature_summary.csv`、`enhanced_patch_threshold_scan.csv` 与样本预览。报告用于在正式训练 CNN 前确认 `rawMajor/rawMinor`、`cov_11/cov_22`、`sigmaX/sigmaY`、三套归一化面积等通道对轻点/重点是否已有可学习差异。
- `android/analysis/export_android_contact_encoding_report.py`：新增不依赖 PyTorch 的 Contact Patch 三方案比对脚本，直接从历史 JSON 的 `touchFrames` 原始字段重建二维高斯热图、多通道表示与协方差矩阵表示，输出 `contact_encoding_report.json`、`contact_encoding_samples.csv`、`contact_encoding_summary.csv` 和 `contact_encoding_threshold_scan.csv`。
- `android/analysis/run_android_touch_ablation.py`：新增不依赖外部 Python 包的 Android 触摸训练级消融脚本，读取清洗后的 `balanced_train_manifest.csv`，固定同一批 train/valid 划分，对 `old13`、`old13_raw_radius`、`old13_covariance`、`old13_multichannel`、`old13_gaussian_summary`、`old13_gaussian_map9x9`、`patch9_only`、`enhanced_patch21_no_confidence` 与 `enhanced_patch22` 执行多 seed 的 diagonal LDA probe，并输出 `ablation_runs.csv`、`ablation_summary.csv`、`ablation_predictions.csv` 与 `ablation_report.json`。该脚本用于在 PyTorch 不可用时先验证各输入组的训练/验证贡献，不替代最终 CNN 权重训练。
- `android/analysis/run_android_touch_cnn_ablation.py`：新增正式 CNN 全量消融批跑脚本，按同一 manifest、同一窗口、同一训练参数逐个调用 `train_android_touch_cnn.py`，对 `old13`、高斯热图、多通道、协方差、当前 22 通道等预设训练真实 1D CNN，并汇总每个 preset 的通道数、训练/验证准确率、平衡准确率、轻重召回和混淆矩阵到 `cnn_ablation_summary.csv` 与 `cnn_ablation_report.json`。
- `android/analysis/train_android_touch_cnn.py`：训练脚本新增自动设备选择，优先使用 CUDA，其次使用 Apple Silicon 的 MPS，最后回退 CPU；模型、输入张量和类别权重会统一放到同一设备上，评估结果再转回 CPU 统计，避免 Mac 本机训练被迫退回纯 CPU。
- `android/analysis/train_android_finger_four_class_cnn.py`：四分类训练脚本同步接入自动设备选择，支持在已完成 preset 基础上继续断点式恢复全量消融，并直接复用同一套 MPS/CUDA/CPU 设备逻辑。
- `android/analysis/train_android_finger_four_class_cnn.py`：四分类训练入口同步适配二分类训练器新增的 `sample_weight` 版 `train_one_epoch` 接口，默认给四分类样本使用权重 `1.0`，避免全量四分类消融在第一项训练时因接口不一致中断。
- `android/analysis/datasets/oneplus_20260625_phone_pull`：保存当前 OnePlus PHK110 手机导出的 `3374` 条原生训练 JSON、`clean_v1` 清洗结果、`finger_four_class_handheld_clean_v1` 与 `finger_four_class_handheld_hard_v1` 两套手持四分类平衡 manifest，以及 clean/hard 两轮核心 CNN 消融报告。hard 版本每类 `264` 条，包含 review 中的大位移/高冲击边界样本，用于比只取 clean 的每类 `116` 条更贴近真实操作。
- `android/analysis/datasets/oneplus_20260625_phone_pull/finger_four_class_temporal_holdout_v1`：新增 OnePlus 四分类时间留出复核报告。该报告指出当前四类数据按长段连续采集，随机切分会把同一操作段同时放入训练和验证，导致 `old13_raw_radius` 的随机 `97.35%` 被高估；时间留出下 `old13_raw_radius` 掉到约 `86.36%`，主要混淆 `index_light` 与 `thumb_light`，因此后续模型选择必须优先看时间/会话留出和同强度手指混淆矩阵。
- `android/analysis/datasets/oneplus_20260625_after_recollect`：保存本轮补采后从 OnePlus PHK110 导出的 `4641` 条原生 JSON、`clean_v1` 清洗结果、`finger_four_class_handheld_clean_v2` 与 `finger_four_class_handheld_hard_v2` 平衡 manifest。hard_v2 使用 `handheld_static` 的 high_quality + review 样本，每类 `321` 条，并生成 `finger_four_class_cnn_ablation_hard_temporal_v2` 与 `finger_four_class_cnn_ablation_hard_temporal10_v2` 两套时间留出消融报告；其中严格 `25%` 留出用于揭示采集顺序分布漂移，`10%` 最新尾段留出用于当前装机模型选择。
- `android/analysis/run_android_patch9_subset_ablation.py`：新增 `patch9` 接触通道组合穷举脚本，固定旧 13 通道和同一时间窗口，对 9 个接触通道的 `511` 个非空子集执行多 seed diagonal LDA probe，输出 `patch9_subset_runs.csv`、`patch9_subset_summary.csv` 与 `patch9_subset_report.json`。该脚本用于筛选“通道数少、信息覆盖完整、性能接近最优”的候选组合，再交给正式 CNN 复训。
- `android/analysis/audit_android_contact_bias.py`：新增接触面积偏置审计脚本，读取清洗 manifest 后自动按最新 `created_at` 回看最近窗口，统计二分类和四分类 CNN 预测在轻/重触维度上的混淆、大接触轻触误判、接触半径/面积/冲击 delta 与重触输出的相关性，并导出最近样本与“大接触轻触被判重触”清单，便于定位模型是否过度依赖半径或面积。
- `android/analysis/run_android_touch_window_ablation.py`：新增 Android 全量原始数据时间窗口消融脚本，读取 `ALL_NATIVE_NO_FILTER_manifest.csv` 后对 `pre/post` 连续时间窗执行多 seed 的 diagonal LDA probe；默认使用 `enhanced_patch22` 特征，缺失帧以零值补齐，确保有效和无效口径之外的全部已落盘原始样本都能参与窗口比较，并输出 `window_ablation_runs.csv`、`window_ablation_summary.csv` 与 `window_ablation_report.json`。
- `android/analysis/train_android_touch_cnn.py`：训练导出脚本支持全量原始 manifest 的 `raw_original_path` 等路径字段，并新增缺失帧处理策略；默认用零值补齐缺失帧，使 CNN 训练口径可与全量窗口消融保持一致。当前已用该脚本导出前五窗口 Android 22 通道 CNN 权重和 scaler，并写入 `aispect_impact_models` assets。
- Android 数据分层规则：所有手机原始 JSON 必须保留；清洗脚本只生成 `high_quality/review/rejected` 标签和 manifest，不删除原始样本。JSON 损坏、窗口帧缺失、多指、拖动/取消等样本会被标记为复核或 rejected；设备能力弱、触摸半径缺失、归一化无效、重触样本过短、大幅滑移或冲击异常的样本也只做质量标注。类别不均衡时可额外生成 `balanced_train_manifest.csv` 作为训练视图，但不得替代全量原始数据。

## 原始触摸数据统一去重补充
- `android/analysis/extract_unique_touch_data.py`：扫描 `data/lemon/lemon`、`data/tao/tao`、`data/collections/*/raw` 以及 `data/` 下的 `tar/tar.gz/tgz` 归档，将原始触摸 JSON 汇总到统一目录。目录源优先、归档源随后按归档原生顺序流式读取，避免压缩归档随机回跳导致重复解压。
- `run_extraction`：使用样本顶层 `id` 作为主唯一键、原始 JSON 字节的 SHA-256 作为次唯一键。相同 ID 和哈希的后续候选记为重复；相同 ID 但哈希不同的候选记为冲突并拒绝覆盖；无 ID、非法 JSON 或非对象 JSON 不进入目标目录。
- `iter_destination_candidates`：每次运行前先索引目标目录。若中断或并发曾留下内容完全相同的目标副本，保留路径较短的确定性版本并删除冗余副本；若目标内部出现同 ID 不同内容，则立即停止，禁止自动选择数据版本。
- `data/deduplicated_raw/<collector>/<device>/<date>/`：统一数据按采集者、设备和本地日期分层保存，路径字段转换为小写 ASCII slug；缺失采集者时按 legacy 数据目录或 collection 名称回退，缺失设备或日期时使用明确的 unknown 分类。
- `data/deduplicated_raw/manifest.csv`：记录每个保留样本的 ID、SHA-256、采集者、设备、日期、来源和相对目标路径。`duplicates.csv` 记录所有未复制候选及其原因与已保留目标；`summary.json` 保存目录数、归档数、发现量、保留量、重复量、冲突量和无效量。
- 当前完整执行扫描了 `6` 个展开目录、`6` 个归档和 `178770` 个源候选，统一保留 `89385` 条样本；其中 lemon `17872` 条、tao `43817` 条、zhuo `27696` 条。独立逐文件复核确认 `89385` 个唯一 ID 与 `89385` 个唯一 SHA-256 完全对应，ID 冲突和无效样本均为 `0`。

## Clean V3 清洗、严格留出与候选训练补充
- `android/analysis/build_deduplicated_clean_v3.py`：只读取 `data/deduplicated_raw/manifest.csv` 引用的 `89385` 条 JSON，不扫描目录中的报告 JSON。脚本提取四分类标签、采集者、设备、日期、原始 session、触摸/IMU 窗口和质量指标，原始 JSON 保持只读。
- Clean V3 身份规则：`data/lemon/lemon`、`data/tao/tao` 与 `data/collections/*/raw` 来源目录是采集者和 collection 的权威来源，JSON 内 `collector.userName` 仅作审计字段。全部 `89385` 条均匹配到唯一来源目录，其中 `26955` 条内嵌采集者与来源目录不一致；训练分组使用来源目录身份，避免把 7 月 22 日交叉采集错误归到设备内保存的用户名。
- Clean V3 质量规则：上下文按设备、运动状态、手指和轻重标签分组，触摸位移、冲击上限和触摸持续帧使用 median、MAD、p95 与 p05 计算，不再使用全设备统一的“重触少于四帧”规则。非法标签、不可用冲击窗口、多指和拖动/取消样本拒绝训练；半径缺失、归一化回退和上下文异常进入 review 并使用 `0.2~0.6` 有界权重。
- Clean V3 全量结果：`84778` 条 high quality、`2729` 条 review、`1878` 条 rejected，共 `87507` 条质量可训练样本。session 内按 `0.25s` 去除连续冗余后保留 `79113` 条，减少 `8394` 条；全部样本分为 `147` 个 session。
- `android/analysis/build_clean_v3_folds.py`：优先使用 `collectionSessionID`，旧数据缺失时按 collection 内相邻样本超过 `60s` 切分 session。生成 `5` 个采集者-设备 domain folds、`6` 个 collection folds与 `1` 个 temporal-tail fold；所有 fold 四类完整，检查确认没有 session 同时进入 train 和 valid。
- `train_android_finger_four_class_cnn.py` / `baseline_imu`：新增 11 通道候选，保留归一化坐标，并将九个 IMU 通道变换为相对当前样本前五帧中位数与有界噪声尺度的变化量；不读取标签、其他样本、未来样本或绝对接触尺寸。
- `android/analysis/run_clean_v3_probe_experiments.py`：可恢复执行 `old13`、`patch21`、`relative_touch`、`baseline_imu` 四候选乘五个严格 domain folds，使用 30 epoch、patience 8、batch 128、seed 22 和训练折 collection-class 平衡权重。20 个 probe 均已完成，报告位于 `models/deduplicated_clean_v3_20260728/`。
- 严格 domain probe 结果：`baseline_imu` 平均 balanced accuracy 约 `42.62%`、最差折约 `32.64%`；`patch21` 约 `42.52% / 25.77%`；`old13` 约 `40.36% / 33.34%`；`relative_touch` 约 `39.37% / 27.84%`。最佳候选未达到平均 `70%`、最差折 `55%`、最差类别召回 `45%` 的门槛，多数 fold 在较早 epoch 达峰，因此停止 120 epoch 多 seed 扩展，不生成或安装新的 Android 默认模型。
- `AispectImpactCNNModelCatalog.json`：为真机人工对照加入五个 Clean V3 `patch21` domain-holdout probe，分别对应 lemon-Xiaomi、tao-LE2110、tao-PHK110、zhuo-LE2110、zhuo-PHK110 留出域；模型只作为可切换实验项，catalog 第一项和 `crossDevice` 的 `DEFAULT_MODEL_ID` 继续保持 `full_old13_seen_devices_current_preprocessed_all_118`，不会因实验模型指标不足而自动替换默认模型。

## Clean V2 旧流程全量复训补充
- `test_train_android_first_touch_mlp.py`：定义按下期首帧实验模型的训练契约，要求训练端只读取 `touchFrames[0]`，使用十个可在 Android `ACTION_DOWN` 同步重建的无量纲特征，并验证缺失接触椭圆时使用 `0.012` 默认 sigma、分层切分可重复、导出 MLP 权重尺寸可被端侧校验。
- `train_android_first_touch_mlp.py`：新增独立首帧四分类 MLP 训练入口，读取 Clean V2 manifest 的全部合格样本和 `sample_weight`，提取归一化位置、压力、size、接触椭圆 sigma、对数长宽比、对数面积与方向正余弦十特征；支持特征缓存、随机分层验证和全量拟合，导出单隐藏层权重、scaler、标签顺序及训练报告，不修改现有抬起期 CNN 管线。
- `test_train_android_first_touch_mlp.py`：新增全量拟合状态测试，要求无验证集时每个 epoch 都更新导出状态，确保指定 72 epoch 时实际导出第 72 轮，而不是按训练集分数回选较早轮次。
- `should_update_best_state`：首帧训练器在存在验证集时仍按 balanced accuracy 保存最佳轮次；`split_strategy=all` 时无条件更新为当前 epoch，使全量模型严格使用验证阶段选出的训练轮数。
- `AispectFirstTouchClassifierTest`：定义 Android 首帧分类器契约，逐值核对十个训练特征、标准化后单隐藏层 ReLU MLP 的四分类 softmax 输出，以及模型资源不可用时返回 `null` 而不影响现有抬起推理。
- `AispectFirstTouchClassifier`：新增可复用首帧 MLP 推理器，从独立 JSON asset 读取 scaler、单隐藏层权重和四类顺序；端侧直接从 `AispectTouchFrame` 与单帧 `AispectContactPatch` 重建训练端十特征，资源或尺寸无效时返回空结果，不耦合现有 CNN catalog。
- `AispectCollectionController.collectTouchFrames`：`ACTION_DOWN` 当前帧落入 `activeTouchFrames` 后立即调用首帧分类器，以 `collector_early_prediction_ready` 发布一次实验早期预测；抬起后的 `finalizeCompletedTouch` 仍运行当前选中 CNN 并覆盖为最终结果，早期事件不满足完成事件条件，因此不会进入现有最终召回计数。
- `models/first_touch_early_v1_20260731`：保存全部 `79878` 条 Clean V2 样本的首帧特征缓存、随机验证模型与 epoch `72` 全量模型。随机 `25%` 验证 balanced accuracy 约 `47.99%`，四类召回约为 `thumb_light 28.25% / thumb_heavy 53.76% / index_light 69.08% / index_heavy 40.86%`；该结果仅高于随机基线，明显低于抬起期模型，因此只作为按下瞬间实验反馈，不替换或参与选择最终 CNN。
- `AispectFirstTouchModel_first_touch_early_v1.json`：将首帧全量 MLP 作为独立 asset 导入，SHA-256 与训练产物一致；模型 ID 不进入现有 CNN catalog，避免用户模型切换同时改变最终抬起模型。
- 首帧真机时序验证：在 OnePlus LE2110 `aa95546c` 上执行 10 秒持续按压，按住第 1 秒截图显示 `first_touch_early_v1`，抬起后截图显示 `full_data_clean_v2_old13_all_118`，证明早期事件发生在抬起前且最终事件正常覆盖。该样例人工标签为拇指轻点，早期模型误判拇指重点 `0.7996`，最终模型正确输出拇指轻点 `0.6327`；截图保存在 `models/first_touch_early_v1_20260731/device_early_press.png` 与 `device_final_press.png`，用于明确记录原型能力边界。
- `test_first_touch_ui_deployment.py`：新增按下期模型可见性回归契约，要求控制器公开首帧模型 ID，面板常驻显示“按下期默认模型”，并把原模型切换区域明确标注为“抬起期 CNN 模型切换”，避免独立 MLP 未进入 CNN catalog 时被误认为没有安装。
- `AispectFirstTouchClassifier.modelId` / `AispectCollectionController.firstTouchModelId`：新增只读首帧模型标识接口，不允许通过 CNN 模型切换器修改独立 MLP。
- `MainActivity` 模型面板：常驻显示“按下期默认模型”及 `first_touch_early_v1`，原 catalog 控件标题改为“抬起期 CNN 模型切换”；用户无需持续按住才能确认早期模型已安装，两个不同格式的模型仍保持独立生命周期。
- `test_latest_model_deployment.py`：crossDevice 最新模型部署契约改为检查 `full_data_clean_v2_old13_all_118` 必须同时成为 flavor 默认模型与 catalog 第一项，并要求对应 scaler、weights 资产实际存在；部署实现前该测试用于产生明确失败，避免只修改显示名称或只复制单个文件。
- `AispectImpactCNNModelCatalog.json` / `crossDevice` 产品变体：新增并置顶 `full_data_clean_v2_old13_all_118`，导入同名 scaler 与 weights，并仅将 crossDevice 的 `DEFAULT_MODEL_ID` 切换到该模型；standard flavor 默认模型与全部旧模型保持不变，便于真机同会话 A/B 测试。
- 当前 `data/deduplicated_raw/manifest.csv` 与旧流程输入 `full_all_collections_manifest.csv` 均为 `89385` 条、`89385` 个唯一 ID，双向 ID 差集均为 `0`；因此本轮不存在尚未进入旧流程的新样本，不重复清洗或复制原始 JSON。
- 继续使用旧流程已经生成的 grouped Clean V2 训练视图，共 `79878` 条且 ID 不重复：`44058` 条 high quality、`8275` 条 review、`27545` 条 legacy unscored；四类分别为 `thumb_light 22862 / thumb_heavy 20856 / index_light 18101 / index_heavy 18059`，全部路径存在。
- `models/full_data_clean_v2_old13_20260728/validation_120`：完全复刻 old13、`-5...+9`、缺帧补零、质量样本权重与类别权重、batch `256`、学习率 `0.001`、weight decay `0.0001`、seed `22`、随机 `25%` 验证和 `120` epoch。最佳 epoch 仍为 `118`，验证平衡准确率约 `79.14%`，与历史同口径 `79.15%` 相差约 `0.007` 个百分点；四类召回约为 `thumb_light 78.22% / thumb_heavy 74.43% / index_light 79.93% / index_heavy 83.99%`。
- `models/full_data_clean_v2_old13_20260728/final_all`：使用验证阶段选出的 `118` epoch 对全部 `79878` 条合格数据重新训练，模型 ID 为 `full_data_clean_v2_old13_all_118`。全量训练平衡准确率约 `85.82%` 只用于确认拟合状态，不视为验证或真机准确率；本轮不修改 Android catalog 和默认模型，待同一手机同一操作会话完成旧模型与该模型 A/B 测试后再决定是否接入。

## 历史按下锚点数据集补充
- `android/analysis/build_historical_down_anchor_datasets.py`：只读扫描 `data/deduplicated_raw/manifest.csv` 指向的去重原始 JSON，分别生成 A、B 两套按下锚点数据集，不修改任何原始 JSON。按下时间优先读取 `touchFrames[0].eventTimeMillis`，缺失时回退到 `touchEvents[0].timestamp`。
- 真实 IMU 帧判定：`delta/x/y/z`、三轴陀螺仪、三轴重力与 `x_norm/y_norm` 中至少一个信号字段非零才视为真实观测帧；所有信号字段均为零的补位帧不参与覆盖范围、帧数与连续性判定。
- 数据集 A `a_down_inside_saved_window`：真实 IMU 帧按绝对时间排序后，最早帧时间不晚于按下时刻且最晚帧时间不早于按下时刻。A 仅保存 `manifest.csv` 与统计摘要，保留源 JSON 的绝对路径和按下定位诊断，不复制源文件。
- 数据集 B `b_complete_down_minus5_plus9`：必须先属于 A；选择距离按下时间最近的真实 IMU 帧作为 0 帧，要求其前方至少 5 帧、后方至少 9 帧，0 帧误差不超过 `0.75` 个采样周期，选中窗口相邻帧间隔不超过 `1.75` 个采样周期。B 因而严格是 A 的子集，且不允许补零、插值或外推。
- B 派生 JSON：写入 `sourceId`、`sourcePath`、`sourceSchemaVersion`、`label`、`downAnchorTimestamp`、`downReferenceFrameIndex`、`sampleRateHz` 和 `downFrames`。`downFrames` 固定为 15 个真实帧，重新编号为 `-5...+9`，同时保留 `sourceFrameIndex`、`sourceRelativeTimestamp` 与 `sourceTimeSinceTouchDown` 供审计。
- A/B `manifest.csv`：共同记录样本 ID、源 SHA-256、源路径、schema、采集者、设备、日期、session、类别、按下时间及来源、原锚点时间、采样率、虚拟按下帧位置、最近源帧、定位误差、按下前后真实帧数、真实帧总数与补位帧数；B 额外记录派生 JSON 相对路径。各级 `summary.json` 汇总保留数、排除原因、schema、采集者、设备和类别分布。
- 构建命令：`python android/analysis/build_historical_down_anchor_datasets.py --source-manifest data/deduplicated_raw/manifest.csv --output-dir data/down_anchor_historical_20260731`。输出先写入同级临时目录，成功后整体发布，避免中断时留下半成品结果。

## Impact Dashboard GitHub Pages 部署补充
- `docs/impact_dashboard/index.html`：作为 GitHub Pages 发布副本，复用本地 `tools/impact_dashboard/dashboard.html` 的静态内容，并将下载链接改为相对路径，确保项目站点子路径下可正常访问 `latest_samples.csv`。
- `docs/impact_dashboard/data`：保存发布用 `latest_samples.json` 与 `latest_samples.csv` 静态数据副本，避免 GitHub Pages 直接暴露工具目录，便于后续独立替换数据文件。
- `docs/index.html`：仓库 Pages 根入口，自动跳转到 `impact_dashboard` 子目录，降低访问路径记忆成本。

## 后续扩展建议
- Aispect 事件路由与模块绑定建议单独抽出 `InteractionRouter`，避免与 UI 组件耦合。
- 消息列表与侧栏建议加入虚拟化计算与滚动监听，满足大规模会话与消息量的性能目标。
- 模块化接口已预留 Live Activity、PiP、Todo、知识库等能力的最小入口，可按模块逐步填充。

## Android 正式构建合同
- Android 构建在项目的 `android` 目录中执行，由 Windows Java 直接加载 `gradle\wrapper\gradle-wrapper.jar`，不依赖 Bash 或 WSL。
- 标准调试 APK 固定执行 `java -classpath gradle\wrapper\gradle-wrapper.jar org.gradle.wrapper.GradleWrapperMain :app:assembleStandardDebug --no-daemon --max-workers=1`。
- 仓库内只有 Unix 版 `android/gradlew`，且为 CRLF 换行；Bash/WSL 可能把路径附带 `\r`，同时当前 WSL 未配置 Java。因此 `android/gradlew` 不是正式构建入口，其失败也不能直接归因于 Android 源码。

---

修改日志：
- 2026-08-11：固定 Android 正式构建合同为 Windows Java 直接启动 Gradle Wrapper，并强制 `--no-daemon --max-workers=1`；明确禁止把 Bash/WSL 与 Unix `android/gradlew` 作为正式构建路径。
- 2026-08-09：将 Android 旧 Contact Patch/归一化/时间特征读取集中到 `aispect/legacy/AispectLegacyFeatureBuilder`，主分类器只按 feature contract 分派；causal 合同继续仅使用双时间门 ZOH 与逐帧 touchFrames。
- 2026-08-04：Android 控制器创建时、iOS 页面可见时预热 IMU 滚动缓存，确保验证模式在 `ACTION_DOWN` 前已有 `-5` 帧；页面销毁或离开时停止传感器。
- 2026-08-04：修正 iOS 传感器不可用或停止时仍能结束当前原始采集会话，保留 `windowComplete` 真实状态而不丢失会话收尾。
- 2026-08-04：移除 iOS 旧冲击回调与旧触摸窗口状态变量，避免按下锚点采集链路再次合成重触输出。
- 2026-08-04：补齐 iOS 按下窗口就绪状态回调和统一本地化文案；旧 iOS CNN 推理器改为显式未就绪占位，保留历史资源但不再计算旧特征。
- 2026-08-04：按下/抬起双锚点采集正式统一为 schema v5；Android 与 iOS 均以 `ACTION_DOWN` 为 `frameIndex=0`，等待真实 `-5...+9` 15 帧窗口并只发布一次“模型待训练”状态，抬起后独立补采 `lift+15` 保存原始序列。本轮不重新训练、不启用旧模型。
- 2026-08-04：新标准 IMU JSON 移除 `delta` 与 `maxAbsDelta` 派生字段，保留原始加速度、陀螺仪、重力、触摸归一化与时间对齐字段；旧数据、旧模型资源和历史分析目录不修改、不混入新标准训练集。
- 2026-07-28：将五个 Clean V3 `patch21` domain-holdout probe 的 scaler/weights 加入 Android 模型 catalog，保留现有默认模型，用于 `crossDevice` App 在 LE2110 真机上手动切换对照。
- 2026-07-28：基于 `89385` 条唯一原始数据建立 Clean V3，修正 `26955` 条采集者元数据冲突，生成 `87507` 条质量训练视图、`79113` 条 session 去冗余训练视图和 12 个无 session 泄漏严格 folds；完成 20 个四候选 domain probe，因跨域指标未达门槛而不替换端侧模型。
- 2026-07-27：新增原始触摸数据统一去重提取器，将 legacy、四个 collection 展开目录及六个归档汇总到 `data/deduplicated_raw`；按 ID 与 SHA-256 双重去重后保留 `89385` 条唯一 JSON，并生成 manifest、重复审计和汇总报告。
- 2026-01-23：初始化 UIKit 主容器、核心数据模型、Aispect 触摸骨架与基础 UI 组件结构。
- 2026-01-23：加入虚拟列表组件与 Aispect 采样平滑策略，侧栏列表改为滚动驱动的可视化复用。
- 2026-01-23：补充虚拟列表在滚动与布局变化时的可见项位置刷新。
- 2026-01-23：连线系统改为基于虚拟布局快照计算锚点坐标。
- 2026-01-23：输入框增加浮标模式切换与建议文本展示接口，并接入 Aispect 示例路由。
- 2026-01-23：新增悬浮式主页容器组件并接入主界面布局。
- 2026-01-23：悬浮容器补充内容裁剪与连续圆角一致性配置。
- 2026-01-23：新增详细气泡与气泡菜单占位，实现未回消息页与顶部按钮切换。
- 2026-01-23：气泡菜单条加入玻璃材质与点击淡入淡出动画。
- 2026-01-23：未回消息页加入布局快照与可见索引计算，连线可基于该页生成。
- 2026-01-23：未回消息页使用详细气泡样式，右侧工具列加入滚动驱动的高度联动与自动回弹。
- 2026-01-23：虚拟列表增加滚动回调，用于侧栏联动控制。
- 2026-01-23：右侧流程列加入滚动挤压联动，气泡菜单按钮加入图标化样式。
- 2026-01-23：右侧账号列加入滚动挤压联动，三列高度按比例切换。
- 2026-01-23：左侧会话头像支持滚动边界固定展示，避免当前会话丢失。
- 2026-01-23：Aispect 手势允许与系统手势同时识别，避免阻断点击与滚动。
- 2026-01-23：Aispect 手势改为不拦截触摸，避免阻断 UI 点击。
- 2026-01-23：左侧会话头像增加停留插入按钮的占位挤压逻辑，并与滚动联动重置。
- 2026-01-23：新增玻璃材质背景组件与 MeshGradient 背景层，顶部/底部/侧栏与气泡改用语义化材质渲染。
- 2026-01-23：Aispect 增加触摸采样事件，MeshGradient 背景与悬浮容器支持触摸压力联动形变。
- 2026-01-23：补充 Aispect 事件路由与热区绑定，长按进度条与重触控制占位视图落地。
- 2026-01-23：项目最低系统版本调整为 iOS 18.6。
- 2026-01-23：补充默认开发团队配置，修复命令行构建签名失败。
- 2026-01-23：开发团队 ID 与本机签名证书一致化，便于自动签名。
- 2026-01-23：开发团队 ID 与本机配置文件一致化，便于自动签名。
- 2026-01-23：修复 MeshGradient 背景编译依赖，补充 Combine 引用与辅助函数整理。
- 2026-01-23：关闭预览构建以避免设备构建的调试动态库签名失败。
- 2026-01-23：长按进度完成触发回调接入，补齐轻触长按触发入口。
- 2026-01-23：重触长按控制视图接入拖拽更新与回弹重置逻辑。
- 2026-01-23：边缘拖拽热区识别与动作回调接入，作为可复用触发入口。
- 2026-01-23：新增触摸半径可视化叠层，用于验证 majorRadius 采样与重触阈值。
- 2026-01-23：下调重触默认倍率与最小基线值，降低重触阈值门槛。
- 2026-01-23：Aispect 采样支持 coalesced touches 与 estimated updates，提升半径变化捕获率。
- 2026-01-23：重触阈值首次命中时立即派发事件，降低延迟感。
- 2026-01-23：加入重触显眼提示视图，并进一步下调默认阈值参数。
- 2026-01-23：新增触摸半径日志记录器，便于统计 majorRadius 变化与阈值分布。
- 2026-01-23：日志记录改为不做额外节流，避免遗漏系统已节流后的样本。
- 2026-01-23：日志写盘频率提高，并在前后台切换时强制 flush。
- 2026-01-23：新增 majorRadius 最小测试面板，便于快速验证数值变化。
- 2026-01-23：测试面板改为使用 intrinsicContentSize，避免约束循环导致卡顿。
- 2026-01-23：Aispect 事件补充 majorRadiusTolerance，并在日志与调试视图中可视化输出。
- 2026-01-23：Aispect 事件补充 force/maxForce，并在调试面板与日志中输出显示。
- 2026-01-23：移除 TouchVisualizer 调试模块，避免引入全局 swizzle 依赖。
- 2026-01-23：majorRadius 调试面板显示原始精度数值，避免被 UI 侧截断影响判断。
- 2026-01-23：调试面板新增 moved-only 的 majorRadius 展示，用于排除 Began 采样误导。
- 2026-01-23：调试阶段将半径平滑关闭并提升采样频率，确保 raw/moved 更接近系统原始值。
- 2026-01-23：重触阈值默认调整为 12.0852，便于观察重触触发行为。
- 2026-01-23：引入半径变化率判定，用于捕捉重触的瞬时增长。
- 2026-01-23：测试面板加入 rate 阈值滑条，可实时调节变化率触发门槛。
- 2026-01-23：变化率显示精度提升，便于观察微小波动。
- 2026-01-23：新增 coalescedTouches 数量展示与日志记录，用于验证采样密度变化。
- 2026-01-23：加入基于加速度计的冲击检测，用于在触摸窗口内补偿重触触发。
- 2026-01-23：冲击检测补充 QuartzCore 依赖以使用高精度时间戳。
- 2026-01-23：测试面板新增冲击阈值滑条，便于快速调整加速度触发灵敏度。
- 2026-01-23：测试面板新增触觉反馈样式选择，并在重触触发时使用选定手感反馈。
- 2026-01-23：冲击采样率统计并显示在测试面板，便于验证实际采样频率。
- 2026-01-23：冲击采样间隔提升至 240Hz 目标值，用于测试系统上限。
- 2026-01-23：测试面板新增冲击 delta 窗口日志显示（前 5 帧、后 15 帧）。
- 2026-01-23：冲击日志改为延迟收集，确保完整显示前 5 帧与后 15 帧。
- 2026-01-23：冲击日志补充基线采样与零填充，避免前置帧为空白。
- 2026-01-23：冲击日志改为增量更新，显示 post remaining 以便观察收集进度。
- 2026-01-23：冲击检测引入轴向权重，默认 X 权重为 0.01、Y 权重为 1.0、Z 权重为 6.5 以增强按压判定。
- 2026-01-23：冲击窗口日志改为输出带正负号的 delta，便于观察方向变化。
- 2026-01-24：移除 majorRadius 全链路与调试可视化/日志，重触判定统一切到加速度冲击检测，测试面板精简为 impact 参数与日志。
- 2026-01-24：删除 AispectRadiusDebugView 与 AispectRadiusLogger 文件，清理旧半径调试残留。
- 2026-01-24：新增 liftOffset 结束偏移与 heavyTap 辅助判定，测试面板展示抬起偏移。
- 2026-01-24：heavyTap 触发时若已发生 heavyPress 则跳过重复触觉反馈。
- 2026-01-24：补充 AispectTouchEvent 显式初始化器，确保 liftOffset 传参与编译一致。
- 2026-01-24：移除 liftOffset 的默认值，修正只读属性重复初始化报错。
- 2026-01-24：impact delta window 输出每帧 delta 与 x/y/z/magnitude，并在测试面板增加 copy log 按钮。
- 2026-01-24：为冲击检测与 liftOffset 判定补充注释，说明触发窗口与重复触觉防护。
- 2026-01-24：更新 AGENTS 规则并将新增注释统一为中文注释。
- 2026-01-24：补充 Aispect 冲击采样、触摸事件流、测试面板与路由关键代码的中文注释。
- 2026-01-24：新增 ARKit 闭眼检测模块与测试面板延迟/状态显示，并补充相机权限说明。
- 2026-01-24：补充入口、连线层与测试面板的中文注释，统一注释规范。
- 2026-01-24：轻点触发闭眼检测时重置面板延迟与状态显示。
- 2026-01-24：闭眼状态文案改为中文资源，面板显示“检测到闭眼/未检测到闭眼”等清晰状态。
- 2026-01-24：测试面板新增闭眼结果行，用于明确是否检测到闭眼。
- 2026-02-14：修复 AispectBlinkDetector 配置字段拼写错误（`blinkHoldDurantion` -> `blinkHoldDuration`），消除编译报错。
- 2026-04-04：接入基于 CNN 的冲击二次分类器，保留 delta 阈值候选触发并在手机端使用本地权重/归一化参数完成重触识别。
- 2026-04-04：新增四类真值标签的数据采集模块与测试面板控件，支持日常采集 `static/walking × light/heavy` 训练样本并写入本地 JSON 数据集目录。
- 2026-04-05：将采集模块改为任务式交互，采集中锁定标签、丢弃拖动与取消事件、按真实触摸时间对齐窗口，并补齐冲击测试面板中文文案。
- 2026-04-05：采集模块新增系统侧活动证据记录，接入 `CMMotionActivityManager` 并把 activity/confidence、采集 session ID、App 版本信息写入样本，同时在面板显示系统活动摘要并补充运动权限说明。
- 2026-04-05：修复采集器启动阶段同步扫描历史数据导致的界面卡死问题，改为后台异步汇总本地样本统计。
- 2026-04-05：修复测试面板在 iPhone 上不可见且无法滚动的问题，新增面板滚动查看与收起为小按钮的交互。
- 2026-04-05：修复测试面板整体超出屏幕导致滚动回弹异常的问题，按安全区限制展开态最大高度。
- 2026-04-05：修复主页背景未占满全屏、上下安全区出现空白的问题，主容器改为全屏铺满并由内部内容自行遵守安全区。
- 2026-04-05：将主页布局改为“全屏视觉层 + 独立安全区内容 guide”，并为全屏容器切换到 `edgeToEdge` 样式，避免装饰性容器承担安全区职责导致的边缘留白与缩边问题。
- 2026-04-05：将背景层统一改为纯黑底，关闭彩色 MeshGradient 底色，确保全屏背景视觉一致。
- 2026-04-05：将根控制器与 SwiftUI 入口宿主层同步改为黑底，修复安全区仍然暴露白色背景的问题。
- 2026-04-05：训练数据采集的运动标签扩展为“手持静止 / 放置静止 / 走路”，并兼容历史 `static` 标签映射到手持静止。
- 2026-04-07：新增 `docs/impact_dashboard` GitHub Pages 发布目录与根入口跳转页，并修正发布副本里的数据下载相对路径。
- 2026-04-12：将 iOS App 目录、Xcode 工程、Target、产品名、Bundle Identifier 与 SwiftUI 入口从 `zhifaIOS` 统一重命名为 `ZHIFAeyes`，便于后续以新品牌名继续开发与打开工程。
- 2026-04-12：闭眼检测主链从 ARKit 切换为 Vision + AVCaptureSession，新增低功耗前置相机流水线、ARKit 兜底分支、权限拒绝文案与相机用途说明更新。
- 2026-04-12：闭眼检测新增页面级预热、低帧率待机与触发窗口升频机制，减少重复冷启动带来的启动时延与待机功耗。
- 2026-04-12：闭眼检测补充预热、首帧、锁脸、判定四个关键延迟埋点，并把真机调试面板升级为多阶段性能观测视图。
- 2026-04-12：根据真机实测“总在超时前未命中”的现象，下调闭眼保持时长、提高检测窗口帧率、加入开眼回滞与更轻平滑，修正 Vision 版闭眼判定过于保守的问题。
- 2026-04-12：继续把闭眼检测向低延迟极限推进，加入强闭眼快速通道、缩短常规保持窗口、降低基线样本门槛，并移除每帧主线程方向读取的调度开销。
- 2026-04-12：修复 Vision 检测态升频对设备帧率区间过于苛刻的问题，改为自动协商最接近的受支持帧率，避免检测会话意外停留在低帧率待机档导致持续漏判。
- 2026-04-12：为闭眼检测关键阈值与保持时长补充面向人工微调的中文注释，便于在 Xcode 内直接逐项试参。
- 2026-04-12：加入闭眼排查版调试链路，默认关闭强闭眼快速通道、把判定融合改为原始值优先的加权平均，并在测试面板显示实时基线与比值快照。
- 2026-04-12：修复预热期基线容易被异常 eye landmarks 污染的问题，新增基线异常值过滤、坏基线重置、向上跳变限制与向下快速恢复逻辑。
- 2026-04-12：将 Vision 眼部开合度从包围盒高宽比切换为点距 `EAR`，减少眼轮廓异常点对基线与闭眼判定的破坏。
- 2026-04-12：修复预热阶段错误复用路过人脸基线的逻辑缺陷，正式检测改为仅基于当前点击后的样本重新建立开眼基线。
- 2026-06-04：新增 Android 移植工程，优先完成非声学采集链路，接入 Android 触摸半径/压力/面积/方向字段的标准 axis 兼容读取、设备能力分级、IMU 窗口、CNN 诊断与 JSON 样本落盘。
- 2026-06-05：将桌面 `zhifaIOS_old.app` 中可迁移的旧触控 CNN 模型资源接入 Android 工程，示例 App 更名为旧触控安卓适配版，并让 Android 诊断优先使用旧包 13 通道增强模型。
- 2026-06-05：Android 示例 App 补齐旧 iOS 触控采集面板格式，新增真实 CNN 模型 catalog 切换、采集开关任务锁定、模型/标签状态日志与数据目录复制，并让样本诊断优先使用当前手动选择的模型。
- 2026-06-05：Android 示例 App 从表单式采集页改为旧 iOS 式全屏主界面与左上悬浮冲击测试面板，训练采集改由全屏非面板区域承接；同时把触摸采样间隔调到约 250Hz、拖动判定半径放宽到 28dp、重触抬起偏移阈值降到 4dp，提升小米真机采集灵敏度并减少轻微手抖误丢样本。
- 2026-06-10：补充 Android 触摸训练级消融脚本，基于当前小米均衡训练集对旧 13 通道、半径、协方差、多通道、高斯热图摘要、高斯 9x9 热图与增强 22 通道输入进行多 seed probe 对比，并生成 ablation 报告。
- 2026-06-11：Android 采集落盘改为全量原始事件保存，未来拖动/取消等样本也会写入同一数据目录，避免界面计数和可导出 JSON 数量不一致。
- 2026-06-11：新增 Android 全量原始数据时间窗口消融脚本，基于 9185 条手机原始 JSON 比较 `pre/post` 时间窗，并输出前五窗口排序报告。
- 2026-06-12：基于全量原始数据训练并接入前五时间窗口 22 通道 CNN 模型，Android 手机端支持采集开关打开时在两次触摸之间热切换模型，并默认选中 `-5...+9` 第一名窗口。
- 2026-06-05：继续细化 Android 对 iOS `AispectImpactTestView` 的复刻，面板展开/收起尺寸、采集 switch、冲击阈值 slider、实时事件行与冲击窗口日志对齐 iOS 交互；全屏采集改为按触摸起点锁定整轮事件，IMU 目标采样周期调整到 4167us。
- 2026-06-05：修复小米 Android 15 点击训练采集开关即闪退的问题，新增高频传感器权限与 IMU 采样率降级兜底；Android 采集链路新增三种触摸归一化方案选择，并把三套结果、设备内分位、短期历史偏移与字段质量标记同时写入训练样本 JSON。
- 2026-06-05：修复 Android 悬浮测试面板收起/展开不灵敏的问题，收起态改为 132x64dp 可点击小面板，展开态顶部 58dp 整条头部均可触发收起，并在事件分发入口增加外扩 36dp 的优先热区。
- 2026-06-06：Android 示例 App 新增重触视觉与振动反馈，依据实时触摸半径/面积估算强度并映射到触摸环大小、暖色冲击闪烁和短振动。
- 2026-06-06：新增 Android 真机数据清洗与 CNN 训练导出脚本，将 471 条小米样本分为优质、复核、剔除三层，并生成类别均衡训练清单，明确旧 CNN 输出只作诊断而不作为训练真值。
- 2026-06-08：新增接触区域三方案离线比对脚本，按二维高斯热图、多通道表示、协方差矩阵三类 CNN 输入重建已采集样本，确认当前 JSON 数据保留了后续比对所需的触摸半径、中心、方向、有效性与接触 patch 信息。
- 2026-06-08：Android 面板将旧 CNN 输出改为诊断文案并显示当前训练标签，轻点采集任务开启时关闭重触振动与红色冲击反馈，避免轻点采集被旧模型或即时反馈误读为重点识别。
- 2026-06-08：Android 训练采集进一步改为按面板锁定标签重写触摸结束事件和 `sourceTouchKind`，关闭旧 CNN 实时诊断，清理训练背景无关 UI，并把收起/测试切换限制到明确按钮区域以降低误触。
- 2026-06-08：修复 Android 悬浮面板滚动后收起按钮位置漂移的问题，切换面板时不再单独改变按钮层级，保证按钮稳定停在面板顶部。
- 2026-06-08：Android CNN 训练入口从旧 `old13` 默认升级为 `enhanced_patch22`，推理器与采集控制器补齐 `contactPatch` / `touchNormalization` 输入传递，并新增增强触摸特征报告脚本。
- 2026-06-13：正式接入 Contact Patch 三方案 CNN 特征预设，补齐 Android 推理端 `gaussian.*`、`multi.*`、`covariance.*` 同名特征读取，新增正式 CNN 批量消融脚本，并在 9185 条全量 Android 原始样本上完成 12 项 CNN 消融汇总。
- 2026-06-14：为 Android 触摸归一化、手机端 CNN 特征读取和离线 CNN 训练脚本补充简洁中文注释，重点标明 Contact Patch 三方案、缺失值占位、`validMask` 与 `confidence` 的真实语义。
- 2026-06-16：Android 采集标签新增拇指/食指维度，面板增加手指类型选择并写入新样本 JSON 与文件名；清洗 manifest 增加 `finger_type`，同时扩大面板收起/展开热区。
- 2026-06-23：Android 验证模式支持在采集开关关闭时直接跑当前 CNN 四分类识别并显示人工标签对比、置信度、四类概率和四类召回统计；该模式不保存训练 JSON。
- 2026-06-13：新增 `patch9` 九通道组合穷举消融脚本，基于 9185 条全量 Android 原始样本、`-5...+9` 窗口和 3 个 seed 跑完 511 个非空子集，用于筛选少通道但信息覆盖完整的接触特征组合候选。
- 2026-06-22：新增历史 Android 样本拇指标记脚本，已将本地 9185 条唯一原始样本及其展开整理副本补齐为 `thumb`，并让清洗脚本对缺失手指字段的旧样本默认按拇指处理。
- 2026-06-22：从已连接小米手机导出 12131 条 raw 样本，识别其中新增 2946 条为食指样本；基于 `handheld_static` 场景构建四类各 733 条的拇指/食指轻重点均衡数据集，并训练 `enhanced_patch22` 四分类 CNN，验证集平衡准确率约 92.49%。
- 2026-06-22：完成拇指/食指轻重点四分类 CNN 全量消融 12 项；最佳 preset 为 `old13_raw_radius`，验证集平衡准确率约 92.90%、最小类别召回约 88.52%，`old13_multichannel` 与 `enhanced_patch22` 以约 92.49% 平衡准确率并列其后。
- 2026-06-23：将拇指/食指轻重点四分类消融前五模型打入 Android assets，并扩展手机端 CNN 推理器、预测结果结构、采集日志与面板展示，使模型切换功能可以直接切换四分类模型。
- 2026-06-23：将 `lemon + tao` 全量四分类 CNN 消融前五名导入 Android assets 并置顶 catalog，手机端默认使用新的 `lemon_tao_full4_enhanced_patch21_no_confidence` 模型。
- 2026-07-01：整理 GitHub 上传结构，新增根 README，将两人标签数据以压缩包归档到 `data/`，将全量四分类前五模型与消融报告归档到 `models/`，启用 Git LFS 跟踪数据压缩包、manifest CSV 与 CNN 模型资源，并忽略本机 Android 配置与完整消融中间结果目录。
- 2026-06-23：Android 采集面板新增四分类实时召回显示框，按本轮 accepted 样本实时累计四类召回、macro recall 与总体 accuracy；显示数据不再受训练数据采集开关清零影响，仅在切换 CNN 模型时重置。
- 2026-06-23：Android 手机采集阶段去除有效/无效样本判断，拖动、取消和普通点击都会按当前人工标签保存为 accepted 样本；界面计数同步改为保存数与写入失败数。
- 2026-06-23：修复去除有效/无效判断后同一次触摸可能由多个结束类事件重复落盘的问题，采集控制器新增单次触摸保存闸门，保证一次按压最多保存一条样本。
- 2026-06-23：Android 面板新增验证模式开关，每次触摸后显示人工标签、CNN 输出、对错、置信度与当前组合统计，支持采集时直接观察四分类准确性。
- 2026-06-23：Android 四分类消融训练脚本接入自动 CUDA/MPS/CPU 设备选择，Mac 本机可直接启用 MPS 加速，已完成 preset 可继续恢复跑剩余消融。
- 2026-06-23：修复四分类训练脚本调用 `train_one_epoch` 时未传样本权重的问题，保持四分类默认等权训练并恢复全量消融可运行性。
- 2026-06-23：新增 Android 接触面积偏置审计脚本，并基于 `f31039_20260623_phone_pull` 生成清洗、最近五分钟审计、二分类/四分类 CNN 消融、`patch9` 子集穷举和接触特征分布报告，用于排查大接触轻触被识别为重触的问题。
- 2026-06-23：修复 Android 四分类召回验证一次触摸可能累计多次的问题，控制器新增完成预测序号并在新触摸开始时清空旧预测，面板召回统计改为按预测序号去重；训练采集仍保持面板外触摸全量 accepted 保存。
- 2026-06-23：Android 示例 App 调整实时验证口径，验证模式打开后采集和非采集触摸都会持续累计四分类召回，采集开关、模型切换和任务切换不再清空累计显示。
- 2026-06-23：Android 面板隐藏常驻冲击窗口日志、复制日志按钮和接触编码明细数据，优先保留四分类验证结果、模型切换与训练采集控件显示空间。
- 2026-06-23：Android 采集界面移除振动、红色冲击闪屏和重触色彩暗示，训练 JSON 不再写入 `cnnPrediction`，并新增最近保存原生样本摘要显示。
- 2026-06-23：Android 面板将 CNN 诊断结果改为固定高度方框，并修正人工标签占位符与传参顺序，避免每次预测结果文本长短变化或格式化异常推动四分类数据框上下跳动。
- 2026-06-23：Android 采集教程 `1...5` 文案区域改为固定五行高度，避免采集开关切换锁定提示时继续推动下面的数据框。
- 2026-06-23：Android 二分类 heavy 线去掉 `confidence` 训练通道，接入大面积轻触 hard negative 加权，全量 high_quality 复训后回装 `android_patch21_no_confidence_pre5_post9`；端侧加入“冲击 delta + 形态或时间动态”证据约束。
- 2026-06-25：导出当前 OnePlus PHK110 手机 `3374` 条原生训练样本，生成 `oneplus_20260625_phone_pull` 清洗数据集；基于 `handheld_static` 四类构建 clean 与 hard 两套平衡 manifest，并完成 5 个核心 preset 的四分类 CNN 消融。
- 2026-06-25：将 OnePlus hard-core 消融第一名 `oneplus_four_class_hard_core_old13_raw_radius` 导入 Android assets 并置顶 catalog；该模型在 hard 平衡集上验证平衡准确率约 `97.35%`、最弱类召回约 `93.94%`，端侧四分类默认模型同步切换到该模型。
- 2026-06-25：针对手机实测食指/拇指同强度混淆，新增 OnePlus 四分类时间留出复核；确认随机切分因连续采集段泄漏而高估 `old13_raw_radius`，端侧默认四分类模型改为时间留出更稳的 `oneplus_four_class_hard_core_old13`，并保留 raw radius 版本作为第二项对照。
- 2026-06-25：补采后从 OnePlus PHK110 导出 `4641` 条原生 JSON，修正四分类数据构建脚本优先使用真实 `finger_type`，训练和消融脚本新增 temporal split；基于 hard_v2 每类 `321` 条样本完成 `25%` 与 `10%` 最新尾段时间留出全量消融，并将 `10%` 留出当前第一名 preset 的 nearfull 装机版 `oneplus_four_class_after_recollect_hard_nearfull_old13_raw_radius` 置顶 catalog 且设为端侧默认四分类模型，`temporal10` 训练版保留为第二项对照。
- 2026-06-26：修复四分类训练未生效使用 `sample_weight` 的缺陷，重新跑 weighted temporal10 核心消融；`old13_multichannel` 以约 `87.50%` 平衡准确率、`81.25%` 最小召回和更少同强度手指互混胜出，新增 weighted nearfull 装机模型 `oneplus_four_class_after_recollect_hard_nearfull_weighted_old13_multichannel` 并切为端侧默认。
- 2026-07-21：端侧默认模型切换为 Lemon+Tao 四分类验证集表现最佳的 `lemon_tao_full4_enhanced_patch21_no_confidence`，保留其他模型作为可选对照。
- 2026-07-21：根据 OnePlus 实机验证反馈，将端侧默认模型切回针对 OnePlus 设备训练的 `oneplus_four_class_after_recollect_hard_nearfull_weighted_old13_multichannel`，Lemon+Tao 模型保留为可选对照。
- 2026-07-22：Android 训练采集新增用户名称和匿名稳定设备标识；用户名称由采集者填写并在会话中锁定，设备标识首次运行时生成并保存在私有偏好设置，新样本使用 `schemaVersion: 3` 并写入 `collector.userName` 与 `collector.deviceId`。
- 2026-07-23：新增跨设备统一训练 manifest 构建脚本，统一旧 Lemon/Tao 与新 LE2110/PHK110 高质量数据字段；四分类训练器新增 `predefined` 切分，支持把 PHK110 整体保留为未见设备验证集，避免随机切分造成设备与会话泄漏。
- 2026-07-23：Android App 新增 `crossDevice` 产品变体，使用独立包名并默认加载旧+新已见设备训练的 `full_old13_seen_devices_120`；标准变体继续保留原包名和原默认模型，两个 App 可在同一手机并存测试。
- 2026-07-24：完成四批新增数据的设备场景自适应清洗、软权重训练视图、五分钟会话分组和时间冗余压缩；四个 collection 留出折均确认无重复 ID 和跨 split 会话组。
- 2026-07-24：四分类训练器增加带输入签名的样本缓存与验证平衡准确率早停；使用前天同组超参数和 `batch_size=128` 完成四折严格留出及随机对照，确认随机验证约 `83.77%` 但整批留出均值不足上线要求，未替换 Android 端侧模型。
- 2026-07-24：仅使用现有数据完成 `imu11_no_time`、`invariant_shape`、`relative_touch` 三种设备去偏输入的 12 个严格留出实验，并加入训练折 collection-class 平衡权重；最优诊断方案 `relative_touch` 平均约 `44.72%`，未达到部署门槛，未补采、未合成数据、未替换端侧模型。
- 2026-07-24：四分类训练器改为完整张量 CPU 常驻、训练和验证按批搬入 CUDA，修复 Windows WDDM 中断恢复后整折张量映射导致的小额显存分配失败和 `resource already mapped`。
- 2026-07-24：核对并完全复刻 `Full old13 seen devices 120` 的训练方式，使用当前预处理后的 `79878` 条数据训练 `full_old13_seen_devices_current_preprocessed_120`；随机验证平衡准确率约 `79.15%`，较旧模型约 `81.20%` 下降约 `2.05` 个百分点，该模型保留为最终全量训练的 epoch 选择依据。
- 2026-07-24：将最终全量模型 `full_old13_seen_devices_current_preprocessed_all_118` 接入 Android assets，并设为 `crossDevice` 变体默认模型；带随机验证版本、旧跨设备模型和标准变体默认配置继续保留用于对照。
- 2026-07-24：根据当前 pooled old13 随机验证的最佳 epoch `118`，新增 `split_strategy=all` 并使用全部 `79878` 条现有预处理数据训练最终模型 `full_old13_seen_devices_current_preprocessed_all_118`；报告明确 valid count 为零，模型产物已导入 `crossDevice` App 并安装到 PHK110 真机。
- 2026-07-28：确认当前去重目录与 7 月 24 日旧流程输入的 `89385` 个 ID 完全一致，按原 grouped Clean V2 + pooled old13 参数重新训练；随机验证仍在 epoch `118` 达到约 `79.14%` 平衡准确率，随后生成全量 `79878` 条模型 `full_data_clean_v2_old13_all_118`，暂不接入 Android，避免未经同会话真机 A/B 直接替换。
- 2026-07-28：将 `full_data_clean_v2_old13_all_118` 的 scaler 与 weights 导入 Android assets、置顶 catalog，并设为 crossDevice flavor 默认模型；42 个分析测试及 Gradle 构建通过后，将新 APK 覆盖安装到 OnePlus LE2110 `a2d0a964`，真机 UI 的 CNN 诊断与验证模式均确认当前模型 ID 为该模型且已完成推理，standard flavor 默认模型和全部旧模型保持不变。
- 2026-07-31：基于全部 `79878` 条 Clean V2 样本训练 `ACTION_DOWN` 首帧十特征实验 MLP，随机验证 balanced accuracy 约 `47.99%`；新增 Android 独立首帧分类器，在按下当前帧落盘到内存后立即发布早期预测，抬起后继续由现有 `full_data_clean_v2_old13_all_118` 输出最终结果，早期预测不计入最终召回。
- 2026-07-31：将首帧实验模型安装到 OnePlus LE2110 并用 10 秒持续按压完成抬起前后截图验证；早期模型在按住阶段可见、最终模型在抬起后覆盖，本次早期样例发生轻重点误判，与离线约 `47.99%` balanced accuracy 一致，因此功能可测试但不作为正式分类结果。
- 2026-07-31：修复首帧 MLP 已运行但未出现在 CNN catalog 列表而被误认为没有安装的问题；模型面板新增常驻“按下期默认模型 first_touch_early_v1”，原切换列表改名为“抬起期 CNN 模型切换”。49 个分析测试、4 个 Java 测试和 crossDevice 构建通过后覆盖安装到 LE2110，真机 UI 层级确认三个目标文本同时可见。
- 2026-08-02：采集链路改为按下/抬起双锚点原始窗口，保存 `down-15...lift+15` 的可变长度真实 IMU 序列；训练窗口与端侧固定 CNN 输入继续分离，Android/iOS 同步记录双锚点元数据并在中断时标记不完整、不补零伪造原始帧。
- 2026-07-31：新增历史按下锚点 A/B 数据集筛选脚本与审计结构；A 保留按下时刻位于真实 IMU 覆盖范围内的样本，B 从 A 中进一步保留可完整提取真实 `-5...+9` 连续窗口的样本，明确排除全零补位帧且不修改源 JSON。
- 2026-08-06：四分类训练器新增 schema v5 原始目录计划准备、按下/抬起锚点对照、跨运动场景留出、session 时间尾段留出及按按压时长分组报告；原始 JSON 保持只读，抬起时长不作为端侧模型输入。
- 2026-08-07：新增 schema v5 按下锚点完整因子消融调度器及测试，按 12 个特征预设 x 7 个窗口 x 3 个运动场景严格留出统一管理训练；已完成项按完整模型产物跳过，结果仅保留离线评估，不修改 Android 默认模型。
- 2026-08-07：新增按下期因果窗口过滤、确定性训练开关和 84 个时间窗口 x 3 个场景留出的 252 次消融运行器；完成全部严格训练，`-1...+2` 四帧窗口在当前数据上胜出，未自动部署模型。
- 2026-08-07：新增 84 个按下期窗口的 CSV 总览表，合并三个运动状态留出折指标、综合平均/最差指标、覆盖率与前十标记，便于直观检查候选窗口的离散分布。
- 2026-08-07：将 84 窗口 CSV 总览表改为中英对照表头，并将前十布尔标记改为“是 (Yes) / 否 (No)”，保留全部排名和数值指标不变。
- 2026-08-07：新增 4至15 帧密集窗口遍历计划，保留原84窗口计划作为默认；新计划完整遍历 114 种前/后帧分配，并扩展为342次三场景留出训练任务。
- 2026-08-07：按工作进程并行训练需求，将密集窗口调度改为窗口级并发；每个窗口内三折串行以保护特征缓存，不同窗口以六组并行完成本轮训练，主调度进程统一写入进度。
- 2026-08-07：Android 原始采集升级为 schema v6。每个 IMU 帧记录硬件与接收 `elapsedRealtimeNanos`，每个触摸帧记录换算后的事件与接收单调时间，记录级 `timeAlignment` 明确按下/抬起时间和 ZOH 因果可用性规则；双锚点窗口同步改用共同单调时间轴。
- 2026-08-09：按下期因果合同增加触摸事件时间与 IMU 硬件时间的第二道未来门；缺少 IMU/触摸事件或接收时间时拒绝构建。基础合同升级为含 `touch_valid` 的 `causal_imu12_v1`，完整触摸合同继续使用 `causal_imu11_touch9_v1`。
- 2026-08-09：新四分类训练器与旧 schema v5 训练脚本解除主动依赖，通用训练工具移至 `training_common.py`，旧路径标记为 legacy-only；训练分割出口强制拒绝缺失或跨 split 的原始 session。
- 2026-08-07：按下期训练窗口改为强制活动按压约束，旧样本缓存以 `causalPressActiveV1` 失效；252 窗口消融新增训练/验证计数审计、总可输出覆盖率、等待延迟和四目标 Pareto 前沿，全因子命令同步显式传入 `--require-press-active`。

## 跨设备因果时间网格补充
- `build_causal_time_grid_dataset.py`：新增第二层离线导出器。它只读 schema v6 原始 JSON，以源文件 SHA-256 去重并把每条来源唯一归档为 accepted 或 rejected；接受样本分别输出 `causal_time_grid_imu11_v2` 的 `[9,11]` 与 `causal_time_grid_imu11_touch9_v2` 的 `[9,20]` JSONL 矩阵。输出同时冻结 source manifest、拒绝原因、网格质量、特征合同和源哈希清单；不修改源 JSON，不将时间戳或整次触摸汇总写入矩阵。
- `build_time_grid_training_dataset.py`：新增第三层训练清洗构建器。它按 source SHA-256 逐条核对 `[9,11]` 与 `[9,20]` 第二层矩阵及来源 manifest，生成可训练的 `causal_time_grid_imu11_v2` 和 `causal_time_grid_imu11_touch9_mask6_v3` `[9,26]` 视图。后者仅追加压力、面积、长轴、短轴、方向、指针数的逐帧可用性掩码；训练候选按 `platform + collectorDeviceId + sessionId` 严格分组，已知多指触控进入排除清单，完全相同的同组同类输入才降权。清洗报告显式标记 session-标签绑定和全量缺失的触摸字段，训练脚本可据此禁用无信息通道。
- `train_time_grid_strict_cnn.py` / `run_time_grid_strict_baselines.py`：新增第四层严格训练入口。训练器从冻结 JSONL 和三折 session manifest 读取数据，`touch23` 固定删除全量缺失的方向三通道，scaler 只在训练折拟合并对缺失触摸值保持零；轻量一维 CNN 使用 AdamW、梯度裁剪、早停和有界 session/class 权重。调度器固定运行 `imu11` 与 `touch23` 各三折，输出逐类召回、最差折、泛化差距、参数量和确定性复训报告，未通过候选门槛时不部署模型。
- `train_pose_invariant_variants.py`：新增全量姿态不变双模型训练入口，复用冻结的 `training_touch26.jsonl`、GroupNorm CNN、鲁棒 scaler、session/class 权重和 80 轮无早停配置。`no_gravity` 生成 `[9,20]` 输入并完全删除 `gravity.x/y/z`；`delta_gravity` 生成 `[9,23]` 输入，以每个时间槽的重力减去前一时间槽重力，第一槽固定为零，只保留旋转变化。两组产物均标记为 `all_data_fit_only`、`deploymentAllowed=false`，不覆盖现有模型或 Android 默认配置。
- `time_grid_v6_all_no_gravity_20260811_experimental` / `time_grid_v6_all_delta_gravity_20260811_experimental`：使用 29,736 条历史与最新 schema v6 原始记录重建固定物理时间网格；29,521 条通过归一化，清洗后 29,513 条进入 80 轮全量拟合。两模型分别使用 `[9,20]` 无重力合同和 `[9,23]` 重力差分合同，继续标记为 `all_data_fit_only`、`deploymentAllowed=false`，只加入可选模型目录，不替换默认模型。
- `cross_device_causal_features.py`：新增 `causal_time_grid_imu11_v2` 与 `causal_time_grid_imu11_touch9_v2` 契约。以 `timeAlignment.downEventElapsedRealtimeNanos` 为 `0ms`，固定重采样到 `-15,-10,-5,0,5,10,15,20,25ms` 九个物理时间槽；IMU 连续通道按硬件时间左右帧线性插值，端点缺少右支撑、间隔超过 `max(2*5ms,4*median_dt)`、时间戳异常或抬起早于 `+25ms` 时拒绝样本。
- `cross_device_causal_features.py`：跨平台输入必须声明平台、传感器来源、单位、轴坐标系与 `transformVersion`。Android 含重力加速度计只有在 `gravityAvailable=true` 且离线执行加速度减重力后才允许进入 v2；iOS `user_acceleration`、Android 线性加速度统一为 `m/s^2`，陀螺仪统一为 `rad/s`，轴变换矩阵必须是正交单位矩阵（允许镜像）。
- `cross_device_causal_features.py`：触摸输入只读取逐帧 `touchFrames[]`。按触摸帧接收时间相对 IMU 接收时间做零阶保持，并额外检查触摸事件时间不能晚于目标硬件时间；触摸屏幕尺寸按像素/point/毫米单位归一化，缺失压力、面积、半径或方向时写入数值占位并在 `missingMasks` 中保留独立可用性标记，根级 Contact Patch、整次汇总和抬起后字段不进入 v2 特征。
- `train_android_finger_four_class_cnn.py`：时间网格训练强制使用九个物理槽和按下期活动约束（抬起时间必须不早于 `down+25ms`），样本缓存签名记录平台适配版本、坐标变换版本、网格速率、偏移、插值和间隔规则；scaler/training report 写入相同 `featureContract` 元数据，避免 v2 结果误走 v1 原始帧索引路径。
- `AispectCausalPressFeatureBuilder` / `AispectCollectionController`：Android 端按 scaler 的 `featureContract` 显式分派 v1/v2；v2 只有在真实 IMU 数据覆盖正端点并具备右插值支撑时才发布预测，缺少触摸双时钟或窗口质量不合格时不补零、不外推。
- `AispectCausalPressFeatureBuilder`：Android 时间网格增加两个姿态不变推理合同。`causal_time_grid_touch7_mask5_no_gravity_v1` 从原 23 通道因果矩阵中删除索引 `8..10` 的绝对重力，输出 `[9,20]`；`causal_time_grid_imu11_touch7_mask5_delta_gravity_v1` 保持 `[9,23]`，将三个重力通道替换为当前网格减前一网格的因果差分，第一网格固定为零。两者复用原触摸逐帧值、可用性掩码和 GroupNorm CNN 运行时。
- `AispectTimeGridGroupNormRuntime`：触摸测量值的缺失掩码通过 `featureNames` 动态查找对应的 `pressure_available`、`size_available`、`major_available`、`minor_available` 或 `pointer_count_available`，不依赖 23 通道合同中的固定列号。20 通道无重力合同删除三个绝对重力列后仍能使用同一归一化运行时；声明需要触摸可用性但缺少对应掩码通道的模型会在加载时被拒绝。
- `android/analysis/audit_cross_device_capture_contract.py`：schema v6 原始合同审计强制每个 IMU 帧携带合法 `sensorSource/streamEpoch`，检查姿态观察帧不得超出最终 IMU 窗口，并把触摸轴倒置、长短轴不可区分、工具轴完全别名和恒定压力聚合为不重复的设备退化告警。`deviceCapability` 必须包含轴语义、可区分性与独立性字段，且 `observedFrames` 必须严格等于本条 `touchFrames[]` 数量；累计画像、当前样本退化证据不一致或帧数越界都会阻止样本直接进入后续训练层，普通退化告警仍不删除原始数据。
- 2026-08-09：开始落地跨设备 v2 因果特征契约，补齐元数据拒绝、触摸双时钟校验、固定物理时间网格支持闸门、含重力加速度计线性化和训练缓存契约签名；旧 v1 模型与原始 schema 保持兼容且不自动替换。
- 2026-08-09：补充跨设备目标与数据结构总文档 `goal文档.md`；统一 Python/Android 的间隔中位数规则、缺失指针/触摸形状掩码、原始坐标回退归一化和按下期抬起端点过滤。
- 2026-08-09：Android 加速度计回退路径新增同一单调时间轴的重力对齐门，未获得有效重力支撑时不写入伪线性加速度帧。
- 2026-08-09：时间网格构建结果增加 `gridQuality`（实际采样率、中位间隔、最大间隔、覆盖时长和接受状态），用于训练 manifest 与跨设备严格留出审计。
- 2026-08-09：训练端样本载入支持保留每条 v2 样本的 `gridQuality`，避免只留下矩阵而丢失跨设备质量审计信息。
- 2026-08-09：时间间隔质量统计改为只使用固定物理窗口及其左右支撑帧，并与 Android 使用相同的上中位数规则，避免窗口外异常间隔污染拒绝阈值。
- 2026-08-09：统一无触摸占位行的可用性掩码结构，`pointerCount` 与压力/面积/半径/方向字段始终有明确的缺失标记。
- 2026-08-10：新增 schema v6 第二层因果时间网格离线导出器和行为测试；源 JSON 通过 SHA-256 冻结，重复来源与不可构建窗口进入拒绝清单，训练矩阵只来自显式特征合同的固定物理时间网格。
- 2026-08-10：新增第三层时间网格训练清洗器和测试；训练输入与第二层 source SHA-256 一一对应，触摸缺失掩码扩展为 `[9,26]` 合同，session 严格折不允许同组泄露，当前阶段不产生或部署模型。
- 2026-08-10：训练清洗报告增加 `sessionLabelBinding`、每类 session 数和全量缺失掩码字段，使 session 与标签耦合、无信息通道成为训练前可机器检查的风险门槛。
- 2026-08-10：完成 `0810_time_grid_strict_baseline` 六次严格训练与一次确定性探针；`imu11` 平均 balanced accuracy `40.23%`，`touch23` `42.68%`，但两者平均泛化差距分别为 `52.99%` 和 `51.75%`，均超过过拟合门槛，因此聚合报告将决策锁定为保留基线、不允许部署。
- 2026-08-10：新增 `train_time_grid_all_data_ablation.py` 与 `run_time_grid_channel_effectiveness_ablation.py`。该独立入口固定使用全部 eligible 样本、80 epochs、原严格基线的优化器与样本权重，并最多以 7 个独立进程并行运行 9 个通道合同；不读取 session 折、不早停，所有产物显式标记 `all_data_fit_only` 和 `deploymentAllowed=false`。输出目录保存严格基线配置快照、脚本 SHA-256、每合同权重/scaler/训练历史及训练集拟合排名；其用途仅限通道冗余与拟合差异审计，不能用于部署或泛化结论。
- 2026-08-10：全量通道消融合同增加唯一性测试；将与 `touch15_geometry` 重复的 `imu15_geometry_ratio` 替换为 `imu12_touch_valid`，用于独立测量触摸因果可用状态的拟合贡献。首轮含重复合同的训练产物仅保留为无效审计副本，不作为最终排名依据。
- 2026-08-10：Android 增加实验性 `causal_time_grid_imu11_touch7_mask5_v3` 合同，固定输出九个物理时间槽的 `[9,23]` 因果矩阵：11 个位置/IMU 通道、5 个触摸数值通道、指针数、触摸有效位及 5 个逐帧可用性掩码。新增独立 `time_grid_groupnorm_cnn_v1` 运行时，严格执行 `Conv1d -> GroupNorm -> GELU -> Conv1d -> GroupNorm -> GELU -> MeanPool -> Linear`，按导出的逐通道 scaler 做缺失掩码、鲁棒缩放与截断。模型 `time_grid_all_data_touch23_experimental` 仅加入可选目录，不替换默认模型；其来源为全量拟合通道消融产物，明确标记 `evaluationScope=all_data_fit_only` 与 `deploymentAllowed=false`，仅可用于端侧兼容性和同条件真机验证。
- 2026-08-10：新增姿态不变双模型全量训练器与行为测试，分别训练删除绝对重力的 `no_gravity` `[9,20]` 模型，以及用相邻物理时间槽因果差分替换绝对重力的 `delta_gravity` `[9,23]` 模型；两组共同使用 2,582 条 eligible 样本、80 轮训练和两个并行进程，产物独立保存且不自动部署。
- 2026-08-10：Android 端新增无绝对重力与重力因果差分两个时间网格合同，通用 GroupNorm 导出器支持 20/23 通道合同；两套全量实验模型加入可选 CNN catalog，不替换默认模型，并构建 standard debug APK 安装到当前 ADB 设备用于真机 A/B。
- 2026-08-10：修复按下期时间网格模型的运行链路：模型推理改为直接读取完整 IMU 滚动缓存，经固定物理时间网格重采样后输入，旧模型继续使用历史 `-5...+9` 帧窗口；验证模式在抬起后保留最近一次按下期预测，并持续预热 IMU，直到下一次 `ACTION_DOWN` 才清空结果，避免有效预测被收尾流程覆盖。
- 2026-08-10：修复 20 通道 `no_gravity` 实验模型真机推理时因触摸可用性掩码仍使用 23 通道固定索引而越界崩溃的问题；GroupNorm 运行时改为按特征名动态解析掩码列，并增加完整 `predict()` 回归测试。模型权重与 scaler 数值参数不变，无需重新训练。
- 2026-08-10：Android 原始双锚点窗口升级为 `85ms + 15帧` 双下限；抬起后改为 85ms 首次检查、8ms 条件轮询和 500ms 超时，schema v6 新增版本化 `captureWindow` 审计对象。按下期九槽模型输入保持不变，iOS 同构改造仍待完成。
- 2026-08-11：修复 Android 触摸时间轴的亚毫秒倒序；触摸事件时间改为按 `ACTION_DOWN` 固定时钟原点转换，时间对齐合同升级为 `timeAlignment.schemaVersion=2`，增加固定转换与旧数据重建说明；不改变模型输入和训练参数。
- 2026-08-11：修复触摸输入元数据对 `InputDevice.getSourceName` 非公开 API 的依赖，改为无损十六进制 source 名称，恢复 compileSdk 34 的干净构建。
- 2026-08-11：依据《跨设备采集字段采集方式回退研究》升级 Android 原始采集证据层：新增 MotionRange 与设备/显示元数据、逐轴来源和合成未知掩码、IMU/姿态流健康状态及 `rotationFrames[]` 观察流；姿态流固定不进入模型，触摸编码禁止 `toolMajor/toolMinor` 静默覆盖触摸轴，并新增跨设备采集合同审计脚本与测试。
- 2026-08-11：根据规格复核补齐 Android 传感器注册返回值、首帧超时和停流后的候选切换；统一 `accelerometer_minus_gravity` 离线来源合同，禁止无重力普通加速度进入训练，并彻底移除触摸归一化与清洗脚本中的工具轴覆盖。采集审计新增 MotionRange 子字段、逐轴三态、双时钟因果、姿态来源一致性和原始加速度泄露拒绝测试。
- 2026-08-11：为 IMU 帧和姿态观察帧补充 `sensorSource/streamEpoch`，按来源切换隔离旧帧；触摸元数据补充 `metadataEpoch/windowInsets`（不可用时显式标记），审计新增 IMU 双时钟单调性、状态/来源一致性和时间戳回退拒绝门。
- 2026-08-11：修复双锚点窗口重建丢失逐帧 `sensorSource/streamEpoch`、`rotationFrames[]` 保存整段滚动缓存和触摸轴退化被评为 Level A/high 的问题；姿态帧改按最终 `85ms + 15帧` 双下限 IMU 窗口裁剪，能力画像与归一化诊断增加轴语义、长短轴可区分性与工具轴独立性门，major/minor 全程相等时按单半径 Level B/medium 处理，并升级 Python 合同回放。同步修复旧特征构建器错误要求时间槽数等于通道数的输入守卫。
- 2026-08-11：继续依据《跨设备采集字段采集方式回退研究》修复传感器来源合同：普通加速度减重力后的逐帧来源统一为 `accelerometer_minus_gravity`；每个 IMU 帧新增来源派生的 `trainingInput`，原始或不可用加速度只能作为证据保存；重力健康变化导致语义切换时递增 `streamEpoch`。姿态审计改为允许不同来源跨 epoch 保留、禁止同一 epoch 混源，并要求最终健康来源与最后姿态帧一致。
- 2026-08-11：修复 `deviceCapability` 跨触摸累计污染；单条样本保存时改为仅根据本次 `activeTouchFrames` 重建能力快照，保证长短轴可区分性、信息性和工具轴别名状态不会继承上一条触摸。累计 profiler 仅保留为运行界面诊断，不再写入单条训练 JSON；Python 合同审计同步要求 `observedFrames == touchFrames.length`，防止语义布尔值碰巧一致时漏检累计污染。
- 2026-08-11：为每个 `touchFrames[]` 元素写入单调 `metadataEpoch`，记录级 epoch 固定对应首帧；同一元数据指纹保持版本，记录级显示快照、逐帧 View 尺寸或输入设备元数据变化时递增。采集审计新增正整数、根级首帧一致和禁止回退三道门，并保留未逐事件采集旋转/Insets 的能力边界。
- 2026-08-11：使用全部 29,736 条 schema v6 原始记录重建因果时间网格，29,513 条清洗样本完成无重力与重力差分双模型 80 轮全量拟合；导出两个独立 Android 实验模型资源并加入可选目录，保留旧默认模型用于真机 A/B。

## 远程模型下发与运行时激活

### 目标与边界

- Android 端远程下发只支持 `time_grid_groupnorm_cnn_v1` 的 runtime bundle；旧 `12x21` CNN 继续通过 APK assets 目录加载，不改变其已有回退路径。
- 远程 bundle 仅接受两套因果时间网格合同：`causal_time_grid_imu11_touch7_mask5_delta_gravity_v1` 的 `[9,23]`，或 `causal_time_grid_touch7_mask5_no_gravity_v1` 的 `[9,20]`。
- 训练产物必须显式写入 `deploymentAllowed=true` 才允许发布到手机。`evaluationScope=all_data_fit_only` 的全量拟合实验模型即使能推理，也不得作为线上下发目标。

### Android 更新链路

1. `AispectModelUpdateScheduler` 在 App 前台启动时通过 WorkManager 提交带网络约束的唯一更新任务；构建参数 `MODEL_API_BASE_URL` 为空时不发起网络请求。
2. `AispectModelRefreshWorker` 生成稳定安装 ID、登记设备信息，并创建 `AispectModelHttpClient` 请求模型 assignment、下载 `manifest.json`、`weights.json`、`scaler.json` 三件套和上报更新结果。
3. `AispectRemoteModelUpdater` 校验 assignment 的模型格式和特征 schema、逐件核对大小与 SHA-256，再交给 `AispectRemoteModelStore` 暂存和激活。
4. `AispectRemoteBundleValidator` 对 manifest、标签顺序、输入矩阵、feature names、GroupNorm 张量形状、文件哈希和 `deploymentAllowed` 执行严格校验。任何不满足的 bundle 都不会影响现有运行模型。
5. `AispectRemoteModelStore` 只允许 `modelId/version` 的安全路径段；先在 `staging` 验证，再原子移动至 `versions`，状态文件维护 `current` 和 `previous`。新版本损坏时可恢复上一已验证版本。
6. `AispectImpactCNNClassifier` 优先加载已激活的远程 bundle；远程模型不可用、无效或加载失败时，自动回退到 APK assets catalog。
7. `AispectModelRuntimeController` 在触摸期间只记录 reload pending；触摸结束后才通过主线程 reload，避免同一次按压的前后帧使用不同模型。

### 后端合同

- assignment 通过 `/api/v1/model-assignment` 返回模型 ID、version、schema、format 及三件 artifact 的 URL、大小和 SHA-256。
- 更新结果通过 `/api/v1/model-update-events` 回执；设备信息通过 `/api/v1/devices/register` 登记。登记或回执失败不会回滚已经完成的本地激活。
- 后端发布校验器接受 `aispect-time-grid-groupnorm-v1` / `aispect-time-grid-json-v1`，并要求 manifest 与 scaler 的 `inputContract` 一致。

### 构建依赖

- `aispect-core` 使用 `androidx.work:work-runtime:2.9.1` 承担网络约束、唯一任务与失败重试；应用和 library manifest 都声明 `INTERNET`、`ACCESS_NETWORK_STATE`。
- Android Gradle 仓库按 `maven.aliyun.com/repository/google -> google() -> mavenCentral()` 回退。该镜像仅用于 Windows 开发机的依赖可达性，不改变 APK 的联网目标或远程模型 API。

### 修改日志

- 2026-08-13：新增远程时间网格模型的 assignment 下载、哈希/合同校验、暂存原子激活、上一版本回退、触摸结束安全 reload 与后端发布合同；新增 Google Maven 镜像回退以保证 WorkManager 依赖可构建。
- 2026-08-13：远程模型文件读写改为基于 `java.io` 的 API 24 兼容实现，避免 Android 低版本设备调用 `java.nio.file` 导致运行时崩溃；核心校验、下载和激活合同保持不变。

## 可复用 Android 触摸 SDK 整合

- 目标目录 `C:\codeDev\cnn\sdk\aispect-touch-android` 以 `eyes/android/aispect-core` 的公开 `com.zhifaios.eyes.touch.*` API、Maven 发布、canonical assignment、下载模型不可变版本存储与 current/previous 回滚为产品边界；不复制宿主 App、采集 UI、JSON 数据集、训练脚本、构建缓存或实验模型目录。
- `AispectImpactCNNClassifier` 同时保持旧版 release CNN 路径与新 `time_grid_groupnorm_cnn_v1` 路径。模型选择只依赖显式 `runtimeArchitecture` 和 `featureContract`，不根据通道数推断；远程时间网格只接受 `causal_time_grid_imu11_touch7_mask5_delta_gravity_v1` `[9,23]` 与 `causal_time_grid_touch7_mask5_no_gravity_v1` `[9,20]`。
- `AispectRemoteModelAssignment`、`AispectCanonicalModelAssignment` 与 `AispectModelBundleValidator` 为两种时间网格合同增加独立校验分支。通过 canonical assignment 的 `modelType=time_grid_groupnorm_cnn_v1` 和 `featureSchemaId=aispect-time-grid-groupnorm-v1` 声明，仍须逐项校验标签顺序、九槽矩阵、特征顺序、GroupNorm 张量和 SHA-256；失败下载不能改变当前已验证模型。
- `AispectImpactSignalCollector` 为每个 IMU 帧写入硬件 `sensorTimestampElapsedRealtimeNanos` 与接收 `receivedElapsedRealtimeNanos`。`AispectCollectionController` 以 `ACTION_DOWN` 转换得到的同一 elapsed-realtime 原点记录触摸帧，只有 `-15ms...+25ms` 九槽都具有真实左右支撑且未在 `+25ms` 前抬起时才发布一次因果结果。该结果使用公开 `PRESS` 事件回调，触摸结束时按序列号去重；触摸期间的模型更新继续延迟到触摸完成。
- SDK 内置资源只保留一个稳定的旧版四分类 fallback `lemon_tao_full4_enhanced_patch21_no_confidence`。历史、消融和全量拟合实验模型不再随 SDK 分发；新因果模型只能经远程合同验证后激活。

### 修改日志

- 2026-08-14：新建独立 Android SDK 暂存整合目录并完成 causal GroupNorm 运行时、canonical 远程模型合同、双时钟 `PRESS` 发布门和单 fallback 资产收缩；新增合同、远程校验、发布门和公开 PRESS 映射的 JVM 回归测试。

## 可复用 iOS 触摸 SDK 正式化

- 目标目录 `C:\codeDev\cnn\sdk\aispect-touch-ios` 建立 Swift Package 双层结构：`AispectTouchCore` 只含 Foundation/CryptoKit 可测试的因果特征、模型合同与运行时基础类型；`AispectTouch` 保留 UIKit/CoreMotion 采集与宿主生命周期集成。该分层避免传感器/界面代码进入模型和远程更新逻辑。
- `AispectFeatureContract` 固定支持 `causal_time_grid_touch7_mask5_no_gravity_v1` `[9,20]` 与 `causal_time_grid_imu11_touch7_mask5_delta_gravity_v1` `[9,23]`。物理时间槽为 `-15,-10,-5,0,5,10,15,20,25ms`，标签顺序固定为 `thumb_light`、`thumb_heavy`、`index_light`、`index_heavy`。
- `AispectCausalPressFeatureBuilder` 接收具有传感器时间与接收时间的 `AispectIMUSample`，按传感器时间线性插值并限制最大支撑间隔为 `max(10ms, 4*局部中位间隔)`；触摸 `AispectTouchObservation` 以事件时间和接收时间共同执行零阶保持，禁止读取目标时间之后发生或尚未到达的触摸信息。抬起早于 `down+25ms`、端点无支撑、时间倒序和无效尺寸均拒绝，不补零或外推。
- 触摸特征保留坐标归一化、压力、面积、长短轴、轴比、指针数和独立可用性掩码；尺寸按当前屏幕对角线归一化，缺失字段以数值占位及掩码表达。`no_gravity` 删除绝对重力；`delta_gravity` 仅使用相邻物理槽的重力差，首槽固定为零。

### 修改日志

- 2026-08-16：开始 iOS 正式 SDK 落地，新增 Swift Package 核心合同、九槽因果特征构造器和对应单元测试；Windows 机器未安装 Swift，测试待 macOS/Xcode 环境执行。
- 2026-08-16：新增 iOS JSON GroupNorm CNN 运行时与 bundle 验证器，逐项校验 iOS 平台、部署许可、四分类标签、两种九槽特征合同、张量形状、模型文件路径/大小/SHA-256；当前只形成本地 SDK 核心，远程下载与 UIKit 生命周期接入尚未完成。
- 2026-08-16：新增 iOS `AispectPendingModelActivation`，模型更新在触摸活动期间只进入 pending，触摸空闲后才能切换，保证同一按压序列不会跨模型。
