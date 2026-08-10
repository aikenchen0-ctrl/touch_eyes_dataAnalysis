# 跨设备按下期触摸分类：目标与数据结构约束

## 1. 总目标

建立一个可以在 Android、iOS 及不同采样率设备上复用的按下期触摸分类系统。系统需要在按下后尽早判断四类标签：

- `thumb_light`：拇指轻触
- `thumb_heavy`：拇指重触
- `index_light`：食指轻触
- `index_heavy`：食指重触

模型窗口的语义按真实物理时间定义，而不是按某一台设备的回调序号定义。当前正式 v2 网格为：

```text
按下事件 = 0ms
-15, -10, -5, 0, +5, +10, +15, +20, +25ms
```

设备端仍然逐帧接收数据，但必须等真实 IMU 硬件时间覆盖 `+25ms`，并具备右侧插值支撑后，才组装九个网格点并输出一次分类。这个输出仍属于按下期输出，只是延迟由物理窗口决定。

## 2. 数据分层

| 层级 | 数据对象 | 是否进入模型 | 服务目的 |
| --- | --- | --- | --- |
| 原始证据层 | schema v6 JSON、文件哈希、采集者、设备、session | 否 | 可追溯、去重、重建任意训练视图 |
| 记录对齐层 | `timeAlignment`、传感器/触摸元数据 | 不直接进入 | 把不同来源放到共同单调时钟，校验单位、轴和因果边界 |
| IMU 原始层 | `frames[]` 中的真实加速度、陀螺仪、重力和硬件/接收时间 | 先适配，再进入 | 提供运动变化和设备姿态信息 |
| 触摸原始层 | `touchFrames[]` 中逐事件坐标、压力、面积、半径、方向、指针数和可用性 | 先逐帧 ZOH，再进入 | 提供每个时间点实际可获得的接触信息 |
| 派生训练层 | 固定物理网格矩阵 `[9,11]` 或 `[9,20]` | 是 | 作为训练和端侧推理的固定输入 |
| 模型参数层 | scaler、权重、`featureContract`、网格规则 | 否 | 保证训练端和 Android 端解释完全一致 |

原始层只读保存。清洗、剔除、去重和窗口筛选都输出新的 manifest 或派生目录，不修改源 JSON。

## 3. schema v6 原始结构

### 3.1 记录级字段

记录保存标签、采集者、设备能力、session、窗口完整性和双锚点索引。`downReferenceFrameIndex` 固定表示按下帧，`liftReferenceFrameIndex` 表示抬起帧在同一 IMU 时间轴上的位置。Android 新记录的 `captureWindow.contract=dual_anchor_min_time_and_frames_v1`：开始位置取 `down-85ms` 与按下前 15 个真实 IMU 帧中更早者，结束位置取 `lift+85ms` 与抬起后 15 个真实 IMU 帧中更晚者；`windowComplete` 只有在前后两侧均同时满足 85ms、15 帧且序列连续时才为真。历史 schema v6 缺少该对象时按旧纯帧数合同识别；iOS 当前仍未同步该新合同。原始冗余不进入模型，v2 仍只取 `-15...+25ms` 九槽。

### 3.2 `timeAlignment`

`timeAlignment` 保存：

- `downEventElapsedRealtimeNanos`、`downReceivedElapsedRealtimeNanos`
- `liftEventElapsedRealtimeNanos`、`liftReceivedElapsedRealtimeNanos`
- IMU 硬件时间来源 `SensorEvent.timestamp`
- 触摸事件时间来源 `MotionEvent.eventTime` 的单调时钟换算
- 可用性规则：`touchReceived <= imuReceived` 且 `touchEvent <= imuSensor`

这些时间戳只用于对齐、插值、质量检查和等待调度，禁止作为模型通道。

### 3.3 `sensorMetadata`

必须声明 `platform`、`sensorSource`、`accelerationUnit`、`rotationRateUnit`、`sensorAxisFrame`、`axisTransformMatrix` 和 `transformVersion`。

- Android 线性加速度和 iOS `userAcceleration` 统一为 `m/s^2`。
- 陀螺仪统一为 `rad/s`。
- Android 普通加速度计只有在同时存在同一时间轴的重力数据时，才执行 `linear = acceleration - gravity`；缺失重力时拒绝 v2 样本。
- 所有轴先变换到 canonical device 坐标：`x=device_right`、`y=device_up`、`z=device_outward`。
- 轴变换矩阵必须是正交单位矩阵，不能缺省猜测设备方向。

### 3.4 `frames[]`

每个 IMU 帧同时保留真实硬件时间和应用接收时间。原始 JSON 中可见的字段包含：

| 中文 | English | 作用 |
| --- | --- | --- |
| 帧索引 | `frameIndex` | 双锚点原始序列中的位置 |
| 相对时间 | `relativeTimestamp` | 兼容查看和历史分析 |
| 加速度三轴 | `x`,`y`,`z` | 传感器原始/适配后的运动值 |
| 归一化触摸坐标 | `x_norm`,`y_norm` | 当前触摸位置快照，不能替代 `touchFrames[]` |
| 陀螺仪三轴 | `rotationRateX/Y/Z` | 旋转运动 |
| 重力三轴 | `gravityX/Y/Z` | 姿态与重力参考 |
| 按下后时间 | `timeSinceTouchDown` | 原始审计字段，不能进入 v2 模型 |
| 锚点后时间 | `timeSinceAnchor` | 原始审计字段，不能进入 v2 模型 |
| IMU 硬件时间 | `sensorTimestampElapsedRealtimeNanos` | 时间对齐和插值 |
| IMU 接收时间 | `receivedElapsedRealtimeNanos` | 因果可用性边界 |

`delta`、`maxAbsDelta` 不属于 schema v6 原始帧；冲击变化量需要时在派生阶段临时计算。

## 4. 触摸数据结构

`touchFrames[]` 是触摸事件采样序列，不是一次触摸的结束汇总。每个元素保存 `x/y`、`x_norm/y_norm`、`pressure`、`size`、`touchMajor`、`touchMinor`、`toolMajor`、`toolMinor`、`orientation`、`pointerCount`、视图尺寸、事件时间、接收时间和 `axisAvailability`。

跨设备处理规则：

1. 先按触摸单位和视图尺寸归一化坐标、半径和面积；缺少尺寸时拒绝需要形状的契约。
2. 每个重采样点使用覆盖它的真实 IMU 帧接收时间执行零阶保持，只选择已收到且事件时间不晚于该 IMU 硬件时间的最后一个触摸帧。
3. 缺失压力、面积、长轴、短轴、方向或指针数量写入数值占位，同时在 `missingMasks` 保留独立可用性标记；缺失不能伪装成真实小半径。
4. 根级 `contactPatch`、`contactEncoding`、`touchNormalization`、高斯图、协方差、面积/半径汇总不进入 v2 特征。

## 5. 训练输入结构

### 5.1 v1 兼容契约

- `causal_imu12_v1`：`[T,12]`，坐标 2 + IMU 9 + `touch_valid`。
- `causal_imu11_touch9_v1`：`[T,20]`，坐标 2 + IMU 9 + 触摸细节 9。
- v1 仍按原始 `frameIndex` 取帧，只用于已有模型兼容和历史对照。

### 5.2 v2 正式跨设备契约

- `causal_time_grid_imu11_v2`：`[9,11]`。
- `causal_time_grid_imu11_touch9_v2`：`[9,20]`。
- 九个时间槽固定对应上述物理毫秒偏移；不把时间戳本身加入通道。
- IMU 连续值按硬件时间线性插值；触摸值按 IMU 接收时间 ZOH。
- 左右支撑不完整、相邻间隔超过质量阈值、抬起早于 `+25ms`、平台元数据缺失时拒绝样本，不零填充、不外推。

训练和端侧必须读取同一 `featureContract`、`gridRateHz`、`gridOffsetsMs`、`interpolation`、`maxGapRule`、平台适配版本和坐标变换版本。不得仅凭通道数量猜测契约。

## 6. 约束与服务目的

| 约束 | 保护的目标 | 违反后的处理 |
| --- | --- | --- |
| 真实硬件时间优先 | 跨采样率物理一致 | 不使用名义 Hz 或数组下标推时 |
| 接收时间因果门 | 不读取尚未到达的触摸信息 | 触摸帧不可用，必要时拒绝窗口 |
| `+25ms` 右支撑 | 不外推未来 IMU 值 | 继续等待或拒绝 |
| 原始双侧冗余 | 为未来窗口重建保留跨采样率证据 | Android 前后各同时要求至少 85ms 和 15 个真实帧，不完整样本由清洗层拒绝 |
| 抬起边界 | 按下期模型不读取抬起后数据 | 训练过滤，端侧停止按下期发布 |
| session/设备整体留出 | 防止同会话、同设备污染验证 | 训练命令拒绝泄漏切分 |
| 单位/轴元数据完整 | 不把量纲和方向差异学成类别 | 缺元数据直接拒绝 v2 |
| 原始 JSON 只读 | 可追溯和未来重建 | 只在派生 manifest 中剔除 |
| `featureContract` 显式分派 | 训练/端侧同构 | 合同不匹配不加载模型 |

## 7. 验收标准

- 同一物理信号在 120Hz、200Hz、240Hz 和不规则采样下重采样结果在误差范围内一致。
- Android 与 iOS 轴排列、符号和屏幕旋转经适配后落入同一 canonical 空间。
- 修改窗口端点之后的未来帧、抬起时间、根级汇总字段不会改变 v2 输入矩阵。
- Android 新原始样本必须带版本化 `captureWindow`，完整样本前后两侧均同时覆盖至少 85ms 和 15 个真实 IMU 帧；扩大原始窗口不得改变九槽模型输入。
- 训练报告同时包含 balanced accuracy、每类召回、最差设备/平台召回、窗口拒绝率和实际等待时长。
- v2 模型未通过跨设备严格留出和端侧 A/B 前，不替换默认模型；v1 旧模型和原始数据保持可用。
