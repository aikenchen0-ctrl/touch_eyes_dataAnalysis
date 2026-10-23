# Android 引导式采集流程

日期：2026-10-08。工作范围：独立仓库 `F:\sa\touch_eyes_dataAnalysis`。

## 采集者怎么用

1. 填一个采集者代号，不必填真实姓名；成功开始后会记住代号。
2. 点击“场景”，选择当前准备采用的姿势。“每组次数”默认 10，可点开输入 1–1000，自动记住。
3. 看当前动作，例如“拇指 · 轻触”，再点“开始这一组”。
4. 面板自动收起；按提示点击空白区域，底部显示当前动作和已记录次数。
5. 成功保存到设定次数后自动暂停并展开面板。确认下一组动作后，点“开始下一组”。
6. 四组完成后，选择其他场景。需要休息时，打开“采集面板”后点“暂停采集”。

每个场景依次安排：拇指轻触、拇指重触、食指轻触、食指重触，每组采用同一设定次数。
调整次数需先暂停采集；已有记录保留，从第一组核对剩余数量。进度显示、停采配额与新记录的 `targetPerCell` 使用同一设置。
轻触按日常习惯完成，不必刻意减轻力度；重触指接触时有意更用力，不是按住更久。
任务标签由计划安排，模型预测不会替换任务标签。

刚才动作做错时点“刚才那次不采用”：暂停采集、保存复核记录、扣减当前组可用次数，随后可继续补采。
“更多信息（可选）”内可恢复最近一次记录或标记不确定；复核不删除原始数据。
握持和支撑可以不填，不确定时保持未指定，不用为了开始采集逐项猜测。

## 统一场景入口

| 界面名称 | taskId | 导出的 motionState |
| --- | --- | --- |
| 倾斜手持 | tilted_handheld | handheld_static |
| 手机竖直 · 顶部朝下 | inverted | handheld_static |
| 仰卧 · 举手机 | supine | handheld_static |
| 左侧卧 · 自然使用 | side_left | handheld_static |
| 右侧卧 · 自然使用 | side_right | handheld_static |
| 走路 · 手持手机 | walking | walking |
| 桌面平放 | tabletop | placed_static |
| 手持 · 其他姿势 | handheld | handheld_static |

当前八个场景中，倾斜手持表示屏幕朝向自己、手机不竖直也不平放，不限制坐姿；“手持 · 其他姿势”不约束角度。
旧 `seated_natural` 和 `upright` 标识保留兼容，但不再作为当前采集入口；旧坐姿记录不会改写为倾斜手持。
场景是采集指令，仍标记 `task_instruction_not_verified`，不声称系统已验证身体姿态。
旧的运动状态、轻重、手指按钮已从采集主流程移除；由同一计划生成标签，避免出现“走路 + 仰卧”等冲突组合。
模型选择、特征参数、原始记录与预测默认放在折叠的“研究员设置与诊断”中。

## 标签与状态保护

- `AispectCollectionPlan` 管理场景及四组动作，达到配额不会自行切换下一组标签。
- `AispectCollectionController.Config.targetSamplesPerCell` 默认是 0，保留 SDK 其他调用者的连续采集行为；本采集 App 从计划同步用户设定，初始默认 10。
- 只有原始样本保存成功才增加进度，写入失败不增加次数。
- 采集中、触摸未完成、写入或复核尚未完成时锁定采集者、场景及标签。
- 复核按钮只指向当前采集者、场景、握持、支撑、手指和力度组合的最近样本，切场景后不能误改上一个场景。
- 复核前暂停接收新样本，避免恢复计数与满额停采交错。
- 同一进程中切换场景会检查已有计数，不重复开始已满额的组。

当前进度只在本次 App 进程内统计。重启 App 后计数重新开始，已保存的原始文件和复核旁车仍保留；本轮没有实现跨进程续采或跨设备合并进度。
采集目录导出仍须包含 `annotations` 子目录。

## 首版验证记录（调整场景与次数之前）

- 新增计划测试：6 / 6 通过；已有采集任务、进度及协议测试：7 / 7 通过。
- Python 标注复核测试：9 / 9 通过。
- 完整 JVM 测试：59 项，57 通过、2 失败。失败项与 2026-10-05 记录一致：
  - `AispectCausalPressFeatureBuilderTest.builderDoesNotUseTouchEventThatIsPhysicallyAfterImuSample`
  - `AispectLegacyFeatureBuilderTest.legacyBuilderPreservesHistoricalTimeFeatureOnlyForLegacyCaller`
  尚未解决，不将完整测试描述为全部通过。
- `assembleStandardDebug` 和 `assembleCrossDeviceDebug` 均成功。
- 独立、只读 Android API 30 模拟器中：两个 APK 均安装启动成功，崩溃缓冲区无记录。
- 标准版走通四组、满额自动暂停、额外点击不增加计数、明确开始下一组、不采用后补采、恢复记录、场景切换和代号记忆。
- 读取模拟器实际保存的 44 个原始文件及 2 个复核旁车：四组各 10 条可用记录；倒置、走路、桌面平放各 1 条；另 1 条标记不采用且原文件保留。恢复历史有连续记录。
- 界面动作与落盘的 `taskId`、运动状态、手指、力度逐项核对一致。上述模拟器数据仅用于流程验证，不并入训练集，不代表真人动作或识别准确率。
- 本轮没有安装或操作连接的真机。真实采集者体验和大字体、小屏设备仍需回归。

## 场景与可调次数修订验证

- 当前入口是八个场景，默认倾斜手持；坐姿和正向竖直不再出现。
- 计划相关测试 10 / 10，旧任务及复核合同测试 7 / 7，Python 标注复核测试 9 / 9 通过。
- 完整 JVM 测试 63 项，61 通过、2 失败；仍是上文列出的两项特征测试，未新增失败。
- 两个 Debug APK 均重新完成 assemble。组合任务因上述两项测试失败返回非零，不代表 APK 组装失败，也不宣称整套构建检查全绿。
- API 30 独立只读模拟器验证：输入 0 保持设置对话框并提示合法范围；设置 3 后准确在 3 条停止，额外点击不保存。
- 将目标从 3 改为 5，页面保留 3 条进度；补 2 条后停止，不重复采已完成部分。
- 重启 App 后“每组 5 次”设置保留；组进度仍按本次进程统计。
- 实际保存的 5 条记录均为 `tilted_handheld`，身体姿势为 `unspecified`；前三条 `targetPerCell=3`、后两条 `targetPerCell=5`，旧记录没有被改写。
- 本轮未安装真机，模拟器样本不进入训练集。

## 更新后的构建产物

- `android/app/build/outputs/apk/standard/debug/app-standard-debug.apk`
- `android/app/build/outputs/apk/crossDevice/debug/app-crossDevice-debug.apk`

构建使用 Windows JDK 17，独立 Gradle 缓存位于 `F:\sa\.touch-collection-runtime\gradle`。
本次修改没有进入 SuperAppAndroid 工作树，也没有修改 iOS 采集界面。
