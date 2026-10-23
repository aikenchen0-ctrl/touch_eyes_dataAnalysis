# 姿态补采与跨端标签协议 v1

## 本轮边界

仅修改独立仓库 `F:\sa\touch_eyes_dataAnalysis`，不修改 SuperAppAndroid、SDK 模型、模型阈值或已有权重。
这轮实现的是采集与复核闭环，不代表已经提升了真机准确率；需要新数据、独立留出测试和端侧复测才能确认。

## 采集操作

1. 填用户代号，不要求真实姓名。同一个人跨设备使用同一代号，安装实例另外生成设备标识。
2. 选择任务、轻／重、拇指／食指；握持与支撑只需每组选择一次，可以保留未指定。
3. 开始采集后锁定标签；先停止再切换。不在达到目标后自动改任务，避免人仍在做旧动作而文件标签已变化。
4. 每个任务四个“手指 × 力度”组合，各试采 10 条，即每任务 40 条、六任务 240 条。
   这是首轮补采目标，不是统计充分性保证，也不要求把全部姿态维度做笛卡尔积。
5. 最近样本写入完成、手指离开后，可以选择“刚才做错了”“刚才不确定”“确认符合任务／恢复”。
   只修改最近已保存样本的复核旁车，不删除信号，不把“做错”自动翻成另一种力度。
6. 界面进度是本次打开采集器后的有效计数，按用户／任务／握持／支撑／手指／力度区分。
   暂停切组后保留；应用重建不恢复此进度，原始文件仍保留。复核为做错或不确定会扣除，恢复会加回。

| 任务 ID | 用户指令 | 物理姿态指令 |
| --- | --- | --- |
| `seated_natural` | 坐姿自然握持 | 不限制角度 |
| `upright` | 近竖直、顶部朝上 | `vertical_top_up` |
| `inverted` | 近竖直、顶部朝下 | `vertical_top_down` |
| `supine` | 仰卧举手机 | `screen_down` |
| `side_left` | 左侧卧自然操作 | 不限制角度 |
| `side_right` | 右侧卧自然操作 | 不限制角度 |

六任务都采轻、重两类，不能把某个身体姿势等同于重触。不要为模拟重触而额外摇动手机。
“重”指开始接触时的有意重触；先轻按再逐渐加力不属于这轮指令。做错就复核标记，不依据现有 CNN 结果判断是否采对。
长期评测应保留未见过的人、设备和独立采集批次；同时报告各姿态的轻点误触发率、重触召回和最差分组，不用训练集拟合率代表真实准确率。

## 数据结构

### `collectionContext`

根级对象新增 `schemaVersion=1`、`contract=touch_collection_context_v1`：

- `taskId`、`bodyPosture`、`requestedDevicePose`：任务描述。
- `sceneSource=task_instruction_not_verified`：不是通过传感器证明用户实际姿势。
- `grip`、`support`：用户选填，不从手机倾角推测。
- `contactMode=natural`、`gestureIntent=tap`、`strengthOnset=at_contact`：动作指令。
- `labelStatus=task_assigned`：人工选定任务标签，尚未逐条确认。
- `targetPerCell=10`：试采目标。

`light/heavy` 与 `thumb/index` 仍在原 `label` 中；场景字段只用于覆盖审计、分组与评估，
不拼进模型输入矩阵，不生成“倒置重触”等混合类别。旧构造器保持可用，缺失场景为未知。

### `automaticContext`

- Android：保持原始 schema 6。记录采集来源、写入阶段的显示旋转及按下前最近 IMU 携带的重力向量，单位 `m/s^2`。
  向量不存在或无有效重力时标记不可用，不用零冒充实测。重力没有独立硬件时间戳，明确 `gravityTimestampKnown=false`。
- iOS：保留 schema 5，新增平台、采集者、任务、显示方向、传感器来源和重力可用性。
  CoreMotion 路径与原始加速度计回退分别标记；重力／加速度单位为 `g`，未采的触摸形状为 `not_recorded`。
  这里记录的 IMU 时间仍是旧回调时间，**不能当作硬件传感器时间**。
- 内外屏、折叠状态等本轮没有可靠入口的字段保持 `unknown`；没有推断或伪造。
- iOS 新增手指选择，旧标签解码默认 `unknown`，不默认为拇指。

两端已共用任务与复核语义；**尚未完成 iOS 原始触摸双时钟／硬件时间的 schema 6 改造**。
现有严格物理时间网格导出器仍要求 schema 6，不能直接把本轮 iOS schema 5 改版本号后混入。
跨端共同训练仍需匹配真实字段能力的适配器；本轮没有宣称“任意旧数据均已进入同一 CNN”。
接触半径不是新标签协议的必填字段，单位、可用性掩码与因果归一化路径均未被本轮改动。

## 复核与导出

复核文件位于原样本同级的 `annotations/<sampleId>.json`，契约为 `touch_sample_review_v1`：

- `sampleId`、`sampleFileName` 同时绑定原文件。
- `labelStatus` 为 `confirmed`、`mistake`、`uncertain`。
- `history` 追加 `revision`、`status`、`timestampMs`、`source=user_review`，保留恢复历史。
- Android 使用 `AtomicFile`，iOS 使用原子文件替换。写入失败不扣减进度，不报告成功。

**导出时复制整个采集目录，包括 `annotations`；不要只挑根目录 JSON。**
若旁车被手动丢弃，单靠未改写的原始文件无法知道曾有复核，因此目录完整性是导出合同。

`collection_annotations.py` 统一校验旁车，损坏、身份错配、残留原子写事务全部关闭训练入口。
第二层 `build_causal_time_grid_dataset.py`：

- 不将 `annotations` 下文件作为原始样本。
- 剔除做错、不确定；相同原始副本的复核状态冲突时全部隔离。
- 将场景、最终复核状态及复核文件 SHA-256 写入来源 manifest 和派生 JSONL。

第三层 `build_time_grid_training_dataset.py` 验证这些审计字段与来源一致，并保留到训练视图。
它不自动回读源目录；复核变化后必须重新导出第二、三层冻结数据，不能继续训练旧快照。
`train_android_finger_four_class_cnn.py` 的原始数据入口也读取复核；缓存签名覆盖原始与复核内容，
复核变化（包括排除后恢复）要求重建缓存。未改造的历史训练脚本不属于此次复核闭环，不能绕开指定入口训练新数据。

## 其他采集完整性修复

- iOS 原先落盘异常仍增加成功计数的问题已修正；只有真实写入成功才计数。
- iOS 面板按钮操作不进入采样，采集期间不执行分类触觉反馈或闭眼相机触发，减少自行产生的干扰。
- 本轮不根据接触面积、晃动幅度或模型低置信度删除难例。

## 验证边界

- Android 构建：Windows JDK 17、Gradle 8.7、SDK 34；两个 flavor 的 Debug APK 均已生成。
- 独立 Gradle 缓存：`F:\sa\.touch-collection-runtime\gradle`，未借用 SuperAppAndroid 的缓存或构建脚本。
- JVM 测试补充真实 `org.json` 测试依赖，避免 Android 本地桩阻止 JSON 协议验证。
- iOS：本机没有 Xcode／Swift 编译器，只有源码复核，未宣称编译或真机通过。
- Git LFS 权重在本地仍可能是占位文件；本轮不训练、不下载或切换模型。构建成功不等于旧模型验证页可用。
- 本轮未安装到任何手机；六任务的人体操作、旋转、后台恢复及数据导出仍需真机验收。

### 本轮实测结果（2026-10-05）

| 验证 | 结果 |
| --- | --- |
| 新增 Android 任务／进度／协议单测 | 7 / 7 通过 |
| Python 复核、跨设备特征、第二／三层导出回归 | 29 / 29 通过，其中新增复核测试 9 项 |
| 完整 `aispect-core:testDebugUnitTest` | 53 项，51 通过、2 失败 |
| 额外原始训练入口回归 | 39 项，37 通过；2 项因环境缺少 `torch` 无法执行 |
| `assembleStandardDebug`、`assembleCrossDeviceDebug` | 均成功，未安装手机 |
| `git diff --check` | 通过 |
| iOS 编译与真机操作 | 未执行，本机没有 Xcode／Swift 编译器 |

两项 JVM 失败分别为：

- `AispectCausalPressFeatureBuilderTest.builderDoesNotUseTouchEventThatIsPhysicallyAfterImuSample`：
  期望 `0.25`，实际约 `0.00083333335`。
- `AispectLegacyFeatureBuilderTest.legacyBuilderPreservesHistoricalTimeFeatureOnlyForLegacyCaller`：
  返回矩阵为空导致空指针。

相关测试和特征实现相对仓库 HEAD 均无本轮差异；尚未在独立基线环境复现，因此只标记为未解决失败，
不将其归因为本轮或声称全部回归通过，不为让测试变绿而修改模型特征。

标准采集包：
`android/app/build/outputs/apk/standard/debug/app-standard-debug.apk`

SHA-256：
`2fe24089b4f4dfedceadbce4f58e5a69c98ee0fd3a26c2c3eefc1ef3f34e72c2`
