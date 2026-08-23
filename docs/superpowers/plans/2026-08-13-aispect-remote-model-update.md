# Aispect Remote Model Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让当前 Aispect Android 应用完整支持后端分配、下载、校验、原子激活和运行时加载 `time_grid_groupnorm_cnn_v1` 的远程模型，同时保留内置模型回退。

**Architecture:** 在 `aispect-core` 内新增与旧 collector 解耦的远程模型更新子系统：HTTP 客户端只负责 API 传输，bundle 校验器只负责合同和哈希，模型存储器只负责暂存/原子激活，更新协调器负责下载和回执。运行时通过一个可替换的模型源优先读取已激活 bundle，失败时继续读取 APK assets；当前触摸期间不切换，安全时机统一 reload。

**Tech Stack:** Java 17, Android library/application, WorkManager 2.9.1, `HttpURLConnection`, `org.json`, JUnit 4.

---

### Task 1: 新模型合同与校验器

**Files:**
- Create: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectRemoteBundleValidator.java`
- Create: `android/aispect-core/src/test/java/com/zhifaios/eyes/aispect/AispectRemoteBundleValidatorTest.java`
- Modify: `FunctionAndDataStructureDescription.md`

- [ ] 写测试覆盖 `9x23 delta-gravity`、`9x20 no-gravity` 合同、标签顺序、张量形状、非法架构和 manifest 文件哈希。
- [ ] 运行测试确认先失败。
- [ ] 实现仅支持 `time_grid_groupnorm_cnn_v1` 的通用合同校验器，保留 featureContract 和 featureNames 精确匹配。
- [ ] 运行测试确认通过。

### Task 2: 安全下载存储与更新协调器

**Files:**
- Create: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectRemoteModelStore.java`
- Create: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectRemoteModelUpdater.java`
- Create: `android/aispect-core/src/test/java/com/zhifaios/eyes/aispect/AispectRemoteModelStoreTest.java`
- Create: `android/aispect-core/src/test/java/com/zhifaios/eyes/aispect/AispectRemoteModelUpdaterTest.java`
- Modify: `android/aispect-core/build.gradle`

- [ ] 测试暂存目录、路径段校验、原子激活、当前/上一版本和损坏 bundle 拒绝。
- [ ] 测试 updater 对 artifact 大小/SHA-256、assignment 合同和更新状态的处理。
- [ ] 实现最小可用存储及 updater 接口，网络和文件系统均可注入测试。

### Task 3: 运行时远程模型优先加载和安全 reload

**Files:**
- Modify: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectImpactCNNClassifier.java`
- Modify: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectTimeGridGroupNormRuntime.java`
- Create/modify tests under `android/aispect-core/src/test/java/com/zhifaios/eyes/aispect/`

- [ ] 增加从已激活目录读取 weights/scaler 的入口，复用现有 JSON runtime loader。
- [ ] 远程 bundle 优先，找不到或校验失败时回退到 catalog assets。
- [ ] 绑定 `ModelRuntimeController`，触摸活动期间延迟 reload。

### Task 4: Android 更新调度与后端合同

**Files:**
- Create: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectModelHttpClient.java`
- Create: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectModelRefreshWorker.java`
- Create: `android/aispect-core/src/main/java/com/zhifaios/eyes/aispect/AispectModelUpdateScheduler.java`
- Modify: `android/app/build.gradle`
- Modify: `android/app/src/main/AndroidManifest.xml`
- Modify: `C:/Users/血饮/Desktop/work-content/轻重触Web端后端/app/services/model_validation.py`
- Modify: `C:/Users/血饮/Desktop/work-content/轻重触Web端后端/tests/test_sdk_manifest.py`

- [ ] 增加 Android 新 feature schema，后端接受并校验两个 9 帧模型合同。
- [ ] Android 端实现手动/前台/周期拉取，保留离线 assets 回退。
- [ ] 构建并运行 Android 单元测试与后端测试。

### Task 5: 文档和最终验证

**Files:**
- Modify: `FunctionAndDataStructureDescription.md`
- Modify: `docs/data/四层数据链路说明.md` only if runtime boundary changed

- [ ] 更新接口、数据结构、错误状态和模型激活优先级。
- [ ] 运行完整 Gradle 构建、单元测试和后端 pytest。
- [ ] 检查 git diff，确认没有覆盖用户既有改动。
