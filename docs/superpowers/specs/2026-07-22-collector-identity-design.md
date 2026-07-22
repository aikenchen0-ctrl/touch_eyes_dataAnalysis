# 采集者与设备标识设计

## 目标

在不改变现有轻重、手指、运动状态和单一会话采集流程的前提下，为每条 Android 原始样本增加用户名称与匿名稳定设备标识。

## 数据结构

原始 JSON 的 `schemaVersion` 升级为 `3`，新增：

```json
"collector": {
  "userName": "用户填写的名称",
  "deviceId": "App 自动生成的匿名稳定标识"
}
```

`deviceId` 首次运行时使用 UUID 生成并保存到应用私有偏好设置，后续采集保持稳定。它不读取 IMEI、手机号、序列号或其他硬件隐私标识。

## 交互

采集区域增加用户名称输入框。用户名称去除首尾空格后不能为空；未填写时禁止开始采集并显示明确提示。开始采集后，用户名称与现有采集标签一起锁定；停止采集后恢复编辑。

## 数据流

`MainActivity` 读取用户输入并交给 `AispectCollectionController`。控制器在会话开始时冻结 `userName` 与 `deviceId`，保存样本时传给 `AispectDataLogger`。Logger 只负责将结构化身份对象写入 JSON。

## 兼容性

保留所有现有顶层字段和文件命名。读取旧版 `schemaVersion: 2` 数据时将缺少 `collector` 视为未知采集者，不改变已有训练脚本行为。

## 验证

- 未填写用户名称时不能开始采集。
- 采集中不能修改用户名称。
- 同一次安装产生的多个样本具有相同 `deviceId`。
- 每条新样本包含正确的 `collector.userName` 和 `collector.deviceId`。
- 现有 Android Debug APK 能成功构建并安装。

