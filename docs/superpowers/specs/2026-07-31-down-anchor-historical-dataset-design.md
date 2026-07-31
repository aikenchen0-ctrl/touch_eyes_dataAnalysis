# Historical Down-Anchor Dataset Design

## Goal

从不可变的 `data/deduplicated_raw/` 中分别构建两个历史数据集：

1. A 数据集：按下时刻位于已保存真实 IMU 帧覆盖范围内。
2. B 数据集：能够围绕按下时刻提取完整真实 IMU 窗口 `-5...+9`。

两个数据集独立输出，不混合，不覆盖、不改写原始 JSON。

## Source Contract

数据源为：

- `data/deduplicated_raw/manifest.csv`
- manifest 引用的 89,385 个原始 JSON

按下时间优先取：

```text
touchFrames[0].eventTimeMillis / 1000
```

缺失时回退到：

```text
touchEvents[0].timestamp
```

抬起窗口锚点取 `anchorTimestamp`。每个已保存 IMU 帧的绝对时间按以下方式恢复：

```text
anchorTimestamp + relativeTimestamp
```

并使用 `timeSinceTouchDown` 交叉验证。

## Placeholder Detection

旧窗口构建器会用全零 IMU 帧补齐越界位置。补零帧不能作为 A 或 B 的有效证据。

当一帧的以下动态字段全部为零时，将其判为占位帧：

- `delta`
- `x / y / z`
- `rotationRateX / rotationRateY / rotationRateZ`
- `gravityX / gravityY / gravityZ`
- `x_norm / y_norm`

判定结果写入审计字段，不修改源帧。

## Dataset A

目录：

```text
data/down_anchor_historical_20260731/a_down_inside_saved_window/
```

入选条件：

1. JSON、按下时间、锚点和采样率有效。
2. 至少存在两个真实 IMU 帧。
3. 真实帧中至少一帧位于按下时刻之前或等于按下时刻。
4. 真实帧中至少一帧位于按下时刻之后或等于按下时刻。

这表示按下时间落在真实已保存 IMU 时间范围内。A 不保证按下前有 5 帧，也不保证按下后有 9 帧。

每条 manifest 记录：

- 源样本 ID 与路径
- schema、采集者、设备、日期、会话、类别
- 按下时间、锚点、采样率
- 虚拟按下帧位置
- 最近真实帧索引和时间误差
- 按下前/后真实帧数量
- 真实帧和占位帧数量

## Dataset B

目录：

```text
data/down_anchor_historical_20260731/b_complete_down_minus5_plus9/
```

入选条件：

1. 首先满足 A。
2. 按真实时间排序后，选取距离按下最近的真实帧作为 0 帧。
3. 0 帧之前至少连续存在 5 个真实帧。
4. 0 帧之后至少连续存在 9 个真实帧。
5. 相邻帧间隔不得超过基于样本采样率计算周期的 1.75 倍。
6. 0 帧与按下时间的误差不得超过采样周期的 0.75 倍。

B 的派生样本只保存15个选中帧，并重新编号为：

```text
-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9
```

每个派生 JSON 同时保留源 ID、源路径、原始帧索引、原始时间和按下相对时间，保证可审计。

如果没有任何样本满足 B，仍然生成空 manifest 和 summary，明确记录 `retained = 0`，不得通过补零或外推制造样本。

## Outputs

A 输出：

- `manifest.csv`
- `summary.json`

B 输出：

- `manifest.csv`
- `summary.json`
- `samples/*.json`

总输出：

- `data/down_anchor_historical_20260731/summary.json`
- 排除原因计数
- 按 schema、采集者、设备、类别统计
- A/B 样本 ID 交集一致性检查，B 必须是 A 的子集

## Implementation

新增独立脚本：

```text
android/analysis/build_historical_down_anchor_datasets.py
```

脚本使用流式 JSON 读取，避免把 3.5 GB 数据同时载入内存。输出先写入临时目录，全部成功后再替换目标目录，防止中断留下半成品。

## Verification

按 TDD 实现以下测试：

1. 按下位于真实帧范围外时排除 A。
2. 按下位于真实帧范围内但前后帧不足时只进入 A。
3. 完整连续 `-5...+9` 时同时进入 A 和 B。
4. 补零帧不能帮助样本进入 A 或 B。
5. 帧间隔异常时不能进入 B。
6. B 派生帧严格为15个且重新编号正确。
7. B 样本 ID 必须全部存在于 A。
8. 空 B 仍生成有效 manifest 和 summary。

完成后运行脚本处理全量89,385条数据，并核对输出计数、JSON可解析性、ID唯一性和源文件未修改。
