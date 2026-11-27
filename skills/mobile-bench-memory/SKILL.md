---
name: mobile-bench-memory
description: Use when analyzing inference memory usage — RSS/PSS reporting, native-heap growth over time, and leak detection. Sampled from the platform around mobench run and aggregated as canonical records.
---

# 内存占用分析

深入分析推理进程内存：区分 RSS 与 PSS、输出峰值与稳态、观察 native heap 增长并做泄漏判定。由 `mobench memory` 驱动真实采样与曲线生成。

## 命令化（`mobench memory`，adb）

需要进程内存曲线时用一条命令在真机完成「后台长生成 + 轮询 VmRSS + 曲线」：

```bash
# 产物: results/<engine>/mem_<ts>/
#   mem.html（RSS 曲线图）· mem.csv（时间序列）· mem.json（峰值/稳态/KV 增长）· status_samples.log（原始采样）
python3 -m mobench memory --config configs/local/<档>.yml --threads 4 --gen 512 --json
# --gen 给大生成值(如 512/1024)才有够长的 decode 采样窗口; --interval/--max-sec 调轮询
```

- 取数：被测引擎在 adb 会话运行，host 轮询 `/proc/<pid>/status` 的 VmRSS（默认每 0.2s）；曲线与 `kv_growth` 均来自真实采样，原始输出落 `status_samples.log` 溯源。
- `peak_rss_kb` 含模型加载；`kv_growth_kb` = decode 中段→末段 RSS 中位差（排除加载，反映 KV/上下文随生成的增长）；`possible_leak` 仅当末段单调不降且增长 >10MB 才置位。
- 单进程一次生成的 RSS 增长主要是 KV cache，**不是泄漏**；连续多次推理的泄漏判定留待多-run 扩展。

## 取数方式（可溯源）

内存是**进程维度**指标，核心 run 不内置采样；用被测平台的真实命令在负载前后/期间取数，并把**原始输出存成日志**作为 source：

- host Linux：`/proc/<pid>/status`（VmRSS）、`/proc/<pid>/smaps` 聚合
- adb：`dumpsys meminfo <pkg|pid>`（PSS/RSS）、`adb shell top`/`/proc/<pid>/status`

把每次采样命令的输出保存为日志文件，随后按规范 schema 追加为 records（每行含 engine/workload_id/metric/value/unit + `source_log` 指向该日志）。这样 `mobench stats`/`report` 可统一聚合。禁止手工填数（红线 R1/R11）——任何值都必须对应一份真实命令日志。

## 指标与报告

| 指标 | metric 建议 | 说明 |
|------|-------------|------|
| RSS 峰值/稳态 | `rss_kb` | /proc 来源 |
| PSS | `pss_kb` | dumpsys meminfo |
| native heap | `heap_kb` | 随推理次数采样 |

报告必须区分 RSS/PSS 并标注来源命令；输出峰值与稳态。

## 泄漏判定

连续 N 次推理后采样 RSS：若持续增长且不回落、累计增量 > 10 MB（单调不降）→ 输出 `Possible memory leak`，附增长曲线证据（时间/次数 vs heap 的序列）。

## 关联

与 `mobile-bench-llm` 的长上下文 KV-cache 趋势、`mobile-bench-thermal` 的长时间负载叠加采样时，保持同一 records/日志口径，统一溯源。
