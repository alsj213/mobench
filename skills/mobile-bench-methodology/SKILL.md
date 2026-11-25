---
name: mobile-bench-methodology
description: Use when a benchmark experiment needs statistical rigor — warmup handling, multi-run statistics (mean/CV/P50), environment state checks, fair cross-framework comparison, and reproducibility snapshots. Maps to mobench stats / env / report and the methodology spec.
---

# 测评方法学（多轮统计 · 环境 · 可复现性）

正式测量需要统计可信、可复现、跨框架公平。本 skill 是 `agents/mobile-bench-agent.md` 7 步协议之上的统计/环境增强层，规范落地于 `openspec/specs/benchmark-methodology/spec.md`。

## 多轮统计与方差

测量以 ≥5 轮为一组，记录归一化为 `mobench/record@1`，按 `(engine, workload_id, metric, threads, precision)` 分组汇总：

```bash
python3 -m mobench stats results/records.jsonl        # mean/std/CV/P50/P90/P99
python3 -m mobench report results/records.jsonl --out results/   # MD/HTML 报告
```

- 轮次 < 5：stats 标注不足（不输出统计量）。
- CV > 5%：标注 `High variance (CV=xx%)`，提示检查降频/后台进程/环境控制。

## 预热与冷启动

正式测量前做 warmup ≥ 10 次；首轮冷启动耗时不计入统计。warmup 次数从 `mobench.yml` / `--rounds` 读取，不得硬编码。不足 10 次时报告标注 `Warmup insufficient (< 10)`。

## 环境控制与状态检查

`mobench run` 执行前记录电量/充电/温度等平台状态；无控制权时打标 `Env control degraded`，绝不假装已控制。充电或温度 > 45°C 的数据按红线 R9 处理（降频风险，不使用/如实标注）。

## 可复现性快照

```bash
python3 -m mobench env --json
```

输出含平台身份/OS/内核、被测引擎 commit/version、binary 与模型 MD5、温度；每字段带来源命令。作为 `mobench regression` 基线对比的环境依据。

## 公平对比协议

- 跨引擎对比前对齐：线程数、精度、输入/上下文长度。
- 锁定引擎与模型版本（`env` 快照中的 commit/version/MD5）。
- 实际加速器与声明不符 → 报告标注 `Accelerator mismatch`。
- 功耗维度 → `mobile-bench-power`；跨 commit 回归 → `mobile-bench-regression`；长时间热稳定性 → `mobile-bench-thermal`。

## 数据真实性

遵守红线 R1–R11（见 agent）。所有统计值必须来自 records（其 `source_log` 追溯到 tee 日志），禁止手工构造行、禁止跨会话复用。
