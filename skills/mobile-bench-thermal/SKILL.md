---
name: mobile-bench-thermal
description: Use when assessing thermal stability and throttling under sustained load — soak test (15-30 min), time-vs-performance throttling curve, and stability index, consuming mobench run records and env temperatures.
---

# 热稳定性与降频（Soak test）

评估持续负载下的热稳定性，量化降频对性能的影响（仿 AI Benchmark throttling 测试），基于 `mobench run` 时间序列与 `mobench env` 温度快照执行判读。

## 方法

1. **Soak**：对同一 workload 持续 15–30 分钟压力负载，**每 5 分钟**跑一轮 `mobench run`，同时取温度：
   ```bash
   python3 -m mobench run --engine <engine> --model <id> --threads <N> --rounds 5
   python3 -m mobench env --json        # 记录当前温度与平台状态
   ```
2. **时间序列**：每轮的 `run.json`/`records.jsonl` 带时间戳与温度；把各轮 P50/吞吐与温度整理为结构化时间序列。
3. **稳定性指数** = 末尾区间平均性能 / 起始区间平均性能。等级判定：≥0.9 正常；明显低于 0.9 为「降频显著」。

## 判读

- 首轮性能下降 > 10%：定位 `Throttle onset: 第 N 轮, 温度 xx°C`，附前后数据对比。
- 每轮温度 > 45°C：按红线 R9 处理——如实报告，数据不当作正常值（建议冷却后重测）。
- 单点温度阈值检查不足以定性；结论要基于曲线与稳定性指数。

## 数据

序列中各点都追溯到 `mobench run` 的 tee 日志 / `env` 温度来源，禁止凭印象补点。最终结论与曲线数据一并交给用户。
