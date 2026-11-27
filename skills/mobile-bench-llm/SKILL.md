---
name: mobile-bench-llm
description: Use when benchmarking LLM inference — decode tokens/s, prefill latency, TTFT, KV-cache/memory, and parameter-matrix sweeps. Built on the llamacpp / mnn adapters' gen_tok_s and prompt_eval metrics, aggregated through mobench records.
---

# LLM 专项指标

LLM 推理需要区分 prefill 与 decode、量化生成速度与首 token 延迟；指标由 `mobench` 引擎适配器归一化产出并按参数矩阵汇总。

## 指标来源

`mobench run` 的 adapter 解析决定产出哪些指标：

| 引擎 | decode 速度 | prefill |
|------|-------------|---------|
| llamacpp（`llama-bench --json`） | `gen_tok_s` | `prompt_eval_ms_tok`（每 token prefill 毫秒） |
| mnn（LLM 工具） | `gen_tok_s` | — |

decode 阶段的生成速度与 prefill 延迟分开报告；报告标注 `max-tokens`/`batch`/prompt 长度等参数（参数写入 workload/run meta）。

## 用法

```bash
# 单配置测量 → records
python3 -m mobench run --engine <engine> --model <id> --threads <N> --rounds 5

# 跨参数组合矩阵（不同 precision/threads/上下文/prompt 档位）分别 run 后统一统计
# run 记录落在 results/run_*/records.jsonl，stats/report 对目录递归聚合
python3 -m mobench stats results/
python3 -m mobench report results/ --out results/
```

stats 按 `(engine, workload_id, metric, threads, precision)` 分组，即可横向比较不同量化精度/线程数的生成速度——这就是参数矩阵的汇总。标出最优配置。

## 图表收尾（多 prompt 档位）

表格之外用内置图表报告收尾（`report --format chart` 输出自包含 HTML、内联 SVG、深/浅主题自适应）：

```bash
python3 -m mobench report results/ --format chart --chart-by engine --out results/
# 产物: results/benchmark_chart.html
```

- 面板 = 平台 × 模型 × (Prefill / Decode)，**x 轴 = Prompt Length**（取自 record 的 `params.prompt_tokens`），曲线/柱按 `--chart-by` 分组（`backend` 默认 / `engine` / `matrix`）。
- 完整曲线需矩阵内含 **≥2 个 prompt 档位**；单档位只剩单点。定点单档对比（如两引擎各一次 p512）更适合按单点手绘柱状/趋势图，而不是内置的多档位面板图。

## 超出核心指标的项

TTFT、KV-cache/峰值内存、长上下文内存趋势等核心未内置的维度，按被测引擎工具可输出为准，用**真实工具日志**驱动（见 `mobile-bench-profiling` 的平台取数 + `mobile-bench-memory` 的内存曲线方法）。任何附加指标必须能溯源到日志（红线 R3/R7），禁止估算。

长上下文增长趋势发现非线性风险时，纳入报告结论。红线协议遵循 agent 文档。
