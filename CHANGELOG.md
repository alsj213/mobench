# Changelog

All notable changes to **mobench** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2026-10-16

### Added
- **完整 12 子命令 CLI**：`run`、`profile`、`memory`、`trace`、`env`、`stats`、`regression`、`validate`、`power`、`report`、`prepare`、`runs`。
- **双平台执行与环境控制**：桌面 Linux (`host`) 与 Android (`adb`) 统一流水线，支持自动目录同步、温度门禁、CPU 绑核（`cpu_pin`）与锁频（`lock_freq`）以及运行中频率有效性与降频门禁。
- **内置引擎适配器**：`llamacpp`（`llama-bench` 新旧 JSON 契约）与 `mnn`（`llm_bench` / `benchmark.out`），引擎差异完全收敛至 `mobench/adapters/`。
- **精度校验门禁（`validate`）**：支持 `greedy`（覆盖 decode）与 `ppl`（覆盖 prefill）双仪器校验，A/B 同条件强校验与离线对比判定（`PASS/WARN=0`、`FAIL=2`、不可比=`1`）。
- **分析与可视化报告**：`stats`（多轮统计与 CV 告警）、`regression`（基线保存与 Mann-Whitney 显著性回归对比）、`power`（RAPL / batterystats 能耗分析）、`report`（Markdown / HTML / 多面板基准图）、`profile`（perf / simpleperf 火焰图与 IPC）、`memory`（RSS 曲线与峰值）、`trace`（atrace 频率追踪）及 `runs`（历史运行索引）。
- **MCP Server 工具面**：提供 `env`、`run`、`profile`、`stats`、`regression`、`power`、`report`、`runs` 共 8 个 MCP 工具，支持 `expect_handoff` 二进制 MD5 前置锚定校验。
- **Claude Code 插件资产**：内置 10 个领域 `skills/`、核心 `agents/mobile-bench-agent.md` 与数据真实性红线 `rules/mobile-bench-integrity.md`（R1–R11）。
