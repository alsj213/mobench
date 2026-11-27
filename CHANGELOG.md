# Changelog

All notable changes to **mobench** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `mobench --version` / `-V` CLI 选项，直接输出当前版本号。
- MCP Server 新增 `validate` 工具，支持以结构化结果消费 `validate compare`（含 `FAIL` 退出码 `2`）。
- 仓库新增 `.gitattributes` 与 `.github/dependabot.yml`，CI 补强字节码编译检查、`--cov-fail-under=80` 门禁及配置 Schema/卫生扫描。

### Changed
- 同步更新 `schema/mobench.schema.json`，补全 `build.required_cmake_args`、`workload` KV/深度控制、`methodology` 冷却/频率门禁/日志探针及 `llamacpp`/`mnn` 配置块。
- 清理仓库文档、模板与技能文件中的失效内部路径与关联项目引用，明确 `memory`/`trace` 当前仅支持 `adb`、`report` 不支持 `--json`，并修正 `CODE_OF_CONDUCT.md` 私密举报渠道。

### Fixed
- 修复 `AdbPlatform` 远程 shell 命令拼接缺失 `shlex.quote` 转义及 `trace.py` `atrace` 类别分隔符（空格而非逗号）的问题，并增加类别/事件/policy 名白名单校验。
- 修复 `llamacpp.py` `_extract_json` 在存在前导日志行且输出单元素顶层 JSON 数组时误截成内层字典的问题。
- 修复 `prepare.download` 缺失 URL 协议限制（仅允许 `http`/`https`）、下载超时与 `.part` 原子落盘问题。
- 为 `flamegraph.flame_html`、`trace._html` 与 `scripts/md2report.py` 补全 `html.escape` 转义，消除 HTML 报告注入风险。

## [0.9.0] - 2026-10-16

### Added
- **完整 12 子命令 CLI**：`run`、`profile`、`memory`、`trace`、`env`、`stats`、`regression`、`validate`、`power`、`report`、`prepare`、`runs`。
- **双平台执行与环境控制**：桌面 Linux (`host`) 与 Android (`adb`) 统一流水线，支持自动目录同步、温度门禁、CPU 绑核（`cpu_pin`）与锁频（`lock_freq`）以及运行中频率有效性与降频门禁。
- **内置引擎适配器**：`llamacpp`（`llama-bench` 新旧 JSON 契约）与 `mnn`（`llm_bench` / `benchmark.out`），引擎差异完全收敛至 `mobench/adapters/`。
- **精度校验门禁（`validate`）**：支持 `greedy`（覆盖 decode）与 `ppl`（覆盖 prefill）双仪器校验，A/B 同条件强校验与离线对比判定（`PASS/WARN=0`、`FAIL=2`、不可比=`1`）。
- **分析与可视化报告**：`stats`（多轮统计与 CV 告警）、`regression`（基线保存与 Mann-Whitney 显著性回归对比）、`power`（RAPL / batterystats 能耗分析）、`report`（Markdown / HTML / 多面板基准图）、`profile`（perf / simpleperf 火焰图与 IPC）、`memory`（RSS 曲线与峰值）、`trace`（atrace 频率追踪）及 `runs`（历史运行索引）。
- **MCP Server 工具面**：提供 `env`、`run`、`profile`、`stats`、`regression`、`power`、`report`、`runs` 共 8 个 MCP 工具，支持 `expect_handoff` 二进制 MD5 前置锚定校验。
- **Claude Code 插件资产**：内置 10 个领域 `skills/`、核心 `agents/mobile-bench-agent.md` 与数据真实性红线 `rules/mobile-bench-integrity.md`（R1–R11）。

[Unreleased]: https://github.com/alsj213/mobench/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/alsj213/mobench/releases/tag/v0.9.0
