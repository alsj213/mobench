# Changelog

All notable changes to **mobench** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 统一本地与 CI 验证入口 `scripts/verify.py`（字节码编译、版本同步、Schema/YAML、仓库卫生、AST 签名/裸 `open()`/边界守门、核心依赖隔离与 >= 80% 分支覆盖率门禁）及端到端契约测试套件。
- 新增 `mobench/io_utils.py`，提供流式 MD5/SHA256、临时文件 + `os.replace` 原子 JSON 落盘与冲突安全目录生成 `make_run_dir`。
- `run` 新增 `--cooldown-c`、`--cooldown-per-round`、`--max-attempts` 与 `methodology.timeout_s`；`stats` / `regression` / `report` 新增 `--include-invalid`；`regression compare` 新增 `--threshold`；`power` 新增 `--duration`；`env` 新增 `--no-save`。
- `mobench --version` / `-V` CLI 选项，直接输出当前版本号。
- MCP Server 新增 `validate` 工具，支持以结构化结果消费 `validate compare`（含 `FAIL` 退出码 `2`）。
- 仓库新增 `.gitattributes` 与 `.github/dependabot.yml`，CI 复用 `scripts/verify.py`。

### Changed
- 统一 `profile` / `memory` / `trace` 共享 `measurement_session` 生命周期并在 `run.py` 闭合每轮跑前冷却、跑后温度/频率审计与下游无效 run 过滤。
- 合并 `profiling.py` 双平台重复流水线；将模型转换、校验身份参数与进程名查找下沉至 `mobench/adapters/`，`config.py` 补全运行期配置校验并委派适配器自检。
- 同步更新 `schema/mobench.schema.json`，补全 `build.required_cmake_args`、`workload` KV/深度控制、`methodology` 冷却/频率门禁/重试/超时/日志探针及 `llamacpp`/`mnn` 配置块，移除未实现消费的死字段。
- 清理仓库文档、模板与技能文件中的失效内部路径与关联项目引用，明确 `memory`/`trace` 当前仅支持 `adb`、`report` 不支持 `--json`，并修正 `CODE_OF_CONDUCT.md` 私密举报渠道。

### Fixed
- 修正功耗量纲换算（`MAH_TO_JOULE = 12.96`）、`stats` 样本标准差、跨目录 `source_log` 解析、`backend`/`platform` 分组键隔离、`mnn` 多长度 `pp<n>`/`tg<n>` 与 `cli_args` 透传、以及 `validate` A/B 两侧独立适配器与覆盖交集。
- 加固 `AdbPlatform`：精确在线状态校验、单 shell 配对热区读取、跨进程 MD5 去重推包、模型目录哈希防碰撞、`bin` 目录陈旧文件清理，以及 `HostPlatform` 文件句柄上下文管理与 hwmon 最大温度选取；`artifact_set_md5` 覆盖 `*.so.*` 并接入 `profile` / `memory` / `trace`；MCP `expect_handoff` 检查改用 `env --no-save` 实现零副作用。
- 修复 `AdbPlatform` 远程 shell 命令拼接缺失 `shlex.quote` 转义及 `trace.py` `atrace` 类别分隔符（空格而非逗号）的问题，并增加类别/事件/policy 名白名单校验。
- 修复 `llamacpp.py` `_extract_json` 在存在前导日志行且输出单元素顶层 JSON 数组时误截成内层字典的问题。
- 修复 `prepare.download` 缺失 URL 协议限制（仅允许 `http`/`https`）、下载超时与 `.part` 原子落盘问题。
- 为 `flamegraph.flame_html`、`trace._html` 与 `scripts/md2report.py` 补全 `html.escape` 转义，消除 HTML 报告注入风险。

### Removed
- 删除独立的 `scripts/mobench-t4.py`（功能已收编进 `mobench run`）、孤儿模块 `mobench/analysis/compare_precision.py`，以及 Schema 与模板中的失效字段 `workload.batch` / `methodology.power` / `methodology.thermal`。

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
