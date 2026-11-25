<p align="center">
  <strong>mobench</strong> · 通用推理引擎基准测试核心
  <br/>
  Generic inference-engine benchmarking core — traceable, engine-agnostic performance measurement, profiling, validation &amp; analysis for llama.cpp, MNN, and custom adapters.
</p>

<p align="center">
  <a href="https://github.com/alsj213/mobench/actions/workflows/ci.yml"><img src="https://github.com/alsj213/mobench/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Android-orange.svg" alt="Linux / Android">
  <img src="https://img.shields.io/badge/engines-llama.cpp%20%7C%20MNN-green.svg" alt="llama.cpp / MNN">
</p>

<p align="center">
  <a href="#quickstart">快速开始</a> · <a href="#installation">安装</a> · <a href="#configuration">配置</a> ·
  <a href="#usage">使用</a> · <a href="#engine-support">引擎支持</a> · <a href="#mcp-server">MCP</a> ·
  <a href="#testing">测试</a> · <a href="#contributing">贡献</a> · <a href="#license">License</a>
</p>

---

**mobench** 是**引擎无关**的推理性能基准核心：直接驱动你自己构建的 `llama.cpp`、`MNN`（或任何已注册适配器）原生测速工具完成性能测量、剖析与精度校验。所有测量结果统一沉淀为可溯源记录（`mobench/record@1`），统计、回归、功耗、报告与多面板图表全部由同一份数据驱动。

**数据可信红线 R1–R11**（见 [`rules/mobile-bench-integrity.md`](rules/mobile-bench-integrity.md) 与 [`agents/mobile-bench-agent.md`](agents/mobile-bench-agent.md)）：不编造数字、不复用旧数据、每值标注原始日志来源、异常降频/高温（>45°C）显式告警不隐瞒。

## Table of Contents

- [Features](#features)
- [Quickstart](#quickstart)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Engine support](#engine-support)
- [Architecture](#architecture)
- [MCP server](#mcp-server)
- [Extending to a new engine](#extending-to-a-new-engine)
- [Testing](#testing)
- [Known boundaries & Roadmap](#known-boundaries--roadmap)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)

---

## Features

| 能力 | 说明 |
|------|------|
| 引擎无关 | 引擎差异收敛于 `mobench/adapters/`；内置 `llamacpp`、`mnn`，扩展无需改动核心 |
| 框架原生 CLI | 直接调用被测工程原生工具（`llama-bench`、MNN `llm_bench`/`benchmark.out`），无二次包装偏差 |
| 双平台统一 | 桌面 Linux（`host`）与 Android（`adb`）统一 run 语义；`adb` 自动同步二进制、动态库与模型目录 |
| 环境控制与门禁 | 支持温度门禁、CPU 绑核（`cpu_pin`）、cpufreq 锁频（`lock_freq`）以及运行中频率降频检测 |
| 完整可溯源 | 每次测量 tee 原始日志 → canonical records（`mobench/record@1`，含 `source_log`）→ 报告逐值标来源 |
| 一体化分析套件 | 内置 `stats` / `regression` / `validate` / `profile` / `memory` / `trace` / `power` / `report` / `runs` |
| MCP 工具面 | 原生提供 MCP server，便于 AI Agent 以可审计、fail-fast 的方式驱动基准测试 |

## Quickstart

```bash
# 1. 克隆并安装
git clone https://github.com/alsj213/mobench.git
cd mobench
pip install -e .

# 2. 准备配置文件 mobench.yml（模板见 examples/ 与 configs/templates/）
mobench prepare --config mobench.yml          # 自备校验 / 下载(sha256) / 借用引擎工具转换
mobench run     --config mobench.yml          # 完整流水线 -> records.jsonl + run.json
mobench stats   results/                      # mean / std / CV / P50 / P90 / P99
mobench report  results/ --out results/       # Markdown + HTML 报告
```

30 秒新引擎接入指南见 [`docs/adapter-guide.md`](docs/adapter-guide.md)。

## Installation

### 作为 Python 包（推荐）

要求 Python **3.9+**：

```bash
# 核心 CLI（仅依赖 PyYAML）
pip install -e .

# 按需安装可选功能组
pip install -e ".[regression]"   # 统计显著性回归检测（scipy）
pip install -e ".[mcp]"          # MCP Server（fastmcp）
pip install -e ".[dev]"          # 全部可选依赖 + pytest / pytest-cov
```

验证安装：

```bash
mobench --help                   # 12 个子命令
# 或 python3 -m mobench --help
```

### 作为 Claude Code 插件

本仓库同时包含 Claude Code 插件定义（[`.claude-plugin/`](.claude-plugin/plugin.json)），提供领域 `skills/`、`agents/` 与数据真实性 `rules/`：

```bash
claude plugins install https://github.com/alsj213/mobench
```

## Configuration

配置文件默认读取当前目录下的 `mobench.yml`，也可通过 `--config <路径>` 显式指定（完整 JSON Schema 见 [`schema/mobench.schema.json`](schema/mobench.schema.json)）：

```yaml
engine: llamacpp                 # llamacpp | mnn（对应 mobench/adapters/<engine>.py）
platform:
  type: host                     # host | adb
  # adb 平台补充:
  # adb: /path/to/adb
  # device_id: <serial>
build:
  bin_dir: ./build/bin           # 被测工程二进制目录（host）/ 本地 Android 产物目录（adb）
  bench_bin: llama-bench         # 被测测速工具名
  source_dir: .                  # 引擎源码根（用于 git commit 溯源）
workload:
  threads: [8]
  rounds: 5                      # 正式统计建议 >= 5
  prompt_tokens: 128             # LLM：prompt 长度（可选）
  gen_tokens: 128                # LLM：生成 token 数（可选）
methodology:
  cpu_pin: true                  # 自动或显式绑核（默认启用）
  lock_freq: true                # 锁频并在结束后恢复（默认启用）
models:
  - id: tinyllama
    path: models/tinyllama-q8_0.gguf          # 单文件模型
  - id: qwen2-0.5b
    path: models/qwen2/config.json            # 目录模型（MNN LLM）
    dir:  models/qwen2
  - id: squeezenet
    path: models/sq/squeezenet.mnn            # 目录式测速工具（MNN benchmark.out）
    dir:  models/sq
results_dir: results
```

更多开箱即用模板见 [`examples/README.md`](examples/README.md) 与 [`configs/README.md`](configs/README.md)。

## Usage

`mobench` 提供 **12 个确定性子命令**（均支持 `--json` 稳定输出，失败以非零码退出）：

| 子命令 | 作用 |
|--------|------|
| `prepare [model_ids]` | 模型准备三态：自备校验 / 下载 + sha256 / 借用引擎工具转换（`convert_with`） |
| `env` | 环境快照：平台身份 + 温度 / 引擎版本 / binary 与 model MD5（每字段附来源命令） |
| `run` | 完整基准流水线：preflight → 温度门禁 → 版本/哈希 → 模型检查 → 环境控制 → 执行（tee） → records |
| `profile` | 性能剖面采样：host `perf` / adb `simpleperf`，产出火焰图、热点 top 与 IPC 统计 |
| `memory` | 内存采样：在长序列运行期间轮询进程 RSS，输出内存曲线与峰值 |
| `trace` | 频率追踪：采集运行期间的 `atrace` CPU 频率轨迹 |
| `stats <inputs…>` | 分组统计 mean / std / CV / P50 / P90 / P99，CV > 5% 自动告警 |
| `report <inputs…> --format md\|html\|both\|chart` | 带来源标注的基准报告（高温 >45°C 自动打降频横幅）；`chart` 输出多面板基准图（支持 `--chart-by backend\|engine\|matrix`） |
| `regression save\|compare` | 基线保存与跨 commit 回归对比（Mann-Whitney 显著性检验 + 阈值告警） |
| `validate run\|compare` | 精度校验门禁：`run` 产出自描述 `validate.json`；`compare` 离线判定 `PASS/WARN=0`、`FAIL=2`、不可比=`1` |
| `power` | 能耗分析：每推理能耗（J）与 perf/Watt（消费 RAPL / batterystats 文本，不可用明确声明） |
| `runs` | 历史运行索引：扫描 `results/` 下所有 `run.json`，汇总时间、标签、产物 MD5、记录数与审计状态 |

`run` 支持常用覆盖参数：`--config` / `--engine` / `--model` / `--threads` / `--rounds` / `--label`。

### 精度校验（`validate`）

用于回答「某次优化或构建改动是否改变了模型输出」，并让仪器盲区**结构可见**——只跑 PPL 的工件会显式标注 `decode NOT COVERED`，绝不假装覆盖单 token decode 路径：

```yaml
validation:
  prompts: [prompts/long-01.txt]   # greedy 仪器：固定 prompt + 贪心解码 → 覆盖 decode
  gen_tokens: 64
  ppl:                             # ppl 仪器：llama-perplexity → 覆盖 prefill
    dataset: wiki.test.raw
    ctx: 1024
    chunks: 8
```

- `mobench validate run --config A.yml [--config-b B.yml]`：输出 `validate.json`（含 `controls` / `identity` 双层指纹 + coverage + 原始日志）；`--config-b` 为 A/B 同条件双列，若模型 MD5、线程数、prompt 集哈希或采样参数不一致会在**执行前**拒绝。
- `mobench validate compare --baseline golden.json [--candidate new.json]`：支持跨工件或单 A/B 工件对内判定，默认逐字节/逐位 exact 对比，也可通过 `--tolerance` / `--ppl-tolerance` 设定容差并报告首分歧位置。
- 引擎未实现校验仪器时明确返回 unavailable，不伪造结果。

## Engine support

引擎差异仅存在于 **adapter**（命令拼装 + 输出解析），核心无引擎耦合：

| engine | 被测工具 | 命令形态 | 输出解析 |
|--------|---------|----------|----------|
| `llamacpp` | llama.cpp `llama-bench` | `-m <gguf> -t N -r 1 -o json [-p -n]` | 新版 `-o json` 顶层数组（`avg_ts`）→ `prompt_eval_ms_tok` / `gen_tok_s` / `gen_ms_tok`；旧版 `--json` 经 `llamacpp: {output: --json}` 回退 |
| `mnn`（LLM） | MNN `llm_bench` | `-m <config.json> -a cpu -c 2 -t N [-p -n]` | stdout `pp<n>` / `tg<n>` + `t/s` 表格 |
| `mnn`（CV/NLP） | MNN `benchmark.out` | `<模型目录> 1 1 0 N 2` | `max/min/avg` summary 行 → `latency_ms` + `fps`；兼容 JSON / `key: value` |
| 自定义扩展 | 任意原生测速工具 | 自由定义 | 实现 `EngineAdapter` 即可复用全套分析/报告 |

## Architecture

```text
mobench/                  # 确定性 Python 编排核心
├── cli.py                #   12 子命令 CLI 入口
├── config.py             #   mobench.yml 载入与校验（BUILTIN_ENGINES）
├── records.py            #   canonical records: mobench/record@1 JSONL（source_log 溯源）
├── adapters/             #   引擎唯一耦合点（base.py + llamacpp.py / mnn.py）
├── platform/             #   host / adb（preflight·温度·绑核锁频·run/tee·restore）
├── run.py                #   基准执行流水线与运行中频率有效性检查
├── validation.py         #   精度校验仪器执行（greedy / ppl）
├── profiling.py          #   perf / simpleperf 采样与热点统计
├── flamegraph.py         #   火焰图生成
├── memory.py             #   RSS 内存曲线采样
├── trace.py              #   atrace 频率轨迹采集
├── snapshot.py           #   环境与产物哈希快照
├── prepare.py            #   模型准备（自备 / 下载 / 转换）
├── report.py             #   Markdown / HTML / 多面板基准图生成
├── runs_index.py         #   results/ 历史运行索引
├── mcp_server.py         #   FastMCP stdio 工具面
└── analysis/             #   stats / regression / correctness / power / compare_precision
configs/templates/        # 通用 YAML 配置模板（占位符）
examples/                 # host / adb 快速上手配置示例
schema/                   # mobench.schema.json
scripts/                  # 辅助脚本（mobench / mobench-t4.py / md2report.py）
tools/                    # WSL→Windows adb 路径映射包装脚本
skills/                   # 10 个领域知识 skill（Claude Code 插件）
agents/                   # mobile-bench-agent.md（7 步协议 + R1–R11）
rules/                    # 数据真实性强制规则（R1–R11）
docs/                     # 适配器接入指南、MCP 示例与架构设计文档
tests/                    # pytest 单元测试与端到端测试套件
```

## MCP server

`mobench` 可作为 MCP server 暴露给 AI Agent——所有工具经 `python -m mobench <verb> …` 子进程执行，核心零改动，失败原样带 stderr 抛错（R4）：

```bash
pip install -e ".[mcp]"
python3 -m mobench.mcp_server    # stdio 模式启动
```

| 工具 | 参数 | 作用 |
|------|------|------|
| `env` | `config` | 环境快照：平台身份 + 温度 / 引擎版本 / binary 与 model MD5 |
| `run` | `config`，可选 `expect_handoff` / `overrides` | 执行基准；指定 `expect_handoff` 时先校验二进制 MD5（fail-fast），匹配才开跑 |
| `profile` | `config`，可选 `extra` | 性能剖面采样（`flamegraph` / `top` / IPC） |
| `stats` | `inputs: list[str]`，可选 `extra` | 分组统计（records 文件或目录） |
| `regression` | `action`（`save`/`compare`），可选 `extra` | 基线保存与显著性回归对比 |
| `power` | `config` + `source`，可选 `extra` | 能耗 / perf-per-Watt 分析 |
| `report` | `inputs: list[str]`，可选 `out` / `extra` | 生成 Markdown / HTML / 图表报告工件 |
| `runs` | 可选 `results`（默认 `"results"`） | 扫描 `results/` 目录返回已完成运行的索引表 |

客户端注册示例见 [`docs/mcp.example.json`](docs/mcp.example.json)。

## Extending to a new engine

接入新框架**不需要修改核心**，只需 4 步：

1. 新增 `mobench/adapters/<engine>.py`（实现 `EngineAdapter` 并导出 `ADAPTER`）
2. 在 `mobench/config.py` 的 `BUILTIN_ENGINES` 中添加 engine key
3. 在 `schema/mobench.schema.json` 的 `engine.enum` 中同步枚举值
4. 新增 `tests/test_adapter_<engine>.py` 并更新文档

完整步骤与真实踩坑记录见 **[`docs/adapter-guide.md`](docs/adapter-guide.md)**。

## Testing

```bash
pip install -e ".[dev]"
python3 -m pytest tests/ -q --cov=mobench
```

测试套件涵盖 CLI、配置校验、双平台（host / mock adb）、引擎适配器、统计/回归/精度校验、火焰图与 MCP server（GitHub Actions 在 Python 3.9–3.12 矩阵持续验证）。

## Known boundaries & Roadmap

- **声明式零代码接入**：当前新引擎需编写轻量 adapter（约 60–120 行）。规划支持在 `mobench.yml` 中声明二进制、参数模板与输出映射的通用适配器。
- **Android 真机测试**：CI 环境无物理手机，真机路径由 mock 单测与本地实机验证共同覆盖。
- **功耗采集**：消费平台功耗工具输出（RAPL / batterystats 文本），不在核心内置采集守护进程。

## Contributing

欢迎提交 Issue 与 Pull Request！请先阅读：

- 贡献指南：[`CONTRIBUTING.md`](CONTRIBUTING.md)
- 行为准则：[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
- 变更日志：[`CHANGELOG.md`](CHANGELOG.md)

## Security

如发现安全漏洞，请勿公开提交 Issue，请按 [`SECURITY.md`](SECURITY.md) 指引通过 GitHub Private Vulnerability Reporting 私密反馈。

## License

本项目基于 [MIT License](LICENSE) 开源。
