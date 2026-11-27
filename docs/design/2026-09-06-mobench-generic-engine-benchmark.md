# mobench v0.4 —— 通用引擎评测核心 设计文档

> **历史设计文档说明**：本文记录了 `mobench` 从早期插件架构重构为通用引擎评测核心（v0.4）时的初始设计方案；当前实际代码已演进至 v0.9.0，具体子命令、配置字段与实现细节请以 [`README.md`](../../README.md) 与 [`schema/mobench.schema.json`](../../schema/mobench.schema.json) 为准。

- 日期: 2026-09-06
- 状态: 历史设计归档（v0.4 初始设计）
- 关联仓库: mobench（Claude Code 插件与 Python 核心）

## 1. 背景与动机

当前 `mobile-bench` 插件被设计成对**一个约定的外部 benchmark harness 工程**做基准测试：

- 假设存在外部工程，目录固定 `build_android/src/benchmark_inference` + `models/{detection,nlp,classification}`，二进制遵守自定 CLI（`--model/--backend/--runs --json`）。
- `.benchmarkrc.yml` 放在该外部工程根，读 `device.adb`、`project.binary`、`ndk.path`。
- 构建 / 执行 / 环境控制 / 模型准备依赖**外部 13 个脚本**（`build_android.sh`、`run_benchmark_android.sh`、`download_pretrained.py`、`convert_models.sh`、`setup_test_environment.sh`、`restore_test_environment.sh`、`atrace_capture.sh`、`perfetto_trace.sh`、`profile_benchmark.sh`、`simpleperf_profile.sh`、`build_host_tools.sh`、`compile_tvm_model.py`、`setup_deps.sh`），本仓 `scripts/` 中并不存在。
- 模型集写死（mobilenetv2/resnet50/yolov8n/bert/qwen2_0.5b）；精度对比以 ONNX Runtime 为标杆。
- 执行假设平台 = Android + adb + root（batterystats、thermal_zone、governor/zygote）。

**目标场景**：用户在 llama.cpp / MNN 等引擎源码上做开发后，希望直接调用本插件对**自己 build 出的引擎产物**做性能测试与分析（延迟 / 吞吐 / 回归 / 功耗 / 热 / LLM 专项），不再依赖一个约定的外部 harness 工程。

### 头脑风暴已确认的决策

| 决策点 | 结论 |
|--------|------|
| 接入形态（核心契约） | **框架原生 CLI 适配器**：调用引擎自带基准工具（llama.cpp → llama-bench；MNN → benchmark/LLM 工具），适配器把输出归一化为统一 record |
| 目标平台 | **桌面 Linux + Android 双平台**（host 直跑 + RAPL/perf/thermal；Android 走 adb + batterystats/thermal_zone） |
| 模型供给 | **用户自备路径为主**；支持 `source` 下载；可用引擎工程自带工具做格式转换/量化 |
| 实现路径 | **方案 A：Python 编排核心**（records / adapters / platform / run / report / analysis），skills & agent 变薄编排层 |
| 配置与调用 | 新配置 **`mobench.yml`** 放被测引擎工程根；skill/CLI 以该目录为工作目录调用（CWD 式） |
| 首批适配器 | **llama.cpp + MNN**（registry 留扩展位，ncnn/tvm/mslite 后续按需补） |

## 2. 目标 / 非目标

### 目标

1. 解耦「被测对象 = 单一外部 benchmark harness 工程」的假设；被测对象变为**用户自研/自改的 llama.cpp / MNN 工程**。
2. 桌面 Linux（开发期）与 Android（端侧）都能执行；功耗 / 热 / 锁频 / 环境控制方法按平台分发。
3. 模型**自备路径为主**，同时支持 `source` 下载与「借用引擎工程自带工具」做格式转换/量化（llama.cpp 的 `convert_hf_to_gguf.py` / `llama-quantize`、MNN 的 `MNNConvert`）。
4. 消灭 prompt / SKILL 内联的命令与正则解析逻辑（消除 HTML 报告双生成、日志解析双实现的漂移风险）。
5. 保留并复用以现有分析/方法论能力：多轮统计、性能回归、功耗能效、热稳定、LLM 专项、内存、数据真实性红线。

### 非目标（本变更不做）

- ncnn / tvm / MindSpore Lite 适配器（仅提供 adapter registry 扩展位）。
- 跨引擎指标的强归一化（不做「把所有引擎压到同一条 P50/FPS 曲线」；各引擎用其自带 metric 词表，对比只在同引擎同 workload 下进行）。
- `--project <path>` 任意目录指向（v0.4 用 CWD 式：进入被测引擎工程根目录调用）。后续按需。
- MCP 封装（v0.4 不暴露为 MCP server；但核心做成**确定性 CLI**，为后续 MCP 打底）。
- 对旧 `.benchmarkrc.yml` / `benchmark_inference` 日志的向后兼容解析（v0.4 直接迁移，不双轨运行）。

## 3. 概念模型

```
mobench.yml ──▶ run() ──▶ 原始日志(tee) ──▶ adapter.parse() ──▶ records.jsonl
(引擎工程根)      │                                             │
                  ├─ platform: host | adb（执行/功耗/热/锁频）    ├─▶ analysis:
                  └─ adapter: llamacpp | mnn（命令构造+解析）      stats / regression / power / report
```

四个概念，引擎知识只在一处（adapter）：

- **platform**：执行与观测的物理层。
  - `host`：Linux 直跑；`uname`/`lscpu`/`nproc`；温度读 `thermal_zone`/`hwmon`；功耗读 RAPL（`perf`/`turbostat`）；环境控制尽力而为（nice/优先级；有 sudo 时 cpupower/清缓存）。
  - `adb`：push 二进制/模型/动态库到设备 → 远程执行 → 拉取结果；温度 `thermal_zone`（mC）；功耗 batterystats；有 root 时 governor 锁频/清缓存/zygote。
  - 无 root/sudo 时优雅降级：只记录状态并打标（沿用现状语义）。
- **adapter**：引擎唯一的耦合点。契约 = `probe() → version → build_cmd() → parse() → records`，并声明该引擎的 metric 词表。首批 `llamacpp`、`mnn`。
- **record**：引擎无关的规范化测量点（见 §5），JSONL 落盘；所有分析/报告只消费 record。
- **engine bin**：`mobench.yml` 指向用户在引擎工程里 build 出的可执行与模型路径。

## 4. 仓库结构迁移

```
（v0.4 迁移后目标结构）
mobench/                  # ← 取代 scripts/，Python 包，核心逻辑在此
  __main__.py             #   python3 -m mobench <cmd>
  cli.py                  #   子命令: run / env / stats / regression / power / report / prepare
  config.py               #   mobench.yml 载入 + 校验（对接 JSON Schema）
  records.py              #   record schema + JSONL 读写 + metric 词表
  platform/
    base.py  host.py  adb.py
  adapters/
    base.py  llamacpp.py  mnn.py
  run.py                  #   编排: 快照→环境控制→tee 执行→恢复→records
  report.py               #   消费 records 生成 Markdown/HTML
  prepare.py              #   模型下载(HF) / 借用引擎工具转换量化
  analysis/
    stats.py              #   迁入 stats_summary 逻辑
    regression.py         #   迁入 regression_compare 逻辑
    power.py              #   迁入 power_analysis 逻辑（batterystats / RAPL）
    env_snapshot.py       #   迁入 env_snapshot 逻辑（按 platform 采集）
    compare_precision.py  #   ORT 参考对比（仅 onnx 图场景，可选）
scripts/mobench            # 薄入口（pip 安装后亦可为 PATH 上 `mobench`）
schema/mobench.schema.json
skills/…                  # 重写为薄编排层（命令/解析/正则不再内联）
agents/mobile-bench-agent.md  # 7 步协议保留，step 内改调 mobench
tests/…                   # 按模块重写（fixture 驱动）
pyproject.toml
requirements.txt           # 顺带修复「依赖未形式化」
docs/design/…             # 本设计文档
```

要点：

- 现有 7 个独立脚本（parse_log / generate_report / compare_precision / env_snapshot / power_analysis / regression_compare / stats_summary）的**数学与分析逻辑保留**，迁入 `analysis/` 或作为 `records` 消费者，入口统一为 `mobench <subcmd>`。
- `mobile-bench-run/SKILL.md` 内联约 170 行 `gen_html` 与 HTML 模板**删除** → 由 `mobench report` 生成（消除 HTML 双实现、日志双解析漂移）。
- 旧的 `scripts/*.py` 独立文件在迁移完成后移除（版本内不双轨）。

## 5. mobench.yml（草案）与 record schema

### mobench.yml

```yaml
engine: llamacpp            # mnn | llamacpp（registry key）
platform:
  type: host                # host | adb
  # type: adb 时:
  # adb: /path/to/adb
  # device_id: <serial>
build:
  source_dir: .             # 引擎工程根（默认 CWD）
  bin_dir: build/bin        # 产物目录
  bench_bin: llama-bench    # 基准可执行名（adapter 契约内解析）
models:
  - id: tinyllama
    path: models/tinyllama-q8_0.gguf     # 用户自备
    kind: llm                            # llm | cv | nlp
    precision: q8_0
    source:                              # 可选：下载 + 转换
      url: hf://TheBloke/TinyLlama...GGUF
      convert_with: llama.cpp            # 借用引擎自带工具
workload:
  threads: [1, 4]
  rounds: 5                 # >= 5 才出正式统计（对齐 methodology 要求）
  prompt_tokens: 64         # LLM 专用
  gen_tokens: 128           # LLM 专用
  batch: 1                  # CV 专用（如需）
methodology:
  env_control: auto         # auto | none | 具体开关
  power: true
  thermal: true
results_dir: results
```

说明：旧 `.benchmarkrc.yml` 的 `device/project(build_dir/binary/models_dir)/ndk` 语义被 `platform/build/models/workload` 取代；`engine` 决定走哪个 adapter。

### record（records.jsonl，一行一条）

```jsonc
{
  "schema": "mobench/record@1",
  "engine": "llamacpp",           // adapter id
  "engine_version": "b1234",      // --version / git
  "workload_id": "tinyllama",     // 与 models[].id 对应
  "metric": "gen_tok_s",          // 见 metric 词表
  "value": 42.7,
  "unit": "tok/s",
  "threads": 4,
  "gpu": false,
  "precision": "q8_0",
  "round": 2,                     // 第几轮（>=1）
  "runs": 5,                      // 单次引擎调用内部重复次数
  "platform": "host",
  "device": { "os": "…", "cpu": "…", "temp_c": null },
  "source_log": "results/run_…/raw.log",   // tee 原始日志，可追溯
  "ts": "2026-09-06T10:00:00",
  "raw": { /* 引擎原始字段，适配器透传 */ }
}
```

- **metric 词表**由 adapter 声明。首批示意：
  - `llamacpp`: `prompt_eval_ms_tok`（prefill ms/token）、`gen_ms_tok`、`gen_tok_s`、`load_ms`、`mem_mb`。
  - `mnn`: `latency_ms`、`fps`、`op_*`（逐算子，若有）、LLM 路径的 `gen_tok_s` 等。
- **分析/报告只消费 record**：多轮统计、回归、功耗归因全部按 `(engine, workload_id, metric, threads, precision, round)` 分组（stats 的旧 `(backend, model, precision, threads)` 泛化为此）。

## 6. 通用流水线（7 步协议映射，对 agent/红线不变）

| 步骤 | 旧动作 | v0.4 动作 |
|------|--------|-----------|
| 0 | 读 `.benchmarkrc.yml` | 读 `mobench.yml`（config.py 校验） |
| 1 平台握手+温度 | `adb devices` + `thermal_zone` | platform.preflight：host=CPU/内核/温度；adb=设备在线/型号/温度(mC) |
| 2 代码版本 | 外部工程 git | 被测引擎 `git` + `engine --version`（adapter.probe） |
| 3 二进制验证 | glob `build_dir/**/binary` + md5 | adapter 解析出的 bench 可执行路径 + md5/大小 |
| 4 模型检查 | 固定目录树 | `mobench.yml models[].path` + 扩展名校验 + md5；缺失时 `mobench prepare` |
| 5 环境控制 | setup_test_environment.sh | platform.control（锁频/清缓存/governor，尽力而为 + 降级） |
| 6 执行(tee) | run_benchmark_android.sh | adapter.build_cmd + platform 分发执行，`tee` 原始日志 |
| 7 恢复+records+报告 | restore + grep + 内联 HTML | platform.restore → adapter.parse → records.jsonl → `mobench report`（MD/HTML） |

数据真实性红线 R1–R11 **原样保留**：每行指标可追溯到 `source_log`；来源命令变为 `mobench …` 及其 tee 的原始引擎输出；禁止旧数据/伪造/跨会话复用等规则不变。

## 7. 首批适配器行为约定

- **llamacpp**：调用引擎工程 build 出的 `llama-bench`，参数含 `-m <gguf> -p <prompt_tokens> -n <gen_tokens> -t <threads> -r <runs> --json`。`parse()` 读取其 JSON（build / models / results），映射 prompt eval 与 generation 的耗时与 tok/s 到 record；不做 ORT cosine（GGUF 无 onnx 参考可比）。
- **mnn**：CV/NLP 模型走 MNN 基准工具（`benchmark.out` 等，文本输出逐 op + 总耗时）解析出 latency/fps；若模型以 onnx 表述且本地有 ORT，可经 `compare_precision` 做参考对比（可选）。
- adapter registry：`adapters/base.py` 定义基类与注册表；v0.4 内置 `llamacpp`、`mnn`；后续新增引擎 = 新增一个 adapter 模块 + 一个测试 fixture。

## 8. 与现有文件的关系（迁移/删除清单）

| 文件/对象 | 处置 |
|-----------|------|
| 引用 13 个外部脚本、`build_android/src/benchmark_inference`、`models/{detection,nlp,classification}` 的文字与命令 | 全部删除/改写 |
| `scripts/parse_log.py` `generate_report.py` | 逻辑并入 `mobench/analysis`/`report.py`，随 records 重写；HTML 只由 `mobench report` 产出 |
| `scripts/env_snapshot.py` `power_analysis.py` `stats_summary.py` `regression_compare.py` `compare_precision.py` | 逻辑迁入 `mobench/analysis/`，按 platform + records 泛化 |
| `schema/benchmarkrc.schema.json` | 替换为 `schema/mobench.schema.json`（配置 schema 校验） |
| `skills/*`（run/model-prep/profiling/methodology/power/thermal/regression/llm/memory/integrate） | 重写为薄编排层；命令与解析交 `mobench`；`integrate` 改述为「如何新增 adapter」；`model-prep` 改述为 `mobench prepare`（下载 + 借用引擎转换工具） |
| `agents/mobile-bench-agent.md` | 7 步协议与 R1–R11 保留，step 内命令改调 `mobench`，去掉外部脚本引用 |
| `rules/mobile-bench-integrity.md` | 保留（红线来源） |
| `tests/*`（5 个文件） | 按模块语义迁移/重写，fixture 驱动，覆盖率 >= 80% |
| `.benchmarkrc.yml` 默认值/解析 | 移除（不留双轨） |
| `README.md` / `TEST_REPORT.md` | v0.4 重写 |

## 9. 测试策略

- adapters：用 fixture 日志做单元测试（llama-bench JSON、MNN 文本），覆盖解析与 metric 映射；新增引擎即新增 fixture。
- platform：host/adb 用 mock 子进程/adb，覆盖握手、环境控制、降级路径。
- records：roundtrip 读写、分组键、非法记录拒绝。
- run：用假引擎二进制做端到端编排（tee→parse→records），验证 7 步顺序与恢复。
- config：对 `mobench.yml` 用 schema 校验，缺必填/非法引擎报错。
- analysis：stats/regression/power 基于 records fixture 验证；迁移现有 bug 回归用例（如 batterystats 换行、Capacity 误读、裸 JSON 等，保留其回归价值）。
- report：MD/HTML 从 records 生成，golden 快照或断言关键区块。

## 10. 风险 / 开放问题

- **引擎 CLI 差异**：llama-bench 与 MNN 工具的输出格式/指标字段会随上游版本演化 → adapter 内解析集中、fixture 可替换、`raw` 透传兜底。
- **RAPL / 热 / 锁频在 host 与 adb 上的权限**：无 root/sudo 时指标缺失 → 沿用「缺失则打标 + 降级提示」语义，不阻塞主链路。
- **CWD 式调用**：若用户想在多个引擎 checkout 间轮换，v0.4 需切换目录；`--project` 作为后续增强。
- **旧日志/旧配置兼容**：明确不做（见非目标），需在 README 的升级说明中写明一次性迁移。

## 11. 后续动作

1. 本设计评审通过。
2. 拆解为可执行需求清单与验收用例（`schema/mobench.schema.json` + 单元测试）。
3. 规划分阶段实现（先 records+config+llamacpp/host 最小闭环，再 mnn、adb、analysis 迁移、skills/agent 重写）。
