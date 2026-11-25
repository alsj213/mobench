---
name: mobile-bench-run
description: Use when running inference performance benchmarks on a llama.cpp / MNN (or adapter-registered) engine build via the mobench core — driving the run pipeline from mobench.yml, verifying record provenance, and producing MD/HTML reports. The redline protocol (R1–R11) lives in agents/mobile-bench-agent.md.
---

# 运行通用推理基准（mobench run）

在**被测引擎工程**（llama.cpp / MNN，或已注册 adapter 的框架）的构建产物上执行性能基准。所有测量统一由 `mobench` 核心完成；本 skill 描述如何驱动它、验证产物真实、并解读结果。

## 适用场景

- 对用户自己 build 出的引擎二进制（如 `llama-bench`、`MNNBenchmark`/`llm.mnn` 工具）做性能测量
- 需要规范化的 records + 可溯源日志 + MD/HTML 报告
- 跑 host（Linux）或 Android（adb）平台

## 前置

1. 有基准入口 `mobench.yml`（schema 见 `schema/mobench.schema.json`），用 `--config <路径>` 显式定位——本机实例在 `<mobile-bench>/configs/local/`、模板在 `<mobile-bench>/configs/templates/`（分层见 `configs/README.md`）。字段含义与缺省值由 `mobench config` 载入；引擎 key：`llamacpp` / `mnn`。
2. 平台握手：host 直接可用；adb 需设备在线。`platform.type: adb` 时预检失败会报错中止（红线 R5），不会"假设设备在线"。

## 驱动方式

```bash
# 基准入口用 --config 显式定位（默认读 cwd 的 mobench.yml；以下从 <mobile-bench> 仓 cwd 跑）
CFG=configs/local/<档>.yml

# 环境快照：平台身份/温度 + 引擎版本 + binary/model MD5（每字段带来源）
python3 -m mobench env --config $CFG --json

# 模型准备三态：自备校验 / 下载 / 借用引擎工具转换
python3 -m mobench prepare --config $CFG

# 执行完整流水线
python3 -m mobench run --config $CFG
# 常用覆盖：--engine llamacpp --model <id> --threads 8 --rounds 5 --json
```

`run` 内部按序完成：配置载入 → 平台预检/温度 → 引擎版本与产物哈希 → 模型检查（缺失即中止并提示 `mobench prepare`）→ 环境控制（无控制权打标 `Env control degraded`）→ 执行（`tee` 保存原始日志）→ 恢复 → 追加规范化 records。任一失败按步报错并退出非零（红线 R4）。

## 产物与验证

| 产物 | 位置 | 验证点 |
|------|------|--------|
| 原始日志 | `log_path`（`run.json` 指明） | 非空；每行可溯源（红线 R7/R8） |
| records | `results/records.jsonl` | schema=`mobench/record@1`；每行含 engine/workload_id/metric/value/unit/`source_log` |
| run meta | `results/run.json` | 平台/温度/版本/哈希/时间戳齐全 |

### 温度
`env`/`run` 报告温度。> 45°C 打降频标注，数据不得当作正常值使用（红线 R9），建议冷却后重跑。

## 解读结果

```bash
# 多轮统计：按 (engine, workload_id, metric, threads, precision) 分组 ≥5 轮输出 mean/std/CV/P50/P90/P99
python3 -m mobench stats results/

# 报告：MD + HTML，数值标注来源记录
python3 -m mobench report results/ --out results/
```

跨框架对比、功耗、回归、长时热稳定性分别由 `mobile-bench-methodology`、`-power`、`-regression`、`-thermal` skills 覆盖。

## 红线（摘要）

R1 不编造数字 · R2 不用旧数据 · R3 每值标注来源 · R4 不静默忽略失败 · R5 不跳设备握手 · R6 流水线缺一不可 · R7 必须 tee 原始日志 · R8 同时给原始输出与表格 · R9 温度 >45°C 不使用 · R10 记录二进制/模型哈希 · R11 禁止跨会话复用/伪造日志。完整表格见 `agents/mobile-bench-agent.md`。
