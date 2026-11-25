---
name: mobile-bench-integrate
description: "Use when adding a new inference framework backend (adapter) to the mobench core — the exact files to touch, the EngineAdapter contract, and fixture coverage, so run/report/analysis stay engine-agnostic. Detailed walkthrough + real-world pitfalls: docs/adapter-guide.md"
---

# 接入新引擎适配器（mobench）

mobench 把引擎差异收敛在 `mobench/adapters/`：**每框架一个模块，暴露 `ADAPTER`**。核心
（run/report/analysis/records/platform/CLI）不感知引擎，只消费 records。

完整教程（四步、模型声明、真实踩坑）见 **`docs/adapter-guide.md`**。本 skill 是可执行清单。

## 要改哪些文件

| 动作 | 文件 | 内容 |
|------|------|------|
| ✅ 新增 | `mobench/adapters/<engine>.py` | `EngineAdapter`：`_bench_name` / `resolve_binary` / `version` / `build_cmds` / `parse` + 末尾 `ADAPTER = <Class>` |
| ✅ 加一行 | `mobench/config.py` | `BUILTIN_ENGINES` 加 `<engine>`（否则 `mobench.yml` 报错并列出现有适配器） |
| ✅ 加枚举 | `schema/mobench.schema.json` | `properties.engine.enum` 加 `<engine>` |
| ✅ 新增 | `tests/test_adapter_<engine>.py` | registry / `build_cmds` 命令形态 / `parse` 真实样例与不可解析路径 / `version` 兜底 |
| 🔁 建议 | `README.md` 契约表、`docs/adapter-guide.md` | 记录该引擎的命令与输出形态 |

**不要改**：`run.py`、`report.py`、`records.py`、`platform/*`、`analysis/*`。

## 实现步骤

1. **先拿真实事实**：被测工具的二进制路径、真实 CLI 形态、真实输出片段（JSON 对象/数组？
   表格行？`key:value`？summary 单行？）。别照旧文档猜——llama.cpp 曾把 `--json`→`-o json`
   且输出顶层结构变化。
2. **写 adapter**：契约见 `mobench/adapters/base.py`，模板对照 `llamacpp.py`/`mnn.py`。
   - `resolve_binary` 用 `_find_binary(cfg, name)`（先 `build.bin_dir` 后 PATH）。
   - `build_cmds` 返回真实 CLI；模型可能是文件（`model.path`）、目录模型（`path`=config +
     `dir`=目录）、目录式工具（`dir`=单模型目录）。
   - `parse` 读 tee 日志 → records；**每条必须带 `source_log`**；解析失败抛 `AdapterError`。
   - metric 方向一致：延迟 `*ms*`/`latency_ms`（低好）；吞吐 `*tok_s`/`fps`（高好）。
   - `version()`：无参裸跑会崩的工具（MNN `llm_bench`）跳过探针，走 `git -C build.source_dir`。
3. **注册**：config `BUILTIN_ENGINES` + schema `engine` enum。
4. **fixture 与验证**：
   ```bash
   python3 -m pytest tests/ -q        # 全绿（覆盖率 ≥80% 门槛）
   ```
   host 真跑一轮 →（如适用）adb 真机跑一轮。

## adb 提示

`bin_dir` 指向**本地 Android 产物目录**（binary+动态库同一次构建、自洽）；mobench 整目录
push 并以 `LD_LIBRARY_PATH=.` 运行。`models[].dir` 会把目录模型/模型目录同步到设备。
设备上多版本混装库会导致 `cannot locate symbol`——务必用同一次构建的产物。

## 完成判据

- `python3 -m mobench env` / `run` 在 host 与 adb（如适用）对该引擎各跑通一轮
- records 的 `source_log` 可溯源到真实日志
- 单测覆盖 registry、命令形态、解析（含不可解析拒绝）

详细教程与真实踩坑清单：`docs/adapter-guide.md`。
