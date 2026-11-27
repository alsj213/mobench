# benchmark 入口配置（configs）

统一收纳**基准入口配置**（mobench.yml），不再要求散落在被测引擎工程根目录。

## 分层

| 目录 | 内容 | git | 说明 |
|------|------|-----|------|
| `configs/templates/` | 通用模板（引擎×被测模型档位，路径用 `<占位符>`） | ✅ 跟踪 | 可分享；新机/新档位照此实例化 |
| `configs/local/` | 本机实例（真实路径：模型/adb/device/fork 目录） | ❌ ignore | 不入库，避免个人路径进共享仓 |

## 契约（与 skills/agents 文档一致）

- 基准入口 = 任一 `mobench.yml`，字段 schema 见 `schema/mobench.schema.json`（位置无关）。
- 用 **`--config <路径>`** 显式定位，可在任意 cwd 执行：

```bash
cd <mobench 仓>                                    # 本机实例的 results_dir 相对此 cwd
PYTHONPATH=<mobench 仓> python3 -m mobench prepare --config configs/local/<档>.yml
PYTHONPATH=<mobench 仓> python3 -m mobench run     --config configs/local/<档>.yml --json
```

- `build.source_dir` 仍指向**被测引擎仓**（git commit 溯源不受 config 位置影响）。
- 被测引擎工程根目录**不再要求**有 `mobench.yml`（旧契约已废弃）；被测仓内仅保留构建产物与 wrapper 工具。

## 模板 → 实例

模板里 `<ADB_EXE>` `<DEVICE_ID>` `<FORK_DIR>` `<FORK_BIN_DIR>` `<MODELS_DIR>` 替换为真实值，另存到 `configs/local/`：

```bash
mkdir -p configs/local
sed -e 's#<ADB_EXE>#/path/to/adb#' \
    -e 's#<DEVICE_ID>#YOUR_DEVICE_SERIAL#' \
    -e 's#<FORK_DIR>#/path/to/llama.cpp#' \
    -e 's#<FORK_BIN_DIR>#/path/to/llama.cpp/build-android/bin#' \
    -e 's#<MODELS_DIR>#/path/to/models#' \
    configs/templates/llamacpp-qwen3-0.6b.yml > configs/local/llamacpp-qwen3-0.6b.yml
```

档位（prompt/gen/threads/rounds）按测量需要编辑实例 `workload:`；不同档位建议各存一档，例如 `…-p512g1024.yml`。

## 测试条件：绑核与锁频（默认启用）

`methodology` 段新增三个字段，**默认即启用**，既有档位无需改动：

| 字段 | 默认 | 含义 |
|------|------|------|
| `cpu_pin` | `true` | 用 `taskset` 把测量进程绑定到选定 CPU 集合 |
| `pin_cpus` | `null` | `null` = 按线程数自动探测；也可 `"4-7"` / `[4,5,6,7]` 显式指定 |
| `lock_freq` | `true` | 把所选核所属 cpufreq policy 锁到 `performance`，结束后恢复 |

`env_control: none` 仍是总闸：一次关掉绑核 + 锁频 + governor。

**自动取核规则（优先整簇）**：簇以内核声明的 cpufreq policy 为准，先求「若干**完整**簇的并集、大小恰好 = 线程数」（取频率和最大者），否则取「能容纳该线程数的最高频簇」的前 N 个。以 1+3+4 的 SM8250 为例：

| threads | 选中 | 说明 |
|---------|------|------|
| 1 | `cpu7` | prime 核 |
| 2 | `cpu4-5` | 无整簇组合 → 最高频簇前 2 |
| 3 | `cpu4-6` | **整簇命中**，不跨簇 |
| 4 | `cpu4-7` | 整簇并集（prime + 大核簇） |

绝不**部分占用**某个簇——同簇线程与跨簇线程频率不一致，正是绑核要消除的噪声。

**失败分级**（按成因，实现见 `mobench/platform/base.py` 与 `mobench/platform/cpu.py`）：

- **配置错误**（显式 `pin_cpus` 含不存在的核号、格式非法）→ **中止测量**
- **环境不支持**（平台无 cpufreq，如 WSL2）→ **降级不绑核并打标**，测量照常；显式指定 `pin_cpus` 时仍会正常绑核
- **锁频失败**（无 root/sudo）→ **降级打标 `Env control degraded`**

**可溯源**：实际生效的核集合、掩码、锁频状态写入 `run.json` 与**每条 record** 的 `device` 字段，报告与图表标注「测量条件」列。该状态同时是**分组维度**——绑核与未绑核的数据不会被合并统计。

```bash
# 关掉绑核（例如在不关心的快速验证里）
methodology:
  cpu_pin: false
# 复现单核实验
methodology:
  pin_cpus: "7"
```

## 插件读取规则

- **mobench 核心是插件；config 是数据，不属于插件缓存副本**（`~/.claude/plugins/cache/.../<ver>/` 会被重装覆盖）。
- 会话内引用 config 一律指向 **mobench 源仓** 的 `configs/`：本仓 cwd 用相对路径；跨目录用绝对路径。
- 插件技能/agent 装载到别机时，随插件分发的是 `configs/templates/`；`configs/local/` 需按上节实例化后才可 `run`。
- 运行产物（records/results）落 cwd 下 `results/`，已在根 `.gitignore` 忽略。
