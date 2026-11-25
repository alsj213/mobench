---
name: mobile-bench-profiling
description: Use when analyzing latency bottlenecks in an inference run — operator-level timing, CPU sampling, and flame graphs, dispatched by platform (host perf / adb simpleperf & perfetto) and pointed at the engine under test's own profiling facilities.
---

# 推理性能剖析（方法文档级）

定位延迟瓶颈：先看算子级时间，再 CPU 采样，最后叠火焰图。剖析目标是**被测引擎自己的构建产物**，不依赖本仓库内置采样脚本。

## 平台分发

| 平台 | 系统级工具 | 说明 |
|------|-----------|------|
| host Linux | `perf record/report` | CPU 采样 + 调用栈 |
| adb Android | `simpleperf`、`perfetto`/`atrace` | 采样 + 系统 trace（sched/freq） |

工具链路径：host `perf` 由 `mobench` 探测（装 `linux-tools`，WSL 回退 `/usr/lib/linux-tools-*/perf`）；adb `simpleperf` 取被测工程/NDK 自带版本。平台身份与设备状态按 `mobench` 的 adb/host 语义（离线即中止）。

## 命令化（`mobench profile`，host/adb 双平台）

需要火焰图/热点时，一条命令完成「采样 → 折叠 → 渲染」，平台由 config `platform.type` 分发：

```bash
# host(Linux) 剖析：perf record -g + perf stat -> flamegraph/top/IPC
python3 -m mobench profile --config configs/local/<档>-host.yml --threads 4 --json
# 真机(adb) 剖析：simpleperf record + NDK report_sample 折叠
python3 -m mobench profile --config configs/local/<档>.yml --threads 4 --json
# 产物均为 results/<engine>/profile_<ts>/: flamegraph.html · top.txt · stat.txt · profile.json
# 可选: --freq 采样率(默认2000) · --events <stat 事件集> · --ndk-dir <NDK simpleperf 目录>
```

- host 前置：装 `linux-tools`（`sudo apt-get install linux-tools-generic`）。WSL 的 `/usr/bin/perf` wrapper 会因内核版本树缺失而失败——`mobench` 自动回退扫描 `/usr/lib/linux-tools-*/perf`。
- host 热点以引擎 x86 算子核为主（MNN `_AVX_MNNGemmInt8AddBiasScale_16x4_{Unit,w4}` int8 GEMM）；fp 调用栈对无帧指针的中段帧会留 `[unknown]` 间隙（归并为稳定占位，不打断路径）。
- adb 依赖：设备自带 `simpleperf`；本机需 NDK `simpleperf/report_sample.py`（符号化折叠）。采样窗口含进程加载+推理，短生成档会偏 load/初始化——想看 decode 稳态热点请用较长生成档（如 g≥512）。
- 火焰图宽度 ∝ 采样占比；IPC/cache-miss 由 stat 归一（`simpleperf stat` / host `perf stat`，多路复用时 events 只覆盖部分运行时间，比例仍有效）。

## 命令化（`mobench trace`，adb/atrace freq）

需要 CPU 频率/调度证据（跑在小核? 低频/降频?）时：

```bash
python3 -m mobench trace --config configs/local/<档>.yml --threads 4 --gen 512 --dur 25 --json
# 产物: results/<engine>/trace_<ts>/{trace.html 频率曲线, freq.csv, trace.json, atrace.txt}
```

- 负载窗内并行 atrace(`freq`) → 每核 mean/min/max + `low_freq_share`。**采样窗含 idle 前后段**，low share 是参考非自动降频判定（活跃窗过滤留待后续）。
- 真机示例：MNN t4 采样几乎全落小核 `cpu0-3`（mean ~1.1GHz，max 1804），big/prime 核仅数十事件 → decode 受小核频率与调度限制，可据此排查绑核/亲和。

## 算子级（引擎内建优先）

- **llama.cpp**：`llama-bench`/被测二进制内建分阶段计时；开 debug/perf 计时（GGML_PERF）可看逐 op。
- **MNN**：用被测工程自带 benchmark/profile 入口（如带 profiling 参数的 bench 工具），或设置引擎的 profiling 开关输出逐算子耗时。

产出记录/metric 口径尽量落到 `mobench` records（`source_log` 指向实际日志），可被 stats/report 聚合。

## CPU 采样与火焰图

```text
1. 对被测进程采样调用栈（perf record -g / simpleperf record -g）
2. 反汇编符号化（host: addr2line; adb: 对应 NDK 工具链；用 Debug/带符号构建）
3. 折叠栈 → FlameGraph 生成火焰图
```

## 排查要点

- **火焰图无符号**：换带符号的 Debug 构建，或提供被测工程符号文件。
- 先确认瓶颈是计算、访存还是调度/降频（配合 `mobile-bench-thermal` 温度与 `env` 频率信息），再决定优化方向。
- 所有采样命令的原始输出都要保存，遵循红线 R3/R7 可溯源。
