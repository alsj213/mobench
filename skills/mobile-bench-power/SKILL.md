---
name: mobile-bench-power
description: Use when measuring energy efficiency of inference — energy per inference (J) and perf/Watt from host RAPL or adb power sources, with explicit Power data unavailable labeling when no source is usable. Maps to the mobench power subcommand and the power spec.
---

# 功耗与能效测量（mobench power）

产出业界标准能效指标：每推理能耗（J）与 perf/Watt（对齐 MLPerf Power）。数据来源与公式必须标注；不可用时打标 `Power data unavailable`，禁止伪造或估算充数。

## 采集功耗数据

`mobench` 只消费**平台工具的输出文件**，不内置采集：

- host：RAPL（`/sys/class/powercap` 或 `perf stat -e power/energy-*`）
- adb：`dumpsys batterystats`、Perfetto power rails；无 root 时没有可靠功耗源 → 打标 unavailable（红线：不伪造数字）

优先 power rails；依次降级 batterystats → 电量差估算（估算须标注误差等级）。

## 计算

```bash
python3 -m mobench power --source <power-data.txt> \
  --inference-count <N> --throughput <FPS|tok/s> --json
```

- 每推理能耗 = 窗口内能量(J) / 推理次数
- perf/Watt = 吞吐 / 平均功率
- 解析只认真实能耗信号（如 `Computed drain`、RAPL energy），`Capacity`、`drain 0` 等不是功耗信号，不误读（含换行包裹数字的兼容）

输出样例：`Energy per inference: xx J` + 来源 note；不可用时 `Power data unavailable` + 降级原因。

## 验证

- 有可用信号 → status `ok`，输出 energy_joules / energy_per_inference_j / perf_per_watt
- 无信号 / drain=0 → status `unavailable`，label 明确，不填假数字

能效结论需与 `mobile-bench-run` 的吞吐量配对解读；长时间能效趋势纳入 `mobile-bench-thermal` 的稳定性评估。
