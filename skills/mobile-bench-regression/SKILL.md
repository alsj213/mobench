---
name: mobile-bench-regression
description: Use when detecting performance regressions across code, model, or framework changes — baseline snapshot, cross-commit comparison with statistical significance testing, and threshold alerts. Maps to the mobench regression subcommand and the regression spec.
---

# 回归检测（mobench regression）

检测被测引擎代码/模型/框架版本变更导致的性能回归。基线快照 + 跨 commit 对比 + 显著性检验，使"退化"可自动发现而非仅靠肉眼。

## 保存基线

在一组合格测量（≥5 轮/组、来源可溯）后保存：

```bash
python3 -m mobench regression save --results results/records.jsonl --out results/baseline.json
```

baseline.json 含各组关键指标与 env 字段（引擎/模型哈希、平台、温度），由当前 records 派生，不手工填写。

## 跨 commit 对比

对引擎代码改动用例：记录改动前后 commit，对比同一配置：

```bash
python3 -m mobench regression compare --results results/records.jsonl --baseline results/baseline.json
```

- 差异方向 + Δ%（相对基线）按 metric 语义（吞吐 tok/s/FPS 越高越好；延迟越低越好）判断恶化。
- 同一配置当前与基线各 ≥5 次重复时，用显著性检验（配对 t-test / Mann-Whitney U，scipy 可选，缺失则回退），输出 p 值；不得仅凭单点数值判定。
- 环境差异（设备/OS/温度变化）与真实性能变化区分开——对比输出两侧 commit 与环境是否一致。

## 阈值告警

显著恶化超过阈值（默认延迟类恶化 >5%）：

```bash
REGRESSION DETECTED: <engine> <workload_id> <metric> Δ=+xx%
```

给出当前 commit/版本及可能原因（代码变更、版本升级、环境漂移）。改善或未显著则不告警。

## 数据真实性

对比两侧都来自 records 且 `source_log` 可溯源到 tee 日志；禁止粘贴旧输出、手工构造记录（红线 R11）。基线文件记录创建时间与环境快照，跨会话引用时先核对 env 一致性。
