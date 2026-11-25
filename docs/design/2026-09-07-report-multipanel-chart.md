# Design — report 多面板基准图（multipanel chart）

需求来源：`docs/requirement-report-multipanel-chart.md`（2026-09-07）。参考 MNN 3.6.1 release 基准图：多面板（平台×模型×Prefill/Decode），x=Prompt Length，y=tok/s，面板内 threads/后端成曲线。

## 目标

让 `records → mobench report` 能输出一张**自包含 HTML**（SVG 内联，离线可读、中文无乱码、深浅主题自适应），对 LLM 参数矩阵给出趋势与多序列对比的可视化，同时保留数值可溯源（红线 R1/R3/R7）。

## 核心决策

### D1 输出通道
- `report --format` 枚举扩展为 `{md, html, both, chart}`。
  - `chart` 只生成 `benchmark_chart.html`；
  - 沿用现有语义，不破坏既有调用（md/html/both 不变）。
- chart.html 内**含图 + 明细溯源表**（每个数据点一行：n / mean / CV% / source_log），避免“图上好看但没法对账”。

### D2 绘图技术：纯 Python 标准库自绘 SVG（推荐，采纳）
- 不引入 matplotlib/plotly 等重依赖；输出 **SVG 内联进单 HTML**，浏览器直接打开即渲染，无网络请求。
- 用 CSS token（`prefers-color-scheme`）做深浅主题，中文字体栈与现有 report.html 一致。
- 不做 PNG 编码；如需 PNG 用浏览器/系统工具另存（标注 in-doc，不入代码）。
- 备选（若后续要“PNG 文件/高密度导出”）再评估 matplotlib——本阶段不做。

### D3 面板与序列分组（复用 records，无 schema 变更）
输入沿用 `report inputs`（递归收集，P1 已支持）。分组：
1. **面板键** = `(platform_label, workload_id, stage)`
   - `stage` 由 metric 推导：`gen_tok_s`→`Decode`；`prompt_eval_ms_tok`→`Prefill`（图上换算 tok/s = 1000/mean(ms/tok)）。
   - `platform_label`：优先 record 的 `device.platform`，其次顶层 `platform`（**需小修**：`run` 现在把 record 顶层 `platform` 硬编码为 `'host'` 由 adapter 写入；应改为由 run 按 cfg platform 类型覆盖，adb 才正确。此为低风险小改动）。
   - 其它 metric（CV latency/fps 等）本轮不入 LLM 图；若输入不含任何 LLM 阶段 metric → 输出空态说明而非空面板。
2. **面板内序列**：`threads`（多线程取值各一条曲线）。默认 x = `params.prompt_tokens`（Prompt Length）。
3. 同一 `(panel, x, threads)` 多次运行取均值成点，标注 n 与 CV。
4. 旧记录无 `params`：x 退化为单点（标签“默认”，点=n 均值），仍可画 threads 折线，不报错。

> 需求“后端成曲线”（CPU/Metal/OpenCL）当前 records **无 backend 字段**，而需求 Out-of-scope 也声明“能画的只取决于 records 里有哪个维度”。本设计曲线维度 = threads；backend 维度列为**可选扩展**（若以后引入可选 `backend` 字段，插到曲线维之前即可，面板布局不依赖它）。

### D4 换算、刻度与标注（可读性硬要求）
- y 轴：Prefill 面板统一显示 tok/s（由 ms/tok 换算），Decode 面板直接用 tok/s；**每面板独立刻度**。
- 点信息（悬停 title / 明细表）：n、mean、CV%、首轮 `source_log`（多轮标 `· 共 n 轮`）。
- 图例 = threads；x 轴 = Prompt Length（数字刻度）。

### D5 空态与单点
- 面板无数据：不画，给出“该面板无数据”说明（可用 CSS 提示块）。
- x 仅 1 个值：画点（或极短柱），不假装成趋势；仍给出提示“单点不足以判断趋势”。
- 全部输入无 LLM 阶段 metric：页面说明“记录不含可作图指标（需要 prompt_eval_ms_tok/gen_tok_s）”。

### D6 与既有表格并存
- chart.html 独立文件，不影响 md/html。也可在 html 报告正文里通过相对链接提示 chart（可选）。

## 数据模型 / 触达面

| 要素 | 来源 |
|---|---|
| x | `params.prompt_tokens` |
| Decode y | `metric=gen_tok_s` 均值 |
| Prefill y | `metric=prompt_eval_ms_tok` 均值 → 1000/v |
| 曲线 | `threads` |
| 面板 | `(device.platform∥platform, workload_id, stage)` |
| n / CV / source | 组内统计（沿用现有 summarize 逻辑） |

## 验收标准（映射 AC）

1. 对含两个不同 `params.prompt_tokens`（如 512/1024）与多个 threads 的 run 集，`report --format chart` 生成 chart.html：面板 = 平台×模型×Prefill/Decode，x=Prompt Length，每面板 threads 曲线、图例与刻度正确。
2. 每个点可溯源（明细表列出 source_log / 首轮锚点 + 共 n 轮），图上悬停给 n/CV/来源。
3. 单点/无参数/无 LLM metric 三种退化：不报错、有明确说明。
4. `pytest` 新增 chart 用例通过，覆盖率不回落；TEST_REPORT 记录。
5. 深/浅主题、窄屏可读、中文无乱码（自包含 HTML 验证）。

## 实现落点（后续 tasks 参照）

- `mobench/report.py`：`--format chart` 分支 + `_render_chart(rows, records)`；内部组织面板数据与 SVG 生成（可拆 `mobench/report/` 子模块，若文件超限）。
- `run.py`：小修 record 顶层 `platform` 按 cfg 平台覆盖（配合 D3 platform_label）。
- 测试 `tests/test_report_chart.py`：合成 records（两 prompt 长度 × 两 threads × prefill/decode）断言 HTML 含面板、图例、换算值、明细表、空态分支。
- 文档：README（report 支持 chart）、TEST_REPORT、AC。

## 风险与备注

- SVG 自绘工作量可控但样式打磨需时间；首版先保证正确性与可读性，美学后置。
- `platform` 小修属于独立低风险 bugfix，可先行。
- backend 维度、PNG 导出为明确的后续扩展，不在本设计范围。
