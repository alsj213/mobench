---
name: mobile-bench-integrity
description: 推理 benchmark 数据真实性强制规则（引擎无关：llama.cpp / MNN）。权威全文与 7 步协议见 agents/mobile-bench-agent.md，本规则为其自动加载摘要，禁止两处漂移。
---

# Mobile Bench 数据真实性规则

执行推理 benchmark（host 或 adb、任何已注册引擎）时必须遵守。**权威定义见 `agents/mobile-bench-agent.md`（R1–R11 + 7 步协议）；规则文件若与 agent 冲突，以 agent 为准并修正本文件。**

## 数据真实性

1. **不得编造任何数字** — 指标值必须来自 `mobench run` records（`source_log` 追溯 tee 日志）或平台命令输出
2. **不得使用旧数据** — 每次 request 重新运行，不得回忆/复用前次结果
3. **每行数据必须标注来源** — 表格每值注明 `[来源: source_log 或命令]`
4. **不得静默忽略失败** — 任何命令 exit code != 0（含 `mobench` 报错）须报告并决定停止或继续

## 执行流程

5. **不得跳过平台握手** — adb 设备不在线 / preflight 失败即报错退出
6. **必须保留原始日志** — `mobench run` 内部 tee 日志须存在且非空，不得仅靠 stdout
7. **必须同时展示原始输出/记录与处理后的结果** — 不能只给表格

## 证据链

8. **禁止跨会话数据复用** — 不粘贴旧输出、不手工构造 records/结果行、不以"之前跑过"为由跳过重跑；日志时间戳须匹配当前时间
9. **温度 > 45°C 时标注降频风险 / 不使用数据** — 如实报告温度
10. **二进制和模型文件的哈希/大小必须记录** — 由 `mobench env` 快照保存
11. **所有配置从 `mobench.yml` 读取** — 不硬编码路径/设备 ID；schema 见 `schema/mobench.schema.json`
