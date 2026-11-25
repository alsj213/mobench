# CLAUDE.md — mobench 开发指南

## 项目结构

- `mobench/`：确定性 Python 编排核心（CLI、平台抽象、引擎适配器、分析与报告、MCP server）
- `tests/`：单元测试与端到端测试（pytest）
- `schema/mobench.schema.json`：`mobench.yml` 配置 Schema
- `configs/templates/` 与 `examples/`：开箱即用配置模板
- `skills/`、`agents/`、`rules/`：Claude Code 插件知识与数据真实性规则（R1–R11）

## 开发与测试

```bash
pip install -e ".[dev]"
python3 -m pytest tests/ -q --cov=mobench
```

## 关键约束

- 引擎差异仅允许收敛在 `mobench/adapters/` 中，核心模块不耦合具体推理框架。
- 严格遵守数据可信红线 R1–R11（见 `rules/mobile-bench-integrity.md`）：不编造数据、不静默吞错、每项指标保留真实日志溯源。
- 不要将包含本机绝对路径、设备序列号或原始运行产物（`results/`、`configs/local/`）提交到 Git。
