# 贡献指南（Contributing to mobench）

感谢你关注并为 **mobench** 做出贡献！无论是修复 Bug、完善文档、新增推理引擎适配器，还是改进分析与报告能力，都非常欢迎。

## 开发环境准备

要求 Python 3.9+：

```bash
git clone https://github.com/alsj213/mobench.git
cd mobench
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

运行测试与覆盖率检查：

```bash
python3 -m pytest tests/ -q --cov=mobench
```

## 核心设计原则

1. **引擎无关核心**：具体推理框架的命令构造与日志解析只能放在 `mobench/adapters/<engine>.py`，不向核心模块泄漏框架细节。
2. **数据可信红线（R1–R11）**：详见 [`rules/mobile-bench-integrity.md`](rules/mobile-bench-integrity.md)——
   - 不编造或插值缺失数值；
   - 不静默吞掉子进程或解析错误，失败必须以非零退出并暴露诊断信息；
   - 每条记录必须绑定真实原始日志（`source_log`）与环境/产物哈希，确保完全可溯源。
3. **保持仓库干净**：切勿将包含个人本机绝对路径、真实设备序列号（`device_id`）或本地跑分结果（`results/`、`configs/local/`）提交到 Git。

## 接入新推理引擎适配器

详见 [`docs/adapter-guide.md`](docs/adapter-guide.md)，标准四步流程：

1. 新增 `mobench/adapters/<engine>.py`，实现 `EngineAdapter` 并导出 `ADAPTER`。
2. 在 `mobench/config.py` 的 `BUILTIN_ENGINES` 中注册 engine key。
3. 在 `schema/mobench.schema.json` 的 `engine.enum` 中同步枚举值。
4. 新增 `tests/test_adapter_<engine>.py`（覆盖命令构造、真实输出解析与异常路径），并更新 `README.md` 支持矩阵。

## 提交 Pull Request 流程

1. Fork 本仓库并从 `main` 创建特性/修复分支（如 `feat/adapter-ncnn`、`fix/adb-push-retry`）。
2. 采用清晰的 Conventional Commits 风格提交信息（如 `feat(adapters): ...`、`fix(cli): ...`、`docs: ...`）。
3. 确保所有测试通过且核心覆盖率不低于 80%：
   ```bash
   python3 -m pytest tests/ -q --cov=mobench
   ```
4. 提交 PR 并按模板说明改动动机、验证方式与影响范围。
