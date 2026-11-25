---
name: mobile-bench-model-prep
description: Use when preparing models for benchmark testing — self-provided path verification with MD5, downloading public weights with declared-hash check, or converting/quantizing via the engine's own tools. All three states are driven by the mobench prepare subcommand.
---

# 模型准备（mobench prepare）

通用化后不依赖固定模型树。模型供给分三种状态，统一由 `mobench prepare` 完成；本 skill 描述如何声明与验证。

## 声明方式（mobench.yml）

基准入口 config 位置由 `--config` 显式定位（`configs/` 模板/实例分层见 `configs/README.md`）。每个模型一项 `models[]`，字段见 `schema/mobench.schema.json`。关键差异字段：

| 状态 | 声明 | prepare 行为 |
|------|------|--------------|
| 自备 | 仅 `id` + `path`（指向本地文件） | 校验文件存在并计算 MD5，写入记录 |
| 下载 | `source: <URL>` + `sha256` | 下载到目标路径；哈希与声明不符即丢弃并报错 |
| 转换 | `source` 指向源模型 + `convert_with` | 借用被测引擎工具转换/量化（如 llama.cpp `convert_hf_to_gguf.py`、MNN `MNNConvert`），记录所用命令与结果哈希 |

`path` 可直接指向被测引擎工程已准备/构建出的模型，无需拷贝到本仓库。

## 驱动与验证

```bash
python3 -m mobench prepare                 # 准备全部模型
python3 -m mobench prepare <model-id> ...  # 只准备指定模型
python3 -m mobench prepare --json          # 结构化输出
```

验证点：
- 自备：返回 MD5，与之后 `mobench run`/`env` 快照中模型哈希一致（红线 R10）。
- 下载：`sha256` 不符 → 文件被丢弃、命令失败退出非零，绝不用未校验文件。
- 转换：输出含转换命令与产物哈希；缺失转换器报错并给出被测引擎侧工具提示。

模型就绪后由 `mobile-bench-run` skill 的 `mobench run` 消费。`run` 内部模型检查失败时会中止并提示先执行 `mobench prepare`。
