# 配置示例（examples）

把对应 `mobench.yml` 复制到**被测引擎工程根**（或任意工作目录），按注释改 `path`/`bin_dir`/`device_id` 后运行：

```bash
python3 -m mobench prepare && python3 -m mobench run
python3 -m mobench report results/ --out results/
python3 -m mobench stats  results/
```

| 示例 | 引擎/工具 | 说明 |
|------|-----------|------|
| `llamacpp/mobench.yml` | llama.cpp `llama-bench` | host 与 adb 两种平台形态（单文件 `.gguf` 模型） |
| `mnn/llm_bench/mobench.yml` | MNN `llm_bench` | LLM 目录模型（`config.json` + weights，`models[].dir`） |
| `mnn/benchmark.out/mobench.yml` | MNN `benchmark.out` | CV/NLP 目录式测速（`models[].dir` 放单个 `.mnn`） |

- `platform.type: host` 直跑桌面 Linux 产物；`adb` 需要在线设备，`bin_dir` 指向**本地 Android 产物目录**（含动态库），mobench 会整目录推送。
- 正式测量 `rounds` ≥ 5；示例为便于快速试跑常设 1–2。
- 引擎支持与契约见 README「Engine support」；新框架接入见 `docs/adapter-guide.md`。
