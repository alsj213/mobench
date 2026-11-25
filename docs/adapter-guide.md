# 接入新推理框架（Adapter 接入指南）

把一个新框架（ncnn / ONNX Runtime / TVM / llamafile / TensorRT …）接进 mobench，
只需让 mobench 会**调用它自己的测速工具**并**读懂它的输出**。这份指南说明要改
哪些文件、每一步做什么、以及真实接入中会踩的坑（均来自 llama.cpp / MNN 实测）。

## 结论先行

| 要动的东西 | 是否改动 | 说明 |
|-----------|---------|------|
| mobench 核心（run / report / analysis / records / CLI / platform） | ❌ | 完全不碰，引擎差异已隔离 |
| `mobench/adapters/<engine>.py` | ✅ **新增**（约 60–120 行） | 引擎唯一耦合点：怎么调用 + 怎么解析 |
| `mobench/config.py` | ✅ 加 1 行 | `BUILTIN_ENGINES` 注册 engine key |
| `schema/mobench.schema.json` | ✅ 加 1 个枚举值 | `engine` enum |
| `tests/test_adapter_<engine>.py` | ✅ **新增** | registry/解析/命令构造单测 |
| `README.md` 契约表 + `docs/adapter-guide.md` | ✅（建议） | 记录该引擎的命令与输出形态 |

> engine key 用 kebab-case（如 `ncnn`），与模块名 `mobench/adapters/<engine>.py` 一致。

---

## 你需要先确认的 4 件事

从被测框架拿真实信息（**不要猜**，本会话多次教训来自"按假想契约写 adapter"）：

1. **测速二进制**：哪个可执行文件？（如 `llama-bench` / `llm_bench` / `benchmark.out` / `onnxruntime_perf_test`）
2. **命令行形态**：单文件参数还是目录/flag 式？（`llama-bench -m model.gguf -o json`；MNN `benchmark.out <模型目录> loop warm fw threads prec`；MNN `llm_bench -m <config.json> -a cpu …`）
3. **输出形态**：JSON（对象还是数组）？表格行？`key: value`？summary 单行？（llama.cpp 新版是**顶层数组** `avg_ts`；MNN CV 是 `max/min/avg` 单行）
4. **指标语义**：吞吐类（`gen_tok_s`/`fps`，越高越好）与延迟类（`latency_ms`/`*_ms_tok`，越低越好）——统计与回归按 metric 名判断方向。

---

## 四步接入

### 第 1 步：写 adapter —— 5 个方法

新建 `mobench/adapters/<engine>.py`，继承 `mobench/adapters/base.py` 的 `EngineAdapter`：

```python
from mobench.adapters.base import AdapterError, EngineAdapter, _find_binary

class MyAdapter(EngineAdapter):
    engine = "ncnn"                       # 与模块名一致
    metric_vocab = ("latency_ms", "fps", "gen_tok_s")   # 你产出的 metric

    def _bench_name(self):                # 被测二进制名（build.bench_bin 可覆盖）
        build = self.cfg.get("build") or {}
        return build.get("bench_bin") or "my_bench"

    def resolve_binary(self) -> str:      # 探测可执行文件（bin_dir 再 PATH）
        path = _find_binary(self.cfg, self._bench_name())
        if not os.path.isfile(path):
            raise AdapterError("... binary not found ...")
        return path

    def version(self) -> str:             # 版本/commit（尽力而为；工具无参跑会崩就跳过裸跑）
        # 例：跑 `git -C <build.source_dir> log --oneline -1` 兜底
        return "unknown"

    def build_cmds(self, model, threads, rounds):
        """返回引擎真实 CLI（每外层 round 一次）。model.path / model.get('dir') 可用。"""
        return [[self.resolve_binary(), model["path"], "-t", str(threads)]]

    def parse(self, log_path, model, threads, round_no):
        """读 tee 原始日志 -> 返回规范化 records（list[dict]）。"""
        text = open(log_path, encoding="utf-8", errors="replace").read()
        value = _pull_out_your_number(text)          # 你的解析逻辑
        return [self._rec(model, threads, round_no, "fps", value, "FPS", log_path)]

    def _rec(self, model, threads, round_no, metric, value, unit, log_path):
        return {"schema": "mobench/record@1", "engine": self.engine,
                "workload_id": model["id"], "metric": metric,
                "value": round(float(value), 6), "unit": unit,
                "threads": threads, "precision": model.get("precision"),
                "round": round_no, "platform": "host",
                "source_log": log_path}

ADAPTER = MyAdapter        # 注册表懒加载要读这个类
```

要点：
- **`source_log` 必须指真实日志**（红线 R1/R11：不编造）。解析失败就抛 `AdapterError`。
- 输出若可能有多种形态（JSON 对象/数组、表行、`key: value`、summary），在 `parse` 里依次尝试、命中即返回——参照 `llamacpp.py`（对象/数组双契约）与 `mnn.py`（CV summary + 表行 + JSON/key:value）。
- `version()` 若对无参裸跑会 assert/core（MNN `llm_bench` 就会）→ 跳过探针直接走 git 兜底。

### 第 2 步：注册 engine key

```python
# mobench/config.py
BUILTIN_ENGINES = ("llamacpp", "mnn", "ncnn")
```
否则 `mobench.yml` 填 `engine: ncnn` 会被拒绝并列出已注册适配器。

### 第 3 步：schema enum

```jsonc
// schema/mobench.schema.json
"engine": { "enum": ["llamacpp", "mnn", "ncnn"] }
```

### 第 4 步：测试 + 文档

`tests/test_adapter_<engine>.py` 至少覆盖：registry 返回你的 adapter、`parse` 对真实输出样例
（含不可解析路径抛错）、`build_cmds` 拼出期望命令、`version` 兜底。跑 `python3 -m pytest tests/`。

文档：在 README 契约表加一行 + 本文件补充该引擎的命令/输出形态。

---

## 模型怎么声明（mobench.yml）

| 引擎模型形态 | models 配置 |
|--------------|-------------|
| 单文件（`.gguf` / `.onnx` / `.mnn`） | `path: <文件>` |
| 目录模型（MNN LLM：config + weights + tokenizer） | `path: <目录>/config.json` + `dir: <目录>` |
| 目录式批量测速工具（MNN `benchmark.out` 吃模型目录） | `path: <目录>/model.mnn` + `dir: <目录>`（目录内只放被测这一个模型） |

- host 平台：`build.bin_dir` 指向被测工程产物目录即可。
- **adb 平台**：`bin_dir` 指向**本地 Android 产物目录**（含二进制与动态库）。若被测工具是
  动态链接（RPATH `$ORIGIN`），mobench 会整目录 push 并以 `LD_LIBRARY_PATH=.` 运行；
  目录模型/目录式工具经 `models[].dir` 整目录同步。

---

## 真实接入踩过的坑（务必避开）

1. **版本契约会漂移**：llama.cpp 把 `--json` 改成了 `-o json`，输出从 `{results:[…]}`
   变成**顶层数组**。接入时先拿真实输出确认，别照旧文档写解析。
2. **二进制与动态库必须自洽**：Android 上同一目录里混着多版 `libMNN.so`/`libllama.so`
   会让工具报 `cannot locate symbol`。用**同一次构建**产出的 binary+libs（如某次
   `build_android_llm` 目录），别从设备上多个散目录拼。
3. **adb push 会丢可执行位** → mobench 已自动 `chmod 755`；但自测时记得手动加。
4. **`/tmp` 在 Android 上不存在** → 用 `/data/local/tmp`。
5. **无参跑会崩的工具**（`llm_bench`）不能在 `version()` 里裸执行探针 → 跳过走 git。
6. **WSL→Windows adb server 偶发抖动**（`accept4 110`）→ mobench push 已带重试；手测时
   `adb kill-server && adb start-server` 后重试，且 push 之间留间隔。
7. **指标方向要一致**：延迟 metric 用 `ms`/`ms/token`（越低越好），吞吐用 `tok/s`/`fps`
   （越高越好），否则 `mobench regression` 会把优化当回归。

---

## 零代码方向（可选演进）

若希望多数新框架**只写 mobench.yml、不写 Python**，可加「声明式通用适配器」：用配置描述
二进制、参数模板、输出字段/正则到 metric 的映射。当前引擎 key 仍需注册，暂未实现。

## 关联

- `skills/mobile-bench-integrate/SKILL.md` — 执行态接入清单
- `mobench/adapters/llamacpp.py` / `mnn.py` — 可作模板对照
- `schema/mobench.schema.json`、`mobench/config.py` — 注册点
