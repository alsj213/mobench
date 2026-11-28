"""Unit tests for the MNN adapter (spec: benchmark-core 内置 MNN 适配器)."""
import pytest

from mobench.adapters.base import AdapterError, get_adapter
from mobench.adapters.mnn import MnnAdapter


def make_cfg(tmp_path):
    return {"engine": "mnn", "build": {"bin_dir": str(tmp_path), "bench_bin": "benchmark.out"}}


def test_registry_returns_mnn(tmp_path):
    # Arrange
    (tmp_path / "benchmark.out").write_text("#!/bin/sh\n", encoding="utf-8")
    cfg = make_cfg(tmp_path)

    # Act
    adapter = get_adapter(cfg)

    # Assert
    assert isinstance(adapter, MnnAdapter)
    assert adapter.engine == "mnn"


def test_parse_text_lines(tmp_path):
    # Arrange
    cfg = make_cfg(tmp_path)
    adapter = MnnAdapter(cfg)
    log = tmp_path / "raw.log"
    log.write_text("model: mobilenetv2.mnn\nlatency_ms: 18.52\nfps: 54.0\n",
                   encoding="utf-8")
    model = {"id": "mobilenetv2", "path": "/m.mnn", "precision": "fp16"}

    # Act
    records = adapter.parse(str(log), model, threads=4, round_no=2)

    # Assert
    by = {r["metric"]: r for r in records}
    assert by["latency_ms"]["value"] == pytest.approx(18.52)
    assert by["fps"]["value"] == pytest.approx(54.0)
    assert by["latency_ms"]["round"] == 2
    assert by["latency_ms"]["precision"] == "fp16"


def test_parse_json_and_llm_tok(tmp_path):
    # Arrange
    cfg = make_cfg(tmp_path)
    adapter = MnnAdapter(cfg)
    log = tmp_path / "raw.log"
    log.write_text('{"latency_ms": 1.5, "gen_tok_s": 12.3}', encoding="utf-8")
    model = {"id": "qwen", "path": "/q.mnn", "precision": "q4_0"}

    # Act
    records = adapter.parse(str(log), model, threads=1, round_no=1)

    # Assert
    by = {r["metric"]: r for r in records}
    assert by["latency_ms"]["value"] == pytest.approx(1.5)
    assert by["gen_tok_s"]["value"] == pytest.approx(12.3)
    assert by["gen_tok_s"]["unit"] == "tok/s"


def test_parse_unrecognized_raises(tmp_path):
    # Arrange
    cfg = make_cfg(tmp_path)
    adapter = MnnAdapter(cfg)
    log = tmp_path / "raw.log"
    log.write_text("nothing useful here\n", encoding="utf-8")
    model = {"id": "m", "path": "/m.mnn"}

    # Act / Assert
    with pytest.raises(AdapterError, match="cannot parse"):
        adapter.parse(str(log), model, threads=4, round_no=1)


def test_missing_binary_raises(tmp_path):
    # Arrange
    cfg = make_cfg(tmp_path)

    # Act / Assert
    with pytest.raises(AdapterError, match="MNN"):
        MnnAdapter(cfg).resolve_binary()


# ---- llm_bench (official MNN LLM benchmark) ----------------------------
def llm_cfg(tmp_path):
    (tmp_path / "llm_bench").write_text("#!/bin/sh\n", encoding="utf-8")
    return {"engine": "mnn",
            "build": {"bin_dir": str(tmp_path), "bench_bin": "llm_bench"},
            "workload": {"prompt_tokens": 16, "gen_tokens": 64}}


def test_llm_bench_build_cmds(tmp_path):
    # Arrange
    adapter = MnnAdapter(llm_cfg(tmp_path))
    model = {"id": "qwen2", "path": "/m/qwen/config.json", "precision": "q4_0"}

    # Act
    cmd = adapter.build_cmds(model, threads=8, rounds=1)[0]

    # Assert: llm_bench flag style, prompt/gen from workload
    assert cmd[0] == str(tmp_path / "llm_bench")
    assert "-m" in cmd and "/m/qwen/config.json" in cmd
    assert "-a" in cmd and "cpu" in cmd
    assert "-p" in cmd and "16" in cmd
    assert "-n" in cmd and "64" in cmd


def test_llm_bench_rep_omitted_by_default(tmp_path):
    # Arrange: 未设 mnn.n_repeat 时不传 -rep，行为与既有档位完全一致
    adapter = MnnAdapter(llm_cfg(tmp_path))
    model = {"id": "qwen2", "path": "/m/qwen/config.json", "precision": "q4_0"}

    # Act
    cmd = adapter.build_cmds(model, threads=4, rounds=1)[0]

    # Assert
    assert "-rep" not in cmd


def test_llm_bench_rep_from_config(tmp_path):
    # Arrange: T4 档要一轮一测——llm_bench 默认 -rep 5 会在单轮内连跑 5 次
    # 且中途不冷却，既与"每轮独立冷却"的协议冲突，单轮时长也变成 llama 侧的数倍
    cfg = llm_cfg(tmp_path)
    cfg["mnn"] = {"backend": "cpu", "n_repeat": 1}
    adapter = MnnAdapter(cfg)
    model = {"id": "qwen2", "path": "/m/qwen/config.json", "precision": "q4_0"}

    # Act
    cmd = adapter.build_cmds(model, threads=4, rounds=1)[0]

    # Assert
    assert cmd[cmd.index("-rep") + 1] == "1"


def test_llm_bench_version_skips_probe(tmp_path, monkeypatch):
    # Arrange
    adapter = MnnAdapter(llm_cfg(tmp_path))
    import subprocess as _sp

    calls = {"n": 0}

    def fake_run(argv, **kw):
        calls["n"] += 1
        return _sp.CompletedProcess(argv, 0, stdout="")

    monkeypatch.setattr(_sp, "run", fake_run)
    # No probe binary run for llm_bench -> falls to git (fails here) -> unknown
    assert adapter.version() == "unknown"
    assert calls["n"] >= 1  # only the git fallback, never a bare llm_bench run


def test_parse_cv_benchmark_summary_row(tmp_path):
    # Arrange: real MNN benchmark.out summary row
    cfg = {"engine": "mnn"}
    adapter = MnnAdapter(cfg)
    log = tmp_path / "cv.log"
    log.write_text(
        "MNN benchmark\n"
        "Forward type: CPU thread=8 precision=2\n"
        "[ - ] squeezenetv1.1.mnn          max =    4.681 ms  min =    4.466 ms  avg =    4.574 ms\n",
        encoding="utf-8",
    )
    model = {"id": "squeezenet", "path": "/m/squeezenet.mnn"}

    # Act
    records = adapter.parse(str(log), model, threads=8, round_no=3)

    # Assert: avg ms -> latency_ms + derived fps
    by = {r["metric"]: r for r in records}
    assert by["latency_ms"]["value"] == pytest.approx(4.574)
    assert by["fps"]["value"] == pytest.approx(1000.0 / 4.574)
    assert by["latency_ms"]["round"] == 3


def test_parse_llm_bench_stdout_rows(tmp_path):
    # Arrange: real llm_bench table lines (pp8 / tg16 with t/s)
    cfg = {"engine": "mnn"}
    adapter = MnnAdapter(cfg)
    log = tmp_path / "raw.log"
    log.write_text(
        "The device supports: i8sdot:0\n"
        "| model | modelSize | backend | threads | precision | flashAttention | test | t/s |\n"
        "| ------ | ------: | ------ | ----: | ------ | ------ | ------ | ------: |\n"
        "| qwen2-0.5b | 265 MiB | CPU | 8 | Low | off | pp8 | 309.79 ± 0.00 |\n"
        "| qwen2-0.5b | 265 MiB | CPU | 8 | Low | off | tg16 | 52.21 ± 0.00 |\n",
        encoding="utf-8",
    )
    model = {"id": "qwen2", "path": "/m/qwen/config.json", "precision": "q4_0"}

    # Act
    records = adapter.parse(str(log), model, threads=8, round_no=1)

    # Assert: pp -> prompt_eval_ms_tok, tg -> gen_tok_s + gen_ms_tok
    by = {r["metric"]: r for r in records}
    assert by["gen_tok_s"]["value"] == pytest.approx(52.21)
    assert by["gen_ms_tok"]["value"] == pytest.approx(1000.0 / 52.21)
    assert by["prompt_eval_ms_tok"]["value"] == pytest.approx(1000.0 / 309.79)
    assert all(r["source_log"] == str(log) for r in records)


def test_cv_bench_out_build_cmds_uses_folder_cli(tmp_path):
    # Arrange
    (tmp_path / "benchmark.out").write_text("#!/bin/sh\n", encoding="utf-8")
    cfg = {"engine": "mnn",
           "build": {"bin_dir": str(tmp_path), "bench_bin": "benchmark.out"}}
    adapter = MnnAdapter(cfg)
    model = {"id": "squeezenet", "path": str(tmp_path / "models" / "sq.mnn")}

    # Act
    cmd = adapter.build_cmds(model, threads=8, rounds=1)[0]

    # Assert: <folder> 1(warm)1(loop) 0(cpu) threads 2(precision)
    assert cmd[0] == str(tmp_path / "benchmark.out")
    assert cmd[1] == str(tmp_path / "models")
    assert cmd[2:] == ["1", "1", "0", "8", "2"]


def test_llm_backend_from_config(tmp_path):
    # Arrange
    (tmp_path / "llm_bench").write_text("x", encoding="utf-8")
    cfg = llm_cfg(tmp_path)
    cfg["mnn"] = {"backend": "opencl"}
    adapter = MnnAdapter(cfg)
    model = {"id": "q", "path": "/m/q/config.json", "precision": "q4_0"}

    # build_cmds carries -a opencl
    cmd = adapter.build_cmds(model, threads=4, rounds=1)[0]
    assert "-a" in cmd and cmd[cmd.index("-a") + 1] == "opencl"

    # parsed records carry backend=opencl
    log = tmp_path / "raw.log"
    log.write_text("| q | 1 MiB | CPU | 4 | Low | off | tg8 | 12.0 ± 0.0 |\n",
                   encoding="utf-8")
    recs = adapter.parse(str(log), model, threads=4, round_no=1)
    assert recs[0]["backend"] == "opencl"


def test_build_cmds_forwards_workload_cli_args(tmp_path):
    cfg = llm_cfg(tmp_path)
    cfg["workload"]["cli_args"] = ["--mmap", "1"]
    adapter = MnnAdapter(cfg)
    model = {"id": "q", "path": "/m/q/config.json"}
    cmd = adapter.build_cmds(model, threads=4, rounds=1)[0]
    assert cmd[-2:] == ["--mmap", "1"]


def test_parse_llm_bench_preserves_multiple_pp_and_tg_rows(tmp_path):
    cfg = {"engine": "mnn", "workload": {"prompt_tokens": 128, "gen_tokens": 64}}
    adapter = MnnAdapter(cfg)
    log = tmp_path / "multi.log"
    log.write_text(
        "| q | 1 MiB | CPU | 4 | Low | off | pp128 | 300.0 ± 0.0 |\n"
        "| q | 1 MiB | CPU | 4 | Low | off | pp512 | 350.0 ± 0.0 |\n"
        "| q | 1 MiB | CPU | 4 | Low | off | tg64 | 50.0 ± 0.0 |\n"
        "| q | 1 MiB | CPU | 4 | Low | off | tg128 | 45.0 ± 0.0 |\n",
        encoding="utf-8",
    )
    model = {"id": "q", "path": "/m/q/config.json"}
    recs = adapter.parse(str(log), model, threads=4, round_no=1)
    pp = [r for r in recs if r["metric"] == "prompt_eval_ms_tok"]
    tg = [r for r in recs if r["metric"] == "gen_tok_s"]
    assert [r["params"]["prompt_tokens"] for r in pp] == [128, 512]
    assert [r["params"]["gen_tokens"] for r in tg] == [64, 128]
    assert [r["value"] for r in tg] == [50.0, 45.0]
