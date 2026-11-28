"""Unit tests for the precision-validation contract & flow
(spec: benchmark-correctness; adapters: benchmark-core 框架适配器与扩展注册)."""
import json

import pytest

from mobench.adapters.base import AdapterError, EngineAdapter


class _NoValidateAdapter(EngineAdapter):
    """Minimal adapter without the optional validation contract."""

    engine = "novalidate"


def test_contract_default_raises_unavailable(tmp_path):
    # Arrange
    adapter = _NoValidateAdapter({})
    model = {"id": "m", "path": "m.gguf"}

    # Act / Assert: 缺省契约必须报明确不可用，绝不静默降级
    with pytest.raises(AdapterError, match="does not implement"):
        adapter.build_validate_cmds(model, {"prompts": ["p.txt"]}, {}, 4)


def test_contract_parse_default_raises_unavailable():
    # Arrange
    adapter = _NoValidateAdapter({})

    # Act / Assert
    with pytest.raises(AdapterError, match="does not implement"):
        adapter.parse_validate({"instrument": "ppl", "cmd": []}, "raw.log")


# -- llama.cpp 仪器（fixtures 取自源码核对的真实输出格式）-------------------

from pathlib import Path  # noqa: E402

from mobench.adapters.llamacpp import LlamacppAdapter  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "validate"


def make_llamacpp_cfg(tmp_path, validation=None):
    for name in ("llama-bench", "llama-completion", "llama-perplexity"):
        (tmp_path / name).write_text("#!/bin/sh\n", encoding="utf-8")
    return {
        "engine": "llamacpp",
        "build": {"bin_dir": str(tmp_path), "bench_bin": "llama-bench"},
        "workload": {"threads": [4]},
        "validation": validation or {},
    }


MODEL = {"id": "qwen3", "path": "/models/qwen3.gguf", "precision": "q4_k_m"}
KV_WORKLOAD = {"cache_type_k": "f16", "cache_type_v": "f16", "flash_attn": "off"}


def test_build_validate_greedy_command_shape(tmp_path):
    # Arrange
    adapter = LlamacppAdapter(make_llamacpp_cfg(
        tmp_path, {"prompts": ["/p/long-01.txt"], "gen_tokens": 64}))

    # Act
    jobs = adapter.build_validate_cmds(MODEL, {"prompts": ["/p/long-01.txt"],
                                               "gen_tokens": 64}, KV_WORKLOAD, 4)

    # Assert
    assert len(jobs) == 1
    job = jobs[0]
    assert job["instrument"] == "greedy"
    assert job["prompt"] == "/p/long-01.txt"
    cmd = job["cmd"]
    assert cmd[0].endswith("llama-completion")
    assert "-no-cnv" in cmd
    assert "--temp" in cmd and cmd[cmd.index("--temp") + 1] == "0"
    assert [cmd[cmd.index("-f") + 1]] == ["/p/long-01.txt"]
    assert [cmd[cmd.index("-n") + 1]] == ["64"]
    assert [cmd[cmd.index("-t") + 1]] == ["4"]
    # KV 参数透传（identity 组成部分）
    assert [cmd[cmd.index("-fa") + 1]] == ["off"]
    assert [cmd[cmd.index("-ctk") + 1]] == ["f16"]
    assert [cmd[cmd.index("-ctv") + 1]] == ["f16"]


def test_build_validate_ppl_command_shape(tmp_path):
    # Arrange
    val = {"ppl": {"dataset": "/d/wiki.raw", "ctx": 1024, "chunks": 8}}
    adapter = LlamacppAdapter(make_llamacpp_cfg(tmp_path, val))

    # Act
    jobs = adapter.build_validate_cmds(MODEL, val, {}, 4)

    # Assert
    assert len(jobs) == 1
    assert jobs[0]["instrument"] == "ppl"
    cmd = jobs[0]["cmd"]
    assert cmd[0].endswith("llama-perplexity")
    assert [cmd[cmd.index("-f") + 1]] == ["/d/wiki.raw"]
    assert [cmd[cmd.index("-c") + 1]] == ["1024"]
    assert [cmd[cmd.index("--chunks") + 1]] == ["8"]
    assert [cmd[cmd.index("-t") + 1]] == ["4"]


def test_build_validate_both_instruments_ppl_first(tmp_path):
    # Arrange
    val = {"prompts": ["/p/a.txt"], "gen_tokens": 32,
           "ppl": {"dataset": "/d/wiki.raw"}}
    adapter = LlamacppAdapter(make_llamacpp_cfg(tmp_path, val))

    # Act
    jobs = adapter.build_validate_cmds(MODEL, val, {}, 4)

    # Assert
    assert [j["instrument"] for j in jobs] == ["ppl", "greedy"]


def test_build_validate_missing_binary_errors(tmp_path):
    # Arrange: 只有 llama-bench，无 completion/perplexity
    (tmp_path / "llama-bench").write_text("#!/bin/sh\n", encoding="utf-8")
    cfg = {
        "engine": "llamacpp",
        "build": {"bin_dir": str(tmp_path), "bench_bin": "llama-bench"},
        "workload": {},
    }
    adapter = LlamacppAdapter(cfg)

    # Act / Assert
    with pytest.raises(AdapterError, match="llama-completion"):
        adapter.build_validate_cmds(MODEL, {"prompts": ["p.txt"]}, {}, 4)


def test_parse_greedy_strips_markers_and_perf(tmp_path):
    # Arrange
    adapter = LlamacppAdapter(make_llamacpp_cfg(tmp_path))

    # Act
    result = adapter.parse_validate(
        {"instrument": "greedy", "prompt": "/p/x.txt"},
        str(FIXTURES / "completion_ok.log"))

    # Assert: 流含 prompt 回显（引擎无纯生成通道），性能块与 EOG 标记被剥离
    assert result["text"] == ("The capital of France is Paris, its largest city. "
                              "Paris is known for")
    assert result["n_prompt_tokens"] == 512


def test_parse_greedy_no_eog_keeps_text(tmp_path):
    # Arrange
    adapter = LlamacppAdapter(make_llamacpp_cfg(tmp_path))

    # Act
    result = adapter.parse_validate(
        {"instrument": "greedy", "prompt": "/p/x.txt"},
        str(FIXTURES / "completion_no_eog.log"))

    # Assert
    assert result["text"] == "Line one.\nLine two."
    assert result["n_prompt_tokens"] == 512


def test_parse_greedy_unparseable_raises(tmp_path):
    # Arrange: 截断日志无 generate 头 → 不可用，不猜值
    adapter = LlamacppAdapter(make_llamacpp_cfg(tmp_path))

    # Act / Assert
    with pytest.raises(AdapterError, match="generate"):
        adapter.parse_validate({"instrument": "greedy"},
                               str(FIXTURES / "completion_unparseable.log"))


def test_parse_greedy_truncated_no_perf_raises(tmp_path):
    # Arrange: 有 generate 头但无性能块（进程被杀）→ 完整性无法证明
    log = tmp_path / "truncated.log"
    log.write_text("generate: n_ctx = 8192, n_batch = 2048, n_predict = 64, n_keep = 0\n"
                   "Partial output", encoding="utf-8")
    adapter = LlamacppAdapter(make_llamacpp_cfg(tmp_path))

    # Act / Assert
    with pytest.raises(AdapterError, match="perf"):
        adapter.parse_validate({"instrument": "greedy"}, str(log))


def test_parse_ppl_extracts_value_and_chunks(tmp_path):
    # Arrange
    adapter = LlamacppAdapter(make_llamacpp_cfg(tmp_path))

    # Act
    result = adapter.parse_validate({"instrument": "ppl"},
                                    str(FIXTURES / "perplexity_ok.log"))

    # Assert
    assert result["value"] == 1.0086
    assert result["chunks"] == 8


def test_parse_ppl_unparseable_raises(tmp_path):
    # Arrange
    adapter = LlamacppAdapter(make_llamacpp_cfg(tmp_path))

    # Act / Assert
    with pytest.raises(AdapterError, match="Final estimate"):
        adapter.parse_validate({"instrument": "ppl"},
                               str(FIXTURES / "perplexity_unparseable.log"))


# -- validate run（host 假仪器脚本，spec: benchmark-correctness）------------

import textwrap  # noqa: E402

from mobench.config import load_config  # noqa: E402
from mobench.validation import ValidationError, run_validation  # noqa: E402

_PROMPT_TEXT = "Once upon a time " * 10
_GREEDY_TEXT = _PROMPT_TEXT + "GENERATED-SUFFIX"


def fake_validate_bins(bin_dir):
    """Fake llama-bench (version probe) + completion/perplexity instruments.

    The completion fake echoes the actual -f prompt file (like the engine's
    prompt echo) so the parser's echo verification is exercised.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    bench = bin_dir / "llama-bench"
    bench.write_text("#!/usr/bin/env python3\nprint('version: 7165 (fake)')\n",
                     encoding="utf-8")
    completion = bin_dir / "llama-completion"
    completion.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "argv = sys.argv[1:]\n"
        "prompt = ''\n"
        "if '-f' in argv:\n"
        "    prompt = open(argv[argv.index('-f') + 1], encoding='utf-8').read()\n"
        "print('build: 7165 (fake)')\n"
        "print('llama_context: n_threads     = 4')\n"
        "print('sampler seed: 1')\n"
        "print('generate: n_ctx = 8192, n_batch = 2048, n_predict = 16, n_keep = 0')\n"
        "sys.stdout.write(prompt + 'GENERATED-SUFFIX [end of text]\\n\\n\\n')\n"
        "print('common_perf_print:        load time =      10.00 ms')\n"
        "print('common_perf_print: prompt eval time =       5.00 ms /   512 tokens"
        " (    0.01 ms per token, 102400.00 tokens per second)')\n"
        "print('common_perf_print:       total time =     105.00 ms /   528 tokens')\n",
        encoding="utf-8")
    perplexity = bin_dir / "llama-perplexity"
    perplexity.write_text(
        "#!/usr/bin/env python3\n"
        "print('load_backend: loaded CPU backend from libggml-cpu.so')\n"
        "print('llama_perplexity: calculating perplexity over 8 chunks, n_ctx=1024')\n"
        "print('Final estimate: PPL = 1.0086 +/- 0.05123')\n",
        encoding="utf-8")
    for p in (bench, completion, perplexity):
        p.chmod(0o755)


def setup_validate_project(tmp_path, ab=False):
    """Project with prompts + ppl declared; optionally a -fa off B config."""
    fake_validate_bins(tmp_path / "bin")
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "tiny.gguf").write_bytes(b"GGUF" * 64)
    (tmp_path / "prompts").mkdir(exist_ok=True)
    (tmp_path / "prompts" / "long-01.txt").write_text("Once upon a time " * 10,
                                                      encoding="utf-8")
    (tmp_path / "wiki.test.raw").write_text("the quick brown fox " * 50,
                                            encoding="utf-8")
    body = textwrap.dedent("""
        engine: llamacpp
        platform:
          type: host
        build:
          bin_dir: bin
          bench_bin: llama-bench
        models:
          - id: tiny
            path: models/tiny.gguf
            kind: llm
            precision: q4_k_m
        workload:
          threads: [4]
        methodology:
          env_control: none
        validation:
          prompts: [prompts/long-01.txt]
          gen_tokens: 16
          ppl:
            dataset: wiki.test.raw
            ctx: 1024
            chunks: 8
        """)
    cfg_path = tmp_path / "mobench.yml"
    cfg_path.write_text(body, encoding="utf-8")
    cfg_b_path = None
    if ab:
        cfg_b_path = tmp_path / "mobench-b.yml"
        cfg_b_path.write_text(body + "  flash_attn: off\n", encoding="utf-8")
        # workload 段在 body 里已闭合，B 档单独给 flash_attn：
        cfg_b_path.write_text(
            body.replace("methodology:", "workload:\n  flash_attn: off\nmethodology:"),
            encoding="utf-8")
    return str(cfg_path), (str(cfg_b_path) if cfg_b_path else None)


def test_validate_run_writes_self_describing_artifact(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    cfg_path, _ = setup_validate_project(tmp_path)

    # Act
    art = run_validation(load_config(cfg_path), config_path=cfg_path)

    # Assert: 工件四要素完备（指纹/覆盖/结果/日志引用）
    import hashlib
    assert art["schema"] == "mobench/validate@1"
    ctrl = art["controls"]
    assert ctrl["threads"] == 4
    assert len(ctrl["model_md5"]) == 32
    assert hashlib.sha256(("long-01.txt" + "Once upon a time " * 10)
                          .encode()).hexdigest() == ctrl["prompts_sha256"]
    assert ctrl["sampling"] == "greedy"
    assert ctrl["gen_tokens"] == 16
    assert ctrl["ppl"]["ctx"] == 1024
    assert len(ctrl["ppl"]["dataset_sha256"]) == 64
    cov = art["coverage"]
    assert cov["prefill"]["covered"] is True and cov["prefill"]["by"] == "ppl"
    assert cov["decode"]["covered"] is True
    assert cov["decode"]["n_kv_range"] == [512, 528]
    side = art["sides"]["a"]
    assert side["instruments"]["ppl"]["value"] == 1.0086
    greedy = side["instruments"]["greedy"]["runs"][0]
    assert greedy["text"] == _GREEDY_TEXT
    # 日志引用存在且非空；records.jsonl 不被创建（与性能记录链隔离）
    import os
    for inst in ("ppl", "greedy"):
        entries = ([side["instruments"][inst]] if inst == "ppl"
                   else side["instruments"][inst]["runs"])
        for entry in entries:
            log_path = os.path.join(art["out_dir"], entry["log"])
            assert os.path.getsize(log_path) > 0
    assert not (tmp_path / art["out_dir"] / "records.jsonl").exists()


def test_validate_run_ppl_only_marks_decode_not_covered(tmp_path, monkeypatch):
    # Arrange: 仅 ppl —— 盲区必须显式（不覆盖不宣称通过）
    monkeypatch.chdir(tmp_path)
    cfg_path, _ = setup_validate_project(tmp_path)
    body = open(cfg_path, encoding="utf-8").read()
    body = "\n".join(ln for ln in body.splitlines()
                     if "prompts:" not in ln and "gen_tokens:" not in ln) + "\n"
    open(cfg_path, "w", encoding="utf-8").write(body)

    # Act
    art = run_validation(load_config(cfg_path), config_path=cfg_path)

    # Assert
    assert art["coverage"]["decode"]["covered"] is False
    assert "not configured" in art["coverage"]["decode"]["reason"]


def test_validate_run_missing_prompt_file_fails_fast(tmp_path, monkeypatch):
    # Arrange: 声明的 prompt 文件缺失 → 残缺指纹不得写出工件
    monkeypatch.chdir(tmp_path)
    cfg_path, _ = setup_validate_project(tmp_path)
    import os
    os.remove(tmp_path / "prompts" / "long-01.txt")

    # Act / Assert
    with pytest.raises(ValidationError, match="prompt"):
        run_validation(load_config(cfg_path), config_path=cfg_path)
    assert not list((tmp_path / "results").glob("validate_*/validate.json"))


def test_validate_run_engine_crash_marks_instrument_unavailable(tmp_path, monkeypatch):
    # Arrange: completion 崩溃 → 该仪器 unavailable，工件照常落盘
    monkeypatch.chdir(tmp_path)
    cfg_path, _ = setup_validate_project(tmp_path)
    (tmp_path / "bin" / "llama-completion").write_text(
        "#!/usr/bin/env python3\nimport sys\nsys.exit(3)\n", encoding="utf-8")
    (tmp_path / "bin" / "llama-completion").chmod(0o755)

    # Act
    art = run_validation(load_config(cfg_path), config_path=cfg_path)

    # Assert
    greedy = art["sides"]["a"]["instruments"]["greedy"]
    assert greedy["status"] == "unavailable"
    assert "exit" in greedy["runs"][0]["reason"] or "exited" in greedy["runs"][0]["reason"]
    assert art["coverage"]["decode"]["covered"] is False
    assert art["sides"]["a"]["instruments"]["ppl"]["status"] == "ok"


def test_validate_ab_controls_mismatch_refused(tmp_path, monkeypatch):
    # Arrange: B 档线程数不同 —— controls 不一致必须执行前拒绝
    monkeypatch.chdir(tmp_path)
    cfg_path, _ = setup_validate_project(tmp_path)
    cfg_b_path = str(tmp_path / "mobench-b.yml")
    open(cfg_b_path, "w", encoding="utf-8").write(
        open(cfg_path, encoding="utf-8").read().replace("threads: [4]", "threads: [8]"))

    # Act / Assert
    with pytest.raises(ValidationError, match="controls"):
        run_validation(load_config(cfg_path), config_b=load_config(cfg_b_path),
                       config_path=cfg_path, config_b_path=cfg_b_path)


def test_validate_ab_records_both_sides(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    cfg_path, cfg_b_path = setup_validate_project(tmp_path, ab=True)

    # Act
    art = run_validation(load_config(cfg_path), config_b=load_config(cfg_b_path),
                         config_path=cfg_path, config_b_path=cfg_b_path)

    # Assert: 双列 + 各自 identity（B 带 flash_attn=off 差异）
    assert set(art["sides"]) == {"a", "b"}
    assert art["sides"]["a"]["identity"]["params"].get("flash_attn") in (None, "auto")
    assert art["sides"]["b"]["identity"]["params"]["flash_attn"] == "off"
    assert art["sides"]["a"]["identity"]["binary_md5"]


# -- validate compare（离线判定门，无需设备）--------------------------------

from mobench.analysis.correctness import (CompareError, compare_artifacts,  # noqa: E402
                                          load_artifact)

CTRL = {"model_md5": "a" * 32, "threads": 4, "sampling": "greedy",
        "gen_tokens": 16, "prompts_sha256": "b" * 64,
        "ppl": {"dataset_sha256": "c" * 64, "ctx": 1024, "chunks": 8}}
COV = {"prefill": {"covered": True, "by": "ppl", "ctx": 1024},
       "decode": {"covered": True, "by": "greedy", "n_kv_range": [512, 528]}}


def make_artifact(ppl=1.0086, text="Hello world", n_prompt=512,
                  ppl_status="ok", greedy_status="ok",
                  controls=None, coverage=None, params=None, side_b=None):
    def instruments():
        return {
            "ppl": {"status": ppl_status, "value": ppl, "chunks": 8,
                    "log": "raw_ppl_a.log"},
            "greedy": {"status": "ok", "reason": None,
                       "includes_prompt_echo": True,
                       "runs": [{"prompt": "prompts/long-01.txt",
                                 "status": greedy_status, "text": text,
                                 "n_prompt_tokens": n_prompt,
                                 "log": "raw_greedy0_a.log"}]},
        }

    def side(inst=None):
        return {"identity": {"engine_version": "7165 (fake)",
                             "binary_md5": {"llama-completion": "d" * 32},
                             "params": params or {"flash_attn": "auto"}},
                "instruments": inst or instruments()}

    art = {"schema": "mobench/validate@1", "engine": "llamacpp",
           "model": {"id": "tiny", "path": "models/tiny.gguf", "md5": "a" * 32},
           "threads": 4, "controls": controls or dict(CTRL),
           "coverage": coverage or {k: dict(v) for k, v in COV.items()},
           "sides": {"a": side()}, "out_dir": "/tmp/x", "path": "/tmp/x/validate.json"}
    if side_b is not None:
        art["sides"]["b"] = side_b
    return art


def write_artifact(tmp_path, name, art):
    path = tmp_path / name
    path.write_text(json.dumps(art), encoding="utf-8")
    return str(path)


def test_compare_pass_exact(tmp_path):
    # Arrange
    a = write_artifact(tmp_path, "a.json", make_artifact())
    b = write_artifact(tmp_path, "b.json", make_artifact())

    # Act
    result = compare_artifacts(a, b)

    # Assert
    assert result["verdict"] == "PASS"
    assert result["exit_code"] == 0
    assert result["instruments"]["greedy"]["runs"][0]["verdict"] == "PASS"


def test_compare_intra_ab_mode(tmp_path):
    # Arrange: 单工件双列（A/B run 产物），identity 不同、结果相同
    art = make_artifact(params={"flash_attn": "auto"},
                        side_b={"identity": {"engine_version": "7165 (fake)",
                                             "binary_md5": {"llama-completion": "d" * 32},
                                             "params": {"flash_attn": "off"}},
                                "instruments": make_artifact()["sides"]["a"]["instruments"]})
    path = write_artifact(tmp_path, "ab.json", art)

    # Act
    result = compare_artifacts(path, None)

    # Assert
    assert result["mode"] == "intra"
    assert result["verdict"] == "PASS"
    assert any("flash_attn" in d for d in result["identity_diff"])


def test_compare_greedy_divergence_fails_with_first_diff(tmp_path):
    # Arrange: 第 5 字节起分歧（"Hello world" vs "Hello_world"）
    a = write_artifact(tmp_path, "a.json", make_artifact(text="Hello world"))
    b = write_artifact(tmp_path, "b.json", make_artifact(text="Hello_world extra"))

    # Act
    result = compare_artifacts(a, b)

    # Assert
    assert result["verdict"] == "FAIL"
    assert result["exit_code"] == 2
    run = result["instruments"]["greedy"]["runs"][0]
    assert run["verdict"] == "FAIL"
    assert run["first_divergence"] == 5
    assert run["divergence_rate"] > 0
    assert "Hello" in run["context"]["a"]


def test_compare_tolerance_turns_fail_into_warn(tmp_path):
    # Arrange: 17 字节里 1 字节分歧 ≈ 5.9%
    a = write_artifact(tmp_path, "a.json", make_artifact(text="Hello world"))
    b = write_artifact(tmp_path, "b.json", make_artifact(text="Hello_world"))

    # Act: 容差 10% 内 → WARN；exit 0
    result = compare_artifacts(a, b, tolerance=10.0)

    # Assert
    assert result["verdict"] == "WARN"
    assert result["exit_code"] == 0
    assert result["instruments"]["greedy"]["runs"][0]["first_divergence"] == 5


def test_compare_controls_mismatch_incomparable(tmp_path):
    # Arrange: 线程数不同 —— 数值不可比，拒绝
    a = write_artifact(tmp_path, "a.json", make_artifact())
    bad = dict(CTRL, threads=8)
    b = write_artifact(tmp_path, "b.json", make_artifact(controls=bad))

    # Act
    result = compare_artifacts(a, b)

    # Assert
    assert result["verdict"] == "INCOMPARABLE"
    assert result["exit_code"] == 1
    assert any("threads" in d for d in result["controls_diff"])


def test_compare_coverage_mismatch_incomparable(tmp_path):
    # Arrange: 基线覆盖 decode、候选没有 —— 不可比
    a = write_artifact(tmp_path, "a.json", make_artifact())
    cov = {"prefill": {"covered": True, "by": "ppl", "ctx": 1024},
           "decode": {"covered": False, "by": None, "reason": "instrument not configured"}}
    b = write_artifact(tmp_path, "b.json", make_artifact(coverage=cov))

    # Act
    result = compare_artifacts(a, b)

    # Assert
    assert result["verdict"] == "INCOMPARABLE"
    assert any("decode" in d for d in result["coverage_diff"])


def test_compare_unavailable_is_not_fabricated(tmp_path):
    # Arrange: 候选 ppl 不可用
    a = write_artifact(tmp_path, "a.json", make_artifact())
    b = write_artifact(tmp_path, "b.json", make_artifact(ppl_status="unavailable"))

    # Act
    result = compare_artifacts(a, b)

    # Assert: 不伪造数值，整体 UNAVAILABLE → exit 1
    assert result["verdict"] == "UNAVAILABLE"
    assert result["exit_code"] == 1
    ppl = result["instruments"]["ppl"]
    assert ppl["verdict"] == "UNAVAILABLE"
    assert ppl.get("b") is None or "value" not in ppl.get("b", {})


def test_compare_ppl_tolerance(tmp_path):
    # Arrange: Δ=0.14%
    a = write_artifact(tmp_path, "a.json", make_artifact(ppl=1.0086))
    b = write_artifact(tmp_path, "b.json", make_artifact(ppl=1.0100))

    # Act: exact → FAIL；容差 0.2% → WARN
    exact = compare_artifacts(a, b)
    toler = compare_artifacts(a, b, ppl_tolerance=0.2)

    # Assert
    assert exact["verdict"] == "FAIL"
    assert toler["verdict"] == "WARN"
    assert toler["exit_code"] == 0


def test_compare_pass_but_uncovered_carries_annotation(tmp_path):
    # Arrange: 双列都不覆盖 decode（仅 ppl 工件, intra）
    cov = {"prefill": {"covered": True, "by": "ppl", "ctx": 1024},
           "decode": {"covered": False, "by": None, "reason": "instrument not configured"}}
    inst = make_artifact(coverage=cov)["sides"]["a"]["instruments"]
    art = make_artifact(coverage=cov, side_b={"identity": {},
                                              "instruments": inst})
    path = write_artifact(tmp_path, "ab.json", art)

    # Act
    result = compare_artifacts(path, None)

    # Assert: PASS 但必须显式携带 decode NOT COVERED
    assert result["verdict"] == "PASS"
    assert result["uncovered_paths"] == ["decode"]
    from mobench.analysis.correctness import render_report
    assert "NOT COVERED" in render_report(result)


# -- CLI 接入 ---------------------------------------------------------------

from mobench.cli import main as cli_main  # noqa: E402


def test_validate_cli_help_lazy_no_device(tmp_path, monkeypatch):
    # Arrange: --help 不得拉起设备依赖（惰性导入）
    monkeypatch.chdir(tmp_path)

    # Act
    code = cli_main(["validate", "--help"])

    # Assert
    assert code == 0


def test_validate_run_cli_json_pure(tmp_path, monkeypatch, capsys):
    # Arrange
    monkeypatch.chdir(tmp_path)
    cfg_path, _ = setup_validate_project(tmp_path)

    # Act
    code = cli_main(["validate", "run", "--config", cfg_path, "--json"])

    # Assert: stdout 为纯 JSON 工件（echo 被关闭）
    assert code == 0
    art = json.loads(capsys.readouterr().out)
    assert art["schema"] == "mobench/validate@1"
    assert art["coverage"]["decode"]["covered"] is True


def test_validate_compare_cli_exit_codes(tmp_path, capsys):
    # Arrange: PASS 对与分歧对
    a = write_artifact(tmp_path, "a.json", make_artifact())
    b = write_artifact(tmp_path, "b.json", make_artifact())
    c = write_artifact(tmp_path, "c.json", make_artifact(text="Hello_world"))

    # Act
    code_pass = cli_main(["validate", "compare", "--baseline", a,
                          "--candidate", b, "--json"])
    out_pass = capsys.readouterr().out
    code_fail = cli_main(["validate", "compare", "--baseline", a,
                          "--candidate", c, "--json"])
    out_fail = capsys.readouterr().out

    # Assert: 判定门语义 —— PASS=0 / FAIL=2
    assert code_pass == 0
    assert code_fail == 2
    assert json.loads(out_pass)["verdict"] == "PASS"
    assert json.loads(out_fail)["verdict"] == "FAIL"


def test_validate_compare_cli_human_report_marks_uncovered(tmp_path, capsys):
    # Arrange: intra 工件, decode 不覆盖
    cov = {"prefill": {"covered": True, "by": "ppl", "ctx": 1024},
           "decode": {"covered": False, "by": None,
                      "reason": "instrument not configured"}}
    inst = make_artifact(coverage=cov)["sides"]["a"]["instruments"]
    path = write_artifact(tmp_path, "ab.json",
                          make_artifact(coverage=cov,
                                        side_b={"identity": {}, "instruments": inst}))

    # Act
    code = cli_main(["validate", "compare", "--baseline", path])

    # Assert
    assert code == 0
    out = capsys.readouterr().out
    assert "NOT COVERED" in out


# -- 分支补漏（错误/降级路径；对齐 test_coverage_targets 惯例）--------------

from mobench.analysis import correctness as corr_mod  # noqa: E402
from mobench.validation import render_summary  # noqa: E402


def _write_min_config(tmp_path, extra="", validation="validation:\n  prompts: [p.txt]\n"):
    cfg = tmp_path / "mini.yml"
    cfg.write_text(
        "engine: llamacpp\nplatform:\n  type: host\n"
        "build: {bin_dir: bin, bench_bin: llama-bench}\n"
        "models:\n  - {id: tiny, path: models/tiny.gguf}\n"
        + validation + extra, encoding="utf-8")
    return str(cfg)


def test_validate_run_without_validation_section(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    fake_validate_bins(tmp_path / "bin")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "tiny.gguf").write_bytes(b"G")
    cfg = _write_min_config(tmp_path, validation="")

    # Act / Assert
    with pytest.raises(ValidationError, match="validation"):
        run_validation(load_config(cfg), config_path=cfg)


def test_validate_run_missing_model_file(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    fake_validate_bins(tmp_path / "bin")
    (tmp_path / "models").mkdir()
    (tmp_path / "p.txt").write_text("x", encoding="utf-8")
    cfg = _write_min_config(tmp_path)

    # Act / Assert
    with pytest.raises(ValidationError, match="model file missing"):
        run_validation(load_config(cfg), config_path=cfg)


def test_validate_run_missing_dataset(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    fake_validate_bins(tmp_path / "bin")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "tiny.gguf").write_bytes(b"G")
    cfg = _write_min_config(
        tmp_path, validation="validation:\n  ppl:\n    dataset: gone.raw\n")

    # Act / Assert
    with pytest.raises(ValidationError, match="dataset missing"):
        run_validation(load_config(cfg), config_path=cfg)


def test_validate_ab_engine_mismatch_refused(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    cfg_path, _ = setup_validate_project(tmp_path)
    cfg_b = {"engine": "mnn", "models": [{"id": "tiny"}]}

    # Act / Assert
    with pytest.raises(ValidationError, match="same engine"):
        run_validation(load_config(cfg_path), config_b=cfg_b, config_path=cfg_path)


def test_validate_ab_model_id_mismatch_refused(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    cfg_path, _ = setup_validate_project(tmp_path)
    cfg_b = {"engine": "llamacpp", "models": [{"id": "other"}]}

    # Act / Assert
    with pytest.raises(ValidationError, match="model id"):
        run_validation(load_config(cfg_path), config_b=cfg_b, config_path=cfg_path)


def test_validate_unknown_instrument_parse(tmp_path):
    # Arrange
    adapter = LlamacppAdapter(make_llamacpp_cfg(tmp_path))

    # Act / Assert
    with pytest.raises(AdapterError, match="unknown validation instrument"):
        adapter.parse_validate({"instrument": "bogus"},
                               str(FIXTURES / "perplexity_ok.log"))


def test_validate_run_cli_human_and_error_paths(tmp_path, monkeypatch, capsys):
    # Arrange
    monkeypatch.chdir(tmp_path)
    cfg_path, _ = setup_validate_project(tmp_path)

    # Act: 人类可读输出
    code = cli_main(["validate", "run", "--config", cfg_path])
    out = capsys.readouterr().out
    # 以及错误路径（配置不存在）
    code_err = cli_main(["validate", "run", "--config", "missing.yml"])

    # Assert
    assert code == 0
    assert "coverage" in out
    assert code_err == 1


def test_validate_run_cli_no_usable_instrument_exits_1(tmp_path, monkeypatch, capsys):
    # Arrange: 全部仪器失败 → 工件落盘但仍以 1 退出
    monkeypatch.chdir(tmp_path)
    cfg_path, _ = setup_validate_project(tmp_path)
    for name in ("llama-completion", "llama-perplexity"):
        script = tmp_path / "bin" / name
        script.write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(3)\n",
                          encoding="utf-8")
        script.chmod(0o755)

    # Act
    code = cli_main(["validate", "run", "--config", cfg_path, "--json"])

    # Assert
    assert code == 1
    art = json.loads(capsys.readouterr().out)
    assert art["coverage"]["prefill"]["covered"] is False
    assert not render_summary(art).startswith("x")  # 摘要可渲染


def test_load_artifact_error_paths(tmp_path):
    # Arrange
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    wrong_schema = tmp_path / "wrong.json"
    wrong_schema.write_text(json.dumps({"schema": "other/1"}), encoding="utf-8")
    no_side = tmp_path / "noside.json"
    no_side.write_text(json.dumps({"schema": "mobench/validate@1", "sides": {}}),
                       encoding="utf-8")

    # Act / Assert
    with pytest.raises(CompareError, match="cannot read"):
        load_artifact(str(bad_json))
    with pytest.raises(CompareError, match="not a validate artifact"):
        load_artifact(str(wrong_schema))
    with pytest.raises(CompareError, match="side 'a'"):
        load_artifact(str(no_side))


def test_compare_intra_requires_side_b(tmp_path):
    # Arrange
    path = write_artifact(tmp_path, "single.json", make_artifact())

    # Act / Assert
    with pytest.raises(CompareError, match="side 'b'"):
        compare_artifacts(path, None)


def test_compare_greedy_run_count_mismatch(tmp_path):
    # Arrange
    art_b = make_artifact()
    art_b["sides"]["a"]["instruments"]["greedy"]["runs"].append(
        {"prompt": "p2", "status": "ok", "text": "Hello world",
         "n_prompt_tokens": 512, "log": "l"})
    a = write_artifact(tmp_path, "a.json", make_artifact())
    b = write_artifact(tmp_path, "b.json", art_b)

    # Act
    result = compare_artifacts(a, b)

    # Assert
    assert result["instruments"]["greedy"]["verdict"] == "UNAVAILABLE"
    assert "run count" in result["instruments"]["greedy"]["reason"]


def test_compare_aggregate_unavailable_reason(tmp_path):
    # Arrange: 两侧 greedy 聚合状态都 unavailable
    art_a, art_b = make_artifact(), make_artifact()
    art_a["sides"]["a"]["instruments"]["greedy"]["status"] = "unavailable"
    art_b["sides"]["a"]["instruments"]["greedy"]["status"] = "unavailable"
    a = write_artifact(tmp_path, "a.json", art_a)
    b = write_artifact(tmp_path, "b.json", art_b)

    # Act
    result = compare_artifacts(a, b)

    # Assert
    assert result["instruments"]["greedy"]["verdict"] == "UNAVAILABLE"
    assert result["verdict"] == "UNAVAILABLE"


def test_compare_ppl_missing_on_one_side(tmp_path):
    # Arrange
    art_b = make_artifact()
    del art_b["sides"]["a"]["instruments"]["ppl"]
    a = write_artifact(tmp_path, "a.json", make_artifact())
    b = write_artifact(tmp_path, "b.json", art_b)

    # Act
    result = compare_artifacts(a, b)

    # Assert
    assert result["instruments"]["ppl"]["verdict"] == "UNAVAILABLE"
    assert "not present" in result["instruments"]["ppl"]["reason"]


def test_compare_ppl_zero_baseline_no_fabricated_delta(tmp_path):
    # Arrange: 基线 0 无法算相对差 —— 不得伪造 delta
    a = write_artifact(tmp_path, "a.json", make_artifact(ppl=0.0))
    b = write_artifact(tmp_path, "b.json", make_artifact(ppl=1.0))

    # Act
    result = compare_artifacts(a, b)

    # Assert
    assert result["instruments"]["ppl"]["verdict"] == "FAIL"
    assert result["instruments"]["ppl"]["delta_pct"] is None


def test_correctness_main_paths(tmp_path, capsys):
    # Arrange
    import argparse
    a = write_artifact(tmp_path, "a.json", make_artifact())
    b = write_artifact(tmp_path, "b.json", make_artifact(text="Hello_world"))

    # Act: json 输出
    code = corr_mod.main(argparse.Namespace(
        baseline=a, candidate=b, tolerance=0.0, ppl_tolerance=None, json=True))
    out = capsys.readouterr().out
    # 人类可读输出
    code_h = corr_mod.main(argparse.Namespace(
        baseline=a, candidate=b, tolerance=0.0, ppl_tolerance=None, json=False))
    out_h = capsys.readouterr().out
    # 错误路径
    code_err = corr_mod.main(argparse.Namespace(
        baseline=str(tmp_path / "missing.json"), candidate=None,
        tolerance=0.0, ppl_tolerance=None, json=True))

    # Assert
    assert code == 2 and code_h == 2
    assert json.loads(out)["verdict"] == "FAIL"
    assert "first divergence" in out_h
    assert code_err == 1


# -- 真机双布局解析（fixture = 设备实跑日志，add-correctness-validation）----

def test_parse_greedy_device_split_layout(tmp_path):
    # Arrange: 真机布局 = stdout（回显+生成，无前缀）在前、stderr（日志，带
    # "h.mm.ss.mmm L " 前缀）在后（adb 平台先 stdout 后 stderr 拼接）
    adapter = LlamacppAdapter(make_llamacpp_cfg(tmp_path))

    # Act
    result = adapter.parse_validate(
        {"instrument": "greedy"},
        str(FIXTURES / "completion_device_layout.log"))

    # Assert: 流 = prompt 回显 + 生成段（n_predict=64 截断，无 EOG）
    assert result["text"].startswith("Robert Boulter is an English film")
    assert result["text"].endswith("Ben Whish")
    assert result["n_prompt_tokens"] == 814


def test_parse_greedy_prompt_echo_mismatch_refuses(tmp_path):
    # Arrange: 流不以 prompt 回显开头 → 拒绝抽取（宁可 unavailable 不猜）
    prompt = tmp_path / "p.txt"
    prompt.write_text("EXPECTED PREFIX", encoding="utf-8")
    log = tmp_path / "mismatch.log"
    log.write_text(
        "generate: n_ctx = 8192, n_batch = 2048, n_predict = 16, n_keep = 0\n"
        "WRONG ECHO text [end of text]\n\n\n"
        "common_perf_print: prompt eval time = 5.00 ms / 10 tokens\n",
        encoding="utf-8")
    adapter = LlamacppAdapter(make_llamacpp_cfg(tmp_path))

    # Act / Assert
    with pytest.raises(AdapterError, match="prompt echo"):
        adapter.parse_validate({"instrument": "greedy", "prompt": str(prompt)},
                               str(log))


def test_validate_ab_threads_override_applies_to_both_sides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path, cfg_b_path = setup_validate_project(tmp_path, ab=True)
    art = run_validation(
        load_config(cfg_path),
        config_b=load_config(cfg_b_path),
        threads=8,
        config_path=cfg_path,
        config_b_path=cfg_b_path,
    )
    assert art["threads"] == 8
    assert art["controls"]["threads"] == 8
    assert set(art["sides"]) == {"a", "b"}


def test_validate_ab_side_b_failure_marks_coverage_uncovered(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path, cfg_b_path = setup_validate_project(tmp_path, ab=True)
    # 构造 side b 使用独立 bin 目录，其 llama-completion 退出非零
    bin_b = tmp_path / "bin_b"
    fake_validate_bins(bin_b)
    (bin_b / "llama-completion").write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(2)\n", encoding="utf-8")
    (bin_b / "llama-completion").chmod(0o755)
    body_b = open(cfg_b_path, encoding="utf-8").read().replace("bin_dir: bin", f"bin_dir: {bin_b}")
    open(cfg_b_path, "w", encoding="utf-8").write(body_b)

    art = run_validation(
        load_config(cfg_path),
        config_b=load_config(cfg_b_path),
        config_path=cfg_path,
        config_b_path=cfg_b_path,
    )
    assert art["coverage"]["decode"]["covered"] is False
    assert "side b" in art["coverage"]["decode"]["reason"]
    assert art["coverage"]["prefill"]["covered"] is True
