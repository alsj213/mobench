"""Log probes: forbid aborts the run, expect marks it invalid (测量有效性门禁)."""
import pytest

from mobench.config import load_config
from mobench.log_probe import (LogProbeError, compile_checks, scan_expect,
                               scan_forbid)
from mobench.run import RunError, run_pipeline


def test_compile_none_when_absent():
    assert compile_checks(None) is None
    assert compile_checks({}) is None
    assert compile_checks({"log_checks": {}}) is None


def test_compile_ok():
    checks = compile_checks({"log_checks": {"forbid": ["boom"], "expect": ["ok"]}})
    assert checks["forbid"][0][0] == "boom"
    assert checks["expect"][0][0] == "ok"


def test_compile_rejects_malformed():
    with pytest.raises(LogProbeError):
        compile_checks({"log_checks": {"forbid": ["(unclosed"]}})
    with pytest.raises(LogProbeError):
        compile_checks({"log_checks": {"forbid": "not-a-list"}})
    with pytest.raises(LogProbeError):
        compile_checks({"log_checks": {"expect": [42]}})


def test_scan_forbid_reports_line(tmp_path):
    log = tmp_path / "raw.log"
    log.write_text("line one\ncannot be used with CPU_REPACK\nlast\n",
                   encoding="utf-8")
    checks = compile_checks({"log_checks": {"forbid": ["CPU_REPACK"]}})
    v = scan_forbid(str(log), checks)
    assert len(v) == 1
    assert v[0]["line"] == 2
    assert "CPU_REPACK" in v[0]["matched"]


def test_scan_expect_reports_missing(tmp_path):
    log = tmp_path / "raw.log"
    log.write_text("repack: enabled\n", encoding="utf-8")
    checks = compile_checks(
        {"log_checks": {"expect": ["repack: enabled", "never-here"]}})
    v = scan_expect(str(log), checks)
    assert len(v) == 1
    assert v[0]["pattern"] == "never-here"
    assert v[0]["missing"] is True


# -- pipeline integration ----------------------------------------------------

def fake_llama_bench(path, extra_lines=()):
    """Fake llama-bench emitting JSON plus optional extra stdout lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = "".join(f"print({line!r})\n" for line in extra_lines)
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, json\n"
        "print(json.dumps({'build':'fake','results':[\n"
        "  {'n_prompt':64,'n_gen':128,'t_prompt_processing':3.2,'t_generation':21.0}]}))\n"
        + extra,
        encoding="utf-8",
    )
    path.chmod(0o755)


def setup_project(tmp_path, log_checks_yaml="", extra_lines=()):
    fake_llama_bench(tmp_path / "bin" / "llama-bench", extra_lines)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "models" / "tinyllama.gguf").write_bytes(b"GGUF" * 64)
    cfg = tmp_path / "mobench.yml"
    cfg.write_text(
        "engine: llamacpp\n"
        "platform:\n  type: host\n"
        "build:\n  bin_dir: bin\n  bench_bin: llama-bench\n"
        "models:\n  - id: tinyllama\n    path: models/tinyllama.gguf\n"
        "    kind: llm\n    precision: q8_0\n"
        "workload:\n  threads: [4]\n  rounds: 2\n"
        + ("methodology:\n" + log_checks_yaml if log_checks_yaml else "")
        + "results_dir: results\n",
        encoding="utf-8",
    )
    return str(cfg)


def test_pipeline_forbid_aborts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = setup_project(
        tmp_path,
        log_checks_yaml='  log_checks:\n    forbid:\n      - "CPU_REPACK"\n',
        extra_lines=["cannot be used with CPU_REPACK (test)"])
    with pytest.raises(RunError) as ei:
        run_pipeline(load_config(cfg_path))
    assert "[step:execution]" in str(ei.value)
    assert "CPU_REPACK" in str(ei.value)


def test_pipeline_expect_missing_marks_invalid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = setup_project(
        tmp_path,
        log_checks_yaml='  log_checks:\n    expect:\n      - "repack path active"\n')
    meta = run_pipeline(load_config(cfg_path))
    assert meta["log_audit"]["valid"] is False
    assert len(meta["log_audit"]["violations"]) == 2  # 2 rounds x 1 pattern
    v = meta["log_audit"]["violations"][0]
    assert v["round"] == 1 and v["model"] == "tinyllama" and v["threads"] == 4


def test_pipeline_expect_hit_valid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = setup_project(
        tmp_path,
        log_checks_yaml='  log_checks:\n    expect:\n      - "repack path active"\n',
        extra_lines=["repack path active"])
    meta = run_pipeline(load_config(cfg_path))
    assert meta["log_audit"] == {"valid": True, "violations": []}


def test_pipeline_without_checks_has_no_audit_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = setup_project(tmp_path)
    meta = run_pipeline(load_config(cfg_path))
    assert "log_audit" not in meta


def test_pipeline_forbid_abort_writes_invalid_run_json(tmp_path, monkeypatch):
    """命中 forbid 中止后仍写 run.json，且 log_audit 带 forbid 证据。"""
    import json

    from mobench.records import read_paths

    monkeypatch.chdir(tmp_path)
    cfg_path = setup_project(
        tmp_path,
        log_checks_yaml='  log_checks:\n    forbid:\n      - "CPU_REPACK"\n',
        extra_lines=["cannot be used with CPU_REPACK (test)"])
    with pytest.raises(RunError):
        run_pipeline(load_config(cfg_path))

    run_jsons = list((tmp_path / "results").glob("run_*/run.json"))
    assert len(run_jsons) == 1
    meta = json.loads(run_jsons[0].read_text(encoding="utf-8"))
    assert meta["valid"] is False
    assert meta["aborted"] is True
    assert meta["log_audit"]["valid"] is False
    assert meta["log_audit"]["forbid"]["pattern"] == "CPU_REPACK"
    assert "CPU_REPACK" in meta["error"]
    assert read_paths([str(tmp_path / "results")]) == []
