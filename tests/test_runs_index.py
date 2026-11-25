"""Run index: one compact row per run.json (对话场景测量定位)."""
import json
import os

from mobench.cli import main as cli_main
from mobench.config import load_config
from mobench.run import run_pipeline
from mobench.runs_index import render_rows, scan_runs


def write_run(d, ts, **meta):
    run_dir = d / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    base = {"timestamp": ts, "engine": "llamacpp", "n_records": 6,
            "binary": {"md5": "abcdef1234567890"},
            "models": [{"id": "tinyllama"}]}
    base.update(meta)
    (run_dir / "run.json").write_text(json.dumps(base), encoding="utf-8")
    return run_dir


def test_scan_runs_fields(tmp_path):
    write_run(tmp_path, "20260101_000001", label="arm-a",
              build_recipe={"checked": True},
              log_audit={"valid": False, "violations": [1]})
    write_run(tmp_path, "20260101_000002")  # minimal legacy shape
    rows = scan_runs(str(tmp_path))
    assert len(rows) == 2
    a, b = rows
    assert a["label"] == "arm-a" and a["recipe_checked"] is True
    assert a["log_audit_valid"] is False
    assert b["label"] is None and b["recipe_checked"] is None
    assert b["log_audit_valid"] is None
    assert b["models"] == ["tinyllama"]


def test_scan_runs_skips_broken_json(tmp_path):
    (tmp_path / "run_20260101_000003").mkdir()
    (tmp_path / "run_20260101_000003" / "run.json").write_text(
        "{not json", encoding="utf-8")
    write_run(tmp_path, "20260101_000001")
    assert len(scan_runs(str(tmp_path))) == 1


def test_render_flags(tmp_path):
    write_run(tmp_path, "20260101_000001", label="arm-a",
              build_recipe={"checked": True}, log_audit={"valid": False},
              throttle_risk=True)
    out = render_rows(scan_runs(str(tmp_path)))
    assert "arm-a" in out
    assert "recipe-ok" in out
    assert "AUDIT-INVALID" in out
    assert "throttled" in out


def test_cli_runs_json(tmp_path, capsys):
    write_run(tmp_path, "20260101_000001", label="x")
    code = cli_main(["runs", "--results", str(tmp_path), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["label"] == "x"


def test_cli_runs_empty_dir_exits_nonzero(tmp_path, capsys):
    code = cli_main(["runs", "--results", str(tmp_path)])
    assert code == 1


# -- run --label integration -------------------------------------------------

def fake_llama_bench(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, json\n"
        "print(json.dumps({'build':'fake','results':[\n"
        "  {'n_prompt':64,'n_gen':128,'t_prompt_processing':3.2,'t_generation':21.0}]}))\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def setup_project(tmp_path):
    fake_llama_bench(tmp_path / "bin" / "llama-bench")
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "models" / "tinyllama.gguf").write_bytes(b"GGUF" * 64)
    cfg = tmp_path / "mobench.yml"
    cfg.write_text(
        "engine: llamacpp\n"
        "platform:\n  type: host\n"
        "build:\n  bin_dir: bin\n  bench_bin: llama-bench\n"
        "models:\n  - id: tinyllama\n    path: models/tinyllama.gguf\n"
        "    kind: llm\n    precision: q8_0\n"
        "workload:\n  threads: [4]\n  rounds: 1\n"
        "results_dir: results\n",
        encoding="utf-8",
    )
    return str(cfg)


def test_run_pipeline_writes_label_to_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = setup_project(tmp_path)
    meta = run_pipeline(load_config(cfg_path), label="k8-b-neon")
    assert meta["label"] == "k8-b-neon"
    run_json = os.path.join(os.path.dirname(meta["records"]), "run.json")
    on_disk = json.loads(open(run_json, encoding="utf-8").read())
    assert on_disk["label"] == "k8-b-neon"


def test_run_pipeline_without_label_has_no_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = setup_project(tmp_path)
    meta = run_pipeline(load_config(cfg_path))
    assert "label" not in meta


def test_cli_run_label(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cfg_path = setup_project(tmp_path)
    code = cli_main(["run", "--config", cfg_path, "--label", "cli-lbl", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["label"] == "cli-lbl"
