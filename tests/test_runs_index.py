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
              freq_audit={"valid": False}, throttle_risk=True)
    rows = scan_runs(str(tmp_path))
    assert rows[0]["freq_audit_valid"] is False
    assert rows[0]["valid"] is False
    out = render_rows(rows)
    assert "arm-a" in out
    assert "recipe-ok" in out
    assert "AUDIT-INVALID" in out
    assert "FREQ-INVALID" in out
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


# -- Wave 5 加固：aborted / cooldown 缺失 / 兜底 INVALID -------------------

def test_rows_expose_aborted_and_cooldown_reached(tmp_path):
    write_run(tmp_path, "20260101_000001", valid=False, aborted=True,
              cooldown={"target_c": 40, "reached": False, "temp_c": 52})
    rows = scan_runs(str(tmp_path))
    assert rows[0]["aborted"] is True
    assert rows[0]["cooldown_reached"] is False
    out = render_rows(rows)
    assert "ABORTED" in out
    assert "COOLDOWN-MISS" in out
    # 已有具体 invalid 标记时不再重复 INVALID
    assert "INVALID" not in out


def test_render_flags_falls_back_to_invalid(tmp_path):
    write_run(tmp_path, "20260101_000001", valid=False)
    rows = scan_runs(str(tmp_path))
    out = render_rows(rows)
    assert "INVALID" in out


def test_rows_detect_cooldown_miss_from_per_round_attempts(tmp_path):
    """cooldown_per_round=True 无顶层 cooldown，须从 attempts 聚合 COOLDOWN-MISS。"""
    write_run(
        tmp_path,
        "20260101_000002",
        attempts=[
            {"round": 1, "attempt": 1, "cooldown": {"target_c": 40, "reached": True, "temp_c": 39}},
            {"round": 2, "attempt": 1, "cooldown": {"target_c": 40, "reached": False, "temp_c": 49}},
        ],
    )
    rows = scan_runs(str(tmp_path))
    assert len(rows) == 1
    assert rows[0]["cooldown_reached"] is False
    out = render_rows(rows)
    assert "COOLDOWN-MISS" in out


def test_scan_runs_skips_non_dict_run_json(tmp_path):
    """非 dict JSON（如列表 [1, 2, 3]）的 run.json 不得导致 scan_runs 崩溃。"""
    bad_dir = tmp_path / "run_20260101_000003"
    bad_dir.mkdir()
    (bad_dir / "run.json").write_text("[1, 2, 3]", encoding="utf-8")
    write_run(tmp_path, "20260101_000004")
    rows = scan_runs(str(tmp_path))
    assert len(rows) == 1
    assert rows[0]["timestamp"] == "20260101_000004"
