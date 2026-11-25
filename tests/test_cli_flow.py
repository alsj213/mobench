"""CLI 端到端：驱动每个子命令的 dispatch + main + 错误路径（确定性命令行入口）。

goes through cli.main(argv) so argparse builds the Namespace and the cmd_*
dispatch functions are exercised, not just each module's main().
"""
import json

from mobench.cli import main as cli_main
from mobench.records import RECORD_SCHEMA, append_record


def write_records(tmp_path, values, fname="records.jsonl"):
    raw = tmp_path / "raw.log"
    raw.write_text("raw bench log\n", encoding="utf-8")
    recs = tmp_path / fname
    for i, v in enumerate(values, start=1):
        append_record(str(recs), {
            "schema": RECORD_SCHEMA, "engine": "llamacpp",
            "workload_id": "tinyllama", "metric": "gen_tok_s",
            "value": v, "unit": "tok/s", "threads": 4, "precision": "q8_0",
            "round": i, "platform": "host", "source_log": str(raw),
        })
    return str(recs)


def write_config(tmp_path, body):
    cfg = tmp_path / "mobench.yml"
    cfg.write_text(body, encoding="utf-8")
    return str(cfg)


# ---- stats ------------------------------------------------------------
def test_stats_cli_json_and_text(tmp_path, capsys):
    recs = write_records(tmp_path, [10.0, 10.1, 10.2, 10.3, 10.4])

    assert cli_main(["stats", recs, "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["n"] == 5
    assert rows[0]["samples_min"] is True

    assert cli_main(["stats", recs]) == 0
    assert "多轮统计结果" in capsys.readouterr().out


def test_stats_cli_insufficient_and_error(tmp_path, capsys):
    recs = write_records(tmp_path, [10.0, 10.2], fname="few.jsonl")
    assert cli_main(["stats", recs]) == 0
    out = capsys.readouterr().out
    assert "only 2 run(s)" in out

    assert cli_main(["stats", str(tmp_path / "no_such_dir")]) == 1
    assert "no records" in capsys.readouterr().err


# ---- report -----------------------------------------------------------
def test_report_cli_md_html(tmp_path, capsys):
    recs = write_records(tmp_path, [10.0, 10.2, 10.1])
    out = tmp_path / "rpt"

    assert cli_main(["report", recs, "--out", str(out)]) == 0
    assert (out / "benchmark_report.md").exists()
    assert (out / "benchmark_report.html").exists()

    assert cli_main(["report", "--format", "md", str(recs),
                     "--out", str(out)]) == 0
    capsys.readouterr()  # swallow md-only stdout


def test_report_cli_no_records_error(capsys):
    assert cli_main(["report", "/no_such_records"]) == 1
    assert "no records" in capsys.readouterr().err


# ---- regression -------------------------------------------------------
def test_regression_save_compare_cli(tmp_path, capsys):
    base = write_records(tmp_path, [50.0] * 5, fname="base.jsonl")
    baseline = tmp_path / "baseline.json"
    assert cli_main(["regression", "save", "--results", base,
                     "--out", str(baseline), "--json"]) == 0
    saved = json.loads(capsys.readouterr().out)
    assert saved["schema"] == "mobench/baseline@1"
    assert baseline.exists()

    cur = write_records(tmp_path, [40.0] * 5, fname="cur.jsonl")
    assert cli_main(["regression", "compare", "--results", cur,
                     "--baseline", str(baseline), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["rows"][0]["flagged"] is True
    assert result["rows"][0]["worse"] is True

    # text render path (render_report)
    assert cli_main(["regression", "compare", "--results", cur,
                     "--baseline", str(baseline)]) == 0
    assert "REGRESSION DETECTED" in capsys.readouterr().out


def test_regression_save_requires_out(tmp_path, capsys):
    recs = write_records(tmp_path, [50.0] * 5)
    assert cli_main(["regression", "save", "--results", recs]) == 1
    assert "--out required" in capsys.readouterr().err


def test_regression_no_records_error(tmp_path, capsys):
    assert cli_main(["regression", "compare", "--results", "/no_dir",
                     "--baseline", "/b.json"]) == 1
    assert "no records" in capsys.readouterr().err


# ---- power ------------------------------------------------------------
def test_power_cli_ok_text_and_json(tmp_path, capsys):
    src = tmp_path / "bs.txt"
    src.write_text("Capacity: 5000\nComputed drain: 100 mAh\n", encoding="utf-8")

    assert cli_main(["power", "--source", str(src),
                     "--inference-count", "100", "--throughput", "54.0"]) == 0
    assert "Energy per inference" in capsys.readouterr().out

    assert cli_main(["power", "--source", str(src), "--inference-count", "100",
                     "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["status"] == "ok"
    assert rep["energy_per_inference_j"] is not None


def test_power_cli_no_source_error(capsys):
    assert cli_main(["power"]) == 1
    assert "--source" in capsys.readouterr().err


def test_power_cli_unavailable_label(tmp_path, capsys):
    src = tmp_path / "empty.txt"
    src.write_text("no energy signal\n", encoding="utf-8")
    assert cli_main(["power", "--source", str(src), "--inference-count", "10",
                     "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["status"] == "unavailable"
    assert "unavailable" in rep["label"]


# ---- prepare ----------------------------------------------------------
def _prepare_cfg(tmp_path):
    model = tmp_path / "models" / "tinyllama.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"GGUF" * 8)
    return write_config(tmp_path,
        "engine: llamacpp\n"
        "models:\n  - id: tinyllama\n    path: models/tinyllama.gguf\n"
        "    kind: llm\n")


def test_prepare_cli_self_provided_text(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _prepare_cfg(tmp_path)
    assert cli_main(["prepare", "--config", cfg]) == 0
    assert "[prepare] tinyllama: ready" in capsys.readouterr().out


def test_prepare_cli_json_and_filter(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _prepare_cfg(tmp_path)
    assert cli_main(["prepare", "--config", cfg, "tinyllama", "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res[0]["status"] == "ready"


def test_prepare_cli_unknown_model_filter_error(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _prepare_cfg(tmp_path)
    assert cli_main(["prepare", "--config", cfg, "ghost"]) == 1
    assert "no models to prepare" in capsys.readouterr().err


def test_prepare_cli_missing_self_provided_error(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = write_config(tmp_path,
        "engine: llamacpp\n"
        "models:\n  - id: ghost\n    path: models/ghost.gguf\n    kind: llm\n")
    assert cli_main(["prepare", "--config", cfg]) == 1
    assert "model file missing" in capsys.readouterr().err


# ---- __main__ (python -m mobench) -------------------------------------
def test_python_dash_m_help_ok():
    import subprocess
    import sys

    proc = subprocess.run([sys.executable, "-m", "mobench", "--help"],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0
    assert "run" in proc.stdout
