"""Tests for report --format chart (report-multipanel-chart)."""

from mobench import report as report_mod
from mobench.records import RECORD_SCHEMA, append_record, read_paths


def seed(tmp_path, base="records.jsonl"):
    raw = tmp_path / "raw.log"
    raw.write_text("raw\n", encoding="utf-8")
    out = tmp_path / base
    # Decode + Prefill, threads {4,8} x prompt {512,1024}
    vals = {
        "gen_tok_s": {"4-512": 40.0, "4-1024": 36.0, "8-512": 46.0, "8-1024": 42.0},
        "prompt_eval_ms_tok": {"4-512": 4.0, "4-1024": 5.0, "8-512": 3.0, "8-1024": 3.5},
    }
    for metric, spec in vals.items():
        for key, value in spec.items():
            th_s, prompt_s = key.split("-")
            append_record(str(out), {
                "schema": RECORD_SCHEMA, "engine": "llamacpp",
                "workload_id": "qwen", "metric": metric, "value": value,
                "unit": "tok/s" if metric == "gen_tok_s" else "ms/token",
                "threads": int(th_s), "precision": "q4", "round": 1,
                "platform": "host", "params": {"prompt_tokens": int(prompt_s)},
                "source_log": str(raw)})
    return str(out)


def test_chart_panels_and_legend(tmp_path):
    recs = read_paths([seed(tmp_path)])
    html = report_mod.render_chart_html(recs)
    for marker in ("Decode", "Prefill", "T4", "T8",
                   "qwen", "溯源明细"):
        assert marker in html
    # x ticks and traceability rows
    assert "512" in html and "1024" in html
    assert "tok/s" in html


def test_chart_prefill_conversion_and_detail(tmp_path):
    recs = read_paths([seed(tmp_path)])
    html = report_mod.render_chart_html(recs)
    # prefill 8-thread 512: mean 3.0 ms/tok -> 333.333 tok/s
    assert "333.333" in html
    assert "共 1 轮" not in html  # single round: no suffix


def test_chart_no_llm_metric_has_empty_state(tmp_path):
    raw = tmp_path / "raw.log"; raw.write_text("raw\n", encoding="utf-8")
    p = tmp_path / "cv.jsonl"
    append_record(str(p), {"schema": RECORD_SCHEMA, "engine": "mnn",
                           "workload_id": "sq", "metric": "latency_ms",
                           "value": 4.5, "unit": "ms", "threads": 4,
                           "round": 1, "platform": "host", "source_log": str(raw)})
    html = report_mod.render_chart_html(read_paths([str(p)]))
    assert "不含可作图指标" in html


def test_chart_cli_writes_file(tmp_path, monkeypatch, capsys):
    from mobench.cli import main as cli_main
    src = seed(tmp_path)
    out = tmp_path / "out"
    assert cli_main(["report", src, "--format", "chart", "--out", str(out)]) == 0
    f = out / "benchmark_chart.html"
    assert f.exists()
    assert "多面板基准图" in f.read_text(encoding="utf-8")


def test_chart_backend_curves_labeled_by_backend(tmp_path):
    raw = tmp_path / "raw.log"; raw.write_text("raw\n", encoding="utf-8")
    out = tmp_path / "b.jsonl"
    for backend, value in [("cpu", 40.0), ("opencl", 55.0)]:
        append_record(str(out), {"schema": RECORD_SCHEMA, "engine": "mnn",
                                 "workload_id": "qwen", "metric": "gen_tok_s",
                                 "value": value, "unit": "tok/s", "threads": 4,
                                 "precision": "q4", "round": 1, "backend": backend,
                                 "params": {"prompt_tokens": 512},
                                 "source_log": str(raw)})
    html = report_mod.render_chart_html(read_paths([str(out)]))
    assert "opencl" in html and "cpu" in html
    assert "T4" not in html  # backend present -> not threads label


def _engine_pair(tmp_path):
    raw = tmp_path / "raw.log"; raw.write_text("raw\n", encoding="utf-8")
    out = tmp_path / "x.jsonl"
    for eng, val in [("llamacpp", 40.0), ("mnn", 55.0)]:
        append_record(str(out), {"schema": RECORD_SCHEMA, "engine": eng,
                                 "workload_id": "q", "metric": "gen_tok_s",
                                 "value": val, "unit": "tok/s", "threads": 4,
                                 "backend": "cpu", "precision": "q4", "round": 1,
                                 "params": {"prompt_tokens": 512},
                                 "source_log": str(raw)})
    return read_paths([str(out)])


def test_chart_by_engine_cross_engine_same_backend(tmp_path):
    html = report_mod.render_chart_html(_engine_pair(tmp_path), by="engine")
    assert "llamacpp" in html and "mnn" in html
    assert "cpu" not in html.replace("llamacpp", "").replace("mnn", "")


def test_chart_by_matrix_engine_backend(tmp_path):
    html = report_mod.render_chart_html(_engine_pair(tmp_path), by="matrix")
    assert "llamacpp:cpu" in html and "mnn:cpu" in html


def test_chart_default_multi_engine_prefixes_backend(tmp_path):
    html = report_mod.render_chart_html(_engine_pair(tmp_path), by="backend")
    assert "llamacpp·cpu" in html and "mnn·cpu" in html


def test_chart_does_not_overwrite_same_backend_multiple_threads(tmp_path):
    raw = tmp_path / "raw.log"
    raw.write_text("raw\n", encoding="utf-8")
    out = tmp_path / "mt.jsonl"
    for th, val in [(4, 30.0), (8, 50.0)]:
        append_record(str(out), {
            "schema": RECORD_SCHEMA, "engine": "llamacpp",
            "workload_id": "q", "metric": "gen_tok_s", "value": val,
            "unit": "tok/s", "threads": th, "backend": "cpu",
            "precision": "q4", "round": 1,
            "params": {"prompt_tokens": 512}, "source_log": str(raw),
        })
    recs = read_paths([str(out)])
    data = report_mod._collect_chart(recs, by="backend")
    assert len(data[0]["series"]) == 2
    assert "cpu·T4" in data[0]["series"] and "cpu·T8" in data[0]["series"]


def test_chart_by_engine_does_not_overwrite_multiple_backends(tmp_path):
    raw = tmp_path / "raw.log"
    raw.write_text("raw\n", encoding="utf-8")
    out = tmp_path / "mb.jsonl"
    for be, val in [("cpu", 30.0), ("opencl", 50.0)]:
        append_record(str(out), {
            "schema": RECORD_SCHEMA, "engine": "mnn",
            "workload_id": "q", "metric": "gen_tok_s", "value": val,
            "unit": "tok/s", "threads": 4, "backend": be,
            "precision": "q4", "round": 1,
            "params": {"prompt_tokens": 512}, "source_log": str(raw),
        })
    recs = read_paths([str(out)])
    data = report_mod._collect_chart(recs, by="engine")
    assert len(data[0]["series"]) == 2
    assert "mnn·cpu·T4" in data[0]["series"] and "mnn·opencl·T4" in data[0]["series"]
