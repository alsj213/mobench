"""Unit tests for atrace CPU-frequency parsing / summary and pipeline."""
import csv
import json
import types

import pytest

from mobench import trace

SAMPLE = """# tracer: nop
          atrace-1 (-----) [000] .... 100.000000: tracing_mark_write: x
   llm_bench-5 (-----) [003] .... 100.100000: cpu_frequency: state=2112000 cpu_id=3
   llm_bench-5 (-----) [003] .... 100.250000: cpu_frequency: cpu_id=3 state=1800000
   llm_bench-5 (-----) [001] .... 100.300000: cpu_frequency: state=960000 cpu_id=1
"""


def test_parse_freq_events_both_field_orders():
    evs = list(trace.parse_freq_events(SAMPLE))
    assert len(evs) == 3
    (t0, cpu0, hz0), (t1, cpu1, hz1), (t2, cpu2, hz2) = evs
    assert (round(t0, 3), cpu0, hz0) == (0.0, 3, 2112000)
    assert (cpu1, hz1) == (3, 1800000)      # cpu_id-first order
    assert (cpu2, hz2) == (1, 960000)
    assert t2 > t1 > t0


def test_summarize_low_share_and_mean():
    per = {0: [(0, 2000000), (1, 1800000), (2, 500000), (3, 600000)]}
    s = trace.summarize_freq(per)
    assert s["cpu"][0]["mean_mhz"] == (2_000_000 + 1_800_000 + 500_000 + 600_000) // 4 // 1000
    assert s["cpu"][0]["low_freq_share"] == 0.5   # 2 of 4 below 60% of max
    assert s["overall_max_mhz"] == 2000


def test_trace_html_escapes_metadata():
    doc = trace._html(
        {"engine": "<script>e</script>", "model": "<b>m</b>", "threads": "4<img>", "n_freq_events": 2},
        {0: [(0.0, 1000000), (1.0, 2000000)]},
        "/tmp",
    )
    assert "<script>" not in doc
    assert "<b>m</b>" not in doc
    assert "<img>" not in doc
    assert "&lt;script&gt;e&lt;/script&gt;" in doc


def test_trace_pipeline_rejects_non_adb_platform():
    class HostLike:
        pass

    with pytest.raises(RuntimeError, match="仅支持 adb"):
        trace.trace_pipeline({"engine": "llamacpp", "models": [{"id": "m", "path": "m.gguf"}]},
                             _platform=HostLike())


def test_trace_pipeline_writes_artifacts_and_cli_uses_space_cats(tmp_path, monkeypatch):
    seen_cats = []

    class FakeProc:
        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    class FakePlatform:
        def preflight(self):
            return {"platform": {"value": "adb"}}

        def spawn_trace(self, dur, cats="freq"):
            seen_cats.append(cats)
            return FakeProc()

        def spawn_engine(self, cmd):
            return FakeProc()

        def read_remote_file(self, remote):
            return SAMPLE

        def file_md5(self, path):
            return "md5abc"

    class FakeAdapter:
        def version(self):
            return "v1"

        def resolve_binary(self):
            return "/bin/llama-bench"

        def build_cmds(self, model, threads, rounds=1):
            return [["/bin/llama-bench", "-m", model["path"], "-t", str(threads)]]

    monkeypatch.setattr(trace.time, "sleep", lambda _s: None)
    cfg = {
        "engine": "llamacpp",
        "models": [{"id": "qwen", "path": str(tmp_path / "m.gguf")}],
        "workload": {"threads": [4], "rounds": 1},
        "results_dir": str(tmp_path / "results"),
    }
    meta = trace.trace_pipeline(cfg, dur=1.0, cats="freq sched",
                                _platform=FakePlatform(), _adapter=FakeAdapter())
    assert seen_cats == ["freq sched"]
    assert meta["n_freq_events"] == 3
    assert (tmp_path / "results").exists()
    out_dir = meta["source_log"].rsplit("/", 1)[0]
    assert json.loads(open(out_dir + "/trace.json", encoding="utf-8").read())["overall_max_mhz"] == 2112
    rows = list(csv.reader(open(out_dir + "/freq.csv", encoding="utf-8")))
    assert len(rows) == 4  # header + 3 events

    # CLI passes space-separated categories (not commas)
    called_cats = []
    monkeypatch.setattr(trace, "load_config", lambda _p: cfg)
    monkeypatch.setattr(trace, "trace_pipeline", lambda _c, **kw: called_cats.append(kw["cats"]) or meta)
    rc = trace.main(types.SimpleNamespace(
        config="mobench.yml", model=None, threads=None, dur=2.0,
        atrace_cats=["freq", "sched"], prompt=None, gen=None, json=True,
    ))
    assert rc == 0
    assert called_cats == ["freq sched"]
