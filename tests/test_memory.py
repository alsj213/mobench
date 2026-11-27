"""Unit tests for the memory-curve command core."""
import csv
import json
import os

import pytest

from mobench import memory

PROC_STATUS = """Name:\tllm_bench
VmRSS:\t  620420 kB
VmHWM:\t  640512 kB
RssAnon:\t 500000 kB
"""


def test_parse_proc_status_extracts_kb():
    d = memory.parse_proc_status(PROC_STATUS)
    assert d["VmRSS"] == 620420
    assert d["VmHWM"] == 640512
    assert d["RssAnon"] == 500000


def test_parse_ignores_noisy_lines():
    d = memory.parse_proc_status("Name: x\nsome junk without colon\nVmRSS: 12 kB\n")
    assert d["VmRSS"] == 12


def test_curve_html_renders_line_and_label():
    samples = [(0.0, 100), (1.0, 300), (2.0, 320)]
    svg = memory.rss_curve_html(samples, "VmRSS · mnn")
    assert svg.startswith("<svg")
    assert "polyline" in svg
    assert "VmRSS · mnn" in svg


def test_mem_html_kpi_in_mb():
    doc = memory._mem_html("<svg/>", "/tmp/x", peak=1024 * 100,
                           steady=1024 * 90, engine="mnn", model="q4k")
    assert "100.0 MB" in doc
    assert "90.0 MB" in doc


def test_mem_pipeline_rejects_non_adb_platform():
    class HostLike:
        pass

    with pytest.raises(RuntimeError, match="仅支持 adb"):
        memory.mem_pipeline({"engine": "llamacpp", "models": [{"id": "m", "path": "m.gguf"}]},
                            _platform=HostLike())


def test_mem_pipeline_collects_samples_detects_leak_and_restores(tmp_path, monkeypatch):
    bench = tmp_path / "llama-bench"
    bench.write_text("#!/bin/sh\n", encoding="utf-8")
    # 40 个样本确保 85% 分位后的 tail 至少有 6 个样本 (>=5) 且 kv_growth > 10MB
    rss_series = [100_000 + i * 1_000 for i in range(40)]
    state = {"idx": 0, "restored": False}

    class FakeProc:
        def poll(self):
            return 0 if state["idx"] >= len(rss_series) else None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    class FakePlatform:
        def preflight(self):
            return {"platform": {"value": "adb"}}

        def measure_condition(self, threads):
            return {"pin": {"cpus": [4, 5, 6, 7], "mask": "f0", "origin": "auto"},
                    "freq": {"locked": True}}

        def spawn_engine(self, cmd):
            return FakeProc()

        def pidof(self, name):
            return "4242" if state["idx"] < len(rss_series) else ""

        def read_proc_status(self, pid):
            val = rss_series[state["idx"]]
            state["idx"] += 1
            return f"Name:\tllama-bench\nVmRSS:\t{val} kB\n"

        def file_md5(self, path):
            return "deadbeef"

        def restore(self):
            state["restored"] = True

    monkeypatch.setattr(memory.time, "sleep", lambda _s: None)
    cfg = {
        "engine": "llamacpp",
        "build": {"bin_dir": str(tmp_path), "bench_bin": "llama-bench"},
        "models": [{"id": "qwen", "path": str(tmp_path / "m.gguf")}],
        "workload": {"threads": [4], "rounds": 1},
        "results_dir": str(tmp_path / "results"),
    }
    meta = memory.mem_pipeline(cfg, _platform=FakePlatform())
    assert state["restored"] is True
    assert meta["n_samples"] == len(rss_series)
    assert meta["peak_rss_kb"] == rss_series[-1]
    assert meta["possible_leak"] is True
    assert os.path.isfile(meta["curve_html"])
    assert os.path.isfile(meta["source_log"])
    mem_json = json.loads(open(os.path.join(meta["out_dir"], "mem.json"), encoding="utf-8").read())
    assert mem_json["schema"] == "mobench/mem@1"
    csv_rows = list(csv.reader(open(os.path.join(meta["out_dir"], "mem.csv"), encoding="utf-8")))
    assert len(csv_rows) == len(rss_series) + 1
