"""Unit tests for the memory-curve command core (pure parts)."""
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
