"""Unit tests for the Linux perf (host) profiling branch.

Fixtures are real captures from a WSL host perf run against MNN's x86
llm_bench (fp call graph; ``perf script`` / ``perf stat`` text).
"""
import pathlib
import subprocess

import pytest

from mobench import profiling

FIX = pathlib.Path(__file__).parent / "fixtures" / "perf"


def _load(name):
    return (FIX / name).read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- host stat
def test_parse_perf_stat_strips_user_modifier():
    events = profiling.parse_perf_stat(_load("host_stat.txt"))
    assert events["cpu-cycles"] == 179_954_995_060
    assert events["instructions"] == 336_732_283_103
    assert events["cache-misses"] == 828_228_523
    assert ":u" not in events  # modifiers removed, vocabulary shared with adb


def test_host_stat_metrics():
    events = profiling.parse_perf_stat(_load("host_stat.txt"))
    m = profiling.metrics_from_stat(events)
    assert m["ipc"] == pytest.approx(336_732_283_103 / 179_954_995_060,
                                     rel=1e-4)
    assert m["cache_miss_rate"] == pytest.approx(828_228_523 / 10_727_660_980,
                                                 rel=1e-3)
    assert m["L1_dcache_miss_rate"] == pytest.approx(5_648_595_573 / 65_968_565_553,
                                                     rel=1e-3)


# ---------------------------------------------------------------- host samples
def test_parse_perf_samples_frames_leaf_first():
    samples = list(profiling.parse_perf_samples(_load("host_samples.txt")))
    assert samples, "expected sampled stacks in the fixture"
    # leaf (sampled PC) is the first frame; the decode hot kernel appears
    assert any("MNNGemmInt8AddBiasScale_16x4_Unit" in frames[0]
               for _, frames in samples)
    for _pid, frames in samples:
        assert frames
        assert all(len(f) <= 160 for f in frames)


def test_fold_perf_samples_roots_paths():
    folded = profiling.fold_perf_samples(_load("host_samples.txt"))
    assert folded
    # root-first path: leaf symbol is the last segment
    assert any(p.endswith("MNNGemmInt8AddBiasScale_16x4_Unit")
               for p in folded)
    # every path's root (outermost frame) is a stable placeholder only when
    # fp unwinding stopped at the thread entry; no frame keeps a +0x offset
    assert not any("+0x" in name for p in folded for name in p.split(";"))


def test_top_from_frames_self_shares():
    samples = profiling.parse_perf_samples(_load("host_samples.txt"))
    top = profiling.top_from_frames(samples, limit=10)
    assert top, "expected at least one leaf symbol"
    assert abs(sum(p for p, _ in top)) <= 100.0
    assert all(0 < p <= 100 for p, _ in top)
    assert any("MNNGemmInt8AddBiasScale_16x4_Unit" in name
               for _, name in top[:5])  # decode GEMM dominates the leafs


def test_clean_frame_strips_perf_script_offset_and_dso():
    assert profiling._clean_frame(
        "__cosf_fma+0x4 (/usr/lib/x86_64-linux-gnu/libm.so.6)") == "__cosf_fma"
    assert profiling._clean_frame(
        "_M_invoke+0x3ce (/opt/mnn/build/llm_bench)").startswith("_M_invoke")
    assert profiling._clean_frame(
        "100000002 [unknown] ([unknown])") == "[unknown]"


# ---------------------------------------------------------------- find_perf
def test_find_perf_verifies_candidate(fake_perf_bin, monkeypatch):
    monkeypatch.setattr(profiling, "_find_perf_candidates",
                        lambda: [str(fake_perf_bin)])
    monkeypatch.setattr(profiling.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(
                            a[0], 0, stdout="perf version 5.15"))
    assert profiling.find_perf() == str(fake_perf_bin)


def test_find_perf_rejects_broken_wrapper(fake_perf_bin, monkeypatch):
    # A wrapper that exec-fails (rc != 0 on --version) must be skipped.
    monkeypatch.setattr(profiling, "_find_perf_candidates",
                        lambda: [str(fake_perf_bin), "/does/not/exist"])
    monkeypatch.setattr(profiling.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(
                            a[0], 1, stderr="perf not found for kernel ..."))
    assert profiling.find_perf() == ""


@pytest.fixture
def fake_perf_bin(tmp_path):
    p = tmp_path / "perf"
    p.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    p.chmod(0o755)
    return p
