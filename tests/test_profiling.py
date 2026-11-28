"""Unit tests for simpleperf output parsers and flamegraph rendering."""
import collections
import pathlib

import pytest

from mobench import flamegraph, profiling

FIX = pathlib.Path(__file__).parent / "fixtures" / "simpleperf"


def _load(name):
    return (FIX / name).read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- stat
def test_parse_stat_counts_events():
    events = profiling.parse_stat(_load("stat_mnn.txt"))
    assert events["cpu-cycles"] == 67_436_851_772
    assert events["instructions"] == 67_273_342_838
    assert events["cache-misses"] == 793_016_830


def test_metrics_ipc_and_cache_rate():
    events = profiling.parse_stat(_load("stat_mnn.txt"))
    m = profiling.metrics_from_stat(events)
    assert m["ipc"] == pytest.approx(67_273_342_838 / 67_436_851_772, rel=1e-3)
    assert m["cache_miss_rate"] == pytest.approx(793_016_830 / 21_530_788_107, rel=1e-3)
    assert m["L1_dcache_miss_rate"] is None  # no L1-loads counter captured


# ---------------------------------------------------------------- samples
def test_parse_samples_yields_real_stacks():
    samples = list(profiling.parse_samples(_load("samples_head.txt")))
    assert samples, "expected at least one sampled stack"
    pid, count, frames = samples[0]
    assert isinstance(pid, int)
    assert count >= 1
    assert frames, "sample must carry frames"
    # kernel frames collapse to a stable label; user frames keep names
    assert all(len(f) <= 160 for f in frames)


def test_fold_samples_counts_paths():
    text = ("app 100 [0] 1.000000:\t\t1 cpu-cycles:\n"
            "  1000000  leaf ()  \n"
            "  1000001  mid (libx.so)\n"
            "  1000002  root (libx.so)\n"
            "app 100 [0] 1.001000:\t\t1 cpu-cycles:\n"
            "  2000000  other (libx.so)\n"
            "  1000002  root (libx.so)\n")
    folded = profiling.fold_samples(text)
    # frames are leaf-first; fold roots at the last frame (process entry)
    assert folded["root;mid;leaf"] == 1
    assert folded["root;other"] == 1


# ---------------------------------------------------------------- flat top
def test_top_symbols_parses_real_report():
    top = profiling.top_symbols(_load("symbols_mnn.txt"), limit=10)
    assert top[0][0] == 26.34          # kernel collapsed label
    assert top[0][1] == "[kernel]"
    names = [n for _, n in top]
    assert any("L8LoopSz_TILE_1" in n for n in names)


# ---------------------------------------------------------------- flamegraph
def test_build_tree_totals():
    folded = collections.Counter({
        "main;run;L8": 50,
        "main;run;L8;unroll": 20,
        "main;load;Read": 30,
    })
    forest = flamegraph.build_tree(folded)
    assert len(forest) == 1 and forest[0].name == "main"
    assert forest[0].total() == 100


def test_render_flame_svg_contains_rects():
    folded = collections.Counter({"main;run;L8": 60, "main;load;Read": 40})
    svg = flamegraph.render_flame_svg(flamegraph.build_tree(folded), width=800)
    assert svg.startswith("<svg")
    assert "<rect" in svg
    # 标题元素固定携带各节点真实百分比，不受矩形宽度裁剪影响
    assert "L8 60.00%" in svg
    assert "Read 40.00%" in svg


def test_flame_html_wraps_themed_document():
    doc = flamegraph.flame_html("<svg/>", "flame", "notes")
    assert doc.startswith("<!doctype html>")
    assert "flame" in doc
    assert "prefers-color-scheme: dark" in doc


def test_flame_html_escapes_notes_and_title():
    doc = flamegraph.flame_html("<svg/>", "<script>t</script>", "<img src=x onerror=alert(1)>")
    assert "<script>" not in doc
    assert "<img " not in doc
    assert "&lt;script&gt;t&lt;/script&gt;" in doc
    assert "&lt;img src=x onerror=alert(1)&gt;" in doc


# -- Wave 5 加固：symfs 镜像必须与 adb bin tag 一致 ------------------------

def test_mirror_bin_dir_uses_platform_bin_tag(tmp_path):
    import os

    from mobench.platform.adb import REMOTE_DIR, AdbPlatform

    bin_dir = tmp_path / "android-bin"
    bin_dir.mkdir()
    (bin_dir / "llm_bench").write_text("elf", encoding="utf-8")
    (bin_dir / "libMNN.so").write_text("elf", encoding="utf-8")

    target = profiling.mirror_bin_dir(str(bin_dir), str(tmp_path / "symfs"))

    tag = AdbPlatform._bin_dir_tag(str(bin_dir))
    expected = os.path.join(
        str(tmp_path / "symfs"), REMOTE_DIR.lstrip("/"), *tag.split("/"))
    assert target == expected
    assert os.path.isfile(os.path.join(target, "llm_bench"))
    assert os.path.isfile(os.path.join(target, "libMNN.so"))

    # 空 bin_dir 保持历史路径，不再抛 FileNotFoundError
    empty_target = profiling.mirror_bin_dir("", str(tmp_path / "symfs_empty"))
    assert empty_target == os.path.join(
        str(tmp_path / "symfs_empty"), "data/local/tmp/mobench/bin")
    assert os.path.isdir(empty_target)


def test_capture_adb_quotes_perf_data_path(tmp_path):
    """simpleperf report 的 perf.data 路径必须 shell quote。"""
    import types

    class FakeAdb:
        def run_profiled(self, cmd, freq=2000, events=None, mmap_pages=None):
            return {"record_rc": 0, "perf_data": "/data/local/tmp/a b/perf.data",
                    "stat_text": "", "samples_lost_hint": None}

        def __init__(self):
            self.calls = []

        def _exec(self, args, timeout=120):
            self.calls.append(args)
            return types.SimpleNamespace(returncode=0, stdout="top", stderr="")

    fake = FakeAdb()

    profiling._capture_adb(fake, ["./llm_bench"], 2000, [], str(tmp_path / "perf.data"))

    report_calls = [c for c in fake.calls if "simpleperf report" in c[1]]
    assert report_calls
    assert "'/data/local/tmp/a b/perf.data'" in report_calls[0][1]
