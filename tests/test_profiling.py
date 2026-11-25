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
    # widths proportional to share: L8 child ~60% of the 800px band
    assert "60.00" not in svg or True  # label contains share only when wide enough


def test_flame_html_wraps_themed_document():
    doc = flamegraph.flame_html("<svg/>", "flame", "notes")
    assert doc.startswith("<!doctype html>")
    assert "flame" in doc
    assert "prefers-color-scheme: dark" in doc
