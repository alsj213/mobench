"""Regression tests for QA findings P1–P5 (fix-qa-p1-p5)."""
import json


from mobench import records as recmod
from mobench.adapters.llamacpp import LlamacppAdapter
from mobench.records import RECORD_SCHEMA, append_record, read_paths


def write_recs(tmp_path, values, fname="records.jsonl", params=None, **over):
    raw = tmp_path / "raw.log"
    raw.write_text("raw\n", encoding="utf-8")
    p = tmp_path / fname
    for i, v in enumerate(values, start=1):
        rec = {"schema": RECORD_SCHEMA, "engine": "llamacpp",
               "workload_id": "m", "metric": "gen_tok_s", "value": v,
               "unit": "tok/s", "threads": 4, "precision": "q8_0",
               "round": i, "platform": "host", "source_log": str(raw)}
        if params:
            rec["params"] = params
        rec.update(over)
        append_record(str(p), rec)
    return str(p)


# ---- P1: recursive aggregation ------------------------------------------
def test_read_paths_recurses_nested_run_dirs(tmp_path):
    r1 = tmp_path / "results" / "run_a"
    r2 = tmp_path / "results" / "run_b"
    r1.mkdir(parents=True); r2.mkdir(parents=True)
    write_recs(r1, [10.0, 11.0])
    write_recs(r2, [20.0, 21.0])

    rows = read_paths([str(tmp_path / "results")])
    assert len(rows) == 4  # nested run_*/records.jsonl collected


# ---- P2a/P2b: params provenance + param-dimension grouping ---------------
def test_records_carry_workload_params(tmp_path):
    cfg = {"engine": "llamacpp",
           "build": {"bin_dir": str(tmp_path), "bench_bin": "llama-bench"},
           "workload": {"prompt_tokens": 8, "gen_tokens": 128}}
    (tmp_path / "llama-bench").write_text("x", encoding="utf-8")
    log = tmp_path / "raw.log"
    log.write_text(json.dumps({"results": [{"t_generation": 20.0}]}), encoding="utf-8")
    recs = LlamacppAdapter(cfg).parse(str(log), {"id": "m", "path": "/m.gguf"},
                                      threads=4, round_no=1)
    assert recs[0]["params"] == {"prompt_tokens": 8, "gen_tokens": 128}


def test_group_key_keeps_params_separate_and_legacy_stable(tmp_path):
    a = write_recs(tmp_path, [50.0] * 3, params={"gen_tokens": 32})
    b = write_recs(tmp_path, [40.0] * 3, fname="b.jsonl", params={"gen_tokens": 128})
    c = write_recs(tmp_path, [30.0] * 3, fname="c.jsonl")  # legacy, no params
    recs = read_paths([a, b, c])
    keys = {recmod.group_key(r) for r in recs}
    assert len(keys) == 3  # gen 32 / gen 128 / legacy 三者不合并
    legacy = {k for k in keys if len(k) == 5}
    assert legacy  # 无 params 记录仍是五元组


# ---- P3: run --json pure stdout -----------------------------------------
def test_run_json_stdout_is_pure_json(tmp_path, monkeypatch, capsys):
    from mobench.cli import main as cli_main
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    bench = bin_dir / "llama-bench"
    bench.write_text("#!/usr/bin/env python3\n"
                     "import json\nprint(json.dumps({'results':["
                     "{'t_generation':20.0}]}))\n", encoding="utf-8")
    bench.chmod(0o755)
    model = tmp_path / "m.gguf"; model.write_bytes(b"GGUF" * 4)
    (tmp_path / "mobench.yml").write_text(
        "engine: llamacpp\nplatform: {type: host}\n"
        f"build: {{bin_dir: {bin_dir}, bench_bin: llama-bench}}\n"
        f"workload: {{threads: [4], rounds: 1}}\n"
        "models:\n  - id: m\n" + f"    path: {model}\n    kind: llm\n",
        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert cli_main(["run", "--json"]) == 0
    out = capsys.readouterr().out
    meta = json.loads(out)  # pure JSON on stdout
    assert meta["n_records"] >= 1


# ---- P4: multi-round source provenance ----------------------------------
def test_report_marks_multi_round_source(tmp_path, capsys):
    from mobench import report as report_mod
    recs = write_recs(tmp_path, [10.0, 11.0, 12.0], fname="r.jsonl")
    recs_list = read_paths([recs])
    rows = report_mod.summarize_rows(recs_list)
    md = report_mod.render_markdown(rows, recs_list)
    assert "共 3 轮" in md


# ---- P5: regression no-overlap note -------------------------------------
def test_regression_compare_no_overlap_notes(tmp_path):
    from mobench.analysis import regression as reg
    base = write_recs(tmp_path, [50.0] * 5, fname="base.jsonl",
                      workload_id="base_m")
    cur = write_recs(tmp_path, [50.0] * 5, fname="cur.jsonl",
                     workload_id="other_m")
    baseline = tmp_path / "baseline.json"
    reg.save_baseline(str(baseline), read_paths([base]))
    result = reg.compare(str(baseline), read_paths([cur]))
    assert result["rows"] == []
    assert "no overlapping groups" in (result.get("note") or "").lower()


def test_report_and_stats_split_params_matrix_with_label(tmp_path):
    # P2 display layer: report must keep params groups separate and label them
    from mobench import report as report_mod
    from mobench.analysis import stats as stats_mod
    a = write_recs(tmp_path, [50.0, 51.0, 52.0], fname="a.jsonl",
                   params={"gen_tokens": 32})
    b = write_recs(tmp_path, [40.0, 41.0, 42.0], fname="b2.jsonl",
                   params={"gen_tokens": 128})
    recs = read_paths([a, b])

    rrows = report_mod.summarize_rows(recs)
    assert len(rrows) == 2 and {r["params"].get("gen_tokens") for r in rrows} == {32, 128}
    md = report_mod.render_markdown(rrows, recs)
    assert "gen_tokens=32" in md and "gen_tokens=128" in md

    srows = stats_mod.build_rows(recs)
    assert len(srows) == 2
    smd = stats_mod.render_markdown(srows)
    assert "gen_tokens=32" in smd and "gen_tokens=128" in smd


def test_verify_tracked_files_non_git_excludes_venv_and_results(tmp_path, monkeypatch):
    """非 git 兜底模式下，_tracked_files 必须排除 .venv、results 与 configs/local。"""
    import importlib.util
    import types
    from pathlib import Path

    verify_path = Path(__file__).resolve().parents[1] / "scripts" / "verify.py"
    spec = importlib.util.spec_from_file_location("verify_mod_under_test", verify_path)
    verify_mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(verify_mod)

    for rel in (
        "schema/a.json",
        ".venv/bad.json",
        "results/r.json",
        "configs/local/c.yml",
        "configs/local/bad.json",
    ):
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(verify_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        verify_mod.subprocess,
        "run",
        lambda *a, **kw: types.SimpleNamespace(returncode=128, stdout="", stderr="not a git repo"),
    )

    json_hits = verify_mod._tracked_files("*.json", "**/*.json")
    assert json_hits == ["schema/a.json"]

    all_hits = verify_mod._tracked_files()
    assert all_hits == ["schema/a.json"]
    assert all(
        not p.startswith((".venv/", "results/", "configs/local/"))
        for p in all_hits
    )
