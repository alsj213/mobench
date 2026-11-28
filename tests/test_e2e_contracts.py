"""端到端闭环、CLI 分派、profiling 流水线与 MCP 真子进程契约测试。"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from mobench import profiling
from mobench.artifacts_hash import artifact_set_md5
from mobench.cli import main as cli_main
from mobench.mcp_server import mcp

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _write_fake_llama_bench(bin_dir: Path) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    bench = bin_dir / "llama-bench"
    bench.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "if '--version' in sys.argv:\n"
        "    print('version: 7200 (fake)')\n"
        "    sys.exit(0)\n"
        "print(json.dumps([\n"
        "  {'n_prompt': 64, 'n_gen': 0, 'avg_ts': 320.0},\n"
        "  {'n_prompt': 0, 'n_gen': 128, 'avg_ts': 45.0}\n"
        "]))\n",
        encoding="utf-8",
    )
    bench.chmod(0o755)
    return bench


def _setup_fake_project(tmp_path: Path, rounds: int = 5) -> tuple[Path, Path, Path]:
    bin_dir = tmp_path / "bin"
    bench = _write_fake_llama_bench(bin_dir)
    (bin_dir / "libggml.so").write_bytes(b"FAKE-SO-1")
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model = models_dir / "tiny.gguf"
    model.write_bytes(b"GGUF" * 32)
    cfg = tmp_path / "mobench.yml"
    cfg.write_text(
        f"engine: llamacpp\n"
        f"platform:\n  type: host\n"
        f"build:\n  bin_dir: {bin_dir}\n  bench_bin: llama-bench\n"
        f"models:\n  - id: tiny\n    path: {model}\n    kind: llm\n    precision: q8_0\n"
        f"workload:\n  threads: [4]\n  rounds: {rounds}\n  prompt_tokens: 64\n  gen_tokens: 128\n"
        f"methodology:\n  env_control: none\n"
        f"results_dir: {tmp_path / 'results'}\n",
        encoding="utf-8",
    )
    return cfg, bench, model


def _run_cli_subproc(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "mobench", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)},
    )


def test_subprocess_golden_e2e_loop(tmp_path):
    cfg, _bench, _model = _setup_fake_project(tmp_path, rounds=5)
    results_dir = tmp_path / "results"

    p_env = _run_cli_subproc(["env", "--config", str(cfg), "--json"], tmp_path)
    assert p_env.returncode == 0, p_env.stderr
    env_data = json.loads(p_env.stdout)
    assert env_data["schema"] == "mobench/env@1"
    assert env_data["binary"]["md5"]
    assert env_data["models"][0]["md5"]

    p_run = _run_cli_subproc(["run", "--config", str(cfg), "--label", "golden", "--json"], tmp_path)
    assert p_run.returncode == 0, p_run.stderr
    run_data = json.loads(p_run.stdout)
    assert run_data["schema"] == "mobench/run@1"
    assert run_data["label"] == "golden"
    assert run_data["n_records"] == 15
    rec_file = Path(run_data["records"])
    assert rec_file.is_file()
    lines = [json.loads(ln) for ln in rec_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 15
    for rec in lines:
        for req in ("schema", "engine", "workload_id", "metric", "value", "unit", "source_log"):
            assert rec.get(req) not in (None, "")

    p_stats = _run_cli_subproc(["stats", str(results_dir), "--json"], tmp_path)
    assert p_stats.returncode == 0, p_stats.stderr
    stats_rows = json.loads(p_stats.stdout)
    assert len(stats_rows) == 3
    assert all(r["n"] == 5 and r["samples_min"] is True for r in stats_rows)

    baseline = tmp_path / "baseline.json"
    p_save = _run_cli_subproc(
        ["regression", "save", "--results", str(rec_file), "--out", str(baseline), "--json"], tmp_path
    )
    assert p_save.returncode == 0, p_save.stderr
    base_data = json.loads(p_save.stdout)
    assert base_data["schema"] == "mobench/baseline@1"

    p_cmp = _run_cli_subproc(
        ["regression", "compare", "--results", str(rec_file), "--baseline", str(baseline), "--json"], tmp_path
    )
    assert p_cmp.returncode == 0, p_cmp.stderr
    cmp_data = json.loads(p_cmp.stdout)
    assert len(cmp_data["rows"]) == 3
    assert all(r["flagged"] is False for r in cmp_data["rows"])

    out_rpt = tmp_path / "rpt"
    p_rpt = _run_cli_subproc(["report", str(rec_file), "--format", "both", "--out", str(out_rpt)], tmp_path)
    assert p_rpt.returncode == 0, p_rpt.stderr
    assert (out_rpt / "benchmark_report.md").is_file()
    assert (out_rpt / "benchmark_report.html").is_file()

    p_chart = _run_cli_subproc(["report", str(rec_file), "--format", "chart", "--out", str(out_rpt)], tmp_path)
    assert p_chart.returncode == 0, p_chart.stderr
    assert (out_rpt / "benchmark_chart.html").is_file()

    p_runs = _run_cli_subproc(["runs", "--results", str(results_dir), "--json"], tmp_path)
    assert p_runs.returncode == 0, p_runs.stderr
    runs_rows = json.loads(p_runs.stdout)
    assert len(runs_rows) == 1
    assert runs_rows[0]["label"] == "golden"


@pytest.mark.parametrize("subcmd", ["profile", "memory", "trace"])
def test_cli_dispatch_profile_memory_trace(subcmd, tmp_path, monkeypatch, capsys):
    assert cli_main([subcmd, "--config", str(tmp_path / "missing.yml")]) == 1
    assert "error" in capsys.readouterr().err

    if subcmd == "profile":
        monkeypatch.setattr(
            profiling,
            "profile_pipeline",
            lambda *a, **kw: {"out_dir": str(tmp_path), "flamegraph": str(tmp_path / "f.html")},
        )
    elif subcmd == "memory":
        from mobench import memory

        monkeypatch.setattr(
            memory,
            "mem_pipeline",
            lambda *a, **kw: {"n_samples": 1, "peak_rss_kb": 1024, "steady_rss_kb": 1024, "kv_growth_kb": 0},
        )
    else:
        from mobench import trace

        monkeypatch.setattr(
            trace,
            "trace_pipeline",
            lambda *a, **kw: {"n_freq_events": 1, "overall_max_mhz": 2000, "trace_html": str(tmp_path / "t.html")},
        )

    cfg, _b, _m = _setup_fake_project(tmp_path, rounds=1)
    assert cli_main([subcmd, "--config", str(cfg), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, dict)


def test_profile_pipeline_host_end_to_end(tmp_path, monkeypatch):
    cfg_path, _bench, _model = _setup_fake_project(tmp_path, rounds=1)
    from mobench.config import load_config

    cfg = load_config(str(cfg_path))
    fake_perf = tmp_path / "fake_perf.py"
    stat_file = FIXTURES / "perf" / "host_stat.txt"
    samples_file = FIXTURES / "perf" / "host_samples.txt"
    fake_perf.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "sub = sys.argv[1]\n"
        "if sub == 'record':\n"
        "    out = sys.argv[sys.argv.index('-o') + 1]\n"
        "    pathlib.Path(out).write_bytes(b'PERFDATA')\n"
        "elif sub == 'stat':\n"
        f"    sys.stdout.write(pathlib.Path({str(stat_file)!r}).read_text(encoding='utf-8'))\n"
        "elif sub == 'script':\n"
        f"    sys.stdout.write(pathlib.Path({str(samples_file)!r}).read_text(encoding='utf-8'))\n",
        encoding="utf-8",
    )
    fake_perf.chmod(0o755)
    monkeypatch.setattr(profiling, "find_perf", lambda: str(fake_perf))

    meta = profiling.profile_pipeline(cfg, json_mode=True)
    out_dir = Path(meta["out_dir"])
    assert (out_dir / "profile.json").is_file()
    assert (out_dir / "stat.txt").is_file()
    assert (out_dir / "samples.txt").is_file()
    assert (out_dir / "top.txt").is_file()
    assert (out_dir / "flamegraph.html").is_file()
    assert meta["events_summary"]["ipc"] is not None
    assert meta["top"]
    assert meta["n_samples_folded"] > 0


def test_profile_pipeline_adb_end_to_end(tmp_path):
    cfg_path, bench, _model = _setup_fake_project(tmp_path, rounds=1)
    from mobench.config import load_config

    cfg = load_config(str(cfg_path))
    cfg["platform"] = {"type": "adb", "adb": "adb", "device_id": "dev1"}

    ndk_dir = tmp_path / "ndk_simpleperf"
    ndk_dir.mkdir()
    samples_head = FIXTURES / "simpleperf" / "samples_head.txt"
    (ndk_dir / "report_sample.py").write_text(
        "import pathlib, sys\n"
        f"sys.stdout.write(pathlib.Path({str(samples_head)!r}).read_text(encoding='utf-8'))\n",
        encoding="utf-8",
    )

    stat_text = (FIXTURES / "simpleperf" / "stat_mnn.txt").read_text(encoding="utf-8")
    symbols_text = (FIXTURES / "simpleperf" / "symbols_mnn.txt").read_text(encoding="utf-8")
    calls = []

    class FakeAdbPlatform:
        name = "adb"

        def preflight(self):
            return {"platform": {"value": "adb"}}

        def measure_condition(self, threads):
            calls.append("measure")
            return {"pin": {"cpus": [4], "mask": "10", "origin": "auto"}, "freq": {"locked": True}}

        def run_profiled(self, cmd, **kwargs):
            calls.append("run_profiled")
            return {
                "record_rc": 0,
                "perf_data": "/data/local/tmp/mobench/perf.data",
                "stat_text": stat_text,
                "samples_lost_hint": "",
            }

        def _exec(self, args, timeout=120):
            if args[0] == "pull":
                Path(args[2]).write_bytes(b"PERF2")
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")
            return types.SimpleNamespace(returncode=0, stdout=symbols_text, stderr="")

        def restore(self):
            calls.append("restore")

        def file_md5(self, path):
            return artifact_set_md5(str(bench))

    meta = profiling.profile_pipeline(cfg, ndk_dir=str(ndk_dir), _platform=FakeAdbPlatform())
    out_dir = Path(meta["out_dir"])
    assert calls == ["measure", "run_profiled", "restore"]
    assert meta["perf_data_pulled"] is True
    assert (out_dir / "flamegraph.html").is_file()
    assert (out_dir / "top.txt").is_file()
    assert (out_dir / "stat.txt").is_file()
    assert (out_dir / "profile.json").is_file()
    assert meta["top"]
    assert meta["events_summary"]["ipc"] is not None


async def test_mcp_real_subprocess_integration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT))
    cfg, bench, _model = _setup_fake_project(tmp_path, rounds=2)
    real_md5 = artifact_set_md5(str(bench))

    good_handoff = tmp_path / "good_handoff.json"
    good_handoff.write_text(json.dumps({"binary_md5": real_md5}), encoding="utf-8")
    bad_handoff = tmp_path / "bad_handoff.json"
    bad_handoff.write_text(json.dumps({"binary_md5": "0" * 32}), encoding="utf-8")

    async with Client(mcp) as c:
        env_res = (await c.call_tool("env", {"config": str(cfg)})).data
        assert env_res["schema"] == "mobench/env@1"
        assert env_res["binary"]["md5"] == real_md5

        with pytest.raises(ToolError, match="I6"):
            await c.call_tool("run", {"config": str(cfg), "expect_handoff": str(bad_handoff)})
        assert not list((tmp_path / "results").glob("run_*"))

        run_res = (await c.call_tool("run", {"config": str(cfg), "expect_handoff": str(good_handoff)})).data
        assert run_res["handoff_check"]["match"] is True
        rec_path = run_res["run"]["records"]

        stats_res = (await c.call_tool("stats", {"inputs": [rec_path]})).data
        assert "results" in stats_res and len(stats_res["results"]) == 3

        def _val_art(text: str) -> dict:
            return {
                "schema": "mobench/validate@1",
                "controls": {
                    "model_md5": "abc",
                    "threads": 4,
                    "sampling": "greedy",
                    "gen_tokens": 16,
                    "prompts_sha256": "p",
                },
                "coverage": {
                    "prefill": {"covered": False, "by": None, "reason": "instrument not configured"},
                    "decode": {"covered": True, "by": "greedy"},
                },
                "sides": {
                    "a": {
                        "identity": {"engine_version": "v1", "params": {}, "binary_md5": {}},
                        "instruments": {
                            "greedy": {"status": "ok", "runs": [{"prompt": "p.txt", "status": "ok", "text": text}]}
                        },
                    }
                },
            }

        va = tmp_path / "val_a.json"
        vb = tmp_path / "val_b.json"
        va.write_text(json.dumps(_val_art("Hello world")), encoding="utf-8")
        vb.write_text(json.dumps(_val_art("Hello_world")), encoding="utf-8")
        cmp_res = (
            await c.call_tool("validate", {"action": "compare", "extra": ["--baseline", str(va), "--candidate", str(vb)]})
        ).data
        assert cmp_res["verdict"] == "FAIL"
        assert cmp_res["exit_code"] == 2
