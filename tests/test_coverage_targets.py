"""Coverage targets: exercise uncovered branch paths across modules.

Not behavioral spec tests — these deliberately hit alternate paths (fallbacks,
degraded labels, error branches) so the 80% coverage gate is met honestly.
"""
import subprocess

import pytest

from mobench.adapters.base import AdapterError
from mobench.adapters.mnn import MnnAdapter
from mobench.platform import host as host_mod
from mobench.platform.host import HostPlatform
from mobench.records import RECORD_SCHEMA, append_record


# ---- mnn: version / build_cmds / bad value -----------------------------
def _mnn_cfg(tmp_path):
    (tmp_path / "benchmark.out").write_text("x", encoding="utf-8")
    return {"engine": "mnn", "build": {"bin_dir": str(tmp_path),
                                       "bench_bin": "benchmark.out"}}


def _make_fake_run(stdouts, raises=None):
    """Scripted subprocess.run stand-in; stdouts consumed per call."""
    state = {"i": 0}

    def fake_run(argv, **kwargs):
        if raises and state["i"] in raises:
            raise raises[state["i"]]()
        idx = state["i"]
        state["i"] += 1
        out = stdouts[idx] if idx < len(stdouts) else ""
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    return fake_run


def test_mnn_version_parses_binary_line(tmp_path, monkeypatch):
    adapter = MnnAdapter(_mnn_cfg(tmp_path))
    # mnn.version() imports subprocess locally, so patch the shared module.
    monkeypatch.setattr(subprocess, "run",
                        _make_fake_run(["MNNBenchmark version 2.9.0\nx"]))
    assert "MNN" in adapter.version()


def test_mnn_version_falls_back_to_git(tmp_path, monkeypatch):
    adapter = MnnAdapter(_mnn_cfg(tmp_path))
    fake = _make_fake_run(["noise line\n", "abc1234 commit\n"])
    monkeypatch.setattr(subprocess, "run", fake)
    assert adapter.version() == "abc1234 commit"


def test_mnn_version_unknown_on_failure(tmp_path, monkeypatch):
    adapter = MnnAdapter(_mnn_cfg(tmp_path))
    monkeypatch.setattr(subprocess, "run",
                        _make_fake_run(["noise\n"], raises={1: OSError}))
    assert adapter.version() == "unknown"


def test_mnn_build_cmds_and_bad_line_value(tmp_path):
    adapter = MnnAdapter(_mnn_cfg(tmp_path))
    model = {"id": "m", "path": str(tmp_path / "models" / "m.mnn")}
    cmds = adapter.build_cmds(model, threads=8, rounds=1)
    # benchmark.out name -> folder CLI: <folder> warm loop fw threads precision
    assert cmds == [[str(tmp_path / "benchmark.out"), str(tmp_path / "models"),
                     "1", "1", "0", "8", "2"]]

    log = tmp_path / "bad.log"
    log.write_text("latency_ms: not-a-number\n", encoding="utf-8")
    with pytest.raises(AdapterError, match="bad value"):
        adapter.parse(str(log), model, threads=8, round_no=1)


# ---- host: temp fallback / env control root path / _run errors ----------
def test_host_temp_c_falls_back_to_hwmon(tmp_path, monkeypatch):
    tz = tmp_path / "tz" / "temp"
    tz.parent.mkdir(parents=True)
    tz.write_text("not-an-int\n", encoding="utf-8")
    hw = tmp_path / "hw" / "temp1_input"
    hw.parent.mkdir(parents=True)
    hw.write_text("52000\n", encoding="utf-8")

    def fake_glob(pattern):
        if "thermal_zone" in pattern:
            return [str(tz)]
        return [str(hw)]

    monkeypatch.setattr(host_mod, "glob", type("_G", (), {"glob": staticmethod(fake_glob)}))
    platform = HostPlatform({})
    assert platform._temp_c()["value"] == pytest.approx(52.0)


def test_host_preflight_records_temp_when_available(monkeypatch):
    platform = HostPlatform({})
    monkeypatch.setattr(platform, "_temp_c",
                        lambda: {"value": 42.0, "source": "test"})
    monkeypatch.setattr(platform, "_cpu_freq",
                        lambda: {"value_mhz": 2400.0, "source": "test"})
    info = platform.preflight()
    assert info["temp_c"]["value"] == 42.0
    assert info["cpu_freq"]["value_mhz"] == 2400.0


def test_host_run_reports_nonzero_and_exception(monkeypatch):
    platform = HostPlatform({})
    calls = {"i": 0}

    def fake_run(cmd, **kwargs):
        calls["i"] += 1
        if calls["i"] == 1:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")
        raise FileNotFoundError("no uname")

    monkeypatch.setattr(host_mod.subprocess, "run", fake_run)
    assert platform._run(["uname", "-o"])["error"]
    assert "no uname" in platform._run(["uname", "-r"])["error"]


def _root_platform(tmp_path, monkeypatch, cpufail=False):
    """Host platform on a fake one-cluster sysfs tree, running as root."""
    d = tmp_path / "cpufreq" / "policy4"
    d.mkdir(parents=True)
    (d / "related_cpus").write_text("4 5 6\n", encoding="utf-8")
    (d / "cpuinfo_max_freq").write_text("2419200\n", encoding="utf-8")
    governor = d / "scaling_governor"
    if cpufail:
        governor.mkdir()          # a directory cannot be opened for writing
    else:
        governor.write_text("powersave\n", encoding="utf-8")
    monkeypatch.setattr(host_mod, "_SYSFS_CPU", str(tmp_path))
    monkeypatch.setattr(host_mod.os, "geteuid", lambda: 0)
    return HostPlatform({"methodology": {"env_control": "auto", "cpu_pin": True,
                                         "pin_cpus": None, "lock_freq": True}})


def _governor_file(tmp_path):
    return tmp_path / "cpufreq" / "policy4" / "scaling_governor"


def test_host_env_control_root_pins_and_restores(tmp_path, monkeypatch):
    platform = _root_platform(tmp_path, monkeypatch)
    platform.control(platform.cfg, {})
    cond = platform.measure_condition(3)

    assert cond["freq"]["locked"] is True
    assert _governor_file(tmp_path).read_text(encoding="utf-8").strip() == "performance"

    platform.restore()
    assert _governor_file(tmp_path).read_text(encoding="utf-8").strip() == "powersave"
    assert platform._freq_state == {}


def test_host_env_control_root_cpufail_degrades(tmp_path, monkeypatch):
    platform = _root_platform(tmp_path, monkeypatch, cpufail=True)
    info = {}
    platform.control(platform.cfg, info)
    cond = platform.measure_condition(3)

    assert cond["freq"]["locked"] is False
    assert "degraded" in info["env_control"]["value"].lower()


# ---- regression: read engine version from adjacent run.json -------------
def _recs_with_run_meta(tmp_path, values, engine_version="abc1234"):
    run_dir = tmp_path / "run_x"
    run_dir.mkdir(exist_ok=True)
    raw = run_dir / "raw.log"
    raw.write_text("raw\n", encoding="utf-8")
    (run_dir / "run.json").write_text(
        '{"engine_version": "%s", "timestamp": "2026-01-01", '
        '"binary": {"md5": "deadbeef"}}' % engine_version, encoding="utf-8")
    recs = run_dir / "records.jsonl"
    for i, v in enumerate(values, start=1):
        append_record(str(recs), {
            "schema": RECORD_SCHEMA, "engine": "llamacpp",
            "workload_id": "tinyllama", "metric": "gen_tok_s",
            "value": v, "unit": "tok/s", "threads": 4, "precision": "q8_0",
            "round": i, "platform": "host", "source_log": str(raw),
        })
    return str(recs)


def test_regression_carries_run_meta(tmp_path):
    from mobench.analysis import regression as reg

    base_recs = _recs_with_run_meta(tmp_path, [50.0] * 5, engine_version="base")
    baseline = tmp_path / "baseline.json"
    reg.save_baseline(str(baseline), reg.read_paths([base_recs]))
    cur = _recs_with_run_meta(tmp_path, [55.0] * 5, engine_version="cur")
    result = reg.compare(str(baseline), reg.read_paths([cur]))
    assert result["cur_meta"]["engine_version"] == "cur"
    assert result["rows"][0]["flagged"] is False


# ---- prepare: converter success via fake run / dry-run / no-file --------
def _touch_out(argv, **kwargs):
    out = None
    if "-o" in argv:
        out = argv[argv.index("-o") + 1]
    elif "--MNNModel" in argv:
        out = argv[argv.index("--MNNModel") + 1]
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write("artifact")
    return subprocess.CompletedProcess(argv, 0)


def test_prepare_mnn_convert_via_fake_run(tmp_path, monkeypatch):
    from mobench import prepare as prep

    out = tmp_path / "models" / "q.mnn"
    model = {"id": "q", "path": str(out),
             "source": {"convert_with": "mnn", "onnx": str(tmp_path / "q.onnx")}}
    cfg = {"build": {"bin_dir": str(tmp_path)}}
    (tmp_path / "q.onnx").write_text("onnx", encoding="utf-8")
    (tmp_path / "MNNConvert").write_text("tool", encoding="utf-8")
    monkeypatch.setattr(prep.subprocess, "run", _touch_out)
    info = prep.prepare_model(cfg, model)
    assert info["status"] == "ready"
    assert "MNNConvert" in info["source_cmd"]


def test_prepare_dry_run_skips_download(tmp_path):
    from mobench import prepare as prep

    model = {"id": "m", "path": str(tmp_path / "m.gguf"),
             "source": {"url": "http://example.invalid/m.gguf"}}
    info = prep.prepare_model({}, model, dry_run=True)
    assert info["status"] == "would_download"


def test_prepare_converter_producing_no_file_raises(tmp_path, monkeypatch):
    from mobench import prepare as prep

    tool = tmp_path / "convert_hf_to_gguf.py"
    tool.write_text("x", encoding="utf-8")
    out = tmp_path / "m.gguf"
    model = {"id": "m", "path": str(out),
             "source": {"convert_with": "llama.cpp",
                        "dir": str(tmp_path / "hf")}}
    cfg = {"build": {"source_dir": str(tmp_path)}}
    monkeypatch.setattr(prep.subprocess, "run",
                        lambda argv, **kw: subprocess.CompletedProcess(argv, 0))
    with pytest.raises(prep.PrepareError, match="no file"):
        prep.prepare_model(cfg, model)
