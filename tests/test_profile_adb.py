"""Unit tests for adb simpleperf profiling support (mocked device)."""
import types

from mobench.platform.adb import REMOTE_DIR, AdbPlatform


def _platform(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llm_bench").write_text("elf")
    (bin_dir / "libMNN.so").write_text("elf")
    cfg = {
        "engine": "mnn",
        "platform": {"type": "adb", "adb": "adb", "device_id": "dev1"},
        "build": {"bin_dir": str(bin_dir), "bench_bin": "llm_bench"},
        "models": [{"id": "m", "path": str(tmp_path / "config.json")}],
    }
    plat = AdbPlatform(cfg)
    plat._sync_dir = lambda *a, **k: REMOTE_DIR + "/bin"  # no real push
    calls = []

    def fake_exec(args, timeout=None):
        calls.append(list(args))
        return types.SimpleNamespace(returncode=0, stdout="ok-table",
                                     stderr="simpleperf I lost 3% of samples")

    monkeypatch.setattr(plat, "_exec", fake_exec)
    return plat, calls


def test_run_profiled_record_command(tmp_path, monkeypatch):
    plat, calls = _platform(tmp_path, monkeypatch)
    cmd = [str(tmp_path / "bin" / "llm_bench"), "-m",
           str(tmp_path / "config.json")]
    out = plat.run_profiled(cmd, freq=2000, events=["cpu-cycles", "instructions"])
    assert out["record_rc"] == 0
    assert len(calls) == 2                     # record + stat
    rec_shell = calls[0][1]
    assert rec_shell.startswith("cd " + REMOTE_DIR + "/bin && LD_LIBRARY_PATH=. ")
    assert "simpleperf record -f 2000 " in rec_shell
    assert "-g -- ./llm_bench" in rec_shell
    assert "perf.data" in rec_shell
    stat_shell = calls[1][1]
    assert "simpleperf stat -e cpu-cycles,instructions -- ./llm_bench" in stat_shell
    assert out["samples_lost_hint"]  # hint surfaced


def test_deploy_command_remaps_bin_and_ld_library(tmp_path, monkeypatch):
    plat, _calls = _platform(tmp_path, monkeypatch)
    cmd = [str(tmp_path / "bin" / "llm_bench"), "-m", "/no/such/model.json"]
    shell = plat._deploy_command(cmd)
    assert shell.startswith("cd " + REMOTE_DIR + "/bin && LD_LIBRARY_PATH=. ")
    assert shell.endswith("./llm_bench -m /no/such/model.json")
