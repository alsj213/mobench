"""adb platform tests (spec: benchmark-core 平台抽象 host 与 adb)."""
import subprocess

import pytest

from mobench.platform import adb as adb_mod
from mobench.platform.adb import AdbPlatform
from mobench.platform.base import BasePlatform, PlatformError


def make_cfg():
    return {"platform": {"type": "adb", "adb": "/fake/adb", "device_id": "SN123"},
            "engine": "llamacpp"}


def fake_devices_list(stdout_lines):
    class Proc:
        returncode = 0
        stdout = "\n".join(stdout_lines) + "\n"
        stderr = ""

    def fake_run(args, *a, **k):
        return Proc()

    return fake_run


def test_factory_selects_adb_platform():
    # Act
    platform = BasePlatform.make(make_cfg())

    # Assert
    assert isinstance(platform, AdbPlatform)


def test_preflight_raises_when_device_offline(monkeypatch):
    # Arrange
    monkeypatch.setattr(adb_mod.subprocess, "run", fake_devices_list(["List of devices attached", ""]))
    platform = AdbPlatform(make_cfg())

    # Act / Assert
    with pytest.raises(PlatformError, match="not online"):
        platform.preflight()


def test_preflight_ok_when_device_present(monkeypatch):
    # Arrange: device listed -> online branch continues into shell commands.
    calls = {}

    class Proc:
        returncode = 0
        stdout = ""
        stderr = ""

        def __init__(self, stdout=""):
            self.stdout = stdout

    def fake_run(args, *a, **k):
        key = args[1] if len(args) > 1 and args[1] == "devices" else "shell"
        calls[key] = calls.get(key, 0) + 1
        if key == "devices":
            return Proc("List of devices attached\nSN123\tdevice\n")
        return Proc("1\n")

    monkeypatch.setattr(adb_mod.subprocess, "run", fake_run)
    platform = AdbPlatform(make_cfg())

    # Act
    info = platform.preflight()

    # Assert
    assert info["platform"]["value"] == "adb"
    assert info["device_id"]["value"] == "SN123"


def test_run_pushes_and_remaps(tmp_path, monkeypatch):
    # Arrange
    payload = {"pushed": [], "shell_cmds": []}

    class Proc:
        returncode = 0
        stdout = ""
        stderr = ""

        def __init__(self, stdout=""):
            self.stdout = stdout

    def fake_run(args, *a, **k):
        joined = " ".join(args)
        if joined.endswith("devices"):
            return Proc("List of devices attached\nSN123\tdevice\n")
        if " push " in joined:
            payload["pushed"].append(args[-1])
            return Proc("1 file pushed")
        # shell
        payload["shell_cmds"].append(joined)
        return Proc("remote-output\n")

    monkeypatch.setattr(adb_mod.subprocess, "run", fake_run)
    platform = AdbPlatform(make_cfg())
    bin_path = tmp_path / "llama-bench"
    bin_path.write_text("#!/bin/sh\n", encoding="utf-8")
    log = tmp_path / "out.log"

    # Act
    rc = platform.run([str(bin_path), "-m", "x.gguf"], str(log))

    # Assert
    assert rc == 0
    assert len(payload["pushed"]) == 1  # binary pushed
    assert "remote-output" in log.read_text(encoding="utf-8")


def _fake_push_shell(payload):
    class Proc:
        returncode = 0
        stdout = ""
        stderr = ""

        def __init__(self, stdout=""):
            self.stdout = stdout

    def fake_run(args, *a, **k):
        joined = " ".join(args)
        if " push " in joined:
            payload["pushed"].append(args[-1])
            return Proc("1 file pushed")
        payload["shell_cmds"].append(joined)
        return Proc("bench-out\n")

    return fake_run


def test_run_syncs_local_bin_dir_and_runs_there(tmp_path, monkeypatch):
    # Arrange: local android bin dir with the engine binary + dynamic libs
    bin_dir = tmp_path / "android-bin"
    bin_dir.mkdir()
    (bin_dir / "llama-bench").write_text("x", encoding="utf-8")
    (bin_dir / "libllama-common.so").write_text("y", encoding="utf-8")
    model = tmp_path / "q.gguf"
    model.write_bytes(b"GGUF" * 4)
    payload = {"pushed": [], "shell_cmds": []}
    monkeypatch.setattr(adb_mod.subprocess, "run", _fake_push_shell(payload))
    cfg = make_cfg()
    cfg["build"] = {"bin_dir": str(bin_dir), "bench_bin": "llama-bench"}
    platform = AdbPlatform(cfg)
    log = tmp_path / "out.log"

    # Act
    rc = platform.run([str(bin_dir / "llama-bench"), "-m", str(model)], str(log))

    # Assert: whole bin dir + model pushed once; runs from bin dir w/ LD_LIBRARY_PATH
    assert rc == 0
    remote_push = sorted(p for p in payload["pushed"] if "mobench/bin" in p)
    assert [p.rsplit("/", 1)[-1] for p in remote_push] == ["libllama-common.so", "llama-bench"]
    assert any(p.endswith("/q.gguf") and "mobench/" in p for p in payload["pushed"])
    script = payload["shell_cmds"][-1]
    assert "cd /data/local/tmp/mobench/bin && LD_LIBRARY_PATH=. ./llama-bench" in script
    assert "bench-out" in log.read_text(encoding="utf-8")


def test_run_syncs_model_dir_for_mnn_llm(tmp_path, monkeypatch):
    # Arrange: MNN LLM model = directory (config + weights) via models[].dir
    model_dir = tmp_path / "qwen-mnn"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model":"q"}', encoding="utf-8")
    (model_dir / "llm.mnn.weight").write_bytes(b"w" * 16)
    cfg = make_cfg()
    cfg["models"] = [{"id": "qwen", "kind": "llm",
                      "path": str(model_dir / "config.json"),
                      "dir": str(model_dir)}]
    payload = {"pushed": [], "shell_cmds": []}
    monkeypatch.setattr(adb_mod.subprocess, "run", _fake_push_shell(payload))
    platform = AdbPlatform(cfg)
    bin_path = tmp_path / "llm_bench"
    bin_path.write_text("#!/bin/sh\n", encoding="utf-8")
    log = tmp_path / "out.log"

    # Act
    rc = platform.run([str(bin_path), "-m", str(model_dir / "config.json")], str(log))

    # Assert: model dir synced to mobench/models/<name>; -m remapped to its config
    assert rc == 0
    remote_model = [p for p in payload["pushed"] if "mobench/models/qwen-mnn/" in p]
    names = sorted(p.rsplit("/", 1)[-1] for p in remote_model)
    assert names == ["config.json", "llm.mnn.weight"]
    script = payload["shell_cmds"][-1]
    assert ("-m /data/local/tmp/mobench/models/qwen-mnn/config.json"
            in script)
    assert "cd /data/local/tmp/mobench &&" in script


def test_run_maps_model_folder_arg(tmp_path, monkeypatch):
    # Arrange: CV benchmark.out passes a models folder (models[].dir)
    model_dir = tmp_path / "cv-models"
    model_dir.mkdir()
    (model_dir / "squeezenet.mnn").write_bytes(b"mnn" * 8)
    cfg = make_cfg()
    cfg["models"] = [{"id": "sq", "kind": "cv",
                      "path": str(model_dir / "squeezenet.mnn"),
                      "dir": str(model_dir)}]
    payload = {"pushed": [], "shell_cmds": []}
    monkeypatch.setattr(adb_mod.subprocess, "run", _fake_push_shell(payload))
    platform = AdbPlatform(cfg)
    bin_path = tmp_path / "benchmark.out"
    bin_path.write_text("#!/bin/sh\n", encoding="utf-8")
    log = tmp_path / "o.log"

    # Act
    rc = platform.run([str(bin_path), str(model_dir), "1", "1", "0", "8", "2"], str(log))

    # Assert: folder arg remapped to synced remote dir; model pushed
    assert rc == 0
    assert any(p.endswith("/squeezenet.mnn") and "models/cv-models" in p
               for p in payload["pushed"])
    script = payload["shell_cmds"][-1]
    assert "/data/local/tmp/mobench/models/cv-models" in script


# -- push timeout scales with payload ---------------------------------------
# 固定 120 s 只够推约 2.3 GB（实测 19.5 MB/s），4B q4_k_m 经 gguf2mnn 重建后
# 单个 llm.mnn.weight 就有 3.2 GB —— 会把 push 打断在半路。

def _sparse_file(path, size):
    with open(path, "wb") as fh:
        fh.truncate(size)
    return path


def test_push_timeout_keeps_default_for_small_files(tmp_path):
    # Arrange
    small = tmp_path / "small.bin"
    small.write_bytes(b"x" * 1024)

    # Act / Assert
    assert adb_mod.push_timeout(str(small)) == adb_mod.PUSH_MIN_TIMEOUT_S


def test_push_timeout_scales_with_model_size(tmp_path):
    # Arrange: 3.2 GB 稀疏文件，等价 4B q4k-m 的 llm.mnn.weight
    big = _sparse_file(tmp_path / "llm.mnn.weight", 3_200_000_000)

    # Act
    timeout = adb_mod.push_timeout(str(big))

    # Assert: 必须比 120 s 大，且不低于按保守速率估出的下界
    assert timeout > adb_mod.PUSH_MIN_TIMEOUT_S
    assert timeout >= 3_200_000_000 / adb_mod.PUSH_ASSUMED_BYTES_PER_S


def test_push_timeout_falls_back_when_path_missing(tmp_path):
    # Arrange / Act / Assert: 取不到体积时退回默认，不抛异常
    assert adb_mod.push_timeout(str(tmp_path / "nope")) == adb_mod.PUSH_MIN_TIMEOUT_S


def test_push_file_passes_scaled_timeout(tmp_path, monkeypatch):
    # Arrange
    seen = {}

    class Proc:
        returncode = 0
        stdout = "1 file pushed"
        stderr = ""

    def fake_run(args, *a, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return Proc()

    monkeypatch.setattr(adb_mod.subprocess, "run", fake_run)
    platform = AdbPlatform(make_cfg())
    big = _sparse_file(tmp_path / "llm.mnn.weight", 3_200_000_000)

    # Act
    platform._push_file(str(big), "/data/local/tmp/mobench/models/x/llm.mnn.weight")

    # Assert
    assert seen["timeout"] > adb_mod.PUSH_MIN_TIMEOUT_S


# -- thermal / freq sampling (add-thermal-band-monitor) ---------------------


def test_temp_c_takes_hottest_zone(monkeypatch):
    """2026-10-09 教训：取第一个 zone 落在不敏感区（aoss 36°C vs CPU 60°C）。"""
    # Arrange
    platform = AdbPlatform(make_cfg())

    def fake_shell(cmd, timeout=120):
        if cmd[1].endswith("/type"):
            return 0, "aoss0-usr\ncpu-1-7-usr\n"
        return 0, "36000\n60100\n"

    monkeypatch.setattr(platform, "_shell", fake_shell)

    # Act
    info = platform._temp_c()

    # Assert
    assert info["value"] == 60.1
    assert info["zone"] == "cpu-1-7-usr"
    assert "max" in info["source"]


def test_temp_c_none_when_unreadable(monkeypatch):
    platform = AdbPlatform(make_cfg())
    monkeypatch.setattr(platform, "_shell", lambda cmd, timeout=120: (1, ""))
    assert platform._temp_c()["value"] is None


def test_cpu_freq_reports_max_mhz(monkeypatch):
    platform = AdbPlatform(make_cfg())
    monkeypatch.setattr(platform, "_shell",
                        lambda cmd, timeout=120: (0, "1804800\n2419200\n1804800\n"))
    assert platform._cpu_freq()["value_mhz"] == 2419.2


def test_sample_thermal_combines_temp_and_freq(monkeypatch):
    platform = AdbPlatform(make_cfg())

    def fake_shell(cmd, timeout=120):
        if "thermal_zone" in cmd[1]:
            return 0, "40000\n"
        return 0, "2000000\n"

    monkeypatch.setattr(platform, "_shell", fake_shell)
    sample = platform.sample_thermal()
    assert sample["temp_c"]["value"] == 40.0
    assert sample["cpu_freq"]["value_mhz"] == 2000.0
