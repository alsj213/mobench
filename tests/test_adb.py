"""adb platform tests (spec: benchmark-core 平台抽象 host 与 adb)."""

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
    bin_tag = AdbPlatform._bin_dir_tag(str(bin_dir))
    remote_push = sorted(p for p in payload["pushed"] if f"mobench/{bin_tag}" in p)
    assert [p.rsplit("/", 1)[-1] for p in remote_push] == ["libllama-common.so", "llama-bench"]
    assert any(p.endswith("/q.gguf") and "mobench/" in p for p in payload["pushed"])
    script = payload["shell_cmds"][-1]
    assert f"cd /data/local/tmp/mobench/{bin_tag} && LD_LIBRARY_PATH=. ./llama-bench" in script
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

    # Assert: model dir synced to mobench/models/<name>_<hash>; -m remapped to its config
    assert rc == 0
    expected_tag = AdbPlatform._model_dir_tag(str(model_dir))
    assert "models/qwen-mnn_" in expected_tag
    remote_model = [p for p in payload["pushed"] if f"mobench/{expected_tag}/" in p]
    names = sorted(p.rsplit("/", 1)[-1] for p in remote_model)
    assert names == ["config.json", "llm.mnn.weight"]
    script = payload["shell_cmds"][-1]
    assert f"-m /data/local/tmp/mobench/{expected_tag}/config.json" in script
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
    expected_tag = AdbPlatform._model_dir_tag(str(model_dir))
    assert "models/cv-models_" in expected_tag
    assert any(p.endswith("/squeezenet.mnn") and expected_tag in p
               for p in payload["pushed"])
    script = payload["shell_cmds"][-1]
    assert f"/data/local/tmp/mobench/{expected_tag}" in script


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
    """单 shell 按行配对读取 zone|temp，取最热区。"""
    # Arrange
    platform = AdbPlatform(make_cfg())

    def fake_shell(cmd, timeout=120):
        return 0, "aoss0-usr|36000\ncpu-1-7-usr|60100\n"

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
        if "thermal_zone" in cmd[0]:
            return 0, "cpu-0|40000\n"
        return 0, "2000000\n"

    monkeypatch.setattr(platform, "_shell", fake_shell)
    sample = platform.sample_thermal()
    assert sample["temp_c"]["value"] == 40.0
    assert sample["cpu_freq"]["value_mhz"] == 2000.0


def test_deploy_parts_quotes_spaces_and_metacharacters(tmp_path, monkeypatch):
    # Arrange: paths and cli_args containing spaces / shell metacharacters
    bin_dir = tmp_path / "android bin"
    bin_dir.mkdir()
    bench = bin_dir / "llama-bench"
    bench.write_text("x", encoding="utf-8")
    model = tmp_path / "my model;id.gguf"
    model.write_bytes(b"GGUF")
    cfg = make_cfg()
    cfg["build"] = {"bin_dir": str(bin_dir), "bench_bin": "llama-bench"}
    plat = AdbPlatform(cfg)
    monkeypatch.setattr(plat, "_sync_dir", lambda d, tag, chmod_exec=False, **kw: "/data/local/tmp/mobench/" + tag)
    monkeypatch.setattr(plat, "_push", lambda p: "/data/local/tmp/mobench/my model;id.gguf")

    # Act
    shell_cmd = plat._deploy_command([str(bench), "-m", str(model), "--note", "a b;rm -rf /"])

    # Assert: arguments with spaces/metacharacters are shell-quoted
    assert "'/data/local/tmp/mobench/my model;id.gguf'" in shell_cmd
    assert "'a b;rm -rf /'" in shell_cmd


def test_spawn_trace_formats_space_separated_categories_and_rejects_injection(monkeypatch):
    plat = AdbPlatform(make_cfg())
    popens = []

    def fake_popen(cmd, **kwargs):
        popens.append(cmd)
        return object()

    monkeypatch.setattr(adb_mod.subprocess, "Popen", fake_popen)

    # Comma or space-separated categories become space-separated args for atrace
    plat.spawn_trace(10, cats="freq,sched", out_remote="/data/local/tmp/mobench/a trace.txt")
    shell_str = popens[-1][-1]
    assert "atrace -b 16384 -t 10 freq sched > '/data/local/tmp/mobench/a trace.txt' 2>&1" == shell_str

    plat.spawn_trace(5, cats=["freq", "idle"])
    assert "atrace -b 16384 -t 5 freq idle > /data/local/tmp/mobench/atrace.txt 2>&1" == popens[-1][-1]

    # Injection attempt rejected
    with pytest.raises(PlatformError, match="invalid atrace"):
        plat.spawn_trace(5, cats="freq;rm -rf /")


def test_run_profiled_rejects_invalid_perf_event(tmp_path, monkeypatch):
    plat = AdbPlatform(make_cfg())
    monkeypatch.setattr(plat, "_deploy_parts", lambda cmd: ("cd /data/local/tmp/mobench && ", "./llm_bench"))

    class Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(plat, "_exec", lambda args, timeout=120: Proc())

    with pytest.raises(PlatformError, match="invalid perf event"):
        plat.run_profiled(["./llm_bench"], events=["cpu-cycles;id"])


def test_remote_helpers_quote_arguments(monkeypatch):
    plat = AdbPlatform(make_cfg())
    seen = []

    def fake_shell(cmd, timeout=120):
        seen.append(cmd)
        return 0, "123\n"

    monkeypatch.setattr(plat, "_shell", fake_shell)
    assert plat.pidof("my bench;id") == "123"
    assert plat.read_proc_status("123;id") == "123\n"
    assert plat.read_remote_file("/data/local/tmp/a b.txt") == "123\n"
    assert seen[0] == ["pidof 'my bench;id'"]
    assert seen[1] == ["cat '/proc/123;id/status'"]
    assert seen[2] == ["cat '/data/local/tmp/a b.txt'"]


def test_preflight_rejects_device_substring_when_offline(monkeypatch):
    cfg = {"platform": {"type": "adb", "adb": "/fake/adb", "device_id": "my-device-1"},
           "engine": "llamacpp"}
    for status in ("offline", "unauthorized", "recovery"):
        monkeypatch.setattr(
            adb_mod.subprocess,
            "run",
            fake_devices_list(["List of devices attached", f"my-device-1\t{status}"]),
        )
        platform = AdbPlatform(cfg)
        with pytest.raises(PlatformError, match="not online"):
            platform.preflight()


def test_temp_c_single_shell_handles_missing_type_without_misalignment(monkeypatch):
    platform = AdbPlatform(make_cfg())
    monkeypatch.setattr(
        platform,
        "_shell",
        lambda cmd, timeout=120: (0, "|36000\ncpu-1-7-usr|61500\nbad|N/A\n|55000\n"),
    )
    info = platform._temp_c()
    assert info["value"] == 61.5
    assert info["zone"] == "cpu-1-7-usr"

    monkeypatch.setattr(platform, "_shell", lambda cmd, timeout=120: (0, "|55000\n"))
    info2 = platform._temp_c()
    assert info2["value"] == 55.0
    assert info2["zone"] is None


def test_same_basename_model_dirs_do_not_collide(tmp_path, monkeypatch):
    dir_a = tmp_path / "dirA" / "qwen"
    dir_b = tmp_path / "dirB" / "qwen"
    for d, content in ((dir_a, b"A"), (dir_b, b"B")):
        d.mkdir(parents=True)
        (d / "config.json").write_bytes(content)
    cfg = make_cfg()
    cfg["models"] = [
        {"id": "qa", "path": str(dir_a / "config.json"), "dir": str(dir_a)},
        {"id": "qb", "path": str(dir_b / "config.json"), "dir": str(dir_b)},
    ]
    payload = {"pushed": [], "shell_cmds": []}
    monkeypatch.setattr(adb_mod.subprocess, "run", _fake_push_shell(payload))
    platform = AdbPlatform(cfg)
    remote_a = platform._map_arg(str(dir_a / "config.json"))
    remote_b = platform._map_arg(str(dir_b / "config.json"))
    assert remote_a != remote_b
    assert remote_a.endswith("/config.json") and remote_b.endswith("/config.json")


def test_cross_instance_skips_push_when_remote_md5_matches(tmp_path, monkeypatch):
    from mobench.io_utils import file_md5

    model = tmp_path / "m.gguf"
    model.write_bytes(b"MODEL-CONTENT")
    model_md5 = file_md5(model)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    bench = bin_dir / "llama-bench"
    bench.write_text("#!/bin/sh\n", encoding="utf-8")
    bench_md5 = file_md5(bench)

    cfg = make_cfg()
    cfg["build"] = {"bin_dir": str(bin_dir), "bench_bin": "llama-bench"}
    pushed = []

    class Proc:
        def __init__(self, rc=0, stdout=""):
            self.returncode = rc
            self.stdout = stdout
            self.stderr = ""

    def fake_run(args, *a, **kw):
        if "push" in args:
            pushed.append(args[-1])
            return Proc(0, "pushed")
        shell_str = " ".join(args[4:]) if len(args) > 4 else ""
        if shell_str.startswith("md5sum "):
            if "m.gguf" in shell_str:
                return Proc(0, f"{model_md5}  remote\n")
            if "llama-bench" in shell_str:
                return Proc(0, f"{bench_md5}  remote\n")
        return Proc(0, "")

    monkeypatch.setattr(adb_mod.subprocess, "run", fake_run)
    p2 = AdbPlatform(cfg)
    p2.run([str(bench), "-m", str(model)], str(tmp_path / "r.log"))
    assert pushed == []


def test_sync_dir_bin_cleans_stale_remote_files_and_keeps_current(tmp_path, monkeypatch):
    from mobench.io_utils import file_md5

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    bench = bin_dir / "llama-bench"
    bench.write_text("x", encoding="utf-8")
    bench_md5 = file_md5(bench)

    shell_cmds = []

    class Proc:
        def __init__(self, rc=0, stdout=""):
            self.returncode = rc
            self.stdout = stdout
            self.stderr = ""

    def fake_run(args, *a, **kw):
        shell_str = " ".join(args[4:]) if len(args) > 4 else ""
        shell_cmds.append(shell_str)
        if shell_str.startswith("ls -1 "):
            return Proc(0, "llama-bench\nstale.so\nold-bin\n")
        if shell_str.startswith("md5sum ") and "llama-bench" in shell_str:
            return Proc(0, f"{bench_md5}  remote\n")
        return Proc(0, "")

    monkeypatch.setattr(adb_mod.subprocess, "run", fake_run)
    plat = AdbPlatform(make_cfg())
    plat._sync_dir(str(bin_dir), "bin", chmod_exec=True, clean_stale=True)

    rm_cmds = [c for c in shell_cmds if c.startswith("rm -f ")]
    assert len(rm_cmds) == 1
    assert "/data/local/tmp/mobench/bin/stale.so" in rm_cmds[0]
    assert "/data/local/tmp/mobench/bin/old-bin" in rm_cmds[0]
    assert "llama-bench" not in rm_cmds[0]


def test_adb_run_applies_configured_timeout(tmp_path, monkeypatch):
    cfg = make_cfg()
    cfg["methodology"] = {"timeout_s": 45}
    plat = AdbPlatform(cfg)
    monkeypatch.setattr(plat, "_deploy_command", lambda cmd: "echo ok")
    seen = {}

    class Proc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def fake_exec(args, timeout=120):
        seen["timeout"] = timeout
        return Proc()

    monkeypatch.setattr(plat, "_exec", fake_exec)
    plat.run(["echo"], str(tmp_path / "out.log"))
    assert seen["timeout"] == 45


# -- Wave 5 加固：bin 目录哈希隔离 / 超时落盘 / 设备核数回退 ----------------

def test_different_bin_dirs_map_to_isolated_remote_dirs(tmp_path, monkeypatch):
    """同名 bin 目录（不同绝对路径）必须映射到不同远端目录，互不删除。"""
    bin_a = tmp_path / "A" / "bin"
    bin_b = tmp_path / "B" / "bin"
    for d in (bin_a, bin_b):
        d.mkdir(parents=True)
        (d / "llama-bench").write_text("x", encoding="utf-8")
        (d / "libMNN.so").write_text("y", encoding="utf-8")

    tag_a = AdbPlatform._bin_dir_tag(str(bin_a))
    tag_b = AdbPlatform._bin_dir_tag(str(bin_b))
    assert tag_a != tag_b
    assert tag_a.startswith("bin/bin_") and tag_b.startswith("bin/bin_")

    shell_cmds = []

    class Proc:
        def __init__(self, rc=0, stdout=""):
            self.returncode = rc
            self.stdout = stdout
            self.stderr = ""

    def fake_run(args, *a, **kw):
        shell_str = " ".join(args[4:]) if len(args) > 4 else ""
        shell_cmds.append(shell_str)
        if shell_str.startswith("ls -1 "):
            return Proc(0, "stale.so\n")
        return Proc(0, "")

    monkeypatch.setattr(adb_mod.subprocess, "run", fake_run)
    plat = AdbPlatform(make_cfg())
    remote_a = plat._sync_dir(str(bin_a), tag_a, chmod_exec=True, clean_stale=True)
    remote_b = plat._sync_dir(str(bin_b), tag_b, chmod_exec=True, clean_stale=True)

    assert remote_a != remote_b
    # 缓存键是 (tag, local_dir)，不同目录不会命中同一条目
    assert (tag_a, str(bin_a)) in plat._synced
    assert (tag_b, str(bin_b)) in plat._synced
    rm_cmds = [c for c in shell_cmds if c.startswith("rm -f ")]
    assert rm_cmds, "clean_stale 应清理各自目录内的陈旧文件"
    for cmd in rm_cmds:
        for other in (tag_a, tag_b):
            if other in cmd:
                assert f"mobench/{other}/stale.so" in cmd


def test_sync_dir_md5_check_uses_scaled_push_timeout(tmp_path, monkeypatch):
    """远端 md5 探测也必须使用按体积放大的超时，避免大权重读校验被 120s 截断。"""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    big = _sparse_file(bin_dir / "llm.mnn.weight", 3_200_000_000)
    monkeypatch.setattr(adb_mod, "file_md5", lambda path: "0" * 32)
    seen = []
    plat = AdbPlatform(make_cfg())

    def fake_shell(cmd, timeout=120):
        seen.append(timeout)
        return 1, ""

    monkeypatch.setattr(plat, "_shell", fake_shell)
    monkeypatch.setattr(plat, "_push_file", lambda *a, **k: None)
    plat._sync_dir(str(bin_dir), AdbPlatform._bin_dir_tag(str(bin_dir)),
                   chmod_exec=True, clean_stale=True)

    assert max(seen) >= adb_mod.push_timeout(str(big)) > adb_mod.PUSH_MIN_TIMEOUT_S


def test_adb_run_timeout_writes_partial_output_and_cleans_remote_process(tmp_path, monkeypatch):
    """adb 超时：部分输出 + 显式标记落盘，并 best-effort 清理设备孤儿进程。"""
    plat = AdbPlatform(make_cfg())
    monkeypatch.setattr(plat, "_deploy_command", lambda cmd: "./llama-bench -m x")
    cleanup = []

    def fake_exec(args, timeout=120):
        raise adb_mod.subprocess.TimeoutExpired(
            args, timeout, output=b"partial-benchtop\n", stderr="partial-err\n")

    def fake_shell(cmd, timeout=120):
        cleanup.append((cmd, timeout))
        return 0, ""

    monkeypatch.setattr(plat, "_exec", fake_exec)
    monkeypatch.setattr(plat, "_shell", fake_shell)
    log = tmp_path / "timeout.log"

    with pytest.raises(adb_mod.subprocess.TimeoutExpired):
        plat.run(["/local/bin/llama-bench", "-m", "x"], str(log))

    text = log.read_text(encoding="utf-8")
    assert "partial-benchtop" in text
    assert "partial-err" in text
    assert "[mobench] adb command timed out after 600s" in text
    assert cleanup
    cleanup_cmd, cleanup_timeout = cleanup[0]
    assert "pkill -9 -x llama-bench" in cleanup_cmd[0]
    assert cleanup_timeout == 15


def test_known_cpus_falls_back_to_device_nproc(monkeypatch):
    """无 cpufreq 策略时用设备 nproc，而不是宿主机核数。"""
    plat = AdbPlatform(make_cfg())
    monkeypatch.setattr(plat, "_ensure_policies", lambda: [])
    monkeypatch.setattr(plat, "_shell", lambda cmd, timeout=60: (0, "8\n"))

    assert plat._known_cpus() == set(range(8))


def test_control_syncs_degraded_state_into_preflight_info(monkeypatch):
    """control() 的降级状态必须同步回 preflight info（run.json 可溯源）。"""
    plat = AdbPlatform(make_cfg())
    monkeypatch.setattr(plat, "_ensure_policies", lambda: [])
    monkeypatch.setattr(plat, "_is_root", lambda: False)
    info = {"platform": {"value": "adb"}}

    plat.control({"methodology": {"env_control": "auto"}}, info)

    assert "degraded" in info["env_control"]["value"].lower()
    assert info["env_control"]["value"] == plat._env_control_label
