"""绑核与锁频控制测试 (add-cpu-pin-lockfreq).

Host coverage runs against a fake sysfs tree so the governor writes and the
restore path are exercised for real, with no dependency on the machine the
tests run on.
"""
import subprocess

import pytest

from mobench.platform import host as host_mod
from mobench.platform.base import PlatformError
from mobench.platform.host import HostPlatform

# 1+3+4 (Snapdragon 865 / Xiaomi Mi 10)
SM8250 = [("policy0", [0, 1, 2, 3], 1804800),
          ("policy4", [4, 5, 6], 2419200),
          ("policy7", [7], 2841600)]


def make_sysfs(root, policies=SM8250, governor="schedutil"):
    """Build a fake /sys/devices/system/cpu tree with cpufreq policies."""
    for name, cpus, freq in policies:
        d = root / "cpufreq" / name
        d.mkdir(parents=True)
        (d / "related_cpus").write_text(
            " ".join(str(c) for c in cpus) + "\n", encoding="utf-8")
        (d / "cpuinfo_max_freq").write_text(f"{freq}\n", encoding="utf-8")
        (d / "scaling_governor").write_text(governor + "\n", encoding="utf-8")
    return root


def _cfg(**methodology):
    return {"platform": {"type": "host"},
            "methodology": {"env_control": "auto", "cpu_pin": True,
                            "pin_cpus": None, "lock_freq": True,
                            **methodology}}


@pytest.fixture
def host(tmp_path, monkeypatch):
    """HostPlatform wired to a fake sysfs tree, running as root."""
    make_sysfs(tmp_path)
    monkeypatch.setattr(host_mod, "_SYSFS_CPU", str(tmp_path))
    monkeypatch.setattr(host_mod.os, "geteuid", lambda: 0)
    return HostPlatform(_cfg())


def _governor(tmp_path, policy):
    return (tmp_path / "cpufreq" / policy / "scaling_governor").read_text(
        encoding="utf-8").strip()


# -- cluster-aware pinning --------------------------------------------------


def test_measure_condition_selects_whole_cluster(host):
    # Act
    cond = host.measure_condition(3)

    # Assert
    assert cond["pin"]["cpus"] == [4, 5, 6]
    assert cond["pin"]["mask"] == "70"
    assert cond["pin"]["origin"] == "auto"


def test_measure_condition_four_threads(host):
    # Act
    cond = host.measure_condition(4)

    # Assert
    assert cond["pin"]["cpus"] == [4, 5, 6, 7]
    assert cond["pin"]["mask"] == "f0"


def test_pin_prefix_is_taskset_with_hex_mask(host):
    # Act
    host.measure_condition(4)

    # Assert
    assert host.pin_prefix() == ["taskset", "f0"]


def test_explicit_pin_cpus_overrides_detection(tmp_path, monkeypatch):
    # Arrange
    make_sysfs(tmp_path)
    monkeypatch.setattr(host_mod, "_SYSFS_CPU", str(tmp_path))
    monkeypatch.setattr(host_mod.os, "geteuid", lambda: 0)
    platform = HostPlatform(_cfg(pin_cpus="4-5"))

    # Act
    cond = platform.measure_condition(1)

    # Assert
    assert cond["pin"]["cpus"] == [4, 5]
    assert cond["pin"]["origin"] == "explicit"


def test_explicit_pin_cpus_out_of_range_is_refused(host):
    # Arrange
    host.cfg["methodology"]["pin_cpus"] = [4, 99]

    # Act / Assert
    with pytest.raises(PlatformError, match="99"):
        host.measure_condition(2)


def test_explicit_set_larger_than_thread_count_is_allowed(host):
    """An explicit set may exceed the thread count: taskset binds the set and
    the threads share it. The actual set is what gets recorded."""
    # Arrange
    host.cfg["methodology"]["pin_cpus"] = [4, 5, 6, 7]

    # Act
    cond = host.measure_condition(2)

    # Assert
    assert cond["pin"]["cpus"] == [4, 5, 6, 7]
    assert host.pin_prefix() == ["taskset", "f0"]


def test_disabled_pin_reports_uncontrolled(host):
    # Arrange
    host.cfg["methodology"]["cpu_pin"] = False

    # Act
    cond = host.measure_condition(4)

    # Assert
    assert cond["pin"]["cpus"] == []
    assert cond["pin"]["origin"] == "disabled"
    assert host.pin_prefix() == []


def test_env_control_none_skips_everything(host):
    # Arrange
    host.cfg["methodology"]["env_control"] = "none"

    # Act
    cond = host.measure_condition(4)

    # Assert
    assert cond["pin"]["cpus"] == []
    assert cond["freq"]["locked"] is False


# -- failure semantics ------------------------------------------------------


def test_no_cpufreq_degrades_instead_of_aborting(tmp_path, monkeypatch):
    """D4: an environment with no cpufreq (WSL2 and similar) cannot be asked
    which CPUs share a clock domain, so it degrades instead of aborting —
    but the run must be labelled unavailable, never 'pinned'."""
    # Arrange (empty sysfs root)
    monkeypatch.setattr(host_mod, "_SYSFS_CPU", str(tmp_path))
    monkeypatch.setattr(host_mod.os, "geteuid", lambda: 0)
    platform = HostPlatform(_cfg())
    platform.control(platform.cfg, {})

    # Act
    cond = platform.measure_condition(4)

    # Assert
    assert cond["pin"]["cpus"] == []
    assert cond["pin"]["origin"] == "unavailable"
    assert cond["freq"]["locked"] is False
    assert platform.pin_prefix() == []
    assert "degraded" in platform._env_control_label.lower()


def test_explicit_pin_cpus_works_without_cpufreq(tmp_path, monkeypatch):
    """An explicit CPU set needs no cluster information, so it still binds."""
    # Arrange
    monkeypatch.setattr(host_mod, "_SYSFS_CPU", str(tmp_path))
    monkeypatch.setattr(host_mod.os, "geteuid", lambda: 0)
    platform = HostPlatform(_cfg(pin_cpus="0-1"))

    # Act
    cond = platform.measure_condition(2)

    # Assert
    assert cond["pin"]["cpus"] == [0, 1]
    assert cond["pin"]["origin"] == "explicit"
    assert platform.pin_prefix() == ["taskset", "3"]


def test_unwritable_governor_degrades_without_raising(host, tmp_path):
    # Arrange: a directory where the governor file should be
    bad = tmp_path / "cpufreq" / "policy4" / "scaling_governor"
    bad.unlink()
    bad.mkdir()

    # Act
    cond = host.measure_condition(3)

    # Assert
    assert cond["freq"]["locked"] is False
    assert "degraded" in host._env_control_label.lower()
    # pinning still succeeded
    assert cond["pin"]["cpus"] == [4, 5, 6]


def test_no_root_degrades_lock_freq(tmp_path, monkeypatch):
    # Arrange
    make_sysfs(tmp_path)
    monkeypatch.setattr(host_mod, "_SYSFS_CPU", str(tmp_path))
    monkeypatch.setattr(host_mod.os, "geteuid", lambda: 1000)
    platform = HostPlatform(_cfg())
    platform.control(platform.cfg, {})

    # Act
    cond = platform.measure_condition(3)

    # Assert
    assert cond["freq"]["locked"] is False
    assert cond["pin"]["cpus"] == [4, 5, 6]      # pinning needs no root
    assert _governor(tmp_path, "policy4") == "schedutil"


# -- frequency locking ------------------------------------------------------


def test_lock_freq_targets_policies_of_selected_cpus(host, tmp_path):
    # Act
    cond = host.measure_condition(4)

    # Assert
    assert _governor(tmp_path, "policy4") == "performance"
    assert _governor(tmp_path, "policy7") == "performance"
    # the little cluster was never selected -> untouched
    assert _governor(tmp_path, "policy0") == "schedutil"
    assert cond["freq"]["locked"] is True
    assert sorted(cond["freq"]["policies"]) == ["policy4", "policy7"]


def test_lock_freq_records_restore_point(host):
    # Act
    cond = host.measure_condition(4)

    # Assert
    assert cond["freq"]["restored_from"] == {"policy4": "schedutil",
                                             "policy7": "schedutil"}


def test_restore_returns_original_governors(host, tmp_path):
    # Arrange
    host.measure_condition(4)

    # Act
    host.restore()

    # Assert
    assert _governor(tmp_path, "policy4") == "schedutil"
    assert _governor(tmp_path, "policy7") == "schedutil"


def test_restore_is_idempotent(host, tmp_path):
    # Arrange
    host.measure_condition(4)

    # Act / Assert (a second restore must not raise or clobber)
    host.restore()
    host.restore()
    assert _governor(tmp_path, "policy4") == "schedutil"


def test_restore_without_measurement_is_safe(host):
    # Act / Assert
    host.restore()


# -- command prefixing ------------------------------------------------------


def test_run_prefixes_taskset(host, tmp_path, monkeypatch):
    """`run` must apply the pin transparently, so every measurement path is
    covered without each caller remembering to."""
    # Arrange
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, b"ok")

    monkeypatch.setattr(host_mod.subprocess, "run", fake_run)
    host.measure_condition(4)
    log = tmp_path / "raw.log"

    # Act
    host.run(["llama-bench", "-m", "m.gguf"], str(log))

    # Assert
    assert seen[0][:2] == ["taskset", "f0"]
    assert seen[0][2:] == ["llama-bench", "-m", "m.gguf"]


def test_run_without_pin_is_unchanged(tmp_path, monkeypatch):
    # Arrange
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, b"ok")

    make_sysfs(tmp_path)
    monkeypatch.setattr(host_mod, "_SYSFS_CPU", str(tmp_path))
    monkeypatch.setattr(host_mod.subprocess, "run", fake_run)
    platform = HostPlatform(_cfg(cpu_pin=False))
    platform.measure_condition(1)

    # Act
    platform.run(["llama-bench"], str(tmp_path / "raw.log"))

    # Assert
    assert seen[0] == ["llama-bench"]


# -- adb platform: the two pre-existing defects ----------------------------

from mobench.platform.adb import AdbPlatform  # noqa: E402

POLICY_OUT = ("policy0|0 1 2 3|1804800\n"
              "policy4|4 5 6|2419200\n"
              "policy7|7|2841600\n")


def _adb(cfg=None):
    platform = AdbPlatform(cfg or _cfg())
    platform.adb = "adb"
    platform.device_id = "serial"
    return platform


def _wire(platform, monkeypatch, governor="schedutil", root=True):
    """Fake the device shell: cpufreq tree, root probe and reads."""
    def fake_shell(cmd, timeout=120):
        if cmd[0].startswith("for p in"):
            return 0, POLICY_OUT
        if cmd[:2] == ["id", "-u"]:
            return 0, ("0\n" if root else "2000\n")
        if cmd[0] == "cat":
            return 0, governor + "\n"
        return 0, ""

    ok = subprocess.CompletedProcess([], 0, "", "")
    monkeypatch.setattr(platform, "_shell", fake_shell)
    monkeypatch.setattr(platform, "_exec", lambda args, timeout=120: ok)
    return platform


def test_adb_discover_passes_script_as_single_argument(monkeypatch):
    """Regression (found on a real device): adb joins the arguments after
    ``shell`` with spaces, so ``sh -c <script>`` leaves the device's shell
    with just ``for`` -> `syntax error: unexpected 'do'`. The script must be
    handed over as one argument."""
    # Arrange
    platform = _adb()
    seen = []

    def fake_shell(cmd, timeout=120):
        seen.append(cmd)
        return 0, POLICY_OUT

    monkeypatch.setattr(platform, "_shell", fake_shell)

    # Act
    policies = platform.discover_policies()

    # Assert
    assert [p.name for p in policies] == ["policy0", "policy4", "policy7"]
    assert len(seen[0]) == 1, f"script must be one argument, got {seen[0]!r}"


def test_adb_locks_policies_of_selected_cpus_not_cpu0(monkeypatch):
    """Defect fix: locking targeted cpu0 (a little core) no matter which cores
    were actually selected for the measurement."""
    # Arrange
    platform = _wire(_adb(), monkeypatch)

    # Act
    cond = platform.measure_condition(3)          # big cluster cpu4-6

    # Assert
    assert cond["pin"]["cpus"] == [4, 5, 6]
    assert cond["freq"]["policies"] == ["policy4"]
    assert platform._freq_state == {"policy4": "schedutil"}
    assert "policy0" not in platform._freq_state


def test_adb_lock_freq_degrades_without_root(monkeypatch):
    # Arrange
    platform = _wire(_adb(), monkeypatch, root=False)

    # Act
    cond = platform.measure_condition(3)

    # Assert
    assert cond["freq"]["locked"] is False
    assert cond["pin"]["cpus"] == [4, 5, 6]       # pinning needs no root
    assert "degraded" in platform._env_control_label.lower()


def test_adb_restore_writes_back_original_governor(monkeypatch):
    """Defect fix: restore() was a no-op, leaving the device pinned."""
    # Arrange
    platform = _wire(_adb(), monkeypatch)
    writes = []

    def fake_su(script):
        writes.append(script)
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(platform, "_su_shell", fake_su)
    platform.measure_condition(3)

    # Act
    platform.restore()

    # Assert
    assert any("schedutil" in w and "policy4" in w for w in writes)


def test_profile_adb_restores_governor_after_run(monkeypatch, tmp_path):
    """Regression (found on a real device): the profile path locked the
    governor and never restored it, leaving the device pinned to performance
    and poisoning every later measurement. Every execution path that applies
    the condition must undo it."""
    # Arrange
    from mobench import profiling

    calls = []

    class FakePlatform:
        name = "adb"

        def preflight(self):
            return {"platform": {"value": "adb"}}

        def measure_condition(self, threads):
            calls.append("measure")
            return {"pin": {"cpus": [4, 5, 6, 7], "mask": "f0", "origin": "auto"},
                    "freq": {"locked": True, "governor": "performance",
                             "policies": ["policy4", "policy7"],
                             "restored_from": {"policy4": "schedutil"}}}

        def run_profiled(self, cmd, **kwargs):
            calls.append("run")
            return {"record_rc": 0, "perf_data": "/remote/perf.data",
                    "stat_text": "", "samples_lost_hint": None}

        def _exec(self, args, timeout=120):
            return subprocess.CompletedProcess(args, 0, "", "")

        def restore(self):
            calls.append("restore")

        def file_md5(self, path):
            return "0" * 32

    class FakeAdapter:
        def version(self):
            return "v"

        def resolve_binary(self):
            return __file__

        def build_cmds(self, model, threads, rounds):
            return [["echo", "hi"]]

    cfg = {"engine": "mnn", "models": [{"id": "m", "path": __file__}],
           "workload": {"threads": [1]}, "results_dir": str(tmp_path)}
    monkeypatch.setattr(profiling, "find_simpleperf_dir", lambda ndk_dir=None: "")

    # Act
    meta = profiling._profile_pipeline_adb(cfg, _platform=FakePlatform(),
                                           _adapter=FakeAdapter())

    # Assert
    assert calls[0] == "measure"
    assert calls[-1] == "restore", f"profile must restore, got {calls!r}"
    assert meta["condition"]["freq"]["locked"] is True
    assert meta["condition"]["pin"]["cpus"] == [4, 5, 6, 7]


def test_profile_host_restores_governor_on_failure(monkeypatch, tmp_path):
    """A failing profile run must still undo the condition it applied."""
    # Arrange
    from mobench import profiling

    calls = []

    class FakePlatform:
        name = "host"
        cfg = {}

        def preflight(self):
            return {"platform": {"value": "host"}}

        def measure_condition(self, threads):
            calls.append("measure")
            return {"pin": {"cpus": []}, "freq": {"locked": True}}

        def with_pin(self, cmd):
            return list(cmd)

        def restore(self):
            calls.append("restore")

        def file_md5(self, path):
            return "0" * 32

    class FakeAdapter:
        def version(self):
            return "v"

        def resolve_binary(self):
            return __file__

        def build_cmds(self, model, threads, rounds):
            return [["echo", "hi"]]

    cfg = {"engine": "mnn", "models": [{"id": "m", "path": __file__}],
           "workload": {"threads": [1]}, "results_dir": str(tmp_path)}
    # perf exists but fails, i.e. the failure happens *after* the condition
    # has been applied -- the case that used to leave the box pinned.
    monkeypatch.setattr(profiling, "find_perf", lambda: "/bin/false")

    # Act / Assert
    with pytest.raises(RuntimeError):
        profiling._profile_host(cfg, _platform=FakePlatform(),
                                _adapter=FakeAdapter())
    assert calls == ["measure", "restore"], f"unexpected calls {calls!r}"


def test_adb_run_applies_pin_to_remote_command(monkeypatch, tmp_path):
    # Arrange
    platform = _wire(_adb(), monkeypatch)
    seen = []
    ok = subprocess.CompletedProcess([], 0, "", "")

    def fake_exec(args, timeout=120):
        seen.append(args)
        return ok

    monkeypatch.setattr(platform, "_exec", fake_exec)
    platform.measure_condition(4)

    # Act
    platform.run(["llama-bench", "-m", "x.gguf"], str(tmp_path / "raw.log"))

    # Assert
    remote = seen[-1][-1]
    assert "taskset f0" in remote
    assert remote.index("taskset") < remote.index("llama-bench")
