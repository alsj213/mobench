"""Host platform — direct execution on Linux (spec: benchmark-core 平台抽象).

Preflight verifies the platform is usable (never assumes). Environment control
is best-effort: with root it pins the cpufreq policies covering the selected
CPUs to the performance governor and restores them afterwards; without root it
labels ``Env control degraded`` and records state. CPU pinning itself needs no
privileges and is applied to every executed command.
"""
from __future__ import annotations

import glob
import os
import subprocess

from mobench.platform.base import GOVERNOR, BasePlatform, PlatformError
from mobench.platform.cpu import parse_policies

_SYSFS_CPU = "/sys/devices/system/cpu"


class HostPlatform(BasePlatform):
    name = "host"

    def __init__(self, cfg):
        super().__init__(cfg)

    # -- preflight ----------------------------------------------------------
    def preflight(self) -> dict:
        info = {
            "platform": {"value": "host", "source": "mobench.yml platform.type"},
            "os": self._run(["uname", "-o"]),
            "kernel": self._run(["uname", "-r"]),
            "cpu": self._run(["lscpu", "-J"]),
            "nproc": self._run(["nproc"]),
        }
        info["temp_c"] = self._temp_c()
        info["cpu_freq"] = self._cpu_freq()
        info["env_control"] = {"value": self._env_control_label, "source": "platform.control"}
        return info

    def _temp_c(self):
        """Hottest readable thermal zone in °C (与 adb 同语义：最热区)。

        2026-10-09 跨平台同步升级：旧实现取「第一个可读 zone」，可能落在
        不敏感区（实测设备侧偏差 24°C）；改为取最大值，hwmon 兜底。
        """
        best = None
        for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
            try:
                value = int(open(path, encoding="utf-8").read().strip()) / 1000.0
            except (ValueError, OSError):
                continue
            if best is None or value > best:
                best = value
        if best is not None:
            return {"value": best, "source": "thermal_zone max", "unit": "celsius"}
        for path in glob.glob("/sys/class/hwmon/hwmon*/temp*_input"):
            try:
                value = int(open(path, encoding="utf-8").read().strip()) / 1000.0
                return {"value": value, "source": "hwmon", "unit": "celsius"}
            except (ValueError, OSError):
                continue
        return {"value": None, "source": "thermal_zone", "error": "not available"}

    def _cpu_freq(self):
        """Max current core frequency in MHz (performance-band signal)."""
        values = []
        for path in glob.glob(
                os.path.join(_SYSFS_CPU, "cpu*/cpufreq/scaling_cur_freq")):
            try:
                values.append(int(open(path, encoding="utf-8").read().strip()))
            except (ValueError, OSError):
                continue
        if values:
            return {"value_mhz": round(max(values) / 1000.0, 1),
                    "source": "cpufreq scaling_cur_freq max"}
        return {"value_mhz": None, "source": "cpufreq", "error": "not available"}

    # -- cpufreq policies & governor ---------------------------------------
    def discover_policies(self) -> list:
        """Read cpufreq policies from sysfs (raises PlatformError if none).

        The kernel's policy layout is used rather than clustering cores by
        frequency: it is the granularity a governor is written at.
        """
        entries = []
        for directory in sorted(glob.glob(
                os.path.join(_SYSFS_CPU, "cpufreq", "policy*"))):
            entry = {"name": os.path.basename(directory)}
            for key in ("related_cpus", "cpuinfo_max_freq"):
                try:
                    with open(os.path.join(directory, key), encoding="utf-8") as fh:
                        entry[key] = fh.read().strip()
                except OSError:
                    entry[key] = None
            entries.append(entry)
        policies = parse_policies(entries)
        if not policies:
            raise PlatformError(
                f"no cpufreq policies under {_SYSFS_CPU}/cpufreq")
        return policies

    def _governor_path(self, policy_name: str) -> str:
        return os.path.join(_SYSFS_CPU, "cpufreq", policy_name,
                            "scaling_governor")

    def _lock_policies(self, policies) -> tuple:
        """Pin each policy's governor to performance; degrade on failure."""
        restored_from = {}
        if os.geteuid() != 0:
            self._env_control_label = "Env control degraded (no sudo)"
            return False, restored_from
        failure = None
        for policy in policies:
            path = self._governor_path(policy.name)
            try:
                with open(path, encoding="utf-8") as fh:
                    original = fh.read().strip()
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(GOVERNOR + "\n")
            except OSError as exc:
                failure = exc
                continue
            self._freq_state[policy.name] = original
            restored_from[policy.name] = original
        if failure is not None:
            self._env_control_label = f"Env control degraded ({failure})"
            return False, restored_from
        self._env_control_label = (
            "governor pinned to performance (restored on restore)")
        return True, restored_from

    # -- env control (best effort) -----------------------------------------
    def control(self, cfg: dict, env: dict) -> None:
        mode = (cfg.get("methodology") or {}).get("env_control", "auto")
        if mode == "none":
            self._env_control_label = "env control disabled (methodology.env_control=none)"
            self._sync_env_control(env)
            return
        # Probe here so an unusable platform surfaces at the env-control step;
        # a missing tree is reported by measure_condition with an actionable
        # message instead of aborting preflight.
        self._ensure_policies()
        if os.geteuid() != 0:
            self._env_control_label = "Env control degraded (no sudo)"
            self._sync_env_control(env)
            return
        self._env_control_label = "governor will be pinned to performance"
        self._sync_env_control(env)

    def restore(self) -> None:
        """Return every touched governor to its recorded original value."""
        for name, original in list(self._freq_state.items()):
            try:
                with open(self._governor_path(name), "w", encoding="utf-8") as fh:
                    fh.write(original + "\n")
            except OSError:
                pass  # best effort; the original stays recorded in run.json
        self._freq_state = {}
        self._pin_cpus, self._pin_prefix = [], []

    # -- execution ----------------------------------------------------------
    def run(self, cmd, log_path: str, cwd=None, echo=True) -> int:
        cmd = self.with_pin(cmd)
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, cwd=cwd)
            out = proc.stdout.decode("utf-8", errors="replace")
            fh.write(out)
            if echo:  # R7: raw output is shown alongside tables (unless --json)
                print(out, end="")
            return proc.returncode

    # -- files --------------------------------------------------------------
    def file_exists(self, path: str) -> bool:
        return os.path.isfile(path)

    # helpers ---------------------------------------------------------------
    def _run(self, cmd):
        source = " ".join(cmd)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            out = proc.stdout.strip()
            if proc.returncode != 0:
                return {"value": None, "source": source,
                        "error": proc.stderr.strip() or f"exit {proc.returncode}"}
            return {"value": out, "source": source, "error": None}
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            return {"value": None, "source": source, "error": str(exc)}
