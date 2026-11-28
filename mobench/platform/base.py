"""Platform contract (spec: benchmark-core 平台抽象 host 与 adb).

The same benchmark semantics run on ``host`` (direct execution on Linux) or
``adb`` (push + remote execution on an Android device). Environment control
is best-effort: without root/sudo it degrades to recording state and labeling
``Env control degraded`` without blocking measurement.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import subprocess

from mobench.config import parse_cpus
from mobench.io_utils import file_md5 as _stream_file_md5
from mobench.platform.cpu import CpuSelectionError, cpu_mask, select_cpus

GOVERNOR = "performance"


class PlatformError(Exception):
    """Platform preflight failure (offline device, missing binary, ...)."""


def _local_run(cmd, timeout=60):
    """Run a command locally; return {returncode, stdout, stderr}."""
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


@contextmanager
def measurement_session(platform, cfg: dict, threads: int, info: dict | None = None):
    """统一测量会话：control -> measure_condition -> 执行 -> 保证 restore()。"""
    target = info if isinstance(info, dict) else {}
    if hasattr(platform, "control"):
        platform.control(cfg, target)
    condition = None
    try:
        if hasattr(platform, "measure_condition"):
            condition = platform.measure_condition(threads)
        yield condition
    finally:
        if hasattr(platform, "restore"):
            platform.restore()


class BasePlatform:
    name = "base"

    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self._policies = None        # None = not probed yet, [] = probed empty
        self._pin_cpus = []
        self._pin_prefix = []
        self._freq_state = {}        # policy name -> governor to restore
        self._env_control_label = "env control not run"
        self._active_env = None

    def _sync_env_control(self, env: dict | None = None) -> None:
        """把最新的 _env_control_label 同步写回 preflight info / run meta。"""
        if isinstance(env, dict):
            self._active_env = env
        target = self._active_env
        if not isinstance(target, dict):
            return
        entry = {"value": self._env_control_label, "source": "platform.control"}
        plat = target.get("platform")
        if isinstance(plat, dict) and "env_control" in plat:
            plat["env_control"] = entry
        else:
            target["env_control"] = entry

    def measurement_session(self, threads: int, info: dict | None = None):
        return measurement_session(self, self.cfg, threads, info=info)

    # -- CPU pinning & frequency locking (add-cpu-pin-lockfreq) -------------
    def discover_policies(self) -> list:
        """Return this platform's cpufreq policies (raises PlatformError)."""
        raise NotImplementedError

    def _lock_policies(self, policies) -> tuple:
        """Pin the given policies to GOVERNOR. Returns (locked, restored_from).

        Best effort: a failure degrades to a labelled state, never raises.
        """
        raise NotImplementedError

    def _ensure_policies(self) -> list:
        if self._policies is None:
            try:
                self._policies = self.discover_policies()
            except (PlatformError, OSError):
                self._policies = []
        return self._policies

    def _policies_for(self, cpus) -> list:
        """Policies owning at least one of ``cpus``."""
        wanted = set(cpus)
        return [p for p in self._ensure_policies() if wanted & set(p.cpus)]

    def measure_condition(self, threads: int) -> dict:
        """Select CPUs for ``threads``, bind them and lock their frequencies.

        Returns the condition dict that goes into run.json and into every
        record. CPU pinning failure raises PlatformError (fail-fast): binding
        needs no privileges, so a failure is a configuration or probe error,
        and a silent downgrade would make pinned and unpinned data
        indistinguishable. Frequency locking degrades to a labelled state.
        """
        methodology = self.cfg.get("methodology") or {}
        if methodology.get("env_control") == "none":
            self._pin_cpus, self._pin_prefix = [], []
            self._env_control_label = "env control disabled (methodology.env_control=none)"
            self._sync_env_control()
            return {"pin": {"cpus": [], "mask": None, "origin": "disabled"},
                    "freq": {"locked": False, "governor": None,
                             "policies": [], "restored_from": {}}}
        pin = self._apply_pin(methodology, threads)
        cond = {"pin": pin, "freq": self._apply_lock_freq(methodology, pin["cpus"])}
        self._sync_env_control()
        return cond

    def _known_cpus(self) -> set:
        """CPU indexes this platform can address.

        Cluster policies when available, otherwise the reported core count:
        an explicit CPU set is still bindable on a machine with no cpufreq.
        """
        policies = self._ensure_policies()
        if policies:
            return {c for p in policies for c in p.cpus}
        return set(range(os.cpu_count() or 1))

    def _apply_pin(self, methodology, threads) -> dict:
        if not methodology.get("cpu_pin", True):
            self._pin_cpus, self._pin_prefix = [], []
            return {"cpus": [], "mask": None, "origin": "disabled"}
        try:
            explicit = parse_cpus(methodology.get("pin_cpus"))
        except Exception as exc:
            raise PlatformError(f"methodology.pin_cpus is invalid: {exc}") from exc
        if explicit:
            known = self._known_cpus()
            unknown = sorted(c for c in explicit if c not in known)
            if unknown:
                raise PlatformError(
                    f"methodology.pin_cpus selects CPU(s) {unknown}, which do "
                    f"not exist on this platform (available: "
                    f"{sorted(known) or 'none'})")
            cpus, origin = explicit, "explicit"
        else:
            policies = self._ensure_policies()
            if not policies:
                # The environment cannot say which CPUs share a clock domain
                # (WSL2 and similar), so there is nothing sound to select.
                # Degrade rather than abort: this is an environment limit, not
                # a configuration error (design D4). The origin label keeps
                # this unpinned run distinguishable from a pinned one.
                self._pin_cpus, self._pin_prefix = [], []
                self._env_control_label = (
                    "Env control degraded (no cpufreq: cannot auto-select "
                    "CPUs; set methodology.pin_cpus to pin explicitly)")
                return {"cpus": [], "mask": None, "origin": "unavailable"}
            try:
                cpus = select_cpus(policies, threads)
            except CpuSelectionError as exc:
                raise PlatformError(f"cannot pin CPUs: {exc}") from exc
            origin = "auto"
        try:
            mask = cpu_mask(cpus)
        except CpuSelectionError as exc:
            raise PlatformError(f"cannot pin CPUs: {exc}") from exc
        self._pin_cpus = cpus
        self._pin_prefix = ["taskset", mask]
        return {"cpus": cpus, "mask": mask, "origin": origin}

    def _apply_lock_freq(self, methodology, cpus) -> dict:
        empty = {"locked": False, "governor": None,
                 "policies": [], "restored_from": {}}
        if not methodology.get("lock_freq", True) or not cpus:
            return empty
        targets = self._policies_for(cpus)
        if not targets:
            return empty
        locked, restored_from = self._lock_policies(targets)
        return {"locked": locked,
                "governor": GOVERNOR if locked else None,
                "policies": [p.name for p in targets],
                "restored_from": restored_from}

    def pin_prefix(self) -> list:
        """Command prefix applying the current measurement condition."""
        return list(self._pin_prefix)

    def with_pin(self, cmd) -> list:
        """``cmd`` prefixed with the pin, ready to execute."""
        return self.pin_prefix() + list(cmd)

    def preflight(self) -> dict:
        """Verify the platform is usable and return identity/temperature info.

        Raises PlatformError when the platform is not usable (never assume).
        """
        raise NotImplementedError

    def control(self, cfg: dict, env: dict) -> None:
        """Best-effort environment control (freq/thermal/caches). Degrades only."""

    def restore(self) -> None:
        """Restore environment after measurement (no-op on graceful degrade)."""

    def run(self, cmd, log_path: str, cwd=None, echo: bool = True) -> int:
        """Execute ``cmd`` on the platform, teeing raw output into ``log_path``.

        Returns the process exit code. Callers must surface non-zero codes.
        """
        raise NotImplementedError

    def file_exists(self, path: str) -> bool:
        raise NotImplementedError

    def file_md5(self, path: str) -> str:
        """Streaming md5 hex of a local file (adb impl may hash after pull)."""
        return _stream_file_md5(path)

    # -- thermal / performance-band sampling (add-thermal-band-monitor) -----
    def sample_thermal(self) -> dict:
        """One-shot thermal + frequency sample (hottest zone, max core freq).

        Recorded per measurement round so a run carries its performance-band
        trajectory.  2026-10-09 实证：连续跑测的热积累让「快档」自身漂移
        >1%（4.605→4.573→4.670ms），<1% 量级的判决随之失效——采样面是
        判决可信度的前提。
        """
        return {"temp_c": self._temp_c(), "cpu_freq": self._cpu_freq()}

    def _temp_c(self) -> dict:
        """Hottest readable thermal zone: {"value": °C, "zone": name|None}."""
        return {"value": None, "source": "n/a", "error": "not implemented"}

    def _cpu_freq(self) -> dict:
        """Max current core frequency: {"value_mhz": MHz|None}."""
        return {"value_mhz": None, "source": "n/a", "error": "not implemented"}

    @staticmethod
    def make(cfg: dict) -> "BasePlatform":
        """Factory: pick platform class by config platform.type."""
        from mobench.platform import host

        ptype = (cfg.get("platform") or {}).get("type", "host")
        if ptype == "adb":
            from mobench.platform import adb  # lazily: adb needs the adb binary

            return adb.AdbPlatform(cfg)
        return host.HostPlatform(cfg)
