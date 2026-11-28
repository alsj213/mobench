"""Android adb platform (spec: benchmark-core 平台抽象 host 与 adb).

Preflight verifies the device is online (never assumes). Binaries/models are
pushed to a device work dir before each command; model paths in the engine
command are remapped to their remote locations. Environment control is
best-effort and degrades when the device shell is not root.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import time

from mobench.platform.base import GOVERNOR, BasePlatform, PlatformError
from mobench.platform.cpu import parse_policies

REMOTE_DIR = "/data/local/tmp/mobench"
CPUFREQ_DIR = "/sys/devices/system/cpu/cpufreq"

_POLICY_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_ATRACE_CAT_RE = re.compile(r"^[A-Za-z0-9_:-]+$")
_PERF_EVENT_RE = re.compile(r"^[A-Za-z0-9_.:/-]+$")

# adb push 的固定 120 s 只够推约 2.3 GB（实测 19.5 MB/s），而 4B q4_k_m 经
# gguf2mnn 重建后单个权重文件就有 3.2 GB —— 必然超时中断。按体积放大超时，
# 速率取下限 5 MB/s 估，小文件仍维持原有 120 s 不变。
PUSH_MIN_TIMEOUT_S = 120.0
PUSH_ASSUMED_BYTES_PER_S = 5 * 1024 * 1024


def push_timeout(local_path: str) -> float:
    """adb push 该文件应给的超时（秒）：随体积放大，小文件维持 120 s。"""
    try:
        size = os.path.getsize(local_path)
    except OSError:
        return PUSH_MIN_TIMEOUT_S
    return max(PUSH_MIN_TIMEOUT_S, size / PUSH_ASSUMED_BYTES_PER_S)


class AdbPlatform(BasePlatform):
    name = "adb"

    def __init__(self, cfg):
        super().__init__(cfg)
        p = cfg.get("platform") or {}
        self.adb = p.get("adb")
        self.device_id = p.get("device_id")
        self._pushed = {}  # local path -> remote rel path
        self._synced = {}  # tag -> synced remote dir (cache)

    # -- adb helpers --------------------------------------------------------
    def _exec(self, args, timeout=120):
        return subprocess.run([self.adb, "-s", self.device_id] + args,
                              capture_output=True, text=True, timeout=timeout)

    def _shell(self, cmd, timeout=120):
        """Run a remote command, returning (returncode, output)."""
        proc = self._exec(["shell"] + cmd, timeout=timeout)
        return proc.returncode, proc.stdout

    def _adb_base(self):
        return [self.adb, "-s", self.device_id]

    # -- preflight ----------------------------------------------------------
    def preflight(self) -> dict:
        # devices listing
        proc = subprocess.run([self.adb, "devices"], capture_output=True,
                              text=True, timeout=60)
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        online = any(ln.split("\t")[0] == self.device_id and "device" in ln
                     for ln in lines[1:]) if len(lines) > 1 else False
        if not online:
            raise PlatformError(
                f"adb device {self.device_id} not online "
                f"({self.adb} devices showed no match).")
        self._shell(["mkdir", "-p", REMOTE_DIR])
        rc, model = self._shell(["getprop", "ro.product.model"])
        rc2, osv = self._shell(["getprop", "ro.build.version.release"])
        rc3, cores = self._shell(["nproc"])
        temp = self._temp_c()
        info = {
            "platform": {"value": "adb", "source": "mobench.yml platform.type"},
            "device_id": {"value": self.device_id, "source": "mobench.yml platform.device_id"},
            "model": {"value": model.strip() if rc == 0 else None,
                      "source": "adb shell getprop ro.product.model"},
            "os": {"value": osv.strip() if rc2 == 0 else None,
                   "source": "adb shell getprop ro.build.version.release"},
            "nproc": {"value": cores.strip() if rc3 == 0 else None,
                      "source": "adb shell nproc"},
            "temp_c": temp,
            "cpu_freq": self._cpu_freq(),
            "env_control": {"value": self._env_control_label, "source": "platform.control"},
        }
        return info

    def _temp_c(self):
        """Hottest readable thermal zone in Celsius (values are mC on Android).

        2026-10-09 实测教训：旧实现「取第一个可解析 zone」落在 aoss 等
        不敏感区（36°C），而同期 CPU zone 60°C+——温度监控形同虚设，
        throttle 判定（>45°C）永不触发。改为 type/temp 两次读取按行配对、
        取最大值（最热区即热压力信号），并附 zone 名用于诊断。
        （glob 展开序在同 shell 下稳定，两次读取按行对齐；type 缺行最坏只
        影响 zone 名标注，不影响最大值语义。）
        """
        rc_t, types = self._shell(["cat", "/sys/class/thermal/thermal_zone*/type"])
        rc_v, values = self._shell(["cat", "/sys/class/thermal/thermal_zone*/temp"])
        names = types.split()
        best = None
        for idx, raw in enumerate(values.split()):
            try:
                value = int(raw.strip()) / 1000.0
            except ValueError:
                continue
            if best is None or value > best[0]:
                best = (value, names[idx] if idx < len(names) else None)
        if best is not None:
            return {"value": best[0], "zone": best[1],
                    "source": "adb thermal_zone max (mC)", "unit": "celsius"}
        return {"value": None, "source": "adb thermal_zone",
                "error": "not available", "rc": rc_v or rc_t}

    def _cpu_freq(self):
        """Max current core frequency in MHz (performance-band signal).

        scaling_cur_freq is in kHz; 取最大值（通常即大核），与最热 zone 一起
        构成「性能档」观测面（add-thermal-band-monitor）。
        """
        rc, out = self._shell(
            ["cat", "/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"])
        values = []
        for raw in out.split():
            try:
                values.append(int(raw.strip()))
            except ValueError:
                continue
        if values:
            return {"value_mhz": round(max(values) / 1000.0, 1),
                    "source": "adb cpufreq scaling_cur_freq max"}
        return {"value_mhz": None, "source": "adb cpufreq",
                "error": "not available", "rc": rc}

    # -- cpufreq policies & governor ---------------------------------------
    def _governor_path(self, policy_name: str) -> str:
        return f"{CPUFREQ_DIR}/{policy_name}/scaling_governor"

    def discover_policies(self) -> list:
        """Read device cpufreq policies in one shell round-trip.

        The script travels as a *single* adb shell argument: adb joins the
        arguments after ``shell`` with spaces, so ``sh -c <script>`` would
        hand the device's shell only the word ``for`` (observed on a real
        device as ``syntax error: unexpected 'do'``).
        """
        script = (
            'for p in %s/policy*; do [ -d "$p" ] || continue; '
            'echo "$(basename $p)|$(cat $p/related_cpus 2>/dev/null)|'
            '$(cat $p/cpuinfo_max_freq 2>/dev/null)"; done' % CPUFREQ_DIR)
        _rc, out = self._shell([script], timeout=60)
        entries = []
        for line in (out or "").splitlines():
            parts = line.strip().split("|")
            if len(parts) != 3 or not parts[1].strip():
                continue
            entries.append({"name": parts[0].strip(),
                            "related_cpus": parts[1],
                            "cpuinfo_max_freq": parts[2]})
        policies = parse_policies(entries)
        if not policies:
            raise PlatformError(
                f"no cpufreq policies found on the device under {CPUFREQ_DIR}")
        return policies

    def _is_root(self) -> bool:
        rc, who = self._shell(["id", "-u"], timeout=60)
        if rc == 0 and who.strip() == "0":
            return True
        rc, who = self._shell(["su", "-c", "id -u"], timeout=60)
        return rc == 0 and who.strip() == "0"

    def _su_shell(self, script: str):
        """Run a shell snippet as root on the device.

        The whole ``su -c '<script>'`` travels as a single adb shell argument
        so the device shell performs the redirection, rather than adb's own
        argument splitting leaving ``su`` with a bare ``echo``.
        """
        return self._exec(["shell", "su -c " + shlex.quote(script)], timeout=60)

    def _lock_policies(self, policies) -> tuple:
        """Pin each policy's governor via root; degrade rather than raise.

        Locking is applied per policy *of the selected CPUs*: writing a fixed
        core such as cpu0 would lock the little cluster while the measurement
        runs on the big cores.
        """
        restored_from = {}
        if not self._is_root():
            self._env_control_label = "Env control degraded (non-root adb shell)"
            return False, restored_from
        failure = None
        for policy in policies:
            if not _POLICY_NAME_RE.match(policy.name):
                failure = f"invalid policy name {policy.name!r}"
                continue
            path = self._governor_path(policy.name)
            rc, original = self._shell(["cat", path], timeout=60)
            if rc != 0:
                failure = f"cannot read {path}"
                continue
            proc = self._su_shell(
                f"echo {shlex.quote(GOVERNOR)} > {shlex.quote(path)}"
            )
            if proc.returncode != 0:
                failure = ((proc.stderr or proc.stdout) or "").strip() or \
                    f"cannot write {path}"
                continue
            self._freq_state[policy.name] = original.strip()
            restored_from[policy.name] = original.strip()
        if failure is not None:
            self._env_control_label = f"Env control degraded ({failure})"
            return False, restored_from
        self._env_control_label = (
            "governor pinned to performance (restored on restore)")
        return True, restored_from

    def _known_cpus(self) -> set:
        """查询设备侧真实核数，避免无 cpufreq 时误退回到宿主机核数。"""
        policies = self._ensure_policies()
        if policies:
            return {c for p in policies for c in p.cpus}
        rc, out = self._shell(["nproc"], timeout=60)
        if rc == 0:
            try:
                count = int((out or "").strip())
                if count >= 1:
                    return set(range(count))
            except ValueError:
                pass
        return super()._known_cpus()

    # -- env control (best effort) -----------------------------------------
    def control(self, cfg: dict, env: dict) -> None:
        mode = (cfg.get("methodology") or {}).get("env_control", "auto")
        if mode == "none":
            self._env_control_label = "env control disabled (methodology.env_control=none)"
            self._sync_env_control(env)
            return
        self._ensure_policies()
        if not self._is_root():
            self._env_control_label = "Env control degraded (non-root adb shell)"
            self._sync_env_control(env)
            return
        self._env_control_label = "governor will be pinned to performance"
        self._sync_env_control(env)

    def restore(self) -> None:
        """Return every touched governor to its recorded original value."""
        for name, original in list(self._freq_state.items()):
            if not _POLICY_NAME_RE.match(name):
                continue
            self._su_shell(
                f"echo {shlex.quote(original)} > {shlex.quote(self._governor_path(name))}"
            )
        self._freq_state = {}
        self._pin_cpus, self._pin_prefix = [], []

    # -- execution ----------------------------------------------------------
    def _push_file(self, local_path, remote):
        """Push one file with a couple of retries (WSL->adb server vsock jitter)."""
        last = None
        for attempt in range(3):
            proc = self._exec(["push", local_path, remote],
                              timeout=push_timeout(local_path))
            if proc.returncode == 0:
                return
            last = proc
            time.sleep(0.4 * (attempt + 1))
        raise PlatformError(
            f"adb push failed for {local_path}: {(last.stderr or last.stdout).strip()}")

    def _push(self, local_path):
        if local_path in self._pushed:
            return self._pushed[local_path]
        if not os.path.isfile(local_path):
            raise PlatformError(f"local file missing for adb push: {local_path}")
        remote = REMOTE_DIR + "/" + os.path.basename(local_path)
        self._push_file(local_path, remote)
        self._pushed[local_path] = remote
        return remote

    def _sync_dir(self, local_dir, tag, chmod_exec=False):
        """Push a local directory to a cached remote dir (used for bin & models)."""
        local_dir = os.path.abspath(local_dir)
        if tag in self._synced:
            return self._synced[tag]
        remote = REMOTE_DIR + "/" + tag
        self._shell([f"mkdir -p {shlex.quote(remote)}"])
        for name in sorted(os.listdir(local_dir)):
            if name.startswith("."):
                continue
            path = os.path.join(local_dir, name)
            if not os.path.isfile(path):
                continue
            self._push_file(path, remote + "/" + name)
        if chmod_exec:  # adb push resets the exec bit
            self._shell([f"chmod 755 {shlex.quote(remote)}/*"])
        self._synced[tag] = remote
        return remote

    def _map_arg(self, arg):
        """Remap a local path arg: model dir/file -> synced remote, else push."""
        if os.path.isdir(arg):
            # CV benchmark.out takes a models folder; map it to its synced dir.
            for m in self.cfg.get("models") or []:
                mdir = m.get("dir")
                if mdir and os.path.isdir(mdir) and os.path.abspath(arg) == os.path.abspath(mdir):
                    return self._sync_dir(mdir, "models/" + os.path.basename(os.path.abspath(mdir).rstrip("/")))
            return arg
        if not os.path.isfile(arg):
            return arg
        for m in self.cfg.get("models") or []:
            mdir = m.get("dir")
            if not mdir or not os.path.isdir(mdir):
                continue
            if os.path.abspath(os.path.dirname(os.path.abspath(arg))) == os.path.abspath(mdir):
                tag = "models/" + os.path.basename(os.path.abspath(mdir).rstrip("/"))
                remote = self._sync_dir(mdir, tag)
                return remote + "/" + os.path.basename(arg)
        return self._push(arg)

    def _deploy_parts(self, cmd):
        """Return (env_prefix, body) for a local engine command.

        Pushes/syncs whatever the command references (bin dir, model files),
        remaps args to device paths. ``env_prefix`` carries the cd + library
        environment; ``body`` is the pure remote command line (ready to run
        directly, or as the target after ``simpleperf … -- ``).
        """
        bin_dir = (self.cfg.get("build") or {}).get("bin_dir")
        arg0 = cmd[0]
        in_bin_dir = bool(bin_dir) and os.path.isdir(bin_dir) and (
            os.path.abspath(arg0) == os.path.abspath(
                os.path.join(bin_dir, os.path.basename(arg0))))
        remote_cmd = []
        if in_bin_dir:
            # Dynamic-linked engine tools (RPATH $ORIGIN): sync the whole dir
            # and run with LD_LIBRARY_PATH pointing at the synced libraries.
            self._sync_dir(bin_dir, "bin", chmod_exec=True)
            remote_cmd.append("./" + os.path.basename(arg0))
        else:
            remote_cmd.append(self._map_arg(arg0))
        for arg in cmd[1:]:
            remote_cmd.append(self._map_arg(arg))
        env_prefix = (
            f"cd {shlex.quote(REMOTE_DIR + '/bin')} && LD_LIBRARY_PATH=. "
            if in_bin_dir
            else f"cd {shlex.quote(REMOTE_DIR)} && "
        )
        return env_prefix, " ".join(shlex.quote(str(a)) for a in remote_cmd)

    def _pinned_body(self, body):
        """Prefix a remote command with the CPU pin.

        One place applies the measurement condition, so run / profile / memory
        all execute under the same one (design D5).
        """
        prefix = " ".join(shlex.quote(str(p)) for p in self._pin_prefix)
        return (prefix + " " + body) if prefix else body

    def _deploy_command(self, cmd):
        env, body = self._deploy_parts(cmd)
        return env + self._pinned_body(body)

    def run(self, cmd, log_path: str, cwd=None, echo=True) -> int:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        shell_script = self._deploy_command(cmd)
        proc = self._exec(["shell", shell_script], timeout=600)
        out = proc.stdout
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(out)
            fh.write(proc.stderr)
        if echo:
            print(out, end="")
        return proc.returncode

    # -- memory sampling (long engine run + pid poll) ---------------------
    def spawn_engine(self, cmd):
        """Start the engine command in its own adb shell (non-blocking Popen).

        The engine runs in a foreground adb session; the caller polls its pid
        (``pidof``) to sample memory. ``&``-based remote backgrounds hang on
        this adb/device, so we keep the remote process foreground instead.
        """
        env_prefix, body = self._deploy_parts(cmd)
        return subprocess.Popen(
            self._adb_base() + ["shell", env_prefix + self._pinned_body(body)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def pidof(self, name):
        """Device pid(s) for a process name ('' when not running)."""
        rc, out = self._shell([f"pidof {shlex.quote(str(name))}"], timeout=60)
        return out.strip() if rc == 0 else ""

    def read_proc_status(self, pid):
        """Return /proc/<pid>/status text, or None once the process is gone."""
        remote = f"/proc/{pid}/status"
        rc, out = self._shell([f"cat {shlex.quote(remote)}"], timeout=60)
        return out if rc == 0 else None

    # -- system trace (atrace) -------------------------------------------
    def spawn_trace(self, dur, cats="freq", out_remote=None):
        """Start a device atrace capture into a remote file (non-blocking)."""
        out_remote = out_remote or REMOTE_DIR + "/atrace.txt"
        if isinstance(cats, str):
            cat_list = [c for c in re.split(r"[\s,]+", cats.strip()) if c]
        else:
            cat_list = [str(c).strip() for c in (cats or []) if str(c).strip()]
        if not cat_list or any(not _ATRACE_CAT_RE.match(c) for c in cat_list):
            raise PlatformError(f"invalid atrace categories: {cats!r}")
        cats_str = " ".join(shlex.quote(c) for c in cat_list)
        cmd = "atrace -b 16384 -t %d %s > %s 2>&1" % (
            int(dur), cats_str, shlex.quote(out_remote)
        )
        return subprocess.Popen(
            self._adb_base() + ["shell", cmd],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def read_remote_file(self, remote, timeout=240):
        rc, out = self._shell([f"cat {shlex.quote(remote)}"], timeout=timeout)
        return out if rc == 0 else ""

    # -- profiling (simpleperf) ------------------------------------------
    def run_profiled(self, cmd, freq=2000, mmap_pages=None, events=None,
                     timeout=900) -> dict:
        """Run the engine command under device simpleperf sampling.

        Returns {"record_rc", "perf_data", "stat_text", "samples_lost_hint"}
        where perf_data is the remote record path. The benchmark output itself
        is consumed by simpleperf; capture stderr for the engine table.
        """
        env_prefix, body = self._deploy_parts(cmd)
        perf_remote = REMOTE_DIR + "/perf.data"
        mmap_flag = f"-m {int(mmap_pages)}" if mmap_pages else ""
        rec = (env_prefix
               + "simpleperf record -f %d %s -o %s -g -- %s"
               % (int(freq),
                  mmap_flag,
                  shlex.quote(perf_remote), self._pinned_body(body)))
        rec_proc = self._exec(["shell", rec], timeout=timeout)
        stat_text = ""
        if events:
            event_list = [str(ev) for ev in events]
            for ev in event_list:
                if not _PERF_EVENT_RE.match(ev):
                    raise PlatformError(f"invalid perf event name: {ev!r}")
            stat = (env_prefix
                    + "simpleperf stat -e %s -- %s"
                    % (shlex.quote(",".join(event_list)), self._pinned_body(body)))
            stat_proc = self._exec(["shell", stat], timeout=timeout)
            stat_text = (stat_proc.stdout or "") + (stat_proc.stderr or "")
        hint = ""
        for ln in ((rec_proc.stdout or "") + "\n" + (rec_proc.stderr or "")).splitlines():
            low = ln.lower()
            if ("samples lost" in low or "lost" in low) and "%" in low:
                hint = ln.strip()
        return {"record_rc": rec_proc.returncode, "perf_data": perf_remote,
                "stat_text": stat_text, "samples_lost_hint": hint}

    # -- files --------------------------------------------------------------
    def file_exists(self, path: str) -> bool:
        return os.path.isfile(path)

    def file_md5(self, path: str) -> str:
        # Content-hash is computed locally; adb push is checksum-protected.
        return super().file_md5(path)
