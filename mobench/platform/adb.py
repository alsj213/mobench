"""Android adb platform (spec: benchmark-core 平台抽象 host 与 adb).

Preflight verifies the device is online (never assumes). Binaries/models are
pushed to a device work dir before each command; model paths in the engine
command are remapped to their remote locations. Environment control is
best-effort and degrades when the device shell is not root.
"""
from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
import time

from mobench.io_utils import file_md5
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
        self._synced = {}  # (tag, local_dir) -> synced remote dir (cache)

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

    @staticmethod
    def _is_online_line(line: str, device_id: str | None) -> bool:
        parts = line.split("\t")
        return len(parts) >= 2 and parts[0].strip() == device_id and parts[1].strip() == "device"

    # -- preflight ----------------------------------------------------------
    def preflight(self) -> dict:
        # devices listing
        proc = subprocess.run([self.adb, "devices"], capture_output=True,
                              text=True, timeout=60)
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        online = (
            any(self._is_online_line(ln, self.device_id) for ln in lines[1:])
            if len(lines) > 1
            else False
        )
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

        同 shell 循环配对读取每个可读 thermal_zone 的 type 与 temp，
        防止个别 zone 不可读导致 type/temp 行错位，取最热 zone 温度作热压力信号。
        """
        script = (
            'for z in /sys/class/thermal/thermal_zone*; do '
            '[ -r "$z/temp" ] || continue; '
            'echo "$(cat "$z/type" 2>/dev/null)|$(cat "$z/temp" 2>/dev/null)"; '
            'done'
        )
        rc, out = self._shell([script])
        best = None
        for line in (out or "").splitlines():
            zone, sep, raw = line.partition("|")
            if not sep:
                continue
            try:
                value = int(raw.strip()) / 1000.0
            except ValueError:
                continue
            if best is None or value > best[0]:
                best = (value, zone.strip() or None)
        if best is not None:
            return {"value": best[0], "zone": best[1],
                    "source": "adb thermal_zone max (mC)", "unit": "celsius"}
        return {"value": None, "source": "adb thermal_zone",
                "error": "not available", "rc": rc}

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

    def _remote_file_matches(self, remote_path: str, local_md5: str,
                             timeout: float = 120) -> bool:
        """检查设备端文件 MD5 是否与本地一致，实现跨进程幂等跳过推包。"""
        rc, out = self._shell(
            [f"md5sum {shlex.quote(remote_path)} 2>/dev/null"],
            timeout=timeout,
        )
        if rc != 0 or not out:
            return False
        tokens = out.strip().split()
        return bool(tokens) and tokens[0].lower() == local_md5.lower()

    def _push(self, local_path):
        if local_path in self._pushed:
            return self._pushed[local_path]
        if not os.path.isfile(local_path):
            raise PlatformError(f"local file missing for adb push: {local_path}")
        local_md5 = file_md5(local_path)
        remote_dir = f"{REMOTE_DIR}/files/{local_md5[:12]}"
        remote = f"{remote_dir}/{os.path.basename(local_path)}"
        if not self._remote_file_matches(
                remote, local_md5, timeout=push_timeout(local_path)):
            self._shell([f"mkdir -p {shlex.quote(remote_dir)}"])
            self._push_file(local_path, remote)
        self._pushed[local_path] = remote
        return remote

    @staticmethod
    def _model_dir_tag(local_dir: str) -> str:
        abs_dir = os.path.abspath(local_dir).rstrip("/")
        digest = hashlib.md5(abs_dir.encode("utf-8")).hexdigest()[:8]
        name = os.path.basename(abs_dir) or "root"
        return f"models/{name}_{digest}"

    @staticmethod
    def _bin_dir_tag(local_dir: str) -> str:
        """bin 目录的远端唯一标签：同名不同路径不得共享远端目录。

        多个 A/B 构建目录若都叫 ``bin``，旧实现会把它们映射到同一个
        ``mobench/bin``，``clean_stale=True`` 时会互删对方产物。把绝对路径
        哈希进标签即可隔离并发/交替运行的双方。
        """
        abs_dir = os.path.abspath(local_dir).rstrip("/")
        digest = hashlib.md5(abs_dir.encode("utf-8")).hexdigest()[:8]
        name = os.path.basename(abs_dir) or "bin"
        return f"bin/{name}_{digest}"

    def _sync_dir(self, local_dir, tag, chmod_exec=False, clean_stale=False):
        """Push a local directory to a cached remote dir (used for bin & models)."""
        local_dir = os.path.abspath(local_dir)
        cache_key = (tag, local_dir)
        if cache_key in self._synced:
            return self._synced[cache_key]
        remote = REMOTE_DIR + "/" + tag
        self._shell([f"mkdir -p {shlex.quote(remote)}"])
        local_files = [
            name
            for name in sorted(os.listdir(local_dir))
            if not name.startswith(".") and os.path.isfile(os.path.join(local_dir, name))
        ]
        if clean_stale:
            rc, out = self._shell([f"ls -1 {shlex.quote(remote)} 2>/dev/null"])
            if rc == 0 and out:
                keep = set(local_files)
                stale = [
                    line.strip()
                    for line in out.splitlines()
                    if line.strip()
                    and line.strip() not in (".", "..")
                    and "/" not in line
                    and line.strip() not in keep
                ]
                if stale:
                    rm_args = " ".join(
                        shlex.quote(f"{remote}/{name}") for name in stale
                    )
                    self._shell([f"rm -f {rm_args}"])
        for name in local_files:
            path = os.path.join(local_dir, name)
            remote_file = remote + "/" + name
            local_md5 = file_md5(path)
            if not self._remote_file_matches(
                    remote_file, local_md5, timeout=push_timeout(path)):
                self._push_file(path, remote_file)
        if chmod_exec and local_files:  # adb push resets the exec bit
            self._shell([f"chmod 755 {shlex.quote(remote)}/*"])
        self._synced[cache_key] = remote
        return remote

    def _map_arg(self, arg):
        """Remap a local path arg: model dir/file -> synced remote, else push."""
        if os.path.isdir(arg):
            # CV benchmark.out takes a models folder; map it to its synced dir.
            for m in self.cfg.get("models") or []:
                mdir = m.get("dir")
                if mdir and os.path.isdir(mdir) and os.path.abspath(arg) == os.path.abspath(mdir):
                    return self._sync_dir(mdir, self._model_dir_tag(mdir))
            return arg
        if not os.path.isfile(arg):
            return arg
        for m in self.cfg.get("models") or []:
            mdir = m.get("dir")
            if not mdir or not os.path.isdir(mdir):
                continue
            if os.path.abspath(os.path.dirname(os.path.abspath(arg))) == os.path.abspath(mdir):
                tag = self._model_dir_tag(mdir)
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
        remote_bin = None
        if in_bin_dir:
            # Dynamic-linked engine tools (RPATH $ORIGIN): sync the whole dir
            # and run with LD_LIBRARY_PATH pointing at the synced libraries.
            # 远端目录按本地绝对路径哈希隔离，A/B 两个 bin 目录不会互删。
            remote_bin = self._sync_dir(
                bin_dir, self._bin_dir_tag(bin_dir),
                chmod_exec=True, clean_stale=True)
            remote_cmd.append("./" + os.path.basename(arg0))
        else:
            remote_cmd.append(self._map_arg(arg0))
        for arg in cmd[1:]:
            remote_cmd.append(self._map_arg(arg))
        env_prefix = (
            f"cd {shlex.quote(remote_bin)} && LD_LIBRARY_PATH=. "
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
        raw_timeout = (self.cfg.get("methodology") or {}).get("timeout_s")
        timeout = 600 if raw_timeout is None else raw_timeout
        with open(log_path, "w", encoding="utf-8") as fh:
            try:
                proc = self._exec(["shell", shell_script], timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                # 超时同样要留证：把已产生的 stdout/stderr 落盘再追加显式标记，
                # 否则日志为空会让“超时”看起来像“没有任何输出”。
                for stream in (exc.stdout, exc.stderr):
                    if not stream:
                        continue
                    if isinstance(stream, bytes):
                        stream = stream.decode("utf-8", errors="replace")
                    fh.write(stream)
                fh.write(f"\n[mobench] adb command timed out after {timeout}s\n")
                fh.flush()
                # best-effort 清理设备端孤儿进程；清理失败不得掩盖超时本身。
                proc_name = os.path.basename(str(cmd[0]))
                try:
                    self._shell([
                        f"pkill -9 -x {shlex.quote(proc_name)} 2>/dev/null "
                        f"|| killall -9 {shlex.quote(proc_name)} 2>/dev/null "
                        f"|| true"
                    ], timeout=15)
                except Exception:
                    pass
                raise
            out = proc.stdout
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
