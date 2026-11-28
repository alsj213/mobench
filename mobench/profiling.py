"""Profiling output parsers (spec: benchmark-core profiling).

Consumes plain-text sampling outputs from either platform so every derived
metric traces to a real source log (redline R3):

* adb  — Android simpleperf: ``stat -e`` -> counters/IPC/cache,
         ``report --sort symbol`` -> flat hot table,
         ``report_sample.py`` -> per-sample call stacks (fold).
* host — Linux perf: ``perf stat`` counter block, ``perf script`` dump whose
         leaf-first fp stacks fold into the same flamegraph artifacts.

Each parser is pure over text so it is unit-testable without a device.
"""
from __future__ import annotations

import re
from collections import Counter

_SAMPLE_HEAD = re.compile(
    r"^(\S+)\s+(\d+)\s+\[(\d+)\]\s+([\d.]+)\s*:\s+(\d+)\s+(\S+)\s*:")
# perf script sample header has no [cpu] column and no count modifier:
#   llm_bench 1243333 1984651.575802:         29 cycles:u: <trailing space>
_HOST_HEAD = re.compile(r"^(\S+)\s+(\d+)\s+([\d.]+):\s+(\d+)\s+(\S+):\s*$")
_FRAME = re.compile(r"^\s+([0-9a-fA-F]+)\s+(.*)$")
_KERNEL_ADDR = re.compile(r"\[\+([0-9a-fA-F]{6,})\]")
_OFFSET = re.compile(r"\+0x[0-9a-fA-F]+$")       # perf script `sym+0x1a0` suffix
_UNKNOWN_FRAME = re.compile(r"^[0-9a-fA-F]+\s+\[unknown\]$")
_STAT_ROW = re.compile(r"^\s*([\d,]+)\s+(\S+)\s+#.*$")
# perf stat rows: `<count> <event>:u` (modifier appended under paranoid>=2)
_PERF_STAT = re.compile(r"^\s*([\d,]+)\s+(\S+)")
_OVERHEAD_ROW = re.compile(r"^\s*(\d+(?:\.\d+)?)%\s+(.*)$")


def parse_stat(text: str) -> dict:
    """Parse `simpleperf stat` counter block into event -> count."""
    events = {}
    for line in text.splitlines():
        m = _STAT_ROW.match(line)
        if m:
            events[m.group(2)] = int(m.group(1).replace(",", ""))
    return events


def parse_perf_stat(text: str) -> dict:
    """Parse a Linux ``perf stat`` counter block into event -> count.

    Host perf appends a ``:u``/``:k`` modifier to each event name (paranoid
    >= 2 samples user-space only); the modifier is stripped so metrics share
    the simpleperf vocabulary.
    """
    events = {}
    for line in text.splitlines():
        m = _PERF_STAT.match(line)
        if m:
            events[m.group(2).split(":")[0]] = int(
                m.group(1).replace(",", ""))
    return events


def metrics_from_stat(events: dict) -> dict:
    """Derive IPC / cache-miss rate / L1 miss rate (None when unmeasured)."""
    def _ratio(num, den):
        return round(num / den, 4) if den else None

    cyc = events.get("cpu-cycles")
    ins = events.get("instructions")
    miss = events.get("cache-misses")
    ref = events.get("cache-references")
    l1m = events.get("L1-dcache-load-misses")
    l1r = events.get("L1-dcache-loads")
    return {
        "cpu_cycles": cyc,
        "instructions": ins,
        "ipc": _ratio(ins, cyc),
        "cache_misses": miss,
        "cache_references": ref,
        "cache_miss_rate": _ratio(miss, ref),
        "L1_dcache_miss_rate": _ratio(l1m, l1r),
    }


def _clean_frame(rest: str) -> str:
    """Trim a report_sample frame suffix into a short readable label.

    Inputs look like ``execve (/apex/.../libc.so)`` or
    ``[kernel.kallsyms][+ffffff...] ([kernel.kallsyms])``.
    """
    rest = rest.strip()
    dso = ""
    if rest.endswith(")"):
        k = rest.rfind(" (")
        if k > 0:
            dso = rest[k + 2:-1]
            rest = rest[:k].strip()
    if rest.startswith("[kernel.kallsyms]"):
        # Kernel frames: keep a stable coarse label (symbol not available).
        rest = "[kernel]"
    rest = _KERNEL_ADDR.sub("[k]", rest)
    rest = _OFFSET.sub("", rest).strip()
    rest = _UNKNOWN_FRAME.sub("[unknown]", rest)
    if not rest and dso:
        rest = dso.rsplit("/", 1)[-1]
    # Collapse to a bounded length for labels/aggregation readability.
    return rest[:160]


def parse_samples(text: str):
    """Yield (pid, count, frames) per sampled stack (frames leaf-first)."""
    pid = count = None
    frames = []
    for line in text.splitlines():
        m = _SAMPLE_HEAD.match(line)
        if m:
            if frames:
                yield pid, count, frames
            pid, count = int(m.group(2)), int(m.group(5))
            frames = []
            continue
        if pid is None:
            continue
        fm = _FRAME.match(line)
        if fm:
            rest = fm.group(2).strip()
            if rest:
                frames.append(_clean_frame(rest))
    if frames:
        yield pid, count, frames


def fold_samples(text: str) -> Counter:
    """Fold per-sample stacks into root-first path counts (flamegraph input).

    report_sample lists frames leaf-first (PC first); flamegraphs root first,
    so paths are reversed.
    """
    folded = Counter()
    for pid, count, frames in parse_samples(text):
        if not frames:
            continue
        # Frequency sampling: each sample weighs 1 (report_sample's count
        # column can carry huge aggregate periods for catch-up samples, which
        # would otherwise skew shares).
        folded[";".join(reversed(frames))] += 1
    return folded


def parse_perf_samples(text: str):
    """Yield (pid, frames) per ``perf script`` sample (frames leaf-first).

    perf script lists the sampled PC first, then its callers; the fp walk
    leaves ``[unknown]`` frames where a middle frame kept no frame pointer —
    those are kept as stable placeholders so paths stay contiguous.
    """
    pid = None
    frames = []
    for line in text.splitlines():
        m = _HOST_HEAD.match(line)
        if m:
            if frames:
                yield pid, frames
            pid = int(m.group(2))
            frames = []
            continue
        if pid is None:
            continue
        fm = _FRAME.match(line)
        if fm:
            rest = _clean_frame(fm.group(2))
            if rest:
                frames.append(rest)
    if frames:
        yield pid, frames


def fold_perf_samples(text: str) -> Counter:
    """Fold ``perf script`` per-sample stacks into root-first paths."""
    folded = Counter()
    for _pid, frames in parse_perf_samples(text):
        if not frames:
            continue
        folded[";".join(reversed(frames))] += 1
    return folded


def top_from_frames(samples, limit: int = 25) -> list:
    """[(pct, leaf_symbol)] — self-time shares from sampled-PC (leaf) frames."""
    counts = Counter()
    for _pid, frames in samples:
        if frames:
            counts[frames[0]] += 1
    total = sum(counts.values()) or 1
    return [(round(n / total * 100, 2), name)
            for name, n in counts.most_common(limit)]


def top_symbols(text: str, limit: int = 25) -> list:
    """Parse a flat `--sort symbol` report into [(pct, name)] top entries."""
    rows = []
    for line in text.splitlines():
        m = _OVERHEAD_ROW.match(line)
        if not m:
            continue
        pct = float(m.group(1))
        name = m.group(2).strip()
        if name.startswith("[.] "):
            name = name[4:]
        if name.startswith("[kernel.kallsyms]"):
            name = "[kernel]"
        rows.append((pct, name))
    rows.sort(reverse=True)
    return rows[:limit]


# =====================================================================
# profile pipeline (adb/simpleperf first target)
# =====================================================================
import glob  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

from mobench import flamegraph  # noqa: E402
from mobench.adapters.base import AdapterError, get_adapter  # noqa: E402
from mobench.artifacts_hash import artifact_set_md5  # noqa: E402
from mobench.config import ConfigError, load_config  # noqa: E402
from mobench.io_utils import atomic_write_json, make_run_dir  # noqa: E402
from mobench.platform.base import BasePlatform, PlatformError, measurement_session  # noqa: E402

DEFAULT_EVENTS = ["cpu-cycles", "instructions", "cache-misses",
                  "cache-references", "L1-dcache-loads",
                  "L1-dcache-load-misses"]
DEFAULT_FREQ = 2000


def find_simpleperf_dir(ndk_dir=None) -> str:
    """Locate an NDK simpleperf checkout holding report_sample.py.

    Preference order avoids older ANDROID_NDK_HOME copies whose report_sample
    CLI differs (positional vs ``-i``): SDK ``~/Android/Sdk/ndk`` first, then
    ANDROID_HOME, then ANDROID_NDK_HOME. Returns the first match with the
    script present.
    """
    if ndk_dir and os.path.isfile(os.path.join(ndk_dir, "report_sample.py")):
        return ndk_dir
    if ndk_dir:
        return ndk_dir  # caller-supplied path is authoritative even if bare
    candidates = [os.path.expanduser("~/Android/Sdk/ndk/*/simpleperf"),
                  os.environ.get("ANDROID_HOME", "") + "/ndk/*/simpleperf",
                  os.environ.get("ANDROID_NDK_HOME", "") + "/simpleperf"]
    for pat in candidates:
        if not pat or "/simpleperf" not in pat:
            continue
        for d in sorted(glob.glob(pat), reverse=True):
            if os.path.isfile(os.path.join(d, "report_sample.py")):
                return d
    return ""


def _find_perf_candidates() -> list:
    """Perf binaries to probe: kernel-package tool dirs, then PATH.

    Ubuntu ships ``/usr/bin/perf`` as a wrapper that only works when a
    matching ``linux-tools-<uname -r>/`` tree exists — never the case for a
    WSL microsoft-standard kernel, so installed ``linux-tools-*`` trees are
    scanned first and the PATH wrapper last.
    """
    candidates = []
    for pat in ("/usr/lib/linux-tools-*/perf",
                "/usr/lib/linux-tools/*/perf"):
        candidates.extend(sorted(glob.glob(pat)))
    which = shutil.which("perf")
    if which:
        candidates.append(which)
    return candidates


def find_perf() -> str:
    """Return the first usable ``perf`` binary, else ``""``.

    A candidate is usable when it exists, is executable, and its
    ``--version`` probe returns 0 (the PATH wrapper exec-fails and is skipped).
    """
    for cand in _find_perf_candidates():
        if not (os.path.isfile(cand) and os.access(cand, os.X_OK)):
            continue
        try:
            proc = subprocess.run([cand, "--version"], capture_output=True,
                                  text=True, timeout=10)
            if proc.returncode == 0:
                return cand
        except (OSError, subprocess.SubprocessError):
            continue
    return ""


def mirror_bin_dir(bin_dir: str, symfs_root: str) -> str:
    """Mirror local bin products under the device dso path a symfs expects.

    report_sample resolves symbols by prepending --symfs to the recorded dso
    path (``/data/local/tmp/mobench/bin/libMNN.so``), so copy the shipped
    binaries into that mirrored tree.
    """
    target = os.path.join(symfs_root, "data/local/tmp/mobench/bin")
    os.makedirs(target, exist_ok=True)
    for name in os.listdir(bin_dir):
        src = os.path.join(bin_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(target, name))
    return target


def _pull(platform, remote, local):
    proc = platform._exec(["pull", remote, local], timeout=300)
    return proc.returncode == 0 and os.path.isfile(local)


def _report_sample(ndk_dir, symfs_root, perf_data):
    script = os.path.join(ndk_dir, "report_sample.py")
    proc = subprocess.run(
        [sys.executable, script, "--symfs", symfs_root, perf_data],
        capture_output=True, text=True, timeout=1200)
    return proc.stdout, proc.stderr


def _write_flamegraph(folded, out_dir, cfg, model, threads, freq, extra_note="") -> tuple[str, int]:
    """根据折叠调用栈写出 flamegraph.html，返回 (路径, 折叠样本总数)。"""
    forest = flamegraph.build_tree(folded)
    svg = flamegraph.render_flame_svg(forest)
    notes = f"{cfg['engine']} {model['id']} t{threads} freq={freq}Hz · {len(folded)} distinct paths"
    if extra_note:
        notes += f" · {extra_note}"
    doc = flamegraph.flame_html(svg, f"flame · {cfg['engine']} {model['id']} t{threads}", notes)
    flame_path = os.path.join(out_dir, "flamegraph.html")
    with open(flame_path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return flame_path, sum(folded.values())


def _capture_host(platform, perf_bin, cmd, freq, events, out_dir, perf_local) -> dict:
    """Host Linux perf record + stat + script 采样，返回统一捕获结构。"""
    def _run_tool(args):
        try:
            return subprocess.run(
                args, capture_output=True, text=True, timeout=1800, cwd=out_dir
            )
        except (OSError, subprocess.SubprocessError):
            return None

    pinned_cmd = platform.with_pin(cmd)
    record = _run_tool(
        [perf_bin, "record", "-g", "-F", str(freq), "-o", perf_local, "--"]
        + list(pinned_cmd)
    )
    if record is None or record.returncode != 0:
        detail = ((record.stderr or "") + (record.stdout or "")) if record else "spawn error"
        raise RuntimeError(
            "[step:execution] perf record failed rc=%s; %s"
            % (record.returncode if record else "n/a", detail.strip()[:300])
        )
    stat_text = ""
    stat = _run_tool([perf_bin, "stat", "-e", ",".join(events), "--"] + list(pinned_cmd))
    if stat is not None:
        stat_text = (stat.stdout or "") + (stat.stderr or "")
    script = _run_tool([perf_bin, "script", "-i", perf_local])
    samples_text = (script.stdout or "") if script else ""
    return {
        "stat_text": stat_text,
        "samples_text": samples_text,
        "top_text": "",
        "pulled": None,
        "samples_lost_hint": None,
    }


def _capture_adb(platform, cmd, freq, events, perf_local) -> dict:
    """Android simpleperf record + flat top + 拉取 perf.data，返回统一捕获结构。"""
    cap = platform.run_profiled(cmd, freq=freq, events=events, mmap_pages=16384)
    if cap["record_rc"] != 0:
        raise RuntimeError(
            f"[step:execution] simpleperf record failed rc={cap['record_rc']}; "
            f"sample a workload the engine accepts (raw log saved if any)."
        )
    try:
        proc = platform._exec(
            [
                "shell",
                f"simpleperf report -i {cap['perf_data']} --sort symbol --percent-limit 0.3",
            ],
            timeout=300,
        )
        top_text = proc.stdout or ""
    except Exception:
        top_text = ""
    pulled = _pull(platform, cap["perf_data"], perf_local)
    return {
        "stat_text": cap.get("stat_text") or "",
        "samples_text": "",
        "top_text": top_text,
        "pulled": pulled,
        "samples_lost_hint": cap.get("samples_lost_hint"),
    }


def profile_pipeline(cfg, model_id=None, threads=None, freq=DEFAULT_FREQ,
                     events=None, ndk_dir=None, json_mode=False,
                     _platform=None, _adapter=None):
    """Run one sampling profile pass across host (perf) or adb (simpleperf).

    Returns analysis metadata written under ``results_dir/profile_<ts>/``:
    flamegraph.html, top.txt, stat.txt, samples.txt, perf.data, profile.json.
    """
    events = events or list(DEFAULT_EVENTS)
    platform = _platform or BasePlatform.make(cfg)
    is_host = platform.name == "host"
    try:
        info = platform.preflight()
    except PlatformError as exc:
        raise RuntimeError("[step:platform preflight] " + str(exc))

    perf_bin = None
    if is_host:
        perf_bin = find_perf()
        if not perf_bin:
            raise RuntimeError(
                "[step:tool] linux perf not found; install linux-tools "
                "(e.g. sudo apt-get install linux-tools-generic)"
            )

    adapter = _adapter or get_adapter(cfg)
    try:
        engine_version = adapter.version()
        binary = adapter.resolve_binary()
    except AdapterError as exc:
        raise RuntimeError("[step:binary probe] " + str(exc))

    models = cfg["models"]
    if model_id:
        models = [m for m in models if m["id"] == model_id]
    if not models:
        raise RuntimeError("[step:model check] no model matches filter")
    m = models[0]
    threads = threads or (cfg.get("workload") or {}).get("threads", [4])[0]
    cmd = adapter.build_cmds(m, threads, rounds=1)[0]

    # Host perf runs with cwd=out_dir, so require an absolute output dir there.
    raw_results = cfg.get("results_dir", "results")
    results_dir = os.path.abspath(raw_results) if is_host else raw_results
    ts, out_dir = make_run_dir(results_dir, "profile")
    perf_local = os.path.join(out_dir, "perf.data")

    try:
        with measurement_session(platform, cfg, threads, info=info) as condition:
            if is_host:
                cap = _capture_host(
                    platform, perf_bin, cmd, freq, events, out_dir, perf_local
                )
            else:
                cap = _capture_adb(platform, cmd, freq, events, perf_local)
    except PlatformError as exc:
        raise RuntimeError("[step:env control] " + str(exc))

    binary_info = (
        {"path": binary, "md5": artifact_set_md5(binary), "md5_scope": "artifact_set(binary+*.so*)"}
        if os.path.isfile(binary)
        else {"path": binary, "md5": platform.file_md5(binary)}
    )
    meta = {
        "schema": "mobench/profile@1",
        "timestamp": ts,
        "engine": cfg["engine"],
        "engine_version": engine_version,
        "binary": binary_info,
        "model": m["id"],
        "threads": threads,
        "freq": freq,
        "events": events,
        "platform": info,
        "condition": condition,
        "out_dir": out_dir,
    }
    if is_host:
        meta["perf"] = perf_bin
        meta["perf_data"] = perf_local
    else:
        meta["samples_lost_hint"] = cap.get("samples_lost_hint")
        meta["perf_data_pulled"] = cap.get("pulled")

    stat_text = cap["stat_text"]
    with open(os.path.join(out_dir, "stat.txt"), "w", encoding="utf-8") as fh:
        fh.write(stat_text or "(stat not collected)")
    if stat_text:
        counts = parse_perf_stat(stat_text) if is_host else parse_stat(stat_text)
        metric = metrics_from_stat(counts)
        metric["counts"] = dict(counts)
        meta["events_summary"] = metric

    if is_host:
        samples_text = cap["samples_text"]
        with open(os.path.join(out_dir, "samples.txt"), "w", encoding="utf-8") as fh:
            fh.write(samples_text)
        folded = fold_perf_samples(samples_text)
        if folded:
            top = top_from_frames(parse_perf_samples(samples_text), limit=10)
            meta["top"] = [{"pct": p, "name": n} for p, n in top]
            with open(os.path.join(out_dir, "top.txt"), "w", encoding="utf-8") as fh:
                for p, n in top:
                    fh.write(f"{p:6.2f}%  {n}\n")
            flame_path, n_folded = _write_flamegraph(
                folded, out_dir, cfg, m, threads, freq
            )
            meta["n_samples_folded"] = n_folded
            meta["flamegraph"] = flame_path
        else:
            meta["flame_skipped"] = "no call stacks folded from perf script"
    else:
        top_text = cap["top_text"]
        with open(os.path.join(out_dir, "top.txt"), "w", encoding="utf-8") as fh:
            fh.write(top_text)
        if top_text:
            meta["top"] = [
                {"pct": p, "name": n} for p, n in top_symbols(top_text, limit=10)
            ]
        ndk = find_simpleperf_dir(ndk_dir)
        if cap["pulled"] and ndk:
            symfs = os.path.join(out_dir, "symfs")
            mirror_bin_dir((cfg.get("build") or {}).get("bin_dir") or "", symfs)
            samples, err = _report_sample(ndk, symfs, perf_local)
            with open(os.path.join(out_dir, "samples.txt"), "w", encoding="utf-8") as fh:
                fh.write(samples)
            folded = fold_samples(samples)
            if folded:
                extra = f"samples_hint={cap.get('samples_lost_hint') or 'ok'}"
                flame_path, n_folded = _write_flamegraph(
                    folded, out_dir, cfg, m, threads, freq, extra_note=extra
                )
                meta["n_samples_folded"] = n_folded
                meta["flamegraph"] = flame_path
            meta["report_sample_stderr"] = err[:400] if err else ""
        else:
            meta["flamegraph"] = ""
            meta["flame_skipped"] = (
                "ndk simpleperf not found; give --ndk-dir"
                if not ndk
                else "perf.data not pulled"
            )

    atomic_write_json(os.path.join(out_dir, "profile.json"), meta)
    return meta


def main(args) -> int:
    try:
        cfg = load_config(args.config)
        meta = profile_pipeline(cfg, model_id=args.model, threads=args.threads,
                                freq=args.freq, events=args.events,
                                ndk_dir=args.ndk_dir, json_mode=args.json)
    except (ConfigError, AdapterError, PlatformError, RuntimeError) as exc:
        print(f"profile: error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(meta, indent=2, ensure_ascii=False))
    else:
        print(f"[profile] ok: {meta['out_dir']}"
              + (f" · flame {meta.get('flamegraph')}" if meta.get("flamegraph")
                 else " · flame skipped"))
    return 0
