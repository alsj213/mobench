"""System-level CPU frequency trace (atrace) — power/frequency behaviour.

Captures atrace (freq category) while the engine runs, then derives a per-core
CPU frequency timeline, averages/minima and drop detection. All figures trace
to the captured text (redline R3).
"""
from __future__ import annotations

import copy
import csv
import datetime
import html
import json
import os
import re
import sys
import time

from mobench.adapters.base import AdapterError, get_adapter
from mobench.config import ConfigError, load_config
from mobench.platform.base import BasePlatform, PlatformError, measurement_session

_HEAD = re.compile(r"\[(\d+)\]\s+\S+\s+([0-9.]+):")
_STATE_FIRST = re.compile(r"state=(\d+).*?cpu_id=(\d+)")
_CPU_FIRST = re.compile(r"cpu_id=(\d+).*?state=(\d+)")


def parse_freq_events(text: str):
    """Yield (t_sec, cpu, hz) from an atrace capture."""
    base = None
    for line in text.splitlines():
        i = line.find("cpu_frequency")
        if i < 0:
            continue
        m = _HEAD.search(line[:i])
        if not m:
            continue
        ts = float(m.group(2))
        sm = (_STATE_FIRST.search(line[i:]) or _CPU_FIRST.search(line[i:]))
        if not sm:
            continue
        # sm may match cpu_id= then state= or state= then cpu_id=; decide by index
        f = sm.start()
        if "state=" in line[i:i + 200] and "cpu_id=" in line[i:i + 200]:
            si = line.find("state=", i)
            ci = line.find("cpu_id=", i)
            state = int(re.search(r"state=(\d+)", line[si:]).group(1))
            cid = int(re.search(r"cpu_id=(\d+)", line[ci:]).group(1))
        else:
            continue
        if base is None:
            base = ts
        yield ts - base, cid, state


def freq_timeline(events):
    """Group events per cpu -> [(t, hz)]; extrapolate each core's current freq."""
    per = {}
    for t, cpu, hz in events:
        per.setdefault(cpu, []).append((t, hz))
    return per


def summarize_freq(per_cpu, window=None) -> dict:
    """Per-core mean/min/max and share of samples below 60% of that core's max.

    The capture window includes idle before/after the engine, so a low-freq
    share is expected; treat it as a hint, not an automatic throttle verdict
    (active-window filtering is left as a follow-up).
    """
    out = {}
    overall_max = max((max(v for _, v in seq) for seq in per_cpu.values()),
                      default=0)
    for cpu in sorted(per_cpu):
        seq = per_cpu[cpu]
        hzs = [h for _, h in seq]
        mn = min(hzs) if hzs else 0
        mx = max(hzs) if hzs else 0
        low = sum(1 for h in hzs if mx and h < 0.6 * mx)
        out[cpu] = {"n": len(hzs), "mean_mhz": int(sum(hzs) / len(hzs) / 1000)
                    if hzs else 0, "min_mhz": mn // 1000, "max_mhz": mx // 1000,
                    "low_freq_share": round(low / len(hzs), 2) if hzs else None}
    return {"cpu": out, "overall_max_mhz": overall_max // 1000}


def _html(meta, per_cpu, out_dir) -> str:
    lines = ""
    colors = ["#4f46e5", "#d97706", "#0d9488", "#dc2626", "#7c3aed",
              "#0891b2", "#65a30d", "#db2777"]
    for i, cpu in enumerate(sorted(per_cpu)):
        seq = per_cpu[cpu]
        if len(seq) < 2:
            continue
        x0, x1 = seq[0][0], seq[-1][0]
        y1 = max(h for _, h in seq) or 1
        pts = " ".join("%.1f,%.1f" % (40 + (t - x0) / (x1 - x0 or 1) * 700,
                                      120 - h / y1 * 100) for t, h in seq)
        c = colors[i % len(colors)]
        lines += (f'<polyline points="{pts}" fill="none" stroke="{c}" '
                  f'stroke-width="1.6"/>')
        lines += (f'<text x="744" y="{120 - seq[-1][1] / y1 * 100:.0f}" '
                  f'font-size="9" fill="{c}">cpu{cpu}</text>')
    return f"""<!doctype html><html><meta charset="utf-8"><title>freq trace</title>
<style>body{{margin:0;background:#eef1f6;color:#17202f;font:14px/1.5 -apple-system,"Segoe UI",Roboto,"PingFang SC",sans-serif}}
.wrap{{max-width:820px;margin:0 auto;padding:24px 18px 50px}}</style>
<div class="wrap"><h1>CPU 频率 · {html.escape(str(meta['engine']))} · {html.escape(str(meta['model']))} t{html.escape(str(meta['threads']))}</h1>
<p style="color:#5a6577">n_freq={meta.get('n_freq_events')}（采样窗含 idle; 低频占比参考用）</p>
<svg viewBox="0 0 760 140" style="max-width:100%">{lines}</svg></div></html>"""


def trace_pipeline(cfg, model_id=None, threads=None, dur=20.0,
                   cats="freq", prompt=None, gen=None,
                   json_mode=False, _platform=None, _adapter=None) -> dict:
    platform = _platform or BasePlatform.make(cfg)
    if not hasattr(platform, "spawn_trace"):
        raise RuntimeError("[step:platform] trace 首版仅支持 adb 平台")
    try:
        info = platform.preflight()
    except PlatformError as exc:
        raise RuntimeError("[step:platform preflight] " + str(exc))
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
    if prompt is not None or gen is not None:
        c = copy.deepcopy(cfg)
        wl = dict(c["workload"])
        if prompt is not None:
            wl["prompt_tokens"] = prompt
        if gen is not None:
            wl["gen_tokens"] = gen
        c["workload"] = wl
        adapter = get_adapter(c)
    cmd = adapter.build_cmds(m, threads, rounds=1)[0]

    results_dir = cfg.get("results_dir", "results")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(results_dir, f"trace_{ts}")
    try:
        with measurement_session(platform, cfg, threads, info=info) as condition:
            os.makedirs(out_dir, exist_ok=True)
            trace_out = platform.spawn_trace(dur, cats=cats)
            proc = platform.spawn_engine(cmd)
            time.sleep(dur)                 # let the engine run under the capture
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
            try:
                trace_out.wait(timeout=30)
            except Exception:
                trace_out.kill()
            text = platform.read_remote_file("/data/local/tmp/mobench/atrace.txt")
    except PlatformError as exc:
        raise RuntimeError("[step:env control] " + str(exc))
    with open(os.path.join(out_dir, "atrace.txt"), "w",
              encoding="utf-8", errors="replace") as fh:
        fh.write(text)

    events = list(parse_freq_events(text))
    per_cpu = freq_timeline(events)
    summ = summarize_freq(per_cpu)
    meta = {
        "schema": "mobench/trace@1", "timestamp": ts,
        "engine": cfg["engine"], "engine_version": engine_version,
        "binary": {"path": binary, "md5": platform.file_md5(binary)},
        "model": m["id"], "threads": threads,
        "platform": info, "condition": condition, "n_freq_events": len(events),
        "source_log": os.path.join(out_dir, "atrace.txt"),
        "note": "采样窗含 idle 前后段; low_freq_share 含 idle, 非降频判定",
        **summ,
    }
    csv_path = os.path.join(out_dir, "freq.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_sec", "cpu", "hz"])
        w.writerows((round(t, 4), c, h) for t, c, h in events)
    html_path = os.path.join(out_dir, "trace.html")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(_html(meta, per_cpu, out_dir))
    meta["trace_html"] = html_path
    with open(os.path.join(out_dir, "trace.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    return meta


def main(args) -> int:
    try:
        cfg = load_config(args.config)
        meta = trace_pipeline(cfg, model_id=args.model, threads=args.threads,
                              dur=args.dur, cats=" ".join(args.atrace_cats),
                              prompt=args.prompt, gen=args.gen,
                              json_mode=args.json)
    except (ConfigError, AdapterError, PlatformError, RuntimeError) as exc:
        print(f"trace: error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(meta, indent=2, ensure_ascii=False))
    else:
        print(f"[trace] ok: n_freq={meta['n_freq_events']} "
              f"overall_max={meta['overall_max_mhz']}MHz "
              f"-> {meta.get('trace_html')}")
    return 0
