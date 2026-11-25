"""Process-memory curve sampling (spec: benchmark-memory 命令化).

Samples the engine process RSS while it runs a (long) generation, then reports
peak/steady RSS and a growth curve. Every sample traces to a real
``/proc/<pid>/status`` read whose raw text is saved, so no value is fabricated.
"""
from __future__ import annotations

import copy
import csv
import datetime
import html
import json
import os
import sys
import time

from mobench.adapters.base import AdapterError, get_adapter
from mobench.config import ConfigError, load_config
from mobench.platform.base import BasePlatform, PlatformError

DEFAULT_INTERVAL = 0.2
DEFAULT_MAX_SEC = 300


def parse_proc_status(text: str) -> dict:
    """Extract VmRSS/VmHWM (kB) from a /proc/<pid>/status dump."""
    out = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        rest = rest.strip().split()[0] if rest.strip() else None
        key = key.strip()
        if key in ("VmRSS", "VmHWM", "RssAnon", "RssFile", "VmSize") and rest:
            try:
                out[key] = int(rest)
            except ValueError:
                pass
    return out


def rss_curve_html(samples, title: str) -> str:
    """Render (t_sec, rss_kb) samples as a small inline-SVG line chart."""
    if not samples:
        return "<p>no samples</p>"
    W, H = 720, 240
    xs = [s[0] for s in samples]
    ys = [s[1] for s in samples]
    x0, x1 = xs[0], xs[-1]
    y0, y1 = 0, max(ys) * 1.08 or 1
    pad_l, pad_b, pad_t = 64, 26, 12

    def px(i):
        t, v = samples[i]
        x = pad_l + (t - x0) / (x1 - x0 or 1) * (W - pad_l - 8)
        y = H - pad_b - (v - y0) / (y1 - y0) * (H - pad_b - pad_t)
        return x, y

    pts = " ".join("%.1f,%.1f" % px(i) for i in range(len(samples)))
    grid = ""
    for g in range(0, int(y1), max(1, int(y1 // 4))):
        y = H - pad_b - (g - y0) / (y1 - y0) * (H - pad_b - pad_t)
        grid += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W}" y2="{y:.1f}" stroke="var(--grid)" />'
        grid += f'<text x="{pad_l - 6}" y="{y + 3:.1f}" text-anchor="end" font-size="10" fill="var(--mut)">{g}</text>'
    return f"""<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="max-width:100%">
<g>{grid}</g>
<polyline points="{pts}" fill="none" stroke="var(--ink)" stroke-width="2"/>
<text x="{W-8}" y="{H-8}" text-anchor="end" font-size="10" fill="var(--mut)">seconds</text>
<text x="10" y="14" font-size="12" font-weight="600" fill="var(--ink)">{html.escape(title)}</text>
</svg>"""


def _mem_html(svg, out_dir, peak, steady, engine, model):
    return f"""<!doctype html><html><meta charset="utf-8">
<title>{html.escape(engine)} memory curve</title>
<style>
:root{{--bg:#eef1f6;--ink:#17202f;--mut:#5a6577;--grid:#dbe2ec}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#0d1320;--ink:#e9eef7;--mut:#94a1b8;--grid:#26314a}}}}
:root[data-theme="dark"]{{--bg:#0d1320;--ink:#e9eef7;--mut:#94a1b8;--grid:#26314a}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,"Segoe UI",Roboto,"PingFang SC",sans-serif}}
.wrap{{max-width:860px;margin:0 auto;padding:26px 18px 50px}} h1{{font-size:20px;margin:0 0 10px}}
.kpis{{display:flex;gap:26px;margin:0 0 18px;flex-wrap:wrap}}
.kpi b{{font-size:22px;display:block}}
</style><div class="wrap"><h1>{html.escape(engine)} · {html.escape(model)} 内存曲线</h1>
<div class="kpis"><div class="kpi">峰值 RSS<b>{peak / 1024:.1f} MB</b></div>
<div class="kpi">稳态 RSS<b>{steady / 1024:.1f} MB</b></div></div>{svg}
<p style="color:var(--mut);font-size:12px">数据目录: <code>{html.escape(out_dir)}</code></p></div></html>"""


def mem_pipeline(cfg, model_id=None, threads=None, prompt=None, gen=None,
                 interval=DEFAULT_INTERVAL, max_sec=DEFAULT_MAX_SEC,
                 json_mode=False, _platform=None, _adapter=None) -> dict:
    platform = _platform or BasePlatform.make(cfg)
    if not (hasattr(platform, "spawn_engine")
            and hasattr(platform, "pidof")
            and hasattr(platform, "read_proc_status")):
        raise RuntimeError("[step:platform] memory 首版仅支持 adb 平台"
                           " (被测引擎在 adb 会话运行 + /proc/<pid> 轮询)")
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

    work = dict(cfg["workload"])
    if prompt is not None:
        work["prompt_tokens"] = prompt
    if gen is not None:
        work["gen_tokens"] = gen
    c = copy.deepcopy(cfg)
    c["workload"] = work
    cmd = get_adapter(c).build_cmds(m, threads, rounds=1)[0]
    # Same measurement condition as `run`/`profile` (design D5): the engine is
    # spawned through spawn_engine, which applies the pin transparently.
    try:
        condition = platform.measure_condition(threads)
    except PlatformError as exc:
        raise RuntimeError("[step:env control] " + str(exc))

    results_dir = cfg.get("results_dir", "results")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(results_dir, f"mem_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    proc = platform.spawn_engine(cmd)
    bench = (cfg.get("build") or {}).get("bench_bin", "llm_bench")
    samples = []          # (t_sec, rss_kb)
    raw_log = os.path.join(out_dir, "status_samples.log")
    start = time.monotonic()
    gone = 0              # consecutive polls without a pid before we stop
    try:
        with open(raw_log, "w", encoding="utf-8") as fh:
            while time.monotonic() - start <= max_sec:
                pid = platform.pidof(bench)
                if not pid:
                    gone += 1
                    if gone >= 25 or proc.poll() is not None:
                        break
                else:
                    gone = 0
                    text = platform.read_proc_status(pid.split()[0])
                    if text:
                        rss = parse_proc_status(text).get("VmRSS")
                        if rss is not None:
                            samples.append((time.monotonic() - start, rss))
                        fh.write("==== t=%.2f pid=%s ====\n%s\n"
                                 % (time.monotonic() - start, pid, text))
                time.sleep(interval)
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
    finally:
        # Never leave the device pinned to performance (design D4 restore).
        platform.restore()

    peak = max((s[1] for s in samples), default=0)

    def _median(vals):
        vals = sorted(vals)
        n = len(vals)
        return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2

    if len(samples) >= 8:
        _mids = [v for _, v in samples[len(samples) // 4: len(samples) // 2]]
        _ends = [v for _, v in samples[int(len(samples) * 0.8):]]
        mid_rss, end_rss = _median(_mids), _median(_ends)
        steady = int(end_rss)
        kv_growth = int(end_rss - mid_rss)   # decode-phase growth, load excluded
        # leak: monotone non-decreasing tail with real growth beyond noise
        tail = [v for _, v in samples[int(len(samples) * 0.85):]]
        monotone = all(b >= a for a, b in zip(tail, tail[1:]))
        possible_leak = bool(kv_growth > 10 * 1024 and monotone and len(tail) >= 5)
    else:
        steady = samples[-1][1] if samples else 0
        kv_growth = 0
        possible_leak = False
    growth = steady - (samples[0][1] if samples else 0)  # load+decode total

    csv_path = os.path.join(out_dir, "mem.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_sec", "rss_kb"])
        w.writerows((round(t, 3), v) for t, v in samples)

    meta = {
        "schema": "mobench/mem@1",
        "timestamp": ts,
        "engine": cfg["engine"],
        "engine_version": engine_version,
        "binary": {"path": binary, "md5": platform.file_md5(binary)},
        "model": m["id"], "threads": threads,
        "prompt_tokens": prompt, "gen_tokens": gen,
        "platform": info,
        "condition": condition,
        "n_samples": len(samples),
        "peak_rss_kb": peak, "steady_rss_kb": steady,
        "growth_kb": growth, "kv_growth_kb": kv_growth,
        "possible_leak": possible_leak,
        "source_log": raw_log,
        "out_dir": out_dir,
    }
    if samples:
        svg = rss_curve_html(samples, f"VmRSS · {cfg['engine']} · {m['id']} · t{threads}")
        with open(os.path.join(out_dir, "mem.html"), "w", encoding="utf-8") as fh:
            fh.write(_mem_html(svg, out_dir, peak, steady, cfg["engine"], m["id"]))
        meta["curve_html"] = os.path.join(out_dir, "mem.html")
    with open(os.path.join(out_dir, "mem.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    return meta


def main(args) -> int:
    try:
        cfg = load_config(args.config)
        meta = mem_pipeline(cfg, model_id=args.model, threads=args.threads,
                            prompt=args.prompt, gen=args.gen,
                            interval=args.interval, max_sec=args.max_sec,
                            json_mode=args.json)
    except (ConfigError, AdapterError, PlatformError, RuntimeError) as exc:
        print(f"memory: error: {exc}", file=sys.stderr)
        return 1
    leak = "  [LEAK?]" if meta.get("possible_leak") else ""
    if args.json:
        print(json.dumps(meta, indent=2, ensure_ascii=False))
    else:
        print(f"[memory] ok: n={meta['n_samples']} peak={meta['peak_rss_kb']/1024:.1f}MB "
              f"steady={meta['steady_rss_kb']/1024:.1f}MB kv_growth={meta['kv_growth_kb']/1024:.2f}MB{leak}")
    return 0
