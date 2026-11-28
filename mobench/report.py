"""Report generation from canonical records (spec: benchmark-core 执行流水线).

Markdown + HTML are produced only from records; every numeric cell can be
traced to its ``source_log``. Throttle risk (temp > 45C) is flagged when any
record carries an elevated device temperature.
"""
from __future__ import annotations

import html
import os
import statistics

from mobench.records import empty_result_hint, group_key, read_paths

TEMP_THROTTLE_C = 45.0


def _values(recs):
    return [r["value"] for r in recs]


def _compact_cpus(cpus):
    """``[4,5,6,7]`` -> ``"4-7"``; ``[4,5,7]`` -> ``"4,5,7"``."""
    cpus = sorted(cpus)
    if len(cpus) == 1:
        return str(cpus[0])
    if cpus == list(range(cpus[0], cpus[-1] + 1)):
        return f"{cpus[0]}-{cpus[-1]}"
    return ",".join(str(c) for c in cpus)


def condition_label(device):
    """Human label for the measurement condition a record was produced under.

    Every value must be able to state what produced it (R3), and an unpinned
    run must never read as a pinned one -- so the unavailable/disabled origins
    get their own wording rather than a blank cell.
    """
    if not isinstance(device, dict):
        return "-"
    pin = device.get("pin") or {}
    freq = device.get("freq") or {}
    cpus = pin.get("cpus")
    if cpus:
        lock = "freq锁" if freq.get("locked") else "freq未锁"
        return f"cpu{_compact_cpus(cpus)}·{lock}"
    origin = pin.get("origin")
    if origin == "unavailable":
        return "未绑核(cpufreq不可用)"
    if origin == "disabled":
        return "未绑核(已关闭)"
    if origin is None and not freq:
        return "-"          # legacy record: no condition information
    return "未绑核"


def summarize_rows(records):
    """Aggregate records by group_key (engine/workload/metric/threads/precision + params)."""
    groups = {}
    order = []
    for rec in records:
        key = group_key(rec)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(rec)
    rows = []
    for key in order:
        recs = groups[key]
        vals = _values(recs)
        mean = statistics.fmean(vals)
        row = {
            "engine": key[0], "workload_id": key[1], "metric": key[2],
            "threads": key[3], "precision": key[4],
            "backend": recs[0].get("backend"),
            "platform": recs[0].get("platform"),
            "params": recs[0].get("params"),
            "condition": condition_label(recs[0].get("device")),
            "n": len(recs),
            "mean": round(mean, 6),
            "min": round(min(vals), 6),
            "max": round(max(vals), 6),
            "source_log": recs[0].get("source_log"),
        }
        if len(vals) >= 2:
            row["cv_pct"] = round(statistics.stdev(vals) / mean * 100, 2) if mean else 0.0
        else:
            row["cv_pct"] = None
        rows.append(row)
    return rows


def _param_label(params):
    if not isinstance(params, dict) or not params:
        return "-"
    return " ".join(f"{k}={v}" for k, v in sorted(params.items()))


def _src_label(row):
    """Source column: first log anchored; multi-round groups note the total."""
    base = (row.get("source_log") or "").split("/")[-1]
    if row.get("n") and row["n"] > 1:
        return f"{base} · 共 {row['n']} 轮"
    return base


def throttle_risk(records):
    for r in records:
        dev = r.get("device")
        if not isinstance(dev, dict):
            continue
        for key in ("temp_c", "temp_after_c", "temp_max_c"):
            val = dev.get(key)
            if isinstance(val, (int, float)) and val > TEMP_THROTTLE_C:
                return True
    return False


def render_markdown(rows, records):
    lines = ["## mobench 评测报告", ""]
    if throttle_risk(records):
        lines.append("> ⚠️ **降频风险**: 存在 device temp > 45°C 的记录，建议冷却后重跑")
        lines.append("")
    if not rows:
        lines.append("无可用记录。")
        return "\n".join(lines) + "\n"
    lines.append("| engine | workload | metric | threads | precision | params | 测量条件 | n | mean | CV% | 来源日志 |")
    lines.append("|--------|----------|--------|---------|-----------|--------|----------|---|------|-----|----------|")
    for r in rows:
        cv = f"{r['cv_pct']:.2f}" if r["cv_pct"] is not None else "-"
        src = _src_label(r)
        lines.append(
            f"| {r['engine']} | {r['workload_id']} | {r['metric']} | "
            f"{r['threads']} | {r['precision']} | {_param_label(r.get('params'))} | "
            f"{r.get('condition', '-')} | {r['n']} | {r['mean']} | {cv} | {src} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def render_html(rows, records):
    if throttle_risk(records):
        banner = ('<p class="warn">⚠️ 降频风险: 存在 device temp &gt; 45°C 的记录，'
                  '建议冷却后重跑</p>')
    else:
        banner = ""
    thead = ("<tr><th>engine</th><th>workload</th><th>metric</th><th>threads</th>"
             "<th>precision</th><th>params</th><th>测量条件</th><th>n</th><th>mean</th>"
             "<th>CV%</th><th>来源</th></tr>")
    body = ""
    for r in rows:
        cv = f"{r['cv_pct']:.2f}" if r["cv_pct"] is not None else "-"
        src = html.escape(_src_label(r))
        body += (
            f"<tr><td>{html.escape(str(r['engine']))}</td>"
            f"<td>{html.escape(str(r['workload_id']))}</td>"
            f"<td>{html.escape(str(r['metric']))}</td><td>{r['threads']}</td>"
            f"<td>{html.escape(str(r['precision']))}</td>"
            f"<td>{html.escape(_param_label(r.get('params')))}</td>"
            f"<td>{html.escape(str(r.get('condition', '-')))}</td><td>{r['n']}</td>"
            f"<td>{r['mean']}</td><td>{cv}</td><td>{src}</td></tr>"
        )
    return (
        "<!DOCTYPE html>\n<html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<title>mobench 评测报告</title><style>"
        "body{font-family:system-ui;margin:24px;color:#222}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:6px 10px;text-align:left}"
        "th{background:#1a1a2e;color:#fff}.warn{color:#721c24;background:#f8d7da;padding:8px;border-radius:4px}"
        "</style></head><body><h1>mobench 评测报告</h1>"
        + banner +
        "<table><thead>" + thead + "</thead><tbody>" + body + "</tbody></table>"
        "</body></html>"
    )


CHART_METRICS = {"gen_tok_s": "Decode", "prompt_eval_ms_tok": "Prefill"}
PALETTE = ["#0b6ea8", "#a85a00", "#177a3d", "#8e44ad", "#b3261e", "#2c3a4a"]


def _curve_label(rec):
    """Curve label: backend name (cpu/opencl/…) when known, else T<threads>."""
    b = rec.get("backend")
    if b is not None:
        return str(b)
    return f"T{rec.get('threads')}"


def _label_sort_key(lbl):
    if isinstance(lbl, str) and lbl.startswith("T") and lbl[1:].isdigit():
        return (0, int(lbl[1:]))
    return (1, str(lbl))


def _collect_chart(records, by="backend"):
    """Panel model: (platform, workload, stage) -> series -> x -> points.

    ``by`` selects the comparison dimension inside a panel:
      backend - one bar per engine backend (same engine); when several engines
               share a panel, series are prefixed with the engine to avoid merging.
      engine  - one bar per engine (cross-engine, same backend comparison).
      matrix  - one bar per engine x backend pair.
    """
    panels = {}
    for rec in records:
        stage = CHART_METRICS.get(rec.get("metric"))
        if not stage:
            continue
        dev = rec.get("device") or {}
        plat = dev.get("platform") or rec.get("platform") or "host"
        params = rec.get("params") or {}
        x = params.get("prompt_tokens") if isinstance(params, dict) and params.get("prompt_tokens") is not None else "default"
        key = (plat, rec.get("workload_id"), stage)
        eng = rec.get("engine") or "?"
        be = rec.get("backend")
        th = rec.get("threads")
        raw = (eng, be, th)
        pt = (panels.setdefault(key, {}).setdefault(raw, {}).setdefault(
            x, {"vals": [], "src": None, "rounds": set(), "precision": None,
                "device": None}))
        pt["vals"].append(rec["value"])
        pt["src"] = pt["src"] or rec.get("source_log")
        if pt["precision"] is None and rec.get("precision") is not None:
            pt["precision"] = rec.get("precision")
        if pt["device"] is None and rec.get("device") is not None:
            pt["device"] = rec.get("device")
        if rec.get("round") is not None:
            pt["rounds"].add(rec["round"])
    out = []
    for (plat, wl, stage), series in panels.items():
        engines = {r[0] for r in series}
        multi_engine = len(engines) > 1
        def _base_label(eng, be, th):
            b = be if be is not None else f"T{th}"
            if by == "engine":
                return eng
            if by == "matrix":
                return f"{eng}:{b}"
            # backend view: disambiguate when several engines share the panel
            return f"{eng}·{b}" if multi_engine else b

        raw_items = list(series.items())
        base_labels = [_base_label(eng, be, th) for (eng, be, th), _ in raw_items]
        label_counts = {}
        for bl in base_labels:
            label_counts[bl] = label_counts.get(bl, 0) + 1

        series_out = {}
        for ((eng, be, th), xs), base_lbl in zip(raw_items, base_labels):
            lbl = base_lbl
            if label_counts[base_lbl] > 1:
                # 同一面板多条系列撞名时，按缺失维度（backend / threads）自动消歧，禁止静默覆盖
                suffix_parts = []
                if by == "engine" and be is not None:
                    suffix_parts.append(str(be))
                if th is not None:
                    suffix_parts.append(f"T{th}")
                if suffix_parts:
                    lbl = f"{base_lbl}·{'·'.join(suffix_parts)}"
                while lbl in series_out:
                    lbl = f"{lbl}*"
            points = []
            for x, pt in sorted(xs.items(),
                                key=lambda kv: (0 if kv[0] == "default" else int(kv[0]))):
                vals = pt["vals"]
                mean = statistics.fmean(vals)
                y = mean if stage == "Decode" else (1000.0 / mean if mean else 0.0)
                cv = (round(statistics.stdev(vals) / mean * 100, 2)
                      if len(vals) >= 2 and mean else None)
                points.append({"x": x, "y": round(y, 3), "mean": round(mean, 4),
                               "n": len(vals), "cv": cv, "src": pt["src"],
                               "precision": pt["precision"],
                               "condition": condition_label(pt["device"]),
                               "rounds": len(pt["rounds"])})
            series_out[lbl] = points
        out.append({"platform": plat, "workload": wl, "stage": stage,
                    "series": series_out})
    return out


def _panel_svg(panel, color_map):
    """Grouped bar panel: x = prompt length, one bar per backend/thread color."""
    pad_l, pad_r, pad_t, pad_b = 62, 14, 18, 52
    W, H = 560, 258
    cats = sorted({p["x"] for ps in panel["series"].values() for p in ps},
                  key=lambda x: (0 if x == "default" else int(x)))
    labels = sorted(panel["series"], key=_label_sort_key)
    ymax = max((p["y"] for ps in panel["series"].values() for p in ps), default=1) or 1
    ymax *= 1.1
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b
    def sy(y):
        return pad_t + (1 - y / ymax) * plot_h
    cat_w = plot_w / max(len(cats), 1)
    parts = []
    parts.append(f'<line x1="{pad_l}" y1="{H-pad_b}" x2="{W-pad_r}" y2="{H-pad_b}" stroke="currentColor"/>')
    for i in range(5):
        g = ymax / 4 * i
        yy = sy(g)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{W-pad_r}" y2="{yy:.1f}" stroke="currentColor" opacity="0.12"/>')
        parts.append(f'<text x="{pad_l-6}" y="{yy+4:.1f}" text-anchor="end" font-size="10">{g:.0f}</text>')
    # axis titles
    parts.append(f'<text x="{pad_l+plot_w/2:.1f}" y="{H-6}" text-anchor="middle" '
                 'font-size="11" fill="currentColor">Prompt Length</text>')
    label_y = pad_t + plot_h / 2
    parts.append(f'<text x="12" y="{label_y:.1f}" text-anchor="middle" font-size="11" '
                 f'fill="currentColor" transform="rotate(-90 12 {label_y:.1f})">tok/s</text>')
    slot = cat_w / max(len(labels), 1)
    bw = min(46.0, slot * 0.62)
    for ci, x in enumerate(cats):
        cx = pad_l + cat_w * (ci + 0.5)
        lbl = str(x) if x != "default" else "default"
        parts.append(f'<text x="{cx:.1f}" y="{H-pad_b+16}" text-anchor="middle" font-size="10">{html.escape(lbl)}</text>')
        n = len(labels)
        for j, lab in enumerate(labels):
            pts = {p["x"]: p for p in panel["series"][lab]}
            pt = pts.get(x)
            if pt is None:
                continue
            bx = cx + (j - (n - 1) / 2) * slot
            y0 = sy(0.0)
            y1 = sy(pt["y"])
            col = color_map[lab]
            tip = (f"{lab} x={x} y={pt['y']} tok/s n={pt['n']} "
                   f"cv={pt['cv']}% mean={pt['mean']} src={pt['src'] or ''}")
            parts.append(
                f'<rect x="{bx-bw/2:.1f}" y="{y1:.1f}" width="{bw:.1f}" '
                f'height="{max(y0-y1,0):.1f}" fill="{col}">'
                f'<title>{html.escape(tip)}</title></rect>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" '
            f'aria-label="{html.escape(panel["workload"] + " " + panel["stage"])}">'
            + "".join(parts) + "</svg>")


def render_chart_html(records, by="backend"):
    """Self-contained grouped-bar multipanel chart (platform x model x stage)."""
    css = (
        ":root{--bg:#f5f7fa;--card:#fff;--ink:#17202b;--mut:#5c6b7a;--line:#dfe5ec;"
        "--acc:#0b6ea8}"
        "@media (prefers-color-scheme:dark){:root:not([data-theme='light']){"
        "--bg:#0f141b;--card:#161d27;--ink:#e8edf3;--mut:#8b99a8;--line:#2a3441;--acc:#5cb1e8}}"
        "body{font-family:'IBM Plex Sans',system-ui,sans-serif;background:var(--bg);"
        "color:var(--ink);margin:24px}"
        "h1{font-size:18px}h2{font-size:15px;margin:18px 0 6px}"
        ".srow{display:grid;grid-template-columns:64px repeat(auto-fit,minmax(300px,1fr));"
        "gap:12px;align-items:start}"
        ".stage-label{font-size:12px;color:var(--mut);padding-top:8px;font-weight:600}"
        ".panel{border:1px solid var(--line);border-radius:8px;padding:8px;background:var(--card);min-width:0}"
        ".panel h3{font-size:13px;margin:0 0 2px}.dim{color:var(--mut);font-size:12px;padding:20px 8px}"
        "svg{width:100%;height:auto;color:var(--mut)}"
        ".note{color:#b3261e;background:#fbe7e5;padding:10px;border-radius:6px}"
        ".legend{font-size:11px;color:var(--mut);margin:2px 0 4px}"
        "table{border-collapse:collapse;width:100%;margin-top:16px;font-size:12px}"
        "th,td{border:1px solid var(--line);padding:4px 6px;text-align:left}"
    )
    head = ('<!DOCTYPE html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>mobench 多面板基准图</title><style>' + css + "</style></head><body>"
            "<h1>mobench 多面板基准图</h1>")
    if not any(rec.get("metric") in CHART_METRICS for rec in records):
        return head + ('<p class="note">记录不含可作图指标（需要 gen_tok_s / '
                       'prompt_eval_ms_tok），未生成面板。</p></body></html>')
    data = _collect_chart(records, by=by)
    if not data:
        return head + '<p class="note">无可用数据。</p></body></html>'
    labels_all = sorted({l for p in data for l in p["series"]}, key=_label_sort_key)
    color_map = {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(labels_all)}
    platforms = sorted({p["platform"] for p in data})
    body = []
    for pf in platforms:
        body.append(f'<h2>{html.escape(pf)}</h2>')
        models = sorted({p["workload"] for p in data if p["platform"] == pf})
        for stage in ("Prefill", "Decode"):
            cells = [f'<div class="stage-label">{stage}</div>']
            for md in models:
                pan = next((p for p in data
                            if p["platform"] == pf and p["workload"] == md
                            and p["stage"] == stage), None)
                if pan is None:
                    cells.append(f'<div class="panel dim">无 {html.escape(md)} {stage} 数据</div>')
                    continue
                legend = "".join(
                    f'<span style="color:{color_map[l]}"> {html.escape(l)}</span>'
                    for l in sorted(pan["series"], key=_label_sort_key))
                cells.append(f'<div class="panel"><h3>{html.escape(md)}</h3>'
                             f'<div class="legend">{legend}</div>{_panel_svg(pan, color_map)}</div>')
            body.append('<div class="srow">' + "".join(cells) + "</div>")
    rows = []
    for panel in data:
        for lbl, pts in sorted(panel["series"].items(),
                               key=lambda kv: _label_sort_key(kv[0])):
            for p in pts:
                cv = f"{p['cv']}" if p["cv"] is not None else "-"
                rows.append(
                    f"<tr><td>{html.escape(panel['platform'])}</td>"
                    f"<td>{html.escape(panel['workload'])}</td><td>{panel['stage']}</td>"
                    f"<td>{html.escape(str(lbl))}</td>"
                    f"<td>{html.escape(str(p.get('precision') or ''))}</td>"
                    f"<td>{html.escape(str(p.get('condition') or '-'))}</td>"
                    f"<td>{html.escape(str(p['x']))}</td><td>{p['y']}</td><td>{p['n']}</td>"
                    f"<td>{p['mean']}</td><td>{cv}</td>"
                    f"<td>{html.escape(str(p['src'] or ''))}"
                    + (f" · 共 {p['rounds']} 轮" if p["rounds"] > 1 else "") + "</td></tr>")
    table = ("<h2>溯源明细</h2><table><tr><th>platform</th><th>workload</th><th>stage</th>"
             "<th>曲线</th><th>precision</th><th>测量条件</th><th>prompt</th><th>tok/s</th>"
             "<th>n</th><th>mean</th><th>CV%</th><th>来源</th></tr>"
             + "".join(rows) + "</table>")
    return head + "".join(body) + table + "</body></html>"


def main(args) -> int:
    records = read_paths(args.inputs, include_invalid=getattr(args, "include_invalid", False))
    if not records:
        print("report: error: no records found in inputs "
              + empty_result_hint(args.inputs), file=__import__("sys").stderr)
        return 1
    rows = summarize_rows(records)
    out_dir = args.out or "results"
    os.makedirs(out_dir, exist_ok=True)
    if args.format in ("md", "both"):
        md_path = os.path.join(out_dir, "benchmark_report.md")
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(render_markdown(rows, records))
        print(f"Markdown report: {md_path}")
    if args.format in ("html", "both"):
        html_path = os.path.join(out_dir, "benchmark_report.html")
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(render_html(rows, records))
        print(f"HTML report: {html_path}")
    if args.format == "chart":
        chart_path = os.path.join(out_dir, "benchmark_chart.html")
        with open(chart_path, "w", encoding="utf-8") as fh:
            fh.write(render_chart_html(records, by=args.chart_by))
        print(f"Chart report: {chart_path}")
    return 0
