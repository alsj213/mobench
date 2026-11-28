"""Multi-run statistics over canonical records (spec: methodology 多轮统计与方差报告).

Groups records by (engine, workload_id, metric, threads, precision) and reports
mean/std/CV/P50/P90/P99 per group over its rounds. CV above a threshold flags
high variance. Groups with <5 rounds are marked insufficient for formal stats.
"""
from __future__ import annotations

import json
import statistics

from mobench.records import empty_result_hint, group_key, read_paths

CV_WARN_THRESHOLD = 5.0
MIN_RUNS = 5


def percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def summarize(values):
    if not values:
        return None
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) >= 2 else 0.0
    cv = (std / mean * 100.0) if mean else 0.0
    s = sorted(values)
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "cv": round(cv, 2),
        "p50": round(percentile(s, 50), 4),
        "p90": round(percentile(s, 90), 4),
        "p99": round(percentile(s, 99), 4),
    }


def build_rows(records, cv_threshold=CV_WARN_THRESHOLD):
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
        vals = [r["value"] for r in recs]
        summ = summarize(vals)
        row = {
            "engine": key[0], "workload_id": key[1], "metric": key[2],
            "threads": key[3], "precision": key[4],
            "backend": recs[0].get("backend"),
            "platform": recs[0].get("platform"),
            "params": recs[0].get("params"),
            "n": len(vals),
            "samples_min": len(vals) >= MIN_RUNS,
            "stats": summ,
            "high_variance": bool(summ and summ["cv"] > cv_threshold),
        }
        rows.append(row)
    return rows


def _param_label(params):
    if not isinstance(params, dict) or not params:
        return "-"
    return " ".join(f"{k}={v}" for k, v in sorted(params.items()))


def _dash(value):
    return "-" if value in (None, "") else value


def render_markdown(rows):
    lines = ["## 多轮统计结果", ""]
    lines.append("| engine | backend | platform | workload | metric | threads | precision | params | n | mean | std | CV% | P50 | P90 | P99 | 方差 |")
    lines.append("|--------|---------|----------|----------|--------|---------|-----------|--------|----|------|-----|-----|-----|-----|-----|------|")
    for r in rows:
        s = r["stats"] or {}
        warn = "**HIGH**" if r["high_variance"] else ("insufficient" if not r["samples_min"] else "ok")
        lines.append(
            f"| {r['engine']} | {_dash(r.get('backend'))} | {_dash(r.get('platform'))} | "
            f"{r['workload_id']} | {r['metric']} | {r['threads']} | "
            f"{r['precision']} | {_param_label(r.get('params'))} | {r['n']} | "
            f"{s.get('mean', 0):.2f} | {s.get('std', 0):.2f} | "
            f"{s.get('cv', 0):.1f} | {s.get('p50', 0):.2f} | {s.get('p90', 0):.2f} | "
            f"{s.get('p99', 0):.2f} | {warn} |")
    lines.append("")
    for r in rows:
        if r["high_variance"]:
            lines.append(
                f"> ⚠️ **High variance (CV={r['stats']['cv']:.1f}%)** for "
                f"{r['engine']} {r['workload_id']} {r['metric']} (n={r['n']}). "
                f"Check thermal throttling / background processes / env control.")
        elif not r["samples_min"]:
            lines.append(
                f"> ℹ️ {r['engine']} {r['workload_id']} {r['metric']}: only {r['n']} run(s); "
                f">= {MIN_RUNS} required for formal statistics.")
    return "\n".join(lines) + "\n"


def main(args) -> int:
    records = read_paths(args.inputs, include_invalid=getattr(args, "include_invalid", False))
    if not records:
        print("stats: error: no records found "
              + empty_result_hint(args.inputs), file=__import__("sys").stderr)
        return 1
    rows = build_rows(records, cv_threshold=args.cv_threshold)
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(render_markdown(rows))
    return 0
