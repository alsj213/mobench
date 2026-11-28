"""Baseline save / compare over canonical records (spec: benchmark-regression).

Baseline.json holds per (engine, workload_id, metric, threads, precision)
samples plus environment fields (engine version when a run.json is present).
Comparison reports Δ% and — with >=5 repeats on each side and scipy available —
a Mann-Whitney p-value. A statistically-significant regression beyond the
threshold emits ``REGRESSION DETECTED``.
"""
from __future__ import annotations

import datetime
import json
import os
import statistics

from mobench.io_utils import atomic_write_json
from mobench.records import group_key, read_paths

DEFAULT_THRESHOLD_PCT = 5.0
HIGHER_BETTER_MARKERS = ("tok_s", "fps", "throughput")


def is_higher_better(metric):
    return any(m in metric for m in HIGHER_BETTER_MARKERS)


def group_samples(records):
    groups = {}
    order = []
    for rec in records:
        key = group_key(rec)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(rec["value"])
    return {k: groups[k] for k in order}


def _find_run_meta(records):
    """Best-effort engine version/commit from an adjacent run.json."""
    for rec in records:
        src = rec.get("source_log")
        if not src:
            continue
        run_json = os.path.join(os.path.dirname(src), "run.json")
        if os.path.isfile(run_json):
            try:
                with open(run_json, encoding="utf-8") as fh:
                    meta = json.load(fh)
                return {
                    "engine_version": meta.get("engine_version"),
                    "timestamp": meta.get("timestamp"),
                    "binary_md5": (meta.get("binary") or {}).get("md5"),
                }
            except (OSError, ValueError):
                continue
    return {}


def save_baseline(baseline_path, records, extra=None):
    payload = {
        "schema": "mobench/baseline@1",
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "engine": records[0].get("engine") if records else None,
        **(extra or {}),
        "groups": {json.dumps(k, ensure_ascii=False): v for k, v in group_samples(records).items()},
    }
    atomic_write_json(baseline_path, payload)
    return payload


def _as_hashable(x):
    """JSON turns nested tuples (folded params/control key) into lists;
    restore them recursively so the key stays usable as a dict key."""
    if isinstance(x, list):
        return tuple(_as_hashable(i) for i in x)
    return x


def load_baseline(baseline_path):
    with open(baseline_path, encoding="utf-8") as fh:
        data = json.load(fh)
    groups = {}
    for key_json, values in (data.get("groups") or {}).items():
        groups[_as_hashable(json.loads(key_json))] = values
    return groups, data


def compare(baseline_path, records, threshold_pct=DEFAULT_THRESHOLD_PCT, json_out=True):
    base_groups, base_meta = load_baseline(baseline_path)
    cur_groups = group_samples(records)
    cur_meta = _find_run_meta(records)
    rows = []
    for key in cur_groups:
        if key not in base_groups:
            continue
        cur = cur_groups[key]
        base = base_groups[key]
        mean_cur = statistics.fmean(cur)
        mean_base = statistics.fmean(base)
        delta_pct = ((mean_cur - mean_base) / mean_base * 100.0) if mean_base else None
        worse = None
        if delta_pct is not None:
            hb = is_higher_better(key[2])
            worse = (delta_pct < 0) if hb else (delta_pct > 0)
        p_value = None
        if len(cur) >= 5 and len(base) >= 5:
            try:
                from scipy import stats as _stats

                # higher-better metric: current being lower than base is a regression.
                alternative = "less" if is_higher_better(key[2]) else "greater"
                p_value = float(_stats.mannwhitneyu(cur, base,
                                                    alternative=alternative).pvalue)
            except Exception:
                p_value = None
        significant = p_value is not None and p_value < 0.05
        flagged = bool(worse) and (significant or p_value is None) and (
            delta_pct is None or abs(delta_pct) >= threshold_pct)
        rows.append({
            "key": key,
            "base_mean": round(mean_base, 6),
            "cur_mean": round(mean_cur, 6),
            "delta_pct": round(delta_pct, 4) if delta_pct is not None else None,
            "worse": worse,
            "p_value": round(p_value, 6) if p_value is not None else None,
            "significant": significant,
            "method": "mannwhitneyu" if p_value is not None else "descriptive",
            "flagged": flagged,
        })
    result = {"rows": rows,
              "base_meta": base_meta,
              "cur_meta": cur_meta}
    if not rows:  # P5: distinguish "all fine" from "nothing to compare"
        result["note"] = ("no overlapping groups between current results "
                          "and baseline (engine/workload/metric/params mismatch?)")
    return result


def _key_params(key):
    """Folded params/control suffixes of a group key beyond the base five-tuple.

    The params suffix is a tuple of (k, v) pairs; render it readably. Other
    suffixes (control condition) fall back to str().
    """
    parts = []
    for x in key[5:]:
        if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str):
            parts.append(f"{x[0]}={x[1]}")
        elif (isinstance(x, tuple) and x
                and all(isinstance(p, tuple) and len(p) == 2 for p in x)):
            parts.append(" ".join(f"{p[0]}={p[1]}" for p in x))
        else:
            parts.append(str(x))
    return "; ".join(parts) if parts else "-"


def render_report(result, threshold_pct):
    lines = ["## 回归对比结果", ""]
    lines.append("| engine | workload | metric | threads | precision | params | base | cur | Δ% | p | 判定 |")
    lines.append("|--------|----------|--------|---------|-----------|--------|------|-----|----|---|------|")
    for r in result["rows"]:
        k = r["key"]  # variable length: base five-tuple + folded params/control
        pv = f"{r['p_value']:.4f}" if r["p_value"] is not None else "-"
        if r["flagged"]:
            tag = "REGRESSION" if r.get("method", "mannwhitneyu") == "mannwhitneyu" else "REGRESSION(descriptive)"
        elif r["worse"]:
            tag = "worse"
        else:
            tag = "ok"
        lines.append(
            f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} | {_key_params(k)} | "
            f"{r['base_mean']} | {r['cur_mean']} | "
            f"{r['delta_pct']} | {pv} | {tag} |")
    lines.append("")
    lines.append(f"> 阈值: Δ > {threshold_pct}% (statistically significant 或样本不足时以描述判断)")
    for r in result["rows"]:
        if r["flagged"]:
            k = r["key"]
            note = "" if r.get("method", "mannwhitneyu") == "mannwhitneyu" else " [descriptive only]"
            lines.append(f"> **REGRESSION DETECTED**: {k[0]} {k[1]} {k[2]} Δ={r['delta_pct']:+.2f}%{note}")
    return "\n".join(lines) + "\n"


def main(args) -> int:
    import sys

    records = read_paths(args.results, include_invalid=getattr(args, "include_invalid", False))
    if not records:
        print("regression: error: no records found in --results", file=sys.stderr)
        return 1
    if args.action == "save":
        out = args.out
        if not out:
            print("regression: error: --out required for save", file=sys.stderr)
            return 1
        payload = save_baseline(out, records)
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"Baseline saved: {out}")
        return 0
    # compare
    threshold = getattr(args, "threshold", DEFAULT_THRESHOLD_PCT)
    if threshold is None:
        threshold = DEFAULT_THRESHOLD_PCT
    result = compare(args.baseline, records, threshold_pct=threshold)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(render_report(result, threshold))
        if result.get("note"):
            print(f"note: {result['note']}", file=sys.stderr)
    return 0
