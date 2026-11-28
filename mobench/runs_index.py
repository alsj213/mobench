"""Index of completed runs: one compact row per run.json under results/.

With dozens of run dirs scattered across ``results/``, answering "which run
was which variant, built with what recipe, in what audit state" meant manual
digging every time. This scans run.json files and emits a stable index
(aligned table, or ``--json`` for machine consumption — e.g. an evidence
reference or an A/B pair lookup).

Fields carried per row: timestamp, engine, label, artifact md5, record
count, build-recipe verification, log-probe audit state, throttle flag.
"""
from __future__ import annotations

import json
import os
import sys


def row_from_meta(meta, path):
    binary = meta.get("binary") or {}
    md5 = binary.get("md5")
    if isinstance(md5, dict):  # tolerate odd shapes rather than crash the index
        md5 = md5.get("value") or md5.get("md5")
    recipe = meta.get("build_recipe") or {}
    audit = meta.get("log_audit")
    freq_audit = meta.get("freq_audit")
    log_ok = None if audit is None else audit.get("valid")
    freq_ok = None if freq_audit is None else freq_audit.get("valid")
    valid = meta.get("valid")
    if valid is None:
        if log_ok is False or freq_ok is False:
            valid = False
        elif log_ok is True or freq_ok is True:
            valid = True
    return {
        "timestamp": meta.get("timestamp"),
        "engine": meta.get("engine"),
        "label": meta.get("label"),
        "binary_md5": md5,
        "n_records": meta.get("n_records"),
        "models": [m.get("id") for m in (meta.get("models") or [])],
        "recipe_checked": recipe.get("checked"),
        "log_audit_valid": log_ok,
        "freq_audit_valid": freq_ok,
        "valid": valid,
        "throttle_risk": meta.get("throttle_risk"),
        "run_dir": os.path.dirname(path),
    }


def scan_runs(results_dir="results"):
    """All run.json rows under results_dir, sorted by run directory."""
    rows = []
    for root, _dirs, files in os.walk(results_dir):
        if "run.json" not in files:
            continue
        path = os.path.join(root, "run.json")
        try:
            with open(path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue  # a broken run.json must not sink the whole index
        rows.append(row_from_meta(meta, path))
    rows.sort(key=lambda r: r["run_dir"])
    return rows


def _flags(row):
    flags = []
    if row["recipe_checked"] is True:
        flags.append("recipe-ok")
    if row["log_audit_valid"] is False:
        flags.append("AUDIT-INVALID")
    if row.get("freq_audit_valid") is False:
        flags.append("FREQ-INVALID")
    if row["throttle_risk"]:
        flags.append("throttled")
    return ",".join(flags) or "-"


def render_rows(rows):
    lines = []
    for r in rows:
        n = r["n_records"]
        lines.append(
            "{ts:<15} {eng:<9} {label:<14} md5:{md5:<9} recs:{n:<5} {flags:<22} {path}".format(
                ts=r["timestamp"] or "?",
                eng=r["engine"] or "?",
                label=r["label"] or "-",
                md5=(r["binary_md5"] or "-")[:8],
                n=n if n is not None else "?",
                flags=_flags(r),
                path=r["run_dir"]))
    return "\n".join(lines)


def main(args) -> int:
    rows = scan_runs(args.results)
    if not rows:
        print(f"runs: no run.json found under {args.results}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(render_rows(rows))
        print(f"\n{len(rows)} runs")
    return 0
