"""Power / energy analysis (spec: benchmark-power).

Parses platform power data (adb batterystats or host RAPL) into Joules and
computes energy-per-inference and perf/Watt. When power data is unavailable or
cannot be parsed it is labeled ``Power data unavailable`` and never fabricated.
"""
from __future__ import annotations

import json
import re

MAH_TO_JOULE = 12.96  # 0.001 A * 3600 s * 3.6 V = 12.96 J


def energy_joules(text, source="auto"):
    """Return {joules, unit, note} or None when nothing usable is present.

    ``Capacity`` values are never treated as power (batterystats prints
    ``Capacity: 5000``); only drain/energy signals are used.
    """
    text = re.sub(r"\s+", " ", text)  # handles wrapped numeric lines
    note = None
    if source in ("auto", "batterystats"):
        m = re.search(r"Computed drain:\s*([\d.,]+)\s*(mAh|mwh|j|J)?", text)
        if m:
            value = float(m.group(1).replace(",", ""))
            # drain of 0 (after a batterystats reset) is not a power signal
            if value > 0:
                unit = (m.group(2) or "mAh").lower()
                if unit == "mah":
                    note = "estimate: mAh x 12.96 J (3.6V assumed)"
                    return {"joules": value * MAH_TO_JOULE, "unit": "J", "note": note}
                if unit == "mwh":
                    note = "estimate: mWh x 3.6 J"
                    return {"joules": value * 3.6, "unit": "J", "note": note}
                return {"joules": value, "unit": "J", "note": note}
    if source in ("auto", "rapl", "perf"):
        m = re.search(r"(?:energy|rapl)[^\n]*?([\d.]+)\s*J", text, re.IGNORECASE)
        if m:
            return {"joules": float(m.group(1)), "unit": "J", "note": "host RAPL"}
    return None


def report(text, source="auto", inference_count=None, throughput=None, duration_s=None):
    """Build a power report dict; label unavailable rather than fabricate."""
    parsed = energy_joules(text, source)
    if parsed is None or parsed["joules"] is None:
        return {
            "status": "unavailable",
            "label": "Power data unavailable",
            "reason": "no usable energy signal in source",
        }
    joules = parsed["joules"]
    energy_per = joules / inference_count if inference_count else None
    ops_per_joule = (inference_count / joules) if (inference_count and joules > 0) else None
    avg_watts = (joules / duration_s) if (duration_s and duration_s > 0) else None
    # perf/Watt 必须用吞吐量除以平均功率（W），缺测量时长时置 None 绝不混用焦耳
    eff = (throughput / avg_watts) if (throughput and avg_watts and avg_watts > 0) else None
    return {
        "status": "ok",
        "energy_joules": round(joules, 4),
        "unit": "J",
        "note": parsed["note"],
        "energy_per_inference_j": round(energy_per, 6) if energy_per is not None else None,
        "ops_per_joule": round(ops_per_joule, 6) if ops_per_joule is not None else None,
        "avg_watts": round(avg_watts, 6) if avg_watts is not None else None,
        "perf_per_watt": round(eff, 6) if eff is not None else None,
        "throughput_provided": bool(throughput),
    }


def main(args) -> int:
    import sys

    if not args.source:
        print("power: error: --source <power-data-file> required", file=sys.stderr)
        return 1
    try:
        with open(args.source, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as exc:
        print(f"power: error: cannot read {args.source}: {exc}", file=sys.stderr)
        return 1
    throughput = getattr(args, "throughput", None)
    duration_s = getattr(args, "duration", None)
    rep = report(
        text,
        inference_count=args.inference_count,
        throughput=throughput,
        duration_s=duration_s,
    )
    if args.json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    else:
        print(rep["label"] if rep["status"] == "unavailable"
              else f"Energy per inference: {rep['energy_per_inference_j']} J "
                   f"({rep['note']})")
    return 0
