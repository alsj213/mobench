"""Power / energy analysis (spec: benchmark-power).

Parses platform power data (adb batterystats or host RAPL) into Joules and
computes energy-per-inference and perf/Watt. When power data is unavailable or
cannot be parsed it is labeled ``Power data unavailable`` and never fabricated.
"""
from __future__ import annotations

import json
import re
import statistics

MAH_TO_JOULE = 3.6  # approx at 3.6V, annotated as an estimate


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
                    note = "estimate: mAh x 3.6 J (3.6V assumed)"
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


def report(text, source="auto", inference_count=None, throughput=None):
    """Build a power report dict; label unavailable rather than fabricate."""
    parsed = energy_joules(text, source)
    if parsed is None or parsed["joules"] is None:
        return {
            "status": "unavailable",
            "label": "Power data unavailable",
            "reason": "no usable energy signal in source",
        }
    energy_per = parsed["joules"] / inference_count if inference_count else None
    eff = (throughput / parsed["joules"]) if (throughput and parsed["joules"]) else None
    return {
        "status": "ok",
        "energy_joules": round(parsed["joules"], 4),
        "unit": "J",
        "note": parsed["note"],
        "energy_per_inference_j": round(energy_per, 6) if energy_per is not None else None,
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
    rep = report(text, inference_count=args.inference_count, throughput=throughput)
    if args.json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    else:
        print(rep["label"] if rep["status"] == "unavailable"
              else f"Energy per inference: {rep['energy_per_inference_j']} J "
                   f"({rep['note']})")
    return 0
