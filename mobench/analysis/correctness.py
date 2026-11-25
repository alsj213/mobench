"""Offline precision comparison over validate.json artifacts
(spec: benchmark-correctness 判定语义与退出码).

The gate semantics differ deliberately from the advisory regression report:
exit 0 = PASS/WARN, 2 = FAIL, 1 = cannot-compare (controls/coverage mismatch)
or UNAVAILABLE instruments. A verdict is never fabricated: an instrument that
is missing or unavailable on either side yields UNAVAILABLE, not a number.

Two modes:
- cross: ``compare_artifacts(baseline, candidate)`` — side ``a`` of each
  artifact (the golden workflow: a known-good artifact vs a new one);
- intra: ``compare_artifacts(ab_artifact)`` — side ``a`` vs side ``b`` of one
  A/B artifact (the same-run two-config comparison).
"""
from __future__ import annotations

import json
import sys

# Verdict precedence: a known failure outranks an unknown, which outranks a
# tolerated divergence.
_PRECEDENCE = {"PASS": 0, "WARN": 1, "UNAVAILABLE": 2, "FAIL": 3}
_EXIT_CODES = {"PASS": 0, "WARN": 0, "FAIL": 2, "UNAVAILABLE": 1,
               "INCOMPARABLE": 1}


class CompareError(Exception):
    """Artifact missing/unreadable/not a validate artifact."""


def load_artifact(path):
    try:
        with open(path, encoding="utf-8") as fh:
            artifact = json.load(fh)
    except (OSError, ValueError) as exc:
        raise CompareError(f"cannot read artifact {path}: {exc}") from exc
    schema = artifact.get("schema") or ""
    if not schema.startswith("mobench/validate"):
        raise CompareError(
            f"{path} is not a validate artifact (schema={schema!r})")
    if "a" not in (artifact.get("sides") or {}):
        raise CompareError(f"{path} has no side 'a'")
    return artifact


def _first_divergence(a, b):
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else limit


def _divergence_count(a, b):
    return sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))


def _worst(verdicts):
    vals = [v for v in verdicts if v]
    if not vals:
        return None
    return max(vals, key=lambda v: _PRECEDENCE.get(v, 0))


def _field_diff(a, b, label_a="a", label_b="b"):
    diffs = []
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key), b.get(key)
        if va != vb:
            diffs.append(f"{key}: {label_a}={va!r} != {label_b}={vb!r}")
    return diffs


def _coverage_diff(cov_a, cov_b):
    diffs = []
    for path in ("prefill", "decode"):
        ca, cb = cov_a.get(path) or {}, cov_b.get(path) or {}
        if bool(ca.get("covered")) != bool(cb.get("covered")):
            diffs.append(
                f"{path}: covered a={bool(ca.get('covered'))} "
                f"!= b={bool(cb.get('covered'))}")
    return diffs


def _identity_diff(identity_a, identity_b):
    diffs = []
    identity_a, identity_b = identity_a or {}, identity_b or {}
    for key in ("engine_version", "params", "binary_md5"):
        va, vb = identity_a.get(key), identity_b.get(key)
        if va == vb:
            continue
        if isinstance(va, dict) and isinstance(vb, dict):
            for sub in sorted(set(va) | set(vb)):
                if va.get(sub) != vb.get(sub):
                    diffs.append(f"{key}.{sub}: a={va.get(sub)!r} != b={vb.get(sub)!r}")
        else:
            diffs.append(f"{key}: a={va!r} != b={vb!r}")
    return diffs


def _incomparable(baseline_path, candidate_path, mode, **extra):
    result = {
        "mode": mode, "baseline": baseline_path, "candidate": candidate_path,
        "verdict": "INCOMPARABLE", "exit_code": _EXIT_CODES["INCOMPARABLE"],
        "controls_diff": [], "coverage_diff": [], "identity_diff": [],
        "instruments": {},
    }
    result.update(extra)
    return result


def _compare_ppl(result, side_a, side_b, ppl_tolerance):
    a = (side_a.get("instruments") or {}).get("ppl")
    b = (side_b.get("instruments") or {}).get("ppl")
    if a is None or b is None:
        entry = {"verdict": "UNAVAILABLE",
                 "reason": f"instrument not present on side "
                           f"{'a' if a is None else 'b'}"}
    elif a.get("status") != "ok" or b.get("status") != "ok":
        entry = {"verdict": "UNAVAILABLE",
                 "a": {"status": a.get("status"), "reason": a.get("reason")},
                 "b": {"status": b.get("status"), "reason": b.get("reason")}}
    else:
        va, vb = a.get("value"), b.get("value")
        entry = {"a": {"value": va}, "b": {"value": vb}}
        if va == vb:  # 逐位一致：同一文本的解析值
            entry.update({"verdict": "PASS", "delta_pct": 0.0})
        else:
            delta = abs(vb - va) / abs(va) * 100.0 if va else None
            entry["delta_pct"] = round(delta, 4) if delta is not None else None
            within = (delta is not None and ppl_tolerance is not None
                      and delta <= ppl_tolerance)
            entry["verdict"] = "WARN" if within else "FAIL"
    result["instruments"]["ppl"] = entry


def _compare_greedy(result, side_a, side_b, tolerance, context):
    ga = (side_a.get("instruments") or {}).get("greedy")
    gb = (side_b.get("instruments") or {}).get("greedy")
    entry = {"runs": []}
    if ga is None or gb is None:
        entry.update({"verdict": "UNAVAILABLE",
                      "reason": "instrument not present on one side"})
    elif ga.get("status") not in ("ok", "partial") or \
            gb.get("status") not in ("ok", "partial"):
        entry.update({"verdict": "UNAVAILABLE",
                      "reason": f"a={ga.get('status')} b={gb.get('status')}"})
    elif len(ga["runs"]) != len(gb["runs"]):
        entry.update({"verdict": "UNAVAILABLE", "reason": "run count mismatch"})
    else:
        for ra, rb in zip(ga["runs"], gb["runs"]):
            run = {"prompt": ra.get("prompt") or rb.get("prompt")}
            if ra.get("status") != "ok" or rb.get("status") != "ok":
                run.update({"verdict": "UNAVAILABLE",
                            "a": ra.get("reason"), "b": rb.get("reason")})
            else:
                ta, tb = ra.get("text") or "", rb.get("text") or ""
                idx = _first_divergence(ta, tb)
                if idx is None:
                    run.update({"verdict": "PASS", "first_divergence": None,
                                "divergence_rate": 0.0})
                else:
                    longest = max(len(ta), len(tb))
                    rate = (100.0 * _divergence_count(ta, tb) / longest
                            if longest else 0.0)
                    run.update({
                        "verdict": "WARN" if rate <= tolerance else "FAIL",
                        "first_divergence": idx,
                        "divergence_rate": round(rate, 4),
                        "context": {"a": ta[max(0, idx - context):idx + context],
                                    "b": tb[max(0, idx - context):idx + context]},
                    })
            entry["runs"].append(run)
        entry["verdict"] = _worst(r["verdict"] for r in entry["runs"]) or "PASS"
    result["instruments"]["greedy"] = entry


def compare_artifacts(baseline_path, candidate_path=None, tolerance=0.0,
                      ppl_tolerance=None, context=8):
    """Compare two artifacts (side a vs side a) or one A/B artifact (a vs b).

    ``tolerance`` is a percent divergence allowance for greedy token streams;
    ``ppl_tolerance`` the percent allowance for PPL. Both default to exact.
    """
    base = load_artifact(baseline_path)
    if candidate_path:
        mode = "cross"
        cand = load_artifact(candidate_path)
        controls_diff = _field_diff(base.get("controls") or {},
                                    cand.get("controls") or {})
        if controls_diff:
            return _incomparable(
                baseline_path, candidate_path, mode,
                controls_diff=controls_diff,
                note="controls differ — different thread counts, models, "
                     "prompts or sampling are not a meaningful comparison")
        coverage_diff = _coverage_diff(base.get("coverage") or {},
                                       cand.get("coverage") or {})
        if coverage_diff:
            return _incomparable(
                baseline_path, candidate_path, mode,
                coverage_diff=coverage_diff,
                note="coverage differs — the sides do not evidence the same "
                     "inference paths")
        side_a, side_b = base["sides"]["a"], cand["sides"]["a"]
        coverage = cand.get("coverage") or {}
    else:
        mode = "intra"
        if "b" not in (base.get("sides") or {}):
            raise CompareError(
                f"{baseline_path} has no side 'b': intra comparison needs an "
                f"A/B artifact (run with --config-b) or pass --candidate")
        side_a, side_b = base["sides"]["a"], base["sides"]["b"]
        coverage = base.get("coverage") or {}

    result = {
        "mode": mode,
        "baseline": baseline_path,
        "candidate": candidate_path,
        "controls_diff": [],
        "coverage_diff": [],
        "identity_diff": _identity_diff(side_a.get("identity"),
                                        side_b.get("identity")),
        "coverage": coverage,
        "uncovered_paths": [p for p in ("prefill", "decode")
                            if not (coverage.get(p) or {}).get("covered")],
        "instruments": {},
    }
    _compare_ppl(result, side_a, side_b, ppl_tolerance)
    _compare_greedy(result, side_a, side_b, tolerance, context)
    verdict = _worst([entry.get("verdict")
                      for entry in result["instruments"].values()]) or "PASS"
    result["verdict"] = verdict
    result["exit_code"] = _EXIT_CODES.get(verdict, 1)
    return result


def render_report(result) -> str:
    uncovered = result.get("uncovered_paths") or []
    annotation = ("".join(f" — {p.upper()} NOT COVERED" for p in uncovered))
    lines = [f"validate compare [{result.get('mode')}]: "
             f"{result['verdict']} (exit {result['exit_code']}){annotation}"]
    if result.get("note"):
        lines.append(f"  note: {result['note']}")
    coverage = result.get("coverage") or {}
    for path in ("prefill", "decode"):
        cov = coverage.get(path) or {}
        if cov.get("covered"):
            detail = cov.get("by")
            if cov.get("ctx"):
                detail += f"@{cov['ctx']}"
            lines.append(f"  coverage {path}: covered ({detail})")
        else:
            lines.append(f"  coverage {path}: NOT COVERED "
                         f"({cov.get('reason') or 'unknown'})")
    for diff in result.get("controls_diff") or []:
        lines.append(f"  controls diff: {diff}")
    for diff in result.get("coverage_diff") or []:
        lines.append(f"  coverage diff: {diff}")
    for diff in result.get("identity_diff") or []:
        lines.append(f"  identity diff (measured variable): {diff}")
    ppl = (result.get("instruments") or {}).get("ppl")
    if ppl:
        if ppl["verdict"] == "UNAVAILABLE":
            lines.append(f"  ppl: UNAVAILABLE "
                         f"({ppl.get('reason') or 'see sides'})")
        else:
            lines.append(f"  ppl: a={ppl['a']['value']} b={ppl['b']['value']} "
                         f"delta={ppl.get('delta_pct')}% {ppl['verdict']}")
    greedy = (result.get("instruments") or {}).get("greedy")
    if greedy:
        if not greedy.get("runs"):
            lines.append(f"  greedy: UNAVAILABLE ({greedy.get('reason')})")
        for run in greedy["runs"]:
            line = f"  greedy[{run.get('prompt')}]: {run['verdict']}"
            if run.get("first_divergence") is not None:
                line += (f" — first divergence at byte "
                         f"{run['first_divergence']} "
                         f"(rate {run['divergence_rate']}%) "
                         f"a={run['context']['a']!r} "
                         f"b={run['context']['b']!r}")
            lines.append(line)
    return "\n".join(lines)


def main(args) -> int:
    try:
        result = compare_artifacts(
            args.baseline,
            getattr(args, "candidate", None),
            tolerance=getattr(args, "tolerance", 0.0) or 0.0,
            ppl_tolerance=getattr(args, "ppl_tolerance", None))
    except (CompareError, OSError, ValueError) as exc:
        print(f"validate compare: error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(render_report(result))
    return result["exit_code"]
