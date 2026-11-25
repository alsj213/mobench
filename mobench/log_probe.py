"""Log probes: forbid/expect checks over each round's raw engine log.

``config.methodology.log_checks: {forbid: [regex, ...], expect: [regex, ...]}``

- A **forbidden** pattern in a round's log invalidates the whole run: abort
  immediately (fail fast — continuing burns device time on data that cannot
  be trusted). The K8 incident shape: 311 ``cannot be used with CPU_REPACK``
  lines meant the repack path was dead, yet the measurement still produced
  plausible-looking numbers.
- An **expected** pattern missing from a round's log records a violation for
  that round; after all rounds run.json carries ``log_audit.valid=false``
  with the details (data is kept, clearly marked — never silently discarded).

Patterns are case-sensitive regexes, compiled once up front so a bad regex
fails the run before any device time is spent. No log_checks -> zero behavior.
"""
from __future__ import annotations

import re


class LogProbeError(Exception):
    """log_checks declared but malformed (bad regex or wrong shape)."""


def compile_checks(methodology):
    """Compile methodology.log_checks into ``{role: [(pattern, rx), ...]}``.

    Returns None when no checks are declared (zero behavior).
    """
    checks = (methodology or {}).get("log_checks")
    if not checks:
        return None
    out = {"forbid": [], "expect": []}
    for role in out:
        pats = checks.get(role) or []
        if not isinstance(pats, list):
            raise LogProbeError(
                f"log_checks.{role} 应为模式列表，收到 {type(pats).__name__}")
        for p in pats:
            if not isinstance(p, str):
                raise LogProbeError(f"log_checks.{role} 条目应为字符串: {p!r}")
            try:
                out[role].append((p, re.compile(p)))
            except re.error as exc:
                raise LogProbeError(f"log_checks.{role} 正则非法 {p!r}: {exc}") from exc
    return out


def scan_forbid(path: str, checks: dict):
    """Violations for forbidden patterns, one entry per match (first per line)."""
    violations = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            for pat, rx in checks["forbid"]:
                if rx.search(line):
                    violations.append({
                        "role": "forbid", "pattern": pat, "log": path,
                        "line": lineno, "matched": line.strip()[:200],
                    })
                    break  # one violation per line is enough to identify the signal
    return violations


def scan_expect(path: str, checks: dict):
    """Violations for expected patterns with no match anywhere in the log."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    return [{"role": "expect", "pattern": pat, "log": path, "missing": True}
            for pat, rx in checks["expect"] if rx.search(text) is None]
