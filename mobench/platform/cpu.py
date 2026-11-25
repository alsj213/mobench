"""Cluster-aware CPU selection and taskset masks (add-cpu-pin-lockfreq).

Clusters come from the kernel's own cpufreq policies rather than from
clustering cores by frequency ourselves: the policy boundary is exactly the
granularity a governor is written at, so it is both more reliable and directly
actionable.

Selection deliberately avoids *partially* occupying a cluster. Threads placed
on part of a cluster while their peers sit on another one run at a different
frequency, which is the measurement noise this module exists to remove.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

# taskset masks are a single machine word here; wider sets use a notation that
# differs between the toybox and util-linux builds, so they are refused.
MASK_BITS = 64


class CpuSelectionError(ValueError):
    """CPU selection or mask generation failed (bad input, unsupported size)."""


@dataclass(frozen=True)
class Policy:
    """One cpufreq policy: the CPUs sharing a clock domain, and its ceiling."""

    cpus: List[int]
    max_freq_khz: int = 0
    name: str = ""


def parse_policies(entries: Optional[Iterable[dict]]) -> List[Policy]:
    """Build policies from raw sysfs entries.

    Each entry maps ``related_cpus`` (space separated) and
    ``cpuinfo_max_freq`` (kHz). Entries without a usable CPU list are skipped;
    a missing or unreadable frequency is kept as 0 rather than dropping the
    cluster, so cores are never silently lost from the selectable set.
    """
    policies = []
    for entry in entries or []:
        cpus = []
        for token in str(entry.get("related_cpus") or "").split():
            try:
                cpus.append(int(token))
            except ValueError:
                continue
        if not cpus:
            continue
        try:
            freq = int(str(entry.get("cpuinfo_max_freq")).strip())
        except (TypeError, ValueError):
            freq = 0
        cpus = sorted(set(cpus))
        # sysfs names the directory after the cluster's first CPU.
        policies.append(Policy(cpus=cpus, max_freq_khz=freq,
                               name=str(entry.get("name") or f"policy{cpus[0]}")))
    return policies


def _fastest_first(policies: Sequence[Policy]) -> List[Policy]:
    """Order clusters by frequency; ties prefer the larger cluster."""
    return sorted(policies,
                  key=lambda p: (-p.max_freq_khz, -len(p.cpus), p.cpus[0]))


def _whole_cluster_cover(policies: Sequence[Policy], n: int):
    """Highest-frequency union of *whole* clusters summing exactly to ``n``.

    Enumerates cluster subsets -- a device has a single-digit number of
    policies, so the 2**k search is trivial and keeps the rule exact.
    """
    best, best_score = None, None
    k = len(policies)
    for mask in range(1, 1 << k):
        chosen = [policies[i] for i in range(k) if mask & (1 << i)]
        if sum(len(p.cpus) for p in chosen) != n:
            continue
        score = sum(p.max_freq_khz for p in chosen)
        if best_score is None or score > best_score:
            best, best_score = chosen, score
    if best is None:
        return None
    return sorted(c for p in best for c in p.cpus)


def select_cpus(policies: Sequence[Policy], n: int) -> List[int]:
    """Pick ``n`` CPUs, preferring whole cpufreq clusters.

    1. a union of whole clusters summing exactly to ``n`` (highest frequency sum wins)
    2. otherwise the front of the fastest cluster that can hold ``n``
    3. otherwise greedy whole clusters, then the front of what remains

    ``n`` at or above the core count selects every core (oversubscription is a
    legitimate workload, not a configuration error).
    """
    usable = [p for p in (policies or []) if p.cpus]
    if not usable:
        raise CpuSelectionError("no CPUs found in cpufreq policies")
    if n < 1:
        raise CpuSelectionError(f"thread count must be >= 1, got {n}")

    ordered = _fastest_first(usable)
    if n >= sum(len(p.cpus) for p in ordered):
        return sorted(c for p in ordered for c in p.cpus)

    cover = _whole_cluster_cover(ordered, n)
    if cover is not None:
        return cover

    for p in ordered:                       # one cluster can hold n outright
        if len(p.cpus) >= n:
            return sorted(p.cpus[:n])

    picked = []                             # greedy whole clusters first
    for p in ordered:
        if len(picked) + len(p.cpus) <= n:
            picked.extend(p.cpus)
        if len(picked) == n:
            return sorted(picked)
    for p in ordered:                       # then top up from what is left
        for cpu in p.cpus:
            if cpu in picked:
                continue
            picked.append(cpu)
            if len(picked) == n:
                return sorted(picked)
    return sorted(picked)


def cpu_mask(cpus: Iterable[int]) -> str:
    """Taskset mask for ``cpus`` in *bare* hexadecimal (``[4,5,6,7]`` -> ``f0``).

    Two format constraints, both observed on real runs rather than assumed:

    * hex, not decimal -- a decimal mask silently binds the wrong cores;
    * **no ``0x`` prefix** -- util-linux accepts either form, but Android's
      toybox ``taskset`` rejects the prefixed one outright
      (``taskset: bad mask '0xf0'``), so the bare form is the only one that
      works on both platforms.
    """
    cpus = sorted({int(c) for c in cpus})
    if not cpus:
        raise CpuSelectionError("cpu_mask needs at least one CPU")
    if cpus[0] < 0:
        raise CpuSelectionError(f"CPU indexes must be non-negative, got {cpus[0]}")
    if cpus[-1] >= MASK_BITS:
        raise CpuSelectionError(
            f"CPU {cpus[-1]} needs a mask wider than {MASK_BITS} bits; set "
            "methodology.pin_cpus explicitly or disable cpu_pin")
    value = 0
    for cpu in cpus:
        value |= 1 << cpu
    return format(value, "x")
