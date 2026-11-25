"""Canonical measurement records (spec: benchmark-core 规范化测量记录).

A record is one measured sample point in an engine-independent form. All
analysis/reporting consumes records — never raw engine output. Every record
must trace back to a non-empty raw log via ``source_log``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterator

RECORD_SCHEMA = "mobench/record@1"

# Fields whose absence makes a record untrustworthy / unusable.
REQUIRED_FIELDS = ("schema", "engine", "workload_id", "metric", "value", "unit", "source_log")
# Optional but normalized fields (defaults applied by normalize_record).
OPTIONAL_FIELDS = {
    "threads": None,
    "gpu": False,
    "precision": None,
    "round": None,
    "runs": None,
    "platform": None,
    "device": None,
    "ts": None,
    "raw": None,
    "params": None,  # workload params (e.g. prompt/gen tokens) for matrix grouping
    "backend": None,  # engine backend (cpu/opencl/metal/npu) when known
}


class RecordError(ValueError):
    """Raised when a record is malformed or untraceable."""


def validate_record(rec: dict) -> list:
    """Return a list of human-readable problems with ``rec``.

    An empty list means the record is acceptable.
    """
    errors = []
    if not isinstance(rec, dict):
        return ["record is not an object"]
    for field in REQUIRED_FIELDS:
        if field not in rec or rec[field] in (None, ""):
            errors.append(f"missing required field '{field}'")
    if "schema" in rec and rec["schema"] != RECORD_SCHEMA:
        errors.append(f"schema must be {RECORD_SCHEMA!r}")
    value = rec.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append("'value' must be numeric")
    source_log = rec.get("source_log")
    if source_log and not os.path.exists(source_log):
        errors.append(f"source_log does not exist: {source_log}")
    return errors


def normalize_record(rec: dict) -> dict:
    """Return a copy of ``rec`` with optional fields defaulted.

    Does not mutate the caller's dict.
    """
    merged = {k: v for k, v in OPTIONAL_FIELDS.items() if k not in rec}
    merged.update(rec)
    return merged


def append_record(path: str, rec: dict) -> None:
    """Validate and append one record (one JSON line) to ``path``.

    Raises RecordError if the record is invalid; creates parent dirs as needed.
    """
    errors = validate_record(rec)
    if errors:
        raise RecordError("invalid record: " + "; ".join(errors))
    out = normalize_record(rec)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(out, ensure_ascii=False) + "\n")


def iter_records(path: str) -> Iterator[dict]:
    """Yield parsed records from a records JSONL file."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if validate_record(rec):
                # Rows failing validation are refused at write time; skip a
                # hand-edited bad row rather than surfacing partial data.
                continue
            yield normalize_record(rec)


def _control_key(rec: dict):
    """Measurement-condition dimension: pinned CPU set + frequency lock state.

    A pinned run and an unpinned one are different measurement conditions and
    must not be merged into a single group -- the same rationale that folds
    ``params`` into the key. Records without a control block (produced before
    this dimension existed) return None and keep the historical grouping.
    """
    device = rec.get("device")
    if not isinstance(device, dict):
        return None
    pin, freq = device.get("pin"), device.get("freq")
    if not isinstance(pin, dict) or not isinstance(freq, dict):
        return None
    cpus, locked = pin.get("cpus"), freq.get("locked")
    if cpus is None and locked is None:
        return None
    return (tuple(cpus or ()), bool(locked))


def group_key(rec: dict) -> tuple:
    """Analysis grouping key.

    Base: (engine, workload_id, metric, threads, precision). When a record
    carries workload ``params`` (P2: param-matrix provenance), they are folded
    into the key so runs with different parameters stay separate; likewise the
    CPU-pin/frequency-lock condition. Legacy records without either keep the
    plain five-tuple.
    """
    key = [
        rec.get("engine"),
        rec.get("workload_id"),
        rec.get("metric"),
        rec.get("threads"),
        rec.get("precision"),
    ]
    params = rec.get("params")
    if isinstance(params, dict) and params:
        key.append(tuple(sorted((str(k), str(v)) for k, v in params.items())))
    control = _control_key(rec)
    if control is not None:
        key.append(control)
    return tuple(key)


def empty_result_hint(inputs) -> str:
    """When no records were found, explain if an input dir was non-empty."""
    for path in inputs:
        if os.path.isdir(path) and os.listdir(path):
            return (f"(input {path} 非空但未找到记录文件——嵌套 records 已递归收集，"
                    "确认目录内含 records.jsonl)")
    return ""


def _collect_record_files(path: str) -> list:
    """Recursively collect record files under a directory (incl. nested run_*)."""
    found = []
    for root, _dirs, files in os.walk(path):
        for name in files:
            if name.endswith((".records", ".jsonl")) or name == "records.jsonl":
                found.append(os.path.join(root, name))
    return sorted(found)


def read_paths(inputs) -> list:
    """Expand files/dirs into parsed records (dirs scanned recursively)."""
    samples = []
    for path in inputs:
        if os.path.isdir(path):
            files = _collect_record_files(path)
        else:
            files = [path]
        for f in files:
            if not os.path.exists(f):
                continue
            try:
                samples.extend(iter_records(f))
            except Exception as exc:  # tolerate a single bad file
                print(f"Warning: cannot parse {f}: {exc}", file=__import__("sys").stderr)
                continue
    return samples
