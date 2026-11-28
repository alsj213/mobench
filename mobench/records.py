"""Canonical measurement records (spec: benchmark-core 规范化测量记录).

A record is one measured sample point in an engine-independent form. All
analysis/reporting consumes records — never raw engine output. Every record
must trace back to a non-empty raw log via ``source_log``.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Iterator

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


def resolve_source_log(source_log: str, base_dir: str | None = None) -> str | None:
    """解析原始日志路径：优先归档目录，最后才回退当前工作目录。

    绝对路径先看自身是否存在，再退到 ``base_dir/basename``；相对路径在给定
    ``base_dir`` 时优先 ``base_dir/source_log`` 与 ``base_dir/basename``，最后
    才按 CWD 解析。这样 run 目录被整体归档/移动后，同名 CWD 文件不会抢占
    真正的溯源日志。
    """
    if not source_log:
        return None
    if os.path.isabs(source_log):
        if os.path.isfile(source_log):
            return source_log
        if base_dir:
            sibling = os.path.join(base_dir, os.path.basename(source_log))
            if os.path.isfile(sibling):
                return sibling
        return None
    if base_dir:
        for candidate in (
            os.path.join(base_dir, source_log),
            os.path.join(base_dir, os.path.basename(source_log)),
        ):
            if os.path.isfile(candidate):
                return candidate
    if os.path.isfile(source_log):
        return source_log
    return None


def validate_record(rec: dict, base_dir: str | None = None) -> list:
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
    if source_log and resolve_source_log(source_log, base_dir=base_dir) is None:
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
    base_dir = os.path.dirname(os.path.abspath(path))
    errors = validate_record(rec, base_dir=base_dir)
    if errors:
        raise RecordError("invalid record: " + "; ".join(errors))
    out = normalize_record(rec)
    os.makedirs(base_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(out, ensure_ascii=False) + "\n")


def iter_records(path: str) -> Iterator[dict]:
    """Yield parsed records from a records JSONL file."""
    base_dir = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            errs = validate_record(rec, base_dir=base_dir)
            if errs:
                # 明确向 stderr 告警跳过的非法行，绝不静默丢数据
                print(
                    f"Warning: skipping invalid record in {path}:{lineno}: {'; '.join(errs)}",
                    file=sys.stderr,
                )
                continue
            out = normalize_record(rec)
            resolved = resolve_source_log(out.get("source_log", ""), base_dir=base_dir)
            if resolved:
                out["source_log"] = resolved
            yield out


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
    backend = rec.get("backend")
    if backend not in (None, ""):
        key.append(("backend", str(backend)))
    platform = rec.get("platform")
    if platform not in (None, "", "host"):
        key.append(("platform", str(platform)))
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


def _run_invalid_reason(record_file: str) -> str | None:
    """检查同目录 run.json 的审计状态；无效时返回可读原因。"""
    run_json = os.path.join(os.path.dirname(os.path.abspath(record_file)), "run.json")
    if not os.path.isfile(run_json):
        return None
    try:
        with open(run_json, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        # 损坏的 run.json 不能当作“无审计信息”静默放行——它的 valid 状态未知。
        return "unreadable run.json"
    if not isinstance(meta, dict):
        return "malformed run.json"
    reasons = []
    if meta.get("valid") is False:
        reasons.append("valid=false")
    if (meta.get("log_audit") or {}).get("valid") is False:
        reasons.append("log_audit.valid=false")
    if (meta.get("freq_audit") or {}).get("valid") is False:
        reasons.append("freq_audit.valid=false")
    return ", ".join(reasons) if reasons else None


def read_paths(inputs, include_invalid: bool = False) -> list:
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
            if not include_invalid:
                bad_reason = _run_invalid_reason(f)
                if bad_reason:
                    print(
                        f"Warning: skipping invalid run records {f} ({bad_reason}); "
                        "pass --include-invalid to include",
                        file=sys.stderr,
                    )
                    continue
            try:
                samples.extend(iter_records(f))
            except Exception as exc:  # tolerate a single bad file
                print(f"Warning: cannot parse {f}: {exc}", file=sys.stderr)
                continue
    return samples
