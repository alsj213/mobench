"""Precision-validation run: instruments -> self-describing validate.json
(spec: benchmark-correctness 精度校验子命令与证物).

Deliberately NOT the performance pipeline: no cooldown gate, no env control,
no records.jsonl. Token streams and PPL are deterministic given the controls
fingerprint (model, threads, prompts, sampling) and independent of clock/pin/
temperature — the cross-config refusal below is where the comparability
contract is enforced, before any device time is spent.

Every declared file must be readable (a missing file would leave the
fingerprint incomplete — refuse rather than write a partial artifact) and
every instrument result must trace to a non-empty raw log; unparseable output
marks the instrument unavailable instead of guessing values.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys

from mobench.adapters.base import AdapterError, get_adapter
from mobench.artifacts_hash import artifact_set_md5
from mobench.io_utils import atomic_write_json, file_sha256, make_run_dir
from mobench.platform.base import BasePlatform


class ValidationError(Exception):
    """Validation run failed (config/files/platform/controls)."""


SCHEMA = "mobench/validate@1"

# Instrument -> inference path it provides evidence for. The coverage block is
# the structural answer to the llama-perplexity blind spot (batch prefill only,
# never single-token decode): a path with no instrument is NOT COVERED and no
# verdict may imply otherwise.
INSTRUMENT_PATHS = {"ppl": "prefill", "greedy": "decode"}


def _prompts_sha256(prompts):
    """Order-sensitive content hash: basename keeps each entry identifiable."""
    h = hashlib.sha256()
    for path in prompts:
        h.update(os.path.basename(path).encode())
        with open(path, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def build_controls(cfg, model_path, platform, threads) -> dict:
    """The fingerprint that MUST match for a comparison to be meaningful.

    ``threads`` is a control (not identity): the parallel reduction order
    changes floating-point results, so different thread counts cannot be
    compared.
    """
    val = cfg.get("validation") or {}
    controls = {
        "model_md5": platform.file_md5(model_path),
        "threads": threads,
        "sampling": "greedy",
        "gen_tokens": val.get("gen_tokens"),
        "prompts_sha256": _prompts_sha256(val.get("prompts") or []),
    }
    ppl = val.get("ppl")
    if ppl:
        controls["ppl"] = {
            "dataset_sha256": file_sha256(ppl["dataset"]),
            "ctx": ppl.get("ctx"),
            "chunks": ppl.get("chunks"),
        }
    return controls


def controls_diff(controls_a, controls_b, prefix_a="a", prefix_b="b") -> list:
    """Human-readable per-field differences between two controls dicts."""
    diffs = []
    for key in sorted(set(controls_a) | set(controls_b)):
        va, vb = controls_a.get(key), controls_b.get(key)
        if va != vb:
            diffs.append(f"{key}: {prefix_a}={va!r} != {prefix_b}={vb!r}")
    return diffs


def _declared_files(cfg):
    """(label, path) pairs of every file the fingerprint will hash."""
    val = cfg.get("validation") or {}
    files = []
    for prompt in val.get("prompts") or []:
        files.append(("validation prompt file", prompt))
    if val.get("ppl"):
        files.append(("validation ppl dataset", val["ppl"]["dataset"]))
    return files


def _check_files(cfg, model, platform):
    if not platform.file_exists(model["path"]):
        raise ValidationError(
            f"model file missing: {model['path']}. Run `mobench prepare` first.")
    for label, path in _declared_files(cfg):
        if not platform.file_exists(path):
            raise ValidationError(f"{label} missing: {path}")


def _run_side(cfg, cfg_path, side, model, adapter, platform, threads,
              out_dir, json_mode) -> dict:
    """Execute every instrument job for one side; per-job failures degrade."""
    val_cfg = cfg.get("validation") or {}
    workload = cfg.get("workload") or {}
    jobs = adapter.build_validate_cmds(model, val_cfg, workload, threads)
    identity = {
        "config_path": os.path.abspath(cfg_path) if cfg_path else None,
        "config_sha256": file_sha256(cfg_path) if cfg_path else None,
        "engine_version": adapter.version(),
        "binary_md5": {},
        "params": (
            adapter.validation_identity_params(workload)
            if hasattr(adapter, "validation_identity_params")
            else {}
        ),
    }
    instruments = {}
    for idx, job in enumerate(jobs):
        instrument = job["instrument"]
        binary = job["cmd"][0]
        identity["binary_md5"].setdefault(
            os.path.basename(binary), artifact_set_md5(binary))
        suffix = str(idx) if instrument == "greedy" else ""
        log_name = f"raw_{instrument}{suffix}_{side}.log"
        log_path = os.path.join(out_dir, log_name)
        if not json_mode:
            print(f"[validate] side:{side} {' '.join(job['cmd'])}")
        rc = platform.run(job["cmd"], log_path, echo=not json_mode)
        entry = {"log": log_name, "cmd": job["cmd"]}
        if rc != 0:
            entry.update({"status": "unavailable",
                          "reason": f"engine exited {rc}"})
        else:
            try:
                parsed = adapter.parse_validate(job, log_path)
                entry.update({"status": "ok", **parsed})
            except AdapterError as exc:
                entry.update({"status": "unavailable", "reason": str(exc)})
        if instrument == "greedy":
            entry["prompt"] = job.get("prompt")
            greedy = instruments.setdefault("greedy", {
                "status": "ok", "reason": None, "runs": [],
                "includes_prompt_echo": bool(job.get("include_prompt_echo")),
            })
            greedy["runs"].append(entry)
        else:
            instruments[instrument] = entry
    _aggregate_greedy(instruments.get("greedy"))
    return {"identity": identity, "instruments": instruments}


def _aggregate_greedy(greedy):
    if not greedy:
        return
    ok = [r for r in greedy["runs"] if r["status"] == "ok"]
    if not ok:
        greedy["status"] = "unavailable"
        greedy["reason"] = "; ".join(
            r["reason"] for r in greedy["runs"] if r.get("reason")
        ) or "all greedy runs failed"
    elif len(ok) < len(greedy["runs"]):
        greedy["status"] = "partial"
        greedy["reason"] = "some prompts unavailable (see runs)"


def _coverage(val_cfg, instruments) -> dict:
    """Per-path coverage declaration — the structural record of what is and is
    not evidenced. A configured-but-failed instrument covers nothing."""
    coverage = {}
    ppl = instruments.get("ppl")
    if not val_cfg.get("ppl"):
        coverage["prefill"] = {"covered": False, "by": None,
                               "reason": "instrument not configured"}
    elif ppl and ppl["status"] == "ok":
        coverage["prefill"] = {"covered": True, "by": "ppl",
                               "ctx": (val_cfg.get("ppl") or {}).get("ctx")}
    else:
        coverage["prefill"] = {
            "covered": False, "by": "ppl",
            "reason": (ppl or {}).get("reason") or "instrument failed"}
    greedy = instruments.get("greedy")
    if not greedy:
        coverage["decode"] = {"covered": False, "by": None,
                              "reason": "instrument not configured"}
    else:
        ok_runs = [r for r in greedy["runs"] if r["status"] == "ok"]
        if not ok_runs:
            coverage["decode"] = {"covered": False, "by": "greedy",
                                  "reason": greedy.get("reason")}
        else:
            n_prompt = ok_runs[0].get("n_prompt_tokens")
            gen_tokens = val_cfg.get("gen_tokens")
            n_kv_range = ([n_prompt, n_prompt + gen_tokens]
                          if n_prompt is not None and gen_tokens else None)
            entry = {"covered": True, "by": "greedy", "n_kv_range": n_kv_range}
            if n_kv_range is None:
                entry["note"] = "prompt token count unavailable; range unknown"
            coverage["decode"] = entry
    return coverage


def _merge_ab_coverage(cov_a: dict, cov_b: dict) -> dict:
    """A/B 双列对比时取两侧覆盖交集，任一侧未覆盖则显式标注 NOT COVERED。"""
    merged = {}
    for path in ("prefill", "decode"):
        ca = cov_a.get(path) or {}
        cb = cov_b.get(path) or {}
        if ca.get("covered") and cb.get("covered"):
            merged[path] = dict(ca)
        elif not ca.get("covered"):
            merged[path] = dict(ca)
        else:
            merged[path] = {
                "covered": False,
                "by": cb.get("by") or ca.get("by"),
                "reason": f"side b: {cb.get('reason') or 'not covered'}",
            }
    return merged


def run_validation(cfg, config_b=None, out_dir=None, threads=None,
                   config_path=None, config_b_path=None,
                   json_mode=False) -> dict:
    """Execute all instruments for one config (or a same-controls A/B pair).

    Returns the written artifact dict. Raises ValidationError on refusal
    (missing files, controls mismatch, unusable platform).
    """
    platform = BasePlatform.make(cfg)
    try:
        platform.preflight()
    except Exception as exc:
        raise ValidationError(f"platform preflight: {exc}") from exc
    try:
        adapter = get_adapter(cfg)
    except AdapterError as exc:
        raise ValidationError(str(exc)) from exc
    val_cfg = cfg.get("validation")
    if not val_cfg:
        raise ValidationError(
            "config has no 'validation' section: declare prompts (greedy) "
            "and/or a ppl block to run precision validation")
    models = cfg.get("models") or []
    if not models:
        raise ValidationError("config has no models")
    model = models[0]
    threads_override = threads
    threads = threads_override or (cfg.get("workload") or {}).get("threads", [4])[0]
    _check_files(cfg, model, platform)
    controls = build_controls(cfg, model["path"], platform, threads)

    model_b, threads_b, controls_b = None, None, None
    if config_b is not None:
        if config_b.get("engine") != cfg.get("engine"):
            raise ValidationError(
                "A/B configs must use the same engine "
                f"(a={cfg.get('engine')!r} b={config_b.get('engine')!r})")
        models_b = config_b.get("models") or []
        if not models_b or models_b[0].get("id") != model.get("id"):
            raise ValidationError(
                "A/B configs must declare the same model id as models[0]")
        model_b = models_b[0]
        threads_b = threads_override or ((config_b.get("workload") or {}).get("threads") or [threads])[0]
        _check_files(config_b, model_b, platform)
        try:
            controls_b = build_controls(config_b, model_b["path"], platform, threads_b)
        except OSError as exc:
            raise ValidationError(f"cannot fingerprint A/B controls: {exc}") from exc
        diffs = controls_diff(controls, controls_b)
        if diffs:
            raise ValidationError(
                "A/B controls mismatch — the comparison would not be "
                "meaningful; refusing before execution: " + "; ".join(diffs))

    results_dir = cfg.get("results_dir", "results")
    if out_dir:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(out_dir, exist_ok=True)
    else:
        ts, out_dir = make_run_dir(results_dir, "validate")

    sides = {"a": _run_side(cfg, config_path, "a", model, adapter, platform,
                            threads, out_dir, json_mode)}
    if config_b is not None:
        adapter_b = get_adapter(config_b)
        sides["b"] = _run_side(config_b, config_b_path, "b", model_b, adapter_b,
                               platform, threads_b, out_dir, json_mode)

    cov_a = _coverage(val_cfg, sides["a"]["instruments"])
    if "b" in sides:
        cov_b = _coverage(config_b.get("validation") or {}, sides["b"]["instruments"])
        coverage = _merge_ab_coverage(cov_a, cov_b)
    else:
        coverage = cov_a

    artifact = {
        "schema": SCHEMA,
        "created": ts,
        "engine": cfg.get("engine"),
        "model": {"id": model.get("id"), "path": model.get("path"),
                  "md5": controls["model_md5"]},
        "threads": threads,
        "controls": controls,
        "coverage": coverage,
        "sides": sides,
        "out_dir": out_dir,
    }
    path = os.path.join(out_dir, "validate.json")
    artifact["path"] = path
    atomic_write_json(path, artifact)
    return artifact


def any_instrument_ok(artifact) -> bool:
    """True when side a produced at least one usable instrument result."""
    for instrument in (artifact.get("sides", {}).get("a", {})
                       .get("instruments") or {}).values():
        if instrument.get("status") in ("ok", "partial"):
            return True
    return False


def render_summary(artifact) -> str:
    lines = [f"validate run: {artifact['path']}",
             f"  model: {artifact['model']['id']} (md5 {artifact['model']['md5'][:12]}…) "
             f"threads={artifact['threads']}"]
    for name in ("prefill", "decode"):
        cov = artifact["coverage"].get(name) or {}
        if cov.get("covered"):
            detail = f"by {cov.get('by')}"
            if cov.get("ctx"):
                detail += f"@{cov['ctx']}"
            if cov.get("n_kv_range"):
                detail += f" (n_kv {cov['n_kv_range'][0]}-{cov['n_kv_range'][1]})"
            lines.append(f"  coverage {name}: covered — {detail}")
        else:
            lines.append(f"  coverage {name}: NOT COVERED — "
                         f"{cov.get('reason') or 'unknown'}")
    for side, payload in artifact["sides"].items():
        for instrument, entry in payload["instruments"].items():
            if instrument == "greedy":
                lines.append(f"  [{side}] greedy: {entry['status']} "
                             f"({len(entry['runs'])} prompts)")
            else:
                value = entry.get("value")
                lines.append(f"  [{side}] {instrument}: {entry['status']}"
                             + (f" value={value}" if value is not None else ""))
    return "\n".join(lines)


def main(args) -> int:
    from mobench.config import ConfigError, load_config

    try:
        cfg = load_config(args.config)
        config_b = load_config(args.config_b) if getattr(args, "config_b", None) else None
        artifact = run_validation(
            cfg, config_b=config_b, out_dir=args.out,
            threads=getattr(args, "threads", None),
            config_path=args.config,
            config_b_path=getattr(args, "config_b", None),
            json_mode=args.json)
    except (ConfigError, ValidationError, AdapterError) as exc:
        print(f"validate: error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(artifact, indent=2, ensure_ascii=False))
    else:
        print(render_summary(artifact))
    if not any_instrument_ok(artifact):
        # 证据已落盘（工件），但没有任何可用仪器 → 以错误退出
        print("validate: error: no instrument produced a usable result",
              file=sys.stderr)
        return 1
    return 0
