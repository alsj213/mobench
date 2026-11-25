"""Environment snapshot, routed by platform (spec: benchmark-core 平台抽象 &
methodology 可复现性清单).

Collects platform identity/temperature via the platform layer (host direct,
adb remote), plus engine version when an adapter can resolve it. Every field
carries its source command; failures are recorded as annotated errors rather
than guessed values.
"""
from __future__ import annotations

import datetime
import json
import os

from mobench.adapters.base import AdapterError, get_adapter
from mobench.platform.base import BasePlatform, PlatformError


def _best_effort(fn, label):
    try:
        return {"value": fn(), "source": label, "error": None}
    except (AdapterError, PlatformError, OSError) as exc:
        return {"value": None, "source": label, "error": str(exc)}


def snapshot(cfg: dict) -> dict:
    platform = BasePlatform.make(cfg)
    info = platform.preflight()  # raises PlatformError when platform unusable
    engine_ver = _best_effort(
        lambda: get_adapter(cfg).version(),
        f"adapter.version (engine={cfg.get('engine')})")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    binary = _binary_md5(cfg)
    models = _models_md5(cfg)
    return {
        "schema": "mobench/env@1",
        "timestamp": {"value": now, "source": "date"},
        "engine": {"value": cfg.get("engine"), "source": "mobench.yml engine"},
        "engine_version": engine_ver,
        "binary": binary,
        "models": models,
        "platform": info.get("platform", {}),
        **info,
    }


def _md5_of(path):
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _binary_md5(cfg):
    try:
        binary = get_adapter(cfg).resolve_binary()
        from mobench.artifacts_hash import artifact_set_md5
        return {"path": binary, "md5": artifact_set_md5(binary),
                "source": "artifact-set md5 (binary + *.so in its dir)"}
    except (AdapterError, OSError) as exc:
        return {"error": str(exc)}


def _models_md5(cfg):
    out = []
    for m in cfg.get("models") or []:
        try:
            out.append({"id": m["id"], "path": m["path"], "md5": _md5_of(m["path"]),
                        "source": f"md5sum {m['path']}"})
        except OSError as exc:
            out.append({"id": m["id"], "error": str(exc)})
    return out


def main(args) -> int:
    import sys

    from mobench.config import ConfigError, load_config

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"env: error: {exc}", file=sys.stderr)
        return 1
    try:
        snap = snapshot(cfg)
    except PlatformError as exc:
        print(f"env: error: {exc}", file=sys.stderr)
        return 1
    out_path = args.out
    if not out_path:
        results_dir = cfg.get("results_dir", "results")
        os.makedirs(results_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(results_dir, f"env_snapshot_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=2, ensure_ascii=False)
    if args.json or not args.out:
        print(json.dumps(snap, indent=2, ensure_ascii=False))
    else:
        print(f"Env snapshot: {out_path}")
    return 0
