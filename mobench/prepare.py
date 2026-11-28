"""Model preparation (spec: benchmark-model-prep).

Three states: a model may be self-provided (validate + hash), downloaded from
a source URL with optional sha256 verification, or converted/quantized by the
engine's own tools (llama.cpp convert_hf_to_gguf.py / MNN MNNConvert). Every
artifact records its source command and hash for reproducibility.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request

from mobench.adapters.base import AdapterError, get_adapter
from mobench.io_utils import file_md5, file_sha256

MODEL_EXTENSIONS = (".gguf", ".onnx", ".mnn", ".bin", ".param", ".llamafile")
DEFAULT_DOWNLOAD_TIMEOUT = 60
ALLOWED_URL_SCHEMES = ("http", "https")
CONVERTER_ADAPTER_MAP = {
    "llama.cpp": "llamacpp",
    "mnn": "mnn",
}


class PrepareError(Exception):
    """Model preparation failure (missing file, hash mismatch, tool error)."""


def download(url, dest, timeout=DEFAULT_DOWNLOAD_TIMEOUT):
    """Stream ``url`` to ``dest`` atomically using urllib (http/https only)."""
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in ALLOWED_URL_SCHEMES:
        raise PrepareError(
            f"unsupported URL scheme {scheme!r}: only http/https allowed"
        )
    abs_dest = os.path.abspath(dest)
    os.makedirs(os.path.dirname(abs_dest), exist_ok=True)
    tmp_dest = f"{abs_dest}.part"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(
            tmp_dest, "wb"
        ) as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        os.replace(tmp_dest, abs_dest)
    except BaseException:
        if os.path.exists(tmp_dest):
            os.remove(tmp_dest)
        raise


def _finalize(dest, source_cmd=None):
    if not os.path.isfile(dest):
        raise PrepareError(f"prepare produced no file at {dest}")
    return {
        "status": "ready",
        "path": dest,
        "size": os.path.getsize(dest),
        "sha256": file_sha256(dest),
        "md5": file_md5(dest),
        "source_cmd": source_cmd,
    }


def prepare_model(cfg, model, dry_run=False):
    """Ensure a model is available; returns an artifact-info dict.

    Raises PrepareError on missing source, hash mismatch, or tool failure.
    """
    source = model.get("source") or {}
    out_dir = os.path.dirname(os.path.abspath(model.get("path", ".")))
    os.makedirs(out_dir, exist_ok=True)

    convert_with = source.get("convert_with")
    if convert_with:
        adapter_engine = CONVERTER_ADAPTER_MAP.get(convert_with)
        if adapter_engine is None:
            raise PrepareError(
                f"unsupported convert_with {convert_with!r}; "
                f"known: {', '.join(CONVERTER_ADAPTER_MAP)}"
            )
        try:
            adapter = get_adapter({**cfg, "engine": adapter_engine})
            argv = adapter.convert_model(model)
        except AdapterError as exc:
            raise PrepareError(str(exc)) from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise PrepareError(
                f"model conversion failed (rc={exc.returncode}): {detail}"
            ) from exc
        return _finalize(model["path"], source_cmd=" ".join(map(str, argv)))

    if source.get("url"):
        dest = model["path"]
        if dry_run:
            return {"status": "would_download", "path": dest}
        download(source["url"], dest)
        if source.get("sha256"):
            actual = file_sha256(dest)
            if actual != source["sha256"].lower():
                os.remove(dest)
                raise PrepareError(
                    f"sha256 mismatch for {dest}: got {actual}, "
                    f"expected {source['sha256'].lower()} (artifact discarded)"
                )
        return _finalize(dest, source_cmd=f"download {source['url']}")

    # self-provided
    if os.path.isfile(model["path"]):
        return _finalize(model["path"])
    raise PrepareError(
        f"model file missing: {model['path']}. Provide the file or add "
        f"models[].source (url / convert_with) in mobench.yml."
    )


def main(args) -> int:
    from mobench.config import ConfigError, load_config

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"prepare: error: {exc}", file=sys.stderr)
        return 1
    models = cfg["models"]
    if args.model_ids:
        models = [m for m in models if m["id"] in args.model_ids]
    if not models:
        print("prepare: error: no models to prepare", file=sys.stderr)
        return 1
    results = []
    for m in models:
        try:
            info = prepare_model(cfg, m)
            info = {**m, **info}
            if not args.json:
                print(
                    f"[prepare] {m['id']}: {info['status']} -> {info['path']} "
                    f"(sha256 {info['sha256'][:12]}…)"
                )
        except PrepareError as exc:
            if args.json:
                results.append({"id": m["id"], "status": "error", "error": str(exc)})
            else:
                print(f"[prepare] {m['id']}: error: {exc}", file=sys.stderr)
            return 1
        results.append({"id": m["id"], "status": info["status"], "path": info["path"]})
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0
