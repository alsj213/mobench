"""Model preparation (spec: benchmark-model-prep).

Three states: a model may be self-provided (validate + hash), downloaded from
a source URL with optional sha256 verification, or converted/quantized by the
engine's own tools (llama.cpp convert_hf_to_gguf.py / MNN MNNConvert). Every
artifact records its source command and hash for reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.request

MODEL_EXTENSIONS = (".gguf", ".onnx", ".mnn", ".bin", ".param", ".llamafile")


class PrepareError(Exception):
    """Model preparation failure (missing file, hash mismatch, tool error)."""


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    """Stream ``url`` to ``dest`` using urllib (no third-party dep)."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)


def _convert_llama_cpp(cfg, model):
    """convert_hf_to_gguf.py + optional quantize from the llama.cpp source dir."""
    source = model.get("source") or {}
    src_dir = cfg.get("build", {}).get("source_dir") or "."
    tool = os.path.join(src_dir, "convert_hf_to_gguf.py")
    if not os.path.isfile(tool):
        raise PrepareError(f"llama.cpp convert tool not found: {tool} (engine repo missing?)")
    hf_dir = source.get("dir")
    if not hf_dir:
        raise PrepareError("convert_with=llama.cpp requires source.dir (HF model directory)")
    out = model["path"]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    argv = [sys.executable, tool, hf_dir, "-o", out]
    if source.get("outtype"):
        argv += ["--outtype", source["outtype"]]
    subprocess.run(argv, check=True, capture_output=True, text=True)
    return argv


def _convert_mnn(cfg, model):
    """MNNConvert (onnx -> mnn) from the MNN build dir."""
    source = model.get("source") or {}
    build = cfg.get("build") or {}
    tool = os.path.join(build.get("bin_dir", ""), "MNNConvert")
    if not os.path.isfile(tool):
        raise PrepareError(f"MNNConvert not found: {tool} (set build.bin_dir)")
    onnx = source.get("dir") or source.get("onnx")
    if not onnx:
        raise PrepareError("convert_with=mnn requires source.dir (onnx file)")
    out = model["path"]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    argv = [tool, "--modelFile", onnx, "--MNNModel", out]
    subprocess.run(argv, check=True, capture_output=True, text=True)
    return argv


CONVERTERS = {
    "llama.cpp": _convert_llama_cpp,
    "mnn": _convert_mnn,
}


def _finalize(dest, source_cmd=None):
    if not os.path.isfile(dest):
        raise PrepareError(f"prepare produced no file at {dest}")
    return {"status": "ready", "path": dest, "size": os.path.getsize(dest),
            "sha256": _sha256(dest), "md5": _md5(dest), "source_cmd": source_cmd}


def prepare_model(cfg, model, dry_run=False):
    """Ensure a model is available; returns an artifact-info dict.

    Raises PrepareError on missing source, hash mismatch, or tool failure.
    """
    source = model.get("source") or {}
    out_dir = os.path.dirname(os.path.abspath(model.get("path", ".")))
    os.makedirs(out_dir, exist_ok=True)

    if source.get("convert_with"):
        conv = CONVERTERS.get(source["convert_with"])
        if conv is None:
            raise PrepareError(
                f"unsupported convert_with {source['convert_with']!r}; "
                f"known: {', '.join(CONVERTERS)}")
        argv = conv(cfg, model)
        return _finalize(model["path"], source_cmd=" ".join(map(str, argv)))

    if source.get("url"):
        dest = model["path"]
        if dry_run:
            return {"status": "would_download", "path": dest}
        download(source["url"], dest)
        if source.get("sha256"):
            actual = _sha256(dest)
            if actual != source["sha256"].lower():
                os.remove(dest)
                raise PrepareError(
                    f"sha256 mismatch for {dest}: got {actual}, "
                    f"expected {source['sha256'].lower()} (artifact discarded)")
        return _finalize(dest, source_cmd=f"download {source['url']}")

    # self-provided
    if os.path.isfile(model["path"]):
        return _finalize(model["path"])
    raise PrepareError(
        f"model file missing: {model['path']}. Provide the file or add "
        f"models[].source (url / convert_with) in mobench.yml.")


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
                print(f"[prepare] {m['id']}: {info['status']} -> {info['path']} "
                      f"(sha256 {info['sha256'][:12]}…)")
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
