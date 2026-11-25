"""Build-recipe fingerprint check (spec: 测量有效性门禁).

Consumes ``config.build.required_cmake_args``: parses the ``-DKEY=VALUE``
list and verifies each setting is actually baked into the CMake build tree
that produced ``bin_dir`` (read from its CMakeCache.txt). A mismatch — or a
cache that cannot be found — rejects the run up front.

Motivated by the 2026-10-11 K8 incident: a build missing dotprod/fp16
silently deactivated the repack path (311 tensors refused, "cannot be used
with CPU_REPACK"), and a real optimization was measured as "no gain". The
declaration existed in config but nothing consumed it.
"""
from __future__ import annotations

import os
import shlex

_BOOL_TRUTHY = {"ON", "TRUE", "YES", "1", "Y"}
_BOOL_WORDS = _BOOL_TRUTHY | {"OFF", "FALSE", "NO", "0", "N"}


class BuildRecipeError(Exception):
    """required_cmake_args declared but not verifiably baked into the build."""


def parse_required_args(spec: str) -> dict:
    """``'-DKEY=VALUE -DOTHER:STRING=1'`` -> ``{KEY: VALUE, ...}``.

    Only ``-D`` tokens are accepted; anything else is a config error and is
    rejected loudly (a silently-ignored token would defeat the check).
    """
    out = {}
    for tok in shlex.split(spec or ""):
        if not tok.startswith("-D") or "=" not in tok:
            raise BuildRecipeError(
                f"required_cmake_args 条目非法: {tok!r}（应为 -DKEY=VALUE 形态）")
        key, _, value = tok[2:].partition("=")
        key = key.split(":", 1)[0]  # -DKEY:TYPE=VALUE -> KEY
        if not key:
            raise BuildRecipeError(f"required_cmake_args 条目缺少 KEY: {tok!r}")
        out[key] = value
    return out


def find_cmake_cache(bin_dir: str):
    """Locate the CMakeCache.txt of the tree whose bin dir is ``bin_dir``.

    Candidates: ``dirname(bin_dir)/CMakeCache.txt`` (bin_dir = <build>/bin)
    then ``bin_dir/CMakeCache.txt`` (bin_dir = build root). None if absent.
    """
    for cand in (os.path.join(os.path.dirname(bin_dir), "CMakeCache.txt"),
                 os.path.join(bin_dir, "CMakeCache.txt")):
        if cand and os.path.isfile(cand):
            return cand
    return None


def read_cmake_cache(path: str) -> dict:
    """Parse ``KEY:TYPE=VALUE`` lines into ``{KEY: VALUE}``.

    Skips comments (``//``), blanks, and lines without a typed key. Values
    keep their raw spelling (case matters for strings like arch flags).
    """
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            key_type, sep, value = line.partition("=")
            if not sep or ":" not in key_type:
                continue
            key, _, _type = key_type.partition(":")
            out[key.strip()] = value.strip()
    return out


def _values_match(required: str, actual: str) -> bool:
    if required == actual:
        return True
    # CMake normalizes boolean spellings (ON/TRUE/1/YES...); accept equivalents.
    if required.upper() in _BOOL_WORDS and actual.upper() in _BOOL_WORDS:
        return (required.upper() in _BOOL_TRUTHY) == (actual.upper() in _BOOL_TRUTHY)
    return False


def verify_build_recipe(cfg) -> dict:
    """Check ``cfg['build']['required_cmake_args']`` against the tree's cache.

    Returns a run.json-ready record on success; raises BuildRecipeError on
    any mismatch or when the recipe cannot be verified. With no declaration
    this is a zero-behavior no-op (``{'checked': False, ...}``).
    """
    build = (cfg or {}).get("build") or {}
    spec = build.get("required_cmake_args")
    if not spec:
        return {"checked": False, "reason": "no required_cmake_args declared"}
    required = parse_required_args(spec)
    bin_dir = build.get("bin_dir")
    if not bin_dir:
        raise BuildRecipeError(
            "声明了 required_cmake_args 但未设 build.bin_dir——无法定位构建树校验口径")
    cache_path = find_cmake_cache(bin_dir)
    if cache_path is None:
        raise BuildRecipeError(
            f"找不到 CMakeCache.txt（bin_dir={bin_dir}）——无法验证构建口径。"
            "若 bin_dir 非 cmake 构建产物：改用 cmake 构建、或移除 required_cmake_args 声明")
    cache = read_cmake_cache(cache_path)
    mismatches = []
    for key, want in required.items():
        got = cache.get(key)
        if got is None:
            mismatches.append(f"{key}: 期望 {want!r}，cache 中不存在（未设置）")
        elif not _values_match(want, got):
            mismatches.append(f"{key}: 期望 {want!r}，实际 {got!r}")
    if mismatches:
        raise BuildRecipeError(
            "构建口径不符（" + os.path.abspath(cache_path) + "）：\n  - "
            + "\n  - ".join(mismatches)
            + "\n测量将无效（优化路径可能未激活）。修复：用正确 -D 参数重建，或修正 config。")
    return {"checked": True, "cache": cache_path,
            "verified": {k: cache[k] for k in required},
            "n_checked": len(required)}
