"""Adapter contract & lookup (spec: benchmark-core 框架适配器与扩展注册).

Contract: ``resolve_binary`` (probe) -> ``version`` -> ``build_cmds`` ->
``parse`` (raw engine output -> canonical records). Each adapter declares its
metric vocabulary. Engine knowledge never leaks into run/report/analysis.
"""
from __future__ import annotations

import importlib
import os
import shutil
from typing import List


class AdapterError(Exception):
    """Engine probe/build/parse failure (missing binary, unparseable output)."""


def _friendly(engine: str) -> str:
    return {"llamacpp": "llama.cpp", "mnn": "MNN"}.get(engine, engine)


class EngineAdapter:
    engine = None

    def __init__(self, cfg: dict):
        self.cfg = cfg

    # -- probe ---------------------------------------------------------------
    def resolve_binary(self) -> str:
        """Return the benchmark executable path or raise AdapterError."""
        raise NotImplementedError

    def version(self) -> str:
        """Return an engine version string (commit / --version)."""
        raise NotImplementedError

    # -- measure -------------------------------------------------------------
    def build_cmds(self, model: dict, threads: int, rounds: int) -> List[List[str]]:
        """Return the engine commands to run for one outer round."""
        raise NotImplementedError

    def parse(self, log_path: str, model: dict, threads: int, round_no: int) -> List[dict]:
        """Parse a tee'd raw log into canonical records."""
        raise NotImplementedError

    # -- precision validation (optional contract, add-correctness-validation) --
    # Engines that provide precision instruments implement both methods. The
    # defaults raise a clear "not implemented" error: the validate flow marks
    # such engines unavailable — never a silent downgrade or fabricated value.
    def build_validate_cmds(self, model: dict, val_cfg: dict, workload: dict,
                            threads: int) -> List[dict]:
        """Instrument jobs for precision validation of ``model``.

        Returns a list of jobs, each ``{"instrument": <name>, "cmd": [argv…],
        "prompt": <path|None>}``; unparseable/unsupported → AdapterError.
        """
        raise AdapterError(
            f"engine {self.engine!r} does not implement the validation "
            f"contract (no precision instruments)")

    def parse_validate(self, job: dict, log_path: str) -> dict:
        """Parse one executed instrument job's raw log.

        Returns a per-instrument result dict. Unparseable output MUST raise
        AdapterError so the caller marks the instrument unavailable with the
        reason — values are never guessed.
        """
        raise AdapterError(
            f"engine {self.engine!r} does not implement the validation "
            f"contract (no precision instruments)")


def _find_binary(cfg: dict, name: str) -> str:
    """Resolve an engine bench binary from config then PATH."""
    build = cfg.get("build") or {}
    bin_dir = build.get("bin_dir")
    candidates = []
    if bin_dir:
        candidates.append(os.path.join(bin_dir, name))
    if name and os.path.sep in name:
        candidates.append(name)
    candidates.append(shutil.which(name) or name)
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return candidates[0]


from mobench.config import BUILTIN_ENGINES


def get_adapter(cfg: dict) -> EngineAdapter:
    """Return an adapter instance for cfg['engine'] (raises AdapterError)."""
    engine = cfg.get("engine")
    if not engine:
        raise AdapterError("missing engine in config")
    try:
        module = importlib.import_module(f"mobench.adapters.{engine}")
    except ImportError as exc:
        raise AdapterError(
            f"unsupported engine {engine!r}; registered adapters: "
            + ", ".join(BUILTIN_ENGINES)
        ) from exc
    adapter_cls = getattr(module, "ADAPTER", None)
    if adapter_cls is None:
        raise AdapterError(f"adapter module {engine} has no ADAPTER class")
    return adapter_cls(cfg)
