"""mobench.yml loading & validation (spec: benchmark-core 通用配置与 Schema 校验).

Config lives in the engine repo root (CWD-based invocation). All required
defaults are applied here so downstream code never reads a bare dict.
"""
from __future__ import annotations

import os
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - guarded at import for clear error
    yaml = None

DEFAULT_CONFIG_NAME = "mobench.yml"

# Engine keys the core ships. Adding a framework = adding an adapter module
# and extending this set (spec: benchmark-core 框架适配器与扩展注册).
BUILTIN_ENGINES = ("llamacpp", "mnn")
PLATFORMS = ("host", "adb")

DEFAULTS = {
    "platform": {"type": "host"},
    "results_dir": "results",
    "workload": {"threads": [4], "rounds": 5},
    # CPU 绑核与频率锁定默认启用：未声明 methodology 的既有档位同样受控，
    # 使测量条件（核集合/频率）可复现、可溯源（见 add-cpu-pin-lockfreq）。
    "methodology": {
        "env_control": "auto",
        "cpu_pin": True,
        "pin_cpus": None,   # None = 按线程数自动探测
        "lock_freq": True,
        # 可选冷却门（add-thermal-band-monitor）：统一「最热 zone」口径下的
        # 等待阈值与上限；缺省 None = 不等待（零行为，向后兼容）。
        "cooldown_c": None,
        "cooldown_max_s": 600,
    },
}


class ConfigError(Exception):
    """Raised for missing/invalid mobench.yml."""


def _require_yaml():
    if yaml is None:
        raise ConfigError("pyyaml required: pip install pyyaml")


def load_yaml(path: str) -> dict:
    _require_yaml()
    if not os.path.exists(path):
        raise ConfigError(f"{path} not found. Run from the engine repo root containing mobench.yml.")
    with open(path, encoding="utf-8") as fh:
        try:
            cfg = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return cfg


def apply_defaults(cfg: dict) -> dict:
    """Return a copy of ``cfg`` with defaults merged (never mutates input)."""
    merged = {k: cfg.get(k, v) for k, v in DEFAULTS.items()}
    for k, v in cfg.items():
        merged[k] = v
    # platform default nested
    platform = merged["platform"]
    merged["platform"] = {**platform, "type": platform.get("type", "host")}
    workload = merged["workload"]
    workload = {
        "threads": workload.get("threads") or [4],
        "rounds": workload.get("rounds", 5),
        **workload,
    }
    # YAML 1.1 把未加引号的 on/off 解析为布尔（flash_attn 的合法取值恰是
    # on/off/auto）：还原为引擎参数词表值，避免静默丢弃 -fa 标志。
    flash_attn = workload.get("flash_attn")
    if isinstance(flash_attn, bool):
        workload = {**workload, "flash_attn": "on" if flash_attn else "off"}
    merged["workload"] = workload
    # methodology default nested: a partial section keeps the other defaults.
    methodology = merged.get("methodology")
    if not isinstance(methodology, dict):
        methodology = {}
    merged["methodology"] = {**DEFAULTS["methodology"], **methodology}
    return merged


def _int_or_error(raw, whole):
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        raise ConfigError(
            f"pin_cpus has a non-numeric entry {raw!r} (in {whole!r})")


def parse_cpus(value) -> list:
    """Parse ``pin_cpus`` into a sorted, de-duplicated CPU index list.

    Accepts ``None`` (auto-detect -> ``[]``), a range/CSV string
    (``"4-7"`` / ``"4,5,6,7"``) or a list of ints. Raises ConfigError on
    malformed input so a bad selection is refused before any measurement.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ConfigError("pin_cpus must not be an empty string")
        cpus = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                raise ConfigError(f"pin_cpus has an empty entry (in {value!r})")
            if "-" in part:
                lo_raw, _, hi_raw = part.partition("-")
                lo, hi = _int_or_error(lo_raw, value), _int_or_error(hi_raw, value)
                if hi < lo:
                    raise ConfigError(
                        f"pin_cpus range {part!r} is reversed (expected lo-hi)")
                cpus.extend(range(lo, hi + 1))
            else:
                cpus.append(_int_or_error(part, value))
    elif isinstance(value, (list, tuple)):
        cpus = [_int_or_error(v, value) for v in value]
    else:
        raise ConfigError(
            "pin_cpus must be a range string or a list of ints, got "
            f"{type(value).__name__}")
    if not cpus:
        raise ConfigError("pin_cpus must select at least one CPU")
    if any(c < 0 for c in cpus):
        raise ConfigError(f"pin_cpus must be non-negative CPU indexes, got {value!r}")
    return sorted(set(cpus))


def validate(cfg: dict) -> list:
    """Return human-readable problems; empty list means valid."""
    errors = []
    engine = cfg.get("engine")
    if not engine:
        errors.append("missing 'engine' (one of " + ", ".join(BUILTIN_ENGINES) + ")")
    elif engine not in BUILTIN_ENGINES:
        errors.append(
            f"unsupported engine {engine!r}; registered adapters: "
            + ", ".join(BUILTIN_ENGINES)
        )
    platform = cfg.get("platform") or {}
    ptype = platform.get("type")
    if ptype not in PLATFORMS:
        errors.append(f"platform.type must be one of {PLATFORMS}, got {ptype!r}")
    if ptype == "adb":
        if not platform.get("adb"):
            errors.append("platform.adb path required when platform.type=adb")
        if not platform.get("device_id"):
            errors.append("platform.device_id required when platform.type=adb")
    models = cfg.get("models")
    if not isinstance(models, list) or not models:
        errors.append("missing non-empty 'models' list")
    else:
        for i, m in enumerate(models):
            if not isinstance(m, dict):
                errors.append(f"models[{i}] must be a mapping")
                continue
            if not m.get("id"):
                errors.append(f"models[{i}].id is required")
            if not m.get("path"):
                errors.append(f"models[{i}].path is required")
            if m.get("dir") is not None and not isinstance(m.get("dir"), str):
                errors.append(f"models[{i}].dir must be a string path when present")
            if m.get("kind") not in (None, "llm", "cv", "nlp"):
                errors.append(f"models[{i}].kind must be llm|cv|nlp, got {m.get('kind')!r}")
    methodology = cfg.get("methodology") or {}
    if isinstance(methodology, dict):
        for key in ("cpu_pin", "lock_freq"):
            if key in methodology and not isinstance(methodology[key], bool):
                errors.append(
                    f"methodology.{key} must be a boolean, "
                    f"got {methodology[key]!r}")
        # Format-only check: whether the indexes exist on this machine is
        # verified by the platform layer once CPUs are discovered.
        try:
            parse_cpus(methodology.get("pin_cpus"))
        except ConfigError as exc:
            errors.append(str(exc))
    mnn = cfg.get("mnn") or {}
    if isinstance(mnn, dict) and mnn.get("n_repeat") is not None:
        n_repeat = mnn["n_repeat"]
        if isinstance(n_repeat, bool) or not isinstance(n_repeat, int) or n_repeat < 1:
            errors.append(
                f"mnn.n_repeat must be a positive integer (llm_bench -rep), "
                f"got {n_repeat!r}")
    errors.extend(_validation_problems(cfg.get("validation")))
    return errors


def _positive_int(raw) -> bool:
    return (not isinstance(raw, bool) and isinstance(raw, int) and raw >= 1)


def _validation_problems(validation) -> list:
    """Problems in the optional ``validation`` section (mobench validate).

    Absent section = the config has no validation capability (no defaults are
    injected). A declared section must declare at least one instrument; every
    declared value must be usable as-is (paths are checked at run time).
    """
    if validation is None:
        return []
    if not isinstance(validation, dict):
        return ["validation must be a mapping"]
    errors = []
    prompts = validation.get("prompts")
    if prompts is not None and (
            not isinstance(prompts, list) or not prompts
            or not all(isinstance(p, str) and p.strip() for p in prompts)):
        errors.append(
            "validation.prompts must be a non-empty list of prompt file paths")
    gen_tokens = validation.get("gen_tokens")
    if gen_tokens is not None and not _positive_int(gen_tokens):
        errors.append(
            f"validation.gen_tokens must be a positive integer, got {gen_tokens!r}")
    ppl = validation.get("ppl")
    if ppl is not None:
        if not isinstance(ppl, dict):
            errors.append("validation.ppl must be a mapping")
        else:
            dataset = ppl.get("dataset")
            if not (isinstance(dataset, str) and dataset.strip()):
                errors.append(
                    "validation.ppl.dataset is required (a text corpus file)")
            for key in ("ctx", "chunks"):
                value = ppl.get(key)
                if value is not None and not _positive_int(value):
                    errors.append(
                        f"validation.ppl.{key} must be a positive integer, "
                        f"got {value!r}")
    if (not prompts) and ppl is None:
        errors.append(
            "validation declares no instrument: provide 'prompts' (greedy) "
            "and/or a 'ppl' block")
    return errors


def load_config(path: str = DEFAULT_CONFIG_NAME) -> dict:
    """Load, default, and validate mobench.yml. Raises ConfigError on problems."""
    cfg = load_yaml(path)
    cfg = apply_defaults(cfg)
    problems = validate(cfg)
    if problems:
        raise ConfigError("invalid mobench.yml: " + "; ".join(problems))
    return cfg
