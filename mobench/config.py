"""mobench.yml loading & validation (spec: benchmark-core 通用配置与 Schema 校验).

Config lives in the engine repo root (CWD-based invocation). All required
defaults are applied here so downstream code never reads a bare dict.
"""
from __future__ import annotations

import os

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
        "cooldown_per_round": False,
        "max_attempts": 1,
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

    raw_platform = merged["platform"]
    if raw_platform is None:
        merged["platform"] = {"type": "host"}
    elif isinstance(raw_platform, dict):
        merged["platform"] = {**raw_platform, "type": raw_platform.get("type", "host")}

    raw_workload = merged["workload"]
    if raw_workload is None:
        merged["workload"] = {"threads": [4], "rounds": 5}
    elif isinstance(raw_workload, dict):
        workload = {
            "threads": raw_workload.get("threads", [4]),
            "rounds": raw_workload.get("rounds", 5),
            **raw_workload,
        }
        # YAML 1.1 把未加引号的 on/off 解析为布尔（flash_attn 的合法取值恰是
        # on/off/auto）：还原为引擎参数词表值，避免静默丢弃 -fa 标志。
        flash_attn = workload.get("flash_attn")
        if isinstance(flash_attn, bool):
            workload = {**workload, "flash_attn": "on" if flash_attn else "off"}
        merged["workload"] = workload

    # methodology default nested: a partial section keeps the other defaults.
    methodology = merged.get("methodology")
    if "methodology" not in cfg or methodology is None:
        merged["methodology"] = dict(DEFAULTS["methodology"])
    elif isinstance(methodology, dict):
        merged["methodology"] = {**DEFAULTS["methodology"], **methodology}
    return merged


def _int_or_error(raw, whole):
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        raise ConfigError(
            f"pin_cpus has a non-numeric entry {raw!r} (in {whole!r})"
        )


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
                        f"pin_cpus range {part!r} is reversed (expected lo-hi)"
                    )
                cpus.extend(range(lo, hi + 1))
            else:
                cpus.append(_int_or_error(part, value))
    elif isinstance(value, (list, tuple)):
        cpus = [_int_or_error(v, value) for v in value]
    else:
        raise ConfigError(
            "pin_cpus must be a range string or a list of ints, got "
            f"{type(value).__name__}"
        )
    if not cpus:
        raise ConfigError("pin_cpus must select at least one CPU")
    if any(c < 0 for c in cpus):
        raise ConfigError(f"pin_cpus must be non-negative CPU indexes, got {value!r}")
    return sorted(set(cpus))


def _positive_int(raw) -> bool:
    return not isinstance(raw, bool) and isinstance(raw, int) and raw >= 1


def _non_neg_int(raw) -> bool:
    return not isinstance(raw, bool) and isinstance(raw, int) and raw >= 0


def _number(raw) -> bool:
    return not isinstance(raw, bool) and isinstance(raw, (int, float))


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

    platform = cfg.get("platform")
    if not isinstance(platform, dict):
        errors.append("platform must be a mapping")
    else:
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

    workload = cfg.get("workload")
    if not isinstance(workload, dict):
        errors.append("workload must be a mapping")
    else:
        threads = workload.get("threads")
        if (
            not isinstance(threads, list)
            or not threads
            or not all(_positive_int(t) for t in threads)
        ):
            errors.append(
                f"workload.threads must be a non-empty list of positive integers, got {threads!r}"
            )
        rounds = workload.get("rounds")
        if not _positive_int(rounds):
            errors.append(f"workload.rounds must be a positive integer, got {rounds!r}")
        for key in ("prompt_tokens", "gen_tokens", "n_depth"):
            val = workload.get(key)
            if val is not None and not _positive_int(val):
                errors.append(f"workload.{key} must be a positive integer, got {val!r}")
        cli_args = workload.get("cli_args")
        if cli_args is not None and (
            not isinstance(cli_args, list) or not all(isinstance(a, str) for a in cli_args)
        ):
            errors.append(f"workload.cli_args must be a list of strings, got {cli_args!r}")

    methodology = cfg.get("methodology")
    if methodology is not None and not isinstance(methodology, dict):
        errors.append("methodology must be a mapping")
    elif isinstance(methodology, dict):
        env_ctrl = methodology.get("env_control")
        if env_ctrl is not None and env_ctrl not in ("auto", "none"):
            errors.append(
                f"methodology.env_control must be 'auto' or 'none', got {env_ctrl!r}"
            )
        for key in ("cpu_pin", "lock_freq", "cooldown_per_round"):
            if key in methodology and not isinstance(methodology[key], bool):
                errors.append(
                    f"methodology.{key} must be a boolean, got {methodology[key]!r}"
                )
        cooldown_c = methodology.get("cooldown_c")
        if cooldown_c is not None and (not _number(cooldown_c) or cooldown_c < 0):
            errors.append(
                f"methodology.cooldown_c must be a non-negative number, got {cooldown_c!r}"
            )
        cooldown_max_s = methodology.get("cooldown_max_s")
        if cooldown_max_s is not None and not _non_neg_int(cooldown_max_s):
            errors.append(
                f"methodology.cooldown_max_s must be a non-negative integer, got {cooldown_max_s!r}"
            )
        max_attempts = methodology.get("max_attempts")
        if max_attempts is not None and not _positive_int(max_attempts):
            errors.append(
                f"methodology.max_attempts must be a positive integer, got {max_attempts!r}"
            )
        for key in ("freq_floor_mhz", "timeout_s"):
            val = methodology.get(key)
            if val is not None and (not _number(val) or val <= 0):
                errors.append(
                    f"methodology.{key} must be a positive number, got {val!r}"
                )
        # Format-only check: whether the indexes exist on this machine is
        # verified by the platform layer once CPUs are discovered.
        try:
            parse_cpus(methodology.get("pin_cpus"))
        except ConfigError as exc:
            errors.append(str(exc))

    errors.extend(_validation_problems(cfg.get("validation")))

    # 引擎专属配置校验证委派给对应适配器，避免核心硬编码引擎字段
    from mobench.adapters.base import AdapterError, get_adapter

    adapter_targets = []
    if engine in BUILTIN_ENGINES:
        adapter_targets.append(engine)
    for extra_engine in BUILTIN_ENGINES:
        if extra_engine != engine and cfg.get(extra_engine) is not None:
            adapter_targets.append(extra_engine)
    for target_engine in adapter_targets:
        try:
            errors.extend(get_adapter({**cfg, "engine": target_engine}).validate_config(cfg))
        except AdapterError as exc:
            errors.append(str(exc))

    return errors


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
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(p, str) and p.strip() for p in prompts)
    ):
        errors.append(
            "validation.prompts must be a non-empty list of prompt file paths"
        )
    gen_tokens = validation.get("gen_tokens")
    if gen_tokens is not None and not _positive_int(gen_tokens):
        errors.append(
            f"validation.gen_tokens must be a positive integer, got {gen_tokens!r}"
        )
    ppl = validation.get("ppl")
    if ppl is not None:
        if not isinstance(ppl, dict):
            errors.append("validation.ppl must be a mapping")
        else:
            dataset = ppl.get("dataset")
            if not (isinstance(dataset, str) and dataset.strip()):
                errors.append(
                    "validation.ppl.dataset is required (a text corpus file)"
                )
            for key in ("ctx", "chunks"):
                value = ppl.get(key)
                if value is not None and not _positive_int(value):
                    errors.append(
                        f"validation.ppl.{key} must be a positive integer, "
                        f"got {value!r}"
                    )
    if (not prompts) and ppl is None:
        errors.append(
            "validation declares no instrument: provide 'prompts' (greedy) "
            "and/or a 'ppl' block"
        )
    return errors


def load_config(path: str = DEFAULT_CONFIG_NAME) -> dict:
    """Load, default, and validate mobench.yml. Raises ConfigError on problems."""
    cfg = load_yaml(path)
    cfg = apply_defaults(cfg)
    problems = validate(cfg)
    if problems:
        raise ConfigError("invalid mobench.yml: " + "; ".join(problems))
    return cfg
