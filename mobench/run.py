"""Benchmark run pipeline (spec: benchmark-core 执行流水线与报告).

Order: config -> platform preflight/temp -> engine version & artifact hashes ->
model check -> env control -> execute (tee) x round -> restore -> records.
Any failed step aborts and reports which step failed; raw engine output is
always tee'd so every record traces to a non-empty source log.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import time

from mobench import build_recipe, log_probe, records as recmod
from mobench.adapters.base import AdapterError, get_adapter
from mobench.artifacts_hash import artifact_set_md5
from mobench.config import BUILTIN_ENGINES, ConfigError, load_config
from mobench.platform.base import BasePlatform, PlatformError

TEMP_THROTTLE_C = 45.0
STEPS = ("config", "platform preflight", "binary probe", "build recipe",
         "model check", "env control", "execution", "restore", "records")


class RunError(Exception):
    """A pipeline step failed (carries the step name)."""

    def __init__(self, step, message):
        super().__init__(f"[step:{step}] {message}")
        self.step = step


def _model_md5(platform, path):
    try:
        return platform.file_md5(path)
    except OSError as exc:
        return {"error": str(exc)}


def _cooldown_wait(platform, methodology, emit, sleep=time.sleep, interval=30):
    """Wait (optionally) until the hottest zone ≤ methodology.cooldown_c.

    Fail-open：超时带警告继续（reached=false 如实记录），绝不因等待而失败。
    2026-10-09 热漂移实证：连续跑测让「快档」自身漂移 >1%（4.605→4.573→
    4.670ms），run 协议需要冷却间隔才能保住 <1% 判决的可信度
    （add-thermal-band-monitor）。未配置 cooldown_c 时为零行为（返回 None）。
    """
    target = methodology.get("cooldown_c")
    if target is None:
        return None
    max_s = methodology.get("cooldown_max_s", 600)
    waited = 0
    while True:
        temp = platform.sample_thermal()["temp_c"].get("value")
        if temp is None or temp <= target:
            return {"target_c": target, "waited_s": waited,
                    "reached": True, "temp_c": temp}
        if waited >= max_s:
            return {"target_c": target, "waited_s": waited,
                    "reached": False, "temp_c": temp}
        emit("cooldown",
             f"hottest zone {temp}°C > {target}°C; sleeping {interval}s "
             f"({waited + interval}/{max_s}s max)")
        sleep(interval)
        waited += interval


def run_pipeline(cfg, model_filter=None, threads_override=None,
                 rounds_override=None, json_mode=False, label=None) -> dict:
    """Execute the pipeline; returns run metadata. Raises on failure."""
    # -- config / preflight --------------------------------------------------
    platform = BasePlatform.make(cfg)
    try:
        info = platform.preflight()
    except PlatformError as exc:
        raise RunError("platform preflight", str(exc))
    temp_c = info.get("temp_c", {}).get("value") if info.get("temp_c") else None
    throttled = temp_c is not None and temp_c > TEMP_THROTTLE_C

    # -- adapter / binary / engine version -----------------------------------
    try:
        adapter = get_adapter(cfg)
        binary = adapter.resolve_binary()
        engine_version = adapter.version()
    except AdapterError as exc:
        raise RunError("binary probe", str(exc))
    if not platform.file_exists(binary):
        raise RunError("binary probe", f"benchmark binary not found: {binary}")

    # -- build recipe fingerprint (spec: 测量有效性门禁) ------------------------
    # required_cmake_args 声明必须被真正验证（K8 事故：缺 dotprod 的构建让
    # repack 路径静默不激活，「零收益」是无效测量而非负结果）。
    try:
        build_recipe_info = build_recipe.verify_build_recipe(cfg)
    except build_recipe.BuildRecipeError as exc:
        raise RunError("build recipe", str(exc))

    # -- models --------------------------------------------------------------
    models = list(cfg["models"])
    if model_filter:
        wanted = set(model_filter)
        models = [m for m in models if m["id"] in wanted]
    if not models:
        raise RunError("model check", "no models match the requested filter")
    for m in models:
        if not platform.file_exists(m["path"]):
            raise RunError(
                "model check",
                f"model file missing: {m['path']}. Run `mobench prepare` first.",
            )

    # -- run dir & hashes ------------------------------------------------------
    results_dir = cfg.get("results_dir", "results")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    records_path = os.path.join(run_dir, "records.jsonl")
    threads_list = threads_override or cfg["workload"]["threads"]
    rounds = rounds_override or cfg["workload"]["rounds"]

    meta = {
        "schema": "mobench/run@1",
        "timestamp": ts,
        "engine": cfg["engine"],
        "engine_version": engine_version,
        "binary": {"path": binary, "md5": artifact_set_md5(binary),
                   "md5_scope": "artifact_set(binary+*.so)"},
        "build_recipe": build_recipe_info,
        "models": [
            {"id": m["id"], "path": m["path"], "md5": _model_md5(platform, m["path"])}
            for m in models
        ],
        "platform": info,
        "throttle_risk": throttled,
        "records": records_path,
        "n_records": 0,
    }
    if label:
        meta["label"] = label

    # -- log probes (optional; compile before any device time is spent) --------
    try:
        log_checks = log_probe.compile_checks(cfg.get("methodology") or {})
    except log_probe.LogProbeError as exc:
        raise RunError("config", str(exc))

    # -- thermal frequency floor (optional; 跑后采样判违规) ---------------------
    freq_floor = (cfg.get("methodology") or {}).get("freq_floor_mhz")

    # -- env control ----------------------------------------------------------
    platform.control(cfg, meta)

    # -- execute: threads x rounds, tee raw logs ------------------------------
    def _emit(step, message):
        # Progress stays off stdout in --json so stdout is pure JSON (P3).
        stream = sys.stderr if json_mode else sys.stdout
        print(f"[run] step:{step} {message}", file=stream)

    # -- cooldown gate (optional; fail-open) ----------------------------------
    cooldown = _cooldown_wait(platform, cfg.get("methodology") or {}, _emit)
    if cooldown is not None:
        _emit("cooldown",
              f"target ≤{cooldown['target_c']}°C reached={cooldown['reached']} "
              f"after {cooldown['waited_s']}s (now {cooldown['temp_c']}°C)")

    conditions = {}
    log_violations = []
    freq_violations = []
    try:
        for m in models:
            for threads in threads_list:
                # Condition is per thread count: the selected CPU set depends
                # on it (N cores, whole clusters preferred).
                try:
                    condition = platform.measure_condition(threads)
                except PlatformError as exc:
                    raise RunError("env control", str(exc))
                conditions[f"t{threads}"] = condition
                _emit("env control",
                      f"t{threads} pin={condition['pin']['cpus'] or condition['pin']['origin']}"
                      f" freq_locked={condition['freq']['locked']}")
                for rnd in range(1, rounds + 1):
                    cmd = adapter.build_cmds(m, threads, rounds=1)[0]
                    # 每轮跑测前实时采样性能档（热状态是测量条件的一部分；
                    # 2026-10-09 热漂移曾使 <1% 判决失效——add-thermal-band-monitor）
                    band = platform.sample_thermal()
                    log_path = os.path.join(
                        run_dir, f"raw_{m['id']}_t{threads}_r{rnd}.log")
                    _emit("execution", " ".join(cmd))
                    rc = platform.run(cmd, log_path, echo=not json_mode)
                    if rc != 0:
                        raise RunError("execution",
                                       f"engine exited {rc}; raw log: {log_path}")
                    if freq_floor is not None:
                        # 跑后紧贴采样：热降频发生在这里（THERMAL-DIRTY 事故：
                        # 5 连跑降到 1862MHz 的数据直到跑完才被人工发现）。
                        after_band = platform.sample_thermal()
                        mhz = (after_band.get("cpu_freq") or {}).get("value_mhz")
                        if mhz is not None and mhz < freq_floor:
                            freq_violations.append({
                                "model": m["id"], "threads": threads, "round": rnd,
                                "freq_mhz": mhz, "floor_mhz": freq_floor})
                    if log_checks:
                        forb = log_probe.scan_forbid(log_path, log_checks)
                        if forb:
                            v = forb[0]
                            raise RunError(
                                "execution",
                                f"日志命中禁项 {v['pattern']!r}"
                                f"（line {v['line']}: {v['matched']!r}）"
                                f"——该 run 无效，中止；raw log: {log_path}")
                        for v in log_probe.scan_expect(log_path, log_checks):
                            v.update({"model": m["id"], "threads": threads,
                                      "round": rnd})
                            log_violations.append(v)
                    try:
                        parsed = adapter.parse(log_path, m, threads, rnd)
                    except AdapterError as exc:
                        raise RunError("execution", str(exc))
                    for rec in parsed:
                        rec["platform"] = platform.name  # actual run platform (host/adb)
                        rec["device"] = {
                            "temp_c": band["temp_c"].get("value"),
                            "temp_zone": band["temp_c"].get("zone"),
                            "cpu_freq_mhz": band["cpu_freq"].get("value_mhz"),
                            "platform": info.get("platform", {}).get("value"),
                            # Measurement condition travels with every value,
                            # so a report can always state what produced it.
                            "pin": condition["pin"],
                            "freq": condition["freq"],
                        }
                        recmod.append_record(records_path, rec)
                        meta["n_records"] += 1
    finally:
        platform.restore()

    meta["measurement_conditions"] = conditions
    if cooldown is not None:
        meta["cooldown"] = cooldown
    if log_checks:
        meta["log_audit"] = {"valid": not log_violations,
                             "violations": log_violations}
    if freq_floor is not None:
        meta["freq_audit"] = {"valid": not freq_violations,
                              "violations": freq_violations}
    meta_path = os.path.join(run_dir, "run.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    return meta


def main(args) -> int:
    try:
        cfg = load_config(args.config)
        if args.engine:
            if args.engine not in BUILTIN_ENGINES:
                raise ConfigError(
                    f"unsupported engine {args.engine!r}; registered: "
                    + ", ".join(BUILTIN_ENGINES)
                )
            cfg = {**cfg, "engine": args.engine}
        meta = run_pipeline(cfg, model_filter=args.model,
                            threads_override=[args.threads] if args.threads else None,
                            rounds_override=args.rounds, json_mode=args.json,
                            label=getattr(args, "label", None))
    except (ConfigError, RunError, AdapterError, PlatformError) as exc:
        print(f"run: error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(meta, indent=2, ensure_ascii=False))
    else:
        warn = "  [WARN: 降频风险 temp>45C]" if meta["throttle_risk"] else ""
        audit = meta.get("log_audit")
        if audit and not audit["valid"]:
            warn += (f"  [WARN: 日志探针 violations={len(audit['violations'])}"
                     f"，该 run 标记无效]")
        fa = meta.get("freq_audit")
        if fa and not fa["valid"]:
            warn += (f"  [WARN: 频门禁 violations={len(fa['violations'])}"
                     f"，该 run 标记无效]")
        print(f"[run] ok: {meta['n_records']} records -> {meta['records']}{warn}")
    return 0
