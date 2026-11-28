#!/usr/bin/env python3
"""Control helper for the ``verify-mobench`` skill.

The skill is a thin, repo-local verification layer over the real ``mobench``
CLI. This helper only:

* ``write-config`` renders a disposable Android llama.cpp config from env vars,
* ``doctor`` performs read-only preflight checks against that config,
* ``prove`` drives one mapped feature end to end and retains evidence,
* ``cleanup`` checks that the short-lived run left no process/governor behind.

It never edits ``mobench/`` and it never commits anything.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SKILL_REL = Path(".agents/skills/verify-mobench")
DEFAULT_RESULTS = "results/verify-mobench"
DEFAULT_CONFIG = "configs/local/verify-mobench.yml"
TEMP_LIMIT_C = 45.0
DEFAULT_CMAKE_ARGS = "-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod"


def repo_root() -> Path:
    """Find the checkout even if the helper is invoked from another cwd."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (
            parent / "mobench" / "cli.py"
        ).is_file():
            return parent
    raise SystemExit("verify-mobench: run this helper from a mobench checkout")


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def config_path(explicit: str | None = None) -> Path:
    root = repo_root()
    raw = explicit or env("MOBENCH_VERIFY_CONFIG") or DEFAULT_CONFIG
    path = Path(raw)
    return path if path.is_absolute() else root / path


def results_dir() -> str:
    return env("MOBENCH_VERIFY_RESULTS", DEFAULT_RESULTS) or DEFAULT_RESULTS


def _run(cmd, cwd: Path, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _shell(adb: str, device: str, script: str, timeout: float = 60) -> subprocess.CompletedProcess:
    return _run([adb, "-s", device, "shell", script], cwd=repo_root(), timeout=timeout)


def _adb_devices(adb: str) -> list[tuple[str, str]]:
    proc = _run([adb, "devices"], cwd=repo_root(), timeout=30)
    rows = []
    for line in proc.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            rows.append((parts[0], parts[1]))
    return rows


def _online_device(adb: str) -> str | None:
    wanted = env("MOBENCH_VERIFY_DEVICE")
    online = [serial for serial, state in _adb_devices(adb) if state == "device"]
    if wanted:
        return wanted if wanted in online else None
    if len(online) == 1:
        return online[0]
    return None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"verify-mobench: {name} must be an integer, got {raw!r}")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"verify-mobench: {name} must be a number, got {raw!r}")


def _yaml_dump(data: dict) -> str:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - mobench already requires it
        raise SystemExit(f"verify-mobench: PyYAML is required ({exc}); run pip install -e .")
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _copy_text(src: Path, dst: Path) -> None:
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _git_head(root: Path) -> str:
    proc = _run(["git", "rev-parse", "HEAD"], cwd=root, timeout=15)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


# ---------------------------------------------------------------------------
# write-config


def cmd_write_config(args) -> int:
    root = repo_root()
    out = config_path(args.config)
    if out.exists() and not args.force:
        print(f"verify-mobench: config already exists: {out}")
        return 0

    missing = [
        name
        for name in ("MOBENCH_VERIFY_BIN_DIR", "MOBENCH_VERIFY_SOURCE_DIR", "MOBENCH_VERIFY_MODEL")
        if not env(name)
    ]
    if missing:
        print("verify-mobench: set required environment variables first:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        return 1

    adb = env("MOBENCH_VERIFY_ADB")
    if not adb:
        wrapper = root / "tools" / "adb-unc-wrapper.sh"
        if env("ADB_EXE") and wrapper.is_file():
            adb = str(wrapper.relative_to(root))
        else:
            adb = shutil.which("adb") or shutil.which("adb.exe") or ""
    if not adb:
        print(
            "verify-mobench: no adb found; set MOBENCH_VERIFY_ADB or ADB_EXE",
            file=sys.stderr,
        )
        return 1

    device = env("MOBENCH_VERIFY_DEVICE") or _online_device(adb) or ""
    if not device:
        print(
            "verify-mobench: set MOBENCH_VERIFY_DEVICE (exactly one online device is required for auto-detect)",
            file=sys.stderr,
        )
        return 1

    bin_dir = env("MOBENCH_VERIFY_BIN_DIR")
    source_dir = env("MOBENCH_VERIFY_SOURCE_DIR")
    model = env("MOBENCH_VERIFY_MODEL")
    model_id = env("MOBENCH_VERIFY_MODEL_ID") or re.sub(
        r"[^a-z0-9._-]+", "-", Path(model).stem.lower()
    ).strip("-")
    precision = env("MOBENCH_VERIFY_PRECISION", "q4_k_m")
    threads = _env_int("MOBENCH_VERIFY_THREADS", 4)
    rounds = _env_int("MOBENCH_VERIFY_ROUNDS", 2)
    prompt = _env_int("MOBENCH_VERIFY_PROMPT", 64)
    gen = _env_int("MOBENCH_VERIFY_GEN", 64)
    cooldown = os.environ.get("MOBENCH_VERIFY_COOLDOWN_C", "45")
    cmake_args = os.environ.get("MOBENCH_VERIFY_CMAKE_ARGS", DEFAULT_CMAKE_ARGS)

    cfg = {
        "engine": "llamacpp",
        "platform": {"type": "adb", "adb": adb, "device_id": device},
        "build": {
            "bin_dir": bin_dir,
            "bench_bin": env("MOBENCH_VERIFY_BENCH_BIN", "llama-bench"),
            "source_dir": source_dir,
        },
        "workload": {
            "threads": [threads],
            "rounds": rounds,
            "prompt_tokens": prompt,
            "gen_tokens": gen,
        },
        "models": [
            {
                "id": model_id,
                "path": model,
                "kind": "llm",
                "precision": precision,
            }
        ],
        "methodology": {
            "env_control": "auto",
            "cpu_pin": True,
            "lock_freq": True,
            "max_attempts": 2,
        },
        "results_dir": results_dir(),
    }
    if cmake_args:
        cfg["build"]["required_cmake_args"] = cmake_args
    if cooldown and cooldown.lower() != "none":
        cfg["methodology"]["cooldown_c"] = _env_float("MOBENCH_VERIFY_COOLDOWN_C", 45.0)
        cfg["methodology"]["cooldown_max_s"] = 600

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_yaml_dump(cfg), encoding="utf-8")
    print(f"verify-mobench: wrote {out}")
    print(f"  engine=llamacpp platform=adb device={device} model={model_id}")
    print(f"  results_dir={cfg['results_dir']} threads={threads} rounds={rounds}")
    return 0


# ---------------------------------------------------------------------------
# doctor


def _load_cfg(path: Path):
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from mobench.config import ConfigError, load_config, validate

    if not path.is_file():
        raise ConfigError(
            f"config not found: {path}; run `write-config` or set MOBENCH_VERIFY_CONFIG"
        )
    cfg = load_config(str(path))
    problems = validate(cfg)
    if problems:
        raise ConfigError("invalid config: " + "; ".join(problems))
    return cfg


def cmd_doctor(args) -> int:
    root = repo_root()
    cfg_path = config_path(args.config)
    failures: list[str] = []
    warnings: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    def warn(ok: bool, label: str, detail: str = "") -> None:
        print(f"[{'WARN' if not ok else 'PASS'}] {label}" + (f": {detail}" if detail else ""))
        if not ok:
            warnings.append(label)

    print(f"verify-mobench doctor: repo={root}")
    check((root / SKILL_REL / "SKILL.md").is_file(), "skill present", str(root / SKILL_REL))

    proc = _run([sys.executable, "-m", "mobench", "--version"], cwd=root, timeout=30)
    check(proc.returncode == 0, "mobench importable", (proc.stdout or proc.stderr).strip())

    try:
        cfg = _load_cfg(cfg_path)
        check(True, "config valid", str(cfg_path))
    except Exception as exc:  # noqa: BLE001 - doctor reports, does not raise
        check(False, "config valid", str(exc))
        print("\ndoctor: FAIL (fix config before driving the device)")
        return 1

    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', pyproject, re.M)
    expected = match.group(1) if match else "unknown"
    actual = re.search(r"mobench\s+(\S+)", proc.stdout or "")
    check(
        bool(actual) and actual.group(1) == expected,
        "version matches pyproject",
        f"cli={actual.group(1) if actual else '?'} pyproject={expected}",
    )

    platform_cfg = cfg.get("platform") or {}
    check(cfg.get("engine") == "llamacpp", "engine is llamacpp", str(cfg.get("engine")))
    check(platform_cfg.get("type") == "adb", "platform is adb", str(platform_cfg.get("type")))

    build = cfg.get("build") or {}
    bin_dir = Path(build.get("bin_dir") or "")
    if not bin_dir.is_absolute():
        bin_dir = root / bin_dir
    bench = bin_dir / (build.get("bench_bin") or "llama-bench")
    check(bench.is_file(), "bench binary exists", str(bench))
    check(bool(list(bin_dir.glob("lib*.so*"))) if bin_dir.is_dir() else False,
          "shared libraries present", str(bin_dir))

    model_path = Path((cfg.get("models") or [{}])[0].get("path") or "")
    check(model_path.is_file(), "model exists", str(model_path))

    try:
        from mobench.build_recipe import verify_build_recipe

        recipe = verify_build_recipe(cfg)
        check(True, "build recipe verified", json.dumps(recipe, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        check(False, "build recipe verified", str(exc))

    adb = platform_cfg.get("adb")
    device = platform_cfg.get("device_id")
    check(bool(adb) and bool(device), "adb + device configured", f"{adb} {device or ''}".strip())
    if not adb or not device:
        print("\ndoctor: FAIL")
        return 1

    devices = _adb_devices(adb)
    print("  adb devices: " + ", ".join(f"{s}={st}" for s, st in devices))
    online = any(serial == device and state == "device" for serial, state in devices)
    check(online, "device online", str(device))
    if not online:
        print("\ndoctor: FAIL (device offline; stop here)")
        return 1

    model = _shell(adb, device, "getprop ro.product.model")
    osv = _shell(adb, device, "getprop ro.build.version.release")
    cores = _shell(adb, device, "nproc")
    print(
        "  device: "
        f"{model.stdout.strip() or '?'} Android {osv.stdout.strip() or '?'} "
        f"cores {cores.stdout.strip() or '?'}"
    )

    ps = _shell(adb, device, "ps -A 2>/dev/null || ps")
    active = [
        line.strip()
        for line in ps.stdout.splitlines()
        if re.search(r"llama-bench|llm_bench|benchmark\.out", line)
    ]
    check(not active, "no benchmark already running", "; ".join(active[:3]))

    try:
        from mobench.platform.base import BasePlatform

        sample = BasePlatform.make(cfg).sample_thermal()
        temp = (sample.get("temp_c") or {}).get("value")
        if temp is None:
            warn(False, "device temperature readable", "thermal zone unavailable")
        else:
            warn(
                temp <= TEMP_LIMIT_C,
                "device temperature",
                f"{temp} C (limit {TEMP_LIMIT_C} C; run-level cooldown handles a hot start)",
            )
        freq = (sample.get("cpu_freq") or {}).get("value_mhz")
        print(f"  thermal: {temp if temp is not None else '?'} C, max {freq if freq is not None else '?'} MHz")
    except Exception as exc:  # noqa: BLE001
        warn(False, "thermal sample", str(exc))

    if warnings:
        print("  warnings: " + ", ".join(warnings))
    if failures:
        print(f"\ndoctor: FAIL ({len(failures)} failed check(s))")
        return 1
    print("\ndoctor: OK")
    return 0


# ---------------------------------------------------------------------------
# prove


def _check(proof_checks: list[dict], name: str, ok: bool, detail: str = "") -> None:
    proof_checks.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def cmd_prove(args) -> int:
    root = repo_root()
    cfg_path = config_path(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.evidence_dir:
        evidence = Path(args.evidence_dir)
        if not evidence.is_absolute():
            evidence = root / evidence
    else:
        evidence = root / results_dir() / "evidence" / stamp
    evidence.mkdir(parents=True, exist_ok=True)
    report_dir = evidence / "report"

    print(f"verify-mobench prove: evidence={evidence}")
    doctor = _run(
        [sys.executable, str(Path(__file__).resolve()), "doctor", "--config", str(cfg_path)],
        cwd=root,
        timeout=180,
    )
    doctor_text = (doctor.stdout or "") + (doctor.stderr or "")
    (evidence / "doctor.txt").write_text(doctor_text, encoding="utf-8")
    print(doctor_text.rstrip())
    if doctor.returncode != 0:
        print("verify-mobench: doctor failed; refusing to drive the device", file=sys.stderr)
        return 1
    shutil.copy2(cfg_path, evidence / "config.yml")

    label = args.label or env("MOBENCH_VERIFY_LABEL", "skill-proof")
    cmd = [
        sys.executable,
        "-m",
        "mobench",
        "run",
        "--config",
        str(cfg_path),
        "--json",
        "--label",
        label,
    ]
    if args.rounds:
        cmd += ["--rounds", str(args.rounds)]
    print("running: " + " ".join(cmd))
    started = time.time()
    run_proc = _run(cmd, cwd=root, timeout=1800)
    elapsed = time.time() - started
    (evidence / "run.stdout.json").write_text(run_proc.stdout or "", encoding="utf-8")
    (evidence / "run.stderr.log").write_text(run_proc.stderr or "", encoding="utf-8")
    (evidence / "run.rc").write_text(str(run_proc.returncode) + "\n", encoding="utf-8")
    print((run_proc.stderr or "").rstrip()[-4000:])
    print(f"run rc={run_proc.returncode} elapsed={elapsed:.1f}s")
    if run_proc.returncode != 0:
        print("verify-mobench: run failed; see run.stderr.log", file=sys.stderr)
        return 1
    try:
        meta = json.loads(run_proc.stdout)
    except json.JSONDecodeError as exc:
        print(f"verify-mobench: run stdout is not JSON: {exc}", file=sys.stderr)
        return 1

    run_dir = Path(meta.get("records", "")).parent
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    proof_checks: list[dict] = []

    _check(proof_checks, "run directory exists", run_dir.is_dir(), str(run_dir))
    _check(proof_checks, "run.json exists", (run_dir / "run.json").is_file(), str(run_dir / "run.json"))
    n_records = int(meta.get("n_records") or 0)
    _check(proof_checks, "records produced", n_records > 0, f"n_records={n_records}")
    _check(proof_checks, "run valid", meta.get("valid") is True, f"valid={meta.get('valid')!r}")
    _check(proof_checks, "run not aborted", not meta.get("aborted"), f"aborted={meta.get('aborted')!r}")
    _check(proof_checks, "no throttle risk", meta.get("throttle_risk") is False,
           f"throttle_risk={meta.get('throttle_risk')!r}")

    model = ((meta.get("platform") or {}).get("model") or {}).get("value")
    device_id = ((meta.get("platform") or {}).get("device_id") or {}).get("value")
    _check(proof_checks, "device model captured", bool(model), str(model))
    _check(proof_checks, "device id captured", bool(device_id), str(device_id))
    recipe = meta.get("build_recipe") or {}
    _check(proof_checks, "build recipe checked", recipe.get("checked") is True,
           json.dumps(recipe, ensure_ascii=False))
    conditions = meta.get("measurement_conditions") or {}
    _check(proof_checks, "measurement condition captured", bool(conditions),
           ", ".join(sorted(conditions)))

    records_path = Path(meta.get("records", ""))
    if not records_path.is_absolute():
        records_path = root / records_path
    lines = []
    if records_path.is_file():
        lines = [ln for ln in records_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    _check(proof_checks, "records.jsonl readable", bool(lines), str(records_path))
    _check(proof_checks, "record count matches", len(lines) == n_records,
           f"lines={len(lines)} n_records={n_records}")

    missing_logs = []
    for raw in lines:
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            missing_logs.append("unparseable record")
            continue
        src = rec.get("source_log")
        if not src:
            missing_logs.append("missing source_log")
            continue
        src_path = Path(src)
        if not src_path.is_absolute():
            src_path = root / src_path
        if not src_path.is_file() or src_path.stat().st_size == 0:
            missing_logs.append(str(src_path))
    _check(proof_checks, "every record traces to a non-empty raw log", not missing_logs,
           "; ".join(missing_logs[:3]))

    stats = _run(
        [sys.executable, "-m", "mobench", "stats", str(run_dir), "--json"],
        cwd=root,
        timeout=120,
    )
    (evidence / "stats.json").write_text(stats.stdout or "", encoding="utf-8")
    (evidence / "stats.stderr.log").write_text(stats.stderr or "", encoding="utf-8")
    stats_rows = []
    if stats.returncode == 0:
        try:
            stats_rows = json.loads(stats.stdout)
        except json.JSONDecodeError:
            stats_rows = []
    _check(proof_checks, "stats produced", stats.returncode == 0 and bool(stats_rows),
           f"rc={stats.returncode} groups={len(stats_rows)}")

    runs = _run(
        [sys.executable, "-m", "mobench", "runs", "--results", str(run_dir.parent), "--json"],
        cwd=root,
        timeout=120,
    )
    (evidence / "runs.json").write_text(runs.stdout or "", encoding="utf-8")
    (evidence / "runs.stderr.log").write_text(runs.stderr or "", encoding="utf-8")
    _check(proof_checks, "runs index produced", runs.returncode == 0 and bool(runs.stdout.strip()),
           f"rc={runs.returncode}")

    report = _run(
        [
            sys.executable,
            "-m",
            "mobench",
            "report",
            str(run_dir),
            "--out",
            str(report_dir),
            "--format",
            "both",
        ],
        cwd=root,
        timeout=180,
    )
    (evidence / "report.stdout.log").write_text(report.stdout or "", encoding="utf-8")
    (evidence / "report.stderr.log").write_text(report.stderr or "", encoding="utf-8")
    md_path = report_dir / "benchmark_report.md"
    html_path = report_dir / "benchmark_report.html"
    _check(proof_checks, "markdown report produced", report.returncode == 0 and md_path.is_file(),
           str(md_path))
    _check(proof_checks, "html report produced", report.returncode == 0 and html_path.is_file(),
           str(html_path))
    report_md = md_path.read_text(encoding="utf-8") if md_path.is_file() else ""
    _check(
        proof_checks,
        "report cites source logs",
        bool(report_md) and ("raw_" in report_md),
        "source-log column present" if "raw_" in report_md else "no raw_ reference",
    )

    raw_logs = set()
    for raw in lines:
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        src = rec.get("source_log") if isinstance(rec, dict) else None
        if src:
            raw_logs.add(str((root / src).resolve()))

    # A proof directory that only points at the original run dir is not
    # self-contained: copy the small trace files so the evidence survives even
    # if results/ is pruned later.
    run_copy = evidence / "run"
    run_copy.mkdir(parents=True, exist_ok=True)
    for name in ("run.json", "records.jsonl"):
        if (run_dir / name).is_file():
            shutil.copy2(run_dir / name, run_copy / name)
    raw_logs_copy = []
    for src in sorted(raw_logs):
        dst = run_copy / Path(src).name
        shutil.copy2(src, dst)
        raw_logs_copy.append(str(dst))
    _check(proof_checks, "self-contained run copy", (run_copy / "run.json").is_file()
           and (run_copy / "records.jsonl").is_file(), str(run_copy))

    device_label = f"{model}" if model else "unknown device"
    proof = {
        "schema": "mobench/verify-proof@1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "skill": "verify-mobench",
        "repo_head": _git_head(root),
        "config": str(cfg_path),
        "evidence_dir": str(evidence),
        "doctor": {"rc": doctor.returncode, "log": str(evidence / "doctor.txt")},
        "run": {
            "rc": run_proc.returncode,
            "label": label,
            "elapsed_s": round(elapsed, 2),
            "run_dir": str(run_dir),
            "run_copy": str(run_copy),
            "records": str(records_path),
            "n_records": n_records,
            "valid": meta.get("valid"),
            "throttle_risk": meta.get("throttle_risk"),
            "device_model": model,
            "device_id": device_id,
            "engine_version": meta.get("engine_version"),
        },
        "artifacts": {
            "stats": str(evidence / "stats.json"),
            "runs": str(evidence / "runs.json"),
            "report_md": str(md_path),
            "report_html": str(html_path),
            "raw_logs": sorted(raw_logs),
            "raw_logs_copy": raw_logs_copy,
        },
        "checks": proof_checks,
    }
    _write_json(evidence / "proof.json", proof)

    summary = [
        f"# verify-mobench proof — {stamp}",
        "",
        f"- git HEAD: `{proof['repo_head']}`",
        f"- device: {device_label}",
        f"- engine: {meta.get('engine')} (`{meta.get('engine_version')}`)",
        f"- run dir: `{run_dir}`",
        f"- records: {n_records} (valid={meta.get('valid')}, throttle_risk={meta.get('throttle_risk')})",
        f"- elapsed: {elapsed:.1f}s",
        "",
        "## Checks",
        "",
    ]
    for item in proof_checks:
        summary.append(f"- [{'x' if item['ok'] else ' '}] {item['name']}" + (f" — {item['detail']}" if item["detail"] else ""))
    summary += ["", "## Stats", ""]
    if stats_rows:
        summary += [
            "| metric | n | mean | CV% | P50 | P90 |",
            "|---|---|---|---|---|---|",
        ]
        for row in stats_rows:
            s = row.get("stats") or {}
            summary.append(
                f"| {row.get('metric')} | {row.get('n')} | {s.get('mean')} | {s.get('cv')} | "
                f"{s.get('p50')} | {s.get('p90')} |"
            )
    else:
        summary.append("(no stats rows)")
    summary += [
        "",
        "## Retained artifacts",
        "",
        f"- `{evidence / 'doctor.txt'}`",
        f"- `{evidence / 'run.stdout.json'}` / `{evidence / 'run.stderr.log'}`",
        f"- `{run_copy / 'run.json'}` / `{run_copy / 'records.jsonl'}` + `{run_copy}/raw_*.log`",
        f"- `{md_path}` / `{html_path}`",
        f"- `{evidence / 'proof.json'}`",
        "",
        "Do not commit `results/`; evidence is local by design.",
        "",
    ]
    (evidence / "summary.md").write_text("\n".join(summary), encoding="utf-8")

    ok = all(item["ok"] for item in proof_checks)
    print(f"verify-mobench: {'PROOF OK' if ok else 'PROOF FAILED'} -> {evidence}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# cleanup


def _latest_run(results: Path) -> Path | None:
    if not results.is_dir():
        return None
    candidates = [
        p for p in results.rglob("run.json")
        if "evidence" not in p.parts
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1].parent if candidates else None


def cmd_cleanup(args) -> int:
    root = repo_root()
    cfg_path = config_path(args.config)
    failures = []

    try:
        cfg = _load_cfg(cfg_path)
    except Exception as exc:  # noqa: BLE001
        print(f"verify-mobench: cannot read config for cleanup: {exc}", file=sys.stderr)
        return 1

    platform_cfg = cfg.get("platform") or {}
    if platform_cfg.get("type") == "adb":
        adb = platform_cfg.get("adb")
        device = platform_cfg.get("device_id")
        if adb and device:
            ps = _shell(adb, device, "ps -A 2>/dev/null || ps")
            active = [
                line.strip()
                for line in ps.stdout.splitlines()
                if re.search(r"llama-bench|llm_bench|benchmark\.out", line)
            ]
            if active:
                failures.append("benchmark process still running: " + "; ".join(active[:3]))
            else:
                print("PASS: no benchmark process left on the device")

    run_dir = Path(args.run).resolve() if args.run else _latest_run(root / results_dir())
    if not args.run and args.evidence:
        # Prefer the run named by the proof under test over "newest run".
        ev = Path(args.evidence)
        if not ev.is_absolute():
            ev = root / ev
        proof_path = ev / "proof.json"
        if proof_path.is_file():
            try:
                proof_data = json.loads(proof_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                proof_data = {}
            raw_run = (proof_data.get("run") or {}).get("run_dir")
            if raw_run and not Path(raw_run).is_dir():
                raw_run = (proof_data.get("run") or {}).get("run_copy")
            if raw_run:
                run_dir = Path(raw_run).resolve()
    if run_dir and (run_dir / "run.json").is_file():
        try:
            meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        conditions = meta.get("measurement_conditions") or {}
        expected: dict[str, str] = {}
        for cond in conditions.values():
            freq = cond.get("freq") or {}
            for policy, original in (freq.get("restored_from") or {}).items():
                expected[policy] = original
        if expected and platform_cfg.get("type") == "adb":
            adb = platform_cfg.get("adb")
            device = platform_cfg.get("device_id")
            mismatches = []
            for policy, original in sorted(expected.items()):
                proc = _shell(
                    adb,
                    device,
                    f"cat /sys/devices/system/cpu/cpufreq/{policy}/scaling_governor 2>/dev/null",
                )
                current = (proc.stdout or "").strip()
                if current != original:
                    mismatches.append(f"{policy}: current={current or '?'} expected={original}")
            if mismatches:
                failures.append("governor not restored: " + "; ".join(mismatches))
            else:
                print(f"PASS: governors restored ({len(expected)} policies) from {run_dir.name}")
        else:
            print(f"PASS: no governor state to compare for {run_dir.name}")

    if args.evidence:
        ev = Path(args.evidence)
        if not ev.is_absolute():
            ev = root / ev
        proof_path = ev / "proof.json"
        if not proof_path.is_file():
            failures.append(f"evidence proof missing: {proof_path}")
        else:
            try:
                proof_data = json.loads(proof_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                failures.append(f"evidence proof is not valid JSON: {proof_path}: {exc}")
            else:
                bad = [c.get("name") for c in (proof_data.get("checks") or []) if not c.get("ok")]
                if bad:
                    failures.append(f"evidence proof has failed checks: {', '.join(bad)}")
                else:
                    print(f"PASS: evidence proof intact and all checks true: {proof_path}")
    else:
        evidence_root = root / results_dir() / "evidence"
        if evidence_root.is_dir():
            print(f"evidence kept at: {evidence_root}")
        print("tip: pass --evidence <proof dir> to assert a specific proof is still intact")

    if args.remove_config and cfg_path.is_file():
        cfg_path.unlink()
        print(f"removed disposable config: {cfg_path}")

    if failures:
        for item in failures:
            print(f"FAIL: {item}", file=sys.stderr)
        print("cleanup: FAIL (do not measure again until resolved)", file=sys.stderr)
        return 1
    print("cleanup: OK (short-lived CLI exited; nothing was killed by name)")
    return 0


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify-mobench",
        description="Repo-local control helper for the verify-mobench skill.",
    )
    parser.add_argument("--config", help="config path (default: MOBENCH_VERIFY_CONFIG)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("write-config", help="render an Android llama.cpp config from env vars")
    p.add_argument("--config", help="output path (default: MOBENCH_VERIFY_CONFIG)")
    p.add_argument("--force", action="store_true", help="overwrite an existing config")
    p.set_defaults(func=cmd_write_config)

    p = sub.add_parser("doctor", help="read-only preflight checks")
    p.add_argument("--config", help="config path")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("prove", help="drive one mapped feature and retain evidence")
    p.add_argument("--config", help="config path")
    p.add_argument("--label", help="run label (default: skill-proof)")
    p.add_argument("--rounds", type=int, help="override rounds (default: config)")
    p.add_argument("--evidence-dir", help="evidence directory (default: results/verify-mobench/evidence/<UTC>)")
    p.set_defaults(func=cmd_prove)

    p = sub.add_parser("cleanup", help="verify no process/governor was left behind")
    p.add_argument("--config", help="config path")
    p.add_argument("--run", help="specific run dir (default: newest under results)")
    p.add_argument("--evidence", help="assert this proof directory still has proof.json with all checks true")
    p.add_argument("--remove-config", action="store_true", help="delete the disposable config (run prove first)")
    p.set_defaults(func=cmd_cleanup)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
