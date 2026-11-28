---
name: verify-mobench
description: "Drive the mobench CLI against a real Android device to prove benchmark behavior end to end: doctor the adb/llama.cpp setup, run a traced benchmark, inspect records and stats, and generate a source-linked report. Use when verifying mobench changes, reproducing a device benchmark, or checking that a build recipe is actually what you think it is."
---

# Verify mobench

This skill is the project-local control surface for `mobench`. It drives the
real CLI (not the Python API, not a mock) against one Android device running a
`llama.cpp` build, and it keeps the evidence needed to defend the result.

Read `features/README.md` first. It maps the user-facing features and points to
one recipe per feature. One proved feature is enough to accept the skill; the
map exists so later runs can cover the rest.

## Baseline preconditions

- Run from the root of this checkout: `git rev-parse --show-toplevel`.
- Python 3.10+ with the package importable (`python3 -m mobench --version`).
  If that fails, install once with `python3 -m pip install -e .`.
- `adb` can see exactly one online device. On WSL, point `ADB_EXE` at the
  Windows `adb.exe` and use the committed wrapper
  `tools/adb-unc-wrapper.sh`; the wrapper maps Linux paths to Windows/UNC
  paths before `push`. A plain Linux `adb` works too.
- A `llama.cpp` Android build (`llama-bench` plus its `lib*.so*`) and one GGUF
  model exist locally. The build directory must contain `CMakeCache.txt` when
  `build.required_cmake_args` is declared.
- The device is idle: no other `llama-bench`, `llm_bench`, or `benchmark.out`
  process, and no one else is driving it.
- `results/` and `configs/local/` are gitignored. Evidence and disposable
  configs stay local.

## Environment

Export these once per shell. Only `MOBENCH_VERIFY_BIN_DIR`,
`MOBENCH_VERIFY_SOURCE_DIR`, and `MOBENCH_VERIFY_MODEL` are required; the rest
have defaults.

| Variable | Meaning |
|---|---|
| `MOBENCH_VERIFY_CONFIG` | Config path. Default `configs/local/verify-mobench.yml`. |
| `MOBENCH_VERIFY_ADB` | `adb` executable or `tools/adb-unc-wrapper.sh`. Default: wrapper when `ADB_EXE` is set, else `adb` on `PATH`. |
| `MOBENCH_VERIFY_DEVICE` | Device serial. Default: the only online device. |
| `MOBENCH_VERIFY_BIN_DIR` | Local Android `llama.cpp` bin directory. Required. |
| `MOBENCH_VERIFY_SOURCE_DIR` | `llama.cpp` source root for git commit and build-recipe tracing. Required. |
| `MOBENCH_VERIFY_MODEL` | GGUF path. Required. |
| `MOBENCH_VERIFY_MODEL_ID` | Record id. Default from the model filename. |
| `MOBENCH_VERIFY_PRECISION` | Precision label. Default `q4_k_m`. |
| `MOBENCH_VERIFY_THREADS` | Threads. Default `4`. |
| `MOBENCH_VERIFY_ROUNDS` | Rounds. Default `2` for a quick proof; formal stats need 5+. |
| `MOBENCH_VERIFY_PROMPT` / `_GEN` | `-p` / `-n`. Defaults `64` / `64`. |
| `MOBENCH_VERIFY_CMAKE_ARGS` | Required `-D` recipe. Default `-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod`; set empty to omit. |
| `MOBENCH_VERIFY_COOLDOWN_C` | Pre-run thermal target. Default `45`; `none` disables. |
| `MOBENCH_VERIFY_RESULTS` | Results root. Default `results/verify-mobench`. |

`ADB_EXE` is consumed by the wrapper, not by `mobench` itself.

## Launch

```bash
cd "$(git rev-parse --show-toplevel)"
python3 -m mobench --version || python3 -m pip install -e .

export MOBENCH_VERIFY_ADB="$PWD/tools/adb-unc-wrapper.sh"   # WSL + Windows adb
export ADB_EXE="/path/to/adb.exe"                           # wrapper input only
# On a native Linux host instead: export MOBENCH_VERIFY_ADB="$(command -v adb)"
export MOBENCH_VERIFY_DEVICE="$("$MOBENCH_VERIFY_ADB" devices | awk 'NR>1 && $2=="device" {print $1; exit}')"
export MOBENCH_VERIFY_BIN_DIR="/path/to/llama.cpp/build-android/bin"
export MOBENCH_VERIFY_SOURCE_DIR="/path/to/llama.cpp"
export MOBENCH_VERIFY_MODEL="/path/to/model.gguf"

python3 .agents/skills/verify-mobench/scripts/verify_mobench.py write-config
```

`write-config` leaves an existing file untouched and prints `config already
exists` (exit 0); pass `--force` to replace it. Use `MOBENCH_VERIFY_CONFIG` to
point at a config you already maintain; in that case skip `write-config`.

Launch is complete when the config exists and `doctor` passes. There is no
server to keep alive: every command is a short-lived CLI process.

## Doctor

```bash
python3 .agents/skills/verify-mobench/scripts/verify_mobench.py doctor \
  --config "$MOBENCH_VERIFY_CONFIG"
```

Read-only. Exit 0 means the instance is worth driving. It checks:

- the skill and package are importable, and the CLI version matches
  `pyproject.toml`;
- the config parses and passes `mobench.config.validate`;
- engine is `llamacpp`, platform is `adb`;
- `build.bin_dir/llama-bench`, shared libraries, and the model exist;
- `required_cmake_args` are baked into the `CMakeCache.txt` above `bin_dir`;
- the configured device is `device` (not `offline`/`unauthorized`), with model,
  Android version, and core count printed;
- no benchmark process is already running;
- the hottest thermal zone is readable and reported. Above 45 °C is a warning:
  the run-level cooldown waits, but a hot start still risks a dirty result.

If any check fails, stop. Do not drive a device that another process owns, and
do not "fix" a failed build-recipe check by deleting the declaration.

## Drive

Prove the primary feature (`run-device-benchmark`) end to end:

```bash
python3 .agents/skills/verify-mobench/scripts/verify_mobench.py prove \
  --config "$MOBENCH_VERIFY_CONFIG" --label skill-proof --rounds 2
```

`prove` runs `doctor`, then the real `mobench run`, then `stats`, `runs`, and
`report`, and writes a self-checking `proof.json`. A proof is accepted only
when every check in that file is `true`. `--evidence-dir <dir>` overrides the
default evidence location; set `EVIDENCE` to the path `prove` prints for the
cleanup step below.

The equivalent manual core command is:

```bash
python3 -m mobench run --config "$MOBENCH_VERIFY_CONFIG" --json \
  --label skill-proof --rounds 2
```

Exit code `0` and a JSON object on stdout are required. Progress goes to
stderr, so stdout stays parseable. Do not add `2>&1` when capturing the JSON.

To drive one feature by hand, follow its file under `features/`. Do not
substitute a different entry point and call the feature verified.

## Evidence

`prove` writes to `$MOBENCH_VERIFY_RESULTS/evidence/<UTC timestamp>/`
(default `results/verify-mobench/evidence/<UTC timestamp>/`). The run itself
lives in a timestamped directory under `$MOBENCH_VERIFY_RESULTS/`; `prove`
copies its small trace files into `evidence/run/` so the proof is
self-contained.

| Artifact | What it proves |
|---|---|
| `doctor.txt` | The instance was healthy before the run. |
| `config.yml` | The exact effective input (local paths; never commit). |
| `run.stdout.json` | The CLI's machine-readable result: records path, `n_records`, `valid`, engine version, device, build recipe, measurement conditions. |
| `run.stderr.log` / `run.rc` | The action and its exit code. |
| `run/run.json` | Per-run trace: binary/model hashes, recipe, env control, thermal band. |
| `run/records.jsonl` + `run/raw_*.log` | Every metric and the non-empty raw engine log named by its `source_log`. |
| `stats.json` | Grouped n / mean / std / CV / P50 / P90 / P99. |
| `runs.json` | The run index row, including audit/valid flags. |
| `report/benchmark_report.md` / `.html` | Source-linked report, including a thermal banner when relevant. |
| `proof.json` | The accepted/rejected checks, with paths. |
| `summary.md` | Human-readable index of the above. |

Proof standards:

- The command must exercise the real user path: `mobench run` with the real
  config, device, model, and binary. Never call internal setters or test-only
  endpoints.
- Capture the action and the result: command line, stdout, stderr, exit code,
  `run.json`, records, and raw logs.
- Verify side effects, not just the screen: `n_records > 0`, every record's
  `source_log` exists and is non-empty, `valid: true`, `aborted` false or
  absent, and the report cites a `raw_` log.
- A `throttle_risk: true` run is not a performance proof. Cool the device and
  rerun.
- Evidence survives cleanup. Cleanup never deletes `results/`.

## Cleanup

```bash
# Set EVIDENCE to the path `prove` printed.
python3 .agents/skills/verify-mobench/scripts/verify_mobench.py cleanup \
  --config "$MOBENCH_VERIFY_CONFIG" --evidence "$EVIDENCE"
```

The CLI is short-lived, so there is no instance to stop. `cleanup` checks:

- no `llama-bench` / `llm_bench` / `benchmark.out` process is left on the
  device (it reports PIDs; it never kills by name);
- every governor touched by the selected `run.json` is back at its recorded
  `restored_from` value (newest by default, or the run named by `--evidence`).
  If not, restore only those policies with root and rerun cleanup;
- with `--evidence`, the named `proof.json` still exists and all of its checks
  are `true`. Without it, cleanup prints the evidence root and leaves it alone.

`--remove-config` deletes the disposable config; `prove` has already copied it
to `evidence/config.yml`. Use it only after a proof. It never deletes evidence.

## Helpers

- `python3 .agents/skills/verify-mobench/scripts/verify_mobench.py write-config`
  — render the disposable Android config from the environment.
- `python3 .agents/skills/verify-mobench/scripts/verify_mobench.py doctor` —
  read-only preflight.
- `python3 .agents/skills/verify-mobench/scripts/verify_mobench.py prove` —
  doctor + run + stats + runs + report + proof checks.
- `python3 .agents/skills/verify-mobench/scripts/verify_mobench.py cleanup` —
  process/governor check, evidence kept.

All helpers are invoked from the repo root with `python3`; none writes inside
`mobench/`.

## Maintenance

When the CLI, config schema, or feature map changes, run
`/maintain-verification-skill` and update `features/` rather than weakening a
journey. The map is the maintained verification source; a proof through one
convenient entry point is incomplete when the map lists others.
