# Run an Android llama.cpp benchmark

`mobench run` takes a config that names one Android device, a local
`llama-bench` build, and a GGUF model, then executes the whole measurement
pipeline and writes canonical records whose every value points back to a raw
engine log.

## Sub-features

- `run-preflight` verifies the adb device is online and the binary/model exist
  before spending device time.
- `run-build-recipe` refuses a build whose `CMakeCache.txt` lacks the declared
  `required_cmake_args`.
- `run-sync` pushes the binary directory, its `lib*.so*`, and the model to the
  device, skipping files whose MD5 already matches.
- `run-env-control` pins the CPU set for the thread count and locks the
  covering cpufreq policies to `performance`, then restores the recorded
  governors.
- `run-thermal` samples the hottest zone and max frequency before and after
  every round; a value above 45 °C marks the run throttled.
- `run-records` appends `mobench/record@1` rows with `source_log`, plus a
  `run.json` carrying binary/model hashes, build recipe, device, and
  measurement conditions.
- `run-json` keeps stdout pure JSON while progress stays on stderr.

## How to get to it (user POV)

- Run `python3 -m mobench run --config <config>` from the repo root.
- Add `--json` for one machine-readable object instead of the human summary.
- Override one dimension with `--model`, `--threads`, `--rounds`, `--label`,
  `--cooldown-c`, `--cooldown-per-round`, or `--max-attempts`.

## Driving it with verify-mobench

Preconditions:

- `scripts/verify_mobench.py doctor --config "$MOBENCH_VERIFY_CONFIG"` prints
  `doctor: OK`.
- `ADB_EXE`, `MOBENCH_VERIFY_ADB`, `MOBENCH_VERIFY_DEVICE`,
  `MOBENCH_VERIFY_BIN_DIR`, `MOBENCH_VERIFY_SOURCE_DIR`, and
  `MOBENCH_VERIFY_MODEL` are exported.
- No other benchmark is running on the device.
- The device is below 45 °C, or the config declares `methodology.cooldown_c`.

- **Run the pipeline.** Run
  `python3 -m mobench run --config "$MOBENCH_VERIFY_CONFIG" --json --label skill-proof --rounds 2`.
  Exit code `0`; stdout is a JSON object; stderr contains the `[run] step:`
  lines. The object has `n_records > 0`, `valid: true`, `aborted` absent or
  false, and a non-empty `records` path.
- **Read the run trace.** Open the `run.json` next to the records path. It has
  `engine_version`, `binary.md5`, `models[].md5`, `platform.model`,
  `platform.os`, `build_recipe.verified`, and
  `measurement_conditions.t<threads>`.
- **Verify the build recipe.** When the config declares
  `required_cmake_args`, `run.json` has `build_recipe.checked: true` and
  `build_recipe.verified.GGML_CPU_ARM_ARCH` equal to the declared value. A
  mismatch means the pipeline refused to run. With no declaration,
  `checked` is `false` and no claim is made.
- **Verify env control.** `measurement_conditions.t<threads>.pin.cpus` is the
  selected CPU set (for 4 threads on the reference 1+3+4 device, `4,5,6,7`)
  and `.freq.locked` is `true`. On a non-root adb shell it may be `false`;
  that run is labelled degraded and is not a controlled performance proof.
- **Verify records trace to logs.** Every line of `records.jsonl` has a
  `source_log`; the named file exists and is non-empty. The record also carries
  `device.temp_c`, `device.cpu_freq_mhz`, `device.pin`, and `device.freq`.
- **Verify metrics.** The records include `prompt_eval_ms_tok`, `gen_tok_s`,
  and `gen_ms_tok` for each round.
- **Prove the whole feature.** Run
  `python3 .agents/skills/verify-mobench/scripts/verify_mobench.py prove --config "$MOBENCH_VERIFY_CONFIG" --label skill-proof --rounds 2`.
  Exit code `0`; `results/verify-mobench/evidence/<UTC>/proof.json` has every
  check `true`.

## Gotchas

- `required_cmake_args` is verified against `dirname(bin_dir)/CMakeCache.txt`
  or `bin_dir/CMakeCache.txt`. A non-CMake bin directory must omit the
  declaration; deleting the check to make a run pass invalidates the result.
- A missing `GGML_CPU_ARM_ARCH=armv8.2-a+dotprod` silently disables the
  repack/sdot path. The gate exists because that produces a plausible but
  invalid "no gain" result.
- The WSL wrapper maps existing Linux paths to Windows/UNC paths. Pass it as
  `platform.adb`; put the real `adb.exe` in `ADB_EXE`. A relative wrapper path
  in the config is resolved from the repo root.
- A first push can move hundreds of megabytes and take minutes. Do not kill
  it; the wrapper's timeout scales with file size.
- `--json` writes progress to stderr. Capturing `2>&1` corrupts the JSON.
- `results_dir` is relative to the invoking cwd. Run from the repo root so
  `results/` stays inside the checkout and gitignored.
- The pipeline restores governors in a `finally` block. A failed restore marks
  `run.json` invalid and exits non-zero; run `cleanup` and fix the governor
  before the next measurement.
- A run above 45 °C is marked `throttle_risk: true`. It is still recorded, but
  it is not a performance proof.
- Root is required for frequency locking. Without it the run still executes,
  but `freq.locked` is `false`; do not compare it to a locked run.
