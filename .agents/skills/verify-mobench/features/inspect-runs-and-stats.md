# Inspect runs and statistics

`mobench` keeps every completed run as a directory with `run.json` and
`records.jsonl`. The run index, grouped statistics, and baseline comparison
turn that directory into a reviewable result without re-running the device.

## Sub-features

- `runs-index` scans `results/` for `run.json` and emits one row per run with
  label, binary MD5, record count, recipe state, audit flags, throttle flag,
  and cooldown status.
- `stats-groups` groups records by engine, workload, metric, threads, and
  precision, then reports n / mean / std / CV / P50 / P90 / P99.
- `stats-variance` flags CV above the threshold and marks groups with fewer
  than 5 samples as insufficient for formal statistics.
- `stats-invalid` skips runs marked invalid by the log/frequency audit unless
  `--include-invalid` is passed.
- `regression-save` saves grouped samples as a `mobench/baseline@1` file.
- `regression-compare` compares a baseline with current records, including a
  Mann-Whitney test when both sides have at least 5 samples and SciPy is
  installed.

## How to get to it (user POV)

- Run `python3 -m mobench runs --results <results dir>`.
- Run `python3 -m mobench stats <run dir or records.jsonl>`.
- Run `python3 -m mobench regression save --results <run dir> --out <baseline.json>`.
- Run `python3 -m mobench regression compare --results <run dir> --baseline <baseline.json>`.
- Add `--json` to `runs` and `stats`; `regression` accepts it too.

## Driving it with verify-mobench

Preconditions:

- One completed run from `run-device-benchmark` exists under
  `$MOBENCH_VERIFY_RESULTS/` (default `results/verify-mobench/`).
- `$RUN_DIR` names that run directory, or the records file inside it.
- `python3 -m mobench` works from the repo root.

- **Index the run.** Run
  `python3 -m mobench runs --results "$(dirname "$RUN_DIR")" --json`.
  Exit code `0`; the array has one object whose `run_dir` names `$RUN_DIR`,
  with `n_records` matching `run.json` and `recipe_checked: true` when the
  config declared a build recipe.
- **Compute grouped stats.** Run
  `python3 -m mobench stats "$RUN_DIR" --json`. Exit code `0`; the array has
  one row per metric. Each row has `n`, `stats.mean`, `stats.std`,
  `stats.cv`, `stats.p50`, `stats.p90`, `stats.p99`, and
  `samples_min`/`high_variance` booleans.
- **Verify source traceability.** Every stats row is built from records whose
  `source_log` is non-empty; open one named log and confirm it contains the
  engine table used for the metric.
- **Save a baseline.** Run
  `python3 -m mobench regression save --results "$RUN_DIR" --out "$EVIDENCE/baseline.json" --json`.
  Exit code `0`; the baseline has `schema: mobench/baseline@1` and non-empty
  `groups`.
- **Compare against the baseline.** Run
  `python3 -m mobench regression compare --results "$RUN_DIR" --baseline "$EVIDENCE/baseline.json" --json`.
  Exit code `0`; the result has `rows` with `base_mean`, `cur_mean`, and
  `delta_pct`. Comparing a run with itself gives `delta_pct: 0.0`.

## Gotchas

- `stats` refuses an empty input with exit code `1`; a directory with no
  records is not a zero-result run.
- Formal statistics require at least 5 rounds. A 2-round smoke run produces
  `samples_min: false`; do not quote its CV as a stable estimate.
- `--include-invalid` deliberately mixes invalid runs back in. Do not use it
  for a proof unless the invalidity is the thing being studied.
- `regression compare` is advisory: it returns `0` even when a row is flagged
  as worse. Read `rows[].worse` / `significant`, do not infer failure from the
  exit code.
- Mann-Whitney needs SciPy (`pip install -e ".[regression]"`); without it the
  result falls back to `method: descriptive`.
- The run index reads `run.json`, not records. An index row can exist while a
  records file is missing; verify both for a proof.
- `--results` pointed at the wrong parent will silently index other runs. Keep
  the evidence path next to the proof.
