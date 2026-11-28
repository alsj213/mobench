# mobench verification map

This directory is the maintained source for verifying the user-facing behavior
of `mobench`. Read this index before driving the CLI, then use the matching
feature file as the recipe.

## Baseline preconditions

- Launch from the repo root; `python3 -m mobench --version` succeeds.
- `MOBENCH_VERIFY_CONFIG` points at an `adb` + `llamacpp` config. If it does
  not exist, run
  `python3 .agents/skills/verify-mobench/scripts/verify_mobench.py write-config`
  first.
- Export `ADB_EXE` and `MOBENCH_VERIFY_ADB` when using the WSL wrapper; export
  `MOBENCH_VERIFY_DEVICE`, `MOBENCH_VERIFY_BIN_DIR`, `MOBENCH_VERIFY_SOURCE_DIR`,
  and `MOBENCH_VERIFY_MODEL`.
- Run
  `python3 .agents/skills/verify-mobench/scripts/verify_mobench.py doctor`
  and require `doctor: OK`.
- Never drive a device that this verification run did not check as idle.
- Treat `results/` and `configs/local/` as local-only. They are gitignored.

## Driving conventions

- Start every recipe from the baseline state unless its preconditions say
  otherwise.
- Treat every command as literal. Keep paths and flags unchanged.
- Run CLI actions with `python3 -m mobench ...` so a missing console-script
  install does not change behavior.
- Capture stdout and stderr separately. `run --json` keeps progress on stderr
  so stdout is pure JSON.
- Prefer stable identities (config path, `run.json`, record `source_log`,
  metric name) over screen coordinates.
- After a mutation, prove it with a read-only second view (`run.json`,
  `records.jsonl`, `stats`, report) rather than trusting the mutation command.

## Proof and skip reporting

- Capture the action and the resulting state: command, stdout, stderr, exit
  code, run artifacts, and the source log named by every record.
- Performance proof requires `valid: true`, `aborted` false or absent,
  `throttle_risk: false`, and a non-empty `source_log` per record.
- Record the feature ID and entry point used with every artifact.
- Report an unreachable path with the attempted command and the unmet
  precondition. Do not report a skipped entry point as verified through a
  different path.
- One proved feature accepts the generated skill; the rest of the map exists
  for later runs.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the
user-visible behavior. It then uses exactly four H2 sections in this order.

1. `Sub-features` lists short IDs with one line for each behavior.
2. `How to get to it (user POV)` lists every user entry point.
3. `Driving it with verify-mobench` starts with `Preconditions:` and uses
   labeled bullets that pair each user action with an exact command and
   observable result.
4. `Gotchas` lists traps that can waste or invalidate a verification run.

## Features

- [Run an Android llama.cpp benchmark](./run-device-benchmark.md) covers
  preflight, build-recipe gating, artifact sync, environment control, thermal
  sampling, records, and `--json`.
- [Inspect runs and statistics](./inspect-runs-and-stats.md) covers the run
  index, grouped statistics, and regression baselines.
- [Generate a report](./generate-report.md) covers Markdown/HTML reports,
  source-log citations, thermal banners, and the chart variant.
- [Prepare a model](./prepare-model.md) covers self-provided validation,
  download with sha256, and engine-side conversion.
- [Validate correctness](./validate-correctness.md) covers the precision gate:
  `validate run`, coverage declarations, and `validate compare` exit codes.

## Not seeded yet

The map starts with the benchmark/analysis path. These surfaces are real and
should get their own feature files before a full-repo verification claim:

- `mobench profile` (host `perf` / device `simpleperf`)
- `mobench memory` and `mobench trace` (Android sampling)
- `mobench power` (RAPL / batterystats text)
- the MCP server (`mobench.mcp_server`)
- the `host` platform, which shares the pipeline but not the adb transfer and
  device-control steps
