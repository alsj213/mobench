# Generate a report

`mobench report` renders the measured records into a Markdown or HTML report
whose every value names its source log, and into a standalone chart variant
for comparing groups.

## Sub-features

- `report-md` writes `benchmark_report.md` with one row per
  engine/workload/metric/threads/precision group.
- `report-html` writes a self-contained `benchmark_report.html`.
- `report-source` includes the source-log column (short log name plus round
  count) so a reader can jump from a number to its raw evidence.
- `report-thermal` adds a throttling banner when a group's recorded
  temperatures exceed 45 °C.
- `report-chart` writes `benchmark_chart.html`, grouped by `backend`,
  `engine`, or `matrix`.
- `report-invalid` skips invalid runs by default; `--include-invalid` includes
  them explicitly.

## How to get to it (user POV)

- Run `python3 -m mobench report <run dir or records.jsonl> --out <dir>`.
- Choose `--format md`, `html`, `both`, or `chart`.
- With `--format chart`, choose `--chart-by backend|engine|matrix`.

## Driving it with verify-mobench

Preconditions:

- A completed, valid run directory exists under `results/verify-mobench/runs/`.
- `$RUN_DIR` names it and `$EVIDENCE` names the output directory.
- The run's records are readable and not marked invalid.

- **Render both text formats.** Run
  `python3 -m mobench report "$RUN_DIR" --out "$EVIDENCE/report" --format both`.
  Exit code `0`; `$EVIDENCE/report/benchmark_report.md` and
  `benchmark_report.html` exist and are non-empty.
- **Inspect the table.** The Markdown has the engine, workload, metric,
  threads, precision, params, measurement condition, n, mean, CV, and source
  log columns. Every metric row names a `raw_*.log` and the round count.
- **Verify the source link.** Open the named `raw_*.log` in the run directory;
  it contains the engine table whose numbers appear in the report.
- **Render the chart.** Run
  `python3 -m mobench report "$RUN_DIR" --out "$EVIDENCE/report" --format chart --chart-by engine`.
  Exit code `0`; `$EVIDENCE/report/benchmark_chart.html` exists and contains
  the metric groups.
- **Check the thermal banner.** If any record's `device.temp_max_c` exceeds
  45 °C, the report carries the throttling banner; do not present that report
  as a clean performance result.

## Gotchas

- `--format both` produces Markdown and HTML, not the chart. Use `chart`
  separately.
- The output directory is created, but the report file names are fixed. Two
  reports written to the same directory overwrite each other.
- `report` defaults to `results/`; pass `--out` so a proof does not scatter
  artifacts.
- Invalid runs are skipped by default. If a group disappears, check `run.json`
  `valid` / `log_audit` before reaching for `--include-invalid`.
- The report is derived data. It cannot prove a run that has no records; the
  `records.jsonl` and raw logs remain the authority.
- A high temperature banner is a warning, not a hard error. A valid report can
  still describe a throttled run; label it accordingly.
