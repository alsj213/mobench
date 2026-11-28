# Validate correctness

`mobench validate` answers whether a build or optimization changed model
output, and it makes instrument blind spots structurally visible instead of
claiming coverage it does not have.

## Sub-features

- `validate-run` executes every declared instrument and writes a
  self-describing `validate.json` with controls, identity, coverage, and raw
  logs.
- `validate-greedy` uses `llama-completion` with a fixed prompt and greedy
  decoding to cover the decode path.
- `validate-ppl` uses `llama-perplexity` to cover the prefill path.
- `validate-coverage` marks a path `covered: false` / `NOT COVERED` when no
  instrument for it ran.
- `validate-ab` runs two configs side by side only when their model MD5,
  threads, prompt set hash, sampling, and PPL controls match.
- `validate-compare` compares a baseline artifact with a candidate or an A/B
  pair and returns `0` for PASS/WARN, `2` for FAIL, `1` when incomparable.

## How to get to it (user POV)

- Run `python3 -m mobench validate run --config <config> [--config-b <config>]`.
- Run `python3 -m mobench validate compare --baseline <validate.json> [--candidate <validate.json>]`.
- Add `--out` to choose the artifact directory; add `--json` for the artifact
  on stdout.
- Use `--tolerance` and `--ppl-tolerance` only when a non-exact comparison is
  intended.

## Driving it with verify-mobench

Preconditions:

- The config has a `validation` block with `prompts` and/or `ppl`, and the
  prompt/dataset files exist.
- `build.bin_dir` contains `llama-completion` for `greedy` and/or
  `llama-perplexity` for `ppl`. `llama-bench` alone is not enough.
- For A/B, both configs declare the same model, threads, prompts, sampling,
  and PPL controls.

- **Run the instruments.** Run
  `python3 -m mobench validate run --config "$MOBENCH_VERIFY_CONFIG" --out "$EVIDENCE/validate" --json`.
  Exit code `0` only if at least one instrument produced a usable result on
  every side. `$EVIDENCE/validate/validate.json` exists with `schema:
  mobench/validate@1`.
- **Read coverage before the verdict.** `validate.json` has
  `coverage.prefill` and `coverage.decode`; each says `covered` and which
  instrument provides it. A run with only PPL must say decode is `NOT
  COVERED`.
- **Check the controls fingerprint.** `controls` contains `model_md5`,
  `threads`, `sampling`, `gen_tokens`, and `prompts_sha256`; an A/B artifact
  also contains `ppl.dataset_sha256`, `ctx`, and `chunks`.
- **Compare a candidate.** Run
  `python3 -m mobench validate compare --baseline "$EVIDENCE/validate/validate.json" --candidate "$CANDIDATE" --json`.
  Exit code `0` for PASS/WARN, `2` for FAIL, `1` when the artifacts are not
  comparable. Read `verdict`, `controls_diff`, `coverage_diff`,
  `identity_diff`, `coverage`, and `uncovered_paths` rather than only the
  exit code.
- **Check the failure path.** Compare artifacts with a different model MD5 or
  thread count. Exit code `1`; the output names the differing control and no
  measurement is claimed.

## Gotchas

- A build that only contains `llama-bench` cannot run `validate`; build
  `llama-completion` / `llama-perplexity` first. The CLI reports the missing
  instrument instead of inventing a value.
- PPL covers prefill only. It never covers single-token decode; do not let a
  PPL-only artifact imply decode coverage.
- `validate run` does not use cooldown, CPU pinning, or frequency locking.
  Token streams are deterministic given the controls, so this is not a
  performance run.
- A/B is refused before device time when the controls differ. Fix the config,
  do not relax the comparison.
- `validate compare` defaults to exact comparison. `--tolerance` /
  `--ppl-tolerance` turn it into an approximate check; state that in the
  evidence.
- The artifact lives under the configured `results_dir` unless `--out` is
  passed. Keep it with the proof, not in a shared results root.
- Exit code `1` means "cannot compare", not "the model is wrong". Report the
  unmet precondition.
