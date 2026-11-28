# Prepare a model

`mobench prepare` makes the model named in the config available before a run.
It covers three real states: a self-provided file (validate and hash), a
download (with optional sha256), and conversion by the engine's own tool.

## Sub-features

- `prepare-self-provided` confirms the configured file exists and reports its
  size, MD5, and SHA-256.
- `prepare-download` streams an `http`/`https` URL to the configured path and
  verifies `source.sha256` when declared.
- `prepare-convert` invokes the engine's converter (`convert_with: llama.cpp`
  or `mnn`) and finalizes the produced artifact.
- `prepare-json` prints a stable list of `{id, status, path}` objects.
- `prepare-filter` prepares only the requested model ids.

## How to get to it (user POV)

- Run `python3 -m mobench prepare --config <config>`.
- Run `python3 -m mobench prepare --config <config> <model-id> ...`.
- Add `--json` for machine-readable output.

## Driving it with verify-mobench

Preconditions:

- `$MOBENCH_VERIFY_CONFIG` names a config whose `models[].path` exists for the
  self-provided case.
- For download, the config has `models[].source.url` (and `sha256` if the
  source publishes one).
- For conversion, `build.source_dir` contains `convert_hf_to_gguf.py` and
  `models[].source.dir` names the HF directory.

- **Prepare the self-provided model.** Run
  `python3 -m mobench prepare --config "$MOBENCH_VERIFY_CONFIG" --json`.
  Exit code `0`; the array has one object with `status: "ready"` and the
  configured `path`. The human form prints a SHA-256 prefix.
- **Confirm the hash is stable.** Run the command twice; both runs report the
  same size and hash, and the second does not modify the file.
- **Prepare one model id.** Run
  `python3 -m mobench prepare --config "$MOBENCH_VERIFY_CONFIG" qwen3-0.6b-q4_k_m --json`.
  Exit code `0`; exactly that id is in the output.
- **Prove the missing-file path.** Point a disposable config at a non-existent
  model and run the same command. Exit code `1`; stderr says the model file is
  missing and points at `models[].source`.
- **Download integrity (only with a disposable URL).** Add a `source.sha256`,
  download, then corrupt one byte and rerun. The mismatch exits `1` and
  removes the artifact rather than leaving a bad file behind.

## Gotchas

- A self-provided model is hashed, not compared to an expected value unless
  `source.sha256` is declared. Hashing proves identity, not provenance.
- A download without `source.sha256` is not integrity-checked; treat it as
  convenient, not verified.
- Only `http`/`https` URLs are accepted; other schemes are rejected before any
  file is touched.
- A download writes `<path>.part` and then renames. An interrupted process can
  leave the `.part` file; it is not a model.
- Conversion runs the engine's Python converter and can take minutes and a lot
  of CPU. It is not a device benchmark; run it before `doctor`.
- `prepare` creates the model's parent directory. That is the only write in
  the core path; the model itself is not re-downloaded when it already exists
  and no `source.url` is declared.
- The JSON result is intentionally small (`id`, `status`, `path`). The hashes
  are in the human output and in the downstream `run.json`, not in this list.
