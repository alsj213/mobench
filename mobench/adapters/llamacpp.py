"""llama.cpp adapter — drives llama-bench (spec: benchmark-core 内置 llama.cpp 适配器).

Invokes the engine project's own ``llama-bench`` and maps its results to
canonical records (prompt eval / generation timings & tok/s). Supports both
current llama.cpp (``-o json`` -> a top-level array of per-test items carrying
``avg_ts`` throughput) and older versions (``--json`` -> a ``results`` list
with ``t_prompt_processing`` / ``t_generation`` ms). llama.cpp never produces
ONNX-reference output, so no precision records are made here.
"""
from __future__ import annotations

import json
import os
import re
import subprocess

from mobench.adapters.base import AdapterError, EngineAdapter, _find_binary

# Result keys produced by llama-bench --json. Upstream may rename these keys;
# isolating the mapping here keeps that drift out of the core.
JSON_TIMING_KEYS = {
    "t_prompt_processing": ("prompt_eval_ms_tok", "ms/token"),
    "t_generation": ("gen_ms_tok", "ms/token"),
}


class LlamacppAdapter(EngineAdapter):
    engine = "llamacpp"
    metric_vocab = ("prompt_eval_ms_tok", "gen_ms_tok", "gen_tok_s")

    def _bench_name(self):
        build = self.cfg.get("build") or {}
        return build.get("bench_bin") or "llama-bench"

    def resolve_binary(self) -> str:
        path = _find_binary(self.cfg, self._bench_name())
        if not os.path.isfile(path):
            raise AdapterError(
                f"llama-bench binary not found (looked at {path}). "
                f"Set build.bin_dir/build.bench_bin in mobench.yml or build the engine."
            )
        return path

    def version(self) -> str:
        binary = self.resolve_binary()
        try:
            proc = subprocess.run([binary, "--version"], capture_output=True,
                                  text=True, timeout=20)
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip().splitlines()[0]
        except (OSError, subprocess.TimeoutExpired):
            pass
        source_dir = (self.cfg.get("build") or {}).get("source_dir") or "."
        try:
            proc = subprocess.run(["git", "-C", source_dir, "log", "--oneline", "-1"],
                                  capture_output=True, text=True, timeout=15)
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except OSError:
            pass
        return "unknown"

    def build_cmds(self, model: dict, threads: int, rounds: int):
        binary = self.resolve_binary()
        # Current llama.cpp prints JSON via `-o json`; older builds used
        # `--json`. Force the legacy form with `llamacpp: {output: --json}`
        # in mobench.yml when benchmarking an older engine tree.
        override = (self.cfg.get("llamacpp") or {}).get("output")
        output = [override] if override else ["-o", "json"]
        cmd = [binary, "-m", model["path"], "-t", str(threads),
               "-r", str(rounds)] + output
        workload = self.cfg.get("workload") or {}
        if workload.get("prompt_tokens"):
            cmd += ["-p", str(workload["prompt_tokens"])]
        if workload.get("gen_tokens"):
            cmd += ["-n", str(workload["gen_tokens"])]
        # Opt-in decode/KV controls (KV 量化波 2026-10-10): absent keys keep
        # the legacy command shape bit-for-bit, so existing configs' commands
        # and measurements are unchanged. A matrix cell must declare these
        # (even defaults like f16) so its records are self-describing.
        # cli_args 原样透传（如 ["--no-warmup"]）。
        if workload.get("n_depth"):
            cmd += ["-d", str(workload["n_depth"])]
        cmd += self._kv_flags(workload)
        for arg in workload.get("cli_args") or []:
            cmd.append(str(arg))
        return [cmd]

    def _kv_flags(self, workload):
        """KV-identity flags shared by bench and validation commands."""
        flags = []
        if workload.get("cache_type_k"):
            flags += ["-ctk", str(workload["cache_type_k"])]
        if workload.get("cache_type_v"):
            flags += ["-ctv", str(workload["cache_type_v"])]
        if workload.get("flash_attn"):
            flags += ["-fa", str(workload["flash_attn"])]
        return flags

    def parse(self, log_path: str, model: dict, threads: int, round_no: int) -> list:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        payload = _extract_json(text)
        results = payload if isinstance(payload, list) else None
        if results is None and isinstance(payload, dict):
            results = payload.get("results")
        if not isinstance(results, list):
            raise AdapterError(
                f"cannot parse llama-bench output in {log_path}: "
                f"structure does not match `-o json` / `--json`"
            )
        records = []
        for item in results:
            if not isinstance(item, dict):
                continue
            records.extend(self._records_from_item(
                item, model, threads, round_no, log_path))
        if not records:
            raise AdapterError(
                f"cannot parse llama-bench output in {log_path}: no recognized timing fields"
            )
        return records

    def _records_from_item(self, item, model, threads, round_no, log_path):
        """Map one llama-bench test item into canonical records.

        Current llama.cpp (`-o json`) emits a top-level array where each item is
        a single prompt (``n_gen == 0``) or generation (``n_gen > 0``) test
        carrying throughput in ``avg_ts`` tok/s. Older `--json` output wraps a
        ``results`` list whose items carry ``t_prompt_processing`` /
        ``t_generation`` milliseconds.
        """
        recs = []
        params = self._workload_params(item)
        # -- legacy keys (older llama.cpp --json) ---------------------------
        legacy = False
        for key, (metric, unit) in JSON_TIMING_KEYS.items():
            value = item.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                recs.append(self._rec(model, threads, round_no, metric,
                                      float(value), unit, log_path, params))
                legacy = True
        gen_ms = item.get("t_generation")
        if isinstance(gen_ms, (int, float)) and not isinstance(gen_ms, bool) and gen_ms > 0:
            recs.append(self._rec(model, threads, round_no, "gen_tok_s",
                                  1000.0 / float(gen_ms), "tok/s", log_path, params))
            legacy = True
        if legacy:
            return recs
        # -- current llama.cpp: per-test throughput --------------------------
        ts = item.get("avg_ts")
        if not (isinstance(ts, (int, float)) and not isinstance(ts, bool) and ts > 0):
            return recs
        if (item.get("n_gen") or 0) > 0:
            recs.append(self._rec(model, threads, round_no, "gen_tok_s",
                                  float(ts), "tok/s", log_path, params))
            recs.append(self._rec(model, threads, round_no, "gen_ms_tok",
                                  1000.0 / float(ts), "ms/token", log_path, params))
        elif (item.get("n_prompt") or 0) > 0:
            recs.append(self._rec(model, threads, round_no, "prompt_eval_ms_tok",
                                  1000.0 / float(ts), "ms/token", log_path, params))
        return recs

    def _workload_params(self, item=None):
        """Record params: workload echoes + matrix-cell identity.

        Cell controls are recorded under canonical llama-bench names
        (``n_depth`` / ``type_k`` / ``type_v`` / ``flash_attn``), from the
        config first and overridden by the per-test item fields when the
        engine prints them — so records from a multi-test invocation stay
        distinguishable and every value traces to the cell that produced it.
        """
        wl = self.cfg.get("workload") or {}
        params = {k: wl[k] for k in ("prompt_tokens", "gen_tokens") if wl.get(k)}
        controls = {
            "n_depth": wl.get("n_depth"),
            "type_k": wl.get("cache_type_k"),
            "type_v": wl.get("cache_type_v"),
            "flash_attn": wl.get("flash_attn"),
        }
        if isinstance(item, dict):
            for key in controls:
                if item.get(key) is not None:
                    controls[key] = item[key]
        params.update({k: v for k, v in controls.items() if v is not None})
        return params or None

    def _backend(self):
        """Engine backend label from config llamacpp.backend (e.g. cpu/metal)."""
        return (self.cfg.get("llamacpp") or {}).get("backend")

    def _rec(self, model, threads, round_no, metric, value, unit, log_path,
             params=None):
        return {
            "schema": "mobench/record@1",
            "engine": self.engine,
            "workload_id": model["id"],
            "metric": metric,
            "value": round(value, 6),
            "unit": unit,
            "threads": threads,
            "precision": model.get("precision"),
            "round": round_no,
            "params": params if params is not None else self._workload_params(),
            "backend": self._backend(),
            "platform": "host",
            "source_log": log_path,
        }

    # -- precision validation (add-correctness-validation) -------------------
    def _validate_bin(self, name):
        path = _find_binary(self.cfg, name)
        if not os.path.isfile(path):
            raise AdapterError(
                f"{name} binary not found (looked at {path}); build the "
                f"engine or set build.bin_dir in mobench.yml")
        return path

    def build_validate_cmds(self, model, val_cfg, workload, threads):
        """Instrument jobs: ``ppl`` (llama-perplexity) then ``greedy``.

        Command shapes follow the fork's tools (source-verified):
        - greedy uses the classic ``llama-completion`` tool — the new-UI
          ``llama-cli`` prints animation frames and rejects ``-no-cnv``;
          ``-no-cnv`` is required (Qwen3 carries a chat template and would
          otherwise slide into conversation mode);
        - the extracted stream keeps the prompt echo: the engine has no
          generation-only channel (``--no-display-prompt`` suppresses the
          generated text as well), so ``include_prompt_echo`` is stamped.
        """
        val_cfg = val_cfg or {}
        workload = workload or {}
        kv = self._kv_flags(workload)
        jobs = []
        ppl = val_cfg.get("ppl")
        if ppl:
            cmd = [self._validate_bin("llama-perplexity"),
                   "-m", model["path"], "-f", ppl["dataset"],
                   "-t", str(threads)]
            if ppl.get("ctx"):
                cmd += ["-c", str(ppl["ctx"])]
            if ppl.get("chunks"):
                cmd += ["--chunks", str(ppl["chunks"])]
            jobs.append({"instrument": "ppl", "cmd": cmd + kv, "prompt": None,
                         "include_prompt_echo": False})
        gen_tokens = val_cfg.get("gen_tokens")
        for prompt in val_cfg.get("prompts") or []:
            cmd = [self._validate_bin("llama-completion"),
                   "-m", model["path"], "-f", prompt,
                   "-no-cnv", "-t", str(threads), "--temp", "0"]
            if gen_tokens:
                cmd += ["-n", str(gen_tokens)]
            jobs.append({"instrument": "greedy", "cmd": cmd + kv,
                         "prompt": prompt, "include_prompt_echo": True})
        if not jobs:
            raise AdapterError(
                "validation declares no instrument: provide 'prompts' "
                "(greedy) and/or a 'ppl' block")
        return jobs

    def parse_validate(self, job, log_path):
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        instrument = job.get("instrument")
        if instrument == "ppl":
            return self._parse_ppl(text, log_path)
        if instrument == "greedy":
            return self._parse_greedy(text, log_path, job.get("prompt"))
        raise AdapterError(f"unknown validation instrument {instrument!r}")

    def _parse_ppl(self, text, log_path):
        match = _PPL_FINAL_RE.search(text)
        if not match:
            raise AdapterError(
                f"cannot parse llama-perplexity output in {log_path}: "
                f"no 'Final estimate: PPL = …' line")
        chunks = _PPL_CHUNKS_RE.search(text)
        return {"value": float(match.group(1)),
                "chunks": int(chunks.group(1)) if chunks else None}

    def _parse_greedy(self, text, log_path, prompt_path=None):
        """Extract the generated stream from engine output (two layouts).

        The engine routes level-O (stream/prompt echo) messages to stdout and
        every other log level to stderr (common/log.cpp). Host runs merge both
        chronologically — the stream sits between the ``generate:`` header and
        the perf block. adb captures stdout and stderr separately and the
        platform concatenates stdout first — the stream then leads the file
        and the header/perf lines follow. Both layouts are handled; the
        stream is verified to start with the prompt echo when the prompt file
        is readable (a strong extraction-integrity check).
        """
        heads = list(_GEN_HEADER_RE.finditer(text))
        if not heads:
            raise AdapterError(
                f"cannot parse llama-completion output in {log_path}: "
                f"no 'generate: n_ctx = …' header")
        start = heads[-1].end()
        if text.startswith("\n", start):
            start += 1
        perf = _PERF_BLOCK_RE.search(text, start)
        if not perf:
            raise AdapterError(
                f"cannot parse llama-completion output in {log_path}: no "
                f"performance block (common_perf_print:) after the generate "
                f"header — log truncated?")
        # 合并流布局：头与性能块之间即生成流（剔除期间可能出现的带前缀日志行）
        region = _PREFIX_LINE_RE.sub("", text[start:perf.start()])
        region = _strip_stream_tail(region)
        if not region.strip():
            # 拆分流布局（adb 先 stdout 后 stderr）：流在文件头部，
            # 以第一条带前缀的日志行为界
            first_log = _PREFIX_LINE_RE.search(text)
            if first_log and first_log.start() > 0:
                region = _strip_stream_tail(text[:first_log.start()])
        if not region.strip():
            raise AdapterError(
                f"cannot parse llama-completion output in {log_path}: "
                f"generated stream not found in either stream layout")
        if prompt_path and os.path.isfile(prompt_path):
            with open(prompt_path, encoding="utf-8", errors="replace") as fh:
                prompt_text = fh.read().rstrip()
            if prompt_text and not region.startswith(prompt_text):
                raise AdapterError(
                    f"cannot parse llama-completion output in {log_path}: "
                    f"stream does not start with the prompt echo of "
                    f"{prompt_path} — refusing to guess the extraction")
        match = _N_PROMPT_RE.search(text)
        return {"text": region,
                "n_prompt_tokens": int(match.group(1)) if match else None}


ADAPTER = LlamacppAdapter


# -- validation log patterns (source-verified against the engine tools) -----
# llama-perplexity final estimate (tools/perplexity/perplexity.cpp).
_PPL_FINAL_RE = re.compile(r"Final estimate: PPL = ([0-9.]+)")
_PPL_CHUNKS_RE = re.compile(r"calculating perplexity over (\d+) chunks")
# llama-completion stream boundaries (tools/completion/completion.cpp): the
# generation stream starts after the `generate:` LOG_INF header and ends where
# the LOG_INF perf block begins. Engine log lines may carry a
# "<h.mm.ss.mmm> <L> " prefix (the fork enables timestamps/prefix); level-O
# stream text does not.
_GEN_HEADER_RE = re.compile(
    r"^.*generate: n_ctx = \d+, n_batch = \d+, n_predict = \d+, n_keep = \d+$",
    re.MULTILINE)
_PERF_BLOCK_RE = re.compile(r"^.*common_perf_print:", re.MULTILINE)
_PREFIX_LINE_RE = re.compile(r"^\d[\d.]* [DIWE] .*$\n?", re.MULTILINE)
_N_PROMPT_RE = re.compile(
    r"prompt eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) tokens")
_EOG_TAIL = " [end of text]\n\n\n"


def _strip_stream_tail(region):
    """尾部精确剥离：EOG 标记 + 分隔空行（LOG("\\n\\n")）或仅分隔空行，
    尽量保留生成文本自身的换行。"""
    if region.endswith(_EOG_TAIL):
        return region[:-len(_EOG_TAIL)]
    if region.endswith("\n\n"):
        return region[:-2]
    return region


def _extract_json(text):
    """Return parsed JSON object/array or None (whole-file or bracketed subset)."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, (dict, list)):
            return obj
    except json.JSONDecodeError:
        pass
    # bracketed subset -- object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    # bracketed subset -- array (current llama.cpp `-o json`)
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None
