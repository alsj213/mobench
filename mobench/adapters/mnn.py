"""MNN adapter (spec: benchmark-core 内置 MNN 适配器).

Drives the engine's own benchmark/LLM tooling. Two real entry points are
supported:
  * CV/NLP ``benchmark.out``-style tools that emit a JSON object or
    ``key: value`` lines (``latency_ms`` / ``fps``);
  * the official ``llm_bench`` LLM benchmark (``-m <config.json> …``) whose
    stdout table prints ``pp<n>`` / ``tg<n>`` rows with ``t/s`` throughput —
    mapped to prompt-eval / generation records.

ONNX-reference precision is only attempted when the model is an onnx graph
and a local ORT is present (otherwise the analysis layer labels it
unavailable — never fabricated).
"""
from __future__ import annotations

import json
import os
import re
import subprocess

from mobench.adapters.base import AdapterError, EngineAdapter, _find_binary

LINE_METRICS = {
    "latency_ms": ("latency_ms", "ms"),
    "fps": ("fps", "FPS"),
    "gen_tok_s": ("gen_tok_s", "tok/s"),
}

# llm_bench stdout rows: | <model> | … | pp8 | 309.79 ± 0.00 |
LLM_ROW = re.compile(r"\|\s*(pp|tg)(\d+)\s*\|\s*([\d.]+)\s*±")

# MNN CV benchmark.out summary row:
#   [ - ] squeezenetv1.1.mnn  max = 4.681 ms  min = 4.466 ms  avg = 4.574 ms
CV_SUMMARY = re.compile(
    r"\s+max\s*=\s*([\d.]+)\s*ms\s+min\s*=\s*([\d.]+)\s*ms\s+avg\s*=\s*([\d.]+)\s*ms"
)


class MnnAdapter(EngineAdapter):
    engine = "mnn"
    metric_vocab = ("latency_ms", "fps", "gen_tok_s")

    def _bench_name(self):
        build = self.cfg.get("build") or {}
        return build.get("bench_bin") or "benchmark.out"

    def process_name(self) -> str:
        return os.path.basename(self._bench_name())

    def validate_config(self, cfg: dict) -> list[str]:
        errors = []
        mnn = cfg.get("mnn")
        if mnn is not None and not isinstance(mnn, dict):
            return ["mnn must be a mapping"]
        if isinstance(mnn, dict) and mnn.get("n_repeat") is not None:
            n_repeat = mnn["n_repeat"]
            if isinstance(n_repeat, bool) or not isinstance(n_repeat, int) or n_repeat < 1:
                errors.append(
                    f"mnn.n_repeat must be a positive integer (llm_bench -rep), "
                    f"got {n_repeat!r}"
                )
        return errors

    def convert_model(self, model: dict) -> list[str]:
        """MNNConvert (onnx -> mnn) from the MNN build dir."""
        source = model.get("source") or {}
        build = self.cfg.get("build") or {}
        tool = os.path.join(build.get("bin_dir", ""), "MNNConvert")
        if not os.path.isfile(tool):
            raise AdapterError(f"MNNConvert not found: {tool} (set build.bin_dir)")
        onnx = source.get("dir") or source.get("onnx")
        if not onnx:
            raise AdapterError("convert_with=mnn requires source.dir (onnx file)")
        out = model["path"]
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        argv = [tool, "--modelFile", onnx, "--MNNModel", out]
        subprocess.run(argv, check=True, capture_output=True, text=True)
        return argv

    def resolve_binary(self) -> str:
        path = _find_binary(self.cfg, self._bench_name())
        if not os.path.isfile(path):
            raise AdapterError(
                f"MNN benchmark binary not found (looked at {path}). "
                f"Set build.bin_dir/build.bench_bin in mobench.yml or build MNN."
            )
        return path

    def _is_llm_tool(self) -> bool:
        name = self._bench_name()
        return name in ("llm_bench", "llm") or name.endswith("llm_bench")

    def _backend(self):
        """Engine backend (-a) from config mnn.backend; default None (cpu)."""
        return (self.cfg.get("mnn") or {}).get("backend")

    def _n_repeat(self):
        """mnn.n_repeat -> llm_bench ``-rep N``; None 表示不传（沿用其默认 5）。"""
        return (self.cfg.get("mnn") or {}).get("n_repeat")

    def version(self) -> str:
        # llm_bench with no args asserts/crashes — skip the probe for it.
        if not self._is_llm_tool():
            binary = self.resolve_binary()
            try:
                proc = subprocess.run([binary], capture_output=True, text=True, timeout=20)
                for line in (proc.stdout + proc.stderr).splitlines():
                    if "version" in line.lower() or "MNN" in line:
                        return line.strip()
            except (OSError, subprocess.TimeoutExpired):
                pass
        source_dir = (self.cfg.get("build") or {}).get("source_dir") or "."
        try:
            proc = subprocess.run(
                ["git", "-C", source_dir, "log", "--oneline", "-1"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except OSError:
            pass
        return "unknown"

    def _is_cv_bench_out(self) -> bool:
        return "benchmark.out" in self._bench_name()

    def build_cmds(self, model: dict, threads: int, rounds: int):
        binary = self.resolve_binary()
        wl = self.cfg.get("workload") or {}
        extra_args = [str(arg) for arg in (wl.get("cli_args") or [])]
        if self._is_llm_tool():
            # Official MNN llm_bench: -m <config.json>, -a <backend> (config
            # mnn.backend, default cpu), and prompt/gen from workload.
            cmd = [
                binary,
                "-m",
                model["path"],
                "-a",
                self._backend() or "cpu",
                "-c",
                "2",
                "-t",
                str(threads),
                "-fa",
                "0",
            ]
            if wl.get("prompt_tokens"):
                cmd += ["-p", str(wl["prompt_tokens"])]
            if wl.get("gen_tokens"):
                cmd += ["-n", str(wl["gen_tokens"])]
            n_repeat = self._n_repeat()
            if n_repeat is not None:
                cmd += ["-rep", str(n_repeat)]
            return [cmd + extra_args]
        if self._is_cv_bench_out():
            # MNN CV benchmark.out CLI: <models folder> loop warmup fw threads
            # precision. Run one measurement per mobench round; the model's
            # own directory (models[].dir, else its parent) holds it alone.
            model_dir = model.get("dir") or os.path.dirname(os.path.abspath(model["path"]))
            cmd = [binary, model_dir, "1", "1", "0", str(threads), "2"]
            return [cmd + extra_args]
        # Other CV/NLP tools: model file first, threads after.
        return [[binary, model["path"], "-t", str(threads), *extra_args]]

    def parse(self, log_path: str, model: dict, threads: int, round_no: int) -> list:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        records = self._parse_llm_rows(text, log_path, model, threads, round_no)
        if records:
            return records
        metrics = self._extract_metrics(text, log_path)
        records = []
        for key, value in metrics.items():
            metric, unit = LINE_METRICS[key]
            records.append(
                self._rec(model, threads, round_no, metric, float(value), unit, log_path)
            )
        if not records:
            raise AdapterError(
                f"cannot parse MNN benchmark output in {log_path}: "
                f"no recognized latency/fps/tok_s fields"
            )
        return records

    def _workload_params(self):
        wl = self.cfg.get("workload") or {}
        return {k: wl[k] for k in ("prompt_tokens", "gen_tokens") if wl.get(k)} or None

    def _rec(self, model, threads, round_no, metric, value, unit, log_path, params=None):
        return {
            "schema": "mobench/record@1",
            "engine": self.engine,
            "workload_id": model["id"],
            "metric": metric,
            "value": round(float(value), 6),
            "unit": unit,
            "threads": threads,
            "precision": model.get("precision"),
            "round": round_no,
            "params": params if params is not None else self._workload_params(),
            "backend": self._backend(),
            "platform": "host",
            "source_log": log_path,
        }

    def _parse_llm_rows(self, text, log_path, model, threads, round_no):
        """Parse llm_bench stdout rows (``pp<n>``/``tg<n>`` + ``t/s``)."""
        parsed_rows = []
        for kind, n_raw, tps_raw in LLM_ROW.findall(text):
            try:
                n_val = int(n_raw)
                tps = float(tps_raw)
            except ValueError:
                continue
            if tps <= 0:
                continue
            parsed_rows.append((kind, n_val, tps))
        multi_pp = len({n for k, n, _ in parsed_rows if k == "pp"}) > 1
        multi_tg = len({n for k, n, _ in parsed_rows if k == "tg"}) > 1
        base_params = self._workload_params()
        records = []
        for kind, n_val, tps in parsed_rows:
            if kind == "tg":
                params = {**(base_params or {}), "gen_tokens": n_val} if multi_tg else base_params
                records.append(
                    self._rec(
                        model, threads, round_no, "gen_tok_s", tps, "tok/s", log_path, params=params
                    )
                )
                records.append(
                    self._rec(
                        model,
                        threads,
                        round_no,
                        "gen_ms_tok",
                        1000.0 / tps,
                        "ms/token",
                        log_path,
                        params=params,
                    )
                )
            elif kind == "pp":
                params = (
                    {**(base_params or {}), "prompt_tokens": n_val} if multi_pp else base_params
                )
                records.append(
                    self._rec(
                        model,
                        threads,
                        round_no,
                        "prompt_eval_ms_tok",
                        1000.0 / tps,
                        "ms/token",
                        log_path,
                        params=params,
                    )
                )
        return records

    @staticmethod
    def _extract_metrics(text, log_path):
        """Accept MNN CV benchmark.out summary / JSON object / ``key: value`` lines."""
        text = text.strip()
        m = CV_SUMMARY.search(text)
        if m:
            avg = float(m.group(3))
            if avg > 0:
                return {"latency_ms": avg, "fps": 1000.0 / avg}
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return {k: obj[k] for k in LINE_METRICS if k in obj}
        except json.JSONDecodeError:
            pass
        found = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, _, raw = line.partition(":")
            key = key.strip()
            if key in LINE_METRICS:
                try:
                    found[key] = float(raw.strip())
                except ValueError:
                    raise AdapterError(
                        f"cannot parse MNN output in {log_path}: bad value for {key}: {raw!r}"
                    )
        return found


ADAPTER = MnnAdapter
