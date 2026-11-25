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
CV_SUMMARY = re.compile(r"\s+max\s*=\s*([\d.]+)\s*ms\s+min\s*=\s*([\d.]+)\s*ms\s+avg\s*=\s*([\d.]+)\s*ms")


class MnnAdapter(EngineAdapter):
    engine = "mnn"
    metric_vocab = ("latency_ms", "fps", "gen_tok_s")

    def _bench_name(self):
        build = self.cfg.get("build") or {}
        return build.get("bench_bin") or "benchmark.out"

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
        """mnn.n_repeat -> llm_bench ``-rep N``; None 表示不传（沿用其默认 5）。

        llm_bench 默认在一次调用内连跑 5 次取均值，中途不冷却。这与 T4 这类
        "每轮独立冷却、对齐起始温度"的协议冲突：单轮内的发热会被平均进结果，
        且单轮时长变成 llama 侧（``-r 1``）的数倍。需要一轮一测的档位显式设
        ``mnn.n_repeat: 1``；不设则行为与既有档位完全一致。
        """
        return (self.cfg.get("mnn") or {}).get("n_repeat")

    def version(self) -> str:
        import subprocess

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
            proc = subprocess.run(["git", "-C", source_dir, "log", "--oneline", "-1"],
                                  capture_output=True, text=True, timeout=15)
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except OSError:
            pass
        return "unknown"

    def _is_cv_bench_out(self) -> bool:
        return "benchmark.out" in self._bench_name()

    def build_cmds(self, model: dict, threads: int, rounds: int):
        binary = self.resolve_binary()
        if self._is_llm_tool():
            # Official MNN llm_bench: -m <config.json>, -a <backend> (config
            # mnn.backend, default cpu), and prompt/gen from workload.
            wl = self.cfg.get("workload") or {}
            cmd = [binary, "-m", model["path"], "-a", self._backend() or "cpu",
                   "-c", "2", "-t", str(threads), "-fa", "0"]
            if wl.get("prompt_tokens"):
                cmd += ["-p", str(wl["prompt_tokens"])]
            if wl.get("gen_tokens"):
                cmd += ["-n", str(wl["gen_tokens"])]
            n_repeat = self._n_repeat()
            if n_repeat is not None:
                cmd += ["-rep", str(n_repeat)]
            return [cmd]
        if self._is_cv_bench_out():
            # MNN CV benchmark.out CLI: <models folder> loop warmup fw threads
            # precision. Run one measurement per mobench round; the model's
            # own directory (models[].dir, else its parent) holds it alone.
            model_dir = model.get("dir") or os.path.dirname(os.path.abspath(model["path"]))
            cmd = [binary, model_dir, "1", "1", "0", str(threads), "2"]
            return [cmd]
        # Other CV/NLP tools: model file first, threads after.
        return [[binary, model["path"], "-t", str(threads)]]

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
            records.append(self._rec(model, threads, round_no, metric,
                                     float(value), unit, log_path))
        if not records:
            raise AdapterError(
                f"cannot parse MNN benchmark output in {log_path}: "
                f"no recognized latency/fps/tok_s fields"
            )
        return records

    def _workload_params(self):
        wl = self.cfg.get("workload") or {}
        return {k: wl[k] for k in ("prompt_tokens", "gen_tokens") if wl.get(k)} or None

    def _rec(self, model, threads, round_no, metric, value, unit, log_path):
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
            "params": self._workload_params(),
            "backend": self._backend(),
            "platform": "host",
            "source_log": log_path,
        }

    def _parse_llm_rows(self, text, log_path, model, threads, round_no):
        """Parse llm_bench stdout rows (``pp<n>``/``tg<n>`` + ``t/s``)."""
        found = {}
        for kind, _n, tps in LLM_ROW.findall(text):
            try:
                tps = float(tps)
            except ValueError:
                continue
            if tps <= 0:
                continue
            if kind == "tg":
                found.setdefault("gen_tok_s", tps)
            elif kind == "pp":
                # prompt throughput tok/s -> per-token ms for a latency metric
                found.setdefault("prompt_eval_ms_tok", 1000.0 / tps)
        records = []
        for key, tps in found.items():
            if key == "gen_tok_s":
                records.append(self._rec(model, threads, round_no, "gen_tok_s",
                                         tps, "tok/s", log_path))
                records.append(self._rec(model, threads, round_no, "gen_ms_tok",
                                         1000.0 / tps, "ms/token", log_path))
            else:
                records.append(self._rec(model, threads, round_no,
                                         "prompt_eval_ms_tok", tps,
                                         "ms/token", log_path))
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
