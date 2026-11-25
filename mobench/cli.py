"""Deterministic CLI for the mobench core (spec: benchmark-core 确定性命令行入口).

Subcommands: run / validate / profile / memory / trace / env / stats / regression / power / report / prepare.
Every subcommand supports --json stable output; failures exit non-zero.
``validate compare`` additionally returns 2 on a FAIL verdict (a gate, unlike
the advisory regression report). Implementation of each command lives in its
own module and is imported lazily so ``--help`` never pulls in optional
tooling.
"""
from __future__ import annotations

import argparse
import sys


def _config_path_arg(parser, default="mobench.yml"):
    parser.add_argument("--config", default=default,
                        help=f"path to config (default {default})")


def cmd_run(args):
    from mobench import run
    return run.main(args)


def cmd_env(args):
    from mobench import snapshot
    return snapshot.main(args)


def cmd_stats(args):
    from mobench.analysis import stats
    return stats.main(args)


def cmd_runs(args):
    from mobench import runs_index
    return runs_index.main(args)


def cmd_regression(args):
    from mobench.analysis import regression
    return regression.main(args)


def cmd_validate(args):
    if args.action == "run":
        from mobench import validation
        return validation.main(args)
    from mobench.analysis import correctness
    return correctness.main(args)


def cmd_power(args):
    from mobench.analysis import power
    return power.main(args)


def cmd_report(args):
    from mobench import report
    return report.main(args)


def cmd_prepare(args):
    from mobench import prepare
    return prepare.main(args)


def cmd_profile(args):
    from mobench import profiling
    return profiling.main(args)


def cmd_memory(args):
    from mobench import memory
    return memory.main(args)


def cmd_trace(args):
    from mobench import trace
    return trace.main(args)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="mobench",
        description="Generic engine benchmarking core (llama.cpp / MNN).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="execute the full benchmark pipeline")
    _config_path_arg(p)
    p.add_argument("--engine", help="override engine (llamacpp|mnn)")
    p.add_argument("--model", action="append", default=None, help="filter model ids")
    p.add_argument("--threads", type=int, help="override threads")
    p.add_argument("--rounds", type=int, help="override rounds")
    p.add_argument("--label", help="free-form label written into run.json (e.g. k8-b-neon)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("profile", help="sample one run -> flamegraph/top/IPC (host perf | adb simpleperf)")
    _config_path_arg(p)
    p.add_argument("--model", help="model id to profile (default first)")
    p.add_argument("--threads", type=int, help="threads (default first in workload)")
    p.add_argument("--freq", type=int, default=2000, help="sample frequency Hz")
    p.add_argument("--events", nargs="*", default=None,
                   help="stat events (default cpu-cycles,instructions,cache*,L1*)")
    p.add_argument("--ndk-dir", default=None,
                   help="path to NDK simpleperf dir with report_sample.py")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("memory", help="sample process RSS during a long run -> curve/peak")
    _config_path_arg(p)
    p.add_argument("--model", help="model id (default first)")
    p.add_argument("--threads", type=int, help="threads (default first in workload)")
    p.add_argument("--prompt", type=int, help="override prompt tokens (default config)")
    p.add_argument("--gen", type=int, help="override gen tokens (use large e.g. 512+)")
    p.add_argument("--interval", type=float, default=0.2, help="poll seconds")
    p.add_argument("--max-sec", type=float, default=300, help="sampling cap")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_memory)

    p = sub.add_parser("trace", help="capture atrace CPU-frequency trace during a run")
    _config_path_arg(p)
    p.add_argument("--model", help="model id (default first)")
    p.add_argument("--threads", type=int, help="threads (default first in workload)")
    p.add_argument("--dur", type=float, default=20.0, help="capture window seconds")
    p.add_argument("--atrace-cats", nargs="+", default=["freq"],
                   help="atrace categories (default freq)")
    p.add_argument("--prompt", type=int, help="override prompt tokens")
    p.add_argument("--gen", type=int, help="override gen tokens")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_trace)

    p = sub.add_parser("env", help="capture environment snapshot")
    _config_path_arg(p)
    p.add_argument("--out", default=None, help="output JSON path")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("stats", help="multi-run statistics over records")
    p.add_argument("inputs", nargs="+", help="records files or dirs")
    p.add_argument("--cv-threshold", type=float, default=5.0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("regression", help="baseline save/compare over records")
    p.add_argument("action", choices=["save", "compare"])
    p.add_argument("--results", nargs="+", help="records files/dirs")
    p.add_argument("--baseline", help="baseline.json path")
    p.add_argument("--out", default=None, help="output baseline path for save")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_regression)

    p = sub.add_parser(
        "validate",
        help="precision validation: run instruments / compare artifacts "
             "(PASS/WARN=0, FAIL=2, cannot-compare=1)")
    p.add_argument("action", choices=["run", "compare"])
    _config_path_arg(p)
    p.add_argument("--config-b", default=None,
                   help="run: second config for an A/B comparison on "
                        "identical controls")
    p.add_argument("--out", default=None,
                   help="run: output dir (default results/validate_<ts>)")
    p.add_argument("--threads", type=int, default=None,
                   help="run: override threads (a controls field)")
    p.add_argument("--baseline", default=None,
                   help="compare: baseline/golden validate.json; pass alone "
                        "for an A/B artifact (side a vs b)")
    p.add_argument("--candidate", default=None,
                   help="compare: candidate validate.json (cross-artifact)")
    p.add_argument("--tolerance", type=float, default=0.0,
                   help="compare: percent divergence allowance for greedy "
                        "token streams (default exact)")
    p.add_argument("--ppl-tolerance", type=float, default=None,
                   help="compare: percent allowance for PPL (default exact)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("power", help="energy / perf-per-watt from platform data")
    _config_path_arg(p)
    p.add_argument("--source", default=None, help="power data source file")
    p.add_argument("--inference-count", type=int, help="inferences in the measured window")
    p.add_argument("--throughput", type=float, default=None,
                   help="throughput (FPS or tok/s) for perf/Watt")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_power)

    p = sub.add_parser("report", help="generate Markdown/HTML report from records")
    p.add_argument("inputs", nargs="+", help="records files/dirs")
    p.add_argument("--out", default=None, help="output dir (default results/)")
    p.add_argument("--format", choices=["md", "html", "both", "chart"], default="both")
    p.add_argument("--chart-by", choices=["backend", "engine", "matrix"], default="backend",
                   help="chart 面板内对比维度（backend/engine/matrix）")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("prepare", help="prepare models (self-provided/download/convert)")
    _config_path_arg(p)
    p.add_argument("model_ids", nargs="*", default=None, help="model ids to prepare (all if empty)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("runs", help="index completed runs under results/ (one row per run.json)")
    p.add_argument("--results", default="results",
                   help="results directory to scan (default results)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_runs)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse already printed usage for bad input
        return int(exc.code or 0)
    try:
        return args.func(args) or 0
    except Exception as exc:  # deterministic CLI: report and exit non-zero
        print(f"mobench {args.command}: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
