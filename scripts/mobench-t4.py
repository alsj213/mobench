#!/usr/bin/env python3
"""T4 档驱动：逐轮冷却 + mobench 原生流水线（模型只推一次）。

T4 协议要求**每轮独立冷却、对齐起始温度**。MNN 对温度敏感：三轮连续运行会让
设备从 34.5 °C 升到 49.2 °C，pp 由 193 衰减到 136 —— 未做轮间冷却的采集不可比
（见 `docs/compare-qwen3-0.6b-llamacpp-vs-mnn.md` §1.3，以及
`results/t4-pinlock-20260910/` 中那次因此被废弃的采集）。

但 `mobench run` 没有轮间钩子；拆成 N 次调用又不行 —— adb 平台按**进程**缓存
已推送的文件（`AdbPlatform._pushed` / `_synced`），每次调用都会重推整个模型。
实测推送 19.5 MB/s，4B 档一次约 3 GB ≈ 2.5 min，5 轮就是十几分钟纯搬运。

本脚本在**同一进程**内跑完整条 `run_pipeline`（因此只推一次），仅给
`AdbPlatform.run` 套一层「先等设备降到目标温度再执行」的前置。除冷却外一切走
mobench 原生路径，records / run.json / tee 原始日志与 `mobench run` 完全一致；
另在 run 目录落一份 `t4_conditions.json`，逐次尝试记录起始温度与退出码（红线 R3 可溯源）。

引擎非零退出时按 `--max-attempts` 重试（默认 3）。需要重试的真实场景：MNN 4B
档在**加载阶段**有约五成概率被 Android 的指针标记检查中止（rc=134，日志里只有
一行 `Pointer tag … was truncated`），此时尚未产生任何测量输出，重试是干净数据。
失败日志另存为 `<log>.failedN`，每次尝试都留在 `t4_conditions.json` 里。

用法：
    python3 scripts/mobench-t4.py --config configs/local/<档>.yml [--rounds 5]

等价于 `mobench run --rounds N`，但每轮之间会等设备冷却到 `--target-c`。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time

from mobench.adapters.base import AdapterError
from mobench.config import ConfigError, load_config
from mobench.platform.adb import AdbPlatform
from mobench.platform.base import PlatformError
from mobench.run import RunError, run_pipeline

DEFAULT_TARGET_C = 35.0
DEFAULT_TIMEOUT_S = 1800.0
POLL_S = 5.0
# 连续读不到温度多少次就放弃（设备无 thermal zone / 权限不足），避免空转到超时
MAX_UNREADABLE = 6
# 引擎非零退出时的重试上限。MNN 4B 档在**加载阶段**有约五成概率被 Android
# 指针标记检查中止（rc=134，日志只有 "Pointer tag … was truncated"），
# 此时尚未产生任何测量输出，重试拿到的是干净数据；每次尝试都记入
# t4_conditions.json，失败日志另存为 <log>.failedN，不静默吞掉（红线 R4/R8）。
DEFAULT_MAX_ATTEMPTS = 3
# 执行超时在 events 里用的退出码（沿用 coreutils timeout 的约定）
TIMEOUT_EXIT = 124


def _read_temp_c(platform) -> float | None:
    """当前温度（摄氏度）；不可得返回 None。"""
    try:
        info = platform._temp_c()
    except Exception:  # noqa: BLE001 - 温度读不到不该中断测量，交给下面的判据处理
        return None
    value = info.get("value") if isinstance(info, dict) else None
    return float(value) if value is not None else None


def cool_to(platform, target_c: float, timeout_s: float) -> float:
    """阻塞到设备温度 <= target_c，返回起始温度。超时/读不到温度则抛 RunError。"""
    started = time.time()
    unreadable = 0
    last = None
    while True:
        temp = _read_temp_c(platform)
        if temp is None:
            unreadable += 1
            if unreadable >= MAX_UNREADABLE:
                raise RunError(
                    "cooldown",
                    f"设备温度连续 {unreadable} 次读取失败，无法执行 T4 冷却；"
                    f"确认 thermal_zone 可读，或用 --target-c 调低要求")
        else:
            unreadable = 0
            last = temp
            if temp <= target_c:
                return temp
        if time.time() - started > timeout_s:
            raise RunError(
                "cooldown",
                f"设备 {timeout_s:.0f}s 内未降到 {target_c}°C（最后读数 {last}°C）")
        time.sleep(POLL_S)


def install_cooldown(target_c: float, timeout_s: float, events: list,
                     max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> None:
    """给 AdbPlatform.run 加冷却前置与非零退出重试；每次尝试记入 events。"""
    original = AdbPlatform.run

    def cooling_run(self, cmd, log_path, cwd=None, echo=True):
        last_rc = None
        for attempt in range(1, max_attempts + 1):
            temp = cool_to(self, target_c, timeout_s)
            try:
                rc = original(self, cmd, log_path, cwd=cwd, echo=echo)
            except subprocess.TimeoutExpired:
                # 执行超时（mobench 对引擎调用硬编码 600 s）：单轮超时的原因
                # 往往是这一轮本来就比预期长，重试是合理的；但仍记入 events。
                rc = TIMEOUT_EXIT
            events.append({
                "log": os.path.basename(log_path),
                "attempt": attempt,
                "start_temp_c": temp,
                "rc": rc,
                "at": datetime.datetime.now().isoformat(timespec="seconds"),
            })
            if rc == 0:
                print(f"[t4] 冷却就绪 {temp:.1f}°C -> {os.path.basename(log_path)}",
                      file=sys.stderr)
                return rc
            last_rc = rc
            # 保留失败证据：下一次尝试会覆盖同一个 log_path
            failed = f"{log_path}.failed{attempt}"
            if os.path.exists(log_path):
                shutil.copy2(log_path, failed)
            print(f"[t4] 引擎退出 {rc}（第 {attempt}/{max_attempts} 次）"
                  f"起始 {temp:.1f}°C，失败日志 -> {os.path.basename(failed)}",
                  file=sys.stderr)
            if attempt == max_attempts:
                print(f"[t4] 已达重试上限 {max_attempts}，放弃本轮", file=sys.stderr)
        return last_rc

    AdbPlatform.run = cooling_run


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="T4 档驱动：逐轮冷却 + mobench 流水线（模型只推一次）")
    ap.add_argument("--config", required=True, help="mobench.yml 路径")
    ap.add_argument("--rounds", type=int, default=None,
                    help="轮数（缺省用 config 里的 workload.rounds）")
    ap.add_argument("--target-c", type=float, default=DEFAULT_TARGET_C,
                    help=f"每轮起始温度上限，默认 {DEFAULT_TARGET_C}°C")
    ap.add_argument("--cool-timeout", type=float, default=DEFAULT_TIMEOUT_S,
                    help=f"单轮冷却等待上限（秒），默认 {DEFAULT_TIMEOUT_S:.0f}")
    ap.add_argument("--threads", type=int, default=None, help="覆盖线程数")
    ap.add_argument("--model", action="append", help="只跑指定 model id（可重复）")
    ap.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                    help=f"单轮引擎非零退出的重试上限，默认 {DEFAULT_MAX_ATTEMPTS}")
    ap.add_argument("--json", action="store_true", help="stdout 只输出 JSON")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    events: list = []
    try:
        cfg = load_config(args.config)
        install_cooldown(args.target_c, args.cool_timeout, events,
                         max_attempts=args.max_attempts)
        meta = run_pipeline(
            cfg,
            model_filter=args.model,
            threads_override=[args.threads] if args.threads else None,
            rounds_override=args.rounds,
            json_mode=args.json,
        )
    except (ConfigError, RunError, AdapterError, PlatformError) as exc:
        print(f"t4: error: {exc}", file=sys.stderr)
        return 1

    run_dir = os.path.dirname(meta["records"])
    cond_path = os.path.join(run_dir, "t4_conditions.json")
    with open(cond_path, "w", encoding="utf-8") as fh:
        json.dump({
            "target_c": args.target_c,
            "max_attempts": args.max_attempts,
            "n_attempts": len(events),
            "n_ok": sum(1 for e in events if e["rc"] == 0),
            "measurements": events,
        }, fh, indent=2, ensure_ascii=False)

    if args.json:
        print(json.dumps(meta, indent=2, ensure_ascii=False))
    else:
        warn = "  [WARN: 降频风险 temp>45C]" if meta["throttle_risk"] else ""
        print(f"[t4] ok: {meta['n_records']} records -> {meta['records']}{warn}")
        print(f"[t4] 冷却记录 -> {cond_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
