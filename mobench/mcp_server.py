"""mobench MCP 工具面：面向 AI Agent 的可审计、fail-fast 基准测试入口。

薄包装：全部经 `python -m mobench <verb> …` 子进程执行，mobench 核心零改动。
每个 verb 的 argv 按**真实 CLI 形状**逐 verb 拼装：
`env/run/power --config …`、`stats/report inputs…`、`regression {save,compare} …`、
`validate {run,compare} …`；
`--json` 由 `_run_cli` 统一追加末尾（`report` 无此旗标，返回 stdout 原文）。
数据可信红线机械化：
- 没有「读旧数据」工具；run 每次都是新子进程（R2/R11）
- 输出即证据（source_log/哈希都在 CLI JSON 里，R3/R9/R10）
- 失败不吞：非零退出带 stderr 原样抛给调用方（R4）
- 制品锚定：expect_handoff 时**先**校验实测二进制 MD5 与清单文件一致，通过才开跑（fail fast）

并发语义（SDK 层，非本模块可控）：MCP stdio 会话逐条处理请求（mcp SDK
`shared/session.py::_receive_loop` 顺序 await 每条消息）——`run`/`profile` 等长任务
执行期间，同一 server 的后续工具调用**排队等待**（客户端可能等到超时，实测
30min 级）。请按协议顺序调用（run 完成后再调 stats / report），避免在长任务期间
并发调用同一连接；需要并行时请建立独立连接。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

mcp = FastMCP("mobench")

_CLI = [sys.executable, "-m", "mobench"]
DEFAULT_CLI_TIMEOUT = float(os.environ.get("MOBENCH_MCP_TIMEOUT", "1800"))


def _run_cli(verb: str, args: list[str], *, json_out: bool = True,
             timeout: float | None = DEFAULT_CLI_TIMEOUT,
             allowed_exit_codes: tuple[int, ...] = (0,)) -> dict:
    """跑 `python -m mobench <verb> <args…> [--json]` 并返回结果。

    args 是该 verb 之后的完整参数（形状由各工具按 CLI 拼装）；`--json` 统一追加末尾。
    `json_out=False` 用于不支持 `--json` 的 verb（report）——此时不解析 stdout，
    原样返回 `{"stdout": …}`（报告工件路径就在 stdout 里）。
    """
    cmd = [*_CLI, verb, *args, *(["--json"] if json_out else [])]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ToolError(
            f"mobench {verb} 执行超时（timeout={timeout}s）"
        ) from exc
    if p.returncode not in allowed_exit_codes:
        raise ToolError(f"mobench {verb} 失败 rc={p.returncode}: {p.stderr.strip()[-800:]}")
    if not json_out:
        return {"stdout": p.stdout}
    try:
        r = json.loads(p.stdout or "{}")
    except json.JSONDecodeError as e:
        raise ToolError(f"mobench {verb} 输出非 JSON: {e}: {p.stdout[:200]!r}") from e
    # stats 等 verb 的 CLI JSON 顶层是数组；fastmcp 的 structured_content 只收 dict，包一层
    return {"results": r} if isinstance(r, list) else r


@mcp.tool
def env(config: str) -> dict:
    """环境快照：平台/温度/engine 版本/二进制与模型哈希（7 步协议 0–4 步证物；R9/R10）。"""
    return _run_cli("env", ["--config", config])


@mcp.tool
def run(config: str, expect_handoff: str | None = None,
        overrides: list[str] | None = None) -> dict:
    """执行基准（每次新跑，R2/R11）。

    expect_handoff：外部构件清单 JSON 文件路径（需包含 ``"binary_md5"`` 字段）→
    **前置校验（fail fast）**：先读清单文件，再取 env 快照比对实测二进制 MD5；
    不可读 / 无 binary_md5 / 不匹配一律在开跑前拒绝，避免无效实测；通过后才执行 run。
    overrides：CLI 参数原样附加（如 ["--threads", "4"]）。
    """
    out: dict = {}
    if expect_handoff:
        try:
            h = json.loads(Path(expect_handoff).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:   # 缺文件/不可读/非 JSON → 同样拒绝出判决
            raise ToolError(f"I6: 无法读取 handoff: {e}") from e
        want = h.get("binary_md5")
        if not want:
            raise ToolError("I6: handoff 未记录 binary_md5，无法锚定测量")
        snap = _run_cli("env", ["--config", config, "--no-save"])
        got = (snap.get("binary") or {}).get("md5")
        if got != want:
            raise ToolError(f"I6: 二进制哈希不匹配 handoff={want} measured={got}（测的不是那个改动）")
        out["handoff_check"] = {"binary_md5": got, "match": True}
    out["run"] = _run_cli("run", ["--config", config, *(overrides or [])])
    return out


@mcp.tool
def profile(config: str, extra: list[str] | None = None) -> dict:
    """剖面（`profile --config X [--model …] [--threads …] [--freq …] [--events …]`）。

    产 flamegraph/top/IPC 工件（host perf 或 adb simpleperf）。
    """
    return _run_cli("profile", ["--config", config, *(extra or [])])


@mcp.tool
def stats(inputs: list[str], extra: list[str] | None = None) -> dict:
    """统计（`stats [--cv-threshold X] inputs…`：只吃 records；extra 参数原样透传）。

    注意语义：**汇总**，不是对比——传入多组 records 会被合并统计（各组均值差大时
    合并 cv 会虚高）。对比不同实验组请**分组单独调用**本工具各取均值后再比较。
    """
    return _run_cli("stats", [*inputs, *(extra or [])])


@mcp.tool
def regression(action: str, extra: list[str] | None = None) -> dict:
    """回归检测（`regression {save,compare} [--results …] [--baseline …] …`）。

    action ∈ {"save", "compare"}；示例：
    extra=["--results", "results/records.jsonl", "--baseline", "results/baseline.json"]。
    """
    return _run_cli("regression", [action, *(extra or [])])


@mcp.tool
def validate(action: str, extra: list[str] | None = None) -> dict:
    """精度校验门禁（`validate {run,compare} …`）。

    action ∈ {"run", "compare"}；``compare`` 在 FAIL 时 CLI 返回退出码 2，
    本工具将其视为正常的结构化判定结果返回（包含 verdict/exit_code/details）。
    """
    if action not in ("run", "compare"):
        raise ToolError(f"validate action 必须为 'run' 或 'compare'，收到 {action!r}")
    allowed = (0, 2) if action == "compare" else (0,)
    return _run_cli("validate", [action, *(extra or [])],
                    allowed_exit_codes=allowed)


@mcp.tool
def power(config: str, source: str, extra: list[str] | None = None) -> dict:
    """功耗分析（`power --config X --source Y`；source 为功耗数据文件，必填）。"""
    return _run_cli("power", ["--config", config, "--source", source, *(extra or [])])


@mcp.tool
def report(inputs: list[str], out: str | None = None,
           extra: list[str] | None = None) -> dict:
    """出报告工件（`report [--out DIR] [--format …] inputs…`，**无 --json**）。

    返回 {"stdout": "Markdown report: <path>\\n…"}——工件路径即证物。
    """
    args = [*inputs, *(["--out", out] if out else []), *(extra or [])]
    return _run_cli("report", args, json_out=False)


@mcp.tool
def runs(results: str = "results") -> dict:
    """历史 run 索引（扫描 results 目录下全部 run.json）。

    每行 = 时间 / engine / 标签 / artifact md5 / 记录数 / audit 状态 / 路径；
    结果在返回的 results 数组里（空目录 → 错误）。
    """
    return _run_cli("runs", ["--results", results])


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
