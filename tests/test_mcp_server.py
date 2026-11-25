import subprocess
import sys
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from mobench import mcp_server
from mobench.mcp_server import mcp


@pytest.fixture
def fake_cli(monkeypatch):
    """假 CLI：记录调用 (verb, args)，按 verb 返回 canned JSON。"""
    calls = []
    replies = {}

    def fake(verb, args, **kw):
        calls.append((verb, list(args)))
        out = replies.get(verb, {"ok": True})
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(mcp_server, "_run_cli", fake)
    return calls, replies


@pytest.fixture
def capture_cmd(monkeypatch):
    """捕获 _run_cli 真正拼出的 argv（不执行）。"""
    cmds = []

    def fake(cmd, **kw):
        cmds.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    monkeypatch.setattr(mcp_server.subprocess, "run", fake)
    return cmds


async def test_env_and_run_passthrough(fake_cli):
    calls, replies = fake_cli
    replies["env"] = {"binary": {"path": "/b", "md5": "m1"}, "engine_version": "abc"}
    replies["run"] = {"records": "/r/records.jsonl", "n_records": 4}
    async with Client(mcp) as c:
        env = (await c.call_tool("env", {"config": "m.yml"})).data
        run = (await c.call_tool("run", {"config": "m.yml"})).data
    assert env["engine_version"] == "abc"
    assert run["run"]["n_records"] == 4
    assert calls == [("env", ["--config", "m.yml"]),
                     ("run", ["--config", "m.yml"])]     # 每次都是新调用（无缓存可读旧数据）


async def test_run_handoff_anchor_ok(fake_cli, tmp_path):
    calls, replies = fake_cli
    replies["run"] = {"n_records": 4}
    replies["env"] = {"binary": {"path": "/b", "md5": "same"}}
    h = tmp_path / "handoff.json"
    h.write_text('{"binary_md5": "same"}')
    async with Client(mcp) as c:
        out = (await c.call_tool("run", {"config": "m.yml",
                                         "expect_handoff": str(h)})).data
    assert out["handoff_check"]["match"] is True
    assert [v for v, *_ in calls] == ["env", "run"]     # 前置校验通过后才 run


async def test_handoff_consumes_real_engopt_shape(fake_cli, tmp_path):
    """跨仓接缝：engopt Handoff 的 model_dump_json 真实形状必须能被锚定消费。"""
    calls, replies = fake_cli
    replies["run"] = {"n_records": 4}
    replies["env"] = {"binary": {"path": "/b", "md5": "d41d8cd98f00b204e9800998ecf8427e"}}
    h = tmp_path / "handoff.json"
    h.write_text("""{
  "round": 1,
  "engine": "llamacpp",
  "repo_hash": "ffffffffffffffffffffffffffffffffffffffff",
  "dirty": false,
  "build_recipe": {"build_cmd": "cmake --build build -j", "smoke_cmd": "./build/bin/llama-cli --version"},
  "binary_path": "/opt/llama.cpp/build/bin/llama-cli",
  "binary_md5": "d41d8cd98f00b204e9800998ecf8427e",
  "expected_delta": "+10% decode tok/s",
  "mb_config": "configs/local/t4.yml",
  "created_at": "2026-10-05T15:14:46+00:00"
}""")
    async with Client(mcp) as c:
        out = (await c.call_tool("run", {"config": "m.yml",
                                         "expect_handoff": str(h)})).data
    assert out["handoff_check"]["match"] is True
    assert [v for v, *_ in calls] == ["env", "run"]


async def test_run_handoff_mismatch_rejected(fake_cli, tmp_path):
    calls, replies = fake_cli
    replies["run"] = {"n_records": 4}
    replies["env"] = {"binary": {"path": "/b", "md5": "changed"}}
    h = tmp_path / "handoff.json"
    h.write_text('{"binary_md5": "original"}')
    async with Client(mcp) as c:
        with pytest.raises(ToolError, match="I6"):
            await c.call_tool("run", {"config": "m.yml", "expect_handoff": str(h)})
    assert [v for v, *_ in calls] == ["env"]            # run 未被执行：不白烧一次实测


async def test_run_handoff_without_md5_rejected(fake_cli, tmp_path):
    calls, replies = fake_cli
    replies["run"] = {"n_records": 4}
    h = tmp_path / "handoff.json"
    h.write_text("{}")
    async with Client(mcp) as c:
        with pytest.raises(ToolError, match="I6"):
            await c.call_tool("run", {"config": "m.yml", "expect_handoff": str(h)})
    assert calls == []                                  # 连 env 都没跑：开跑前即拒


async def test_run_handoff_file_missing_rejected(fake_cli, tmp_path):
    calls, replies = fake_cli
    replies["run"] = {"n_records": 4}
    missing = tmp_path / "nope.json"
    async with Client(mcp) as c:
        with pytest.raises(ToolError, match="I6"):
            await c.call_tool("run", {"config": "m.yml", "expect_handoff": str(missing)})
    assert calls == []


async def test_run_handoff_invalid_json_rejected(fake_cli, tmp_path):
    calls, replies = fake_cli
    replies["run"] = {"n_records": 4}
    h = tmp_path / "handoff.json"
    h.write_text("{not json")
    async with Client(mcp) as c:
        with pytest.raises(ToolError, match="I6"):
            await c.call_tool("run", {"config": "m.yml", "expect_handoff": str(h)})
    assert calls == []


def test_run_cli_failure_not_swallowed(monkeypatch):
    def bad(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom: model missing")
    monkeypatch.setattr(mcp_server.subprocess, "run", bad)
    with pytest.raises(ToolError, match="boom"):     # R4 精神：失败原样上报
        mcp_server._run_cli("run", ["--config", "m.yml"])


def test_run_cli_argv_contract(capture_cmd):
    """argv 契约钉死：verb 后接 args，--json 统一追加末尾。"""
    assert mcp_server._run_cli("run", ["--config", "m.yml", "--threads", "4"]) == {}
    assert capture_cmd[0] == [sys.executable, "-m", "mobench", "run",
                              "--config", "m.yml", "--threads", "4", "--json"]


def test_run_cli_wraps_list_output(monkeypatch):
    """stats 等 verb 的 CLI JSON 顶层是数组：须包 {"results": …}——
    fastmcp 的 structured_content 只收 dict，裸 list 会让 tools/call 直接报错。
    dict 输出（env/run 等）保持原样透传。"""
    monkeypatch.setattr(mcp_server.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, stdout='[{"metric": "gen_tok_s", "mean": 29.16}]', stderr=""))
    assert mcp_server._run_cli("stats", ["r.jsonl"]) == {
        "results": [{"metric": "gen_tok_s", "mean": 29.16}]}

    monkeypatch.setattr(mcp_server.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, stdout='{"ok": true}', stderr=""))
    assert mcp_server._run_cli("env", ["--config", "m.yml"]) == {"ok": True}


def test_argv_stats(capture_cmd):
    mcp_server.stats(["r1.jsonl", "r2.jsonl"], ["--cv-threshold", "3"])
    assert capture_cmd[0] == [sys.executable, "-m", "mobench", "stats",
                              "r1.jsonl", "r2.jsonl", "--cv-threshold", "3", "--json"]


def test_argv_regression(capture_cmd):
    mcp_server.regression("compare", ["--results", "r.jsonl", "--baseline", "b.json"])
    assert capture_cmd[0] == [sys.executable, "-m", "mobench", "regression", "compare",
                              "--results", "r.jsonl", "--baseline", "b.json", "--json"]


def test_argv_power(capture_cmd):
    mcp_server.power("m.yml", "power.txt")
    assert capture_cmd[0] == [sys.executable, "-m", "mobench", "power",
                              "--config", "m.yml", "--source", "power.txt", "--json"]


def test_argv_runs(capture_cmd):
    mcp_server.runs("results/k8-upstream")
    assert capture_cmd[0] == [sys.executable, "-m", "mobench", "runs",
                              "--results", "results/k8-upstream", "--json"]


async def test_runs_tool_passthrough(fake_cli):
    calls, replies = fake_cli
    replies["runs"] = {"results": [{"label": "k8-b-neon", "run_dir": "results/x"}]}
    async with Client(mcp) as c:
        r = (await c.call_tool("runs", {"results": "results/k8-upstream"})).data
    assert calls == [("runs", ["--results", "results/k8-upstream"])]
    assert r["results"][0]["label"] == "k8-b-neon"


async def test_runs_tool_end_to_end(tmp_path):
    import json as _json
    run_dir = tmp_path / "run_20260101_000001"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(_json.dumps(
        {"timestamp": "20260101_000001", "engine": "llamacpp", "n_records": 3,
         "binary": {"md5": "abcdef12"}, "models": [{"id": "m"}], "label": "e2e"}),
        encoding="utf-8")
    async with Client(mcp) as c:
        r = (await c.call_tool("runs", {"results": str(tmp_path)})).data
    assert r["results"][0]["label"] == "e2e"
    assert r["results"][0]["binary_md5"] == "abcdef12"


def test_argv_report(capture_cmd):
    mcp_server.report(["r.jsonl"], out="outdir")
    assert capture_cmd[0] == [sys.executable, "-m", "mobench", "report",
                              "r.jsonl", "--out", "outdir"]   # report 真身无 --json


def test_argv_profile(capture_cmd):
    """§4.1 后补项：profile 剖面工具（归因输入）。"""
    mcp_server.profile("m.yml", ["--threads", "4", "--freq", "4000"])
    assert capture_cmd[0] == [sys.executable, "-m", "mobench", "profile",
                              "--config", "m.yml", "--threads", "4", "--freq", "4000",
                              "--json"]


def test_real_cli_accepts_generated_argv(tmp_path):
    """真实 CLI 语法验收：工具拼出的每个 argv 真跑，stderr 不得含 unrecognized arguments。

    这是 mock 测试的盲区所在（C1 翻车点）——退出码可非零（垃圾输入本就该失败），
    但参数形状必须被真身 parser 接受。
    """
    junk_yaml = tmp_path / "mobench.yml"
    junk_records = tmp_path / "records.jsonl"
    junk_power = tmp_path / "power.txt"
    for p in (junk_yaml, junk_records, junk_power):
        p.write_text("junk\n")

    cmds = []
    with pytest.MonkeyPatch.context() as mp:      # 只在块内捕获；出块后打真 CLI
        def fake(cmd, **kw):
            cmds.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

        mp.setattr(mcp_server.subprocess, "run", fake)
        mcp_server.env(str(junk_yaml))
        mcp_server.run(str(junk_yaml))
        mcp_server.stats([str(junk_records)])
        mcp_server.regression("compare", ["--results", str(junk_records),
                                          "--baseline", str(tmp_path / "baseline.json")])
        mcp_server.power(str(junk_yaml), str(junk_power))
        mcp_server.report([str(junk_records)])
        mcp_server.profile(str(junk_yaml))

    assert len(cmds) == 7
    repo = Path(__file__).resolve().parents[1]
    for cmd in cmds:
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=repo)
        assert "unrecognized arguments" not in p.stderr, (cmd, p.returncode, p.stderr)


def test_cli_plumbing_alive():
    """真实 subprocess 冒烟：python -m mobench --help 可执行（rc=0）。"""
    p = subprocess.run([sys.executable, "-m", "mobench", "--help"],
                       capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[1])
    assert p.returncode == 0 and "mobench" in p.stdout
