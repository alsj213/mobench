"""End-to-end run pipeline tests with a fake llama-bench (benchmark-core 执行流水线)."""
import json

import pytest

from mobench.cli import main as cli_main
from mobench.config import load_config
from mobench.platform import host as host_mod
from mobench.run import RunError, run_pipeline


def fake_llama_bench(path):
    """Write a fake llama-bench that emits llama-bench-style JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, json\n"
        "print(json.dumps({'build':'fake','results':[\n"
        "  {'n_prompt':64,'n_gen':128,'t_prompt_processing':3.2,'t_generation':21.0}]}))\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def setup_project(tmp_path):
    fake_llama_bench(tmp_path / "bin" / "llama-bench")
    (tmp_path / "models").mkdir(parents=True)
    model = tmp_path / "models" / "tinyllama.gguf"
    model.write_bytes(b"GGUF" * 64)
    cfg = tmp_path / "mobench.yml"
    cfg.write_text(
        "engine: llamacpp\n"
        "platform:\n  type: host\n"
        "build:\n  bin_dir: bin\n  bench_bin: llama-bench\n"
        "models:\n  - id: tinyllama\n    path: models/tinyllama.gguf\n"
        "    kind: llm\n    precision: q8_0\n"
        "workload:\n  threads: [4]\n  rounds: 2\n"
        "results_dir: results\n",
        encoding="utf-8",
    )
    return model, str(cfg)


def test_run_produces_traceable_records(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    _, cfg_path = setup_project(tmp_path)

    # Act
    meta = run_pipeline(load_config(cfg_path))

    # Assert: 2 rounds -> 3 metrics per round
    assert meta["n_records"] == 2 * 3
    recs = [json.loads(line) for line in open(meta["records"], encoding="utf-8")]
    assert all(rec["source_log"] and __import__("os").path.exists(rec["source_log"])
               for rec in recs)
    assert all(rec["engine"] == "llamacpp" for rec in recs)
    # raw logs written and non-empty
    raw_logs = list((tmp_path / "results").glob("run_*/raw_*.log"))
    assert len(raw_logs) == 2
    assert all(log.stat().st_size > 0 for log in raw_logs)


def test_run_cli_json_ok(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    _, cfg_path = setup_project(tmp_path)

    # Act
    code = cli_main(["run", "--config", cfg_path, "--json"])

    # Assert
    assert code == 0


def test_run_aborts_when_model_missing(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    fake_llama_bench(tmp_path / "bin" / "llama-bench")
    cfg = tmp_path / "mobench.yml"
    cfg.write_text(
        "engine: llamacpp\n"
        "build:\n  bin_dir: bin\n  bench_bin: llama-bench\n"
        "models:\n  - id: ghost\n    path: models/ghost.gguf\n    kind: llm\n",
        encoding="utf-8",
    )

    # Act / Assert
    with pytest.raises(RunError) as ei:
        run_pipeline(load_config(str(cfg)))
    assert "model" in str(ei.value)


# -- 测量条件接入流水线（add-cpu-pin-lockfreq） ---------------------------


def _fake_sysfs(root):
    """A 1+3+4 cluster layout, so the default档 selects cpu4-7 at 4 threads."""
    for name, cpus, freq in (("policy0", [0, 1, 2, 3], 1804800),
                             ("policy4", [4, 5, 6], 2419200),
                             ("policy7", [7], 2841600)):
        d = root / "cpufreq" / name
        d.mkdir(parents=True)
        (d / "related_cpus").write_text(
            " ".join(str(c) for c in cpus) + "\n", encoding="utf-8")
        (d / "cpuinfo_max_freq").write_text(f"{freq}\n", encoding="utf-8")
        (d / "scaling_governor").write_text("schedutil\n", encoding="utf-8")
    return root


def test_run_records_measurement_condition(tmp_path, monkeypatch):
    """控制状态进入 run.json 与每条 record 的 device 字段（R3 可溯源）。"""
    # Arrange
    monkeypatch.chdir(tmp_path)
    _, cfg_path = setup_project(tmp_path)
    monkeypatch.setattr(host_mod, "_SYSFS_CPU",
                        str(_fake_sysfs(tmp_path / "sysfs")))
    monkeypatch.setattr(host_mod.os, "geteuid", lambda: 1000)  # 无 root
    # 伪造的 8 核 sysfs 在 <8 核机器（如 CI 4 核 runner）上直接调真 taskset 会失败；
    # 在 PATH 前置 stub taskset，透传真实命令并保证跨机器可复现。
    stub_bin = tmp_path / "stub_bin"
    stub_bin.mkdir()
    taskset_stub = stub_bin / "taskset"
    taskset_stub.write_text(
        "#!/bin/sh\nif [ \"$1\" = \"-c\" ]; then shift 2; else shift 1; fi\nexec \"$@\"\n",
        encoding="utf-8",
    )
    taskset_stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(stub_bin), prepend=":")

    # Act
    meta = run_pipeline(load_config(cfg_path))

    # Assert
    cond = meta["measurement_conditions"]["t4"]
    assert cond["pin"]["cpus"] == [4, 5, 6, 7]
    assert cond["pin"]["mask"] == "f0"
    assert cond["freq"]["locked"] is False          # degraded, not fabricated
    recs = [json.loads(line) for line in open(meta["records"], encoding="utf-8")]
    assert recs
    assert all(r["device"]["pin"]["cpus"] == [4, 5, 6, 7] for r in recs)
    assert all(r["device"]["freq"]["locked"] is False for r in recs)


def test_run_degrades_without_cpufreq_without_aborting(tmp_path, monkeypatch):
    """无 cpufreq 环境（WSL2 等）默认档位降级不绑核，测量照常完成（D4）。"""
    # Arrange
    monkeypatch.chdir(tmp_path)
    _, cfg_path = setup_project(tmp_path)
    monkeypatch.setattr(host_mod, "_SYSFS_CPU", str(tmp_path / "no-sysfs"))
    monkeypatch.setattr(host_mod.os, "geteuid", lambda: 1000)

    # Act
    meta = run_pipeline(load_config(cfg_path))

    # Assert
    cond = meta["measurement_conditions"]["t4"]
    assert cond["pin"]["origin"] == "unavailable"
    assert cond["pin"]["cpus"] == []
    assert meta["n_records"] == 2 * 3


# -- cooldown 门与每轮档位记录（add-thermal-band-monitor） -------------------


class _FakeBandPlatform:
    """sample_thermal 序列喂料（cooldown 逻辑测试用）。"""

    def __init__(self, temps):
        self._temps = list(temps)

    def sample_thermal(self):
        t = self._temps.pop(0) if self._temps else 999.0
        return {"temp_c": {"value": t}, "cpu_freq": {"value_mhz": None}}


def test_cooldown_waits_until_below_target():
    from mobench.run import _cooldown_wait

    # Arrange: 52 → 47 → 41（阈值 45）
    platform = _FakeBandPlatform([52.0, 47.0, 41.0])
    slept = []

    # Act
    res = _cooldown_wait(platform, {"cooldown_c": 45, "cooldown_max_s": 600},
                         lambda *a: None, sleep=slept.append)

    # Assert
    assert res["reached"] is True
    assert res["waited_s"] == 60
    assert slept == [30, 30]
    assert res["temp_c"] == 41.0


def test_cooldown_gives_up_after_max_without_raising():
    from mobench.run import _cooldown_wait

    platform = _FakeBandPlatform([52.0] * 10)
    res = _cooldown_wait(platform, {"cooldown_c": 45, "cooldown_max_s": 60},
                         lambda *a: None, sleep=lambda s: None)
    assert res["reached"] is False
    assert res["waited_s"] == 60


def test_cooldown_noop_when_not_configured():
    from mobench.run import _cooldown_wait

    platform = _FakeBandPlatform([60.0])
    assert _cooldown_wait(platform, {}, lambda *a: None,
                          sleep=lambda s: None) is None


def test_run_records_per_round_band(tmp_path, monkeypatch):
    """每轮 device 段携带实时采样（temp_zone/cpu_freq_mhz 字段在位）。"""
    # Arrange
    monkeypatch.chdir(tmp_path)
    _, cfg_path = setup_project(tmp_path)

    # Act
    meta = run_pipeline(load_config(cfg_path))

    # Assert
    recs = [json.loads(line) for line in open(meta["records"], encoding="utf-8")]
    assert recs
    assert all("temp_zone" in r["device"] for r in recs)
    assert all("cpu_freq_mhz" in r["device"] for r in recs)
    assert all("temp_after_c" in r["device"] for r in recs)
    assert "temp_c" in meta["platform"]
    assert "cpu_freq" in meta["platform"]
    assert meta["platform"]["env_control"]["value"] != "env control not run"


def test_run_per_round_cooldown_runs_before_sample_and_retries_save_failed_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, cfg_path = setup_project(tmp_path)
    # 构造一个第 1 次失败、第 2 次成功的假 llama-bench
    marker = tmp_path / "attempt_count.txt"
    (tmp_path / "bin" / "llama-bench").write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "if '--version' in sys.argv:\n"
        "    print('fake 1.0')\n"
        "    sys.exit(0)\n"
        f"m = pathlib.Path({str(marker)!r})\n"
        "n = int(m.read_text()) + 1 if m.exists() else 1\n"
        "m.write_text(str(n))\n"
        "if n == 1:\n"
        "    print('pointer tag truncated')\n"
        "    sys.exit(134)\n"
        "print(json.dumps({'results': [{'n_prompt': 64, 'n_gen': 128, 't_prompt_processing': 3.2, 't_generation': 21.0}]}))\n",
        encoding="utf-8",
    )
    (tmp_path / "bin" / "llama-bench").chmod(0o755)

    # 温度序列：r1a1 降温 48->34(采样34) -> r1a2 降温 46->33(采样33) -> r1跑后49(>45) -> r2 降温 49->35(采样35) -> r2跑后36
    temps = [48.0, 34.0, 34.0, 46.0, 33.0, 33.0, 49.0, 49.0, 35.0, 35.0, 36.0]
    from mobench.platform.base import BasePlatform

    def fake_sample(self):
        t = temps.pop(0) if temps else 35.0
        return {"temp_c": {"value": t, "zone": "cpu"}, "cpu_freq": {"value_mhz": 2400.0}}

    monkeypatch.setattr(BasePlatform, "sample_thermal", fake_sample)
    monkeypatch.setattr("mobench.run.time.sleep", lambda _s: None)

    cfg = load_config(cfg_path)
    cfg["methodology"].update({
        "cooldown_c": 35.0,
        "cooldown_per_round": True,
        "max_attempts": 2,
    })
    meta = run_pipeline(cfg)
    assert meta["valid"] is True
    assert meta["throttle_risk"] is True
    assert len(meta["attempts"]) == 3
    # 每条成功轮次的起始温度都必须等于冷却后的温度（<=35），而不是冷却前的高温
    recs = [json.loads(line) for line in open(meta["records"], encoding="utf-8")]
    assert all(r["device"]["temp_c"] <= 35.0 for r in recs)
    failed_logs = list((tmp_path / "results").glob("run_*/raw_*.failed1"))
    assert len(failed_logs) == 1
    assert "pointer tag truncated" in failed_logs[0].read_text(encoding="utf-8")


def test_stats_regression_report_skip_invalid_runs_unless_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _, cfg_path = setup_project(tmp_path)
    from mobench.platform.base import BasePlatform

    monkeypatch.setattr(
        BasePlatform,
        "sample_thermal",
        lambda self: {"temp_c": {"value": 32.0, "zone": "cpu"}, "cpu_freq": {"value_mhz": 1500.0}},
    )
    cfg = load_config(cfg_path)
    cfg["methodology"]["freq_floor_mhz"] = 2400.0
    meta = run_pipeline(cfg)
    assert meta["valid"] is False
    assert meta["freq_audit"]["valid"] is False

    # 默认跳过无效 run -> stats 返回错误退出码 1
    assert cli_main(["stats", str(tmp_path / "results")]) == 1
    assert "skipping invalid run records" in capsys.readouterr().err

    # 显式 --include-invalid 后纳入分析
    assert cli_main(["stats", str(tmp_path / "results"), "--include-invalid", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows and rows[0]["n"] == 2

    assert cli_main(["report", str(tmp_path / "results"), "--include-invalid", "--out", str(tmp_path / "rpt")]) == 0
    assert (tmp_path / "rpt" / "benchmark_report.md").is_file()


# -- Wave 5 加固：中止落盘 / 冷却不可读 / CLI 覆盖复验 -----------------------


def _write_failing_round_binary(tmp_path):
    """第 1 轮成功、第 2 轮非零退出的假 llama-bench。"""
    marker = tmp_path / "round_count.txt"
    script = tmp_path / "bin" / "llama-bench"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        "if '--version' in sys.argv:\n"
        "    print('fake 1.0')\n"
        "    sys.exit(0)\n"
        f"m = pathlib.Path({str(marker)!r})\n"
        "n = int(m.read_text()) + 1 if m.exists() else 1\n"
        "m.write_text(str(n))\n"
        "if n >= 2:\n"
        "    sys.stderr.write('second round failed\\n')\n"
        "    sys.exit(1)\n"
        "print(json.dumps({'results': [{'n_prompt': 64, 'n_gen': 128, "
        "'t_prompt_processing': 3.2, 't_generation': 21.0}]}))\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


def test_run_second_round_failure_writes_aborted_run_json(tmp_path, monkeypatch):
    """第 2 轮 rc!=0 中止后 run.json 必须存在并标 valid=false/aborted=true。"""
    # Arrange
    monkeypatch.chdir(tmp_path)
    _, cfg_path = setup_project(tmp_path)
    _write_failing_round_binary(tmp_path)

    # Act
    with pytest.raises(RunError, match="engine exited 1"):
        run_pipeline(load_config(cfg_path))

    # Assert: run.json 落盘；残留 records.jsonl 默认被 read_paths 拦截
    from mobench.records import read_paths

    run_jsons = list((tmp_path / "results").glob("run_*/run.json"))
    assert len(run_jsons) == 1
    meta = json.loads(run_jsons[0].read_text(encoding="utf-8"))
    assert meta["valid"] is False
    assert meta["aborted"] is True
    assert "engine exited 1" in meta["error"]
    assert (run_jsons[0].parent / "records.jsonl").exists()
    assert read_paths([str(tmp_path / "results")]) == []


def test_cooldown_wait_temp_unavailable_never_reports_reached():
    from mobench.run import _cooldown_wait

    # Arrange: 温度不可读（None）——不得标 reached=True
    platform = _FakeBandPlatform([None])
    slept = []

    # Act
    res = _cooldown_wait(platform, {"cooldown_c": 45, "cooldown_max_s": 600},
                         lambda *a: None, sleep=slept.append)

    # Assert
    assert res["reached"] is False
    assert res["temp_c"] is None
    assert res["reason"] == "temperature unavailable"
    assert slept == []


def test_run_cli_rejects_cooldown_per_round_without_cooldown_c(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _, cfg_path = setup_project(tmp_path)

    code = cli_main(["run", "--config", cfg_path, "--cooldown-per-round"])

    assert code == 1
    assert "cooldown_per_round requires methodology.cooldown_c" in capsys.readouterr().err


def test_run_cli_rejects_negative_cooldown_c(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _, cfg_path = setup_project(tmp_path)

    code = cli_main(["run", "--config", cfg_path, "--cooldown-c", "-1"])

    assert code == 1
    assert "cooldown_c" in capsys.readouterr().err


def test_run_timeout_is_recorded_as_rc_124_and_retried(tmp_path, monkeypatch):
    """平台抛 TimeoutExpired -> rc=124 记入 attempts，并按 max_attempts 重试。"""
    import subprocess

    from mobench.platform.host import HostPlatform

    monkeypatch.chdir(tmp_path)
    _, cfg_path = setup_project(tmp_path)
    real_run = HostPlatform.run
    calls = {"n": 0}

    def flaky_run(self, cmd, log_path, cwd=None, echo=True):
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.TimeoutExpired(cmd, 1)
        return real_run(self, cmd, log_path, cwd=cwd, echo=echo)

    monkeypatch.setattr(HostPlatform, "run", flaky_run)
    cfg = load_config(cfg_path)
    cfg["methodology"]["max_attempts"] = 2

    meta = run_pipeline(cfg)

    assert meta["valid"] is True
    assert meta["attempts"][0]["rc"] == 124
    assert meta["attempts"][1]["rc"] == 0


def test_run_cli_forwards_cooldown_and_retry_overrides(tmp_path, monkeypatch):
    """CLI 覆盖必须进入最终 cfg（并被重新校验）。"""
    from mobench import run as run_mod

    monkeypatch.chdir(tmp_path)
    _, cfg_path = setup_project(tmp_path)
    seen = {}

    def fake_pipeline(cfg, **kwargs):
        seen["cfg"] = cfg
        return {"throttle_risk": False, "n_records": 0,
                "records": "x", "valid": True}

    monkeypatch.setattr(run_mod, "run_pipeline", fake_pipeline)

    code = cli_main([
        "run", "--config", cfg_path, "--json",
        "--cooldown-c", "40", "--cooldown-per-round", "--max-attempts", "3",
    ])

    assert code == 0
    meth = seen["cfg"]["methodology"]
    assert meth["cooldown_c"] == 40.0
    assert meth["cooldown_per_round"] is True
    assert meth["max_attempts"] == 3


def test_run_restore_failure_after_success_writes_invalid_run_json(tmp_path, monkeypatch):
    """主流程成功但 restore 抛错：收敛为 RunError 且工件标无效。"""
    from mobench.platform.host import HostPlatform

    monkeypatch.chdir(tmp_path)
    _, cfg_path = setup_project(tmp_path)

    def boom(self):
        raise RuntimeError("restore exploded")

    monkeypatch.setattr(HostPlatform, "restore", boom)

    with pytest.raises(RunError) as exc:
        run_pipeline(load_config(cfg_path))

    assert exc.value.step == "restore"
    assert "restore exploded" in str(exc.value)

    run_json = list((tmp_path / "results").glob("run_*/run.json"))[0]
    meta = json.loads(run_json.read_text(encoding="utf-8"))
    assert meta["valid"] is False
    assert meta["aborted"] is True
    assert "[step:restore]" in meta["error"]
    assert "restore exploded" in meta["error"]


def test_run_cli_restore_failure_exits_1_cleanly(tmp_path, monkeypatch, capsys):
    """CLI 下 restore 抛普通异常：干净报错并返回 1，不输出裸 traceback。"""
    from mobench.platform.host import HostPlatform

    monkeypatch.chdir(tmp_path)
    _, cfg_path = setup_project(tmp_path)

    def boom(self):
        raise RuntimeError("restore exploded")

    monkeypatch.setattr(HostPlatform, "restore", boom)

    code = cli_main(["run", "--config", cfg_path])

    assert code == 1
    err = capsys.readouterr().err
    assert "run: error:" in err
    assert "[step:restore]" in err
    assert "restore exploded" in err
    assert "Traceback" not in err

    run_json = list((tmp_path / "results").glob("run_*/run.json"))[0]
    meta = json.loads(run_json.read_text(encoding="utf-8"))
    assert meta["valid"] is False
    assert meta["aborted"] is True
    assert "[step:restore]" in meta["error"]


def test_restore_failure_does_not_mask_original_run_error(tmp_path, monkeypatch):
    """执行失败 + restore 失败：run.json 保留原始错误，并另记 restore_error。"""
    from mobench.platform.host import HostPlatform

    monkeypatch.chdir(tmp_path)
    _, cfg_path = setup_project(tmp_path)
    _write_failing_round_binary(tmp_path)

    def boom(self):
        raise RuntimeError("restore exploded")

    monkeypatch.setattr(HostPlatform, "restore", boom)

    with pytest.raises(RunError, match="engine exited 1"):
        run_pipeline(load_config(cfg_path))

    run_json = list((tmp_path / "results").glob("run_*/run.json"))[0]
    meta = json.loads(run_json.read_text(encoding="utf-8"))
    assert meta["valid"] is False
    assert meta["aborted"] is True
    assert "engine exited 1" in meta["error"]
    assert "restore exploded" in meta["restore_error"]


def test_run_meta_construction_failure_still_writes_aborted_run_json(tmp_path, monkeypatch):
    """run_dir 已创建后、执行前的元数据/哈希失败也必须留下 aborted run.json。"""
    from mobench import run as run_mod

    monkeypatch.chdir(tmp_path)
    _, cfg_path = setup_project(tmp_path)

    def boom(_binary):
        raise RuntimeError("hash boom")

    monkeypatch.setattr(run_mod, "artifact_set_md5", boom)

    with pytest.raises(RuntimeError, match="hash boom"):
        run_pipeline(load_config(cfg_path))

    run_jsons = list((tmp_path / "results").glob("run_*/run.json"))
    assert len(run_jsons) == 1
    meta = json.loads(run_jsons[0].read_text(encoding="utf-8"))
    assert meta["valid"] is False
    assert meta["aborted"] is True
    assert "hash boom" in meta["error"]


def test_run_pipeline_rejects_zero_max_attempts_before_creating_run_dir(tmp_path, monkeypatch):
    """直接调用 run_pipeline 时 max_attempts=0 必须在创建 run_dir 前被拒绝。"""
    monkeypatch.chdir(tmp_path)
    _, cfg_path = setup_project(tmp_path)
    cfg = load_config(cfg_path)
    cfg["methodology"]["max_attempts"] = 0

    with pytest.raises(RunError, match="max_attempts"):
        run_pipeline(cfg)

    results_dir = tmp_path / "results"
    run_dirs = list(results_dir.glob("run_*")) if results_dir.exists() else []
    assert run_dirs == []
