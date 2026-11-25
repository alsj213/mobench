"""跑中频门禁（methodology.freq_floor_mhz）：跑后采样低于下限 → run 标记无效。

治 THERMAL-DIRTY 事故：5 连跑热降频（1862MHz）的数据直到跑完才被人工发现。
"""
import pytest

from mobench.config import load_config
from mobench.platform.base import BasePlatform
from mobench.run import run_pipeline


def fake_llama_bench(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, json\n"
        "print(json.dumps({'build':'fake','results':[\n"
        "  {'n_prompt':64,'n_gen':128,'t_prompt_processing':3.2,'t_generation':21.0}]}))\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def setup_project(tmp_path, freq_floor=None):
    fake_llama_bench(tmp_path / "bin" / "llama-bench")
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "models" / "tinyllama.gguf").write_bytes(b"GGUF" * 64)
    meth = (f"methodology:\n  freq_floor_mhz: {freq_floor}\n"
            if freq_floor is not None else "")
    cfg = tmp_path / "mobench.yml"
    cfg.write_text(
        "engine: llamacpp\n"
        "platform:\n  type: host\n"
        "build:\n  bin_dir: bin\n  bench_bin: llama-bench\n"
        "models:\n  - id: tinyllama\n    path: models/tinyllama.gguf\n"
        "    kind: llm\n    precision: q8_0\n"
        "workload:\n  threads: [4]\n  rounds: 2\n"
        + meth +
        "results_dir: results\n",
        encoding="utf-8",
    )
    return str(cfg)


def _patch_sample(monkeypatch, mhz):
    """BasePlatform.sample_thermal → 固定读数（host 继承链生效）。"""
    calls = []

    def fake(self):
        calls.append(mhz)
        return {"temp_c": {"value": 30.0, "zone": "t"},
                "cpu_freq": {"value_mhz": mhz}}

    monkeypatch.setattr(BasePlatform, "sample_thermal", fake)
    return calls


def test_freq_floor_violation_marks_invalid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = setup_project(tmp_path, freq_floor=2600)
    _patch_sample(monkeypatch, 1800.0)
    meta = run_pipeline(load_config(cfg_path))
    assert meta["freq_audit"]["valid"] is False
    assert len(meta["freq_audit"]["violations"]) == 2  # 2 轮，每轮跑后一条
    v = meta["freq_audit"]["violations"][0]
    assert v["freq_mhz"] == 1800.0
    assert v["floor_mhz"] == 2600
    assert v["round"] == 1 and v["model"] == "tinyllama" and v["threads"] == 4


def test_freq_floor_ok(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = setup_project(tmp_path, freq_floor=2600)
    _patch_sample(monkeypatch, 2746.0)
    meta = run_pipeline(load_config(cfg_path))
    assert meta["freq_audit"] == {"valid": True, "violations": []}


def test_freq_floor_absent_no_audit_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = setup_project(tmp_path)
    _patch_sample(monkeypatch, 1800.0)  # 即使低频，未声明也不审计
    meta = run_pipeline(load_config(cfg_path))
    assert "freq_audit" not in meta


def test_freq_floor_none_reading_skipped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = setup_project(tmp_path, freq_floor=2600)
    _patch_sample(monkeypatch, None)  # 采样不可得 → 无从判定（fail-open）
    meta = run_pipeline(load_config(cfg_path))
    assert meta["freq_audit"]["valid"] is True


def test_freq_post_sample_count(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = setup_project(tmp_path, freq_floor=2600)
    calls = _patch_sample(monkeypatch, 2746.0)
    run_pipeline(load_config(cfg_path))
    # 每轮：跑前 band 采样 + 跑后门禁采样各一次 → 2 轮 = 4 次
    assert len(calls) == 4
