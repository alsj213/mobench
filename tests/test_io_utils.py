"""io_utils 与 env --no-save 单测。"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from mobench import io_utils
from mobench.cli import main as cli_main


def test_make_run_dir_adds_sequence_suffix_on_collision(tmp_path, monkeypatch):
    class FixedDatetime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 28, 12, 0, 0)

    monkeypatch.setattr(io_utils.datetime, "datetime", FixedDatetime)
    ts1, d1 = io_utils.make_run_dir(tmp_path, "run")
    ts2, d2 = io_utils.make_run_dir(tmp_path, "run")
    ts3, d3 = io_utils.make_run_dir(tmp_path, "run")

    assert ts1 == "20261128_120000"
    assert ts2 == "20261128_120000_01"
    assert ts3 == "20261128_120000_02"
    assert Path(d1).is_dir() and Path(d2).is_dir() and Path(d3).is_dir()


def test_atomic_write_json_cleans_temp_on_failure(tmp_path):
    target = tmp_path / "out.json"
    target.write_text('{"old": 1}', encoding="utf-8")

    # set 对象触发 json.dump TypeError，不应损坏原文件也不应遗留 .tmp 文件
    with pytest.raises(TypeError):
        io_utils.atomic_write_json(target, {"bad": {1, 2}})

    assert json.loads(target.read_text(encoding="utf-8")) == {"old": 1}
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []


def test_env_cli_no_save_skips_snapshot_file(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    bench = bin_dir / "llama-bench"
    bench.write_text("#!/bin/sh\necho v1\n", encoding="utf-8")
    bench.chmod(0o755)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    cfg = tmp_path / "mobench.yml"
    cfg.write_text(
        f"engine: llamacpp\n"
        f"build:\n  bin_dir: {bin_dir}\n  bench_bin: llama-bench\n"
        f"models:\n  - id: m\n    path: {model}\n"
        f"results_dir: {tmp_path / 'results'}\n",
        encoding="utf-8",
    )
    assert cli_main(["env", "--config", str(cfg), "--no-save", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "mobench/env@1"
    assert not list((tmp_path / "results").glob("env_snapshot_*.json"))
