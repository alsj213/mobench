"""CLI surface tests (benchmark-core 确定性命令行入口)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mobench import __version__
from mobench.cli import build_parser, main


def test_help_lists_all_subcommands(capsys):
    # Arrange
    parser = build_parser()

    # Act / Assert (argparse --help exits 0 after printing)
    with pytest.raises(SystemExit) as ei:
        parser.parse_args(["--help"])
    out = capsys.readouterr().out
    assert ei.value.code == 0
    for cmd in ("run", "profile", "memory", "trace", "env", "stats",
                "regression", "validate", "power", "report", "prepare", "runs"):
        assert cmd in out


def test_cli_version_flag(capsys):
    code = main(["--version"])
    assert code == 0
    out = capsys.readouterr().out.strip()
    assert out == f"mobench {__version__}"


def test_python_m_mobench_version_subprocess():
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "mobench", "--version"],
        capture_output=True, text=True, cwd=repo,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == f"mobench {__version__}"


def test_unknown_command_fails(capsys):
    # Act
    code = main(["bogus"])

    # Assert: argparse invalid-choice exits 2 (non-zero), message on stderr
    assert code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_env_prints_platform_fields(tmp_path, monkeypatch):
    # Arrange
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "mobench.yml"
    cfg.write_text(
        "engine: llamacpp\n"
        "platform:\n  type: host\n"
        "models:\n  - id: tinyllama\n    path: m.gguf\n    kind: llm\n",
        encoding="utf-8",
    )

    # Act
    code = main(["env", "--config", str(cfg)])

    # Assert
    assert code == 0
    results_dir = tmp_path / "results"
    snapshots = list(results_dir.glob("env_snapshot_*.json"))
    assert len(snapshots) == 1
    data = json.loads(snapshots[0].read_text(encoding="utf-8"))
    assert data["platform"]["value"] == "host"
    assert data["engine"]["value"] == "llamacpp"
