"""Host platform tests (spec: benchmark-core 平台抽象 & methodology 环境控制降级)."""
import pytest

from mobench.platform import host as host_mod
from mobench.platform.base import BasePlatform, PlatformError
from mobench.platform.host import HostPlatform


def test_factory_returns_host_by_default():
    # Arrange
    cfg = {"platform": {"type": "host"}}

    # Act
    platform = BasePlatform.make(cfg)

    # Assert
    assert isinstance(platform, HostPlatform)


def test_preflight_returns_traceable_fields():
    # Arrange
    platform = HostPlatform({})

    # Act
    info = platform.preflight()

    # Assert
    assert info["platform"]["value"] == "host"
    assert info["os"]["source"].startswith("uname")
    assert "temp_c" in info


def test_control_disabled_when_env_control_none():
    # Arrange
    platform = HostPlatform({})
    cfg = {"methodology": {"env_control": "none"}}

    # Act
    platform.control(cfg, {})

    # Assert
    assert "disabled" in platform._env_control_label


def test_control_degrades_without_sudo(monkeypatch):
    # Arrange
    platform = HostPlatform({})
    monkeypatch.setattr(host_mod.os, "geteuid", lambda: 1000)
    cfg = {"methodology": {"env_control": "auto"}}

    # Act
    platform.control(cfg, {})

    # Assert
    assert "degraded" in platform._env_control_label.lower()


def test_run_tees_output_to_log(tmp_path):
    # Arrange
    platform = HostPlatform({})
    log = tmp_path / "out.log"

    # Act
    rc = platform.run(["echo", "hello-from-bench"], str(log))

    # Assert
    assert rc == 0
    assert "hello-from-bench" in log.read_text(encoding="utf-8")
