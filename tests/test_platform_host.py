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
    info = {}

    # Act
    platform.control(cfg, info)

    # Assert
    assert "disabled" in info["env_control"]["value"]


def test_control_degrades_without_sudo(monkeypatch):
    # Arrange
    platform = HostPlatform({})
    monkeypatch.setattr(host_mod.os, "geteuid", lambda: 1000)
    cfg = {"methodology": {"env_control": "auto"}}
    info = {}

    # Act
    platform.control(cfg, info)

    # Assert
    assert "degraded" in info["env_control"]["value"].lower()


def test_run_tees_output_to_log(tmp_path):
    # Arrange
    platform = HostPlatform({})
    log = tmp_path / "out.log"

    # Act
    rc = platform.run(["echo", "hello-from-bench"], str(log))

    # Assert
    assert rc == 0
    assert "hello-from-bench" in log.read_text(encoding="utf-8")


def test_temp_c_hwmon_takes_maximum(tmp_path, monkeypatch):
    tz = tmp_path / "tz" / "temp"
    tz.parent.mkdir(parents=True)
    tz.write_text("bad\n", encoding="utf-8")
    hw1 = tmp_path / "hw1" / "temp1_input"
    hw2 = tmp_path / "hw2" / "temp2_input"
    hw1.parent.mkdir(parents=True)
    hw2.parent.mkdir(parents=True)
    hw1.write_text("41000\n", encoding="utf-8")
    hw2.write_text("58500\n", encoding="utf-8")

    def fake_glob(pattern):
        if "thermal_zone" in pattern:
            return [str(tz)]
        return [str(hw1), str(hw2)]

    monkeypatch.setattr(host_mod, "glob", type("_G", (), {"glob": staticmethod(fake_glob)}))
    platform = HostPlatform({})
    res = platform._temp_c()
    assert res["value"] == pytest.approx(58.5)
    assert res["source"] == "hwmon max"


def test_host_run_applies_timeout_and_flushes_partial_stdout(tmp_path, monkeypatch):
    platform = HostPlatform({"methodology": {"timeout_s": 12}})
    log = tmp_path / "timeout.log"
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        raise host_mod.subprocess.TimeoutExpired(cmd, 12, output=b"partial line\n")

    monkeypatch.setattr(host_mod.subprocess, "run", fake_run)
    with pytest.raises(host_mod.subprocess.TimeoutExpired):
        platform.run(["echo", "hi"], str(log))
    assert seen["timeout"] == 12
    assert "partial line" in log.read_text(encoding="utf-8")
