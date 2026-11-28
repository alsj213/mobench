"""共享 measurement_session 生命周期的 restore 异常收敛测试。"""
import pytest

from mobench.platform.base import PlatformError, measurement_session


class FakePlatform:
    """最小平台桩：只记录 control/restore 是否被调用。"""

    def __init__(self, restore_error=None):
        self.restore_error = restore_error
        self.restored = False

    def control(self, cfg, env):
        pass

    def measure_condition(self, threads):
        return {"pin": {"cpus": [], "mask": None, "origin": "disabled"},
                "freq": {"locked": False}}

    def restore(self):
        self.restored = True
        if self.restore_error is not None:
            raise self.restore_error


class BodyError(Exception):
    """模拟测量主体抛出的业务异常。"""


def test_measurement_session_wraps_restore_failure_when_body_succeeds(capsys):
    """主体成功但 restore 抛普通异常：统一收敛为 PlatformError。"""
    platform = FakePlatform(restore_error=RuntimeError("restore exploded"))

    with pytest.raises(PlatformError, match="restore failed: restore exploded") as exc:
        with measurement_session(platform, {}, 4):
            pass

    assert platform.restored is True
    assert isinstance(exc.value.__cause__, RuntimeError)
    # 主体没有失败，不应打印「保留原始错误」的告警。
    assert "warning:" not in capsys.readouterr().err


def test_measurement_session_keeps_body_error_and_warns(capsys):
    """主体已失败且 restore 也失败：保留主体异常并向 stderr 告警。"""
    platform = FakePlatform(restore_error=RuntimeError("restore exploded"))

    with pytest.raises(BodyError, match="body exploded"):
        with measurement_session(platform, {}, 4):
            raise BodyError("body exploded")

    assert platform.restored is True
    err = capsys.readouterr().err
    assert "warning:" in err
    assert "restore exploded" in err
    assert "body exploded" in err
