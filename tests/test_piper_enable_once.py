"""模拟单元测试；不加载Piper SDK，不连接CAN。"""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/025_piper_safety_query/enable_once.py"
spec = importlib.util.spec_from_file_location("enable_once", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Clock:
    now = 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Fake:
    """只提供使能接口；误调用运动/复位/失能会使测试直接失败。"""
    calls = 0

    def EnableArm(self, motor):
        assert motor == 7
        self.calls += 1


@pytest.mark.parametrize("case,expected_calls", [
    ("success", 1), ("timeout", 1), ("stale", 1),
    ("fault", 0), ("teaching", 0), ("already_enabled", 0),
    ("missing", 0), ("interrupt", 1),
])
def test_enable_once(monkeypatch, case, expected_calls):
    clock, device, report = Clock(), Fake(), {}

    def sample(_):
        if case == "interrupt" and device.calls:
            raise KeyboardInterrupt
        stamp = 1 + clock.now
        if case == "stale" and device.calls:
            stamp = 1.15
        return {"mode": 2 if case == "teaching" else 0,
                "status": 0, "fault": case == "fault",
                "motor_ids": [0] * 6 if case == "missing" else list(range(1, 7)),
                "enabled": [case == "already_enabled" or
                            (case == "success" and device.calls > 0)] * 6,
                "arm_stamp": stamp, "driver_stamp": stamp}

    monkeypatch.setattr(module, "snapshot", sample)
    if case == "success":
        module.observe_once(device, report, clock=clock.time, sleep=clock.sleep)
        assert report["outcome"] == "all_enabled_reported"
    else:
        error = KeyboardInterrupt if case == "interrupt" else (RuntimeError, TimeoutError)
        with pytest.raises(error):
            module.observe_once(device, report, clock=clock.time, sleep=clock.sleep)
    assert device.calls == expected_calls
