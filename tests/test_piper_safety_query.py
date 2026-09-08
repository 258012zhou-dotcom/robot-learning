"""离线验证回复完整性、协议单位及异常清理，不导入硬件 SDK。"""

from types import SimpleNamespace as NS

import pytest

from robot_learning.piper_safety_query import decode_limits, query_parameters, describe_firmware


def replies():
    angles = [None] + [NS(motor_num=i, min_angle_limit=-900,
                         max_angle_limit=900, max_joint_spd=500) for i in range(1, 7)]
    acc = [None] + [NS(joint_motor_num=i, max_joint_acc=1000) for i in range(1, 7)]
    return angles, acc


def test_protocol_units():
    result = decode_limits(*replies())
    assert len(result) == 6
    assert result[0] == {"joint": 1, "minimum_angle_deg": -90.0,
                         "maximum_angle_deg": 90.0, "maximum_speed_rad_s": 0.5,
                         "maximum_acceleration_rad_s2": 1.0}


def test_missing_joint_is_not_a_zero_limit():
    angles, acc = replies()
    angles[6].motor_num = 0
    with pytest.raises(ValueError, match="joint 6"):
        decode_limits(angles, acc)


def test_query_error_still_disconnects():
    class BrokenDevice:
        disconnected = False

        def ConnectPort(self, *, piper_init):
            assert piper_init is False

        def SearchAllMotorMaxAngleSpd(self):
            raise RuntimeError("simulated query error")

        def DisconnectPort(self):
            self.disconnected = True

    device = BrokenDevice()
    with pytest.raises(RuntimeError, match="simulated query error"):
        query_parameters(device)
    assert device.disconnected


@pytest.mark.parametrize("raw", ["S-V1\x00\x00\x00\x00", "S-V1", -1199])
def test_incomplete_version_is_preserved_but_not_accepted(raw):
    result = describe_firmware(raw)
    assert result["firmware_raw"] == raw
    assert result["firmware_version"] is None
    assert result["firmware_status"] == "unconfirmed"


def test_version_format_does_not_claim_hardware_verification():
    result = describe_firmware("S-V1.7-3")
    assert result["firmware_status"] == "format_valid_unverified"
    assert result["firmware_version"] == "S-V1.7-3"


def test_incomplete_firmware_keeps_limits_after_wait(monkeypatch):
    from robot_learning import piper_safety_query as module
    clock = iter([0.0, 0.1, 4.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)

    class Device:
        disconnected = False
        def ConnectPort(self, *, piper_init):
            assert piper_init is False
        def SearchAllMotorMaxAngleSpd(self):
            pass
        def SearchAllMotorMaxAccLimit(self):
            pass
        def SearchPiperFirmwareVersion(self):
            pass
        def GetAllMotorAngleLimitMaxSpd(self):
            return NS(all_motor_angle_limit_max_spd=NS(motor=replies()[0]))
        def GetAllMotorMaxAccLimit(self):
            return NS(all_motor_max_acc_limit=NS(motor=replies()[1]))
        def GetPiperFirmwareVersion(self):
            return "S-V1\x00\x00\x00\x00"
        def DisconnectPort(self):
            self.disconnected = True

    device = Device()
    result = query_parameters(device)
    assert result["query_status"] == "partial"
    assert result["firmware_version"] is None
    assert len(result["joints"]) == 6
    assert device.disconnected
