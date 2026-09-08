"""PIPER 参数查询：发送查询请求，检查完整回复，再转换单位。"""

import time
import re


def describe_firmware(raw: object) -> dict:
    """格式检查不是固件认证；不删除空字符来猜测版本。"""
    matches = isinstance(raw, str) and re.fullmatch(r"S-V[0-9]+\.[0-9]+-[0-9]+", raw) is not None
    return {
        "firmware_raw": raw,
        "firmware_version": raw if matches else None,
        "firmware_status": "format_valid_unverified" if matches else "unconfirmed",
    }


def decode_limits(angles, accelerations) -> list[dict]:
    """SDK 数组的下标 1–6 对应关节；下标 0 是占位。"""
    joints = []
    for index in range(1, 7):
        angle = angles[index]
        acc = accelerations[index]
        if angle.motor_num != index or acc.joint_motor_num != index:
            raise ValueError(f"joint {index}: response missing")
        joints.append({
            "joint": index,
            "minimum_angle_deg": angle.min_angle_limit * 0.1,
            "maximum_angle_deg": angle.max_angle_limit * 0.1,
            "maximum_speed_rad_s": angle.max_joint_spd * 0.001,
            "maximum_acceleration_rad_s2": acc.max_joint_acc * 0.001,
        })
    return joints


def query_parameters(piper, timeout_seconds: float = 3.0) -> dict:
    """piper 由入口传入，测试可用假设备代替真机。"""
    joints = None
    firmware = None
    try:
        # 禁用自动初始化，明确列出本实验发送的三个查询接口。
        piper.ConnectPort(piper_init=False)
        piper.SearchAllMotorMaxAngleSpd()
        piper.SearchAllMotorMaxAccLimit()
        piper.SearchPiperFirmwareVersion()
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            angles = piper.GetAllMotorAngleLimitMaxSpd()
            accelerations = piper.GetAllMotorMaxAccLimit()
            firmware = piper.GetPiperFirmwareVersion()
            try:
                joints = decode_limits(
                    angles.all_motor_angle_limit_max_spd.motor,
                    accelerations.all_motor_max_acc_limit.motor,
                )
            except ValueError:
                time.sleep(0.05)
                continue
            details = describe_firmware(firmware)
            if details["firmware_version"] is not None:
                return {**details, "query_status": "complete", "joints": joints}
            time.sleep(0.05)
        if joints is not None:
            # 六关节已完整回复，但版本异常：保留部分结果与原始字符串。
            return {**describe_firmware(firmware), "query_status": "partial",
                    "warning": "固件版本未确认，不能据此选择控制协议", "joints": joints}
        raise TimeoutError("未收到完整参数回复；请检查连接，不要把默认零值当作限位")
    finally:
        # 关闭连接不是停止运动指令；此程序从未启动运动。
        piper.DisconnectPort()
