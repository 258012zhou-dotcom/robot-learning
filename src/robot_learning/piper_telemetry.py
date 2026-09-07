"""Piper SDK 反馈值的最小单位转换。"""

import math
from numbers import Real
from typing import Sequence


PIPER_JOINT_COUNT = 6


def joint_raw_to_degrees(values: Sequence[Real]) -> tuple[float, ...]:
    """将六个关节的 SDK 原始值从 0.001 度转换为度。"""
    if len(values) != PIPER_JOINT_COUNT:
        raise ValueError("joint feedback must contain 6 values")

    degrees: list[float] = []
    for value in values:
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError("joint feedback must contain finite values")
        degrees.append(numeric_value * 0.001)
    return tuple(degrees)


def joint_raw_to_radians(values: Sequence[Real]) -> tuple[float, ...]:
    """将六个关节的 SDK 原始值转换为弧度。"""
    degrees = joint_raw_to_degrees(values)
    return tuple(math.radians(value) for value in degrees)


def gripper_raw_to_meters(value: Real) -> float:
    """将夹爪位置从 0.001 mm 转换为米。"""
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError("gripper feedback must be finite")
    return numeric_value * 1e-6
