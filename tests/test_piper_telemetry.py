import math

import pytest

from robot_learning.piper_telemetry import (
    gripper_raw_to_meters,
    joint_raw_to_degrees,
    joint_raw_to_radians,
)


def test_joint_raw_values_are_converted_to_degrees():
    raw_values = (90000, -90000, 180000, 0, 45000, -45000)

    assert joint_raw_to_degrees(raw_values) == (
        90.0,
        -90.0,
        180.0,
        0.0,
        45.0,
        -45.0,
    )


def test_joint_raw_values_are_converted_to_radians():
    radians = joint_raw_to_radians((90000, 0, 0, 0, 0, 0))

    assert radians[0] == pytest.approx(math.pi / 2)


def test_gripper_raw_value_is_converted_to_meters():
    assert gripper_raw_to_meters(80000) == pytest.approx(0.08)


def test_joint_conversion_rejects_wrong_joint_count():
    with pytest.raises(ValueError, match="6 values"):
        joint_raw_to_degrees((0, 0, 0))
