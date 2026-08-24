"""Unit tests for the point robot camera kinematics."""

import math

import pytest

from point_robot_ros.kinematics import camera_forward_point
from point_robot_ros.kinematics import camera_yaw_to_target


def test_forward_point_at_zero_yaw():
    """A zero yaw should point along the positive x axis."""
    point = camera_forward_point(yaw=0.0, distance=1.0)

    assert point == pytest.approx((1.25, 0.0, 0.28))


def test_forward_point_at_positive_ninety_degrees():
    """A positive 90 degree yaw should point along positive y."""
    point = camera_forward_point(
        yaw=math.pi / 2.0,
        distance=1.0,
    )

    assert point == pytest.approx((0.25, 1.0, 0.28))


def test_inverse_kinematics_at_positive_ninety_degrees():
    """A target on positive y should require a positive 90 degree yaw."""
    yaw = camera_yaw_to_target(
        target_x=0.25,
        target_y=1.0,
    )

    assert yaw == pytest.approx(math.pi / 2.0)


def test_forward_inverse_round_trip():
    """Inverse kinematics should recover a reachable input yaw."""
    expected_yaw = -0.4
    point = camera_forward_point(
        yaw=expected_yaw,
        distance=2.0,
    )

    actual_yaw = camera_yaw_to_target(
        target_x=point[0],
        target_y=point[1],
    )

    assert actual_yaw == pytest.approx(expected_yaw)


def test_inverse_kinematics_rejects_target_behind_camera():
    """A target behind the mount should exceed the joint limits."""
    with pytest.raises(ValueError, match="joint limits"):
        camera_yaw_to_target(
            target_x=-1.0,
            target_y=0.0,
        )


def test_forward_kinematics_rejects_negative_distance():
    """A negative distance should be rejected."""
    with pytest.raises(ValueError, match="distance"):
        camera_forward_point(
            yaw=0.0,
            distance=-1.0,
        )
