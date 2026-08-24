"""Unit tests for differential-drive odometry."""

import math

import pytest

from point_robot_ros.odometry import (
    Pose2D,
    normalize_angle,
    update_odometry,
)
from point_robot_ros.odometry import DifferentialDriveOdometry
from point_robot_ros.wheel_encoder import WheelEncoderModel


def test_equal_wheel_travel_moves_straight():
    pose = update_odometry(
        pose=Pose2D(x=0.0, y=0.0, heading=0.0),
        left_distance=1.0,
        right_distance=1.0,
        wheel_base=0.5,
    )

    assert pose.x == pytest.approx(1.0)
    assert pose.y == pytest.approx(0.0)
    assert pose.heading == pytest.approx(0.0)


def test_opposite_wheel_travel_rotates_in_place():
    pose = update_odometry(
        pose=Pose2D(x=0.0, y=0.0, heading=0.0),
        left_distance=-0.25,
        right_distance=0.25,
        wheel_base=0.5,
    )

    assert pose.x == pytest.approx(0.0)
    assert pose.y == pytest.approx(0.0)
    assert pose.heading == pytest.approx(1.0)


def test_one_stationary_wheel_creates_an_arc():
    pose = update_odometry(
        pose=Pose2D(x=0.0, y=0.0, heading=0.0),
        left_distance=0.0,
        right_distance=0.5,
        wheel_base=0.5,
    )

    assert pose.x == pytest.approx(
        0.25 * math.sin(1.0)
    )
    assert pose.y == pytest.approx(
        0.25 * (1.0 - math.cos(1.0))
    )
    assert pose.heading == pytest.approx(1.0)


def test_heading_is_normalized():
    pose = update_odometry(
        pose=Pose2D(
            x=0.0,
            y=0.0,
            heading=math.pi - 0.1,
        ),
        left_distance=-0.05,
        right_distance=0.05,
        wheel_base=0.5,
    )

    assert -math.pi <= pose.heading < math.pi
    assert pose.heading == pytest.approx(
        normalize_angle(math.pi + 0.1)
    )


def make_odometry() -> DifferentialDriveOdometry:
    encoder = WheelEncoderModel(
        wheel_radius=1.0 / (2.0 * math.pi),
        ticks_per_revolution=100,
    )
    return DifferentialDriveOdometry(
        left_encoder=encoder,
        right_encoder=encoder,
        wheel_base=0.5,
    )


def test_first_encoder_sample_only_initializes_counts():
    odometry = make_odometry()

    pose = odometry.update(
        left_count=100,
        right_count=100,
    )

    assert pose == Pose2D(0.0, 0.0, 0.0)


def test_equal_encoder_increments_move_forward():
    odometry = make_odometry()
    odometry.update(left_count=0, right_count=0)

    pose = odometry.update(
        left_count=100,
        right_count=100,
    )

    assert pose.x == pytest.approx(1.0)
    assert pose.y == pytest.approx(0.0)
    assert pose.heading == pytest.approx(0.0)


def test_opposite_encoder_increments_rotate_robot():
    odometry = make_odometry()
    odometry.update(left_count=0, right_count=0)

    pose = odometry.update(
        left_count=-25,
        right_count=25,
    )

    assert pose.x == pytest.approx(0.0)
    assert pose.y == pytest.approx(0.0)
    assert pose.heading == pytest.approx(1.0)


def test_multiple_updates_accumulate_motion():
    odometry = make_odometry()
    odometry.update(left_count=0, right_count=0)
    odometry.update(left_count=50, right_count=50)

    pose = odometry.update(
        left_count=100,
        right_count=100,
    )

    assert pose.x == pytest.approx(1.0)
