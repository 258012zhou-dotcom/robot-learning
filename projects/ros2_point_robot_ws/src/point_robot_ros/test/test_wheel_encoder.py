"""Unit tests for wheel encoder conversion."""

import math

import pytest

from point_robot_ros.wheel_encoder import WheelEncoderModel


def test_one_revolution_matches_wheel_circumference():
    encoder = WheelEncoderModel(
        wheel_radius=0.1,
        ticks_per_revolution=1000,
    )

    distance = encoder.ticks_to_distance(1000)

    assert distance == pytest.approx(2.0 * math.pi * 0.1)


def test_half_revolution_matches_pi_radians():
    encoder = WheelEncoderModel(
        wheel_radius=0.1,
        ticks_per_revolution=1000,
    )

    angle = encoder.ticks_to_angle(500)

    assert angle == pytest.approx(math.pi)


def test_negative_ticks_produce_negative_distance():
    encoder = WheelEncoderModel(
        wheel_radius=0.1,
        ticks_per_revolution=1000,
    )

    distance = encoder.ticks_to_distance(-250)

    assert distance < 0.0


def test_cumulative_counts_are_converted_to_increment():
    encoder = WheelEncoderModel(
        wheel_radius=0.1,
        ticks_per_revolution=1000,
    )

    distance = encoder.count_change_to_distance(
        previous_count=1200,
        current_count=1450,
    )

    assert distance == pytest.approx(
        encoder.ticks_to_distance(250)
    )
