"""Unit tests for synthetic camera and lidar data."""

import math

import pytest

from point_robot_ros.synthetic_camera_publisher import (
    create_test_pattern,
)
from point_robot_ros.synthetic_lidar_publisher import (
    create_test_ranges,
)


def test_camera_pattern_has_expected_byte_count():
    pattern = create_test_pattern(
        width=16,
        height=3,
        frame_index=0,
    )

    assert len(pattern) == 16 * 3


def test_camera_pattern_repeats_the_generated_row():
    pattern = create_test_pattern(
        width=16,
        height=3,
        frame_index=0,
    )

    first_row = pattern[0:16]
    second_row = pattern[16:32]
    third_row = pattern[32:48]

    assert first_row == second_row
    assert second_row == third_row


def test_camera_bright_stripe_moves_between_frames():
    first_frame = create_test_pattern(
        width=32,
        height=1,
        frame_index=0,
    )
    second_frame = create_test_pattern(
        width=32,
        height=1,
        frame_index=2,
    )

    assert first_frame[0] == 255
    assert second_frame[0] == 0

    assert first_frame[15] < 255
    assert second_frame[15] == 255


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (0, 1),
        (1, 0),
        (-1, 1),
    ],
)
def test_camera_rejects_invalid_dimensions(
    width: int,
    height: int,
):
    with pytest.raises(ValueError):
        create_test_pattern(
            width=width,
            height=height,
            frame_index=0,
        )


def test_lidar_ranges_match_virtual_obstacles():
    ranges = create_test_ranges(
        angle_min=-math.pi / 2.0,
        angle_increment=math.pi / 180.0,
        beam_count=181,
    )

    assert len(ranges) == 181

    assert ranges[0] == pytest.approx(4.0)
    assert ranges[90] == pytest.approx(1.5)
    assert ranges[130] == pytest.approx(2.0)


def test_lidar_rejects_too_few_beams():
    with pytest.raises(ValueError):
        create_test_ranges(
            angle_min=0.0,
            angle_increment=0.1,
            beam_count=1,
        )


def test_lidar_rejects_non_positive_angle_increment():
    with pytest.raises(ValueError):
        create_test_ranges(
            angle_min=0.0,
            angle_increment=0.0,
            beam_count=10,
        )
