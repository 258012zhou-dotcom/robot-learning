import numpy as np
import pytest

from robot_learning.transforms_2d import (
    RigidTransform2D,
    rotation_matrix,
)


def test_rotation_matrix_rotates_counterclockwise() -> None:
    rotated = rotation_matrix(np.pi / 2) @ np.array([1.0, 0.0])

    np.testing.assert_allclose(rotated, [0.0, 1.0], atol=1e-12)


def test_transform_applies_rotation_then_translation() -> None:
    transform = RigidTransform2D(
        angle=np.pi / 2,
        translation=(3.0, 2.0),
    )

    transformed = transform.apply_to_point([2.0, 1.0])

    np.testing.assert_allclose(transformed, [2.0, 4.0], atol=1e-12)


def test_translation_does_not_change_direction_vector() -> None:
    transform = RigidTransform2D(
        angle=np.pi / 2,
        translation=(100.0, -50.0),
    )

    transformed = transform.apply_to_vector([1.0, 0.0])

    np.testing.assert_allclose(transformed, [0.0, 1.0], atol=1e-12)


def test_inverse_recovers_original_point() -> None:
    transform = RigidTransform2D(
        angle=0.7,
        translation=(2.0, -3.0),
    )
    point = np.array([1.5, 4.0])

    recovered = transform.inverse().apply_to_point(
        transform.apply_to_point(point)
    )

    np.testing.assert_allclose(recovered, point, atol=1e-12)


def test_composition_matches_sequential_application() -> None:
    world_from_robot = RigidTransform2D(
        angle=0.4,
        translation=(3.0, 1.0),
    )
    robot_from_sensor = RigidTransform2D(
        angle=-0.2,
        translation=(0.5, 0.25),
    )
    sensor_point = np.array([2.0, -1.0])

    sequential = world_from_robot.apply_to_point(
        robot_from_sensor.apply_to_point(sensor_point)
    )
    composed = world_from_robot.compose(
        robot_from_sensor
    ).apply_to_point(sensor_point)

    np.testing.assert_allclose(composed, sequential, atol=1e-12)


def test_homogeneous_matrix_matches_point_transform() -> None:
    transform = RigidTransform2D(
        angle=-0.3,
        translation=(1.0, 2.0),
    )
    homogeneous_point = np.array([2.0, 3.0, 1.0])

    matrix_result = transform.homogeneous_matrix @ homogeneous_point

    np.testing.assert_allclose(
        matrix_result[:2],
        transform.apply_to_point(homogeneous_point[:2]),
        atol=1e-12,
    )
    assert matrix_result[2] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("angle", "translation"),
    [
        (float("nan"), (0.0, 0.0)),
        (0.0, (1.0,)),
        (0.0, (float("inf"), 0.0)),
    ],
)
def test_transform_rejects_invalid_parameters(
    angle: float,
    translation: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        RigidTransform2D(angle=angle, translation=translation)
