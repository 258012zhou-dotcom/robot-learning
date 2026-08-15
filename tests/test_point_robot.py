import numpy as np
import pytest

from robot_learning.point_robot import next_position, simulate_constant_velocity


def test_robot_moves_forward():
    assert next_position(position=0, velocity=2, dt=0.5) == 1


def test_robot_moves_backward():
    assert next_position(position=3, velocity=-1, dt=2) == 1

def test_robot_stays_still_when_velocity_is_zero():
    assert next_position(position=5, velocity=0, dt=1.5) == 5

def test_negative_dt_is_rejected():
    with pytest.raises(ValueError):
        next_position(position=0, velocity=2, dt=-0.5)


def test_trajectory_has_expected_shape_and_initial_position():
    trajectory = simulate_constant_velocity([0, 0], [1, 0.5], dt=0.1, steps=100)

    assert trajectory.shape == (101, 2)
    np.testing.assert_allclose(trajectory[0], [0, 0])


def test_trajectory_has_expected_final_position():
    trajectory = simulate_constant_velocity([0, 0], [1, 0.5], dt=0.1, steps=100)

    np.testing.assert_allclose(trajectory[-1], [10, 5])


@pytest.mark.parametrize(
    ("initial_position", "velocity", "dt", "steps"),
    [
        ([0, 0], [1, 0.5], -0.1, 10),
        ([0, 0], [1, 0.5], 0.1, -1),
        ([0], [1, 0.5], 0.1, 10),
        ([0, 0], [1], 0.1, 10),
    ],
)
def test_invalid_simulation_input_is_rejected(
    initial_position, velocity, dt, steps
):
    with pytest.raises(ValueError):
        simulate_constant_velocity(initial_position, velocity, dt, steps)
