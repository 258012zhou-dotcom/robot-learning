"""Unit tests for the first MuJoCo model and simulation loop."""

from pathlib import Path

import numpy as np
import pytest

from robot_learning.mujoco_basics import (
    load_mujoco_model,
    simulate_constant_control,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "017_mujoco_step"
    / "point_robot.xml"
)


def test_point_robot_model_has_one_state_and_one_control() -> None:
    """One slide joint and one motor should produce nq=nv=nu=1."""
    model = load_mujoco_model(MODEL_PATH)

    assert model.nq == 1
    assert model.nv == 1
    assert model.nu == 1
    assert model.opt.timestep == pytest.approx(0.01)


def test_simulation_includes_initial_state_and_expected_final_time() -> None:
    """Ten steps of 0.01 seconds should end at time 0.10."""
    model = load_mujoco_model(MODEL_PATH)

    result = simulate_constant_control(
        model,
        joint_name="x_slide",
        actuator_name="x_motor",
        control=0.5,
        step_count=10,
    )

    assert result.times.shape == (11,)
    assert result.positions.shape == (11,)
    assert result.velocities.shape == (11,)
    assert result.controls.shape == (11,)
    assert result.times[0] == pytest.approx(0.0)
    assert result.times[-1] == pytest.approx(0.10)


def test_zero_control_keeps_robot_at_rest() -> None:
    """With zero gravity and zero input, zero position should remain stable."""
    model = load_mujoco_model(MODEL_PATH)

    result = simulate_constant_control(
        model,
        joint_name="x_slide",
        actuator_name="x_motor",
        control=0.0,
        step_count=100,
    )

    np.testing.assert_allclose(result.positions, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.velocities, 0.0, atol=1e-12)


def test_positive_control_moves_robot_forward() -> None:
    """A positive motor command should produce positive x and x velocity."""
    model = load_mujoco_model(MODEL_PATH)

    result = simulate_constant_control(
        model,
        joint_name="x_slide",
        actuator_name="x_motor",
        control=1.0,
        step_count=100,
    )

    assert result.positions[-1] > 0.0
    assert result.velocities[-1] > 0.0
    assert np.all(np.diff(result.positions) >= 0.0)


def test_simulation_is_reproducible_from_fresh_data() -> None:
    """Identical model and control should generate exactly the same trajectory."""
    model = load_mujoco_model(MODEL_PATH)
    arguments = {
        "joint_name": "x_slide",
        "actuator_name": "x_motor",
        "control": 0.7,
        "step_count": 50,
    }

    first = simulate_constant_control(model, **arguments)
    second = simulate_constant_control(model, **arguments)

    np.testing.assert_array_equal(first.times, second.times)
    np.testing.assert_array_equal(first.positions, second.positions)
    np.testing.assert_array_equal(first.velocities, second.velocities)


def test_simulation_rejects_unknown_joint_name() -> None:
    """A misspelled MJCF name should fail instead of reading the wrong state."""
    model = load_mujoco_model(MODEL_PATH)

    with pytest.raises(ValueError, match="does not exist"):
        simulate_constant_control(
            model,
            joint_name="missing_joint",
            actuator_name="x_motor",
            control=0.5,
            step_count=10,
        )
