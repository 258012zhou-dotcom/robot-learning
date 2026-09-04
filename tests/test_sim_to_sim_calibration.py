"""Tests for deployment mismatch estimation and compensated control."""

from pathlib import Path

import numpy as np
import pytest

from robot_learning.point_robot_reach_env import PointRobotReachEnv
from robot_learning.sim_to_sim_calibration import (
    CalibratedPDReachPolicy,
    collect_step_response,
    DeploymentCalibration,
    estimate_deployment_calibration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "017_mujoco_step"
    / "point_robot.xml"
)


def make_environment() -> PointRobotReachEnv:
    """Create the same MuJoCo environment used by experiment 022."""
    return PointRobotReachEnv(MODEL_PATH, frame_skip=5, max_episode_steps=400)


def test_step_response_estimates_hidden_gain_delay_and_bias() -> None:
    """Known proxy mismatch should be recovered without reading true info."""
    calibration_action = 0.5
    nominal_environment = make_environment()
    nominal_environment.reset(
        seed=1,
        options={"initial_position": 0.0, "target_position": 2.0},
    )
    nominal_velocity = float(
        nominal_environment.step(
            np.asarray([calibration_action], dtype=np.float32)
        )[0][1]
    )

    target_environment = make_environment()
    trace = collect_step_response(
        target_environment,
        seed=2,
        reset_options={
            "initial_position": 0.0,
            "target_position": 2.0,
            "action_gain": 0.65,
            "action_delay_steps": 3,
            "observation_position_bias": 0.08,
        },
        command=calibration_action,
        step_count=8,
    )
    estimate = estimate_deployment_calibration(
        trace,
        known_initial_position=0.0,
        nominal_step_velocity=nominal_velocity,
        velocity_threshold=1e-6,
    )

    assert estimate.action_gain == pytest.approx(0.65, rel=1e-5)
    assert estimate.action_delay_steps == 3
    assert estimate.observation_position_bias == pytest.approx(0.08)


def test_calibrated_policy_corrects_bias_and_actuator_gain() -> None:
    """A small desired force should be divided by the estimated weak gain."""
    environment = make_environment()
    policy = CalibratedPDReachPolicy(
        proportional_gain=1.0,
        derivative_gain=0.0,
        action_space=environment.action_space,
        control_timestep=environment.control_timestep,
        calibration=DeploymentCalibration(
            action_gain=0.5,
            action_delay_steps=0,
            observation_position_bias=0.2,
        ),
    )
    # Measured position 0.2 minus bias 0.2 gives estimated true position 0.0.
    observation = np.asarray([0.2, 0.0, 0.3, 0.1], dtype=np.float32)

    action = policy(observation)

    # Desired executed action is 0.3; a 0.5 actuator gain needs command 0.6.
    assert action[0] == pytest.approx(0.6)


def test_delay_prediction_reduces_action_when_moving_toward_target() -> None:
    """Forward prediction should brake earlier when commands arrive late."""
    environment = make_environment()
    no_delay_policy = CalibratedPDReachPolicy(
        proportional_gain=1.0,
        derivative_gain=0.0,
        action_space=environment.action_space,
        control_timestep=0.1,
        calibration=DeploymentCalibration(1.0, 0, 0.0),
    )
    delayed_policy = CalibratedPDReachPolicy(
        proportional_gain=1.0,
        derivative_gain=0.0,
        action_space=environment.action_space,
        control_timestep=0.1,
        calibration=DeploymentCalibration(1.0, 3, 0.0),
    )
    observation = np.asarray([0.5, 1.0, 1.0, 0.5], dtype=np.float32)

    no_delay_action = float(no_delay_policy(observation)[0])
    delayed_action = float(delayed_policy(observation)[0])

    assert no_delay_action == pytest.approx(0.5)
    assert delayed_action == pytest.approx(0.2)


@pytest.mark.parametrize(
    "calibration",
    [
        (0.0, 0, 0.0),
        (1.0, -1, 0.0),
        (1.0, 0, np.nan),
    ],
)
def test_deployment_calibration_rejects_invalid_values(calibration) -> None:
    """Impossible estimated parameters should fail close to their source."""
    with pytest.raises(ValueError):
        DeploymentCalibration(*calibration)
