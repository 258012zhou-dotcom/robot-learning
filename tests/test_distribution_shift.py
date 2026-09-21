"""Tests for deterministic target-directed rollout disturbances."""

from pathlib import Path

import numpy as np
import pytest

from robot_learning.distribution_shift import (
    evaluate_velocity_sweep,
    run_episode_with_target_directed_override,
)
from robot_learning.gymnasium_rollout import (
    ProportionalReachPolicy,
    run_episode,
)
from robot_learning.point_robot_reach_env import PointRobotReachEnv


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "experiments/017_mujoco_step/point_robot.xml"


def make_environment(*, max_episode_steps: int = 20) -> PointRobotReachEnv:
    """Create the real one-dimensional MuJoCo environment."""
    return PointRobotReachEnv(
        MODEL_PATH,
        max_episode_steps=max_episode_steps,
    )


def test_zero_duration_matches_normal_rollout_exactly() -> None:
    """The paired baseline must not alter actions, states, or rewards."""
    environment = make_environment()
    policy = ProportionalReachPolicy(0.5, environment.action_space)
    try:
        expected = run_episode(environment, policy, seed=44)
        actual = run_episode_with_target_directed_override(
            environment,
            policy,
            seed=44,
            override_start_step=2,
            override_duration_steps=0,
            override_magnitude=1.0,
        )
    finally:
        environment.close()

    assert actual.override_applied_steps == 0
    np.testing.assert_array_equal(
        actual.episode.observations, expected.observations
    )
    np.testing.assert_array_equal(actual.episode.actions, expected.actions)
    np.testing.assert_array_equal(actual.episode.rewards, expected.rewards)


@pytest.mark.parametrize(
    ("target_position", "expected_action"),
    [(1.0, 0.75), (-1.0, -0.75)],
)
def test_override_uses_target_direction_at_exact_requested_steps(
    target_position: float,
    expected_action: float,
) -> None:
    """Only the selected action indices should receive the signed override."""
    environment = make_environment(max_episode_steps=4)
    policy = ProportionalReachPolicy(0.5, environment.action_space)
    try:
        result = run_episode_with_target_directed_override(
            environment,
            policy,
            seed=45,
            reset_options={
                "initial_position": 0.0,
                "target_position": target_position,
            },
            override_start_step=1,
            override_duration_steps=2,
            override_magnitude=0.75,
        )
    finally:
        environment.close()

    assert result.override_applied_steps == 2
    np.testing.assert_array_equal(
        result.episode.actions[1:3, 0],
        [expected_action, expected_action],
    )
    assert result.episode.actions[0, 0] != expected_action


@pytest.mark.parametrize(
    ("start", "duration", "magnitude"),
    [(-1, 1, 1.0), (0, -1, 1.0), (0, 1, 0.0), (0, 1, 2.0)],
)
def test_invalid_override_configuration_is_rejected(
    start: int,
    duration: int,
    magnitude: float,
) -> None:
    """Invalid timing or actions must fail before producing evidence."""
    environment = make_environment()
    policy = ProportionalReachPolicy(0.5, environment.action_space)
    try:
        with pytest.raises(ValueError):
            run_episode_with_target_directed_override(
                environment,
                policy,
                seed=46,
                override_start_step=start,
                override_duration_steps=duration,
                override_magnitude=magnitude,
            )
    finally:
        environment.close()


def test_proportional_policy_is_invariant_to_velocity_when_error_is_fixed() -> None:
    """The expert formula uses target error and must ignore velocity."""
    environment = make_environment()
    policy = ProportionalReachPolicy(0.5, environment.action_space)
    try:
        result = evaluate_velocity_sweep(
            policy,
            base_observation=np.asarray(
                [0.0, 0.0, 1.0, 1.0], dtype=np.float32
            ),
            velocities=np.asarray([-2.0, 0.0, 2.0], dtype=np.float32),
        )
    finally:
        environment.close()

    np.testing.assert_array_equal(result.velocities, [-2.0, 0.0, 2.0])
    np.testing.assert_array_equal(result.actions[:, 0], [0.5, 0.5, 0.5])


def test_velocity_sweep_rejects_non_vector_observation() -> None:
    """The sensitivity check must vary one complete observation at a time."""
    environment = make_environment()
    policy = ProportionalReachPolicy(0.5, environment.action_space)
    try:
        with pytest.raises(ValueError, match="one-dimensional"):
            evaluate_velocity_sweep(
                policy,
                base_observation=np.zeros((1, 4), dtype=np.float32),
                velocities=np.asarray([0.0], dtype=np.float32),
            )
    finally:
        environment.close()
