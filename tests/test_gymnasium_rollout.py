"""Tests for policies and complete Gymnasium Episode recording."""

from pathlib import Path

import numpy as np
import pytest

from robot_learning.gymnasium_rollout import (
    EpisodeResult,
    ProportionalReachPolicy,
    RandomPolicy,
    run_episode,
    summarize_episodes,
)
from robot_learning.point_robot_reach_env import PointRobotReachEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "017_mujoco_step"
    / "point_robot.xml"
)


def make_environment(**kwargs) -> PointRobotReachEnv:
    """Create the real MuJoCo reach environment used by rollout tests."""
    return PointRobotReachEnv(MODEL_PATH, **kwargs)


def test_proportional_policy_uses_target_error_and_action_limits() -> None:
    """Positive and excessive errors should become positive bounded actions."""
    environment = make_environment()
    policy = ProportionalReachPolicy(0.5, environment.action_space)

    small_action = policy(np.asarray([0.0, 0.0, 1.0, 1.0]))
    clipped_action = policy(np.asarray([0.0, 0.0, 5.0, 5.0]))

    np.testing.assert_allclose(small_action, [0.5])
    np.testing.assert_allclose(clipped_action, [1.0])


def test_random_policy_is_reproducible_and_bounded() -> None:
    """Equal policy seeds should generate equal valid action sequences."""
    environment = make_environment()
    first_policy = RandomPolicy(environment.action_space, seed=20)
    second_policy = RandomPolicy(environment.action_space, seed=20)
    observation = np.zeros(4, dtype=np.float32)

    first_actions = [first_policy(observation) for _ in range(5)]
    second_actions = [second_policy(observation) for _ in range(5)]

    for first, second in zip(first_actions, second_actions):
        np.testing.assert_array_equal(first, second)
        assert environment.action_space.contains(first)


def test_rollout_records_one_more_observation_than_action() -> None:
    """An Episode has an initial observation plus one result per action."""
    environment = make_environment(max_episode_steps=3)
    policy = RandomPolicy(environment.action_space, seed=21)

    result = run_episode(environment, policy, seed=21)

    assert result.step_count == 3
    assert result.observations.shape == (4, 4)
    assert result.actions.shape == (3, 1)
    assert result.rewards.shape == (3,)
    assert result.total_reward == pytest.approx(result.rewards.sum())
    assert not result.terminated
    assert result.truncated


def test_proportional_policy_completes_positive_target_episode() -> None:
    """The hand-designed baseline should reach a deterministic positive target."""
    environment = make_environment(max_episode_steps=400)
    policy = ProportionalReachPolicy(0.5, environment.action_space)

    result = run_episode(
        environment,
        policy,
        seed=22,
        reset_options={"initial_position": 0.0, "target_position": 1.0},
    )

    assert result.is_success
    assert result.terminated
    assert not result.truncated
    assert result.step_count < 400
    assert abs(result.observations[-1, 3]) <= environment.success_tolerance


def make_episode_result(
    *,
    step_count: int,
    total_reward: float,
    final_error: float,
    is_success: bool,
) -> EpisodeResult:
    """Build compact synthetic evidence for testing aggregate calculations."""
    observations = np.zeros((step_count + 1, 4), dtype=np.float32)
    observations[-1, 3] = final_error
    return EpisodeResult(
        observations=observations,
        actions=np.zeros((step_count, 1), dtype=np.float32),
        rewards=np.full(step_count, total_reward / step_count),
        total_reward=total_reward,
        terminated=is_success,
        truncated=not is_success,
        is_success=is_success,
    )


def test_episode_summary_uses_consistent_aggregate_metrics() -> None:
    """Two Episodes should produce transparent arithmetic means and success rate."""
    results = [
        make_episode_result(
            step_count=10,
            total_reward=-5.0,
            final_error=0.02,
            is_success=True,
        ),
        make_episode_result(
            step_count=20,
            total_reward=-15.0,
            final_error=-0.50,
            is_success=False,
        ),
    ]

    summary = summarize_episodes(results)

    assert summary.episode_count == 2
    assert summary.success_count == 1
    assert summary.success_rate == pytest.approx(0.5)
    assert summary.mean_total_reward == pytest.approx(-10.0)
    assert summary.mean_step_count == pytest.approx(15.0)
    assert summary.mean_final_distance == pytest.approx(0.26)


def test_episode_summary_rejects_empty_input() -> None:
    """An empty result set has no meaningful mean or success rate."""
    with pytest.raises(ValueError, match="at least one Episode"):
        summarize_episodes([])
