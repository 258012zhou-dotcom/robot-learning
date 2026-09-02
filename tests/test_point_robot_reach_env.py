"""Contract and behavior tests for the Gymnasium point-robot reach task."""

from pathlib import Path

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from robot_learning.point_robot_reach_env import PointRobotReachEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "017_mujoco_step"
    / "point_robot.xml"
)


def make_environment(**kwargs) -> PointRobotReachEnv:
    """Create a small environment while allowing one test to override settings."""
    return PointRobotReachEnv(MODEL_PATH, **kwargs)


def test_environment_passes_gymnasium_contract_checker() -> None:
    """Gymnasium should accept reset, step, spaces, dtypes, and seeding behavior."""
    environment = make_environment()

    check_env(environment, skip_render_check=True)


def test_reset_returns_valid_observation_and_diagnostics() -> None:
    """Reset must return one observation inside its declared Box space."""
    environment = make_environment()

    observation, info = environment.reset(seed=10)

    assert observation.shape == (4,)
    assert observation.dtype == np.float32
    assert environment.observation_space.contains(observation)
    assert info["elapsed_steps"] == 0
    assert info["distance"] >= environment.minimum_target_distance


def test_reset_with_same_seed_reproduces_target() -> None:
    """The environment RNG should reproduce task targets independently of MuJoCo."""
    environment = make_environment()

    first_observation, _ = environment.reset(seed=11)
    second_observation, _ = environment.reset(seed=11)

    np.testing.assert_array_equal(first_observation, second_observation)


def test_positive_action_moves_robot_forward_and_advances_control_time() -> None:
    """One environment step should apply one action for frame_skip physics steps."""
    environment = make_environment(frame_skip=5)
    environment.reset(
        seed=12,
        options={"initial_position": 0.0, "target_position": 2.0},
    )

    observation, _, terminated, truncated, info = environment.step(
        np.asarray([1.0], dtype=np.float32)
    )

    assert observation[0] > 0.0
    assert observation[1] > 0.0
    assert environment.data.time == pytest.approx(
        environment.control_timestep
    )
    assert info["elapsed_steps"] == 1
    assert not terminated
    assert not truncated


def test_success_terminates_episode() -> None:
    """Being at the target with zero velocity is a task terminal state."""
    environment = make_environment()
    environment.reset(
        seed=13,
        options={"initial_position": 1.0, "target_position": 1.0},
    )

    _, _, terminated, truncated, info = environment.step(
        np.asarray([0.0], dtype=np.float32)
    )

    assert terminated
    assert not truncated
    assert info["is_success"]


def test_time_limit_truncates_unsolved_episode() -> None:
    """An unsolved task should stop at max steps without claiming termination."""
    environment = make_environment(max_episode_steps=2)
    environment.reset(
        seed=14,
        options={"initial_position": 0.0, "target_position": 2.0},
    )

    first = environment.step(np.asarray([0.0], dtype=np.float32))
    second = environment.step(np.asarray([0.0], dtype=np.float32))

    assert not first[2]
    assert not first[3]
    assert not second[2]
    assert second[3]


def test_action_is_clipped_to_mjcf_control_range() -> None:
    """An action of 100 should produce the same first step as the upper bound 1."""
    environment = make_environment()
    reset_options = {"initial_position": 0.0, "target_position": 2.0}

    environment.reset(seed=15, options=reset_options)
    clipped_observation = environment.step(
        np.asarray([100.0], dtype=np.float32)
    )[0]
    environment.reset(seed=15, options=reset_options)
    bounded_observation = environment.step(
        np.asarray([1.0], dtype=np.float32)
    )[0]

    np.testing.assert_allclose(clipped_observation, bounded_observation)


def test_reward_penalizes_distance_and_unnecessary_action() -> None:
    """At equal distance, nonzero action should receive a slightly lower reward."""
    passive_environment = make_environment(frame_skip=1)
    active_environment = make_environment(frame_skip=1)
    options = {"initial_position": 0.0, "target_position": 2.0}
    passive_environment.reset(seed=16, options=options)
    active_environment.reset(seed=16, options=options)

    passive_reward = passive_environment.step(
        np.asarray([0.0], dtype=np.float32)
    )[1]
    active_reward = active_environment.step(
        np.asarray([1.0], dtype=np.float32)
    )[1]

    # The one-step motion is tiny; the explicit action penalty dominates here.
    assert active_reward < passive_reward
