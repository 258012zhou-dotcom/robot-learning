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


def test_reset_applies_and_reports_episode_domain_parameters() -> None:
    """Mass and damping should be explicit diagnostics for every Episode."""
    environment = make_environment()

    _, info = environment.reset(
        seed=17,
        options={
            "target_position": 1.0,
            "body_mass": 1.5,
            "joint_damping": 0.7,
        },
    )

    assert info["body_mass"] == pytest.approx(1.5)
    assert info["joint_damping"] == pytest.approx(0.7)


def test_heavier_body_accelerates_less_under_equal_action() -> None:
    """Equal force over equal time should change a lighter body's speed more."""
    light_environment = make_environment()
    heavy_environment = make_environment()
    reset_base = {"initial_position": 0.0, "target_position": 2.0}
    light_environment.reset(
        seed=18,
        options={**reset_base, "body_mass": 0.5},
    )
    heavy_environment.reset(
        seed=18,
        options={**reset_base, "body_mass": 2.0},
    )

    light_velocity = light_environment.step(
        np.asarray([1.0], dtype=np.float32)
    )[0][1]
    heavy_velocity = heavy_environment.step(
        np.asarray([1.0], dtype=np.float32)
    )[0][1]

    assert light_velocity > heavy_velocity > 0.0


def test_higher_damping_reduces_speed_under_repeated_action() -> None:
    """Damping should oppose motion and lower speed under the same force."""
    low_damping_environment = make_environment(max_episode_steps=200)
    high_damping_environment = make_environment(max_episode_steps=200)
    reset_base = {"initial_position": 0.0, "target_position": 2.0}
    low_damping_environment.reset(
        seed=19,
        options={**reset_base, "joint_damping": 0.1},
    )
    high_damping_environment.reset(
        seed=19,
        options={**reset_base, "joint_damping": 1.0},
    )

    action = np.asarray([0.5], dtype=np.float32)
    for _ in range(20):
        low_observation = low_damping_environment.step(action)[0]
        high_observation = high_damping_environment.step(action)[0]

    assert low_observation[1] > high_observation[1] > 0.0


def test_reset_without_domain_options_restores_nominal_values() -> None:
    """A randomized Episode must not contaminate the next nominal reset."""
    environment = make_environment()
    environment.reset(
        seed=20,
        options={"body_mass": 1.8, "joint_damping": 0.9},
    )

    _, nominal_info = environment.reset(seed=20)

    assert nominal_info["body_mass"] == pytest.approx(
        environment.nominal_body_mass
    )
    assert nominal_info["joint_damping"] == pytest.approx(
        environment.nominal_joint_damping
    )


@pytest.mark.parametrize(
    "invalid_options",
    [
        {"body_mass": 0.0},
        {"body_mass": np.inf},
        {"joint_damping": -0.1},
        {"joint_damping": np.nan},
    ],
)
def test_reset_rejects_nonphysical_domain_parameters(invalid_options) -> None:
    """Nonphysical domains should fail before an Episode starts."""
    environment = make_environment()

    with pytest.raises(ValueError):
        environment.reset(seed=21, options=invalid_options)


def test_action_gain_changes_the_executed_action_and_motion() -> None:
    """A weak actuator should execute less force under the same command."""
    nominal_environment = make_environment(frame_skip=1)
    weak_environment = make_environment(frame_skip=1)
    reset_options = {"initial_position": 0.0, "target_position": 2.0}
    nominal_environment.reset(seed=22, options=reset_options)
    weak_environment.reset(
        seed=22,
        options={**reset_options, "action_gain": 0.5},
    )

    nominal_observation, _, _, _, nominal_info = nominal_environment.step(
        np.asarray([1.0], dtype=np.float32)
    )
    weak_observation, _, _, _, weak_info = weak_environment.step(
        np.asarray([1.0], dtype=np.float32)
    )

    assert nominal_info["commanded_action"] == pytest.approx(1.0)
    assert weak_info["commanded_action"] == pytest.approx(1.0)
    assert nominal_info["executed_action"] == pytest.approx(1.0)
    assert weak_info["executed_action"] == pytest.approx(0.5)
    assert nominal_observation[1] > weak_observation[1] > 0.0


def test_action_delay_executes_a_command_after_exactly_two_steps() -> None:
    """A two-step delay should execute zero, zero, then the first command."""
    environment = make_environment(frame_skip=1)
    environment.reset(
        seed=23,
        options={
            "initial_position": 0.0,
            "target_position": 2.0,
            "action_delay_steps": 2,
        },
    )
    command = np.asarray([1.0], dtype=np.float32)

    first = environment.step(command)
    second = environment.step(command)
    third = environment.step(command)

    assert first[4]["executed_action"] == pytest.approx(0.0)
    assert second[4]["executed_action"] == pytest.approx(0.0)
    assert third[4]["executed_action"] == pytest.approx(1.0)
    assert first[0][0] == pytest.approx(0.0)
    assert second[0][0] == pytest.approx(0.0)
    assert third[0][0] > 0.0


def test_position_bias_changes_observation_but_not_true_success() -> None:
    """A biased sensor must not make the true task look physically solved."""
    environment = make_environment(frame_skip=1)
    observation, reset_info = environment.reset(
        seed=24,
        options={
            "initial_position": 0.0,
            "target_position": 0.2,
            "observation_position_bias": 0.2,
        },
    )

    assert observation[0] == pytest.approx(0.2)
    assert observation[3] == pytest.approx(0.0)
    assert reset_info["position"] == pytest.approx(0.0)
    assert reset_info["distance"] == pytest.approx(0.2)

    _, _, terminated, _, step_info = environment.step(
        np.asarray([0.0], dtype=np.float32)
    )

    assert not terminated
    assert not step_info["is_success"]
    assert step_info["position"] == pytest.approx(0.0)
    assert step_info["observed_position"] == pytest.approx(0.2)


def test_reset_restores_nominal_deployment_interface() -> None:
    """Actuator and sensor mismatch must not leak into the next Episode."""
    environment = make_environment()
    environment.reset(
        seed=25,
        options={
            "action_gain": 0.6,
            "action_delay_steps": 3,
            "observation_position_bias": -0.15,
        },
    )

    _, nominal_info = environment.reset(seed=25)

    assert nominal_info["action_gain"] == pytest.approx(1.0)
    assert nominal_info["action_delay_steps"] == 0
    assert nominal_info["observation_position_bias"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "invalid_options",
    [
        {"action_gain": 0.0},
        {"action_gain": np.inf},
        {"action_delay_steps": -1},
        {"action_delay_steps": 1.5},
        {"action_delay_steps": 21},
        {"observation_position_bias": np.nan},
        {"observation_position_bias": 1.1},
    ],
)
def test_reset_rejects_invalid_deployment_mismatches(invalid_options) -> None:
    """Invalid actuator and sensor settings should fail before simulation."""
    environment = make_environment()

    with pytest.raises(ValueError):
        environment.reset(seed=26, options=invalid_options)
