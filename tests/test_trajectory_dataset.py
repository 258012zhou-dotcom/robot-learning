"""Unit tests for Episode-preserving transition dataset utilities."""

from dataclasses import fields, replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from robot_learning.gymnasium_rollout import EpisodeResult
from robot_learning.point_robot_reach_env import PointRobotReachEnv
from robot_learning.trajectory_dataset import (
    CollectedEpisode,
    TEST_SPLIT_ID,
    TRAIN_SPLIT_ID,
    VALIDATION_SPLIT_ID,
    assign_stratified_episode_splits,
    build_transition_dataset,
    load_transition_dataset,
    save_transition_dataset,
    validate_transition_dataset,
)


def make_result(*, step_count: int, terminated: bool) -> EpisodeResult:
    """Create a short continuous trajectory for deterministic dataset tests."""
    positions = np.arange(step_count + 1, dtype=np.float32)
    observations = np.zeros((step_count + 1, 4), dtype=np.float32)
    observations[:, 0] = positions
    observations[:, 2] = 10.0
    observations[:, 3] = 10.0 - positions
    actions = np.full((step_count, 1), 0.25, dtype=np.float32)
    rewards = -np.arange(1, step_count + 1, dtype=np.float64)
    return EpisodeResult(
        observations=observations,
        actions=actions,
        rewards=rewards,
        total_reward=float(rewards.sum()),
        terminated=terminated,
        truncated=not terminated,
        is_success=terminated,
    )


def make_collected_episodes() -> list[CollectedEpisode]:
    """Return one successful and one truncated Episode with separate splits."""
    return [
        CollectedEpisode(0, 40, 0, TRAIN_SPLIT_ID, make_result(step_count=2, terminated=True)),
        CollectedEpisode(1, 41, 1, TEST_SPLIT_ID, make_result(step_count=3, terminated=False)),
    ]


def test_episode_conversion_preserves_transition_alignment() -> None:
    """Rows must connect observation t, action t, and observation t+1."""
    dataset = build_transition_dataset(make_collected_episodes())

    assert dataset.transition_count == 5
    assert dataset.episode_count == 2
    np.testing.assert_array_equal(dataset.observations[:2, 0], [0.0, 1.0])
    np.testing.assert_array_equal(dataset.next_observations[:2, 0], [1.0, 2.0])
    np.testing.assert_array_equal(dataset.step_ids, [0, 1, 0, 1, 2])


def test_only_final_transition_contains_episode_end_flag() -> None:
    """Termination and truncation should describe the last transition only."""
    dataset = build_transition_dataset(make_collected_episodes())

    np.testing.assert_array_equal(dataset.terminated, [False, True, False, False, False])
    np.testing.assert_array_equal(dataset.truncated, [False, False, False, False, True])


def test_stratified_split_keeps_every_policy_in_every_split() -> None:
    """Each policy group should contribute complete Episodes to all splits."""
    episode_ids = list(range(20))
    policy_ids = [0] * 10 + [1] * 10

    assignments = assign_stratified_episode_splits(
        episode_ids,
        policy_ids,
        train_fraction=0.6,
        validation_fraction=0.2,
        seed=42,
    )

    for policy_id in (0, 1):
        group_ids = [
            episode_id
            for episode_id, assigned_policy in zip(episode_ids, policy_ids)
            if assigned_policy == policy_id
        ]
        group_splits = [assignments[episode_id] for episode_id in group_ids]
        assert group_splits.count(TRAIN_SPLIT_ID) == 6
        assert group_splits.count(VALIDATION_SPLIT_ID) == 2
        assert group_splits.count(TEST_SPLIT_ID) == 2


def test_npz_round_trip_preserves_every_array(tmp_path) -> None:
    """Saving and loading must not change values, dtypes, or Episode IDs."""
    original = build_transition_dataset(make_collected_episodes())
    path = tmp_path / "dataset.npz"

    save_transition_dataset(path, original)
    loaded = load_transition_dataset(path)

    for field in fields(original):
        np.testing.assert_array_equal(
            getattr(loaded, field.name),
            getattr(original, field.name),
        )


def test_validation_rejects_action_outside_declared_bounds() -> None:
    """A numeric file is invalid when executed actions violate the contract."""
    dataset = build_transition_dataset(make_collected_episodes())
    invalid_actions = dataset.actions.copy()
    invalid_actions[0, 0] = 2.0
    invalid_dataset = replace(dataset, actions=invalid_actions)

    with pytest.raises(ValueError, match="actions exceed"):
        validate_transition_dataset(
            invalid_dataset,
            action_low=np.asarray([-1.0]),
            action_high=np.asarray([1.0]),
        )


def test_validation_rejects_broken_episode_continuity() -> None:
    """The next state of one row must equal the current state of the next row."""
    dataset = build_transition_dataset(make_collected_episodes())
    broken_observations = dataset.observations.copy()
    broken_observations[1, 0] = 99.0

    with pytest.raises(ValueError, match="does not continue"):
        validate_transition_dataset(
            replace(dataset, observations=broken_observations)
        )


def test_time_limit_and_success_flags_survive_dataset_round_trip(tmp_path) -> None:
    """A real TimeLimit wrapper may return termination and truncation together."""
    model_path = (
        Path(__file__).resolve().parents[1]
        / "experiments/017_mujoco_step/point_robot.xml"
    )
    environment = gym.wrappers.TimeLimit(
        PointRobotReachEnv(model_path), max_episode_steps=1,
    )
    try:
        observation, _ = environment.reset(
            seed=1, options={"initial_position": 1.0, "target_position": 1.0},
        )
        action = np.asarray([0.0], dtype=np.float32)
        next_observation, reward, terminated, truncated, info = environment.step(action)
        assert terminated and truncated
        result = EpisodeResult(
            observations=np.stack([observation, next_observation]),
            actions=action.reshape(1, 1), rewards=np.asarray([reward]),
            total_reward=reward, terminated=terminated, truncated=truncated,
            is_success=info["is_success"],
        )
        dataset = build_transition_dataset([
            CollectedEpisode(0, 1, 0, TRAIN_SPLIT_ID, result),
        ])
        path = tmp_path / "both_end_flags.npz"
        save_transition_dataset(path, dataset)
        loaded = load_transition_dataset(path)
        assert loaded.terminated.tolist() == [True]
        assert loaded.truncated.tolist() == [True]
    finally:
        environment.close()


def test_unfinished_episode_is_still_rejected() -> None:
    """Allowing both flags must not allow neither flag on a complete Episode."""
    unfinished = replace(
        make_result(step_count=2, terminated=True),
        terminated=False,
        truncated=False,
    )
    with pytest.raises(ValueError, match="must end"):
        build_transition_dataset([
            CollectedEpisode(0, 1, 0, TRAIN_SPLIT_ID, unfinished),
        ])
    dataset = build_transition_dataset(make_collected_episodes())
    ended = dataset.terminated.copy()
    ended[1] = False
    with pytest.raises(ValueError, match="final transition"):
        validate_transition_dataset(replace(dataset, terminated=ended))


@pytest.mark.parametrize("field_name", ["terminated", "truncated"])
def test_end_flags_require_boolean_dtype(field_name) -> None:
    """Non-boolean flags must not be silently interpreted by truthiness."""
    dataset = build_transition_dataset(make_collected_episodes())
    with pytest.raises(ValueError, match="boolean dtype"):
        validate_transition_dataset(replace(
            dataset, **{field_name: getattr(dataset, field_name).astype(float)},
        ))
