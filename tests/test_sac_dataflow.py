"""Unit tests for replay reuse and the SAC Critic target."""

import numpy as np
import pytest
import torch

from robot_learning.gymnasium_rollout import EpisodeResult
from robot_learning.sac_dataflow import sample_replay_batch, soft_q_targets
from robot_learning.trajectory_dataset import (
    CollectedEpisode,
    TEST_SPLIT_ID,
    TRAIN_SPLIT_ID,
    build_transition_dataset,
)


def make_dataset():
    episodes = []
    for episode_id, split_id in ((0, TRAIN_SPLIT_ID), (1, TEST_SPLIT_ID)):
        start = float(episode_id * 10)
        observations = np.asarray(
            [[start], [start + 1.0], [start + 2.0]], dtype=np.float32
        )
        result = EpisodeResult(
            observations=observations,
            actions=np.asarray([[0.25], [0.5]], dtype=np.float32),
            rewards=np.asarray([-1.0, -2.0], dtype=np.float64),
            total_reward=-3.0,
            terminated=True,
            truncated=False,
            is_success=True,
        )
        episodes.append(CollectedEpisode(
            episode_id=episode_id,
            environment_seed=100 + episode_id,
            policy_id=0,
            split_id=split_id,
            result=result,
        ))
    return build_transition_dataset(episodes)


def test_replay_sample_is_seeded_aligned_and_train_only() -> None:
    dataset = make_dataset()
    first = sample_replay_batch(
        dataset, batch_size=12, rng=np.random.default_rng(17)
    )
    second = sample_replay_batch(
        dataset, batch_size=12, rng=np.random.default_rng(17)
    )

    np.testing.assert_array_equal(first.indices, second.indices)
    assert set(first.indices.tolist()) <= {0, 1}
    np.testing.assert_array_equal(first.observations, dataset.observations[first.indices])
    np.testing.assert_array_equal(first.next_observations, dataset.next_observations[first.indices])
    np.testing.assert_array_equal(first.terminated, dataset.terminated[first.indices])
    assert dataset.transition_count == 4


def test_replay_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="positive"):
        sample_replay_batch(make_dataset(), batch_size=0, rng=np.random.default_rng(1))


def test_soft_target_uses_smaller_q_and_only_true_termination_stops_bootstrap() -> None:
    rewards = torch.tensor([1.0, 1.0, 1.0])
    next_q1 = torch.tensor([5.0, 5.0, 5.0], requires_grad=True)
    next_q2 = torch.tensor([4.0, 4.0, 4.0], requires_grad=True)
    next_log_probs = torch.tensor([-0.5, -0.5, -0.5])
    # Row 2 is a time-limit truncation, not a true termination.
    terminated = torch.tensor([False, True, False])

    targets = soft_q_targets(
        rewards, next_q1, next_q2, next_log_probs, terminated,
        gamma=0.9, alpha=0.2,
    )

    torch.testing.assert_close(targets, torch.tensor([4.69, 1.0, 4.69]))
    assert not targets.requires_grad


def test_soft_target_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        soft_q_targets(
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0]),
            torch.tensor([3.0, 4.0]),
            torch.tensor([-0.5, -0.5]),
            torch.tensor([False, False]),
            gamma=0.9,
            alpha=0.2,
        )
