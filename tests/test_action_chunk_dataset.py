"""Unit checks for action-chunk alignment, masking, and split isolation."""

import numpy as np
import pytest

from robot_learning.action_chunk_dataset import build_action_chunks
from robot_learning.gymnasium_rollout import EpisodeResult
from robot_learning.trajectory_dataset import (
    CollectedEpisode,
    TEST_SPLIT_ID,
    TRAIN_SPLIT_ID,
    build_transition_dataset,
)


def short_episode(actions: list[float]) -> EpisodeResult:
    """Make simple observations with T+1 states and T distinct action labels."""
    step_count = len(actions)
    observations = np.zeros((step_count + 1, 4), dtype=np.float32)
    observations[:, 0] = np.arange(step_count + 1)
    observations[:, 2] = 10.0
    observations[:, 3] = 10.0 - observations[:, 0]
    return EpisodeResult(
        observations=observations,
        actions=np.asarray(actions, dtype=np.float32).reshape(-1, 1),
        rewards=np.zeros(step_count),
        total_reward=0.0,
        terminated=True,
        truncated=False,
        is_success=True,
    )


def two_episode_dataset():
    """Put a two-step train Episode directly before a three-step test Episode."""
    return build_transition_dataset([
        CollectedEpisode(10, 1, 0, TRAIN_SPLIT_ID, short_episode([0.0, 0.25])),
        CollectedEpisode(20, 2, 0, TEST_SPLIT_ID, short_episode([-0.3, -0.4, -0.5])),
    ])


def test_chunks_stop_at_episode_and_split_boundaries() -> None:
    dataset = two_episode_dataset()
    chunks = build_action_chunks(dataset, horizon=3)

    np.testing.assert_array_equal(chunks.action_row_indices, [
        [0, 1, -1], [1, -1, -1], [2, 3, 4], [3, 4, -1], [4, -1, -1],
    ])
    np.testing.assert_array_equal(chunks.valid_mask, chunks.action_row_indices >= 0)
    np.testing.assert_array_equal(chunks.split_ids, [0, 0, 2, 2, 2])
    np.testing.assert_array_equal(chunks.observations, dataset.observations)
    for source_row in range(dataset.transition_count):
        for action_row in chunks.action_row_indices[source_row, chunks.valid_mask[source_row]]:
            assert dataset.episode_ids[action_row] == dataset.episode_ids[source_row]
            assert dataset.split_ids[action_row] == dataset.split_ids[source_row]


def test_zero_padding_is_not_confused_with_a_real_zero_action() -> None:
    chunks = build_action_chunks(two_episode_dataset(), horizon=3)

    np.testing.assert_array_equal(chunks.action_chunks[0, :, 0], [0.0, 0.25, 0.0])
    np.testing.assert_array_equal(chunks.valid_mask[0], [True, True, False])
    np.testing.assert_array_equal(chunks.action_chunks[1, :, 0], [0.25, 0.0, 0.0])
    np.testing.assert_array_equal(chunks.valid_mask[1], [True, False, False])


def test_every_transition_remains_a_training_example_and_horizon_is_validated() -> None:
    dataset = two_episode_dataset()
    chunks = build_action_chunks(dataset, horizon=3)

    assert chunks.action_chunks.shape == (dataset.transition_count, 3, 1)
    assert chunks.valid_mask.shape == (dataset.transition_count, 3)
    np.testing.assert_array_equal(chunks.action_row_indices[:, 0], np.arange(5))
    with pytest.raises(ValueError, match="horizon"):
        build_action_chunks(dataset, horizon=0)
