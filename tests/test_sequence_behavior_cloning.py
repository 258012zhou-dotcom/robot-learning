"""Unit tests for masked chunk learning and execution schedules."""

import numpy as np
import pytest
import torch

from robot_learning.behavior_cloning import ObservationNormalization
from robot_learning.gymnasium_rollout import EpisodeResult
from robot_learning.sequence_behavior_cloning import (
    SequenceBCMLP,
    SequenceBCPolicy,
    SequenceBCTemporalEnsemblePolicy,
    masked_chunk_mse,
    prepare_sequence_data,
)
from robot_learning.trajectory_dataset import (
    CollectedEpisode,
    TEST_SPLIT_ID,
    TRAIN_SPLIT_ID,
    VALIDATION_SPLIT_ID,
    build_transition_dataset,
)


def test_masked_chunk_mse_counts_real_zero_but_not_padding() -> None:
    predicted = torch.tensor([[[1.0], [3.0], [100.0]]])
    target = torch.tensor([[[0.0], [1.0], [0.0]]])
    mask = torch.tensor([[True, True, False]])
    assert masked_chunk_mse(predicted, target, mask).item() == 2.5
    with pytest.raises(ValueError, match="valid action"):
        masked_chunk_mse(predicted, target, torch.zeros_like(mask))


def test_model_output_has_horizon_and_action_bounds() -> None:
    model = SequenceBCMLP(4, [-2.0], [2.0], horizon=3, hidden_size=8)
    predicted = model(torch.zeros((5, 4)))
    assert predicted.shape == (5, 3, 1)
    assert torch.all(predicted <= 2.0)
    assert torch.all(predicted >= -2.0)
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros((5, 3)))


class CountingChunkModel(SequenceBCMLP):
    def __init__(self) -> None:
        super().__init__(1, [-10.0], [10.0], horizon=3, hidden_size=4)
        self.query_count = 0

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        self.query_count += 1
        start = float(self.query_count * 10)
        return torch.tensor([[[start], [start + 1], [start + 2]]])


def test_hold_chunk_ignores_new_observation_until_chunk_ends_and_resets() -> None:
    model = CountingChunkModel()
    norm = ObservationNormalization(np.zeros(1), np.ones(1))
    policy = SequenceBCPolicy(model, norm, mode="hold_chunk")
    outputs = [float(policy(np.array([step]))[0]) for step in range(4)]
    assert outputs == [10.0, 11.0, 12.0, 20.0]
    assert model.query_count == 2
    policy.reset()
    assert float(policy(np.array([0.0]))[0]) == 30.0


def test_replan_each_step_uses_new_first_action() -> None:
    model = CountingChunkModel()
    norm = ObservationNormalization(np.zeros(1), np.ones(1))
    policy = SequenceBCPolicy(model, norm, mode="replan_each_step")
    outputs = [float(policy(np.array([step]))[0]) for step in range(3)]
    assert outputs == [10.0, 20.0, 30.0]
    assert model.query_count == 3


def test_hold_chunk_replans_after_fixed_execution_horizon() -> None:
    model = CountingChunkModel()
    norm = ObservationNormalization(np.zeros(1), np.ones(1))
    policy = SequenceBCPolicy(model, norm, mode="hold_chunk", execution_horizon=2)
    outputs = [float(policy(np.array([step]))[0]) for step in range(4)]
    assert outputs == [10.0, 11.0, 20.0, 21.0]
    assert model.query_count == 2
    with pytest.raises(ValueError, match="execution_horizon"):
        SequenceBCPolicy(model, norm, mode="hold_chunk", execution_horizon=4)


def test_temporal_policy_aligns_predictions_for_each_absolute_step() -> None:
    model = CountingChunkModel()
    norm = ObservationNormalization(np.zeros(1), np.ones(1))
    policy = SequenceBCTemporalEnsemblePolicy(model, norm, decay=0.0)
    outputs = [float(policy(np.array([step]))[0]) for step in range(4)]
    np.testing.assert_allclose(outputs, [10.0, 15.5, 21.0, 31.0])
    assert model.query_count == 4
    assert sorted(policy._chunks) == [1, 2, 3]


def test_temporal_policy_reset_discards_prior_episode_chunks() -> None:
    model = CountingChunkModel()
    norm = ObservationNormalization(np.zeros(1), np.ones(1))
    policy = SequenceBCTemporalEnsemblePolicy(model, norm, decay=0.0)
    policy(np.array([0.0]))
    policy(np.array([1.0]))
    policy.reset()
    assert float(policy(np.array([0.0]))[0]) == 30.0
    assert sorted(policy._chunks) == [0]


def test_sequence_data_uses_expert_episodes_and_train_only_normalization() -> None:
    def episode(value: float) -> EpisodeResult:
        observations = np.array([[value], [value + 1]], dtype=np.float32)
        return EpisodeResult(
            observations=observations,
            actions=np.array([[0.0]], dtype=np.float32),
            rewards=np.zeros(1),
            total_reward=0.0,
            terminated=True,
            truncated=False,
            is_success=True,
        )

    dataset = build_transition_dataset([
        CollectedEpisode(0, 10, 1, TRAIN_SPLIT_ID, episode(2.0)),
        CollectedEpisode(1, 11, 1, VALIDATION_SPLIT_ID, episode(100.0)),
        CollectedEpisode(2, 12, 1, TEST_SPLIT_ID, episode(200.0)),
        CollectedEpisode(3, 13, 0, TRAIN_SPLIT_ID, episode(1000.0)),
    ])
    prepared = prepare_sequence_data(dataset, horizon=2, expert_policy_id=1)
    np.testing.assert_array_equal(prepared.normalization.mean, [2.0])
    np.testing.assert_array_equal(prepared.train.episode_ids, [0])
    np.testing.assert_array_equal(prepared.validation.episode_ids, [1])
    np.testing.assert_array_equal(prepared.test.episode_ids, [2])
    np.testing.assert_array_equal(prepared.train.valid_mask, [[True, False]])


def test_sequence_data_keeps_both_routes_without_validation_leakage() -> None:
    def episode(value: float) -> EpisodeResult:
        return EpisodeResult(
            observations=np.array([[value], [value + 1]], dtype=np.float32),
            actions=np.array([[0.0]], dtype=np.float32),
            rewards=np.zeros(1), total_reward=0.0, terminated=True,
            truncated=False, is_success=True,
        )

    collected = [
        CollectedEpisode(index, 10 + index // 2, index % 2,
                         [TRAIN_SPLIT_ID, VALIDATION_SPLIT_ID, TEST_SPLIT_ID][index // 2],
                         episode([2.0, 4.0, 100.0, 120.0, 200.0, 220.0][index]))
        for index in range(6)
    ]
    dataset = build_transition_dataset(collected)
    prepared = prepare_sequence_data(dataset, horizon=2, expert_policy_ids=[0, 1])
    np.testing.assert_array_equal(prepared.normalization.mean, [3.0])
    np.testing.assert_array_equal(prepared.train.episode_ids, [0, 1])
    np.testing.assert_array_equal(prepared.validation.episode_ids, [2, 3])
    np.testing.assert_array_equal(prepared.test.episode_ids, [4, 5])
    with pytest.raises(ValueError, match="missing"):
        prepare_sequence_data(dataset, horizon=2, expert_policy_ids=[0, 2])
