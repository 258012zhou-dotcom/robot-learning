"""Small, testable replay and soft-Q target pieces for SAC learning."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from robot_learning.trajectory_dataset import TRAIN_SPLIT_ID, TransitionDataset


@dataclass(frozen=True)
class ReplayBatch:
    """Aligned transitions sampled from an existing training dataset."""

    indices: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray


def sample_replay_batch(
    dataset: TransitionDataset,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> ReplayBatch:
    """Sample old training transitions, with replacement and without env steps."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    train_indices = np.flatnonzero(dataset.split_ids == TRAIN_SPLIT_ID)
    if train_indices.size == 0:
        raise ValueError("dataset has no training transitions")
    indices = rng.choice(train_indices, size=batch_size, replace=True)
    return ReplayBatch(
        indices=indices,
        observations=dataset.observations[indices],
        actions=dataset.actions[indices],
        rewards=dataset.rewards[indices],
        next_observations=dataset.next_observations[indices],
        terminated=dataset.terminated[indices],
        truncated=dataset.truncated[indices],
    )


def soft_q_targets(
    rewards: Tensor,
    next_q1: Tensor,
    next_q2: Tensor,
    next_log_probabilities: Tensor,
    terminated: Tensor,
    *,
    gamma: float,
    alpha: float,
) -> Tensor:
    """Compute detached SAC targets; time-limit truncation still bootstraps."""
    tensors = (rewards, next_q1, next_q2, next_log_probabilities, terminated)
    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("rewards must be a non-empty vector")
    if any(item.shape != rewards.shape for item in tensors):
        raise ValueError("all target inputs must have the same shape")
    if terminated.dtype != torch.bool:
        raise ValueError("terminated must be boolean")
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1]")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if not all(torch.isfinite(item).all() for item in tensors[:-1]):
        raise ValueError("target inputs must be finite")

    with torch.no_grad():
        next_soft_value = (
            torch.minimum(next_q1, next_q2)
            - alpha * next_log_probabilities
        )
        return rewards + gamma * (~terminated).to(rewards.dtype) * next_soft_value
