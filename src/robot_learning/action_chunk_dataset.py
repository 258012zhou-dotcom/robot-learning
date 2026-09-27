"""Build fixed-length action labels without crossing Episode boundaries."""

from dataclasses import dataclass

import numpy as np

from robot_learning.trajectory_dataset import TransitionDataset, validate_transition_dataset


@dataclass(frozen=True)
class ActionChunkDataset:
    """One observation and a possibly padded future-action chunk per transition."""

    observations: np.ndarray
    action_chunks: np.ndarray
    valid_mask: np.ndarray
    action_row_indices: np.ndarray
    episode_ids: np.ndarray
    step_ids: np.ndarray
    split_ids: np.ndarray


def build_action_chunks(dataset: TransitionDataset, *, horizon: int) -> ActionChunkDataset:
    """Pair observation t with actions t..t+horizon-1 from its own Episode."""
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    validate_transition_dataset(dataset)

    count, action_dimension = dataset.actions.shape
    chunks = np.zeros((count, horizon, action_dimension), dtype=dataset.actions.dtype)
    valid_mask = np.zeros((count, horizon), dtype=np.bool_)
    action_row_indices = np.full((count, horizon), -1, dtype=np.int64)

    for episode_id in np.unique(dataset.episode_ids):
        rows = np.flatnonzero(dataset.episode_ids == episode_id)
        for local_start, source_row in enumerate(rows):
            included_rows = rows[local_start:local_start + horizon]
            length = included_rows.size
            chunks[source_row, :length] = dataset.actions[included_rows]
            valid_mask[source_row, :length] = True
            action_row_indices[source_row, :length] = included_rows

    return ActionChunkDataset(
        observations=dataset.observations.copy(),
        action_chunks=chunks,
        valid_mask=valid_mask,
        action_row_indices=action_row_indices,
        episode_ids=dataset.episode_ids.copy(),
        step_ids=dataset.step_ids.copy(),
        split_ids=dataset.split_ids.copy(),
    )
