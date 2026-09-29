"""Synthetic upper/lower route instructions for the two-route teaching task."""

import numpy as np

from robot_learning.action_chunk_dataset import build_action_chunks
from robot_learning.behavior_cloning import (
    ObservationNormalization,
    fit_observation_normalization,
)
from robot_learning.sequence_behavior_cloning import (
    SequenceBCMLP,
    SequenceBCPolicy,
    SequenceData,
    SequenceSplit,
)
from robot_learning.trajectory_dataset import (
    TEST_SPLIT_ID,
    TRAIN_SPLIT_ID,
    VALIDATION_SPLIT_ID,
    TransitionDataset,
)


ROUTE_BY_POLICY_ID = {0: -1, 1: 1}


def prepare_route_conditioned_data(
    dataset: TransitionDataset, *, horizon: int
) -> SequenceData:
    """Append a route instruction while retaining the original scene split."""
    if set(np.unique(dataset.policy_ids).tolist()) != set(ROUTE_BY_POLICY_ID):
        raise ValueError("dataset must contain exactly both route policy IDs")
    chunks = build_action_chunks(dataset, horizon=horizon)
    raw: dict[int, SequenceSplit] = {}
    scene_sets: list[set[int]] = []
    episode_sets: list[set[int]] = []
    for split_id in (TRAIN_SPLIT_ID, VALIDATION_SPLIT_ID, TEST_SPLIT_ID):
        selected = dataset.split_ids == split_id
        if set(dataset.policy_ids[selected].tolist()) != set(ROUTE_BY_POLICY_ID):
            raise ValueError(f"split {split_id} must contain both routes")
        instruction = np.where(dataset.policy_ids[selected] == 0, -1.0, 1.0)
        observations = np.column_stack(
            (chunks.observations[selected], instruction)
        ).astype(np.float32)
        raw[split_id] = SequenceSplit(
            observations=observations,
            action_chunks=chunks.action_chunks[selected].copy(),
            valid_mask=chunks.valid_mask[selected].copy(),
            episode_ids=chunks.episode_ids[selected].copy(),
        )
        scene_sets.append(set(dataset.environment_seeds[selected].tolist()))
        episode_sets.append(set(dataset.episode_ids[selected].tolist()))
    for sets, name in ((scene_sets, "scene"), (episode_sets, "Episode")):
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"{name} leakage across data splits")
    normalization = fit_observation_normalization(raw[TRAIN_SPLIT_ID].observations)
    normalized = {
        split_id: SequenceSplit(
            observations=normalization.transform(split.observations),
            action_chunks=split.action_chunks,
            valid_mask=split.valid_mask,
            episode_ids=split.episode_ids,
        )
        for split_id, split in raw.items()
    }
    return SequenceData(
        train=normalized[TRAIN_SPLIT_ID],
        validation=normalized[VALIDATION_SPLIT_ID],
        test=normalized[TEST_SPLIT_ID],
        normalization=normalization,
    )


def route_at_obstacle_x(observations: np.ndarray) -> int | None:
    """Classify which side a trajectory crossed at the obstacle center x."""
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or values.shape[0] < 2:
        raise ValueError("observations must have shape (T+1, 7), with T >= 1")
    obstacle_x, obstacle_y, radius = values[0, 4:7]
    for first, second in zip(values[:-1], values[1:], strict=True):
        x1, x2 = first[0], second[0]
        if x1 <= obstacle_x <= x2 and x2 > x1:
            fraction = (obstacle_x - x1) / (x2 - x1)
            crossing_y = first[1] + fraction * (second[1] - first[1])
            if crossing_y > obstacle_y + radius:
                return 1
            if crossing_y < obstacle_y - radius:
                return -1
            return 0
    return None


class RouteConditionedSequenceBCPolicy:
    """Keep a commanded route fixed while the chunk BC policy replans."""

    def __init__(
        self,
        model: SequenceBCMLP,
        normalization: ObservationNormalization,
        *,
        execution_horizon: int,
    ) -> None:
        if model.observation_size != 8:
            raise ValueError("route-conditioned model must take eight input features")
        self.policy = SequenceBCPolicy(
            model,
            normalization,
            mode="hold_chunk",
            execution_horizon=execution_horizon,
        )
        self.route: int | None = None

    def reset(self, *, route: int) -> None:
        if route not in (-1, 1):
            raise ValueError("route instruction must be -1 or +1")
        self.route = route
        self.policy.reset()

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        if self.route is None:
            raise RuntimeError("set route instruction before rollout")
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (7,):
            raise ValueError("environment observation must have seven features")
        conditioned = np.concatenate(
            (value, np.asarray([self.route], dtype=np.float32))
        )
        return self.policy(conditioned)
