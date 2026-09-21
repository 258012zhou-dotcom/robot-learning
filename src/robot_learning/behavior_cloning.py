"""Prepare Episode-separated expert data for behavior cloning."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from robot_learning.trajectory_dataset import (
    TEST_SPLIT_ID,
    TRAIN_SPLIT_ID,
    VALIDATION_SPLIT_ID,
    TransitionDataset,
    validate_transition_dataset,
)


@dataclass(frozen=True)
class ObservationNormalization:
    """Mean and safe scale fitted only on training observations."""

    mean: np.ndarray
    scale: np.ndarray

    def transform(self, observations: np.ndarray) -> np.ndarray:
        """Normalize a matrix without changing its sample membership."""
        values = np.asarray(observations)
        if values.ndim != 2:
            raise ValueError("observations must be a matrix")
        if values.shape[1] != self.mean.shape[0]:
            raise ValueError("observation feature count does not match normalization")
        if not np.all(np.isfinite(values)):
            raise ValueError("observations must contain only finite values")
        return ((values - self.mean) / self.scale).astype(np.float32)


@dataclass(frozen=True)
class BehaviorCloningSplit:
    """Supervised observation-action pairs from complete source Episodes."""

    observations: np.ndarray
    actions: np.ndarray
    episode_ids: np.ndarray


@dataclass(frozen=True)
class BehaviorCloningData:
    """Normalized expert-only train, validation, and test splits."""

    train: BehaviorCloningSplit
    validation: BehaviorCloningSplit
    test: BehaviorCloningSplit
    observation_normalization: ObservationNormalization


class BehaviorCloningMLP(nn.Module):
    """Map normalized observations to bounded continuous actions."""

    def __init__(
        self,
        observation_size: int,
        action_low: Sequence[float],
        action_high: Sequence[float],
        *,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        if type(observation_size) is not int or observation_size <= 0:
            raise ValueError("observation_size must be a positive integer")
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")

        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        if low.ndim != 1 or high.shape != low.shape or low.numel() == 0:
            raise ValueError("action bounds must be equal-length vectors")
        if not torch.isfinite(low).all() or not torch.isfinite(high).all():
            raise ValueError("action bounds must be finite")
        if not torch.all(low < high):
            raise ValueError("every action lower bound must be below its upper bound")

        self.observation_size = observation_size
        self.hidden_size = hidden_size
        self.action_size = int(low.numel())
        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_size),
        )
        self.register_buffer("action_midpoint", (low + high) / 2.0)
        self.register_buffer("action_half_range", (high - low) / 2.0)

    def forward(self, observations: Tensor) -> Tensor:
        """Predict one bounded action vector for every observation row."""
        if (
            observations.ndim != 2
            or observations.shape[1] != self.observation_size
        ):
            raise ValueError(
                f"observations must have shape (N, {self.observation_size})"
            )
        unbounded_actions = self.network(observations)
        return self.action_midpoint + self.action_half_range * torch.tanh(
            unbounded_actions
        )


class BehaviorCloningPolicy:
    """Apply saved normalization before one-step model inference."""

    def __init__(
        self,
        model: BehaviorCloningMLP,
        observation_normalization: ObservationNormalization,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.observation_normalization = observation_normalization
        self.device = torch.device(device)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Convert one raw observation into one bounded NumPy action."""
        value = np.asarray(observation, dtype=np.float32)
        if value.ndim != 1:
            raise ValueError("observation must be one-dimensional")
        normalized = self.observation_normalization.transform(value[None, :])
        inputs = torch.from_numpy(normalized).to(self.device)
        with torch.inference_mode():
            action = self.model(inputs)[0].cpu().numpy()
        return action.astype(np.float32, copy=False)


def save_behavior_cloning_checkpoint(
    path: str | Path,
    model: BehaviorCloningMLP,
    observation_normalization: ObservationNormalization,
    *,
    source_dataset_sha256: str,
    seed: int,
) -> None:
    """Save every parameter and preprocessing value needed for inference."""
    if not source_dataset_sha256:
        raise ValueError("source_dataset_sha256 must not be empty")
    artifact = {
        "format_version": 1,
        "model_config": {
            "observation_size": model.observation_size,
            "action_low": (
                model.action_midpoint - model.action_half_range
            ).detach().cpu(),
            "action_high": (
                model.action_midpoint + model.action_half_range
            ).detach().cpu(),
            "hidden_size": model.hidden_size,
        },
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "observation_mean": torch.from_numpy(
            observation_normalization.mean.copy()
        ),
        "observation_scale": torch.from_numpy(
            observation_normalization.scale.copy()
        ),
        "source_dataset_sha256": source_dataset_sha256,
        "seed": seed,
    }
    torch.save(artifact, Path(path))


def load_behavior_cloning_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[BehaviorCloningMLP, ObservationNormalization, dict[str, Any]]:
    """Rebuild a BC model and its exact train-time preprocessing contract."""
    artifact = torch.load(
        Path(path), map_location="cpu", weights_only=True
    )
    if artifact.get("format_version") != 1:
        raise ValueError("unsupported behavior cloning checkpoint format_version")
    config = artifact["model_config"]
    model = BehaviorCloningMLP(
        observation_size=int(config["observation_size"]),
        action_low=config["action_low"].tolist(),
        action_high=config["action_high"].tolist(),
        hidden_size=int(config["hidden_size"]),
    )
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    model.to(device).eval()

    mean = artifact["observation_mean"].cpu().numpy().copy()
    scale = artifact["observation_scale"].cpu().numpy().copy()
    if mean.ndim != 1 or scale.shape != mean.shape:
        raise ValueError("checkpoint normalization must contain equal-length vectors")
    if mean.shape[0] != model.observation_size:
        raise ValueError("checkpoint normalization does not match model input size")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("checkpoint normalization must be finite")
    if np.any(scale <= 0.0):
        raise ValueError("checkpoint normalization scale must be positive")
    normalization = ObservationNormalization(mean=mean, scale=scale)
    return model, normalization, artifact


def fit_observation_normalization(
    observations: np.ndarray,
    *,
    minimum_scale: float = 1e-6,
) -> ObservationNormalization:
    """Fit per-feature statistics and keep constant features numerically safe."""
    values = np.asarray(observations)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("training observations must be a non-empty matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("training observations must contain only finite values")
    if not np.isfinite(minimum_scale) or minimum_scale <= 0.0:
        raise ValueError("minimum_scale must be positive and finite")

    mean = values.mean(axis=0, dtype=np.float64)
    standard_deviation = values.std(axis=0, dtype=np.float64)
    scale = np.where(standard_deviation < minimum_scale, 1.0, standard_deviation)
    return ObservationNormalization(
        mean=mean.astype(np.float32),
        scale=scale.astype(np.float32),
    )


def prepare_behavior_cloning_data(
    dataset: TransitionDataset,
    *,
    expert_policy_id: int,
) -> BehaviorCloningData:
    """Select expert rows and normalize every split with train-only statistics."""
    validate_transition_dataset(dataset)
    raw_splits = {
        split_id: _select_expert_split(
            dataset,
            expert_policy_id=expert_policy_id,
            split_id=split_id,
        )
        for split_id in (TRAIN_SPLIT_ID, VALIDATION_SPLIT_ID, TEST_SPLIT_ID)
    }
    _validate_episode_separation(raw_splits)

    normalization = fit_observation_normalization(
        raw_splits[TRAIN_SPLIT_ID].observations
    )
    normalized_splits = {
        split_id: replace(
            split,
            observations=normalization.transform(split.observations),
        )
        for split_id, split in raw_splits.items()
    }
    return BehaviorCloningData(
        train=normalized_splits[TRAIN_SPLIT_ID],
        validation=normalized_splits[VALIDATION_SPLIT_ID],
        test=normalized_splits[TEST_SPLIT_ID],
        observation_normalization=normalization,
    )


def _select_expert_split(
    dataset: TransitionDataset,
    *,
    expert_policy_id: int,
    split_id: int,
) -> BehaviorCloningSplit:
    rows = (dataset.policy_ids == expert_policy_id) & (
        dataset.split_ids == split_id
    )
    if not np.any(rows):
        raise ValueError(
            f"expert policy {expert_policy_id} has no rows in split {split_id}"
        )
    return BehaviorCloningSplit(
        observations=dataset.observations[rows].copy(),
        actions=dataset.actions[rows].copy(),
        episode_ids=dataset.episode_ids[rows].copy(),
    )


def _validate_episode_separation(
    splits: dict[int, BehaviorCloningSplit],
) -> None:
    episode_sets = {
        split_id: set(np.unique(split.episode_ids).tolist())
        for split_id, split in splits.items()
    }
    split_ids = list(episode_sets)
    for index, first_id in enumerate(split_ids):
        for second_id in split_ids[index + 1 :]:
            if episode_sets[first_id] & episode_sets[second_id]:
                raise ValueError("an expert Episode appears in multiple splits")
