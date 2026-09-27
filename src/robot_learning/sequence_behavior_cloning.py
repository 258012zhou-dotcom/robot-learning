"""Small, non-ACT behavior-cloning baseline that predicts action chunks."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn

from robot_learning.action_chunk_dataset import build_action_chunks
from robot_learning.behavior_cloning import (
    ObservationNormalization,
    fit_observation_normalization,
)
from robot_learning.trajectory_dataset import (
    TEST_SPLIT_ID,
    TRAIN_SPLIT_ID,
    VALIDATION_SPLIT_ID,
    TransitionDataset,
)
from robot_learning.temporal_ensemble import ensemble_action_at


@dataclass(frozen=True)
class SequenceSplit:
    observations: np.ndarray
    action_chunks: np.ndarray
    valid_mask: np.ndarray
    episode_ids: np.ndarray


@dataclass(frozen=True)
class SequenceData:
    train: SequenceSplit
    validation: SequenceSplit
    test: SequenceSplit
    normalization: ObservationNormalization


def prepare_sequence_data(
    dataset: TransitionDataset, *, horizon: int, expert_policy_id: int
) -> SequenceData:
    """Use expert-only complete Episodes and train-only observation statistics."""
    chunks = build_action_chunks(dataset, horizon=horizon)
    raw: dict[int, SequenceSplit] = {}
    for split_id in (TRAIN_SPLIT_ID, VALIDATION_SPLIT_ID, TEST_SPLIT_ID):
        selected = (dataset.policy_ids == expert_policy_id) & (
            dataset.split_ids == split_id
        )
        if not np.any(selected):
            raise ValueError(f"no expert rows for split {split_id}")
        raw[split_id] = SequenceSplit(
            observations=chunks.observations[selected].copy(),
            action_chunks=chunks.action_chunks[selected].copy(),
            valid_mask=chunks.valid_mask[selected].copy(),
            episode_ids=chunks.episode_ids[selected].copy(),
        )
    episode_sets = [set(np.unique(split.episode_ids)) for split in raw.values()]
    if any(
        episode_sets[i] & episode_sets[j]
        for i in range(len(episode_sets))
        for j in range(i + 1, len(episode_sets))
    ):
        raise ValueError("expert Episodes overlap across splits")
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


class SequenceBCMLP(nn.Module):
    """Map one observation to H bounded actions; no Transformer or CVAE."""

    def __init__(
        self,
        observation_size: int,
        action_low: Sequence[float],
        action_high: Sequence[float],
        *,
        horizon: int,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        if type(observation_size) is not int or observation_size <= 0:
            raise ValueError("observation_size must be positive")
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("horizon must be positive")
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        if low.ndim != 1 or high.shape != low.shape or low.numel() == 0:
            raise ValueError("action bounds must be equal-length vectors")
        if not torch.isfinite(low).all() or not torch.isfinite(high).all():
            raise ValueError("action bounds must be finite")
        if not torch.all(low < high):
            raise ValueError("action lower bounds must be below upper bounds")
        self.observation_size = observation_size
        self.action_size = int(low.numel())
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon * self.action_size),
        )
        self.register_buffer("action_midpoint", (low + high) / 2.0)
        self.register_buffer("action_half_range", (high - low) / 2.0)

    def forward(self, observations: Tensor) -> Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.observation_size:
            raise ValueError("observations have the wrong shape")
        raw = self.network(observations).reshape(
            -1, self.horizon, self.action_size
        )
        return self.action_midpoint + self.action_half_range * torch.tanh(raw)


def masked_chunk_mse(
    predictions: Tensor, targets: Tensor, valid_mask: Tensor
) -> Tensor:
    """Average squared error only over genuine actions, including real zeros."""
    if predictions.ndim != 3 or targets.shape != predictions.shape:
        raise ValueError("prediction and target chunks must match")
    if valid_mask.shape != predictions.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be a boolean (batch, horizon) matrix")
    if not torch.any(valid_mask):
        raise ValueError("a batch must contain at least one valid action")
    squared_error = (predictions - targets).square()
    return (squared_error * valid_mask.unsqueeze(-1)).sum() / (
        valid_mask.sum() * predictions.shape[2]
    )


class SequenceBCPolicy:
    """Execute a whole predicted chunk or replan from each new observation."""

    def __init__(
        self,
        model: SequenceBCMLP,
        normalization: ObservationNormalization,
        *,
        mode: Literal["hold_chunk", "replan_each_step"],
        device: str | torch.device = "cpu",
    ) -> None:
        if mode not in ("hold_chunk", "replan_each_step"):
            raise ValueError("unknown execution mode")
        self.model = model.to(device).eval()
        self.normalization = normalization
        self.mode = mode
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        """Discard actions from the previous Episode."""
        self._chunk: np.ndarray | None = None
        self._cursor = 0

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        if self.mode == "replan_each_step" or self._chunk is None or (
            self._cursor >= self.model.horizon
        ):
            value = np.asarray(observation, dtype=np.float32)
            if value.ndim != 1:
                raise ValueError("observation must be one-dimensional")
            normalized = self.normalization.transform(value[None, :])
            with torch.inference_mode():
                chunk = self.model(
                    torch.from_numpy(normalized).to(self.device)
                )[0].cpu().numpy()
            self._chunk = chunk
            self._cursor = 0
        assert self._chunk is not None
        action = self._chunk[self._cursor].copy()
        self._cursor += 1
        return action.astype(np.float32, copy=False)


class SequenceBCTemporalEnsemblePolicy:
    """Query every step and combine overlapping predictions for that step."""

    def __init__(
        self,
        model: SequenceBCMLP,
        normalization: ObservationNormalization,
        *,
        decay: float,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.normalization = normalization
        self.decay = decay
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        """Discard predictions from the previous Episode."""
        self._step = 0
        self._chunks: dict[int, np.ndarray] = {}

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if value.ndim != 1:
            raise ValueError("observation must be one-dimensional")
        normalized = self.normalization.transform(value[None, :])
        with torch.inference_mode():
            chunk = self.model(
                torch.from_numpy(normalized).to(self.device)
            )[0].cpu().numpy()
        self._chunks[self._step] = chunk
        # Older chunks no longer cover this absolute step.
        oldest_active = self._step - self.model.horizon + 1
        self._chunks = {
            issued: prediction
            for issued, prediction in self._chunks.items()
            if issued >= oldest_active
        }
        result = ensemble_action_at(
            self._chunks, target_step=self._step, decay=self.decay
        )
        self._step += 1
        return result.action.copy()


def save_sequence_checkpoint(
    path: str | Path,
    model: SequenceBCMLP,
    normalization: ObservationNormalization,
    *,
    source_dataset_sha256: str,
    seed: int,
) -> None:
    """Keep model, train-time normalization, and dataset identity together."""
    if not source_dataset_sha256:
        raise ValueError("source_dataset_sha256 must not be empty")
    torch.save(
        {
            "format_version": 1,
            "model_config": {
                "observation_size": model.observation_size,
                "action_low": (
                    model.action_midpoint - model.action_half_range
                ).detach().cpu(),
                "action_high": (
                    model.action_midpoint + model.action_half_range
                ).detach().cpu(),
                "horizon": model.horizon,
                "hidden_size": model.hidden_size,
            },
            "model_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
            "observation_mean": torch.from_numpy(normalization.mean.copy()),
            "observation_scale": torch.from_numpy(normalization.scale.copy()),
            "source_dataset_sha256": source_dataset_sha256,
            "seed": seed,
        },
        Path(path),
    )


def load_sequence_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[SequenceBCMLP, ObservationNormalization, dict[str, Any]]:
    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    if artifact.get("format_version") != 1:
        raise ValueError("unsupported sequence checkpoint format")
    config = artifact["model_config"]
    model = SequenceBCMLP(
        int(config["observation_size"]),
        config["action_low"].tolist(),
        config["action_high"].tolist(),
        horizon=int(config["horizon"]),
        hidden_size=int(config["hidden_size"]),
    )
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    mean = artifact["observation_mean"].cpu().numpy().copy()
    scale = artifact["observation_scale"].cpu().numpy().copy()
    if mean.shape != (model.observation_size,) or scale.shape != mean.shape:
        raise ValueError("checkpoint normalization has the wrong shape")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("checkpoint normalization must be finite and positive")
    model.to(device).eval()
    return model, ObservationNormalization(mean, scale), artifact
