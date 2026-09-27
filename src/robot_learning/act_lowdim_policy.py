"""Inference-only execution of the teaching-scale low-dimensional ACT model."""

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from robot_learning.act_lowdim import ACTLowDim
from robot_learning.behavior_cloning import ObservationNormalization
from robot_learning.temporal_ensemble import ensemble_action_at


def load_act_lowdim_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[ACTLowDim, ObservationNormalization, dict[str, Any]]:
    """Restore model and its train-only observation normalization together."""
    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    if artifact.get("format_version") != 1:
        raise ValueError("unsupported ACT lowdim checkpoint format")
    if not artifact.get("source_dataset_sha256"):
        raise ValueError("checkpoint source dataset identity is missing")
    model = ACTLowDim(**artifact["model_config"])
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    mean = artifact["observation_mean"].cpu().numpy().copy()
    scale = artifact["observation_scale"].cpu().numpy().copy()
    if mean.shape != (model.observation_size,) or scale.shape != mean.shape:
        raise ValueError("checkpoint normalization has the wrong shape")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("checkpoint normalization must be finite")
    if np.any(scale <= 0):
        raise ValueError("checkpoint normalization scale must be positive")
    model.to(device).eval()
    return model, ObservationNormalization(mean, scale), artifact


class ACTLowDimPolicy:
    """Use z=0; execute newest first action or ensemble overlapping chunks."""

    def __init__(
        self,
        model: ACTLowDim,
        normalization: ObservationNormalization,
        *,
        mode: Literal["latest_first_action", "temporal_ensemble"],
        decay: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        if mode not in ("latest_first_action", "temporal_ensemble"):
            raise ValueError("unknown ACT execution mode")
        if not math.isfinite(decay) or decay < 0:
            raise ValueError("decay must be finite and non-negative")
        self.model = model.to(device).eval()
        self.normalization = normalization
        self.mode = mode
        self.decay = decay
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        """Drop all chunks at the boundary between Episodes."""
        self._step = 0
        self._chunks: dict[int, np.ndarray] = {}

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if value.ndim != 1:
            raise ValueError("observation must be one-dimensional")
        normalized = self.normalization.transform(value[None, :])
        with torch.inference_mode():
            # No target action chunk is supplied: ACTLowDim.forward uses z=0.
            output = self.model(torch.from_numpy(normalized).to(self.device))
        if output.posterior_mean is not None or output.posterior_logvar is not None:
            raise RuntimeError("execution must not use the training posterior")
        chunk = output.actions[0].cpu().numpy()
        if self.mode == "latest_first_action":
            return chunk[0].copy()

        self._chunks[self._step] = chunk
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
