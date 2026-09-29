"""Small observation-conditioned diffusion model for low-dimensional action chunks.

This is a teaching model, not the image-based Diffusion Policy paper system.
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from robot_learning.behavior_cloning import ObservationNormalization


def masked_noise_mse(predicted: Tensor, noise: Tensor, valid_mask: Tensor) -> Tensor:
    """Measure epsilon prediction only at real, non-padded action positions."""
    if predicted.ndim != 3 or predicted.shape != noise.shape:
        raise ValueError("predicted noise and target noise must have the same chunk shape")
    if valid_mask.shape != predicted.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean with shape (batch, horizon)")
    if not torch.any(valid_mask):
        raise ValueError("a batch must contain valid actions")
    squared = (predicted - noise).square()
    return (squared * valid_mask.unsqueeze(-1)).sum() / (
        valid_mask.sum() * predicted.shape[-1]
    )


class LowDimActionDiffusion(nn.Module):
    """Predict added noise from observation, noisy action chunk, and diffusion step."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        *,
        horizon: int,
        diffusion_steps: int = 32,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        if any(type(value) is not int or value <= 0 for value in
               (observation_size, action_size, horizon, hidden_size)):
            raise ValueError("model dimensions must be positive integers")
        if type(diffusion_steps) is not int or diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least two")
        self.observation_size = observation_size
        self.action_size = action_size
        self.horizon = horizon
        self.diffusion_steps = diffusion_steps
        self.hidden_size = hidden_size
        time_size = 16
        self.network = nn.Sequential(
            nn.Linear(observation_size + horizon * action_size + time_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon * action_size),
        )

        # Cosine schedule makes the final training state almost pure Gaussian noise.
        grid = torch.arange(diffusion_steps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(
            ((grid / diffusion_steps + 0.008) / 1.008) * math.pi / 2
        ).square()
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(0.0001, 0.999)
        self.register_buffer("betas", betas.to(torch.float32))
        self.register_buffer("alphas", 1 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))

    def _time_embedding(self, timesteps: Tensor) -> Tensor:
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(8, device=timesteps.device, dtype=torch.float32) / 7
        )
        angles = timesteps.float()[:, None] * frequencies[None, :]
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)

    def forward(
        self, observations: Tensor, noisy_actions: Tensor, timesteps: Tensor
    ) -> Tensor:
        batch = observations.shape[0]
        if observations.shape != (batch, self.observation_size):
            raise ValueError("observations have the wrong shape")
        if noisy_actions.shape != (batch, self.horizon, self.action_size):
            raise ValueError("noisy action chunks have the wrong shape")
        if timesteps.shape != (batch,) or timesteps.dtype != torch.long:
            raise ValueError("timesteps must be a long vector per batch")
        if torch.any(timesteps < 0) or torch.any(timesteps >= self.diffusion_steps):
            raise ValueError("timesteps outside diffusion schedule")
        features = torch.cat(
            (observations, noisy_actions.flatten(start_dim=1),
             self._time_embedding(timesteps)), dim=1
        )
        return self.network(features).reshape(batch, self.horizon, self.action_size)

    def add_noise(self, actions: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        """Draw q(x_t|x_0) using caller-provided noise for exact testing."""
        if actions.ndim != 3 or actions.shape[1:] != (self.horizon, self.action_size):
            raise ValueError("actions have the wrong chunk shape")
        if noise.shape != actions.shape or timesteps.shape != (actions.shape[0],):
            raise ValueError("noise or timesteps have the wrong shape")
        if timesteps.dtype != torch.long or torch.any(timesteps < 0) or torch.any(
            timesteps >= self.diffusion_steps
        ):
            raise ValueError("timesteps outside diffusion schedule")
        alpha_bar = self.alpha_bars[timesteps, None, None]
        return alpha_bar.sqrt() * actions + (1 - alpha_bar).sqrt() * noise

    def training_loss(
        self, observations: Tensor, actions: Tensor, valid_mask: Tensor
    ) -> Tensor:
        batch = observations.shape[0]
        timesteps = torch.randint(
            self.diffusion_steps, (batch,), device=observations.device
        )
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        noisy = self.add_noise(actions, timesteps, noise)
        # Padded actions are not trajectory data; hide their noisy values.
        noisy = noisy * valid_mask.unsqueeze(-1)
        predicted = self(observations, noisy, timesteps)
        return masked_noise_mse(predicted, noise, valid_mask)

    @torch.inference_mode()
    def sample(
        self, observations: Tensor, *, generator: torch.Generator
    ) -> Tensor:
        """Ancestral DDPM sampling, with final velocity commands bounded to [-1, 1]."""
        if observations.ndim != 2 or observations.shape[1] != self.observation_size:
            raise ValueError("observations have the wrong shape")
        actions = torch.randn(
            (observations.shape[0], self.horizon, self.action_size),
            device=observations.device,
            dtype=observations.dtype,
            generator=generator,
        )
        for step in range(self.diffusion_steps - 1, -1, -1):
            timesteps = torch.full(
                (observations.shape[0],), step, device=observations.device,
                dtype=torch.long,
            )
            predicted_noise = self(observations, actions, timesteps)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            mean = (actions - self.betas[step] * predicted_noise /
                    (1 - alpha_bar).sqrt()) / alpha.sqrt()
            if step > 0:
                previous_alpha_bar = self.alpha_bars[step - 1]
                variance = self.betas[step] * (1 - previous_alpha_bar) / (1 - alpha_bar)
                noise = torch.randn(
                    actions.shape, device=actions.device, dtype=actions.dtype,
                    generator=generator,
                )
                actions = mean + variance.sqrt() * noise
            else:
                actions = mean
        return actions.clamp(-1.0, 1.0)


class DiffusionChunkPolicy:
    """Sample H actions, execute a short prefix, then replan from new observation."""

    def __init__(
        self,
        model: LowDimActionDiffusion,
        normalization: ObservationNormalization,
        *,
        execution_horizon: int,
        device: str = "cpu",
    ) -> None:
        if type(execution_horizon) is not int or not 1 <= execution_horizon <= model.horizon:
            raise ValueError("execution_horizon must be within prediction horizon")
        self.model = model.to(device).eval()
        self.normalization = normalization
        self.execution_horizon = execution_horizon
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.reset(seed=0)

    def reset(self, *, seed: int | None = None) -> None:
        self._chunk: np.ndarray | None = None
        self._cursor = 0
        if seed is not None:
            self.generator.manual_seed(seed)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        if self._chunk is None or self._cursor >= self.execution_horizon:
            value = np.asarray(observation, dtype=np.float32)
            if value.ndim != 1:
                raise ValueError("observation must be one-dimensional")
            normalized = self.normalization.transform(value[None, :])
            inputs = torch.from_numpy(normalized).to(self.device)
            self._chunk = self.model.sample(inputs, generator=self.generator)[0].cpu().numpy()
            self._cursor = 0
        action = self._chunk[self._cursor].copy()
        self._cursor += 1
        return action.astype(np.float32, copy=False)


def save_diffusion_checkpoint(
    path: str | Path,
    model: LowDimActionDiffusion,
    normalization: ObservationNormalization,
    *,
    source_dataset_sha256: str,
    seed: int,
) -> None:
    if not source_dataset_sha256:
        raise ValueError("source_dataset_sha256 must not be empty")
    torch.save({
        "format_version": 1,
        "model_config": {
            "observation_size": model.observation_size,
            "action_size": model.action_size,
            "horizon": model.horizon,
            "diffusion_steps": model.diffusion_steps,
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
    }, Path(path))


def load_diffusion_checkpoint(
    path: str | Path, *, device: str = "cpu"
) -> tuple[LowDimActionDiffusion, ObservationNormalization, dict[str, Any]]:
    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    if artifact.get("format_version") != 1:
        raise ValueError("unsupported diffusion checkpoint format")
    model = LowDimActionDiffusion(**artifact["model_config"])
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    mean = artifact["observation_mean"].cpu().numpy().copy()
    scale = artifact["observation_scale"].cpu().numpy().copy()
    if mean.shape != (model.observation_size,) or scale.shape != mean.shape:
        raise ValueError("checkpoint normalization has the wrong shape")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("checkpoint normalization must be finite and positive")
    model.to(device).eval()
    return model, ObservationNormalization(mean, scale), artifact
