"""Evaluation-time sampling for a trained discrete PPO actor."""

import numpy as np
import torch

from robot_learning.ppo_training import ACTION_VALUES, DiscreteActorCritic


class SampledDiscretePolicy:
    """Draw actions from the actor with an isolated, reproducible RNG."""

    def __init__(self, model: DiscreteActorCritic, seed: int) -> None:
        if seed < 0:
            raise ValueError("sampling seed must be non-negative")
        self.model = model
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = self.model.actor(
                torch.as_tensor(observation, dtype=torch.float32)
            )
            probabilities = torch.softmax(logits, dim=-1)
            action_index = int(
                torch.multinomial(
                    probabilities, 1, generator=self._generator
                ).item()
            )
        return np.asarray([ACTION_VALUES[action_index]], dtype=np.float32)
