"""Small continuous-action Actor and twin Critic modules for SAC."""

import math

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F


def squashed_log_probability(distribution: Normal, raw_actions: Tensor) -> Tensor:
    """Log probability after tanh, summed across action dimensions."""
    # Stable equivalent of log(1 - tanh(raw_actions) ** 2).
    log_tanh_derivative = 2.0 * (
        math.log(2.0) - raw_actions - F.softplus(-2.0 * raw_actions)
    )
    return (distribution.log_prob(raw_actions) - log_tanh_derivative).sum(dim=-1)


class SquashedGaussianActor(nn.Module):
    """Sample bounded continuous actions with a differentiable log probability."""

    def __init__(
        self, observation_dimension: int = 4, action_dimension: int = 1,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        if min(observation_dimension, action_dimension, hidden_size) <= 0:
            raise ValueError("network dimensions must be positive")
        self.observation_dimension = observation_dimension
        self.network = nn.Sequential(
            nn.Linear(observation_dimension, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_size, action_dimension)
        self.log_std_head = nn.Linear(hidden_size, action_dimension)

    def _distribution(self, observations: Tensor) -> Normal:
        if observations.ndim != 2 or observations.shape[1] != self.observation_dimension:
            raise ValueError("observations must have shape [batch, observation_dimension]")
        features = self.network(observations)
        mean = self.mean_head(features)
        # Bounds avoid extreme standard deviations during early learning.
        log_std = self.log_std_head(features).clamp(-5.0, 2.0)
        return Normal(mean, log_std.exp())

    def sample(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        """Return actions [batch, action_dim] and log probabilities [batch]."""
        distribution = self._distribution(observations)
        raw_actions = distribution.rsample()
        return torch.tanh(raw_actions), squashed_log_probability(distribution, raw_actions)

    def deterministic_action(self, observations: Tensor) -> Tensor:
        """Use the squashed Gaussian mean for repeatable evaluation."""
        return torch.tanh(self._distribution(observations).mean)


class TwinQCritic(nn.Module):
    """Two independent action-value estimates from the same state-action pair."""

    def __init__(
        self, observation_dimension: int = 4, action_dimension: int = 1,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        if min(observation_dimension, action_dimension, hidden_size) <= 0:
            raise ValueError("network dimensions must be positive")
        self.observation_dimension = observation_dimension
        self.action_dimension = action_dimension

        def make_q_network() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(observation_dimension + action_dimension, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        self.q1 = make_q_network()
        self.q2 = make_q_network()

    def forward(self, observations: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        if observations.ndim != 2 or observations.shape[1] != self.observation_dimension:
            raise ValueError("observations must have shape [batch, observation_dimension]")
        if actions.shape != (observations.shape[0], self.action_dimension):
            raise ValueError("actions must have shape [batch, action_dimension]")
        inputs = torch.cat((observations, actions), dim=-1)
        return self.q1(inputs).squeeze(-1), self.q2(inputs).squeeze(-1)
