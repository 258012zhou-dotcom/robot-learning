"""Small on-policy PPO loop for discrete actions in a Gymnasium environment."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor, nn

from robot_learning.ppo_clipping import ppo_clipped_loss


ACTION_VALUES = (-1.0, 0.0, 1.0)


def near_target_speed_cost(observation: np.ndarray, radius: float) -> float:
    """Penalize speed only near the target, using the next observation."""
    observation_array = np.asarray(observation, dtype=np.float64)
    if observation_array.shape != (4,) or not np.all(np.isfinite(observation_array)):
        raise ValueError("observation must be one finite four-vector")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    distance = abs(float(observation_array[3]))
    speed = abs(float(observation_array[1]))
    proximity = max(0.0, 1.0 - distance / radius)
    return proximity * speed


class DiscreteActorCritic(nn.Module):
    """Separate actor and critic networks for four-dimensional observations."""

    def __init__(self, observation_dimension: int = 4, hidden_size: int = 32) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_dimension, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, len(ACTION_VALUES)),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_dimension, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def distribution(self, observations: Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.actor(observations))

    def value(self, observations: Tensor) -> Tensor:
        return self.critic(observations).squeeze(-1)


def generalized_advantages(
    rewards: Tensor,
    values: Tensor,
    final_value: Tensor,
    *,
    terminated: bool,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute GAE and fixed value targets for one complete Episode.

    Natural termination has no future value; time-limit truncation bootstraps
    from the Critic's value at the final observation.
    """
    if rewards.ndim != 1 or rewards.numel() == 0 or rewards.shape != values.shape:
        raise ValueError("rewards and values must be equal non-empty vectors")
    if final_value.numel() != 1:
        raise ValueError("final_value must be scalar")
    if not (0.0 <= gamma <= 1.0 and 0.0 <= gae_lambda <= 1.0):
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    if not all(torch.isfinite(item).all() for item in (rewards, values, final_value)):
        raise ValueError("GAE inputs must be finite")

    fixed_rewards = rewards.detach()
    fixed_values = values.detach()
    following_value = torch.zeros_like(final_value) if terminated else final_value.detach()
    following_advantage = torch.zeros_like(following_value)
    advantages = torch.empty_like(fixed_rewards)
    for index in range(rewards.numel() - 1, -1, -1):
        delta = fixed_rewards[index] + gamma * following_value - fixed_values[index]
        following_advantage = delta + gamma * gae_lambda * following_advantage
        advantages[index] = following_advantage
        following_value = fixed_values[index]
    return advantages, advantages + fixed_values


@dataclass(frozen=True)
class OnPolicyBatch:
    observations: Tensor
    actions: Tensor
    old_log_probabilities: Tensor
    advantages: Tensor
    value_targets: Tensor
    episode_records: list[dict[str, float | int | bool]]


def collect_episodes(
    environment: gym.Env,
    model: DiscreteActorCritic,
    *,
    episode_seeds: range,
    gamma: float,
    gae_lambda: float,
    reward_scale: float,
    velocity_penalty_weight: float = 0.0,
    near_target_radius: float = 0.5,
) -> OnPolicyBatch:
    """Collect complete Episodes with the current frozen policy."""
    if reward_scale <= 0.0:
        raise ValueError("reward_scale must be positive")
    if not np.isfinite(velocity_penalty_weight) or velocity_penalty_weight < 0.0:
        raise ValueError("velocity_penalty_weight must be non-negative and finite")
    if not np.isfinite(near_target_radius) or near_target_radius <= 0.0:
        raise ValueError("near_target_radius must be positive and finite")
    observations: list[Tensor] = []
    actions: list[int] = []
    old_log_probabilities: list[Tensor] = []
    advantages: list[Tensor] = []
    value_targets: list[Tensor] = []
    records: list[dict[str, float | int | bool]] = []
    model.eval()
    for seed in episode_seeds:
        observation, _ = environment.reset(seed=seed)
        episode_observations: list[Tensor] = []
        episode_actions: list[int] = []
        episode_log_probabilities: list[Tensor] = []
        episode_values: list[Tensor] = []
        episode_rewards: list[float] = []
        total_reward = 0.0
        total_speed_penalty = 0.0
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
            with torch.no_grad():
                distribution = model.distribution(observation_tensor)
                action_index = distribution.sample()
                log_probability = distribution.log_prob(action_index)
                value = model.value(observation_tensor)
            action = np.asarray([ACTION_VALUES[int(action_index)]], dtype=np.float32)
            next_observation, reward, terminated, truncated, info = environment.step(action)
            episode_observations.append(observation_tensor)
            episode_actions.append(int(action_index))
            episode_log_probabilities.append(log_probability)
            episode_values.append(value)
            speed_penalty = (
                velocity_penalty_weight
                * near_target_speed_cost(next_observation, near_target_radius)
                if velocity_penalty_weight > 0.0 else 0.0
            )
            episode_rewards.append((float(reward) - speed_penalty) * reward_scale)
            total_reward += float(reward)
            total_speed_penalty += speed_penalty
            observation = next_observation
        with torch.no_grad():
            final_value = model.value(torch.as_tensor(observation, dtype=torch.float32))
        episode_advantages, episode_targets = generalized_advantages(
            torch.tensor(episode_rewards, dtype=torch.float32),
            torch.stack(episode_values),
            final_value,
            terminated=terminated,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        observations.extend(episode_observations)
        actions.extend(episode_actions)
        old_log_probabilities.extend(episode_log_probabilities)
        advantages.append(episode_advantages)
        value_targets.append(episode_targets)
        records.append(
            {
                "seed": seed,
                "target_position": float(info["target_position"]),
                "success": bool(info["is_success"]),
                "terminated": terminated,
                "truncated": truncated,
                "steps": len(episode_rewards),
                "total_reward": total_reward,
                "total_speed_penalty": total_speed_penalty,
                "training_reward_scaled": float(sum(episode_rewards)),
            }
        )
    if not observations:
        raise ValueError("episode_seeds must not be empty")
    return OnPolicyBatch(
        observations=torch.stack(observations),
        actions=torch.tensor(actions, dtype=torch.long),
        old_log_probabilities=torch.stack(old_log_probabilities),
        advantages=torch.cat(advantages),
        value_targets=torch.cat(value_targets),
        episode_records=records,
    )


def update_from_batch(
    model: DiscreteActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: OnPolicyBatch,
    *,
    epochs: int,
    clip_epsilon: float,
    value_loss_weight: float,
    entropy_weight: float,
) -> dict[str, float]:
    """Reuse the collected batch, then discard it before the next collection."""
    if epochs <= 0 or value_loss_weight < 0 or entropy_weight < 0:
        raise ValueError("epochs must be positive and loss weights non-negative")
    fixed_advantages = batch.advantages.detach()
    fixed_advantages = (
        fixed_advantages - fixed_advantages.mean()
    ) / fixed_advantages.std(unbiased=False).clamp_min(1e-8)
    model.train()
    metrics: dict[str, float] = {}
    for _ in range(epochs):
        distribution = model.distribution(batch.observations)
        log_probabilities = distribution.log_prob(batch.actions)
        policy_loss = ppo_clipped_loss(
            log_probabilities,
            batch.old_log_probabilities,
            fixed_advantages,
            clip_epsilon,
        )
        value_loss = torch.square(
            model.value(batch.observations) - batch.value_targets.detach()
        ).mean()
        entropy = distribution.entropy().mean()
        total_loss = policy_loss + value_loss_weight * value_loss - entropy_weight * entropy
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        metrics = {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()),
        }
    return metrics
