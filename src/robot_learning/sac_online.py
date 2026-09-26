"""Collect point-robot transitions and run small online SAC updates."""

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from robot_learning.sac_dataflow import ReplayBatch
from robot_learning.sac_models import SquashedGaussianActor, TwinQCritic
from robot_learning.sac_training import sac_update


@dataclass(frozen=True)
class OnlineTransition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool


class OnlineReplayBuffer:
    """Bounded storage of newly collected transitions, including episode ends."""

    def __init__(self, capacity: int, observation_dimension: int = 4) -> None:
        if capacity <= 0 or observation_dimension <= 0:
            raise ValueError("capacity and observation dimension must be positive")
        self.observation_dimension = observation_dimension
        self.transitions: deque[OnlineTransition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.transitions)

    def append(
        self, observation: np.ndarray, action: np.ndarray, reward: float,
        next_observation: np.ndarray, terminated: bool, truncated: bool,
    ) -> None:
        observation = np.asarray(observation, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_observation = np.asarray(next_observation, dtype=np.float32)
        if observation.shape != (self.observation_dimension,) or next_observation.shape != observation.shape:
            raise ValueError("observations have the wrong shape")
        if action.shape != (1,):
            raise ValueError("action must have shape (1,)")
        if not all(np.all(np.isfinite(item)) for item in (observation, action, next_observation)):
            raise ValueError("transition contains non-finite values")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        if terminated and truncated:
            raise ValueError("transition cannot terminate and truncate together")
        self.transitions.append(OnlineTransition(
            observation.copy(), action.copy(), float(reward), next_observation.copy(),
            bool(terminated), bool(truncated),
        ))

    def sample(self, batch_size: int, rng: np.random.Generator) -> ReplayBatch:
        if batch_size <= 0 or not self.transitions:
            raise ValueError("batch size must be positive and replay non-empty")
        indices = rng.integers(len(self.transitions), size=batch_size)
        sampled = [self.transitions[int(index)] for index in indices]
        return ReplayBatch(
            indices=indices,
            observations=np.stack([item.observation for item in sampled]),
            actions=np.stack([item.action for item in sampled]),
            rewards=np.asarray([item.reward for item in sampled], dtype=np.float32),
            next_observations=np.stack([item.next_observation for item in sampled]),
            terminated=np.asarray([item.terminated for item in sampled], dtype=np.bool_),
            truncated=np.asarray([item.truncated for item in sampled], dtype=np.bool_),
        )


@dataclass(frozen=True)
class OnlineRunSummary:
    environment_steps: int
    gradient_updates: int
    completed_episodes: int
    replay_size: int
    last_critic_loss: float | None
    last_actor_loss: float | None


def run_online_steps(
    environment: gym.Env,
    actor: SquashedGaussianActor,
    critic: TwinQCritic,
    target_critic: TwinQCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    replay: OnlineReplayBuffer,
    *,
    steps: int,
    random_steps: int,
    batch_size: int,
    seed: int,
    rng: np.random.Generator,
    gamma: float,
    alpha: float,
    tau: float,
) -> OnlineRunSummary:
    """Collect exactly `steps` transitions; update once per step after warmup."""
    if steps <= 0 or random_steps < 0 or batch_size <= 0:
        raise ValueError("steps and batch size must be positive; random steps non-negative")
    if not isinstance(environment.action_space, gym.spaces.Box) or (
        environment.action_space.shape != (1,)
        or not np.array_equal(environment.action_space.low, [-1.0])
        or not np.array_equal(environment.action_space.high, [1.0])
    ):
        raise ValueError("environment must have one action bounded by [-1, 1]")
    if replay.observation_dimension != actor.observation_dimension:
        raise ValueError("replay and actor observation dimensions differ")

    observation, _ = environment.reset(seed=seed)
    device = next(actor.parameters()).device
    completed_episodes = gradient_updates = 0
    last_critic_loss = last_actor_loss = None
    for step in range(steps):
        if step < random_steps:
            action = rng.uniform(-1.0, 1.0, size=1).astype(np.float32)
        else:
            with torch.no_grad():
                observation_tensor = torch.as_tensor(
                    observation, dtype=torch.float32, device=device,
                ).unsqueeze(0)
                sampled_action, _ = actor.sample(observation_tensor)
                action = sampled_action.squeeze(0).cpu().numpy()

        next_observation, reward, terminated, truncated, _ = environment.step(action)
        replay.append(observation, action, reward, next_observation, terminated, truncated)
        if step >= random_steps and len(replay) >= batch_size:
            metrics = sac_update(
                actor, critic, target_critic, actor_optimizer, critic_optimizer,
                replay.sample(batch_size, rng), gamma=gamma, alpha=alpha, tau=tau,
            )
            gradient_updates += 1
            last_critic_loss = metrics.critic_loss
            last_actor_loss = metrics.actor_loss

        observation = next_observation
        if terminated or truncated:
            completed_episodes += 1
            if step + 1 < steps:
                observation, _ = environment.reset(seed=seed + completed_episodes)

    return OnlineRunSummary(
        environment_steps=steps,
        gradient_updates=gradient_updates,
        completed_episodes=completed_episodes,
        replay_size=len(replay),
        last_critic_loss=last_critic_loss,
        last_actor_loss=last_actor_loss,
    )
