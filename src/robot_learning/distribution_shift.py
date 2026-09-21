"""Controlled rollout disturbances for studying policy recovery."""

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from robot_learning.gymnasium_rollout import EpisodeResult, Policy


@dataclass(frozen=True)
class ActionOverrideResult:
    """One Episode plus evidence about the applied action override."""

    episode: EpisodeResult
    override_applied_steps: int


@dataclass(frozen=True)
class VelocitySweepResult:
    """Policy actions for one observation while only velocity changes."""

    velocities: np.ndarray
    actions: np.ndarray


def evaluate_velocity_sweep(
    policy: Policy,
    base_observation: np.ndarray,
    velocities: np.ndarray,
    *,
    velocity_index: int = 1,
) -> VelocitySweepResult:
    """Hold every observation feature fixed except measured velocity."""
    observation = np.asarray(base_observation, dtype=np.float32)
    velocity_values = np.asarray(velocities, dtype=np.float32)
    if observation.ndim != 1:
        raise ValueError("base_observation must be one-dimensional")
    if type(velocity_index) is not int or not 0 <= velocity_index < observation.size:
        raise ValueError("velocity_index is outside the observation")
    if velocity_values.ndim != 1 or velocity_values.size == 0:
        raise ValueError("velocities must be a non-empty vector")
    if not np.all(np.isfinite(observation)) or not np.all(
        np.isfinite(velocity_values)
    ):
        raise ValueError("observation and velocities must be finite")

    actions = []
    for velocity in velocity_values:
        current = observation.copy()
        current[velocity_index] = velocity
        action = np.asarray(policy(current), dtype=np.float32)
        if action.ndim != 1 or not np.all(np.isfinite(action)):
            raise ValueError("policy must return one finite action vector")
        actions.append(action)
    return VelocitySweepResult(
        velocities=velocity_values.copy(),
        actions=np.stack(actions),
    )


def run_episode_with_target_directed_override(
    environment: gym.Env,
    policy: Policy,
    *,
    seed: int,
    override_start_step: int,
    override_duration_steps: int,
    override_magnitude: float,
    reset_options: dict[str, Any] | None = None,
) -> ActionOverrideResult:
    """Temporarily replace policy actions with a command toward the target."""
    if type(override_start_step) is not int or override_start_step < 0:
        raise ValueError("override_start_step must be a non-negative integer")
    if type(override_duration_steps) is not int or override_duration_steps < 0:
        raise ValueError(
            "override_duration_steps must be a non-negative integer"
        )
    if not np.isfinite(override_magnitude) or override_magnitude <= 0.0:
        raise ValueError("override_magnitude must be positive and finite")
    if not isinstance(environment.action_space, gym.spaces.Box):
        raise TypeError("environment action_space must be a Box")
    if environment.action_space.shape != (1,):
        raise ValueError("target-directed override requires one action dimension")

    observation, info = environment.reset(seed=seed, options=reset_options)
    target_displacement = float(observation[2] - observation[0])
    if target_displacement == 0.0:
        raise ValueError("target position must differ from initial position")
    override_action = np.asarray(
        [np.sign(target_displacement) * override_magnitude],
        dtype=np.float32,
    )
    if not environment.action_space.contains(override_action):
        raise ValueError("override action exceeds the environment action space")

    observations = [np.asarray(observation).copy()]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated = False
    truncated = False
    elapsed_steps = 0
    override_applied_steps = 0
    override_end_step = override_start_step + override_duration_steps

    while not (terminated or truncated):
        action = np.asarray(policy(observation), dtype=np.float32)
        if override_start_step <= elapsed_steps < override_end_step:
            action = override_action.copy()
            override_applied_steps += 1
        observation, reward, terminated, truncated, info = environment.step(
            action
        )
        observations.append(np.asarray(observation).copy())
        actions.append(action.copy())
        rewards.append(float(reward))
        elapsed_steps += 1

    reward_array = np.asarray(rewards, dtype=np.float64)
    action_array = np.stack(actions)
    episode = EpisodeResult(
        observations=np.stack(observations),
        actions=action_array,
        rewards=reward_array,
        total_reward=float(reward_array.sum()),
        terminated=bool(terminated),
        truncated=bool(truncated),
        is_success=bool(info.get("is_success", False)),
    )
    return ActionOverrideResult(
        episode=episode,
        override_applied_steps=override_applied_steps,
    )
