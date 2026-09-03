"""Policy and rollout utilities for readable Gymnasium experiments.

The environment owns physics and task rules.  A policy only maps the current
observation to the next action.  ``run_episode`` connects those two roles and
records evidence without knowing MuJoCo internals.
"""

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Protocol

import gymnasium as gym
import numpy as np


class Policy(Protocol):
    """Callable interface shared by random and hand-designed policies."""

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Return one action for the current observation."""
        ...


@dataclass(frozen=True)
class EpisodeResult:
    """One complete environment trajectory.

    Shape contract for an Episode containing ``T`` actions:

    - ``observations``: ``(T + 1, observation_dimension)``
    - ``actions``: ``(T, action_dimension)``
    - ``rewards``: ``(T,)``
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    total_reward: float
    terminated: bool
    truncated: bool
    is_success: bool

    @property
    def step_count(self) -> int:
        """Return the number of actions executed in this Episode."""
        return int(self.actions.shape[0])


@dataclass(frozen=True)
class PolicyEvaluation:
    """Aggregate metrics calculated from several complete Episodes."""

    episode_count: int
    success_count: int
    success_rate: float
    mean_total_reward: float
    mean_step_count: float
    mean_final_distance: float


class RandomPolicy:
    """Sample uniformly inside a continuous Gymnasium action space."""

    def __init__(self, action_space: gym.spaces.Box, seed: int) -> None:
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("RandomPolicy requires a Box action space")
        self._low = action_space.low.astype(np.float32, copy=True)
        self._high = action_space.high.astype(np.float32, copy=True)
        self._rng = np.random.default_rng(seed)

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Ignore the observation and sample one bounded random action."""
        del observation
        return self._rng.uniform(self._low, self._high).astype(np.float32)


class ProportionalReachPolicy:
    """Convert target position error into a bounded motor command."""

    def __init__(
        self,
        gain: float,
        action_space: gym.spaces.Box,
        error_observation_index: int = 3,
    ) -> None:
        if gain <= 0.0:
            raise ValueError("gain must be positive")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("action_space must be a Box")
        if action_space.shape != (1,):
            raise ValueError("policy expects one-dimensional actions")
        self.gain = float(gain)
        self._low = float(action_space.low[0])
        self._high = float(action_space.high[0])
        self._error_index = error_observation_index

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Return ``clip(gain * target_error)`` as a one-element action."""
        if observation.ndim != 1:
            raise ValueError("observation must be one-dimensional")
        target_error = float(observation[self._error_index])
        action = np.clip(self.gain * target_error, self._low, self._high)
        return np.asarray([action], dtype=np.float32)


class ProportionalDerivativeReachPolicy:
    """Use target error and measured velocity to produce a bounded action."""

    def __init__(
        self,
        proportional_gain: float,
        derivative_gain: float,
        action_space: gym.spaces.Box,
        *,
        error_observation_index: int = 3,
        velocity_observation_index: int = 1,
    ) -> None:
        if proportional_gain <= 0.0:
            raise ValueError("proportional_gain must be positive")
        if derivative_gain < 0.0:
            raise ValueError("derivative_gain must be non-negative")
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("action_space must be a Box")
        if action_space.shape != (1,):
            raise ValueError("policy expects one-dimensional actions")
        self.proportional_gain = float(proportional_gain)
        self.derivative_gain = float(derivative_gain)
        self._low = float(action_space.low[0])
        self._high = float(action_space.high[0])
        self._error_index = error_observation_index
        self._velocity_index = velocity_observation_index

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Return ``clip(Kp * error - Kd * velocity)``."""
        if observation.ndim != 1:
            raise ValueError("observation must be one-dimensional")
        target_error = float(observation[self._error_index])
        velocity = float(observation[self._velocity_index])
        action = (
            self.proportional_gain * target_error
            - self.derivative_gain * velocity
        )
        clipped_action = np.clip(action, self._low, self._high)
        return np.asarray([clipped_action], dtype=np.float32)


def run_episode(
    environment: gym.Env,
    policy: Policy,
    *,
    seed: int,
    reset_options: dict[str, Any] | None = None,
) -> EpisodeResult:
    """Run from reset until the environment terminates or truncates."""
    observation, info = environment.reset(seed=seed, options=reset_options)
    observations = [np.asarray(observation).copy()]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # The policy always receives the newest observation from the last step.
        action = np.asarray(policy(observation), dtype=np.float32)
        observation, reward, terminated, truncated, info = environment.step(
            action
        )
        observations.append(np.asarray(observation).copy())
        actions.append(action.copy())
        rewards.append(float(reward))

    action_dimension = int(np.prod(environment.action_space.shape))
    action_array = (
        np.stack(actions)
        if actions
        else np.empty((0, action_dimension), dtype=np.float32)
    )
    reward_array = np.asarray(rewards, dtype=np.float64)
    return EpisodeResult(
        observations=np.stack(observations),
        actions=action_array,
        rewards=reward_array,
        total_reward=float(reward_array.sum()),
        terminated=bool(terminated),
        truncated=bool(truncated),
        is_success=bool(info.get("is_success", False)),
    )


def summarize_episodes(
    results: Sequence[EpisodeResult],
    *,
    error_observation_index: int = 3,
) -> PolicyEvaluation:
    """Summarize policy performance without hiding per-Episode evidence."""
    if not results:
        raise ValueError("results must contain at least one Episode")

    success_count = sum(result.is_success for result in results)
    final_distances = [
        abs(float(result.observations[-1, error_observation_index]))
        for result in results
    ]
    return PolicyEvaluation(
        episode_count=len(results),
        success_count=success_count,
        success_rate=success_count / len(results),
        mean_total_reward=float(
            np.mean([result.total_reward for result in results])
        ),
        mean_step_count=float(
            np.mean([result.step_count for result in results])
        ),
        mean_final_distance=float(np.mean(final_distances)),
    )


def calculate_position_overshoot(
    observations: np.ndarray,
    *,
    position_index: int = 0,
    target_index: int = 2,
) -> float:
    """Return the farthest distance travelled beyond a fixed target.

    Multiplying by the target direction turns leftward and rightward tasks into
    the same positive progress axis.  This prevents negative targets from being
    handled with a separate, easy-to-get-wrong formula.
    """
    observation_array = np.asarray(observations)
    if observation_array.ndim != 2 or observation_array.shape[0] == 0:
        raise ValueError("observations must have shape (T, observation_dimension)")

    initial_position = float(observation_array[0, position_index])
    target_position = float(observation_array[0, target_index])
    target_displacement = target_position - initial_position
    if target_displacement == 0.0:
        return 0.0

    direction = float(np.sign(target_displacement))
    progress = direction * (
        observation_array[:, position_index] - initial_position
    )
    overshoot = np.max(progress) - abs(target_displacement)
    return max(0.0, float(overshoot))
