"""Matched target-distance schedules for a SAC direction-coverage comparison."""

import gymnasium as gym
import numpy as np

from robot_learning.point_robot_reach_env import PointRobotReachEnv


class ScheduledTargetEnvironment(gym.Wrapper):
    """Keep target magnitudes fixed while choosing the sign by experiment arm."""

    def __init__(
        self,
        environment: PointRobotReachEnv,
        *,
        target_magnitudes: tuple[float, ...],
        alternate_direction: bool,
    ) -> None:
        super().__init__(environment)
        if not target_magnitudes or any(
            not np.isfinite(magnitude) or magnitude <= 0.0
            for magnitude in target_magnitudes
        ):
            raise ValueError("target magnitudes must be positive and finite")
        if any(
            not environment.minimum_target_distance <= magnitude <= environment.maximum_target_distance
            for magnitude in target_magnitudes
        ):
            raise ValueError("target magnitudes must fit the environment range")
        self.target_magnitudes = target_magnitudes
        self.alternate_direction = alternate_direction
        self.scheduled_targets: list[float] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if options and "target_position" in options:
            raise ValueError("target position is controlled by the schedule")
        index = len(self.scheduled_targets)
        magnitude = self.target_magnitudes[index % len(self.target_magnitudes)]
        direction = -1.0 if self.alternate_direction and index % 2 else 1.0
        target = direction * magnitude
        observation, info = self.env.reset(
            seed=seed, options={**(options or {}), "target_position": target},
        )
        self.scheduled_targets.append(target)
        return observation, info
