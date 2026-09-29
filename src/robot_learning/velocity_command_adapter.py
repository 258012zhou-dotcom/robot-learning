"""A narrow teaching adapter for checking velocity command units."""

from typing import Protocol

import numpy as np


class RouteVelocityPolicy(Protocol):
    def reset(self, *, route: int) -> None: ...

    def __call__(self, observation: np.ndarray) -> np.ndarray: ...


class VelocityCommandAdapter:
    """Pass velocities through, or deliberately double-apply dt for diagnosis."""

    def __init__(self, policy: RouteVelocityPolicy, *, scale: float) -> None:
        if not np.isfinite(scale) or not 0 < scale <= 1:
            raise ValueError("scale must be finite and within (0, 1]")
        self.policy = policy
        self.scale = float(scale)
        self.raw_commands: list[np.ndarray] = []

    def reset(self, *, route: int) -> None:
        self.policy.reset(route=route)
        self.raw_commands = []

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.policy(observation), dtype=np.float32)
        if raw.shape != (2,) or not np.all(np.isfinite(raw)):
            raise ValueError("route policy must produce finite 2D velocity")
        self.raw_commands.append(raw.copy())
        return (raw * self.scale).astype(np.float32)
