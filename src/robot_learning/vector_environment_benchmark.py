"""Fair, reproducible benchmarks for single and vector Gymnasium environments."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Literal

from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import numpy as np

from robot_learning.point_robot_reach_env import PointRobotReachEnv


BackendName = Literal["single", "sync_vector", "async_vector"]


@dataclass(frozen=True)
class PointRobotBenchmarkFactory:
    """Picklable recipe used by the main process and Async workers."""

    xml_path: str
    frame_skip: int
    max_episode_steps: int

    def __call__(self) -> PointRobotReachEnv:
        """Create an independent MuJoCo model and data object."""
        return PointRobotReachEnv(
            Path(self.xml_path),
            frame_skip=self.frame_skip,
            max_episode_steps=self.max_episode_steps,
            # Avoid task termination during a fixed-length performance run.
            success_tolerance=1e-12,
            velocity_tolerance=1e-12,
        )


@dataclass(frozen=True)
class BenchmarkResult:
    """One timed benchmark repetition."""

    backend: BackendName
    environment_count: int
    transition_count: int
    vector_step_count: int
    initialization_seconds: float
    rollout_seconds: float
    transitions_per_second: float
    checksum: float
    trajectory_sha256: str
    warmup_seconds: float
    reset_seconds: float


def deterministic_action_batch(
    vector_step_index: int,
    environment_count: int,
) -> np.ndarray:
    """Create bounded actions without policy-network or RNG timing noise."""
    if type(vector_step_index) is not int or vector_step_index < 0:
        raise ValueError("vector_step_index must be non-negative")
    if type(environment_count) is not int or environment_count <= 0:
        raise ValueError("environment_count must be positive")

    environment_indices = np.arange(environment_count, dtype=np.float64)
    phases = 0.017 * vector_step_index + 0.37 * environment_indices
    actions = 0.75 * np.sin(phases)
    return actions.astype(np.float32).reshape(environment_count, 1)


def run_benchmark_once(
    *,
    backend: BackendName,
    factory: PointRobotBenchmarkFactory,
    vector_environment_count: int,
    total_transition_count: int,
    warmup_vector_steps: int,
    base_seed: int,
) -> BenchmarkResult:
    """Measure identical per-environment trajectories with a shared protocol."""
    _validate_benchmark_settings(
        backend=backend,
        vector_environment_count=vector_environment_count,
        total_transition_count=total_transition_count,
        warmup_vector_steps=warmup_vector_steps,
        base_seed=base_seed,
    )
    environment_count = vector_environment_count
    vector_step_count = total_transition_count // environment_count
    seeds = [base_seed + index for index in range(environment_count)]

    initialization_start = perf_counter()
    if backend == "single":
        environments = _SerialEnvironmentPool(factory, environment_count)
    elif backend == "sync_vector":
        environments = SyncVectorEnv([factory] * environment_count)
    else:
        environments = AsyncVectorEnv([factory] * environment_count)

    try:
        environments.reset(seed=seeds)
        initialization_seconds = perf_counter() - initialization_start
        warmup_start = perf_counter()
        for step_index in range(warmup_vector_steps):
            actions = deterministic_action_batch(step_index, environment_count)
            _, _, terminated, truncated, _ = environments.step(actions)
            if np.any(terminated) or np.any(truncated):
                raise RuntimeError("environment ended during warmup")
        warmup_seconds = perf_counter() - warmup_start

        # All backends reset every environment with the original seed, untimed.
        reset_start = perf_counter()
        observations, _ = environments.reset(seed=seeds)
        reset_seconds = perf_counter() - reset_start
        digest = sha256()
        digest.update(ordered_array_bytes(observations))
        checksum = float(np.sum(observations, dtype=np.float64))
        rollout_start = perf_counter()
        for step_index in range(vector_step_count):
            actions = deterministic_action_batch(step_index, environment_count)
            observations, rewards, terminated, truncated, _ = (
                environments.step(actions)
            )
            if np.any(terminated) or np.any(truncated):
                raise RuntimeError("environment ended during benchmark")
            # Hash immediately: vector observations may reuse shared buffers.
            for values in (actions, observations, rewards, terminated, truncated):
                digest.update(ordered_array_bytes(values))
            checksum += float(np.sum(observations, dtype=np.float64))
            checksum += float(np.sum(rewards, dtype=np.float64))
        rollout_seconds = perf_counter() - rollout_start
    finally:
        environments.close()

    return BenchmarkResult(
        backend=backend,
        environment_count=environment_count,
        transition_count=total_transition_count,
        vector_step_count=vector_step_count,
        initialization_seconds=initialization_seconds,
        rollout_seconds=rollout_seconds,
        transitions_per_second=total_transition_count / rollout_seconds,
        checksum=checksum,
        trajectory_sha256=digest.hexdigest(),
        warmup_seconds=warmup_seconds,
        reset_seconds=reset_seconds,
    )


def ordered_array_bytes(values: np.ndarray) -> bytes:
    """Encode shape and C-order elements, normalizing numeric values to <f8.

    Unlike a sum, this preserves element/environment order and field boundaries.
    Float32 observations/actions convert losslessly; rewards retain float64.
    Evidence covers returned numeric data, not info or hidden simulator state.
    """
    array = np.asarray(values, dtype="<f8")
    return (
        np.asarray([array.ndim, *array.shape], dtype="<i8").tobytes()
        + array.tobytes(order="C")
    )


class _SerialEnvironmentPool:
    """N ordinary environments, stepped in index order without Gym's wrapper."""

    def __init__(
        self, factory: PointRobotBenchmarkFactory, environment_count: int
    ) -> None:
        self.environments: list[PointRobotReachEnv] = []
        try:
            for _ in range(environment_count):
                self.environments.append(factory())
        except BaseException:
            self.close()
            raise

    def reset(self, *, seed: list[int]) -> tuple[np.ndarray, dict]:
        observations = [
            environment.reset(seed=environment_seed)[0]
            for environment, environment_seed in zip(self.environments, seed)
        ]
        return np.stack(observations), {}

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        results = [
            environment.step(action)
            for environment, action in zip(self.environments, actions)
        ]
        observations, rewards, terminated, truncated, _ = zip(*results)
        return (
            np.stack(observations),
            np.asarray(rewards, dtype=np.float64),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            {},
        )

    def close(self) -> None:
        for environment in self.environments:
            environment.close()


def _validate_benchmark_settings(
    *,
    backend: str,
    vector_environment_count: int,
    total_transition_count: int,
    warmup_vector_steps: int,
    base_seed: int,
) -> None:
    """Reject benchmark settings that would produce a misleading comparison."""
    if backend not in {"single", "sync_vector", "async_vector"}:
        raise ValueError(f"unknown benchmark backend: {backend}")
    if type(vector_environment_count) is not int or vector_environment_count <= 0:
        raise ValueError("vector_environment_count must be positive")
    if type(total_transition_count) is not int or total_transition_count <= 0:
        raise ValueError("total_transition_count must be positive")
    if total_transition_count % vector_environment_count != 0:
        raise ValueError(
            "total_transition_count must be divisible by environment_count"
        )
    if type(warmup_vector_steps) is not int or warmup_vector_steps < 0:
        raise ValueError("warmup_vector_steps must be non-negative")
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
