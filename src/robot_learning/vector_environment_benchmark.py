"""Fair, reproducible benchmarks for single and vector Gymnasium environments."""

from dataclasses import dataclass
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
    """Measure one backend while keeping total Transition count fixed."""
    _validate_benchmark_settings(
        backend=backend,
        vector_environment_count=vector_environment_count,
        total_transition_count=total_transition_count,
        warmup_vector_steps=warmup_vector_steps,
        base_seed=base_seed,
    )
    if backend == "single":
        return _run_single_environment(
            factory=factory,
            total_transition_count=total_transition_count,
            warmup_steps=warmup_vector_steps,
            seed=base_seed,
        )
    return _run_vector_environment(
        backend=backend,
        factory=factory,
        environment_count=vector_environment_count,
        total_transition_count=total_transition_count,
        warmup_vector_steps=warmup_vector_steps,
        base_seed=base_seed,
    )


def _run_single_environment(
    *,
    factory: PointRobotBenchmarkFactory,
    total_transition_count: int,
    warmup_steps: int,
    seed: int,
) -> BenchmarkResult:
    """Time ordinary one-environment stepping without a vector wrapper."""
    initialization_start = perf_counter()
    environment = factory()
    observation, _ = environment.reset(seed=seed)
    initialization_seconds = perf_counter() - initialization_start

    try:
        for step_index in range(warmup_steps):
            action = deterministic_action_batch(step_index, 1)[0]
            observation, _, terminated, truncated, _ = environment.step(action)
            if terminated or truncated:
                raise RuntimeError("single environment ended during warmup")

        # Return to an identical initial state so warmup does not change data.
        observation, _ = environment.reset(seed=seed)
        checksum = float(np.sum(observation, dtype=np.float64))
        rollout_start = perf_counter()
        for step_index in range(total_transition_count):
            action = deterministic_action_batch(step_index, 1)[0]
            observation, reward, terminated, truncated, _ = environment.step(
                action
            )
            if terminated or truncated:
                raise RuntimeError("single environment ended during benchmark")
            checksum += float(np.sum(observation, dtype=np.float64))
            checksum += float(reward)
        rollout_seconds = perf_counter() - rollout_start
    finally:
        environment.close()

    return BenchmarkResult(
        backend="single",
        environment_count=1,
        transition_count=total_transition_count,
        vector_step_count=total_transition_count,
        initialization_seconds=initialization_seconds,
        rollout_seconds=rollout_seconds,
        transitions_per_second=total_transition_count / rollout_seconds,
        checksum=checksum,
    )


def _run_vector_environment(
    *,
    backend: Literal["sync_vector", "async_vector"],
    factory: PointRobotBenchmarkFactory,
    environment_count: int,
    total_transition_count: int,
    warmup_vector_steps: int,
    base_seed: int,
) -> BenchmarkResult:
    """Time a vector backend and account for all per-environment Transitions."""
    if total_transition_count % environment_count != 0:
        raise ValueError(
            "total_transition_count must be divisible by environment_count"
        )
    vector_step_count = total_transition_count // environment_count
    environment_factories = [factory for _ in range(environment_count)]
    seeds = [base_seed + index for index in range(environment_count)]

    initialization_start = perf_counter()
    if backend == "sync_vector":
        environments = SyncVectorEnv(environment_factories)
    else:
        environments = AsyncVectorEnv(environment_factories)
    observations, _ = environments.reset(seed=seeds)
    initialization_seconds = perf_counter() - initialization_start

    try:
        for step_index in range(warmup_vector_steps):
            actions = deterministic_action_batch(step_index, environment_count)
            observations, _, terminated, truncated, _ = environments.step(
                actions
            )
            if np.any(terminated) or np.any(truncated):
                raise RuntimeError("vector environment ended during warmup")

        observations, _ = environments.reset(seed=seeds)
        checksum = float(np.sum(observations, dtype=np.float64))
        rollout_start = perf_counter()
        for step_index in range(vector_step_count):
            actions = deterministic_action_batch(step_index, environment_count)
            observations, rewards, terminated, truncated, _ = (
                environments.step(actions)
            )
            if np.any(terminated) or np.any(truncated):
                raise RuntimeError("vector environment ended during benchmark")
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
    )


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
    if type(warmup_vector_steps) is not int or warmup_vector_steps < 0:
        raise ValueError("warmup_vector_steps must be non-negative")
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
