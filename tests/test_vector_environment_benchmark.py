"""Semantic tests for single, synchronous, and asynchronous benchmarks."""

from pathlib import Path
from dataclasses import replace
from hashlib import sha256
import importlib.util

import numpy as np
import pytest

from robot_learning.vector_environment_benchmark import (
    deterministic_action_batch,
    ordered_array_bytes,
    PointRobotBenchmarkFactory,
    run_benchmark_once,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "017_mujoco_step"
    / "point_robot.xml"
)


def make_factory() -> PointRobotBenchmarkFactory:
    """Use a long Episode so fixed-length timing never triggers autoreset."""
    return PointRobotBenchmarkFactory(
        xml_path=str(MODEL_PATH),
        frame_skip=2,
        max_episode_steps=10_000,
    )


def test_deterministic_action_batch_has_vector_shape_and_bounds() -> None:
    """One action row is required for every environment in the batch."""
    first = deterministic_action_batch(10, 4)
    second = deterministic_action_batch(10, 4)

    assert first.shape == (4, 1)
    assert first.dtype == np.float32
    assert np.all(np.abs(first) <= 0.75)
    np.testing.assert_array_equal(first, second)


def test_picklable_factory_creates_independent_environment() -> None:
    """The recipe used by workers should construct a valid environment."""
    environment = make_factory()()
    try:
        observation, _ = environment.reset(seed=1)
        assert observation.shape == (4,)
    finally:
        environment.close()


def test_single_benchmark_counts_every_transition() -> None:
    """The serial pool must execute the same N short trajectories as vectors."""
    result = run_benchmark_once(
        backend="single",
        factory=make_factory(),
        vector_environment_count=2,
        total_transition_count=40,
        warmup_vector_steps=2,
        base_seed=10,
    )

    assert result.environment_count == 2
    assert result.transition_count == 40
    assert result.vector_step_count == 20
    assert result.rollout_seconds > 0.0
    assert result.transitions_per_second > 0.0
    assert np.isfinite(result.checksum)
    assert len(result.trajectory_sha256) == 64
    assert result.warmup_seconds >= 0
    assert result.reset_seconds > 0


def test_sync_vector_benchmark_is_reproducible() -> None:
    """Timing may vary, but fixed seeds and actions should preserve data values."""
    settings = {
        "backend": "sync_vector",
        "factory": make_factory(),
        "vector_environment_count": 2,
        "total_transition_count": 40,
        "warmup_vector_steps": 2,
        "base_seed": 20,
    }

    first = run_benchmark_once(**settings)
    second = run_benchmark_once(**settings)

    assert first.transition_count == second.transition_count == 40
    assert first.vector_step_count == second.vector_step_count == 20
    assert first.trajectory_sha256 == second.trajectory_sha256


def test_sync_and_async_backends_preserve_identical_numerical_results() -> None:
    """Changing execution backend must not change observations or rewards."""
    settings = {
        "factory": make_factory(),
        "vector_environment_count": 2,
        "total_transition_count": 40,
        "warmup_vector_steps": 2,
        "base_seed": 30,
    }

    synchronous = run_benchmark_once(backend="sync_vector", **settings)
    asynchronous = run_benchmark_once(backend="async_vector", **settings)
    serial = run_benchmark_once(backend="single", **settings)

    assert synchronous.transition_count == asynchronous.transition_count == 40
    assert serial.environment_count == synchronous.environment_count == 2
    assert serial.vector_step_count == asynchronous.vector_step_count == 20
    assert (
        serial.trajectory_sha256
        == synchronous.trajectory_sha256
        == asynchronous.trajectory_sha256
    )


@pytest.mark.parametrize("backend", ["single", "sync_vector", "async_vector"])
def test_vector_benchmark_requires_divisible_transition_count(backend) -> None:
    """Unequal work per backend should be rejected before timing starts."""
    with pytest.raises(ValueError, match="divisible"):
        run_benchmark_once(
            backend=backend,
            factory=make_factory(),
            vector_environment_count=4,
            total_transition_count=41,
            warmup_vector_steps=0,
            base_seed=40,
        )


@pytest.mark.parametrize("changed", [
    np.asarray([[2., 1.], [3., 4.]]),  # Component permutation.
    np.asarray([[3., 4.], [1., 2.]]),  # Environment permutation.
    np.asarray([[0., 3.], [3., 4.]]),  # Compensating value changes.
    np.asarray([1., 2., 3., 4.]),      # Same values, different shape.
])
def test_ordered_hash_rejects_equal_sum_arrays(changed) -> None:
    original = np.asarray([[1., 2.], [3., 4.]])
    assert original.sum() == changed.sum()  # Old evidence would accept this.
    assert sha256(ordered_array_bytes(original)).digest() != sha256(
        ordered_array_bytes(changed)
    ).digest()


@pytest.mark.parametrize("backend", ["single", "sync_vector", "async_vector"])
def test_warmup_does_not_change_timed_trajectory(backend) -> None:
    settings = dict(
        backend=backend, factory=make_factory(), vector_environment_count=3,
        total_transition_count=18, base_seed=51,
    )
    cold = run_benchmark_once(warmup_vector_steps=0, **settings)
    warm = run_benchmark_once(warmup_vector_steps=9, **settings)
    assert cold.trajectory_sha256 == warm.trajectory_sha256


def test_serial_pool_receives_per_environment_seeds_actions_and_steps() -> None:
    from robot_learning.point_robot_reach_env import PointRobotReachEnv

    created = []

    class RecordingEnvironment(PointRobotReachEnv):
        def __init__(self):
            super().__init__(MODEL_PATH, max_episode_steps=100)
            self.reset_seeds = []
            self.action_phases = []
            self.was_closed = False

        def reset(self, *, seed=None, options=None):
            self.reset_seeds.append(seed)
            self.action_phases.append([])
            return super().reset(seed=seed, options=options)

        def step(self, action):
            self.action_phases[-1].append(action.copy())
            return super().step(action)

        def close(self):
            self.was_closed = True
            super().close()

    def factory():
        environment = RecordingEnvironment()
        created.append(environment)
        return environment

    run_benchmark_once(
        backend="single", factory=factory, vector_environment_count=3,
        total_transition_count=12, warmup_vector_steps=2, base_seed=71,
    )
    assert len(created) == 3
    for index, environment in enumerate(created):
        assert environment.reset_seeds == [71 + index, 71 + index]
        assert list(map(len, environment.action_phases)) == [2, 4]
        for phase in environment.action_phases:
            for step, action in enumerate(phase):
                np.testing.assert_array_equal(
                    action, deterministic_action_batch(step, 3)[index]
                )
        assert environment.was_closed


def test_report_rejects_hash_mismatch_even_when_checksums_match() -> None:
    spec = importlib.util.spec_from_file_location(
        "benchmark_run",
        PROJECT_ROOT / "experiments/023_vector_environment_benchmark/run.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = run_benchmark_once(
        backend="single", factory=make_factory(), vector_environment_count=2,
        total_transition_count=4, warmup_vector_steps=0, base_seed=81,
    )
    records = {"small": {
        backend: [replace(baseline, backend=backend)] for backend in module.BACKENDS
    }}
    assert all(module.verify_reproducibility(records)["small"].values())
    # Sum remains identical; changing serial evidence must invalidate comparison.
    records["small"]["single"].append(
        replace(baseline, trajectory_sha256="0" * 64)
    )
    checks = module.verify_reproducibility(records)["small"]
    assert not checks["single_repeated_trajectory_match"]
    assert not checks["all_backends_trajectory_match"]
