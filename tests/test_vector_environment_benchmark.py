"""Semantic tests for single, synchronous, and asynchronous benchmarks."""

from pathlib import Path

import numpy as np
import pytest

from robot_learning.vector_environment_benchmark import (
    deterministic_action_batch,
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
    """A single environment should execute one Transition per ordinary step."""
    result = run_benchmark_once(
        backend="single",
        factory=make_factory(),
        vector_environment_count=2,
        total_transition_count=40,
        warmup_vector_steps=2,
        base_seed=10,
    )

    assert result.environment_count == 1
    assert result.transition_count == 40
    assert result.vector_step_count == 40
    assert result.rollout_seconds > 0.0
    assert result.transitions_per_second > 0.0
    assert np.isfinite(result.checksum)


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
    assert first.checksum == pytest.approx(second.checksum, rel=0.0, abs=1e-12)


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

    assert synchronous.transition_count == asynchronous.transition_count == 40
    assert synchronous.checksum == pytest.approx(
        asynchronous.checksum,
        rel=0.0,
        abs=1e-12,
    )


def test_vector_benchmark_requires_divisible_transition_count() -> None:
    """Unequal work per backend should be rejected before timing starts."""
    with pytest.raises(ValueError, match="divisible"):
        run_benchmark_once(
            backend="sync_vector",
            factory=make_factory(),
            vector_environment_count=4,
            total_transition_count=41,
            warmup_vector_steps=0,
            base_seed=40,
        )
