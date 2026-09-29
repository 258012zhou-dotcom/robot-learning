"""Unit and small integration checks for the two-route navigation contract."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from robot_learning.gymnasium_rollout import run_episode
from robot_learning.two_route_navigation import (
    TwoRouteExpert,
    TwoRouteNavigationEnv,
    sample_scene,
    segment_clearance,
)


def test_scene_is_reproducible_and_route_independent() -> None:
    assert sample_scene(12) == sample_scene(12)
    assert sample_scene(12) != sample_scene(13)
    with pytest.raises(ValueError, match="seed"):
        sample_scene(-1)


def test_environment_passes_gymnasium_contract() -> None:
    check_env(TwoRouteNavigationEnv())


def test_segment_collision_catches_crossing_between_endpoints() -> None:
    center = np.zeros(2)
    assert segment_clearance(
        np.array([-1.0, 0.0]), np.array([1.0, 0.0]), center, 0.4
    ) == pytest.approx(-0.4)
    assert segment_clearance(
        np.array([-1.0, 0.6]), np.array([1.0, 0.6]), center, 0.4
    ) == pytest.approx(0.2)


def test_direct_motion_into_obstacle_terminates_without_success() -> None:
    environment = TwoRouteNavigationEnv()
    environment.reset(seed=0)
    for _ in range(30):
        _, _, terminated, truncated, info = environment.step(np.array([1.0, 0.0]))
        if terminated or truncated:
            break
    assert terminated
    assert not truncated
    assert info["collision"]
    assert not info["is_success"]


@pytest.mark.parametrize("seed", range(20))
def test_both_expert_routes_complete_same_scene(seed: int) -> None:
    environment = TwoRouteNavigationEnv()
    results = []
    for route in (-1, 1):
        expert = TwoRouteExpert(route)
        results.append(run_episode(environment, expert, seed=seed))
    assert np.array_equal(results[0].observations[0], results[1].observations[0])
    assert all(
        result.is_success and result.terminated and not result.truncated
        for result in results
    )
    obstacle_y = float(results[0].observations[0, 5])
    assert np.min(results[0].observations[:, 1] - obstacle_y) < -0.5
    assert np.max(results[1].observations[:, 1] - obstacle_y) > 0.5


def test_invalid_action_is_rejected_instead_of_silently_clipped() -> None:
    environment = TwoRouteNavigationEnv()
    environment.reset(seed=0)
    with pytest.raises(ValueError, match="velocity bounds"):
        environment.step(np.array([1.5, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        environment.step(np.array([np.nan, 0.0]))


def test_episode_limit_truncates_and_reset_starts_a_new_scene() -> None:
    environment = TwoRouteNavigationEnv(max_episode_steps=2)
    first, _ = environment.reset(seed=2)
    _, _, terminated, truncated, _ = environment.step(np.zeros(2))
    assert not terminated and not truncated
    _, _, terminated, truncated, _ = environment.step(np.zeros(2))
    assert not terminated and truncated
    with pytest.raises(RuntimeError, match="reset"):
        environment.step(np.zeros(2))
    second, _ = environment.reset(seed=3)
    assert not np.array_equal(first, second)
