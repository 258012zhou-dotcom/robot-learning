"""Tests for interpreting position, speed, and overshoot in reach traces."""

import numpy as np
import pytest

from robot_learning.gymnasium_rollout import EpisodeResult
from robot_learning.reach_trajectory_analysis import summarize_reach_trajectory


def make_episode(
    positions: list[float], velocities: list[float], target: float,
    actions: list[float] | None = None,
) -> EpisodeResult:
    observations = np.column_stack(
        (
            positions,
            velocities,
            np.full(len(positions), target),
            target - np.asarray(positions),
        )
    ).astype(np.float32)
    step_count = len(positions) - 1
    return EpisodeResult(
        observations=observations,
        actions=np.asarray(
            actions if actions is not None else [0.0] * step_count,
            dtype=np.float32,
        ).reshape(-1, 1),
        rewards=np.zeros(step_count),
        total_reward=0.0,
        terminated=False,
        truncated=True,
        is_success=False,
    )


def test_positive_target_crossing_and_fast_near_pass() -> None:
    episode = make_episode(
        [0.0, 0.8, 1.2, 1.5], [0.0, 0.8, 0.6, 0.1], 1.0,
        [1.0, 1.0, -1.0],
    )

    result = summarize_reach_trajectory(
        episode, position_tolerance=0.25, velocity_tolerance=0.1
    )

    assert result["first_near_step"] == 1
    assert result["speed_at_first_near"] == pytest.approx(0.8)
    assert result["first_crossing_step"] == 2
    assert result["speed_at_first_crossing"] == pytest.approx(0.6)
    assert result["maximum_overshoot"] == pytest.approx(0.5)
    assert result["first_opposing_action_step"] == 2
    assert result["opposing_actions_before_crossing"] == 0
    assert result["near_and_slow_steps"] == 0


def test_negative_target_uses_same_overshoot_direction() -> None:
    episode = make_episode(
        [0.0, -0.9, -1.1], [0.0, -0.4, -0.2], -1.0, [-1.0, 1.0]
    )

    result = summarize_reach_trajectory(
        episode, position_tolerance=0.15, velocity_tolerance=0.1
    )

    assert result["first_near_step"] == 1
    assert result["first_crossing_step"] == 2
    assert result["maximum_overshoot"] == pytest.approx(0.1)
    assert result["first_opposing_action_step"] == 1
    assert result["opposing_actions_before_crossing"] == 1
    assert result["near_and_slow_steps"] == 0


def test_trace_without_target_contact_reports_none() -> None:
    episode = make_episode([0.0, 0.1, 0.2], [0.0, 0.1, 0.1], 1.0)

    result = summarize_reach_trajectory(
        episode, position_tolerance=0.05, velocity_tolerance=0.1
    )

    assert result["first_near_step"] is None
    assert result["first_crossing_step"] is None
    assert result["opposing_actions_before_crossing"] is None
    assert result["maximum_overshoot"] == 0.0
