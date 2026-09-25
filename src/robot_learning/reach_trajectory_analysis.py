"""Small diagnostics for overshoot and speed in reach-task Episodes."""

import numpy as np

from robot_learning.gymnasium_rollout import EpisodeResult


def summarize_reach_trajectory(
    episode: EpisodeResult,
    *,
    position_tolerance: float,
    velocity_tolerance: float,
) -> dict[str, float | int | None]:
    """Describe when the robot nears or crosses its fixed target.

    Observation columns are position, velocity, target, and target error.
    Step indices refer to observations: step zero is the reset state.
    """
    observations = episode.observations
    if (
        observations.ndim != 2
        or observations.shape[1] < 4
        or observations.shape[0] != episode.step_count + 1
        or episode.actions.ndim != 2
        or episode.actions.shape != (episode.step_count, 1)
        or position_tolerance <= 0.0
        or velocity_tolerance <= 0.0
    ):
        raise ValueError("invalid reach trajectory shape or tolerances")
    if not np.all(np.isfinite(observations)) or not np.all(np.isfinite(episode.actions)):
        raise ValueError("reach observations and actions must be finite")
    target = float(observations[0, 2])
    if not np.allclose(observations[:, 2], target):
        raise ValueError("target must stay fixed within one Episode")
    initial_position = float(observations[0, 0])
    if np.isclose(target, initial_position):
        raise ValueError("initial position must differ from target")

    direction = float(np.sign(target - initial_position))
    positions = observations[:, 0]
    velocities = observations[:, 1]
    distance = np.abs(positions - target)
    near_steps = np.flatnonzero(distance <= position_tolerance)
    crossing_steps = np.flatnonzero(direction * (positions - target) >= 0.0)
    first_near = int(near_steps[0]) if near_steps.size else None
    first_crossing = int(crossing_steps[0]) if crossing_steps.size else None
    opposing_steps = np.flatnonzero(direction * episode.actions[:, 0] < 0.0)
    first_opposing = int(opposing_steps[0]) if opposing_steps.size else None
    near_and_slow = (distance <= position_tolerance) & (
        np.abs(velocities) <= velocity_tolerance
    )
    return {
        "first_near_step": first_near,
        "speed_at_first_near": (
            abs(float(velocities[first_near])) if first_near is not None else None
        ),
        "first_crossing_step": first_crossing,
        "first_opposing_action_step": first_opposing,
        "opposing_actions_before_crossing": (
            int(np.count_nonzero(opposing_steps < first_crossing))
            if first_crossing is not None else None
        ),
        "speed_at_first_crossing": (
            abs(float(velocities[first_crossing]))
            if first_crossing is not None else None
        ),
        "maximum_overshoot": float(np.maximum(direction * (positions - target), 0.0).max()),
        "near_position_steps": int(near_steps.size),
        "near_and_slow_steps": int(near_and_slow.sum()),
        "final_position": float(positions[-1]),
        "final_velocity": float(velocities[-1]),
    }
