"""Unit checks for the controlled SAC target-distance intervention."""

from pathlib import Path

import numpy as np
import pytest

from robot_learning.point_robot_reach_env import PointRobotReachEnv
from robot_learning.sac_direction_schedule import ScheduledTargetEnvironment


MODEL_PATH = Path(__file__).resolve().parents[1] / "experiments/017_mujoco_step/point_robot.xml"


@pytest.mark.parametrize("alternate,expected", [
    (False, [0.6, 0.6, 1.0, 1.0]),
    (True, [0.6, -0.6, 1.0, -1.0]),
])
def test_schedule_changes_only_the_target_sign(alternate, expected) -> None:
    environment = ScheduledTargetEnvironment(
        PointRobotReachEnv(MODEL_PATH, max_episode_steps=100),
        target_magnitudes=(0.6, 0.6, 1.0, 1.0),
        alternate_direction=alternate,
    )
    try:
        actual = [float(environment.reset(seed=10 + index)[0][2]) for index in range(4)]
    finally:
        environment.close()
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(environment.scheduled_targets, expected)


def test_schedule_rejects_out_of_range_magnitude() -> None:
    environment = PointRobotReachEnv(MODEL_PATH)
    try:
        with pytest.raises(ValueError, match="range"):
            ScheduledTargetEnvironment(
                environment, target_magnitudes=(2.5,), alternate_direction=True,
            )
    finally:
        environment.close()
