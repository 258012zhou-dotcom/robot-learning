"""Unit checks for the velocity-versus-displacement interface boundary."""

import numpy as np
import pytest

from robot_learning.two_route_navigation import TwoRouteNavigationEnv
from robot_learning.velocity_command_adapter import VelocityCommandAdapter


class FixedVelocityPolicy:
    def __init__(self) -> None:
        self.route: int | None = None

    def reset(self, *, route: int) -> None:
        self.route = route

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        del observation
        assert self.route is not None
        return np.array([0.6, self.route * 0.4], dtype=np.float32)


def test_adapter_applies_one_extra_dt_but_keeps_command_shape_and_bounds() -> None:
    raw_policy = FixedVelocityPolicy()
    correct = VelocityCommandAdapter(raw_policy, scale=1.0)
    correct.reset(route=1)
    np.testing.assert_allclose(correct(np.zeros(7)), [0.6, 0.4])
    incorrect = VelocityCommandAdapter(raw_policy, scale=0.1)
    incorrect.reset(route=-1)
    np.testing.assert_allclose(incorrect(np.zeros(7)), [0.06, -0.04])
    np.testing.assert_allclose(incorrect.raw_commands[0], [0.6, -0.4])
    with pytest.raises(ValueError, match="scale"):
        VelocityCommandAdapter(raw_policy, scale=0.0)


def test_environment_integrates_velocity_once_per_step() -> None:
    environment = TwoRouteNavigationEnv(dt=0.1)
    initial, _ = environment.reset(seed=0)
    next_observation, _, _, _, _ = environment.step(np.array([0.6, 0.0]))
    np.testing.assert_allclose(
        next_observation[:2] - initial[:2], [0.06, 0.0], atol=1e-6
    )
    initial, _ = environment.reset(seed=0)
    wrong_next, _, _, _, _ = environment.step(np.array([0.06, 0.0]))
    np.testing.assert_allclose(
        wrong_next[:2] - initial[:2], [0.006, 0.0], atol=1e-6
    )
    environment.close()
