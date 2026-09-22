"""Unit tests for discounted Episode returns."""

import numpy as np
import pytest

from robot_learning.returns import calculate_discounted_returns


def test_zero_discount_uses_only_each_immediate_reward() -> None:
    """Gamma zero removes every future reward from the current return."""
    rewards = np.asarray([1.0, -2.0, 3.0])

    returns = calculate_discounted_returns(rewards, 0.0)

    np.testing.assert_array_equal(returns, rewards)


def test_unit_discount_matches_undiscounted_reward_to_go() -> None:
    """Gamma one should produce the reverse cumulative sum."""
    rewards = np.asarray([1.0, -2.0, 3.0])

    returns = calculate_discounted_returns(rewards, 1.0)

    np.testing.assert_array_equal(returns, [2.0, 1.0, 3.0])


def test_return_satisfies_one_step_bellman_recursion() -> None:
    """Every return must equal current reward plus discounted next return."""
    rewards = np.asarray([0.0, 0.0, 2.0])
    gamma = 0.75

    returns = calculate_discounted_returns(rewards, gamma)

    np.testing.assert_allclose(returns[:-1], rewards[:-1] + gamma * returns[1:])
    assert returns[-1] == rewards[-1]
    assert returns[0] == pytest.approx(2.0 * gamma**2)


@pytest.mark.parametrize(
    ("rewards", "discount_factor"),
    [
        (np.asarray([]), 0.9),
        (np.asarray([[1.0]]), 0.9),
        (np.asarray([np.nan]), 0.9),
        (np.asarray([1.0]), -0.1),
        (np.asarray([1.0]), 1.1),
        (np.asarray([1.0]), np.inf),
    ],
)
def test_invalid_return_input_is_rejected(
    rewards: np.ndarray,
    discount_factor: float,
) -> None:
    """Malformed rewards or gamma must not silently create RL targets."""
    with pytest.raises(ValueError):
        calculate_discounted_returns(rewards, discount_factor)
