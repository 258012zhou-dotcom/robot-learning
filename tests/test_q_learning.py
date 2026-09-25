"""Tests for Q-Learning's target and exploration semantics."""

import numpy as np
import pytest

from robot_learning.q_learning import epsilon_greedy_action, update_q_table


def test_update_uses_max_next_q_even_if_behavior_takes_another_action() -> None:
    """Q-Learning's off-policy target uses the best next action."""
    table = np.asarray([[4.0, 0.0], [6.0, 10.0]])

    result = update_q_table(
        table, 0, 0, 2.0, 1, discount_factor=0.5, learning_rate=0.25
    )

    assert result.target == 7.0
    assert result.error == 3.0
    assert result.updated_value == 4.75
    np.testing.assert_array_equal(table[1], [6.0, 10.0])


def test_terminal_update_ignores_next_state_value() -> None:
    """Terminal transitions have no future actions to maximize."""
    table = np.asarray([[0.0, 0.0], [100.0, 200.0]])

    result = update_q_table(
        table, 0, 0, 1.0, None, discount_factor=0.9, learning_rate=0.5
    )

    assert result.target == 1.0
    assert table[0, 0] == 0.5


def test_greedy_and_exploration_endpoints() -> None:
    """Zero epsilon is deterministic; full exploration visits both actions."""
    rng = np.random.default_rng(7)
    values = np.asarray([0.0, 1.0])

    assert all(epsilon_greedy_action(values, 0.0, rng) == 1 for _ in range(20))
    explored = [epsilon_greedy_action(values, 1.0, rng) for _ in range(100)]
    assert set(explored) == {0, 1}


def test_invalid_learning_settings_are_rejected() -> None:
    """An invalid gamma or epsilon cannot silently change the experiment."""
    table = np.zeros((2, 2))
    with pytest.raises(ValueError, match="discount_factor"):
        update_q_table(table, 0, 0, 0.0, 1, discount_factor=1.1, learning_rate=0.2)
    with pytest.raises(ValueError, match="epsilon"):
        epsilon_greedy_action(table[0], -0.1, np.random.default_rng(0))
