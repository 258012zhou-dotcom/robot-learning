"""Tabular Q-Learning for small discrete teaching environments."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QUpdate:
    """One Q-Learning target, error, and updated table entry."""

    target: float
    error: float
    updated_value: float


def update_q_table(
    table: np.ndarray,
    state: int,
    action: int,
    reward: float,
    next_state: int | None,
    *,
    discount_factor: float,
    learning_rate: float,
) -> QUpdate:
    """Update one entry using the best estimated action at the next state."""
    if table.ndim != 2 or min(table.shape) == 0 or not np.all(np.isfinite(table)):
        raise ValueError("table must be a finite, non-empty matrix")
    if type(state) is not int or not 0 <= state < table.shape[0]:
        raise ValueError("state is outside the table")
    if type(action) is not int or not 0 <= action < table.shape[1]:
        raise ValueError("action is outside the table")
    if next_state is not None and (
        type(next_state) is not int or not 0 <= next_state < table.shape[0]
    ):
        raise ValueError("next_state is outside the table")
    if not np.isfinite(reward):
        raise ValueError("reward must be finite")
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount_factor must be between 0 and 1")
    if not 0.0 < learning_rate <= 1.0:
        raise ValueError("learning_rate must be in (0, 1]")

    next_value = 0.0 if next_state is None else float(np.max(table[next_state]))
    target = float(reward + discount_factor * next_value)
    error = target - float(table[state, action])
    table[state, action] += learning_rate * error
    return QUpdate(target, error, float(table[state, action]))


def epsilon_greedy_action(
    action_values: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """Choose a random action with probability epsilon, otherwise argmax."""
    values = np.asarray(action_values)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("action_values must be a finite, non-empty vector")
    if not np.isfinite(epsilon) or not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")
    if epsilon == 0.0:
        return int(np.argmax(values))
    if rng.random() < epsilon:
        return int(rng.integers(values.size))
    return int(np.argmax(values))
