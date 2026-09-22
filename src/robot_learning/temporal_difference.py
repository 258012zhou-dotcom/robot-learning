"""One-step temporal-difference targets and value updates."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalDifferenceUpdate:
    """Every quantity used by one scalar TD(0) value update."""

    target: float
    error: float
    updated_value: float


def update_state_value_td(
    current_value: float,
    *,
    reward: float,
    next_value: float,
    discount_factor: float,
    learning_rate: float,
    terminated: bool,
) -> TemporalDifferenceUpdate:
    """Apply ``V <- V + alpha * (r + gamma * V_next - V)`` once."""
    values = np.asarray(
        [current_value, reward, next_value, discount_factor, learning_rate],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("TD inputs must be finite")
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError("discount_factor must be between 0 and 1")
    if not 0.0 < learning_rate <= 1.0:
        raise ValueError("learning_rate must be in (0, 1]")
    if type(terminated) is not bool:
        raise ValueError("terminated must be a boolean")

    bootstrap_value = 0.0 if terminated else float(next_value)
    target = float(reward + discount_factor * bootstrap_value)
    error = target - float(current_value)
    updated_value = float(current_value + learning_rate * error)
    return TemporalDifferenceUpdate(
        target=target,
        error=error,
        updated_value=updated_value,
    )
