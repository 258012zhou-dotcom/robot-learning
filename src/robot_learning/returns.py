"""Return calculations shared by reinforcement-learning experiments."""

import numpy as np


def calculate_discounted_returns(
    rewards: np.ndarray,
    discount_factor: float,
) -> np.ndarray:
    """Calculate ``G_t = r_t + gamma * G_(t+1)`` for one Episode."""
    reward_values = np.asarray(rewards, dtype=np.float64)
    if reward_values.ndim != 1 or reward_values.size == 0:
        raise ValueError("rewards must be a non-empty vector")
    if not np.all(np.isfinite(reward_values)):
        raise ValueError("rewards must contain only finite values")

    gamma = float(discount_factor)
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("discount_factor must be finite and between 0 and 1")

    returns = np.empty_like(reward_values)
    future_return = 0.0
    for index in range(reward_values.size - 1, -1, -1):
        future_return = reward_values[index] + gamma * future_return
        returns[index] = future_return
    return returns
