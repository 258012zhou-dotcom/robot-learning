"""Small, explicit value calculations for reinforcement-learning lessons."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PolicyValueAnalysis:
    """State value and per-action advantages under one discrete policy."""

    state_value: float
    advantages: np.ndarray


def analyze_policy_values(
    action_values: np.ndarray,
    action_probabilities: np.ndarray,
) -> PolicyValueAnalysis:
    """Calculate ``V = sum(pi * Q)`` and ``A = Q - V``."""
    values = np.asarray(action_values, dtype=np.float64)
    probabilities = np.asarray(action_probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("action_values must be a non-empty vector")
    if probabilities.shape != values.shape:
        raise ValueError("action_probabilities must match action_values")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(probabilities)):
        raise ValueError("values and probabilities must be finite")
    if np.any(probabilities < 0.0) or not np.isclose(
        probabilities.sum(), 1.0
    ):
        raise ValueError("action_probabilities must be non-negative and sum to 1")

    state_value = float(np.dot(probabilities, values))
    return PolicyValueAnalysis(
        state_value=state_value,
        advantages=values - state_value,
    )


def estimate_discrete_action_values(
    action_ids: np.ndarray,
    returns: np.ndarray,
    *,
    action_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate each Q value by averaging sampled returns for that action."""
    actions = np.asarray(action_ids)
    return_values = np.asarray(returns, dtype=np.float64)
    if actions.ndim != 1 or return_values.shape != actions.shape:
        raise ValueError("action_ids and returns must be equal-length vectors")
    if actions.size == 0 or not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("action_ids must be a non-empty integer vector")
    if type(action_count) is not int or action_count <= 0:
        raise ValueError("action_count must be a positive integer")
    if not np.all(np.isfinite(return_values)):
        raise ValueError("returns must contain only finite values")
    if np.any(actions < 0) or np.any(actions >= action_count):
        raise ValueError("action_ids contain an unknown action")

    counts = np.bincount(actions, minlength=action_count)
    if np.any(counts == 0):
        raise ValueError("every action needs at least one return sample")
    sums = np.bincount(
        actions,
        weights=return_values,
        minlength=action_count,
    )
    return sums / counts, counts
