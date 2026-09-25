"""Basic sampled policy-gradient objective."""

import torch
from torch import Tensor


def reinforce_loss(log_probability: Tensor, advantage: Tensor) -> Tensor:
    """Return the loss whose descent raises probabilities for positive advantage."""
    if log_probability.shape != advantage.shape or log_probability.numel() == 0:
        raise ValueError("log_probability and advantage must have equal non-empty shapes")
    if not torch.isfinite(log_probability).all() or not torch.isfinite(advantage).all():
        raise ValueError("policy-gradient inputs must be finite")
    # A baseline is a target here; gradients update only the policy.
    return -(log_probability * advantage.detach()).mean()
