"""Small PPO-style policy objectives for a fixed batch of old-policy actions."""

import torch
from torch import Tensor


def _ratio_and_advantage(
    new_log_probability: Tensor,
    old_log_probability: Tensor,
    advantage: Tensor,
) -> tuple[Tensor, Tensor]:
    if (
        new_log_probability.shape != old_log_probability.shape
        or new_log_probability.shape != advantage.shape
        or new_log_probability.numel() == 0
    ):
        raise ValueError("policy objective inputs must have equal non-empty shapes")
    if not all(
        torch.isfinite(value).all()
        for value in (new_log_probability, old_log_probability, advantage)
    ):
        raise ValueError("policy objective inputs must be finite")
    ratio = torch.exp(new_log_probability - old_log_probability.detach())
    if not torch.isfinite(ratio).all():
        raise ValueError("policy probability ratio must be finite")
    return ratio, advantage.detach()


def importance_weighted_loss(
    new_log_probability: Tensor,
    old_log_probability: Tensor,
    advantage: Tensor,
) -> Tensor:
    """Unclipped surrogate for repeated updates on the same fixed batch."""
    ratio, fixed_advantage = _ratio_and_advantage(
        new_log_probability, old_log_probability, advantage
    )
    return -(ratio * fixed_advantage).mean()


def ppo_clipped_loss(
    new_log_probability: Tensor,
    old_log_probability: Tensor,
    advantage: Tensor,
    clip_epsilon: float,
) -> Tensor:
    """PPO clipped surrogate; this is not a hard policy-change constraint."""
    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("clip_epsilon must be between zero and one")
    ratio, fixed_advantage = _ratio_and_advantage(
        new_log_probability, old_log_probability, advantage
    )
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    return -torch.minimum(
        ratio * fixed_advantage, clipped_ratio * fixed_advantage
    ).mean()
