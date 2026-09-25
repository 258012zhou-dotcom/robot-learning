"""Tests for the fixed-batch PPO clipping mechanism."""

import pytest
import torch

from robot_learning.ppo_clipping import importance_weighted_loss, ppo_clipped_loss


def _batch_log_probabilities(logit: torch.Tensor) -> torch.Tensor:
    actions = torch.tensor([0.0, 1.0])
    return torch.distributions.Bernoulli(logits=logit).log_prob(actions)


def test_clipped_objective_improves_good_action_inside_clip_range() -> None:
    logit = torch.tensor(0.0, requires_grad=True)
    old_log_probabilities = _batch_log_probabilities(torch.tensor(0.0))
    loss = ppo_clipped_loss(
        _batch_log_probabilities(logit),
        old_log_probabilities,
        torch.tensor([-0.5, 0.5]),
        0.2,
    )
    loss.backward()

    assert logit.grad is not None
    assert logit.grad.item() < 0.0
    updated_logit = logit.detach() - logit.grad
    assert torch.sigmoid(updated_logit) > 0.5


def test_clipping_stops_extra_favorable_push_but_is_not_a_hard_bound() -> None:
    # P(high)=0.7 already exceeds the 0.6 clip threshold for this old policy.
    logit = torch.logit(torch.tensor(0.7)).detach().requires_grad_()
    old_log_probabilities = _batch_log_probabilities(torch.tensor(0.0))
    advantages = torch.tensor([-0.5, 0.5])
    clipped_loss = ppo_clipped_loss(
        _batch_log_probabilities(logit), old_log_probabilities, advantages, 0.2
    )
    clipped_loss.backward()

    assert logit.grad is not None
    assert logit.grad.item() == pytest.approx(0.0)
    assert torch.sigmoid(logit.detach()).item() > 0.6

    logit.grad = None
    importance_weighted_loss(
        _batch_log_probabilities(logit), old_log_probabilities, advantages
    ).backward()
    assert logit.grad is not None
    assert logit.grad.item() < 0.0


def test_clipping_does_not_remove_gradient_for_unfavorable_change() -> None:
    # P(high)=0.3 is outside the range, but in the wrong direction.
    logit = torch.logit(torch.tensor(0.3)).detach().requires_grad_()
    loss = ppo_clipped_loss(
        _batch_log_probabilities(logit),
        _batch_log_probabilities(torch.tensor(0.0)),
        torch.tensor([-0.5, 0.5]),
        0.2,
    )
    loss.backward()

    assert logit.grad is not None
    assert logit.grad.item() < 0.0


def test_old_policy_and_advantage_are_fixed_targets() -> None:
    new_logit = torch.tensor(0.0, requires_grad=True)
    old_logit = torch.tensor(0.0, requires_grad=True)
    advantages = torch.tensor([-0.5, 0.5], requires_grad=True)
    loss = ppo_clipped_loss(
        _batch_log_probabilities(new_logit),
        _batch_log_probabilities(old_logit),
        advantages,
        0.2,
    )
    loss.backward()

    assert new_logit.grad is not None
    assert old_logit.grad is None
    assert advantages.grad is None


def test_invalid_clip_and_shape_are_rejected() -> None:
    with pytest.raises(ValueError, match="clip_epsilon"):
        ppo_clipped_loss(torch.zeros(2), torch.zeros(2), torch.zeros(2), 1.0)
    with pytest.raises(ValueError, match="equal non-empty"):
        ppo_clipped_loss(torch.zeros(2), torch.zeros(1), torch.zeros(2), 0.2)
