"""Tests for the Critic's value target and gradient boundary."""

import pytest
import torch

from robot_learning.actor_critic import value_prediction_loss
from robot_learning.policy_gradient import reinforce_loss


def test_critic_update_moves_prediction_toward_terminal_reward() -> None:
    """For a one-step episode, the observed reward is the full return."""
    prediction = torch.tensor(0.0, requires_grad=True)
    loss = value_prediction_loss(prediction, torch.tensor(2.0))
    loss.backward()

    assert loss.item() == pytest.approx(4.0)
    assert prediction.grad is not None
    assert prediction.grad.item() == pytest.approx(-4.0)
    updated = prediction.detach() - 0.1 * prediction.grad
    assert abs(2.0 - updated.item()) < 2.0


def test_actor_and_critic_gradients_stay_separate() -> None:
    """Actor uses detached advantage; Critic uses a detached return target."""
    actor_logit = torch.tensor(0.0, requires_grad=True)
    critic_value = torch.tensor(1.5, requires_grad=True)
    reward = torch.tensor(2.0)
    distribution = torch.distributions.Bernoulli(logits=actor_logit)

    actor_loss = reinforce_loss(
        distribution.log_prob(torch.tensor(1.0)),
        reward - critic_value,
    )
    actor_loss.backward()
    assert actor_logit.grad is not None
    assert critic_value.grad is None

    critic_loss = value_prediction_loss(critic_value, reward)
    critic_loss.backward()
    assert critic_value.grad is not None
    assert actor_logit.grad.item() == pytest.approx(-0.25)


def test_value_target_is_not_trainable_through_critic_loss() -> None:
    """A future bootstrapped value target must not receive this loss gradient."""
    prediction = torch.tensor(0.0, requires_grad=True)
    target = torch.tensor(2.0, requires_grad=True)

    value_prediction_loss(prediction, target).backward()

    assert prediction.grad is not None
    assert target.grad is None
