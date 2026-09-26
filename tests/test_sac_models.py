"""Unit tests for bounded SAC actions, twin Q values, and actor gradients."""

import torch
from torch.distributions import Normal

from robot_learning.sac_models import (
    SquashedGaussianActor,
    TwinQCritic,
    squashed_log_probability,
)


def test_actor_outputs_bounded_actions_and_finite_log_probabilities() -> None:
    torch.manual_seed(4)
    actor = SquashedGaussianActor()
    observations = torch.randn(5, 4)

    actions, log_probabilities = actor.sample(observations)
    deterministic_actions = actor.deterministic_action(observations)

    assert actions.shape == (5, 1)
    assert log_probabilities.shape == (5,)
    assert deterministic_actions.shape == (5, 1)
    assert torch.isfinite(log_probabilities).all()
    assert torch.all(actions.abs() <= 1.0)
    assert torch.all(deterministic_actions.abs() <= 1.0)


def test_tanh_log_probability_includes_change_of_variables_correction() -> None:
    raw_actions = torch.tensor([[0.4], [-0.7]])
    distribution = Normal(torch.zeros_like(raw_actions), torch.ones_like(raw_actions))

    actual = squashed_log_probability(distribution, raw_actions)
    expected = (
        distribution.log_prob(raw_actions)
        - torch.log1p(-torch.tanh(raw_actions).square())
    ).sum(dim=-1)

    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(squashed_log_probability(distribution, torch.tensor([[20.0]]))).all()


def test_twin_critic_returns_two_finite_values_per_transition() -> None:
    critic = TwinQCritic()
    q1, q2 = critic(torch.randn(5, 4), torch.zeros(5, 1))

    assert q1.shape == q2.shape == (5,)
    assert torch.isfinite(q1).all() and torch.isfinite(q2).all()
    assert critic.q1 is not critic.q2
    assert critic.q1[0].weight is not critic.q2[0].weight


def test_actor_receives_gradient_through_sampled_action_and_critic() -> None:
    torch.manual_seed(7)
    actor = SquashedGaussianActor()
    critic = TwinQCritic()
    observations = torch.randn(8, 4)
    actions, log_probabilities = actor.sample(observations)
    q1, q2 = critic(observations, actions)

    (0.2 * log_probabilities - torch.minimum(q1, q2)).mean().backward()

    gradients = [parameter.grad for parameter in actor.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.any(gradient != 0) for gradient in gradients)
