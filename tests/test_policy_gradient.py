"""Tests for the direction and baseline handling of a policy-gradient step."""

import pytest
import torch

from robot_learning.policy_gradient import reinforce_loss


@pytest.mark.parametrize(
    ("action", "advantage"),
    [(1.0, 0.5), (0.0, -0.5)],
)
def test_sampled_update_raises_probability_of_better_action(
    action: float,
    advantage: float,
) -> None:
    """Both possible samples should favor the two-point action in this toy task."""
    logit = torch.tensor(0.0, requires_grad=True)
    distribution = torch.distributions.Bernoulli(logits=logit)
    loss = reinforce_loss(
        distribution.log_prob(torch.tensor(action)),
        torch.tensor(advantage),
    )
    loss.backward()

    assert logit.grad is not None
    assert logit.grad.item() == pytest.approx(-0.25)
    updated_logit = logit.detach() - 0.1 * logit.grad
    assert torch.sigmoid(updated_logit) > torch.sigmoid(logit.detach())


def test_baseline_is_not_updated_by_policy_loss() -> None:
    """Actor loss must not backpropagate into a future value estimator."""
    logit = torch.tensor(0.0, requires_grad=True)
    baseline = torch.tensor(1.5, requires_grad=True)
    reward = torch.tensor(2.0)
    distribution = torch.distributions.Bernoulli(logits=logit)

    loss = reinforce_loss(
        distribution.log_prob(torch.tensor(1.0)),
        reward - baseline,
    )
    loss.backward()

    assert logit.grad is not None
    assert baseline.grad is None


def test_mismatched_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="equal non-empty"):
        reinforce_loss(torch.zeros(2), torch.zeros(1))
