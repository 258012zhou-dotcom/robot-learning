"""Unit tests for the ACT-style training and inference contracts."""

import pytest
import torch

from robot_learning.act_lowdim import (
    ACTLowDim,
    ACTLowDimOutput,
    act_lowdim_loss,
)


def tiny_model() -> ACTLowDim:
    return ACTLowDim(
        observation_size=2,
        action_size=1,
        horizon=3,
        hidden_size=8,
        latent_size=2,
        attention_heads=2,
    )


def test_training_shapes_gradients_and_bounded_inference_without_targets() -> None:
    model = tiny_model()
    observations = torch.zeros((2, 2))
    targets = torch.tensor([[[0.0], [0.5], [0.0]], [[-0.2], [0.0], [0.0]]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    training_output = model(observations, targets, mask)
    assert training_output.actions.shape == (2, 3, 1)
    assert training_output.posterior_mean is not None
    assert training_output.posterior_mean.shape == (2, 2)
    total, action_l1, kl = act_lowdim_loss(
        training_output, targets, mask, kl_weight=0.01
    )
    assert torch.isfinite(total)
    assert action_l1 >= 0
    assert kl >= -1e-5
    total.backward()
    assert model.posterior_projection.weight.grad is not None
    assert model.action_head.weight.grad is not None

    model.eval()
    with torch.inference_mode():
        first = model(observations)
        second = model(observations)
    assert first.posterior_mean is None
    assert first.posterior_logvar is None
    torch.testing.assert_close(first.actions, second.actions)
    assert first.actions.shape == (2, 3, 1)
    assert torch.all(first.actions <= 1.0)
    assert torch.all(first.actions >= -1.0)


def test_padding_values_do_not_change_posterior_or_masked_loss() -> None:
    model = tiny_model().eval()
    observations = torch.tensor([[0.1, -0.2]])
    targets = torch.tensor([[[0.3], [0.4], [0.0]]])
    mask = torch.tensor([[True, True, False]])
    altered = targets.clone()
    altered[0, 2, 0] = 999.0

    torch.manual_seed(7)
    original_output = model(observations, targets, mask)
    torch.manual_seed(7)
    altered_output = model(observations, altered, mask)
    torch.testing.assert_close(
        original_output.posterior_mean, altered_output.posterior_mean
    )
    torch.testing.assert_close(original_output.actions, altered_output.actions)
    original_loss = act_lowdim_loss(
        original_output, targets, mask, kl_weight=0.01
    )[0]
    altered_loss = act_lowdim_loss(
        altered_output, altered, mask, kl_weight=0.01
    )[0]
    torch.testing.assert_close(original_loss, altered_loss)


def test_loss_has_l1_and_kl_and_requires_training_posterior() -> None:
    output = ACTLowDimOutput(
        actions=torch.tensor([[[0.0], [0.0]]]),
        posterior_mean=torch.ones((1, 1)),
        posterior_logvar=torch.zeros((1, 1)),
    )
    targets = torch.tensor([[[1.0], [100.0]]])
    mask = torch.tensor([[True, False]])
    total, l1, kl = act_lowdim_loss(output, targets, mask, kl_weight=0.2)
    assert l1.item() == 1.0
    assert kl.item() == 0.5
    assert total.item() == pytest.approx(1.1)
    with pytest.raises(ValueError, match="posterior"):
        act_lowdim_loss(
            ACTLowDimOutput(output.actions, None, None),
            targets,
            mask,
            kl_weight=0.2,
        )


def test_rejects_missing_mask_and_invalid_shapes() -> None:
    model = tiny_model()
    with pytest.raises(ValueError, match="together"):
        model(torch.zeros((1, 2)), torch.zeros((1, 3, 1)))
    with pytest.raises(ValueError, match="first action"):
        model(
            torch.zeros((1, 2)),
            torch.zeros((1, 3, 1)),
            torch.tensor([[False, False, False]]),
        )


def test_explicit_zero_latent_matches_default_inference_and_shape_is_checked() -> None:
    model = tiny_model().eval()
    observations = torch.tensor([[0.2, -0.1], [-0.3, 0.4]])
    zero = torch.zeros((2, model.latent_size))
    with torch.inference_mode():
        baseline = model(observations).actions
        intervention = model.predict_with_latent(observations, zero)
    torch.testing.assert_close(baseline, intervention)
    with pytest.raises(ValueError, match="latent"):
        model.predict_with_latent(observations, torch.zeros((1, model.latent_size)))
