import pytest
import torch

from robot_learning.dynamics_model import (
    DynamicsMLP,
    LearnedDynamicsModel,
    select_torch_device,
)


def test_model_has_expected_output_shape():
    model = LearnedDynamicsModel()
    inputs = torch.randn(8, 4)

    predictions = model(inputs)

    assert predictions.shape == (8, 2)


def test_model_rejects_invalid_input_shape():
    model = LearnedDynamicsModel()

    with pytest.raises(ValueError, match=r"\(N, 4\)"):
        model(torch.randn(8, 3))


def test_model_has_expected_parameter_count():
    model = LearnedDynamicsModel()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert parameter_count == 10


def test_backward_computes_finite_parameter_gradients():
    model = LearnedDynamicsModel()
    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 2)

    predictions = model(inputs)
    loss = torch.nn.functional.mse_loss(predictions, targets)
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_model_and_data_run_on_selected_device():
    device = select_torch_device()
    model = LearnedDynamicsModel().to(device)
    inputs = torch.randn(8, 4, device=device)

    predictions = model(inputs)

    assert next(model.parameters()).device == device
    assert predictions.device == device


def test_high_capacity_model_has_expected_shape_and_more_parameters():
    linear_model = LearnedDynamicsModel()
    mlp = DynamicsMLP(hidden_size=16)

    predictions = mlp(torch.randn(8, 4))

    assert predictions.shape == (8, 2)
    assert sum(p.numel() for p in mlp.parameters()) > sum(
        p.numel() for p in linear_model.parameters()
    )


@pytest.mark.parametrize("hidden_size", [0, -1, 1.5, True])
def test_high_capacity_model_rejects_invalid_hidden_size(hidden_size):
    with pytest.raises(ValueError, match="hidden_size"):
        DynamicsMLP(hidden_size=hidden_size)
