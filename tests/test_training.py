import pytest
import torch

from robot_learning.dynamics_model import LearnedDynamicsModel
from robot_learning.motion_dataset import (
    fit_input_standardization,
    generate_motion_samples,
    split_motion_dataset,
    standardize_motion_inputs,
)
from robot_learning.training import (
    create_data_loader,
    evaluate_mse,
    train_model,
)


def build_small_training_problem():
    inputs, targets = generate_motion_samples(240, dt=0.1, seed=42)
    splits = split_motion_dataset(inputs, targets, 0.7, 0.15, seed=7)
    stats = fit_input_standardization(splits.train_inputs)

    train_loader = create_data_loader(
        standardize_motion_inputs(splits.train_inputs, stats),
        splits.train_targets,
        batch_size=32,
        shuffle=True,
        seed=11,
    )
    validation_loader = create_data_loader(
        standardize_motion_inputs(splits.validation_inputs, stats),
        splits.validation_targets,
        batch_size=32,
        shuffle=False,
        seed=11,
    )
    return train_loader, validation_loader


def test_training_reduces_validation_loss_and_restores_best_model():
    torch.manual_seed(3)
    model = LearnedDynamicsModel()
    train_loader, validation_loader = build_small_training_problem()
    device = torch.device("cpu")
    initial_loss = evaluate_mse(model, validation_loader, device)

    result = train_model(
        model,
        train_loader,
        validation_loader,
        device=device,
        epochs=80,
        learning_rate=0.03,
        weight_decay=0.0,
    )
    restored_loss = evaluate_mse(model, validation_loader, device)

    assert result.best_validation_loss < initial_loss * 0.01
    assert restored_loss == pytest.approx(result.best_validation_loss)
    assert result.best_epoch == (
        result.validation_losses.index(min(result.validation_losses)) + 1
    )
    assert len(result.training_losses) == 80
    assert len(result.validation_losses) == 80


def test_training_changes_model_parameters():
    torch.manual_seed(3)
    model = LearnedDynamicsModel()
    initial_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    train_loader, validation_loader = build_small_training_problem()

    train_model(
        model,
        train_loader,
        validation_loader,
        device=torch.device("cpu"),
        epochs=2,
        learning_rate=0.03,
        weight_decay=0.0,
    )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_parameters, model.parameters())
    )


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_data_loader_rejects_invalid_batch_size(batch_size):
    inputs = torch.zeros(4, 4)
    targets = torch.zeros(4, 2)

    with pytest.raises(ValueError, match="batch_size"):
        create_data_loader(
            inputs,
            targets,
            batch_size=batch_size,
            shuffle=True,
            seed=42,
        )


@pytest.mark.parametrize(
    ("epochs", "learning_rate", "weight_decay"),
    [
        (0, 0.01, 0.0),
        (10, 0.0, 0.0),
        (10, 0.01, -0.1),
    ],
)
def test_training_rejects_invalid_hyperparameters(
    epochs,
    learning_rate,
    weight_decay,
):
    model = LearnedDynamicsModel()
    train_loader, validation_loader = build_small_training_problem()

    with pytest.raises(ValueError):
        train_model(
            model,
            train_loader,
            validation_loader,
            device=torch.device("cpu"),
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
