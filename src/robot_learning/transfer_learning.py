"""Utilities for frozen probes and discriminative full fine-tuning."""

from copy import deepcopy
import math
from numbers import Real

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from robot_learning.image_classification import (
    ClassificationTrainingResult,
    evaluate_classifier,
    train_classification_epoch,
)
from robot_learning.visual_representation import EncoderClassifier


def set_encoder_trainable(
    model: EncoderClassifier,
    trainable: bool,
) -> None:
    """Freeze or unfreeze every parameter in the visual encoder."""
    if type(trainable) is not bool:
        raise ValueError("trainable must be boolean")
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(trainable)


def create_source_head_evaluator(
    adapted_model: EncoderClassifier,
    source_model: EncoderClassifier,
) -> EncoderClassifier:
    """Combine an adapted encoder with the original pretrained source head."""
    if (
        adapted_model.encoder.embedding_dimension
        != source_model.encoder.embedding_dimension
    ):
        raise ValueError("adapted and source encoder dimensions must match")
    evaluator = EncoderClassifier(
        deepcopy(adapted_model.encoder),
        num_classes=source_model.classifier.out_features,
    )
    evaluator.classifier.load_state_dict(source_model.classifier.state_dict())
    adapted_device = next(adapted_model.encoder.parameters()).device
    return evaluator.to(adapted_device)


def create_transfer_optimizer(
    model: EncoderClassifier,
    encoder_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """Create AdamW with separate trainable encoder and head groups."""
    for value, name, allow_zero in (
        (encoder_learning_rate, "encoder_learning_rate", False),
        (head_learning_rate, "head_learning_rate", False),
        (weight_decay, "weight_decay", True),
    ):
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value < 0.0
            or (not allow_zero and value == 0.0)
        ):
            raise ValueError(f"{name} has an invalid value")

    parameter_groups = []
    encoder_parameters = [
        parameter
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    ]
    if encoder_parameters:
        parameter_groups.append(
            {
                "params": encoder_parameters,
                "lr": float(encoder_learning_rate),
                "group_name": "encoder",
            }
        )
    head_parameters = [
        parameter
        for parameter in model.classifier.parameters()
        if parameter.requires_grad
    ]
    if not head_parameters:
        raise ValueError("classification head must contain trainable parameters")
    parameter_groups.append(
        {
            "params": head_parameters,
            "lr": float(head_learning_rate),
            "group_name": "head",
        }
    )
    return AdamW(parameter_groups, weight_decay=float(weight_decay))


def train_transfer_classifier(
    model: EncoderClassifier,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device,
    epochs: int,
    encoder_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
    num_classes: int,
) -> ClassificationTrainingResult:
    """Train transfer parameter groups and restore lowest validation loss."""
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    model.to(device)
    optimizer = create_transfer_optimizer(
        model,
        encoder_learning_rate,
        head_learning_rate,
        weight_decay,
    )
    training_losses: list[float] = []
    validation_losses: list[float] = []
    validation_accuracies: list[float] = []
    best_epoch = 0
    best_validation_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    for epoch_index in range(epochs):
        training_losses.append(
            train_classification_epoch(model, train_loader, optimizer, device)
        )
        validation = evaluate_classifier(
            model,
            validation_loader,
            device,
            num_classes,
        )
        validation_losses.append(validation.loss)
        validation_accuracies.append(validation.accuracy)
        if validation.loss < best_validation_loss:
            best_validation_loss = validation.loss
            best_epoch = epoch_index + 1
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return ClassificationTrainingResult(
        training_losses=training_losses,
        validation_losses=validation_losses,
        validation_accuracies=validation_accuracies,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
    )
