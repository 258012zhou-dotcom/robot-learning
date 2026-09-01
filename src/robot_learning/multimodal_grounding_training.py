"""Training and evaluation utilities for multimodal grounding models.

This file is deliberately separate from ``multimodal_grounding.py``:

- ``multimodal_grounding.py`` defines data and model structure.
- this module defines how parameters are optimized and results are measured.

The separation lets readers study the model without first understanding the
training loop.
"""

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from robot_learning.multimodal_grounding import (
    GroundingOutput,
    LanguageOnlyGrounder,
    MultimodalGrounder,
    VisionOnlyGrounder,
    calculate_grounding_loss,
    calculate_mean_center_error,
)


GroundingModel = VisionOnlyGrounder | LanguageOnlyGrounder | MultimodalGrounder


@dataclass(frozen=True)
class GroundingEvaluation:
    """Loss and localization metrics for one complete dataset split."""

    loss: float
    mean_center_error: float
    selection_accuracy: float
    predicted_centers: Tensor


@dataclass(frozen=True)
class GroundingTrainingResult:
    """Epoch histories and the validation-selected checkpoint information."""

    training_losses: list[float]
    validation_losses: list[float]
    validation_center_errors: list[float]
    best_epoch: int
    best_validation_loss: float


def train_grounding_model(
    model: GroundingModel,
    train_images: Tensor,
    train_token_ids: Tensor,
    train_target_centers: Tensor,
    validation_images: Tensor,
    validation_token_ids: Tensor,
    validation_target_centers: Tensor,
    validation_scene_centers: Tensor,
    validation_target_indices: Tensor,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> GroundingTrainingResult:
    """Train one baseline and restore its lowest-validation-loss checkpoint."""
    _validate_training_settings(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
    )
    _validate_paired_inputs(
        train_images,
        train_token_ids,
        train_target_centers,
    )

    # TensorDataset keeps image, instruction, and coordinate rows synchronized
    # while DataLoader shuffles sample order between training epochs.
    training_dataset = TensorDataset(
        train_images,
        train_token_ids,
        train_target_centers,
    )
    generator = torch.Generator().manual_seed(seed)
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    model.to(device)

    training_losses: list[float] = []
    validation_losses: list[float] = []
    validation_center_errors: list[float] = []
    best_epoch = 0
    best_validation_loss = float("inf")
    best_state = deepcopy(model.state_dict())

    for epoch_index in range(epochs):
        model.train()
        total_loss = 0.0
        processed_samples = 0

        for images, token_ids, target_centers in training_loader:
            images = images.to(device)
            token_ids = token_ids.to(device)
            target_centers = target_centers.to(device)

            optimizer.zero_grad()
            output = _forward_model(model, images, token_ids)
            loss = calculate_grounding_loss(
                output.predicted_centers,
                target_centers,
            )
            loss.backward()
            optimizer.step()

            batch_sample_count = images.shape[0]
            total_loss += float(loss.item()) * batch_sample_count
            processed_samples += batch_sample_count

        epoch_training_loss = total_loss / processed_samples
        validation = evaluate_grounding_model(
            model=model,
            images=validation_images,
            token_ids=validation_token_ids,
            target_centers=validation_target_centers,
            scene_centers=validation_scene_centers,
            target_indices=validation_target_indices,
            batch_size=batch_size,
            device=device,
        )
        training_losses.append(epoch_training_loss)
        validation_losses.append(validation.loss)
        validation_center_errors.append(validation.mean_center_error)

        # Validation data chooses the checkpoint; test data remains untouched.
        if validation.loss < best_validation_loss:
            best_validation_loss = validation.loss
            best_epoch = epoch_index + 1
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return GroundingTrainingResult(
        training_losses=training_losses,
        validation_losses=validation_losses,
        validation_center_errors=validation_center_errors,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
    )


@torch.no_grad()
def evaluate_grounding_model(
    model: GroundingModel,
    images: Tensor,
    token_ids: Tensor,
    target_centers: Tensor,
    scene_centers: Tensor,
    target_indices: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> GroundingEvaluation:
    """Evaluate a learned model without updating any parameters."""
    _validate_paired_inputs(images, token_ids, target_centers)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    model.to(device)
    model.eval()
    predicted_batches: list[Tensor] = []

    # Evaluation does not shuffle: prediction row i must still match target i.
    for start_index in range(0, images.shape[0], batch_size):
        end_index = start_index + batch_size
        output = _forward_model(
            model,
            images[start_index:end_index].to(device),
            token_ids[start_index:end_index].to(device),
        )
        predicted_batches.append(output.predicted_centers.cpu())

    predicted_centers = torch.cat(predicted_batches, dim=0)
    loss = calculate_grounding_loss(
        predicted_centers,
        target_centers.cpu(),
    )
    return evaluate_grounding_predictions(
        predicted_centers=predicted_centers,
        target_centers=target_centers.cpu(),
        scene_centers=scene_centers.cpu(),
        target_indices=target_indices.cpu(),
        loss=float(loss.item()),
    )


def evaluate_grounding_predictions(
    predicted_centers: Tensor,
    target_centers: Tensor,
    scene_centers: Tensor,
    target_indices: Tensor,
    *,
    loss: float,
) -> GroundingEvaluation:
    """Measure continuous coordinate error and discrete object selection."""
    if scene_centers.ndim != 3 or scene_centers.shape[2] != 2:
        raise ValueError("scene_centers must have shape (N, K, 2)")
    if target_indices.ndim != 1:
        raise ValueError("target_indices must have shape (N,)")
    sample_count = predicted_centers.shape[0]
    if not (
        target_centers.shape[0]
        == scene_centers.shape[0]
        == target_indices.shape[0]
        == sample_count
    ):
        raise ValueError("all evaluation tensors must share sample count N")

    # Assign each predicted coordinate to the nearest candidate object center.
    candidate_distances = torch.linalg.vector_norm(
        predicted_centers[:, None, :] - scene_centers,
        dim=2,
    )
    predicted_indices = candidate_distances.argmin(dim=1)
    selection_accuracy = float(
        (predicted_indices == target_indices).to(torch.float32).mean().item()
    )

    return GroundingEvaluation(
        loss=float(loss),
        mean_center_error=calculate_mean_center_error(
            predicted_centers,
            target_centers,
        ),
        selection_accuracy=selection_accuracy,
        predicted_centers=predicted_centers,
    )


def _forward_model(
    model: GroundingModel,
    images: Tensor,
    token_ids: Tensor,
) -> GroundingOutput:
    """Call each baseline with only the modalities it is allowed to use."""
    if isinstance(model, VisionOnlyGrounder):
        return model(images)
    if isinstance(model, LanguageOnlyGrounder):
        return model(token_ids)
    if isinstance(model, MultimodalGrounder):
        return model(images, token_ids)
    raise TypeError("unsupported grounding model type")


def _validate_paired_inputs(
    images: Tensor,
    token_ids: Tensor,
    target_centers: Tensor,
) -> None:
    """Ensure three training inputs describe the same N samples."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape (N, 3, H, W)")
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape (N, L)")
    if target_centers.ndim != 2 or target_centers.shape[1] != 2:
        raise ValueError("target_centers must have shape (N, 2)")
    if not (
        images.shape[0]
        == token_ids.shape[0]
        == target_centers.shape[0]
    ):
        raise ValueError("paired inputs must share sample count N")


def _validate_training_settings(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> None:
    """Reject invalid optimization settings before a long experiment begins."""
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
