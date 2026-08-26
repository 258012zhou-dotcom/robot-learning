"""Synthetic shape data and a small CNN for image classification."""

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class ClassificationDatasetSplits:
    """Train, validation, and test tensors with original sample indices."""

    train_images: Tensor
    train_labels: Tensor
    validation_images: Tensor
    validation_labels: Tensor
    test_images: Tensor
    test_labels: Tensor
    train_indices: Tensor
    validation_indices: Tensor
    test_indices: Tensor


@dataclass(frozen=True)
class ClassificationEvaluation:
    """Loss, accuracy, predictions, and a row=true confusion matrix."""

    loss: float
    accuracy: float
    confusion_matrix: Tensor
    predictions: Tensor
    labels: Tensor


@dataclass(frozen=True)
class ClassificationTrainingResult:
    """Training history and the best validation checkpoint information."""

    training_losses: list[float]
    validation_losses: list[float]
    validation_accuracies: list[float]
    best_epoch: int
    best_validation_loss: float


class SmallShapeCNN(nn.Module):
    """A compact CNN that classifies RGB images into shape classes."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least two")

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        """Return one vector of class logits per image."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (N, 3, H, W)")
        features = self.features(images)
        flattened = torch.flatten(features, start_dim=1)
        return self.classifier(flattened)


def generate_shape_classification_dataset(
    sample_count: int,
    image_size: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Generate balanced RGB images of squares and circles."""
    if type(sample_count) is not int or sample_count < 2:
        raise ValueError("sample_count must be an integer of at least two")
    if type(image_size) is not int or image_size < 16:
        raise ValueError("image_size must be an integer of at least 16")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(seed)
    labels = np.arange(sample_count, dtype=np.int64) % 2
    rng.shuffle(labels)
    images_bgr = np.empty(
        (sample_count, image_size, image_size, 3),
        dtype=np.uint8,
    )

    for index, label in enumerate(labels):
        image = rng.integers(
            0,
            31,
            size=(image_size, image_size, 3),
            dtype=np.uint8,
        )
        shape_size = int(
            rng.integers(image_size // 4, image_size // 2 + 1)
        )
        half_size = shape_size // 2
        center_x = int(rng.integers(half_size, image_size - half_size))
        center_y = int(rng.integers(half_size, image_size - half_size))
        color = tuple(
            int(value) for value in rng.integers(150, 256, size=3)
        )

        if label == 0:
            cv2.rectangle(
                image,
                (center_x - half_size, center_y - half_size),
                (center_x + half_size, center_y + half_size),
                color,
                thickness=-1,
            )
        else:
            cv2.circle(
                image,
                (center_x, center_y),
                half_size,
                color,
                thickness=-1,
            )
        images_bgr[index] = image

    images_rgb = images_bgr[..., ::-1].copy()
    image_tensor = (
        torch.from_numpy(images_rgb)
        .permute(0, 3, 1, 2)
        .to(dtype=torch.float32)
        .div(255.0)
    )
    label_tensor = torch.from_numpy(labels)
    return image_tensor, label_tensor


def generate_shape_challenge_dataset(
    sample_count: int,
    image_size: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Generate a controlled OOD set with unseen square rotations."""
    if type(sample_count) is not int or sample_count < 2:
        raise ValueError("sample_count must be an integer of at least two")
    if type(image_size) is not int or image_size < 16:
        raise ValueError("image_size must be an integer of at least 16")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(seed)
    labels = np.arange(sample_count, dtype=np.int64) % 2
    rng.shuffle(labels)
    images_bgr = np.empty(
        (sample_count, image_size, image_size, 3),
        dtype=np.uint8,
    )

    for index, label in enumerate(labels):
        image = rng.integers(
            0,
            31,
            size=(image_size, image_size, 3),
            dtype=np.uint8,
        )
        shape_size = int(
            rng.integers(image_size // 4, image_size // 2 + 1)
        )
        half_size = shape_size // 2
        rotation_margin = math.ceil(half_size * math.sqrt(2.0))
        center_x = int(
            rng.integers(rotation_margin, image_size - rotation_margin)
        )
        center_y = int(
            rng.integers(rotation_margin, image_size - rotation_margin)
        )
        color = tuple(
            int(value) for value in rng.integers(150, 256, size=3)
        )

        if label == 0:
            angle_magnitude = float(rng.uniform(20.0, 70.0))
            angle_degrees = angle_magnitude * int(rng.choice((-1, 1)))
            rotated_square = cv2.boxPoints(
                (
                    (float(center_x), float(center_y)),
                    (float(shape_size), float(shape_size)),
                    angle_degrees,
                )
            ).astype(np.int32)
            cv2.fillConvexPoly(image, rotated_square, color)
        else:
            cv2.circle(
                image,
                (center_x, center_y),
                half_size,
                color,
                thickness=-1,
            )
        images_bgr[index] = image

    images_rgb = images_bgr[..., ::-1].copy()
    image_tensor = (
        torch.from_numpy(images_rgb)
        .permute(0, 3, 1, 2)
        .to(dtype=torch.float32)
        .div(255.0)
    )
    return image_tensor, torch.from_numpy(labels)


def generate_rotation_augmented_shape_dataset(
    sample_count: int,
    image_size: int,
    seed: int,
    probability: float,
    maximum_angle_degrees: float,
) -> tuple[Tensor, Tensor]:
    """Render a training set with random square-angle domain variation."""
    if type(sample_count) is not int or sample_count < 2:
        raise ValueError("sample_count must be an integer of at least two")
    if type(image_size) is not int or image_size < 16:
        raise ValueError("image_size must be an integer of at least 16")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if (
        not isinstance(probability, Real)
        or isinstance(probability, bool)
        or not 0.0 <= probability <= 1.0
    ):
        raise ValueError("probability must be between zero and one")
    if (
        not isinstance(maximum_angle_degrees, Real)
        or isinstance(maximum_angle_degrees, bool)
        or not math.isfinite(float(maximum_angle_degrees))
        or maximum_angle_degrees <= 0.0
    ):
        raise ValueError("maximum_angle_degrees must be positive and finite")

    rng = np.random.default_rng(seed)
    labels = np.arange(sample_count, dtype=np.int64) % 2
    rng.shuffle(labels)
    images_bgr = np.empty(
        (sample_count, image_size, image_size, 3),
        dtype=np.uint8,
    )

    for index, label in enumerate(labels):
        image = rng.integers(
            0,
            31,
            size=(image_size, image_size, 3),
            dtype=np.uint8,
        )
        shape_size = int(
            rng.integers(image_size // 4, image_size // 2 + 1)
        )
        half_size = shape_size // 2
        rotation_margin = math.ceil(half_size * math.sqrt(2.0))
        center_x = int(
            rng.integers(rotation_margin, image_size - rotation_margin)
        )
        center_y = int(
            rng.integers(rotation_margin, image_size - rotation_margin)
        )
        color = tuple(
            int(value) for value in rng.integers(150, 256, size=3)
        )

        if label == 0 and rng.random() < probability:
            angle = float(
                rng.uniform(
                    -maximum_angle_degrees,
                    maximum_angle_degrees,
                )
            )
            rotated_square = cv2.boxPoints(
                (
                    (float(center_x), float(center_y)),
                    (float(shape_size), float(shape_size)),
                    angle,
                )
            ).astype(np.int32)
            cv2.fillConvexPoly(image, rotated_square, color)
        elif label == 0:
            cv2.rectangle(
                image,
                (center_x - half_size, center_y - half_size),
                (center_x + half_size, center_y + half_size),
                color,
                thickness=-1,
            )
        else:
            cv2.circle(
                image,
                (center_x, center_y),
                half_size,
                color,
                thickness=-1,
            )
        images_bgr[index] = image

    images_rgb = images_bgr[..., ::-1].copy()
    image_tensor = (
        torch.from_numpy(images_rgb)
        .permute(0, 3, 1, 2)
        .to(dtype=torch.float32)
        .div(255.0)
    )
    return image_tensor, torch.from_numpy(labels)


def split_classification_dataset(
    images: Tensor,
    labels: Tensor,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> ClassificationDatasetSplits:
    """Split image tensors using reproducible, non-overlapping indices."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape (N, 3, H, W)")
    if labels.ndim != 1 or labels.dtype != torch.int64:
        raise ValueError("labels must be a one-dimensional int64 tensor")
    if images.shape[0] != labels.shape[0]:
        raise ValueError("images and labels must have the same sample count")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    for value, name in (
        (train_fraction, "train_fraction"),
        (validation_fraction, "validation_fraction"),
    ):
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not 0 < value < 1
        ):
            raise ValueError(f"{name} must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("dataset fractions must leave a test split")

    sample_count = images.shape[0]
    train_count = int(sample_count * train_fraction)
    validation_count = int(sample_count * validation_fraction)
    test_count = sample_count - train_count - validation_count
    if min(train_count, validation_count, test_count) <= 0:
        raise ValueError("each dataset split must contain at least one sample")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(sample_count, generator=generator)
    train_end = train_count
    validation_end = train_count + validation_count
    train_indices = indices[:train_end]
    validation_indices = indices[train_end:validation_end]
    test_indices = indices[validation_end:]

    return ClassificationDatasetSplits(
        train_images=images[train_indices],
        train_labels=labels[train_indices],
        validation_images=images[validation_indices],
        validation_labels=labels[validation_indices],
        test_images=images[test_indices],
        test_labels=labels[test_indices],
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )


def create_classification_data_loader(
    images: Tensor,
    labels: Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Build a reproducible image classification DataLoader."""
    _validate_classification_tensors(images, labels)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        TensorDataset(images, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def train_classification_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
) -> float:
    """Train for one epoch and return sample-weighted cross-entropy."""
    model.train()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    sample_count = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        sample_count += batch_size

    return total_loss / sample_count


def evaluate_classifier(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> ClassificationEvaluation:
    """Evaluate loss, accuracy, and confusion matrix without gradients."""
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least two")

    model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    sample_count = 0
    correct_count = 0
    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
    )
    prediction_batches: list[Tensor] = []
    label_batches: list[Tensor] = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_function(logits, labels)
            predictions = logits.argmax(dim=1)

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            sample_count += batch_size
            correct_count += int((predictions == labels).sum().item())

            labels_cpu = labels.cpu()
            predictions_cpu = predictions.cpu()
            for true_label, predicted_label in zip(
                labels_cpu,
                predictions_cpu,
            ):
                confusion_matrix[true_label, predicted_label] += 1
            prediction_batches.append(predictions_cpu)
            label_batches.append(labels_cpu)

    if sample_count == 0:
        raise ValueError("evaluation data loader cannot be empty")
    return ClassificationEvaluation(
        loss=total_loss / sample_count,
        accuracy=correct_count / sample_count,
        confusion_matrix=confusion_matrix,
        predictions=torch.cat(prediction_batches),
        labels=torch.cat(label_batches),
    )


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    num_classes: int,
) -> ClassificationTrainingResult:
    """Train a classifier and restore the lowest validation-loss state."""
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if (
        not isinstance(learning_rate, Real)
        or isinstance(learning_rate, bool)
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be positive and finite")
    if (
        not isinstance(weight_decay, Real)
        or isinstance(weight_decay, bool)
        or not math.isfinite(float(weight_decay))
        or weight_decay < 0
    ):
        raise ValueError("weight_decay must be finite and non-negative")

    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    training_losses: list[float] = []
    validation_losses: list[float] = []
    validation_accuracies: list[float] = []
    best_epoch = 0
    best_validation_loss = float("inf")
    best_state = deepcopy(model.state_dict())

    for epoch_index in range(epochs):
        training_loss = train_classification_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )
        validation = evaluate_classifier(
            model,
            validation_loader,
            device,
            num_classes,
        )
        training_losses.append(training_loss)
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


def _validate_classification_tensors(
    images: Tensor,
    labels: Tensor,
) -> None:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape (N, 3, H, W)")
    if not images.is_floating_point():
        raise ValueError("images must use a floating-point dtype")
    if labels.ndim != 1 or labels.dtype != torch.int64:
        raise ValueError("labels must be a one-dimensional int64 tensor")
    if images.shape[0] != labels.shape[0] or images.shape[0] == 0:
        raise ValueError("images and labels need the same non-zero count")
