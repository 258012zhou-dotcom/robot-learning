"""Synthetic semantic-segmentation data and a compact encoder-decoder."""

from copy import deepcopy
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class SegmentationDatasetSplits:
    """Train, validation, and test tensors with original sample indices."""

    train_images: Tensor
    train_masks: Tensor
    validation_images: Tensor
    validation_masks: Tensor
    test_images: Tensor
    test_masks: Tensor
    train_indices: Tensor
    validation_indices: Tensor
    test_indices: Tensor


@dataclass(frozen=True)
class SegmentationMetrics:
    """Pixel accuracy and intersection-over-union statistics."""

    pixel_accuracy: float
    per_class_iou: Tensor
    mean_iou: float
    foreground_mean_iou: float


@dataclass(frozen=True)
class SegmentationEvaluation:
    """Loss, masks, confusion matrix, and segmentation metrics."""

    loss: float
    predicted_masks: Tensor
    target_masks: Tensor
    confusion_matrix: Tensor
    metrics: SegmentationMetrics


@dataclass(frozen=True)
class SegmentationTrainingResult:
    """Training history and the best validation checkpoint."""

    training_losses: list[float]
    validation_losses: list[float]
    best_epoch: int
    best_validation_loss: float


class ConvolutionBlock(nn.Module):
    """Two spatially preserving convolution and ReLU layers."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Apply both convolution layers."""
        return self.layers(features)


class SmallSemanticSegmenter(nn.Module):
    """A small U-Net-style model with two skip connections."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least two")

        self.encoder_level_1 = ConvolutionBlock(3, 16)
        self.encoder_level_2 = ConvolutionBlock(16, 32)
        self.bottleneck = ConvolutionBlock(32, 64)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.decoder_level_2 = ConvolutionBlock(64 + 32, 32)
        self.decoder_level_1 = ConvolutionBlock(32 + 16, 16)
        self.segmentation_head = nn.Conv2d(
            16,
            num_classes,
            kernel_size=1,
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return one class-logit vector for every input pixel."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (N, 3, H, W)")

        encoder_1 = self.encoder_level_1(images)
        encoder_2 = self.encoder_level_2(self.pool(encoder_1))
        bottleneck = self.bottleneck(self.pool(encoder_2))

        upsampled_2 = functional.interpolate(
            bottleneck,
            size=encoder_2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoder_2 = self.decoder_level_2(
            torch.cat((upsampled_2, encoder_2), dim=1)
        )
        upsampled_1 = functional.interpolate(
            decoder_2,
            size=encoder_1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoder_1 = self.decoder_level_1(
            torch.cat((upsampled_1, encoder_1), dim=1)
        )
        return self.segmentation_head(decoder_1)


def generate_shape_segmentation_dataset(
    sample_count: int,
    image_size: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Generate RGB images and masks for background, square, and circle."""
    if type(sample_count) is not int or sample_count < 2:
        raise ValueError("sample_count must be an integer of at least two")
    if type(image_size) is not int or image_size < 32:
        raise ValueError("image_size must be an integer of at least 32")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(seed)
    foreground_classes = 1 + np.arange(sample_count, dtype=np.int64) % 2
    rng.shuffle(foreground_classes)
    images_bgr = np.empty(
        (sample_count, image_size, image_size, 3),
        dtype=np.uint8,
    )
    masks = np.zeros(
        (sample_count, image_size, image_size),
        dtype=np.int64,
    )

    for index, foreground_class in enumerate(foreground_classes):
        image = rng.integers(
            0,
            31,
            size=(image_size, image_size, 3),
            dtype=np.uint8,
        )
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        object_size = int(
            rng.integers(image_size // 5, image_size // 2 + 1)
        )
        x_min = int(rng.integers(0, image_size - object_size + 1))
        y_min = int(rng.integers(0, image_size - object_size + 1))
        x_max = x_min + object_size
        y_max = y_min + object_size
        color = tuple(
            int(value) for value in rng.integers(150, 256, size=3)
        )

        if foreground_class == 1:
            cv2.rectangle(
                image,
                (x_min, y_min),
                (x_max - 1, y_max - 1),
                color,
                thickness=-1,
            )
            cv2.rectangle(
                mask,
                (x_min, y_min),
                (x_max - 1, y_max - 1),
                int(foreground_class),
                thickness=-1,
            )
        else:
            center = (
                x_min + object_size // 2,
                y_min + object_size // 2,
            )
            radius = max(1, (object_size - 1) // 2)
            cv2.circle(image, center, radius, color, thickness=-1)
            cv2.circle(
                mask,
                center,
                radius,
                int(foreground_class),
                thickness=-1,
            )

        images_bgr[index] = image
        masks[index] = mask

    images_rgb = images_bgr[..., ::-1].copy()
    image_tensor = (
        torch.from_numpy(images_rgb)
        .permute(0, 3, 1, 2)
        .to(dtype=torch.float32)
        .div(255.0)
    )
    return image_tensor, torch.from_numpy(masks)


def calculate_segmentation_loss(
    logits: Tensor,
    masks: Tensor,
    class_weights: Tensor | None = None,
) -> Tensor:
    """Calculate multiclass pixel-wise cross-entropy loss."""
    if logits.ndim != 4 or logits.shape[1] < 2:
        raise ValueError("logits must have shape (N, C, H, W), with C >= 2")
    if masks.ndim != 3:
        raise ValueError("masks must have shape (N, H, W)")
    if logits.shape[0] != masks.shape[0]:
        raise ValueError("logits and masks must have equal batch sizes")
    if logits.shape[-2:] != masks.shape[-2:]:
        raise ValueError("logits and masks must have equal spatial sizes")
    if masks.dtype != torch.int64:
        raise ValueError("masks must use torch.int64 class indices")
    if masks.numel() == 0:
        raise ValueError("masks must not be empty")
    if masks.min().item() < 0 or masks.max().item() >= logits.shape[1]:
        raise ValueError("mask class indices must match the logit channels")
    if class_weights is not None:
        if class_weights.ndim != 1 or class_weights.shape[0] != logits.shape[1]:
            raise ValueError("class_weights must have one value per class")
        if torch.any(class_weights <= 0.0) or not torch.isfinite(
            class_weights
        ).all():
            raise ValueError("class_weights must be finite and positive")
        class_weights = class_weights.to(
            device=logits.device,
            dtype=logits.dtype,
        )
    return functional.cross_entropy(logits, masks, weight=class_weights)


def split_segmentation_dataset(
    images: Tensor,
    masks: Tensor,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> SegmentationDatasetSplits:
    """Create reproducible, non-overlapping segmentation data splits."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape (N, 3, H, W)")
    if masks.ndim != 3:
        raise ValueError("masks must have shape (N, H, W)")
    if images.shape[0] != masks.shape[0]:
        raise ValueError("images and masks must have equal sample counts")
    sample_count = images.shape[0]
    if sample_count < 3:
        raise ValueError("at least three samples are required")
    if (
        not 0.0 < train_fraction < 1.0
        or not 0.0 < validation_fraction < 1.0
        or train_fraction + validation_fraction >= 1.0
    ):
        raise ValueError("split fractions must leave a non-empty test fraction")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    train_count = int(sample_count * train_fraction)
    validation_count = int(sample_count * validation_fraction)
    if train_count == 0 or validation_count == 0:
        raise ValueError("train and validation splits must not be empty")
    if train_count + validation_count >= sample_count:
        raise ValueError("test split must not be empty")

    indices = torch.randperm(
        sample_count,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    train_indices = indices[:train_count]
    validation_indices = indices[
        train_count:train_count + validation_count
    ]
    test_indices = indices[train_count + validation_count:]
    return SegmentationDatasetSplits(
        train_images=images[train_indices],
        train_masks=masks[train_indices],
        validation_images=images[validation_indices],
        validation_masks=masks[validation_indices],
        test_images=images[test_indices],
        test_masks=masks[test_indices],
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )


def create_segmentation_data_loader(
    images: Tensor,
    masks: Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Build a reproducible loader of image and pixel-mask batches."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape (N, 3, H, W)")
    if masks.ndim != 3:
        raise ValueError("masks must have shape (N, H, W)")
    if images.shape[0] != masks.shape[0]:
        raise ValueError("images and masks must have equal sample counts")
    if images.shape[0] == 0:
        raise ValueError("data loader requires at least one sample")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    return DataLoader(
        TensorDataset(images, masks),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )


def calculate_segmentation_class_weights(
    masks: Tensor,
    num_classes: int,
) -> Tensor:
    """Return inverse-frequency weights with average class weight one."""
    if masks.ndim != 3 or masks.dtype != torch.int64:
        raise ValueError("masks must have shape (N, H, W) and dtype int64")
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least two")
    counts = torch.bincount(masks.flatten(), minlength=num_classes).float()
    if counts.shape[0] != num_classes or torch.any(counts == 0):
        raise ValueError("every segmentation class must occur in the masks")
    inverse_frequencies = counts.sum() / counts
    return inverse_frequencies / inverse_frequencies.mean()


def segmentation_confusion_matrix(
    predicted_masks: Tensor,
    target_masks: Tensor,
    num_classes: int,
) -> Tensor:
    """Return a row=true, column=predicted pixel confusion matrix."""
    if predicted_masks.shape != target_masks.shape or predicted_masks.ndim != 3:
        raise ValueError("predicted and target masks must share shape (N, H, W)")
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least two")
    if predicted_masks.dtype != torch.int64 or target_masks.dtype != torch.int64:
        raise ValueError("predicted and target masks must use int64")
    if (
        predicted_masks.min().item() < 0
        or target_masks.min().item() < 0
        or predicted_masks.max().item() >= num_classes
        or target_masks.max().item() >= num_classes
    ):
        raise ValueError("mask values must be valid class indices")

    pair_indices = target_masks.flatten() * num_classes
    pair_indices = pair_indices + predicted_masks.flatten()
    return torch.bincount(
        pair_indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def segmentation_metrics_from_confusion(
    confusion_matrix: Tensor,
) -> SegmentationMetrics:
    """Calculate pixel accuracy and class IoUs from a confusion matrix."""
    if (
        confusion_matrix.ndim != 2
        or confusion_matrix.shape[0] != confusion_matrix.shape[1]
        or confusion_matrix.shape[0] < 2
    ):
        raise ValueError("confusion_matrix must be square with at least two classes")
    matrix = confusion_matrix.to(dtype=torch.float64)
    total_pixels = matrix.sum()
    if total_pixels.item() == 0:
        raise ValueError("confusion_matrix must contain at least one pixel")

    intersections = matrix.diagonal()
    unions = matrix.sum(dim=1) + matrix.sum(dim=0) - intersections
    valid_classes = unions > 0
    per_class_iou = torch.full_like(unions, float("nan"))
    per_class_iou[valid_classes] = (
        intersections[valid_classes] / unions[valid_classes]
    )
    mean_iou = float(per_class_iou[valid_classes].mean().item())
    foreground_ious = per_class_iou[1:]
    valid_foreground = torch.isfinite(foreground_ious)
    foreground_mean_iou = float(
        foreground_ious[valid_foreground].mean().item()
    )
    return SegmentationMetrics(
        pixel_accuracy=float(intersections.sum().item() / total_pixels.item()),
        per_class_iou=per_class_iou,
        mean_iou=mean_iou,
        foreground_mean_iou=foreground_mean_iou,
    )


def train_segmentation_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    class_weights: Tensor,
) -> float:
    """Train one epoch and return a pixel-weighted average loss."""
    model.train()
    total_loss = 0.0
    pixel_count = 0
    for images, masks in data_loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = calculate_segmentation_loss(
            model(images),
            masks,
            class_weights,
        )
        loss.backward()
        optimizer.step()

        batch_pixels = masks.numel()
        total_loss += loss.item() * batch_pixels
        pixel_count += batch_pixels
    return total_loss / pixel_count


def evaluate_segmenter(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_weights: Tensor,
    num_classes: int,
) -> SegmentationEvaluation:
    """Evaluate weighted loss and pixel-level segmentation statistics."""
    model.eval()
    total_loss = 0.0
    pixel_count = 0
    predicted_batches: list[Tensor] = []
    target_batches: list[Tensor] = []
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = calculate_segmentation_loss(
                logits,
                masks,
                class_weights,
            )
            batch_pixels = masks.numel()
            total_loss += loss.item() * batch_pixels
            pixel_count += batch_pixels
            predicted_batches.append(logits.argmax(dim=1).cpu())
            target_batches.append(masks.cpu())

    predicted_masks = torch.cat(predicted_batches)
    target_masks = torch.cat(target_batches)
    confusion = segmentation_confusion_matrix(
        predicted_masks,
        target_masks,
        num_classes,
    )
    return SegmentationEvaluation(
        loss=total_loss / pixel_count,
        predicted_masks=predicted_masks,
        target_masks=target_masks,
        confusion_matrix=confusion,
        metrics=segmentation_metrics_from_confusion(confusion),
    )


def train_segmenter(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    class_weights: Tensor,
    num_classes: int,
) -> SegmentationTrainingResult:
    """Train a segmenter and restore the lowest validation-loss checkpoint."""
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")

    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    training_losses: list[float] = []
    validation_losses: list[float] = []
    best_epoch = 0
    best_validation_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    for epoch_index in range(epochs):
        training_loss = train_segmentation_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weights,
        )
        validation = evaluate_segmenter(
            model,
            validation_loader,
            device,
            class_weights,
            num_classes,
        )
        training_losses.append(training_loss)
        validation_losses.append(validation.loss)
        if validation.loss < best_validation_loss:
            best_validation_loss = validation.loss
            best_epoch = epoch_index + 1
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return SegmentationTrainingResult(
        training_losses=training_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
    )
