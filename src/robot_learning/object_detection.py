"""Synthetic single-object data and a small trainable detector."""

from copy import deepcopy
from dataclasses import dataclass
from numbers import Real

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class DetectionLoss:
    """Total detection loss and its detached component values."""

    total: Tensor
    classification: Tensor
    box_regression: Tensor


@dataclass(frozen=True)
class DetectionDatasetSplits:
    """Train, validation, and test tensors with original sample indices."""

    train_images: Tensor
    train_labels: Tensor
    train_boxes: Tensor
    validation_images: Tensor
    validation_labels: Tensor
    validation_boxes: Tensor
    test_images: Tensor
    test_labels: Tensor
    test_boxes: Tensor
    train_indices: Tensor
    validation_indices: Tensor
    test_indices: Tensor


@dataclass(frozen=True)
class DetectionEvaluation:
    """Losses, predictions, IoUs, and detection quality statistics."""

    total_loss: float
    classification_loss: float
    box_regression_loss: float
    class_accuracy: float
    mean_iou: float
    detection_success_rate: float
    predicted_labels: Tensor
    predicted_boxes: Tensor
    target_labels: Tensor
    target_boxes: Tensor
    ious: Tensor


@dataclass(frozen=True)
class DetectionTrainingResult:
    """Training history and the best validation checkpoint."""

    training_losses: list[float]
    validation_losses: list[float]
    best_epoch: int
    best_validation_loss: float


class SmallShapeDetector(nn.Module):
    """A compact CNN for one-class label and one bounding box per image."""

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
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
        )
        self.classification_head = nn.Linear(128, num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Return class logits and normalized cxcywh boxes."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (N, 3, H, W)")
        shared_features = self.shared(self.features(images))
        return (
            self.classification_head(shared_features),
            self.box_head(shared_features),
        )


def generate_shape_detection_dataset(
    sample_count: int,
    image_size: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate balanced images with one shape and a normalized cxcywh box."""
    if type(sample_count) is not int or sample_count < 2:
        raise ValueError("sample_count must be an integer of at least two")
    if type(image_size) is not int or image_size < 32:
        raise ValueError("image_size must be an integer of at least 32")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(seed)
    labels = np.arange(sample_count, dtype=np.int64) % 2
    rng.shuffle(labels)
    images_bgr = np.empty(
        (sample_count, image_size, image_size, 3),
        dtype=np.uint8,
    )
    boxes_cxcywh = np.empty((sample_count, 4), dtype=np.float32)

    for index, label in enumerate(labels):
        image = rng.integers(
            0,
            31,
            size=(image_size, image_size, 3),
            dtype=np.uint8,
        )
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

        if label == 0:
            cv2.rectangle(
                image,
                (x_min, y_min),
                (x_max - 1, y_max - 1),
                color,
                thickness=-1,
            )
        else:
            center = (
                x_min + object_size // 2,
                y_min + object_size // 2,
            )
            radius = max(1, (object_size - 1) // 2)
            cv2.circle(image, center, radius, color, thickness=-1)

        images_bgr[index] = image
        boxes_cxcywh[index] = (
            (x_min + x_max) / (2.0 * image_size),
            (y_min + y_max) / (2.0 * image_size),
            object_size / image_size,
            object_size / image_size,
        )

    images_rgb = images_bgr[..., ::-1].copy()
    image_tensor = (
        torch.from_numpy(images_rgb)
        .permute(0, 3, 1, 2)
        .to(dtype=torch.float32)
        .div(255.0)
    )
    return (
        image_tensor,
        torch.from_numpy(labels),
        torch.from_numpy(boxes_cxcywh),
    )


def calculate_detection_loss(
    class_logits: Tensor,
    predicted_boxes: Tensor,
    class_labels: Tensor,
    target_boxes: Tensor,
    box_loss_weight: float,
) -> DetectionLoss:
    """Combine cross-entropy classification and Smooth L1 box losses."""
    if class_logits.ndim != 2 or class_logits.shape[1] < 2:
        raise ValueError("class_logits must have shape (N, C), with C >= 2")
    if predicted_boxes.ndim != 2 or predicted_boxes.shape[1] != 4:
        raise ValueError("predicted_boxes must have shape (N, 4)")
    if class_labels.ndim != 1:
        raise ValueError("class_labels must have shape (N,)")
    if target_boxes.ndim != 2 or target_boxes.shape[1] != 4:
        raise ValueError("target_boxes must have shape (N, 4)")
    sample_count = class_logits.shape[0]
    if not (
        predicted_boxes.shape[0]
        == class_labels.shape[0]
        == target_boxes.shape[0]
        == sample_count
    ):
        raise ValueError("all detection tensors must have the same sample count")
    if (
        not isinstance(box_loss_weight, Real)
        or isinstance(box_loss_weight, bool)
        or float(box_loss_weight) <= 0.0
    ):
        raise ValueError("box_loss_weight must be positive")

    classification_loss = functional.cross_entropy(
        class_logits,
        class_labels,
    )
    box_regression_loss = functional.smooth_l1_loss(
        predicted_boxes,
        target_boxes,
    )
    total_loss = (
        classification_loss
        + float(box_loss_weight) * box_regression_loss
    )
    return DetectionLoss(
        total=total_loss,
        classification=classification_loss,
        box_regression=box_regression_loss,
    )


def split_detection_dataset(
    images: Tensor,
    labels: Tensor,
    boxes: Tensor,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> DetectionDatasetSplits:
    """Create reproducible, non-overlapping train/validation/test splits."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape (N, 3, H, W)")
    if labels.ndim != 1:
        raise ValueError("labels must have shape (N,)")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    sample_count = images.shape[0]
    if labels.shape[0] != sample_count or boxes.shape[0] != sample_count:
        raise ValueError("images, labels, and boxes must have equal lengths")
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

    return DetectionDatasetSplits(
        train_images=images[train_indices],
        train_labels=labels[train_indices],
        train_boxes=boxes[train_indices],
        validation_images=images[validation_indices],
        validation_labels=labels[validation_indices],
        validation_boxes=boxes[validation_indices],
        test_images=images[test_indices],
        test_labels=labels[test_indices],
        test_boxes=boxes[test_indices],
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )


def create_detection_data_loader(
    images: Tensor,
    labels: Tensor,
    boxes: Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Build a reproducible loader of image, class, and box batches."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape (N, 3, H, W)")
    if labels.ndim != 1 or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("labels and boxes must have shape (N,) and (N, 4)")
    if images.shape[0] != labels.shape[0] or labels.shape[0] != boxes.shape[0]:
        raise ValueError("images, labels, and boxes must have equal lengths")
    if images.shape[0] == 0:
        raise ValueError("data loader requires at least one sample")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    return DataLoader(
        TensorDataset(images, labels, boxes),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )


def normalized_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert normalized center boxes to clipped normalized xyxy boxes."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    if not torch.isfinite(boxes).all():
        raise ValueError("boxes must contain finite values")
    if torch.any(boxes[:, 2:] <= 0.0):
        raise ValueError("box width and height must be positive")

    centers = boxes[:, :2]
    half_sizes = boxes[:, 2:] / 2.0
    return torch.cat(
        (centers - half_sizes, centers + half_sizes),
        dim=1,
    ).clamp(0.0, 1.0)


def batch_intersection_over_union(
    first_boxes_cxcywh: Tensor,
    second_boxes_cxcywh: Tensor,
) -> Tensor:
    """Compute one IoU per corresponding pair of normalized boxes."""
    if first_boxes_cxcywh.shape != second_boxes_cxcywh.shape:
        raise ValueError("box batches must have the same shape")
    first = normalized_cxcywh_to_xyxy(first_boxes_cxcywh)
    second = normalized_cxcywh_to_xyxy(second_boxes_cxcywh)

    intersection_minimum = torch.maximum(first[:, :2], second[:, :2])
    intersection_maximum = torch.minimum(first[:, 2:], second[:, 2:])
    intersection_sizes = (intersection_maximum - intersection_minimum).clamp(
        min=0.0
    )
    intersection_areas = intersection_sizes.prod(dim=1)
    first_areas = (first[:, 2:] - first[:, :2]).prod(dim=1)
    second_areas = (second[:, 2:] - second[:, :2]).prod(dim=1)
    union_areas = first_areas + second_areas - intersection_areas
    return intersection_areas / union_areas.clamp_min(
        torch.finfo(first.dtype).eps
    )


def train_detection_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    box_loss_weight: float,
) -> float:
    """Train for one epoch and return sample-weighted average total loss."""
    model.train()
    total_loss = 0.0
    sample_count = 0

    for images, labels, target_boxes in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        target_boxes = target_boxes.to(device)

        optimizer.zero_grad(set_to_none=True)
        class_logits, predicted_boxes = model(images)
        loss = calculate_detection_loss(
            class_logits,
            predicted_boxes,
            labels,
            target_boxes,
            box_loss_weight,
        )
        loss.total.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.total.item() * batch_size
        sample_count += batch_size

    return total_loss / sample_count


def evaluate_detector(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    box_loss_weight: float,
    iou_success_threshold: float = 0.5,
) -> DetectionEvaluation:
    """Evaluate losses, class accuracy, IoU, and joint success rate."""
    if not 0.0 <= iou_success_threshold <= 1.0:
        raise ValueError("iou_success_threshold must be in [0, 1]")
    model.eval()
    total_loss_sum = 0.0
    classification_loss_sum = 0.0
    box_loss_sum = 0.0
    predicted_label_batches: list[Tensor] = []
    predicted_box_batches: list[Tensor] = []
    target_label_batches: list[Tensor] = []
    target_box_batches: list[Tensor] = []

    with torch.no_grad():
        for images, labels, target_boxes in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            target_boxes = target_boxes.to(device)
            class_logits, predicted_boxes = model(images)
            loss = calculate_detection_loss(
                class_logits,
                predicted_boxes,
                labels,
                target_boxes,
                box_loss_weight,
            )

            batch_size = images.shape[0]
            total_loss_sum += loss.total.item() * batch_size
            classification_loss_sum += (
                loss.classification.item() * batch_size
            )
            box_loss_sum += loss.box_regression.item() * batch_size
            predicted_label_batches.append(class_logits.argmax(dim=1).cpu())
            predicted_box_batches.append(predicted_boxes.cpu())
            target_label_batches.append(labels.cpu())
            target_box_batches.append(target_boxes.cpu())

    predicted_labels = torch.cat(predicted_label_batches)
    predicted_boxes = torch.cat(predicted_box_batches)
    target_labels = torch.cat(target_label_batches)
    target_boxes = torch.cat(target_box_batches)
    ious = batch_intersection_over_union(predicted_boxes, target_boxes)
    correct_classes = predicted_labels == target_labels
    successful_detections = correct_classes & (
        ious >= iou_success_threshold
    )
    sample_count = target_labels.numel()
    return DetectionEvaluation(
        total_loss=total_loss_sum / sample_count,
        classification_loss=classification_loss_sum / sample_count,
        box_regression_loss=box_loss_sum / sample_count,
        class_accuracy=float(correct_classes.float().mean().item()),
        mean_iou=float(ious.mean().item()),
        detection_success_rate=float(
            successful_detections.float().mean().item()
        ),
        predicted_labels=predicted_labels,
        predicted_boxes=predicted_boxes,
        target_labels=target_labels,
        target_boxes=target_boxes,
        ious=ious,
    )


def train_detector(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    box_loss_weight: float,
) -> DetectionTrainingResult:
    """Train a detector and restore the checkpoint with lowest validation loss."""
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
        training_loss = train_detection_epoch(
            model,
            train_loader,
            optimizer,
            device,
            box_loss_weight,
        )
        validation = evaluate_detector(
            model,
            validation_loader,
            device,
            box_loss_weight,
        )
        training_losses.append(training_loss)
        validation_losses.append(validation.total_loss)
        if validation.total_loss < best_validation_loss:
            best_validation_loss = validation.total_loss
            best_epoch = epoch_index + 1
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return DetectionTrainingResult(
        training_losses=training_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
    )
