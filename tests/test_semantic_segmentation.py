"""Unit tests for synthetic masks and the semantic-segmentation model."""

import pytest
import torch
from torch.optim import AdamW

from robot_learning.semantic_segmentation import (
    SmallSemanticSegmenter,
    calculate_segmentation_class_weights,
    calculate_segmentation_loss,
    create_segmentation_data_loader,
    generate_shape_segmentation_dataset,
    segmentation_confusion_matrix,
    segmentation_metrics_from_confusion,
    split_segmentation_dataset,
    train_segmentation_epoch,
)


def test_segmentation_dataset_has_pixel_label_contract() -> None:
    images, masks = generate_shape_segmentation_dataset(
        sample_count=12,
        image_size=64,
        seed=42,
    )

    assert images.shape == (12, 3, 64, 64)
    assert images.dtype == torch.float32
    assert images.min().item() >= 0.0
    assert images.max().item() <= 1.0
    assert masks.shape == (12, 64, 64)
    assert masks.dtype == torch.int64
    assert torch.unique(masks).tolist() == [0, 1, 2]


def test_segmentation_dataset_balances_foreground_shape_classes() -> None:
    _, masks = generate_shape_segmentation_dataset(12, 64, seed=3)
    foreground_classes = torch.tensor(
        [int(mask.max()) for mask in masks]
    )

    assert torch.bincount(foreground_classes, minlength=3).tolist() == [0, 6, 6]


def test_segmentation_dataset_is_reproducible_for_same_seed() -> None:
    first_images, first_masks = generate_shape_segmentation_dataset(
        8,
        64,
        seed=7,
    )
    second_images, second_masks = generate_shape_segmentation_dataset(
        8,
        64,
        seed=7,
    )

    assert torch.equal(first_images, second_images)
    assert torch.equal(first_masks, second_masks)


def test_non_background_mask_matches_bright_object_pixels() -> None:
    images, masks = generate_shape_segmentation_dataset(10, 64, seed=9)
    bright_pixels = images.max(dim=1).values > (100.0 / 255.0)

    assert torch.equal(bright_pixels, masks > 0)


def test_segmenter_restores_original_spatial_resolution() -> None:
    model = SmallSemanticSegmenter(num_classes=3)
    images = torch.rand(5, 3, 64, 64)

    logits = model(images)

    assert logits.shape == (5, 3, 64, 64)


def test_segmenter_handles_non_multiple_of_four_image_size() -> None:
    model = SmallSemanticSegmenter(num_classes=3)

    logits = model(torch.rand(2, 3, 65, 67))

    assert logits.shape == (2, 3, 65, 67)


def test_segmenter_rejects_wrong_image_layout() -> None:
    model = SmallSemanticSegmenter()

    with pytest.raises(ValueError, match="N, 3, H, W"):
        model(torch.rand(5, 64, 64, 3))


def test_segmentation_loss_is_finite_and_backpropagates_to_all_sections() -> None:
    model = SmallSemanticSegmenter(num_classes=3)
    images, masks = generate_shape_segmentation_dataset(4, 64, seed=5)

    loss = calculate_segmentation_loss(model(images), masks)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert model.encoder_level_1.layers[0].weight.grad is not None
    assert model.bottleneck.layers[0].weight.grad is not None
    assert model.decoder_level_1.layers[0].weight.grad is not None
    assert model.segmentation_head.weight.grad is not None


def test_segmentation_loss_rejects_float_masks() -> None:
    logits = torch.rand(2, 3, 32, 32)
    masks = torch.zeros(2, 32, 32, dtype=torch.float32)

    with pytest.raises(ValueError, match="int64"):
        calculate_segmentation_loss(logits, masks)


def test_segmentation_splits_are_non_overlapping() -> None:
    images, masks = generate_shape_segmentation_dataset(40, 32, seed=3)

    splits = split_segmentation_dataset(
        images,
        masks,
        train_fraction=0.7,
        validation_fraction=0.15,
        seed=11,
    )

    assert splits.train_images.shape[0] == 28
    assert splits.validation_images.shape[0] == 6
    assert splits.test_images.shape[0] == 6
    all_indices = torch.cat(
        (
            splits.train_indices,
            splits.validation_indices,
            splits.test_indices,
        )
    )
    assert torch.unique(all_indices).numel() == 40


def test_segmentation_loader_preserves_image_and_mask_contract() -> None:
    images, masks = generate_shape_segmentation_dataset(10, 32, seed=4)
    data_loader = create_segmentation_data_loader(
        images,
        masks,
        batch_size=4,
        shuffle=False,
        seed=4,
    )

    batch_images, batch_masks = next(iter(data_loader))

    assert batch_images.shape == (4, 3, 32, 32)
    assert batch_masks.shape == (4, 32, 32)
    assert batch_masks.dtype == torch.int64


def test_class_weights_reduce_background_dominance() -> None:
    _, masks = generate_shape_segmentation_dataset(20, 32, seed=6)

    weights = calculate_segmentation_class_weights(masks, num_classes=3)

    assert weights.shape == (3,)
    assert weights.mean().item() == pytest.approx(1.0)
    assert weights[0] < weights[1]
    assert weights[0] < weights[2]


def test_pixel_confusion_matrix_uses_row_true_convention() -> None:
    targets = torch.tensor([[[0, 0, 1], [1, 2, 2]]])
    predictions = torch.tensor([[[0, 1, 1], [0, 2, 1]]])

    confusion = segmentation_confusion_matrix(
        predictions,
        targets,
        num_classes=3,
    )

    assert confusion.tolist() == [
        [1, 1, 0],
        [1, 1, 0],
        [0, 1, 1],
    ]


def test_segmentation_metrics_match_hand_calculation() -> None:
    confusion = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 1, 1],
        ]
    )

    metrics = segmentation_metrics_from_confusion(confusion)

    assert metrics.pixel_accuracy == pytest.approx(3.0 / 6.0)
    assert metrics.per_class_iou.tolist() == pytest.approx(
        [1.0 / 3.0, 1.0 / 4.0, 1.0 / 2.0]
    )
    assert metrics.mean_iou == pytest.approx(13.0 / 36.0)
    assert metrics.foreground_mean_iou == pytest.approx(3.0 / 8.0)


def test_all_background_baseline_exposes_accuracy_imbalance() -> None:
    _, targets = generate_shape_segmentation_dataset(10, 32, seed=8)
    predictions = torch.zeros_like(targets)
    confusion = segmentation_confusion_matrix(
        predictions,
        targets,
        num_classes=3,
    )

    metrics = segmentation_metrics_from_confusion(confusion)

    assert metrics.pixel_accuracy > 0.8
    assert metrics.foreground_mean_iou == pytest.approx(0.0)
    assert metrics.mean_iou < metrics.pixel_accuracy


def test_one_segmentation_epoch_changes_model_parameters() -> None:
    torch.manual_seed(5)
    model = SmallSemanticSegmenter(num_classes=3)
    images, masks = generate_shape_segmentation_dataset(8, 32, seed=5)
    data_loader = create_segmentation_data_loader(
        images,
        masks,
        batch_size=4,
        shuffle=True,
        seed=5,
    )
    weights = calculate_segmentation_class_weights(masks, num_classes=3)
    optimizer = AdamW(model.parameters(), lr=0.001)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = train_segmentation_epoch(
        model,
        data_loader,
        optimizer,
        device=torch.device("cpu"),
        class_weights=weights,
    )

    assert loss > 0.0
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters())
    )


@pytest.mark.parametrize(
    ("sample_count", "image_size"),
    [(1, 64), (8, 16), (8.0, 64)],
)
def test_segmentation_dataset_rejects_invalid_dimensions(
    sample_count,
    image_size,
) -> None:
    with pytest.raises(ValueError):
        generate_shape_segmentation_dataset(
            sample_count,
            image_size,
            seed=42,
        )
