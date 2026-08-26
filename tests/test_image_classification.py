"""Unit tests for synthetic shape data and the small CNN interface."""

import pytest
import torch
from torch.optim import AdamW

from robot_learning.image_classification import (
    SmallShapeCNN,
    create_classification_data_loader,
    evaluate_classifier,
    generate_shape_classification_dataset,
    generate_shape_challenge_dataset,
    generate_rotation_augmented_shape_dataset,
    split_classification_dataset,
    train_classification_epoch,
)


def test_shape_dataset_has_model_ready_contract() -> None:
    images, labels = generate_shape_classification_dataset(
        sample_count=12,
        image_size=32,
        seed=42,
    )

    assert images.shape == (12, 3, 32, 32)
    assert images.dtype == torch.float32
    assert images.min().item() >= 0.0
    assert images.max().item() <= 1.0
    assert labels.shape == (12,)
    assert labels.dtype == torch.int64
    assert torch.bincount(labels).tolist() == [6, 6]


def test_shape_dataset_is_reproducible_for_the_same_seed() -> None:
    first_images, first_labels = generate_shape_classification_dataset(
        8,
        image_size=32,
        seed=7,
    )
    second_images, second_labels = generate_shape_classification_dataset(
        8,
        image_size=32,
        seed=7,
    )

    assert torch.equal(first_images, second_images)
    assert torch.equal(first_labels, second_labels)


def test_challenge_dataset_is_reproducible_and_model_ready() -> None:
    first_images, first_labels = generate_shape_challenge_dataset(
        12,
        image_size=32,
        seed=17,
    )
    second_images, second_labels = generate_shape_challenge_dataset(
        12,
        image_size=32,
        seed=17,
    )

    assert first_images.shape == (12, 3, 32, 32)
    assert first_images.dtype == torch.float32
    assert first_images.min().item() >= 0.0
    assert first_images.max().item() <= 1.0
    assert torch.bincount(first_labels).tolist() == [6, 6]
    assert torch.equal(first_images, second_images)
    assert torch.equal(first_labels, second_labels)


def test_rotation_augmentation_is_reproducible_and_preserves_contract() -> None:
    first_images, first_labels = generate_rotation_augmented_shape_dataset(
        sample_count=8,
        image_size=32,
        seed=22,
        probability=1.0,
        maximum_angle_degrees=70.0,
    )
    second_images, second_labels = generate_rotation_augmented_shape_dataset(
        sample_count=8,
        image_size=32,
        seed=22,
        probability=1.0,
        maximum_angle_degrees=70.0,
    )

    assert first_images.shape == (8, 3, 32, 32)
    assert first_images.dtype == torch.float32
    assert first_images.min().item() >= 0.0
    assert first_images.max().item() <= 1.0
    assert torch.bincount(first_labels).tolist() == [4, 4]
    assert torch.equal(first_images, second_images)
    assert torch.equal(first_labels, second_labels)


def test_classification_splits_have_expected_non_overlapping_sizes() -> None:
    images, labels = generate_shape_classification_dataset(40, 32, seed=3)

    splits = split_classification_dataset(
        images,
        labels,
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


def test_small_cnn_returns_one_logit_vector_per_image() -> None:
    model = SmallShapeCNN(num_classes=2)
    images = torch.rand(5, 3, 32, 32)

    logits = model(images)

    assert logits.shape == (5, 2)
    assert logits.dtype == torch.float32


def test_small_cnn_rejects_wrong_image_layout() -> None:
    model = SmallShapeCNN()

    with pytest.raises(ValueError, match="N, 3, H, W"):
        model(torch.rand(5, 32, 32, 3))


def test_classification_loader_preserves_batch_contract() -> None:
    images, labels = generate_shape_classification_dataset(10, 32, seed=4)
    data_loader = create_classification_data_loader(
        images,
        labels,
        batch_size=4,
        shuffle=False,
        seed=9,
    )

    batch_images, batch_labels = next(iter(data_loader))

    assert batch_images.shape == (4, 3, 32, 32)
    assert batch_labels.shape == (4,)
    assert batch_labels.dtype == torch.int64


def test_one_classification_epoch_changes_model_parameters() -> None:
    torch.manual_seed(5)
    model = SmallShapeCNN()
    images, labels = generate_shape_classification_dataset(16, 32, seed=5)
    data_loader = create_classification_data_loader(
        images,
        labels,
        batch_size=8,
        shuffle=True,
        seed=5,
    )
    optimizer = AdamW(model.parameters(), lr=0.001)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = train_classification_epoch(
        model,
        data_loader,
        optimizer,
        device=torch.device("cpu"),
    )

    assert loss > 0.0
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters())
    )


def test_classifier_evaluation_statistics_are_consistent() -> None:
    model = SmallShapeCNN()
    images, labels = generate_shape_classification_dataset(10, 32, seed=8)
    data_loader = create_classification_data_loader(
        images,
        labels,
        batch_size=4,
        shuffle=False,
        seed=8,
    )

    evaluation = evaluate_classifier(
        model,
        data_loader,
        device=torch.device("cpu"),
        num_classes=2,
    )

    assert 0.0 <= evaluation.accuracy <= 1.0
    assert evaluation.confusion_matrix.shape == (2, 2)
    assert evaluation.confusion_matrix.sum().item() == 10
    assert evaluation.predictions.shape == (10,)
    assert torch.equal(evaluation.labels, labels)


@pytest.mark.parametrize(
    ("sample_count", "image_size"),
    [(1, 32), (8, 8), (8.0, 32)],
)
def test_shape_dataset_rejects_invalid_dimensions(
    sample_count,
    image_size,
) -> None:
    with pytest.raises(ValueError):
        generate_shape_classification_dataset(
            sample_count,
            image_size,
            seed=42,
        )
