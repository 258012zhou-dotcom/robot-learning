"""Unit tests for synthetic detection data, model, and loss."""

import pytest
import torch
from torch.optim import AdamW

from robot_learning.object_detection import (
    SmallShapeDetector,
    batch_intersection_over_union,
    calculate_detection_loss,
    create_detection_data_loader,
    evaluate_detector,
    generate_shape_detection_dataset,
    normalized_cxcywh_to_xyxy,
    split_detection_dataset,
    train_detection_epoch,
)


def test_detection_dataset_has_model_ready_contract() -> None:
    images, labels, boxes = generate_shape_detection_dataset(
        sample_count=12,
        image_size=64,
        seed=42,
    )

    assert images.shape == (12, 3, 64, 64)
    assert images.dtype == torch.float32
    assert images.min().item() >= 0.0
    assert images.max().item() <= 1.0
    assert labels.shape == (12,)
    assert labels.dtype == torch.int64
    assert torch.bincount(labels).tolist() == [6, 6]
    assert boxes.shape == (12, 4)
    assert boxes.dtype == torch.float32
    assert torch.all((boxes > 0.0) & (boxes <= 1.0))


def test_detection_dataset_is_reproducible_for_same_seed() -> None:
    first = generate_shape_detection_dataset(8, 64, seed=7)
    second = generate_shape_detection_dataset(8, 64, seed=7)

    assert all(
        torch.equal(first_tensor, second_tensor)
        for first_tensor, second_tensor in zip(first, second)
    )


def test_target_boxes_enclose_all_bright_object_pixels() -> None:
    image_size = 64
    images, _, boxes = generate_shape_detection_dataset(
        sample_count=10,
        image_size=image_size,
        seed=9,
    )

    for image, box in zip(images, boxes):
        center_x, center_y, width, height = box.tolist()
        x_min = (center_x - width / 2.0) * image_size
        y_min = (center_y - height / 2.0) * image_size
        x_max = (center_x + width / 2.0) * image_size
        y_max = (center_y + height / 2.0) * image_size
        bright_mask = image.max(dim=0).values > (100.0 / 255.0)
        bright_y, bright_x = torch.nonzero(bright_mask, as_tuple=True)

        assert float(bright_x.min()) >= x_min - 1e-5
        assert float(bright_y.min()) >= y_min - 1e-5
        assert float(bright_x.max()) < x_max + 1e-5
        assert float(bright_y.max()) < y_max + 1e-5


def test_detector_returns_class_logits_and_normalized_boxes() -> None:
    model = SmallShapeDetector(num_classes=2)
    images = torch.rand(5, 3, 64, 64)

    class_logits, boxes = model(images)

    assert class_logits.shape == (5, 2)
    assert boxes.shape == (5, 4)
    assert torch.all((boxes >= 0.0) & (boxes <= 1.0))


def test_detector_rejects_wrong_image_layout() -> None:
    model = SmallShapeDetector()

    with pytest.raises(ValueError, match="N, 3, H, W"):
        model(torch.rand(5, 64, 64, 3))


def test_detection_loss_combines_both_components() -> None:
    class_logits = torch.tensor(
        [[2.0, -1.0], [-1.0, 2.0]],
        requires_grad=True,
    )
    predicted_boxes = torch.tensor(
        [[0.5, 0.5, 0.2, 0.2], [0.4, 0.6, 0.3, 0.3]],
        requires_grad=True,
    )
    labels = torch.tensor([0, 1])
    target_boxes = torch.tensor(
        [[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.3, 0.3]],
    )

    loss = calculate_detection_loss(
        class_logits,
        predicted_boxes,
        labels,
        target_boxes,
        box_loss_weight=5.0,
    )

    assert loss.total.item() == pytest.approx(
        loss.classification.item() + 5.0 * loss.box_regression.item()
    )
    assert loss.classification.item() > 0.0
    assert loss.box_regression.item() > 0.0


def test_detection_loss_produces_gradients_for_both_heads() -> None:
    model = SmallShapeDetector()
    images, labels, target_boxes = generate_shape_detection_dataset(
        sample_count=4,
        image_size=64,
        seed=5,
    )
    class_logits, predicted_boxes = model(images)

    loss = calculate_detection_loss(
        class_logits,
        predicted_boxes,
        labels,
        target_boxes,
        box_loss_weight=5.0,
    )
    loss.total.backward()

    assert model.classification_head.weight.grad is not None
    assert model.box_head[0].weight.grad is not None
    assert torch.count_nonzero(
        model.classification_head.weight.grad
    ).item() > 0
    assert torch.count_nonzero(model.box_head[0].weight.grad).item() > 0


def test_detection_splits_are_non_overlapping() -> None:
    images, labels, boxes = generate_shape_detection_dataset(40, 64, seed=3)

    splits = split_detection_dataset(
        images,
        labels,
        boxes,
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


def test_detection_loader_preserves_three_part_batch_contract() -> None:
    images, labels, boxes = generate_shape_detection_dataset(10, 64, seed=4)
    data_loader = create_detection_data_loader(
        images,
        labels,
        boxes,
        batch_size=4,
        shuffle=False,
        seed=4,
    )

    batch_images, batch_labels, batch_boxes = next(iter(data_loader))

    assert batch_images.shape == (4, 3, 64, 64)
    assert batch_labels.shape == (4,)
    assert batch_boxes.shape == (4, 4)


def test_normalized_center_box_conversion_is_correct() -> None:
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.2]])

    converted = normalized_cxcywh_to_xyxy(boxes)

    assert converted == pytest.approx(
        torch.tensor([[0.3, 0.4, 0.7, 0.6]])
    )


def test_batch_iou_matches_known_overlap() -> None:
    first = torch.tensor(
        [
            [0.5, 0.5, 1.0, 1.0],
            [0.25, 0.25, 0.5, 0.5],
        ]
    )
    second = torch.tensor(
        [
            [0.5, 0.5, 1.0, 1.0],
            [0.75, 0.75, 0.5, 0.5],
        ]
    )

    ious = batch_intersection_over_union(first, second)

    assert ious.tolist() == pytest.approx([1.0, 0.0])


def test_one_detection_epoch_changes_model_parameters() -> None:
    torch.manual_seed(5)
    model = SmallShapeDetector()
    images, labels, boxes = generate_shape_detection_dataset(12, 64, seed=5)
    data_loader = create_detection_data_loader(
        images,
        labels,
        boxes,
        batch_size=6,
        shuffle=True,
        seed=5,
    )
    optimizer = AdamW(model.parameters(), lr=0.001)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = train_detection_epoch(
        model,
        data_loader,
        optimizer,
        device=torch.device("cpu"),
        box_loss_weight=5.0,
    )

    assert loss > 0.0
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters())
    )


def test_detector_evaluation_statistics_are_consistent() -> None:
    model = SmallShapeDetector()
    images, labels, boxes = generate_shape_detection_dataset(10, 64, seed=8)
    data_loader = create_detection_data_loader(
        images,
        labels,
        boxes,
        batch_size=4,
        shuffle=False,
        seed=8,
    )

    evaluation = evaluate_detector(
        model,
        data_loader,
        device=torch.device("cpu"),
        box_loss_weight=5.0,
    )

    assert 0.0 <= evaluation.class_accuracy <= 1.0
    assert 0.0 <= evaluation.mean_iou <= 1.0
    assert 0.0 <= evaluation.detection_success_rate <= 1.0
    assert evaluation.predicted_labels.shape == (10,)
    assert evaluation.predicted_boxes.shape == (10, 4)
    expected_successes = (
        (evaluation.predicted_labels == evaluation.target_labels)
        & (evaluation.ious >= 0.5)
    )
    assert evaluation.detection_success_rate == pytest.approx(
        expected_successes.float().mean().item()
    )


@pytest.mark.parametrize(
    ("sample_count", "image_size"),
    [(1, 64), (8, 16), (8.0, 64)],
)
def test_detection_dataset_rejects_invalid_dimensions(
    sample_count,
    image_size,
) -> None:
    with pytest.raises(ValueError):
        generate_shape_detection_dataset(
            sample_count,
            image_size,
            seed=42,
        )
