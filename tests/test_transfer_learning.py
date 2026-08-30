"""Unit tests for freezing and discriminative fine-tuning utilities."""

import pytest
import torch

from robot_learning.image_classification import (
    create_classification_data_loader,
    generate_shape_classification_dataset,
)
from robot_learning.transfer_learning import (
    create_source_head_evaluator,
    create_transfer_optimizer,
    set_encoder_trainable,
    train_transfer_classifier,
)
from robot_learning.visual_representation import EncoderClassifier, SmallVisualEncoder


def create_small_loaders():
    """Return deterministic train and validation loaders for update tests."""
    images, labels = generate_shape_classification_dataset(24, 32, seed=50)
    train_loader = create_classification_data_loader(
        images[:16], labels[:16], 8, shuffle=True, seed=51
    )
    validation_loader = create_classification_data_loader(
        images[16:], labels[16:], 8, shuffle=False, seed=51
    )
    return train_loader, validation_loader


def test_frozen_probe_updates_head_but_not_encoder() -> None:
    torch.manual_seed(52)
    model = EncoderClassifier(SmallVisualEncoder(8), num_classes=2)
    set_encoder_trainable(model, False)
    original_encoder = model.encoder.projection.weight.detach().clone()
    original_head = model.classifier.weight.detach().clone()
    train_loader, validation_loader = create_small_loaders()

    train_transfer_classifier(
        model,
        train_loader,
        validation_loader,
        device=torch.device("cpu"),
        epochs=2,
        encoder_learning_rate=0.0001,
        head_learning_rate=0.001,
        weight_decay=0.0,
        num_classes=2,
    )

    torch.testing.assert_close(model.encoder.projection.weight, original_encoder)
    assert not torch.equal(model.classifier.weight, original_head)


def test_full_finetuning_updates_encoder_and_head() -> None:
    torch.manual_seed(53)
    model = EncoderClassifier(SmallVisualEncoder(8), num_classes=2)
    original_encoder = model.encoder.projection.weight.detach().clone()
    original_head = model.classifier.weight.detach().clone()
    train_loader, validation_loader = create_small_loaders()

    train_transfer_classifier(
        model,
        train_loader,
        validation_loader,
        device=torch.device("cpu"),
        epochs=2,
        encoder_learning_rate=0.0002,
        head_learning_rate=0.001,
        weight_decay=0.0,
        num_classes=2,
    )

    assert not torch.equal(model.encoder.projection.weight, original_encoder)
    assert not torch.equal(model.classifier.weight, original_head)


def test_transfer_optimizer_keeps_discriminative_learning_rates() -> None:
    model = EncoderClassifier(SmallVisualEncoder(8), num_classes=2)

    optimizer = create_transfer_optimizer(model, 0.0001, 0.001, 0.0002)

    groups = {group["group_name"]: group["lr"] for group in optimizer.param_groups}
    assert groups == {"encoder": pytest.approx(0.0001), "head": pytest.approx(0.001)}


def test_frozen_optimizer_omits_encoder_parameter_group() -> None:
    model = EncoderClassifier(SmallVisualEncoder(8), num_classes=2)
    set_encoder_trainable(model, False)

    optimizer = create_transfer_optimizer(model, 0.0001, 0.001, 0.0002)

    assert [group["group_name"] for group in optimizer.param_groups] == ["head"]


def test_source_evaluator_combines_adapted_encoder_and_original_head() -> None:
    adapted = EncoderClassifier(SmallVisualEncoder(8), num_classes=2)
    source = EncoderClassifier(SmallVisualEncoder(8), num_classes=2)
    with torch.no_grad():
        adapted.encoder.projection.weight.fill_(2.0)
        adapted.classifier.weight.fill_(3.0)
        source.classifier.weight.fill_(5.0)

    evaluator = create_source_head_evaluator(adapted, source)

    torch.testing.assert_close(
        evaluator.encoder.projection.weight,
        adapted.encoder.projection.weight,
    )
    torch.testing.assert_close(
        evaluator.classifier.weight,
        source.classifier.weight,
    )
    assert not torch.equal(evaluator.classifier.weight, adapted.classifier.weight)
    evaluator_devices = {
        parameter.device for parameter in evaluator.parameters()
    }
    assert evaluator_devices == {
        next(adapted.encoder.parameters()).device
    }
