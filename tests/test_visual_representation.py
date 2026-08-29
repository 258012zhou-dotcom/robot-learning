"""Unit tests for contrastive visual embeddings and retrieval."""

import pytest
import torch
from torch.optim import AdamW

from robot_learning.image_classification import (
    generate_shape_classification_dataset,
)
from robot_learning.visual_representation import (
    EncoderClassifier,
    SmallVisualEncoder,
    calculate_embedding_statistics,
    create_contrastive_views,
    encode_images,
    evaluate_one_nearest_neighbor,
    nt_xent_loss,
    train_contrastive_epoch,
)


def test_contrastive_views_preserve_contract_but_change_pixels() -> None:
    images, _ = generate_shape_classification_dataset(8, 32, seed=3)

    first_view, second_view = create_contrastive_views(images, seed=10)

    assert first_view.shape == images.shape
    assert second_view.shape == images.shape
    assert first_view.dtype == torch.float32
    assert first_view.min().item() >= 0.0
    assert first_view.max().item() <= 1.0
    assert not torch.equal(first_view, images)
    assert not torch.equal(first_view, second_view)


def test_contrastive_views_are_reproducible_for_same_seed() -> None:
    images, _ = generate_shape_classification_dataset(8, 32, seed=4)

    first_run = create_contrastive_views(images, seed=11)
    second_run = create_contrastive_views(images, seed=11)

    assert torch.equal(first_run[0], second_run[0])
    assert torch.equal(first_run[1], second_run[1])


def test_visual_encoder_returns_unit_length_embeddings() -> None:
    model = SmallVisualEncoder(embedding_dimension=16)

    embeddings = model(torch.rand(5, 3, 32, 32))

    assert embeddings.shape == (5, 16)
    torch.testing.assert_close(
        torch.linalg.vector_norm(embeddings, dim=1),
        torch.ones(5),
    )


def test_visual_encoder_rejects_wrong_image_layout() -> None:
    model = SmallVisualEncoder()

    with pytest.raises(ValueError, match="N, 3, H, W"):
        model(torch.rand(5, 32, 32, 3))


def test_aligned_positive_pairs_have_lower_loss_than_mismatched_pairs() -> None:
    first = torch.eye(4)
    aligned_second = first.clone()
    mismatched_second = first.roll(shifts=1, dims=0)

    aligned_loss = nt_xent_loss(first, aligned_second, temperature=0.1)
    mismatched_loss = nt_xent_loss(
        first,
        mismatched_second,
        temperature=0.1,
    )

    assert aligned_loss < mismatched_loss


def test_contrastive_loss_backpropagates_to_encoder() -> None:
    model = SmallVisualEncoder(embedding_dimension=16)
    images, _ = generate_shape_classification_dataset(8, 32, seed=5)
    first_view, second_view = create_contrastive_views(images, seed=12)

    loss = nt_xent_loss(
        model(first_view),
        model(second_view),
        temperature=0.2,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert model.features[0].weight.grad is not None
    assert model.projection.weight.grad is not None
    assert torch.count_nonzero(model.projection.weight.grad).item() > 0


def test_embedding_statistics_detect_identical_collapsed_vectors() -> None:
    collapsed = torch.ones(6, 8)

    statistics = calculate_embedding_statistics(collapsed)

    assert statistics.mean_dimension_standard_deviation == pytest.approx(0.0)
    assert statistics.mean_pairwise_cosine_similarity == pytest.approx(1.0)


def test_one_nearest_neighbor_uses_cosine_similarity() -> None:
    reference_embeddings = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    )
    reference_labels = torch.tensor([0, 0, 1, 1])
    query_embeddings = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    query_labels = torch.tensor([0, 1])

    evaluation = evaluate_one_nearest_neighbor(
        reference_embeddings,
        reference_labels,
        query_embeddings,
        query_labels,
    )

    assert evaluation.predictions.tolist() == [0, 1]
    assert evaluation.accuracy == pytest.approx(1.0)


def test_encoder_classifier_returns_class_logits() -> None:
    encoder = SmallVisualEncoder(embedding_dimension=12)
    model = EncoderClassifier(encoder, num_classes=2)

    logits = model(torch.rand(5, 3, 32, 32))
    logits.sum().backward()

    assert logits.shape == (5, 2)
    assert encoder.features[0].weight.grad is not None
    assert model.classifier.weight.grad is not None


def test_contrastive_epoch_updates_encoder_parameters() -> None:
    torch.manual_seed(20)
    encoder = SmallVisualEncoder(embedding_dimension=8)
    optimizer = AdamW(encoder.parameters(), lr=0.001)
    images, _ = generate_shape_classification_dataset(12, 32, seed=8)
    original_projection = encoder.projection.weight.detach().clone()

    loss = train_contrastive_epoch(
        encoder,
        images,
        optimizer,
        device=torch.device("cpu"),
        batch_size=6,
        temperature=0.2,
        seed=21,
    )

    assert loss > 0.0
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(original_projection, encoder.projection.weight)


def test_encode_images_preserves_count_and_unit_norm() -> None:
    encoder = SmallVisualEncoder(embedding_dimension=10)
    images, _ = generate_shape_classification_dataset(7, 32, seed=9)

    embeddings = encode_images(
        encoder,
        images,
        device=torch.device("cpu"),
        batch_size=3,
    )

    assert embeddings.shape == (7, 10)
    assert embeddings.device.type == "cpu"
    torch.testing.assert_close(
        torch.linalg.vector_norm(embeddings, dim=1),
        torch.ones(7),
    )


def test_contrastive_epoch_rejects_single_sample_batches() -> None:
    encoder = SmallVisualEncoder()
    optimizer = AdamW(encoder.parameters(), lr=0.001)
    images = torch.rand(4, 3, 32, 32)

    with pytest.raises(ValueError, match="at least two"):
        train_contrastive_epoch(
            encoder,
            images,
            optimizer,
            device=torch.device("cpu"),
            batch_size=1,
            temperature=0.2,
            seed=22,
        )
