"""Unit tests for the small vision-language dual encoder contract."""

import pytest
import torch
from torch.optim import AdamW

from robot_learning.vision_language import (
    SimpleVocabulary,
    SmallTextEncoder,
    VisionLanguageDualEncoder,
    calculate_image_text_similarities,
    create_unique_concept_batches,
    evaluate_aligned_image_text_pairs,
    evaluate_semantic_image_text_retrieval,
    evaluate_semantic_similarity_matrix,
    generate_image_text_dataset,
    symmetric_image_text_contrastive_loss,
    train_vision_language_epoch,
)


def test_vocabulary_is_deterministic_and_reserves_special_tokens() -> None:
    vocabulary = SimpleVocabulary.from_texts(
        ["a red square", "a blue circle"]
    )

    assert vocabulary.padding_id == 0
    assert vocabulary.unknown_id == 1
    assert vocabulary.token_to_id["a"] == 2
    assert vocabulary.token_to_id["blue"] == 3


def test_vocabulary_encodes_unknown_words_and_padding() -> None:
    vocabulary = SimpleVocabulary.from_texts(["a red square"])

    token_ids = vocabulary.encode("a green square", maximum_token_count=4)

    assert token_ids == [
        vocabulary.token_to_id["a"],
        vocabulary.unknown_id,
        vocabulary.token_to_id["square"],
        vocabulary.padding_id,
    ]


def test_vocabulary_rejects_silent_text_truncation() -> None:
    vocabulary = SimpleVocabulary.from_texts(["a red square"])

    with pytest.raises(ValueError, match="exceeds"):
        vocabulary.encode("a bright red square", maximum_token_count=3)


def test_text_encoder_returns_unit_length_embeddings() -> None:
    encoder = SmallTextEncoder(
        vocabulary_size=10,
        maximum_token_count=4,
        embedding_dimension=16,
        head_count=4,
    )
    token_ids = torch.tensor(
        [[2, 3, 4, 0], [2, 5, 6, 7]],
        dtype=torch.int64,
    )

    embeddings = encoder(token_ids)

    assert embeddings.shape == (2, 16)
    torch.testing.assert_close(
        torch.linalg.vector_norm(embeddings, dim=1),
        torch.ones(2),
    )


def test_text_encoder_rejects_all_padding_sequence() -> None:
    encoder = SmallTextEncoder(
        vocabulary_size=8,
        maximum_token_count=3,
        embedding_dimension=8,
        head_count=2,
    )

    with pytest.raises(ValueError, match="non-padding"):
        encoder(torch.zeros((1, 3), dtype=torch.int64))


def test_dual_encoder_produces_shared_dimension_embeddings() -> None:
    model = VisionLanguageDualEncoder(
        vocabulary_size=10,
        maximum_token_count=4,
        embedding_dimension=16,
        text_head_count=4,
    )
    images = torch.rand(3, 3, 32, 32)
    token_ids = torch.tensor(
        [[2, 3, 4, 0], [2, 5, 6, 0], [2, 7, 4, 0]],
        dtype=torch.int64,
    )

    embeddings = model(images, token_ids)
    similarities = calculate_image_text_similarities(
        embeddings.image_embeddings,
        embeddings.text_embeddings,
    )

    assert embeddings.image_embeddings.shape == (3, 16)
    assert embeddings.text_embeddings.shape == (3, 16)
    assert similarities.shape == (3, 3)


def test_similarity_matrix_prefers_aligned_handcrafted_pairs() -> None:
    image_embeddings = torch.eye(3)
    text_embeddings = torch.eye(3)

    similarities = calculate_image_text_similarities(
        image_embeddings,
        text_embeddings,
        temperature=0.5,
    )

    assert similarities.argmax(dim=1).tolist() == [0, 1, 2]
    torch.testing.assert_close(similarities.diagonal(), torch.full((3,), 2.0))


def test_similarity_rejects_mismatched_embedding_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions must match"):
        calculate_image_text_similarities(
            torch.rand(2, 8),
            torch.rand(3, 7),
        )


def test_image_text_dataset_is_balanced_and_aligned() -> None:
    dataset = generate_image_text_dataset(
        samples_per_concept=3,
        image_size=32,
        seed=40,
    )

    assert dataset.images.shape == (18, 3, 32, 32)
    assert len(dataset.descriptions) == 18
    assert dataset.concept_labels.shape == (18,)
    assert len(dataset.concept_descriptions) == 6
    assert torch.bincount(dataset.concept_labels).tolist() == [3] * 6
    for description, label in zip(
        dataset.descriptions,
        dataset.concept_labels.tolist(),
    ):
        assert description == dataset.concept_descriptions[label]


def test_image_text_dataset_is_reproducible() -> None:
    first = generate_image_text_dataset(2, 32, seed=41)
    second = generate_image_text_dataset(2, 32, seed=41)

    assert torch.equal(first.images, second.images)
    assert first.descriptions == second.descriptions
    assert torch.equal(first.concept_labels, second.concept_labels)


def test_aligned_pairs_have_lower_symmetric_loss_than_mismatched_pairs() -> None:
    image_embeddings = torch.eye(4)
    aligned_text = image_embeddings.clone()
    mismatched_text = aligned_text.roll(shifts=1, dims=0)

    aligned_loss = symmetric_image_text_contrastive_loss(
        image_embeddings,
        aligned_text,
        temperature=0.1,
    )
    mismatched_loss = symmetric_image_text_contrastive_loss(
        image_embeddings,
        mismatched_text,
        temperature=0.1,
    )

    assert aligned_loss < mismatched_loss


def test_contrastive_loss_backpropagates_to_both_encoders() -> None:
    descriptions = (
        "a red square",
        "a red circle",
        "a green square",
        "a green circle",
        "a blue square",
        "a blue circle",
    )
    vocabulary = SimpleVocabulary.from_texts(descriptions)
    model = VisionLanguageDualEncoder(
        vocabulary_size=len(vocabulary),
        maximum_token_count=3,
        embedding_dimension=16,
        text_head_count=4,
    )
    dataset = generate_image_text_dataset(1, 32, seed=42)
    token_ids = vocabulary.encode_batch(dataset.descriptions, 3)

    embeddings = model(dataset.images, token_ids)
    loss = symmetric_image_text_contrastive_loss(
        embeddings.image_embeddings,
        embeddings.text_embeddings,
        temperature=0.2,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert model.image_encoder.features[0].weight.grad is not None
    assert model.text_encoder.token_embedding.weight.grad is not None
    assert torch.count_nonzero(
        model.image_encoder.features[0].weight.grad
    ).item() > 0
    assert torch.count_nonzero(
        model.text_encoder.token_embedding.weight.grad
    ).item() > 0


def test_unique_concept_batches_use_every_sample_once() -> None:
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)

    batches = create_unique_concept_batches(labels, seed=50)

    assert len(batches) == 2
    for batch in batches:
        assert torch.unique(labels[batch]).tolist() == [0, 1, 2]
    all_indices = torch.cat(batches)
    assert sorted(all_indices.tolist()) == list(range(6))


def test_unique_concept_batches_reject_imbalanced_data() -> None:
    labels = torch.tensor([0, 0, 1], dtype=torch.int64)

    with pytest.raises(ValueError, match="same number"):
        create_unique_concept_batches(labels, seed=51)


def test_training_epoch_updates_both_encoder_parameters() -> None:
    torch.manual_seed(52)
    dataset = generate_image_text_dataset(2, 32, seed=52)
    vocabulary = SimpleVocabulary.from_texts(dataset.concept_descriptions)
    token_ids = vocabulary.encode_batch(dataset.descriptions, 3)
    model = VisionLanguageDualEncoder(
        vocabulary_size=len(vocabulary),
        maximum_token_count=3,
        embedding_dimension=16,
        text_head_count=4,
    )
    optimizer = AdamW(model.parameters(), lr=0.001)
    original_image_weights = (
        model.image_encoder.features[0].weight.detach().clone()
    )
    original_text_weights = (
        model.text_encoder.token_embedding.weight.detach().clone()
    )

    loss = train_vision_language_epoch(
        model,
        dataset.images,
        token_ids,
        dataset.concept_labels,
        optimizer,
        device=torch.device("cpu"),
        temperature=0.2,
        seed=53,
    )

    assert loss > 0.0
    assert not torch.equal(
        original_image_weights,
        model.image_encoder.features[0].weight,
    )
    assert not torch.equal(
        original_text_weights,
        model.text_encoder.token_embedding.weight,
    )


def test_aligned_pair_evaluation_reports_perfect_handcrafted_retrieval() -> None:
    class HandcraftedDualEncoder(torch.nn.Module):
        def forward(self, images, token_ids):
            from robot_learning.vision_language import VisionLanguageEmbeddings

            return VisionLanguageEmbeddings(
                image_embeddings=images,
                text_embeddings=token_ids.to(dtype=torch.float32),
            )

    identity = torch.eye(4)
    evaluation = evaluate_aligned_image_text_pairs(
        HandcraftedDualEncoder(),
        identity,
        identity.to(dtype=torch.int64),
        device=torch.device("cpu"),
        temperature=0.1,
    )

    assert evaluation.image_to_text_accuracy == pytest.approx(1.0)
    assert evaluation.text_to_image_accuracy == pytest.approx(1.0)
    assert evaluation.similarities.argmax(dim=1).tolist() == [0, 1, 2, 3]


def test_semantic_metrics_accept_any_correct_image_for_each_text() -> None:
    similarities = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [2.5, 0.1, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 2.6, 0.1],
            [0.0, 0.0, 3.0],
            [0.1, 0.0, 2.7],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)

    evaluation = evaluate_semantic_similarity_matrix(similarities, labels)

    assert evaluation.image_to_text_accuracy == pytest.approx(1.0)
    assert evaluation.text_to_image_accuracy == pytest.approx(1.0)
    assert evaluation.mean_similarity_margin > 2.0
    assert evaluation.confusion_matrix.tolist() == [
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
    ]


def test_negative_semantic_margin_exposes_wrong_predictions() -> None:
    similarities = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
    labels = torch.tensor([0, 1], dtype=torch.int64)

    evaluation = evaluate_semantic_similarity_matrix(similarities, labels)

    assert evaluation.image_to_text_accuracy == pytest.approx(0.0)
    assert evaluation.mean_similarity_margin == pytest.approx(-0.5)


def test_semantic_retrieval_encodes_images_in_batches() -> None:
    dataset = generate_image_text_dataset(2, 32, seed=60)
    vocabulary = SimpleVocabulary.from_texts(dataset.concept_descriptions)
    concept_token_ids = vocabulary.encode_batch(
        dataset.concept_descriptions,
        maximum_token_count=3,
    )
    model = VisionLanguageDualEncoder(
        vocabulary_size=len(vocabulary),
        maximum_token_count=3,
        embedding_dimension=16,
        text_head_count=4,
    )

    evaluation = evaluate_semantic_image_text_retrieval(
        model,
        dataset.images,
        dataset.concept_labels,
        concept_token_ids,
        device=torch.device("cpu"),
        temperature=0.2,
        image_batch_size=5,
    )

    assert evaluation.similarities.shape == (12, 6)
    assert evaluation.predictions.shape == (12,)
    assert evaluation.confusion_matrix.shape == (6, 6)
