"""Tests for the language-conditioned visual-grounding data contract."""

import pytest
import torch

from robot_learning.multimodal_grounding import generate_grounding_dataset
from robot_learning.multimodal_grounding import (
    LanguageOnlyGrounder,
    MultimodalGrounder,
    VisionOnlyGrounder,
    calculate_grounding_loss,
    calculate_mean_center_error,
    predict_fixed_center,
)
from robot_learning.vision_language import SimpleVocabulary


def test_grounding_dataset_has_documented_tensor_shapes() -> None:
    """Every field must share N samples and K objects per scene."""
    dataset = generate_grounding_dataset(
        sample_count=12,
        image_size=48,
        object_count=3,
        seed=10,
    )

    assert dataset.images.shape == (12, 3, 48, 48)
    assert dataset.scene_concept_labels.shape == (12, 3)
    assert dataset.scene_centers.shape == (12, 3, 2)
    assert dataset.target_indices.shape == (12,)
    assert dataset.target_centers.shape == (12, 2)
    assert len(dataset.instructions) == 12


def test_grounding_dataset_uses_normalized_float_images_and_centers() -> None:
    """Model inputs and coordinate targets must remain in the [0, 1] range."""
    dataset = generate_grounding_dataset(8, 48, 3, seed=11)

    assert dataset.images.dtype == torch.float32
    assert dataset.target_centers.dtype == torch.float32
    assert torch.all((0.0 <= dataset.images) & (dataset.images <= 1.0))
    assert torch.all(
        (0.0 <= dataset.target_centers)
        & (dataset.target_centers <= 1.0)
    )


def test_instruction_names_the_selected_scene_object() -> None:
    """The instruction label must agree with target_indices for every sample."""
    dataset = generate_grounding_dataset(30, 48, 4, seed=12)

    for sample_index, instruction in enumerate(dataset.instructions):
        target_index = int(dataset.target_indices[sample_index])
        target_label = int(
            dataset.scene_concept_labels[sample_index, target_index]
        )
        expected = f"select the {dataset.concept_descriptions[target_label]}"
        assert instruction == expected


def test_target_center_is_gathered_from_the_selected_object() -> None:
    """Coordinate supervision must point to the object named by the instruction."""
    dataset = generate_grounding_dataset(20, 48, 3, seed=13)
    sample_indices = torch.arange(20)
    expected_centers = dataset.scene_centers[
        sample_indices,
        dataset.target_indices,
    ]

    torch.testing.assert_close(dataset.target_centers, expected_centers)


def test_concepts_are_unique_inside_each_scene() -> None:
    """Unique concepts ensure that each instruction has exactly one answer."""
    dataset = generate_grounding_dataset(20, 48, 5, seed=14)

    for labels in dataset.scene_concept_labels:
        assert torch.unique(labels).numel() == 5


def test_grounding_dataset_is_reproducible() -> None:
    """The same seed must reproduce inputs and all supervision labels."""
    first = generate_grounding_dataset(10, 48, 3, seed=15)
    second = generate_grounding_dataset(10, 48, 3, seed=15)

    assert torch.equal(first.images, second.images)
    assert first.instructions == second.instructions
    assert torch.equal(
        first.scene_concept_labels,
        second.scene_concept_labels,
    )
    assert torch.equal(first.scene_centers, second.scene_centers)
    assert torch.equal(first.target_indices, second.target_indices)
    assert torch.equal(first.target_centers, second.target_centers)


@pytest.mark.parametrize("object_count", [1, 7, 2.5, True])
def test_grounding_dataset_rejects_invalid_object_count(object_count) -> None:
    """A scene needs at least two and at most six unique concepts."""
    with pytest.raises(ValueError, match="object_count"):
        generate_grounding_dataset(
            sample_count=4,
            image_size=48,
            object_count=object_count,
            seed=16,
        )


def test_fixed_center_baseline_ignores_modalities() -> None:
    """The weakest baseline always predicts normalized image center (0.5, 0.5)."""
    output = predict_fixed_center(sample_count=3)

    torch.testing.assert_close(
        output.predicted_centers,
        torch.full((3, 2), 0.5),
    )
    assert output.attention_weights is None


def test_vision_only_model_returns_spatial_attention() -> None:
    """A 64x64 image becomes an 8x8 grid, or 64 attention probabilities."""
    model = VisionOnlyGrounder(embedding_dimension=16)
    images = torch.rand(4, 3, 64, 64)

    output = model(images)

    assert output.predicted_centers.shape == (4, 2)
    assert output.attention_weights is not None
    assert output.attention_weights.shape == (4, 64)
    torch.testing.assert_close(
        output.attention_weights.sum(dim=1),
        torch.ones(4),
    )


def test_language_only_model_returns_coordinates_without_attention() -> None:
    """Text can produce a guess, but it has no image regions to attend to."""
    vocabulary = SimpleVocabulary.from_texts(
        ["select the red square", "select the blue circle"]
    )
    token_ids = vocabulary.encode_batch(
        ["select the red square", "select the blue circle"],
        maximum_token_count=4,
    )
    model = LanguageOnlyGrounder(
        vocabulary_size=len(vocabulary),
        maximum_token_count=4,
        embedding_dimension=16,
        text_head_count=4,
        padding_id=vocabulary.padding_id,
    )

    output = model(token_ids)

    assert output.predicted_centers.shape == (2, 2)
    assert torch.all(
        (0.0 <= output.predicted_centers)
        & (output.predicted_centers <= 1.0)
    )
    assert output.attention_weights is None


def test_multimodal_model_connects_language_to_visual_regions() -> None:
    """The fusion model returns one spatial distribution for every image-text pair."""
    dataset = generate_grounding_dataset(3, 64, 3, seed=20)
    vocabulary = SimpleVocabulary.from_texts(dataset.instructions)
    token_ids = vocabulary.encode_batch(
        dataset.instructions,
        maximum_token_count=4,
    )
    model = MultimodalGrounder(
        vocabulary_size=len(vocabulary),
        maximum_token_count=4,
        embedding_dimension=16,
        text_head_count=4,
        padding_id=vocabulary.padding_id,
    )

    output = model(dataset.images, token_ids)

    assert output.predicted_centers.shape == (3, 2)
    assert output.attention_weights is not None
    assert output.attention_weights.shape == (3, 64)
    torch.testing.assert_close(
        output.attention_weights.sum(dim=1),
        torch.ones(3),
    )


def test_grounding_loss_is_zero_for_exact_coordinates() -> None:
    """Exact target centers should have no Smooth L1 coordinate penalty."""
    centers = torch.tensor([[0.2, 0.8], [0.5, 0.5]])

    loss = calculate_grounding_loss(centers, centers)

    assert loss.item() == pytest.approx(0.0)


def test_mean_center_error_uses_euclidean_distance() -> None:
    """A normalized 3-4-5 displacement should produce error 0.5."""
    predicted = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[0.3, 0.4]])

    error = calculate_mean_center_error(predicted, target)

    assert error == pytest.approx(0.5)


def test_multimodal_loss_backpropagates_to_both_encoders() -> None:
    """A fusion loss must train both the visual CNN and the text embeddings."""
    dataset = generate_grounding_dataset(4, 64, 3, seed=21)
    vocabulary = SimpleVocabulary.from_texts(dataset.instructions)
    token_ids = vocabulary.encode_batch(dataset.instructions, 4)
    model = MultimodalGrounder(
        vocabulary_size=len(vocabulary),
        maximum_token_count=4,
        embedding_dimension=16,
        text_head_count=4,
        padding_id=vocabulary.padding_id,
    )

    output = model(dataset.images, token_ids)
    loss = calculate_grounding_loss(
        output.predicted_centers,
        dataset.target_centers,
    )
    loss.backward()

    visual_gradient = model.visual_encoder.features[0].weight.grad
    text_gradient = model.text_encoder.token_embedding.weight.grad
    assert visual_gradient is not None
    assert text_gradient is not None
    assert torch.count_nonzero(visual_gradient).item() > 0
    assert torch.count_nonzero(text_gradient).item() > 0


def test_multimodal_model_rejects_mismatched_batch_sizes() -> None:
    """Three images cannot be paired silently with two language instructions."""
    model = MultimodalGrounder(
        vocabulary_size=10,
        maximum_token_count=4,
        embedding_dimension=16,
        text_head_count=4,
    )

    with pytest.raises(ValueError, match="matching images"):
        model(
            torch.rand(3, 3, 64, 64),
            torch.randint(0, 10, (2, 4)),
        )
