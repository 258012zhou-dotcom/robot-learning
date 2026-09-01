"""Tests for multimodal-grounding training and evaluation mechanics."""

import pytest
import torch

from robot_learning.multimodal_grounding import (
    MultimodalGrounder,
    VisionOnlyGrounder,
    generate_grounding_dataset,
)
from robot_learning.multimodal_grounding_training import (
    evaluate_grounding_model,
    evaluate_grounding_predictions,
    train_grounding_model,
)
from robot_learning.vision_language import SimpleVocabulary


def _encode_instructions(instructions: tuple[str, ...]):
    """Build the tiny vocabulary shared by tests and encode four-word commands."""
    vocabulary = SimpleVocabulary.from_texts(instructions)
    token_ids = vocabulary.encode_batch(instructions, maximum_token_count=4)
    return vocabulary, token_ids


def test_prediction_evaluation_recovers_nearest_object_accuracy() -> None:
    """Exact coordinates should select both requested candidates correctly."""
    scene_centers = torch.tensor(
        [
            [[0.2, 0.2], [0.8, 0.8]],
            [[0.2, 0.8], [0.8, 0.2]],
        ]
    )
    target_indices = torch.tensor([1, 0])
    target_centers = torch.tensor([[0.8, 0.8], [0.2, 0.8]])

    evaluation = evaluate_grounding_predictions(
        predicted_centers=target_centers.clone(),
        target_centers=target_centers,
        scene_centers=scene_centers,
        target_indices=target_indices,
        loss=0.0,
    )

    assert evaluation.mean_center_error == pytest.approx(0.0)
    assert evaluation.selection_accuracy == pytest.approx(1.0)


def test_prediction_evaluation_detects_wrong_candidate() -> None:
    """A prediction near the distractor must count as a selection failure."""
    scene_centers = torch.tensor([[[0.2, 0.2], [0.8, 0.8]]])

    evaluation = evaluate_grounding_predictions(
        predicted_centers=torch.tensor([[0.2, 0.2]]),
        target_centers=torch.tensor([[0.8, 0.8]]),
        scene_centers=scene_centers,
        target_indices=torch.tensor([1]),
        loss=0.2,
    )

    assert evaluation.mean_center_error > 0.8
    assert evaluation.selection_accuracy == pytest.approx(0.0)


def test_model_evaluation_preserves_prediction_count() -> None:
    """Batched evaluation must return exactly one prediction per sample."""
    dataset = generate_grounding_dataset(7, 64, 3, seed=30)
    vocabulary, token_ids = _encode_instructions(dataset.instructions)
    model = MultimodalGrounder(
        vocabulary_size=len(vocabulary),
        maximum_token_count=4,
        embedding_dimension=16,
        text_head_count=4,
        padding_id=vocabulary.padding_id,
    )

    evaluation = evaluate_grounding_model(
        model=model,
        images=dataset.images,
        token_ids=token_ids,
        target_centers=dataset.target_centers,
        scene_centers=dataset.scene_centers,
        target_indices=dataset.target_indices,
        batch_size=3,
        device=torch.device("cpu"),
    )

    assert evaluation.predicted_centers.shape == (7, 2)
    assert 0.0 <= evaluation.selection_accuracy <= 1.0
    assert evaluation.mean_center_error >= 0.0


def test_short_training_run_records_and_restores_checkpoint() -> None:
    """Two epochs should produce histories and leave a usable best model."""
    torch.manual_seed(31)
    training_data = generate_grounding_dataset(12, 32, 2, seed=31)
    validation_data = generate_grounding_dataset(8, 32, 2, seed=32)
    all_instructions = (
        training_data.instructions + validation_data.instructions
    )
    vocabulary = SimpleVocabulary.from_texts(all_instructions)
    training_tokens = vocabulary.encode_batch(training_data.instructions, 4)
    validation_tokens = vocabulary.encode_batch(
        validation_data.instructions,
        4,
    )
    model = VisionOnlyGrounder(embedding_dimension=8)

    result = train_grounding_model(
        model=model,
        train_images=training_data.images,
        train_token_ids=training_tokens,
        train_target_centers=training_data.target_centers,
        validation_images=validation_data.images,
        validation_token_ids=validation_tokens,
        validation_target_centers=validation_data.target_centers,
        validation_scene_centers=validation_data.scene_centers,
        validation_target_indices=validation_data.target_indices,
        epochs=2,
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.0,
        device=torch.device("cpu"),
        seed=31,
    )

    assert len(result.training_losses) == 2
    assert len(result.validation_losses) == 2
    assert len(result.validation_center_errors) == 2
    assert result.best_epoch in (1, 2)
    assert result.best_validation_loss == pytest.approx(
        min(result.validation_losses)
    )


def test_training_rejects_misaligned_sample_counts() -> None:
    """Training must not silently pair five images with four instructions."""
    model = VisionOnlyGrounder(embedding_dimension=8)

    with pytest.raises(ValueError, match="sample count"):
        train_grounding_model(
            model=model,
            train_images=torch.rand(5, 3, 32, 32),
            train_token_ids=torch.ones((4, 4), dtype=torch.int64),
            train_target_centers=torch.rand(5, 2),
            validation_images=torch.rand(2, 3, 32, 32),
            validation_token_ids=torch.ones((2, 4), dtype=torch.int64),
            validation_target_centers=torch.rand(2, 2),
            validation_scene_centers=torch.rand(2, 2, 2),
            validation_target_indices=torch.zeros(2, dtype=torch.int64),
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            weight_decay=0.0,
            device=torch.device("cpu"),
            seed=33,
        )
