"""Unit tests for the paired token-order Transformer task."""

import pytest
import torch

from robot_learning.sequence_transformer import (
    TokenOrderTransformer,
    create_sequence_data_loader,
    evaluate_sequence_model,
    generate_token_order_dataset,
)


def test_order_dataset_contains_paired_permutations() -> None:
    dataset = generate_token_order_dataset(6, 7, 20, seed=40)

    assert dataset.token_ids.shape == (12, 7)
    assert dataset.labels.tolist() == [1, 0] * 6
    for pair_index in range(6):
        positive = dataset.token_ids[pair_index * 2]
        negative = dataset.token_ids[pair_index * 2 + 1]
        torch.testing.assert_close(positive.sort().values, negative.sort().values)
        assert torch.nonzero(positive == 1).item() < torch.nonzero(positive == 2).item()
        assert torch.nonzero(negative == 1).item() > torch.nonzero(negative == 2).item()


def test_model_without_positions_is_permutation_invariant_after_pooling() -> None:
    torch.manual_seed(41)
    model = TokenOrderTransformer(
        vocabulary_size=20,
        maximum_token_count=6,
        embedding_dimension=12,
        head_count=3,
        mlp_hidden_dimension=24,
        layer_count=2,
        use_position_embedding=False,
    )
    model.eval()
    sequences = torch.tensor([[1, 5, 7, 2, 9, 4], [2, 5, 7, 1, 9, 4]])

    logits = model(sequences)

    torch.testing.assert_close(logits[0], logits[1])


def test_position_model_can_distinguish_swapped_special_tokens() -> None:
    torch.manual_seed(42)
    model = TokenOrderTransformer(
        vocabulary_size=20,
        maximum_token_count=6,
        embedding_dimension=12,
        head_count=3,
        mlp_hidden_dimension=24,
        layer_count=2,
        use_position_embedding=True,
    )
    model.eval()
    sequences = torch.tensor([[1, 5, 7, 2, 9, 4], [2, 5, 7, 1, 9, 4]])

    logits = model(sequences)

    assert not torch.equal(logits[0], logits[1])


def test_sequence_model_returns_logits_and_receives_gradients() -> None:
    model = TokenOrderTransformer(
        vocabulary_size=20,
        maximum_token_count=6,
        embedding_dimension=12,
        head_count=3,
        mlp_hidden_dimension=24,
        layer_count=1,
        use_position_embedding=True,
    )

    logits = model(torch.randint(0, 20, (4, 6)))
    logits.square().mean().backward()

    assert logits.shape == (4, 2)
    assert model.token_embedding.embedding.weight.grad is not None
    assert model.position_embedding is not None
    assert model.position_embedding.position_vectors.grad is not None
    assert model.blocks[0].attention.query_projection.weight.grad is not None
    assert model.classifier.weight.grad is not None


def test_sequence_evaluation_reports_predictions_and_accuracy() -> None:
    class FixedModel(torch.nn.Module):
        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            return torch.stack(
                ((token_ids[:, 0] == 1).float(), (token_ids[:, 0] == 2).float()),
                dim=1,
            )

    dataset = generate_token_order_dataset(4, 6, 20, seed=43)
    loader = create_sequence_data_loader(dataset, 8, shuffle=False, seed=44)

    evaluation = evaluate_sequence_model(
        FixedModel(),
        loader,
        device=torch.device("cpu"),
    )

    assert evaluation.predictions.shape == (8,)
    assert 0.0 <= evaluation.accuracy <= 1.0
    assert evaluation.loss > 0.0


def test_order_dataset_rejects_vocabulary_without_filler_tokens() -> None:
    with pytest.raises(ValueError, match="at least four"):
        generate_token_order_dataset(4, 6, 3, seed=45)
