"""Unit tests for discrete, patch, and learned-position embeddings."""

import pytest
import torch

from robot_learning.transformer_tokens import (
    DiscreteTokenEmbedding,
    ImagePatchEmbedding,
    LearnedPositionEmbedding,
)


def test_same_token_id_has_same_content_embedding() -> None:
    embedding = DiscreteTokenEmbedding(
        vocabulary_size=10,
        embedding_dimension=6,
    )
    token_ids = torch.tensor([[3, 7, 3]])

    tokens = embedding(token_ids)

    assert tokens.shape == (1, 3, 6)
    torch.testing.assert_close(tokens[:, 0], tokens[:, 2])


def test_learned_positions_distinguish_equal_content_tokens() -> None:
    position_embedding = LearnedPositionEmbedding(
        maximum_token_count=3,
        embedding_dimension=2,
    )
    with torch.no_grad():
        position_embedding.position_vectors.copy_(
            torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]])
        )
    identical_content = torch.zeros(1, 3, 2)

    positioned = position_embedding(identical_content)

    torch.testing.assert_close(
        positioned,
        torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]]),
    )
    assert not torch.equal(positioned[:, 0], positioned[:, 1])


def test_position_embedding_rejects_sequences_beyond_maximum() -> None:
    position_embedding = LearnedPositionEmbedding(4, 8)

    with pytest.raises(ValueError, match="exceeds"):
        position_embedding(torch.rand(2, 5, 8))


def test_discrete_embedding_rejects_non_integer_token_ids() -> None:
    embedding = DiscreteTokenEmbedding(10, 4)

    with pytest.raises(ValueError, match="int64"):
        embedding(torch.tensor([[1.0, 2.0]]))


def test_patch_embedding_returns_expected_token_count() -> None:
    embedding = ImagePatchEmbedding(
        input_channels=3,
        patch_size=8,
        embedding_dimension=16,
    )

    tokens = embedding(torch.rand(2, 3, 32, 48))

    assert tokens.shape == (2, 24, 16)


def test_patch_tokens_follow_row_major_image_regions() -> None:
    embedding = ImagePatchEmbedding(
        input_channels=1,
        patch_size=2,
        embedding_dimension=1,
    )
    with torch.no_grad():
        embedding.projection.weight.fill_(1.0)
        embedding.projection.bias.zero_()
    image = torch.tensor(
        [[[[1.0, 1.0, 2.0, 2.0],
           [1.0, 1.0, 2.0, 2.0],
           [3.0, 3.0, 4.0, 4.0],
           [3.0, 3.0, 4.0, 4.0]]]]
    )

    tokens = embedding(image)

    torch.testing.assert_close(
        tokens.flatten(),
        torch.tensor([4.0, 8.0, 12.0, 16.0]),
    )


def test_patch_embedding_rejects_partial_patches() -> None:
    embedding = ImagePatchEmbedding(3, patch_size=8, embedding_dimension=16)

    with pytest.raises(ValueError, match="divisible"):
        embedding(torch.rand(2, 3, 30, 32))


def test_content_patch_and_position_parameters_receive_gradients() -> None:
    token_embedding = DiscreteTokenEmbedding(10, 8)
    patch_embedding = ImagePatchEmbedding(3, patch_size=8, embedding_dimension=8)
    position_embedding = LearnedPositionEmbedding(16, 8)

    discrete_tokens = position_embedding(
        token_embedding(torch.tensor([[1, 2, 3, 4]]))
    )
    patch_tokens = position_embedding(
        patch_embedding(torch.rand(1, 3, 32, 32))
    )
    (discrete_tokens.square().mean() + patch_tokens.square().mean()).backward()

    assert token_embedding.embedding.weight.grad is not None
    assert patch_embedding.projection.weight.grad is not None
    assert position_embedding.position_vectors.grad is not None
