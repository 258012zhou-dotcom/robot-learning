"""Unit tests for readable scaled dot-product attention."""

import pytest
import torch

from robot_learning.attention import (
    MultiHeadSelfAttention,
    TransformerEncoderBlock,
    create_causal_attention_mask,
    scaled_dot_product_attention,
)


def test_attention_weights_sum_to_one_over_keys() -> None:
    query = torch.rand(2, 3, 4)
    key = torch.rand(2, 5, 4)
    value = torch.rand(2, 5, 6)

    result = scaled_dot_product_attention(query, key, value)

    assert result.weights.shape == (2, 3, 5)
    torch.testing.assert_close(
        result.weights.sum(dim=-1),
        torch.ones(2, 3),
    )


def test_query_assigns_more_weight_to_matching_key() -> None:
    query = torch.tensor([[[2.0, 0.0]]])
    key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    value = torch.tensor([[[10.0], [20.0]]])

    result = scaled_dot_product_attention(query, key, value)

    assert result.weights[0, 0, 0] > result.weights[0, 0, 1]
    assert 10.0 < result.output[0, 0, 0] < 15.0


def test_attention_output_uses_value_feature_dimension() -> None:
    result = scaled_dot_product_attention(
        torch.rand(4, 3, 8),
        torch.rand(4, 5, 8),
        torch.rand(4, 5, 6),
    )

    assert result.output.shape == (4, 3, 6)


def test_causal_mask_blocks_all_future_tokens() -> None:
    token_count = 4
    query = torch.rand(1, token_count, 3)
    key = torch.rand(1, token_count, 3)
    value = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
    mask = create_causal_attention_mask(token_count)

    result = scaled_dot_product_attention(query, key, value, mask)

    assert torch.count_nonzero(result.weights[0].triu(diagonal=1)).item() == 0
    torch.testing.assert_close(result.weights[0, 0], torch.tensor([1.0, 0, 0, 0]))
    torch.testing.assert_close(result.output[0, 0], value[0, 0])


def test_attention_backpropagates_to_query_key_and_value() -> None:
    query = torch.rand(2, 3, 4, requires_grad=True)
    key = torch.rand(2, 5, 4, requires_grad=True)
    value = torch.rand(2, 5, 6, requires_grad=True)

    result = scaled_dot_product_attention(query, key, value)
    result.output.square().mean().backward()

    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(key.grad).all()
    assert torch.isfinite(value.grad).all()


def test_attention_rejects_mismatched_query_and_key_dimensions() -> None:
    with pytest.raises(ValueError, match="feature dimensions"):
        scaled_dot_product_attention(
            torch.rand(2, 3, 4),
            torch.rand(2, 5, 6),
            torch.rand(2, 5, 7),
        )


def test_attention_rejects_mask_with_fully_blocked_query() -> None:
    mask = torch.tensor([[True, False], [False, False]])

    with pytest.raises(ValueError, match="allowed key"):
        scaled_dot_product_attention(
            torch.rand(1, 2, 3),
            torch.rand(1, 2, 3),
            torch.rand(1, 2, 4),
            mask,
        )


def test_multi_head_attention_returns_per_head_weights() -> None:
    model = MultiHeadSelfAttention(embedding_dimension=12, head_count=3)

    result = model(torch.rand(2, 5, 12))

    assert result.output.shape == (2, 5, 12)
    assert result.weights.shape == (2, 3, 5, 5)
    torch.testing.assert_close(
        result.weights.sum(dim=-1),
        torch.ones(2, 3, 5),
    )


def test_multi_head_attention_requires_divisible_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadSelfAttention(embedding_dimension=10, head_count=3)


def test_causal_mask_blocks_future_tokens_in_every_head() -> None:
    model = MultiHeadSelfAttention(embedding_dimension=8, head_count=2)
    mask = create_causal_attention_mask(4)

    result = model(torch.rand(3, 4, 8), mask)

    future_weights = result.weights.triu(diagonal=1)
    assert torch.count_nonzero(future_weights).item() == 0


def test_zero_residual_branches_preserve_original_tokens() -> None:
    block = TransformerEncoderBlock(
        embedding_dimension=8,
        head_count=2,
        mlp_hidden_dimension=16,
    )
    for parameter in block.attention.parameters():
        torch.nn.init.zeros_(parameter)
    for parameter in block.mlp.parameters():
        torch.nn.init.zeros_(parameter)
    tokens = torch.rand(2, 4, 8)

    result = block(tokens)

    torch.testing.assert_close(result.output, tokens)


def test_transformer_without_positions_is_permutation_equivariant() -> None:
    torch.manual_seed(30)
    block = TransformerEncoderBlock(
        embedding_dimension=8,
        head_count=2,
        mlp_hidden_dimension=16,
    )
    block.eval()
    tokens = torch.rand(2, 5, 8)
    permutation = torch.tensor([2, 0, 4, 1, 3])

    original = block(tokens).output
    permuted = block(tokens[:, permutation]).output

    torch.testing.assert_close(permuted, original[:, permutation])


def test_causal_block_output_does_not_use_changed_future_tokens() -> None:
    torch.manual_seed(31)
    block = TransformerEncoderBlock(
        embedding_dimension=8,
        head_count=2,
        mlp_hidden_dimension=16,
    )
    block.eval()
    original_tokens = torch.rand(1, 5, 8)
    changed_tokens = original_tokens.clone()
    changed_tokens[:, 3:] = torch.rand(1, 2, 8) * 100.0
    mask = create_causal_attention_mask(5)

    original_output = block(original_tokens, mask).output
    changed_output = block(changed_tokens, mask).output

    torch.testing.assert_close(original_output[:, :3], changed_output[:, :3])


def test_transformer_block_backpropagates_through_all_sublayers() -> None:
    block = TransformerEncoderBlock(
        embedding_dimension=8,
        head_count=2,
        mlp_hidden_dimension=16,
    )
    tokens = torch.rand(2, 4, 8, requires_grad=True)

    block(tokens).output.square().mean().backward()

    assert tokens.grad is not None
    assert block.attention.query_projection.weight.grad is not None
    assert block.mlp[0].weight.grad is not None
