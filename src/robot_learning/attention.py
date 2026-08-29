"""Readable scaled dot-product attention for learning and testing."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AttentionResult:
    """Attention output values and normalized query-to-key weights."""

    output: Tensor
    weights: Tensor


@dataclass(frozen=True)
class TransformerBlockResult:
    """Transformer output tokens and per-head attention weights."""

    output: Tensor
    attention_weights: Tensor


class MultiHeadSelfAttention(nn.Module):
    """Project one token sequence into multiple self-attention heads."""

    def __init__(
        self,
        embedding_dimension: int,
        head_count: int,
    ) -> None:
        super().__init__()
        if type(embedding_dimension) is not int or embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be a positive integer")
        if type(head_count) is not int or head_count <= 0:
            raise ValueError("head_count must be a positive integer")
        if embedding_dimension % head_count != 0:
            raise ValueError("embedding_dimension must be divisible by head_count")

        self.embedding_dimension = embedding_dimension
        self.head_count = head_count
        self.head_dimension = embedding_dimension // head_count
        self.query_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.key_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.value_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.output_projection = nn.Linear(embedding_dimension, embedding_dimension)

    def forward(
        self,
        tokens: Tensor,
        attention_mask: Tensor | None = None,
    ) -> AttentionResult:
        """Return mixed tokens and weights with shape (N, H, L, L)."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.embedding_dimension:
            raise ValueError("tokens must have shape (N, L, embedding_dimension)")
        if not tokens.is_floating_point():
            raise ValueError("tokens must use a floating-point dtype")

        query = self._split_heads(self.query_projection(tokens))
        key = self._split_heads(self.key_projection(tokens))
        value = self._split_heads(self.value_projection(tokens))
        batch_size, _, token_count, _ = query.shape
        flat_query = query.reshape(
            batch_size * self.head_count,
            token_count,
            self.head_dimension,
        )
        flat_key = key.reshape(
            batch_size * self.head_count,
            token_count,
            self.head_dimension,
        )
        flat_value = value.reshape(
            batch_size * self.head_count,
            token_count,
            self.head_dimension,
        )
        attention = scaled_dot_product_attention(
            flat_query,
            flat_key,
            flat_value,
            attention_mask,
        )
        head_outputs = attention.output.reshape(
            batch_size,
            self.head_count,
            token_count,
            self.head_dimension,
        )
        combined = head_outputs.transpose(1, 2).contiguous().reshape(
            batch_size,
            token_count,
            self.embedding_dimension,
        )
        weights = attention.weights.reshape(
            batch_size,
            self.head_count,
            token_count,
            token_count,
        )
        return AttentionResult(
            output=self.output_projection(combined),
            weights=weights,
        )

    def _split_heads(self, projected_tokens: Tensor) -> Tensor:
        """Change (N, L, D) into (N, H, L, D/H)."""
        batch_size, token_count, _ = projected_tokens.shape
        return projected_tokens.reshape(
            batch_size,
            token_count,
            self.head_count,
            self.head_dimension,
        ).transpose(1, 2)


class TransformerEncoderBlock(nn.Module):
    """A pre-norm self-attention and MLP block with residual connections."""

    def __init__(
        self,
        embedding_dimension: int,
        head_count: int,
        mlp_hidden_dimension: int,
        dropout_probability: float = 0.0,
    ) -> None:
        super().__init__()
        if type(mlp_hidden_dimension) is not int or mlp_hidden_dimension <= 0:
            raise ValueError("mlp_hidden_dimension must be a positive integer")
        if (
            not isinstance(dropout_probability, (int, float))
            or isinstance(dropout_probability, bool)
            or not 0.0 <= dropout_probability < 1.0
        ):
            raise ValueError("dropout_probability must be in [0, 1)")

        self.embedding_dimension = embedding_dimension
        self.attention_normalization = nn.LayerNorm(embedding_dimension)
        self.attention = MultiHeadSelfAttention(
            embedding_dimension,
            head_count,
        )
        self.mlp_normalization = nn.LayerNorm(embedding_dimension)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dimension, mlp_hidden_dimension),
            nn.GELU(),
            nn.Linear(mlp_hidden_dimension, embedding_dimension),
        )
        self.dropout = nn.Dropout(float(dropout_probability))

    def forward(
        self,
        tokens: Tensor,
        attention_mask: Tensor | None = None,
    ) -> TransformerBlockResult:
        """Apply pre-norm attention and MLP residual updates."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.embedding_dimension:
            raise ValueError("tokens must have shape (N, L, embedding_dimension)")
        attention = self.attention(
            self.attention_normalization(tokens),
            attention_mask,
        )
        tokens = tokens + self.dropout(attention.output)
        tokens = tokens + self.dropout(self.mlp(self.mlp_normalization(tokens)))
        return TransformerBlockResult(
            output=tokens,
            attention_weights=attention.weights,
        )


def create_causal_attention_mask(token_count: int) -> Tensor:
    """Return a mask where each token can read itself and earlier tokens."""
    if type(token_count) is not int or token_count <= 0:
        raise ValueError("token_count must be a positive integer")
    return torch.ones(
        (token_count, token_count),
        dtype=torch.bool,
    ).tril()


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None = None,
) -> AttentionResult:
    """Calculate attention for batched query, key, and value sequences.

    A boolean mask uses True for allowed query-key pairs and False for blocked
    pairs. Its shape is (query_token_count, key_token_count).
    """
    _validate_attention_inputs(query, key, value)
    query_token_count = query.shape[1]
    key_token_count = key.shape[1]
    if attention_mask is not None:
        if attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must use boolean dtype")
        if attention_mask.shape != (query_token_count, key_token_count):
            raise ValueError(
                "attention_mask must have shape "
                "(query_token_count, key_token_count)"
            )
        if not attention_mask.any(dim=-1).all():
            raise ValueError("every query token must have an allowed key")

    scores = query @ key.transpose(-2, -1)
    scores = scores / math.sqrt(query.shape[-1])
    if attention_mask is not None:
        mask = attention_mask.to(device=scores.device)
        scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output = weights @ value
    return AttentionResult(output=output, weights=weights)


def _validate_attention_inputs(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> None:
    tensors = (query, key, value)
    if any(tensor.ndim != 3 for tensor in tensors):
        raise ValueError("query, key, and value must have shape (N, L, D)")
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise ValueError("query, key, and value must be floating point")
    if any(tensor.shape[0] == 0 or tensor.shape[1] == 0 for tensor in tensors):
        raise ValueError("batch and token dimensions must be non-empty")
    if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
        raise ValueError("query, key, and value batch sizes must match")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key feature dimensions must match")
    if key.shape[1] != value.shape[1]:
        raise ValueError("key and value token counts must match")
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("query, key, and value must contain finite values")
