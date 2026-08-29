"""Token, patch, and learned-position embeddings for Transformer inputs."""

import torch
from torch import Tensor, nn


class DiscreteTokenEmbedding(nn.Module):
    """Map integer token IDs to trainable content vectors."""

    def __init__(self, vocabulary_size: int, embedding_dimension: int) -> None:
        super().__init__()
        if type(vocabulary_size) is not int or vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be a positive integer")
        if type(embedding_dimension) is not int or embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be a positive integer")
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return embeddings with shape (N, L, D) for int64 token IDs."""
        if token_ids.ndim != 2 or token_ids.dtype != torch.int64:
            raise ValueError("token_ids must have shape (N, L) and int64 dtype")
        if token_ids.shape[0] == 0 or token_ids.shape[1] == 0:
            raise ValueError("token_ids must contain a non-empty batch and sequence")
        if token_ids.min().item() < 0 or token_ids.max().item() >= self.vocabulary_size:
            raise ValueError("token IDs must be inside the configured vocabulary")
        return self.embedding(token_ids)


class LearnedPositionEmbedding(nn.Module):
    """Add one trainable vector for each absolute sequence position."""

    def __init__(
        self,
        maximum_token_count: int,
        embedding_dimension: int,
    ) -> None:
        super().__init__()
        if type(maximum_token_count) is not int or maximum_token_count <= 0:
            raise ValueError("maximum_token_count must be a positive integer")
        if type(embedding_dimension) is not int or embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be a positive integer")
        self.maximum_token_count = maximum_token_count
        self.embedding_dimension = embedding_dimension
        self.position_vectors = nn.Parameter(
            torch.zeros(1, maximum_token_count, embedding_dimension)
        )
        nn.init.normal_(self.position_vectors, mean=0.0, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        """Add learned absolute positions to tokens with shape (N, L, D)."""
        if tokens.ndim != 3 or tokens.shape[-1] != self.embedding_dimension:
            raise ValueError("tokens must have shape (N, L, embedding_dimension)")
        if not tokens.is_floating_point():
            raise ValueError("tokens must use a floating-point dtype")
        if tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError("tokens must contain a non-empty batch and sequence")
        if tokens.shape[1] > self.maximum_token_count:
            raise ValueError("token count exceeds the configured maximum")
        positions = self.position_vectors[:, :tokens.shape[1]].to(
            dtype=tokens.dtype
        )
        return tokens + positions


class ImagePatchEmbedding(nn.Module):
    """Convert non-overlapping square RGB patches into a token sequence."""

    def __init__(
        self,
        input_channels: int,
        patch_size: int,
        embedding_dimension: int,
    ) -> None:
        super().__init__()
        for value, name in (
            (input_channels, "input_channels"),
            (patch_size, "patch_size"),
            (embedding_dimension, "embedding_dimension"),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.embedding_dimension = embedding_dimension
        self.projection = nn.Conv2d(
            input_channels,
            embedding_dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return row-major patch tokens with shape (N, patch_count, D)."""
        if images.ndim != 4 or images.shape[1] != self.input_channels:
            raise ValueError("images must have shape (N, C, H, W)")
        if not images.is_floating_point():
            raise ValueError("images must use a floating-point dtype")
        if images.shape[0] == 0:
            raise ValueError("images must contain a non-empty batch")
        height, width = images.shape[-2:]
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("image height and width must be divisible by patch_size")
        feature_grid = self.projection(images)
        return feature_grid.flatten(start_dim=2).transpose(1, 2)
