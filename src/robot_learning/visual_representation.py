"""Small visual embeddings, contrastive views, and retrieval metrics."""

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from torch.optim import Optimizer


@dataclass(frozen=True)
class EmbeddingStatistics:
    """Simple diagnostics for representation collapse."""

    mean_dimension_standard_deviation: float
    mean_pairwise_cosine_similarity: float


@dataclass(frozen=True)
class NearestNeighborEvaluation:
    """One-nearest-neighbor predictions and classification accuracy."""

    predictions: Tensor
    accuracy: float


class EncoderClassifier(nn.Module):
    """Attach a trainable linear classification head to an image encoder."""

    def __init__(
        self,
        encoder: "SmallVisualEncoder",
        num_classes: int,
    ) -> None:
        super().__init__()
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("num_classes must be an integer of at least two")
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.embedding_dimension, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        """Return class logits while keeping the encoder accessible."""
        return self.classifier(self.encoder(images))


class SmallVisualEncoder(nn.Module):
    """A compact CNN that returns L2-normalized global embeddings."""

    def __init__(self, embedding_dimension: int = 32) -> None:
        super().__init__()
        if type(embedding_dimension) is not int or embedding_dimension < 2:
            raise ValueError("embedding_dimension must be an integer of at least two")
        self.embedding_dimension = embedding_dimension
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(64, embedding_dimension)

    def forward(self, images: Tensor) -> Tensor:
        """Encode NCHW RGB images as unit-length vectors."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (N, 3, H, W)")
        features = self.features(images).flatten(start_dim=1)
        return functional.normalize(self.projection(features), dim=1)


def train_contrastive_epoch(
    encoder: SmallVisualEncoder,
    images: Tensor,
    optimizer: Optimizer,
    device: torch.device,
    batch_size: int,
    temperature: float,
    seed: int,
    *,
    minimum_color_gain: float = 0.7,
    maximum_color_gain: float = 1.3,
    noise_standard_deviation: float = 0.03,
) -> float:
    """Train one contrastive epoch and return sample-weighted NT-Xent loss."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape (N, 3, H, W)")
    if not images.is_floating_point() or images.shape[0] < 2:
        raise ValueError("images must be floating point and contain two samples")
    if type(batch_size) is not int or batch_size < 2:
        raise ValueError("batch_size must be an integer of at least two")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(temperature, Real) or isinstance(temperature, bool):
        raise ValueError("temperature must be a positive finite number")
    if not math.isfinite(float(temperature)) or temperature <= 0.0:
        raise ValueError("temperature must be a positive finite number")

    encoder.train()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(images.shape[0], generator=generator)
    total_loss = 0.0
    trained_sample_count = 0

    for batch_index, start in enumerate(range(0, images.shape[0], batch_size)):
        batch_indices = order[start:start + batch_size]
        if batch_indices.numel() < 2:
            continue
        batch = images[batch_indices].to(device)
        first_view, second_view = create_contrastive_views(
            batch,
            seed=seed + batch_index * 2,
            minimum_color_gain=minimum_color_gain,
            maximum_color_gain=maximum_color_gain,
            noise_standard_deviation=noise_standard_deviation,
        )

        optimizer.zero_grad(set_to_none=True)
        loss = nt_xent_loss(
            encoder(first_view),
            encoder(second_view),
            temperature=float(temperature),
        )
        loss.backward()
        optimizer.step()

        current_batch_size = int(batch_indices.numel())
        total_loss += float(loss.item()) * current_batch_size
        trained_sample_count += current_batch_size

    if trained_sample_count == 0:
        raise ValueError("no complete contrastive batch was available")
    return total_loss / trained_sample_count


def encode_images(
    encoder: SmallVisualEncoder,
    images: Tensor,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    """Encode images in deterministic batches and return CPU embeddings."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape (N, 3, H, W)")
    if not images.is_floating_point() or images.shape[0] == 0:
        raise ValueError("images must be floating point and non-empty")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    encoder.eval()
    batches: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            batch = images[start:start + batch_size].to(device)
            batches.append(encoder(batch).cpu())
    return torch.cat(batches, dim=0)


def _augment_for_contrastive_learning(
    images: Tensor,
    seed: int,
    minimum_color_gain: float,
    maximum_color_gain: float,
    noise_standard_deviation: float,
) -> Tensor:
    """Apply per-image color gains and additive noise without geometry changes."""
    generator = torch.Generator(device=images.device).manual_seed(seed)
    gain_shape = (images.shape[0], images.shape[1], 1, 1)
    gains = minimum_color_gain + (
        maximum_color_gain - minimum_color_gain
    ) * torch.rand(
        gain_shape,
        generator=generator,
        device=images.device,
        dtype=images.dtype,
    )
    noise = torch.randn(
        images.shape,
        generator=generator,
        device=images.device,
        dtype=images.dtype,
    ) * noise_standard_deviation
    return (images * gains + noise).clamp(0.0, 1.0)


def create_contrastive_views(
    images: Tensor,
    seed: int,
    *,
    minimum_color_gain: float = 0.7,
    maximum_color_gain: float = 1.3,
    noise_standard_deviation: float = 0.03,
) -> tuple[Tensor, Tensor]:
    """Create two reproducible appearance-only views of each input image."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape (N, 3, H, W)")
    if not torch.is_floating_point(images):
        raise ValueError("images must use a floating-point dtype")
    if images.shape[0] < 2:
        raise ValueError("contrastive views require at least two images")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    values = (
        minimum_color_gain,
        maximum_color_gain,
        noise_standard_deviation,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("augmentation values must be finite")
    if minimum_color_gain <= 0.0 or maximum_color_gain < minimum_color_gain:
        raise ValueError("color gain range must be positive and ordered")
    if noise_standard_deviation < 0.0:
        raise ValueError("noise_standard_deviation must be non-negative")
    if (
        minimum_color_gain == maximum_color_gain == 1.0
        and noise_standard_deviation == 0.0
    ):
        raise ValueError("at least one contrastive augmentation must be active")

    first_view = _augment_for_contrastive_learning(
        images,
        seed,
        minimum_color_gain,
        maximum_color_gain,
        noise_standard_deviation,
    )
    second_view = _augment_for_contrastive_learning(
        images,
        seed + 1,
        minimum_color_gain,
        maximum_color_gain,
        noise_standard_deviation,
    )
    return first_view, second_view


def nt_xent_loss(
    first_embeddings: Tensor,
    second_embeddings: Tensor,
    temperature: float,
) -> Tensor:
    """Calculate symmetric NT-Xent loss for aligned positive pairs."""
    if (
        first_embeddings.ndim != 2
        or first_embeddings.shape != second_embeddings.shape
    ):
        raise ValueError("embedding views must share shape (N, D)")
    if first_embeddings.shape[0] < 2 or first_embeddings.shape[1] < 2:
        raise ValueError("NT-Xent requires at least two samples and dimensions")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")

    first = functional.normalize(first_embeddings, dim=1)
    second = functional.normalize(second_embeddings, dim=1)
    embeddings = torch.cat((first, second), dim=0)
    similarities = embeddings @ embeddings.T / temperature
    sample_count = first.shape[0]
    diagonal_mask = torch.eye(
        sample_count * 2,
        dtype=torch.bool,
        device=embeddings.device,
    )
    similarities = similarities.masked_fill(diagonal_mask, float("-inf"))
    positive_indices = (
        torch.arange(sample_count * 2, device=embeddings.device)
        + sample_count
    ) % (sample_count * 2)
    return functional.cross_entropy(similarities, positive_indices)


def calculate_embedding_statistics(embeddings: Tensor) -> EmbeddingStatistics:
    """Measure feature variation and average off-diagonal cosine similarity."""
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("embeddings must have shape (N, D), with N >= 2")
    if not torch.isfinite(embeddings).all():
        raise ValueError("embeddings must be finite")
    normalized = functional.normalize(embeddings, dim=1)
    similarities = normalized @ normalized.T
    off_diagonal = ~torch.eye(
        embeddings.shape[0],
        dtype=torch.bool,
        device=embeddings.device,
    )
    return EmbeddingStatistics(
        mean_dimension_standard_deviation=float(
            embeddings.std(dim=0, unbiased=False).mean().item()
        ),
        mean_pairwise_cosine_similarity=float(
            similarities[off_diagonal].mean().item()
        ),
    )


def evaluate_one_nearest_neighbor(
    reference_embeddings: Tensor,
    reference_labels: Tensor,
    query_embeddings: Tensor,
    query_labels: Tensor,
) -> NearestNeighborEvaluation:
    """Classify each query using its most cosine-similar reference sample."""
    if reference_embeddings.ndim != 2 or query_embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if reference_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError("reference and query dimensions must match")
    if reference_labels.ndim != 1 or query_labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if reference_embeddings.shape[0] != reference_labels.shape[0]:
        raise ValueError("reference embeddings and labels must align")
    if query_embeddings.shape[0] != query_labels.shape[0]:
        raise ValueError("query embeddings and labels must align")
    if reference_embeddings.shape[0] == 0 or query_embeddings.shape[0] == 0:
        raise ValueError("reference and query sets must not be empty")

    references = functional.normalize(reference_embeddings, dim=1)
    queries = functional.normalize(query_embeddings, dim=1)
    similarities = queries @ references.T
    nearest_indices = similarities.argmax(dim=1)
    predictions = reference_labels[nearest_indices]
    return NearestNeighborEvaluation(
        predictions=predictions,
        accuracy=float((predictions == query_labels).float().mean().item()),
    )
