"""Small image-text encoders and similarity utilities for VLM learning."""

from dataclasses import dataclass
import math
import re
from typing import Iterable

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from torch.optim import Optimizer

from robot_learning.transformer_tokens import LearnedPositionEmbedding
from robot_learning.visual_representation import SmallVisualEncoder


PADDING_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
COLOR_NAMES = ("red", "green", "blue")
SHAPE_NAMES = ("square", "circle")


@dataclass(frozen=True)
class ImageTextDataset:
    """Synthetic RGB images, canonical descriptions, and concept labels."""

    images: Tensor
    descriptions: tuple[str, ...]
    concept_labels: Tensor
    concept_descriptions: tuple[str, ...]


def generate_image_text_dataset(
    samples_per_concept: int,
    image_size: int,
    seed: int,
) -> ImageTextDataset:
    """Generate balanced color-shape images paired with short descriptions."""
    if type(samples_per_concept) is not int or samples_per_concept <= 0:
        raise ValueError("samples_per_concept must be a positive integer")
    if type(image_size) is not int or image_size < 16:
        raise ValueError("image_size must be an integer of at least 16")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    concept_descriptions = tuple(
        f"a {color_name} {shape_name}"
        for color_name in COLOR_NAMES
        for shape_name in SHAPE_NAMES
    )
    sample_count = len(concept_descriptions) * samples_per_concept
    images_rgb = np.empty(
        (sample_count, image_size, image_size, 3),
        dtype=np.uint8,
    )
    descriptions: list[str] = []
    labels = np.empty(sample_count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    base_colors = {
        "red": np.array((220, 40, 40), dtype=np.int16),
        "green": np.array((40, 200, 40), dtype=np.int16),
        "blue": np.array((40, 70, 220), dtype=np.int16),
    }

    sample_index = 0
    for concept_label, description in enumerate(concept_descriptions):
        _, color_name, shape_name = description.split()
        for _ in range(samples_per_concept):
            image = rng.integers(
                0,
                26,
                size=(image_size, image_size, 3),
                dtype=np.uint8,
            )
            shape_size = int(
                rng.integers(image_size // 4, image_size // 2 + 1)
            )
            half_size = shape_size // 2
            center_x = int(
                rng.integers(half_size, image_size - half_size)
            )
            center_y = int(
                rng.integers(half_size, image_size - half_size)
            )
            color_jitter = rng.integers(-15, 16, size=3)
            color = tuple(
                int(value)
                for value in np.clip(
                    base_colors[color_name] + color_jitter,
                    0,
                    255,
                )
            )
            if shape_name == "square":
                cv2.rectangle(
                    image,
                    (center_x - half_size, center_y - half_size),
                    (center_x + half_size, center_y + half_size),
                    color,
                    thickness=-1,
                )
            else:
                cv2.circle(
                    image,
                    (center_x, center_y),
                    half_size,
                    color,
                    thickness=-1,
                )
            images_rgb[sample_index] = image
            descriptions.append(description)
            labels[sample_index] = concept_label
            sample_index += 1

    order = rng.permutation(sample_count)
    image_tensor = (
        torch.from_numpy(images_rgb[order].copy())
        .permute(0, 3, 1, 2)
        .to(dtype=torch.float32)
        .div(255.0)
    )
    ordered_descriptions = tuple(descriptions[index] for index in order)
    return ImageTextDataset(
        images=image_tensor,
        descriptions=ordered_descriptions,
        concept_labels=torch.from_numpy(labels[order].copy()),
        concept_descriptions=concept_descriptions,
    )


def _split_text(text: str) -> list[str]:
    """Return lowercase alphanumeric tokens for a short English description."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        raise ValueError("text must contain at least one alphanumeric token")
    return tokens


@dataclass(frozen=True)
class SimpleVocabulary:
    """A deterministic word-level vocabulary with padding and unknown tokens."""

    token_to_id: dict[str, int]

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "SimpleVocabulary":
        """Build a vocabulary from unique words in sorted order."""
        words: set[str] = set()
        for text in texts:
            words.update(_split_text(text))
        if not words:
            raise ValueError("texts must contain at least one description")
        token_to_id = {PADDING_TOKEN: 0, UNKNOWN_TOKEN: 1}
        token_to_id.update(
            {word: index + 2 for index, word in enumerate(sorted(words))}
        )
        return cls(token_to_id=token_to_id)

    @property
    def padding_id(self) -> int:
        """Return the ID used to pad shorter descriptions."""
        return self.token_to_id[PADDING_TOKEN]

    @property
    def unknown_id(self) -> int:
        """Return the ID used for words absent from the training vocabulary."""
        return self.token_to_id[UNKNOWN_TOKEN]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, maximum_token_count: int) -> list[int]:
        """Encode one description and right-pad it to a fixed token count."""
        if type(maximum_token_count) is not int or maximum_token_count <= 0:
            raise ValueError("maximum_token_count must be a positive integer")
        words = _split_text(text)
        if len(words) > maximum_token_count:
            raise ValueError("text exceeds maximum_token_count")
        token_ids = [
            self.token_to_id.get(word, self.unknown_id)
            for word in words
        ]
        return token_ids + [self.padding_id] * (
            maximum_token_count - len(token_ids)
        )

    def encode_batch(
        self,
        texts: Iterable[str],
        maximum_token_count: int,
    ) -> Tensor:
        """Encode descriptions as an int64 tensor with shape (N, L)."""
        encoded = [self.encode(text, maximum_token_count) for text in texts]
        if not encoded:
            raise ValueError("texts must contain at least one description")
        return torch.tensor(encoded, dtype=torch.int64)


class SmallTextEncoder(nn.Module):
    """Encode padded word-token sequences as unit-length text embeddings."""

    def __init__(
        self,
        vocabulary_size: int,
        maximum_token_count: int,
        embedding_dimension: int = 32,
        head_count: int = 4,
        mlp_hidden_dimension: int = 64,
        layer_count: int = 1,
        padding_id: int = 0,
    ) -> None:
        super().__init__()
        integer_values = (
            (vocabulary_size, "vocabulary_size"),
            (maximum_token_count, "maximum_token_count"),
            (embedding_dimension, "embedding_dimension"),
            (head_count, "head_count"),
            (mlp_hidden_dimension, "mlp_hidden_dimension"),
            (layer_count, "layer_count"),
        )
        for value, name in integer_values:
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if embedding_dimension % head_count != 0:
            raise ValueError("embedding_dimension must be divisible by head_count")
        if type(padding_id) is not int or not 0 <= padding_id < vocabulary_size:
            raise ValueError("padding_id must be inside the vocabulary")

        self.vocabulary_size = vocabulary_size
        self.maximum_token_count = maximum_token_count
        self.embedding_dimension = embedding_dimension
        self.padding_id = padding_id
        self.token_embedding = nn.Embedding(
            vocabulary_size,
            embedding_dimension,
            padding_idx=padding_id,
        )
        self.position_embedding = LearnedPositionEmbedding(
            maximum_token_count,
            embedding_dimension,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dimension,
            nhead=head_count,
            dim_feedforward=mlp_hidden_dimension,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=layer_count,
            enable_nested_tensor=False,
        )
        self.final_normalization = nn.LayerNorm(embedding_dimension)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return one normalized embedding for each padded token sequence."""
        if token_ids.ndim != 2 or token_ids.dtype != torch.int64:
            raise ValueError("token_ids must have shape (N, L) and int64 dtype")
        if token_ids.shape[0] == 0 or token_ids.shape[1] == 0:
            raise ValueError("token_ids must contain a non-empty batch and sequence")
        if token_ids.shape[1] > self.maximum_token_count:
            raise ValueError("token count exceeds the configured maximum")
        if token_ids.min().item() < 0 or token_ids.max().item() >= self.vocabulary_size:
            raise ValueError("token IDs must be inside the configured vocabulary")

        padding_mask = token_ids.eq(self.padding_id)
        if padding_mask.all(dim=1).any():
            raise ValueError("every text sequence must contain a non-padding token")
        tokens = self.position_embedding(self.token_embedding(token_ids))
        encoded = self.transformer(
            tokens,
            src_key_padding_mask=padding_mask,
        )
        encoded = self.final_normalization(encoded)
        valid_tokens = (~padding_mask).unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * valid_tokens).sum(dim=1)
        pooled = pooled / valid_tokens.sum(dim=1)
        return functional.normalize(pooled, dim=1)


@dataclass(frozen=True)
class VisionLanguageEmbeddings:
    """Normalized image and text vectors in one shared embedding space."""

    image_embeddings: Tensor
    text_embeddings: Tensor


@dataclass(frozen=True)
class PairRetrievalEvaluation:
    """Symmetric loss and top-one retrieval accuracy for aligned pairs."""

    loss: float
    image_to_text_accuracy: float
    text_to_image_accuracy: float
    similarities: Tensor


@dataclass(frozen=True)
class SemanticRetrievalEvaluation:
    """Concept-level classification, retrieval, margin, and confusion matrix."""

    loss: float
    image_to_text_accuracy: float
    text_to_image_accuracy: float
    mean_similarity_margin: float
    predictions: Tensor
    confusion_matrix: Tensor
    similarities: Tensor


class VisionLanguageDualEncoder(nn.Module):
    """Encode images and descriptions independently into a shared dimension."""

    def __init__(
        self,
        vocabulary_size: int,
        maximum_token_count: int,
        embedding_dimension: int = 32,
        text_head_count: int = 4,
        text_mlp_hidden_dimension: int = 64,
        text_layer_count: int = 1,
        padding_id: int = 0,
    ) -> None:
        super().__init__()
        self.image_encoder = SmallVisualEncoder(embedding_dimension)
        self.text_encoder = SmallTextEncoder(
            vocabulary_size=vocabulary_size,
            maximum_token_count=maximum_token_count,
            embedding_dimension=embedding_dimension,
            head_count=text_head_count,
            mlp_hidden_dimension=text_mlp_hidden_dimension,
            layer_count=text_layer_count,
            padding_id=padding_id,
        )

    def forward(
        self,
        images: Tensor,
        token_ids: Tensor,
    ) -> VisionLanguageEmbeddings:
        """Return normalized image and text embeddings."""
        return VisionLanguageEmbeddings(
            image_embeddings=self.image_encoder(images),
            text_embeddings=self.text_encoder(token_ids),
        )


def calculate_image_text_similarities(
    image_embeddings: Tensor,
    text_embeddings: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Return all pairwise cosine similarities with shape (N_image, N_text)."""
    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if image_embeddings.shape[0] == 0 or text_embeddings.shape[0] == 0:
        raise ValueError("embedding batches must be non-empty")
    if image_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError("image and text embedding dimensions must match")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    if not torch.isfinite(image_embeddings).all():
        raise ValueError("image embeddings must be finite")
    if not torch.isfinite(text_embeddings).all():
        raise ValueError("text embeddings must be finite")
    images = functional.normalize(image_embeddings, dim=1)
    texts = functional.normalize(text_embeddings, dim=1)
    return images @ texts.T / temperature


def symmetric_image_text_contrastive_loss(
    image_embeddings: Tensor,
    text_embeddings: Tensor,
    temperature: float,
) -> Tensor:
    """Apply symmetric cross-entropy to one-to-one image-text pairs."""
    if image_embeddings.shape[0] != text_embeddings.shape[0]:
        raise ValueError("image and text batches must contain equal pair counts")
    if image_embeddings.shape[0] < 2:
        raise ValueError("contrastive loss requires at least two pairs")
    similarities = calculate_image_text_similarities(
        image_embeddings,
        text_embeddings,
        temperature,
    )
    targets = torch.arange(
        similarities.shape[0],
        device=similarities.device,
    )
    image_to_text = functional.cross_entropy(similarities, targets)
    text_to_image = functional.cross_entropy(similarities.T, targets)
    return 0.5 * (image_to_text + text_to_image)


def create_unique_concept_batches(
    concept_labels: Tensor,
    seed: int,
) -> list[Tensor]:
    """Create batches containing exactly one randomly ordered sample per concept."""
    if concept_labels.ndim != 1 or concept_labels.dtype != torch.int64:
        raise ValueError("concept_labels must be a one-dimensional int64 tensor")
    if concept_labels.numel() == 0:
        raise ValueError("concept_labels must be non-empty")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    unique_labels = torch.unique(concept_labels, sorted=True)
    if unique_labels.numel() < 2:
        raise ValueError("at least two concepts are required")
    indices_by_concept = [
        torch.nonzero(concept_labels == label, as_tuple=False).flatten()
        for label in unique_labels
    ]
    sample_counts = {indices.numel() for indices in indices_by_concept}
    if len(sample_counts) != 1:
        raise ValueError("every concept must contain the same number of samples")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    shuffled_indices = [
        indices[torch.randperm(indices.numel(), generator=generator)]
        for indices in indices_by_concept
    ]
    samples_per_concept = shuffled_indices[0].numel()
    return [
        torch.stack(
            [indices[step] for indices in shuffled_indices],
        )
        for step in range(samples_per_concept)
    ]


def train_vision_language_epoch(
    model: VisionLanguageDualEncoder,
    images: Tensor,
    token_ids: Tensor,
    concept_labels: Tensor,
    optimizer: Optimizer,
    device: torch.device,
    temperature: float,
    seed: int,
) -> float:
    """Train one epoch using one sample from every concept in each batch."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape (N, 3, H, W)")
    if token_ids.ndim != 2 or token_ids.dtype != torch.int64:
        raise ValueError("token_ids must have shape (N, L) and int64 dtype")
    sample_count = images.shape[0]
    if token_ids.shape[0] != sample_count or concept_labels.shape[0] != sample_count:
        raise ValueError("images, token IDs, and labels must have equal counts")

    model.train()
    total_loss = 0.0
    trained_sample_count = 0
    for batch_indices in create_unique_concept_batches(concept_labels, seed):
        batch_images = images[batch_indices].to(device)
        batch_token_ids = token_ids[batch_indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        embeddings = model(batch_images, batch_token_ids)
        loss = symmetric_image_text_contrastive_loss(
            embeddings.image_embeddings,
            embeddings.text_embeddings,
            temperature,
        )
        loss.backward()
        optimizer.step()

        batch_size = batch_indices.numel()
        total_loss += float(loss.item()) * batch_size
        trained_sample_count += batch_size
    return total_loss / trained_sample_count


def evaluate_aligned_image_text_pairs(
    model: VisionLanguageDualEncoder,
    images: Tensor,
    token_ids: Tensor,
    device: torch.device,
    temperature: float,
) -> PairRetrievalEvaluation:
    """Evaluate one-to-one paired retrieval without updating model parameters."""
    if images.shape[0] != token_ids.shape[0]:
        raise ValueError("images and token IDs must contain equal pair counts")
    model.eval()
    with torch.no_grad():
        embeddings = model(images.to(device), token_ids.to(device))
        similarities = calculate_image_text_similarities(
            embeddings.image_embeddings,
            embeddings.text_embeddings,
            temperature,
        )
        loss = symmetric_image_text_contrastive_loss(
            embeddings.image_embeddings,
            embeddings.text_embeddings,
            temperature,
        )
    targets = torch.arange(similarities.shape[0], device=similarities.device)
    return PairRetrievalEvaluation(
        loss=float(loss.item()),
        image_to_text_accuracy=float(
            (similarities.argmax(dim=1) == targets).float().mean().item()
        ),
        text_to_image_accuracy=float(
            (similarities.argmax(dim=0) == targets).float().mean().item()
        ),
        similarities=similarities.cpu(),
    )


def evaluate_semantic_similarity_matrix(
    similarities: Tensor,
    concept_labels: Tensor,
) -> SemanticRetrievalEvaluation:
    """Evaluate many images against one canonical text per semantic concept."""
    if similarities.ndim != 2 or similarities.shape[0] == 0:
        raise ValueError("similarities must have non-empty shape (N, C)")
    if similarities.shape[1] < 2:
        raise ValueError("similarities must contain at least two concepts")
    if concept_labels.ndim != 1 or concept_labels.dtype != torch.int64:
        raise ValueError("concept_labels must be a one-dimensional int64 tensor")
    if concept_labels.shape[0] != similarities.shape[0]:
        raise ValueError("similarities and labels must contain equal samples")
    if concept_labels.min().item() < 0:
        raise ValueError("concept labels must be non-negative")
    if concept_labels.max().item() >= similarities.shape[1]:
        raise ValueError("concept labels exceed the text concept count")
    if not torch.isfinite(similarities).all():
        raise ValueError("similarities must be finite")

    predictions = similarities.argmax(dim=1)
    image_accuracy = (predictions == concept_labels).float().mean()
    best_image_indices = similarities.argmax(dim=0)
    expected_concepts = torch.arange(
        similarities.shape[1],
        device=similarities.device,
    )
    text_accuracy = (
        concept_labels[best_image_indices] == expected_concepts
    ).float().mean()

    row_indices = torch.arange(similarities.shape[0], device=similarities.device)
    correct_similarities = similarities[row_indices, concept_labels]
    incorrect_similarities = similarities.clone()
    incorrect_similarities[row_indices, concept_labels] = float("-inf")
    margins = correct_similarities - incorrect_similarities.max(dim=1).values
    concept_count = similarities.shape[1]
    flat_confusion_indices = concept_labels * concept_count + predictions
    confusion_matrix = torch.bincount(
        flat_confusion_indices,
        minlength=concept_count * concept_count,
    ).reshape(concept_count, concept_count)
    return SemanticRetrievalEvaluation(
        loss=float(functional.cross_entropy(similarities, concept_labels).item()),
        image_to_text_accuracy=float(image_accuracy.item()),
        text_to_image_accuracy=float(text_accuracy.item()),
        mean_similarity_margin=float(margins.mean().item()),
        predictions=predictions.cpu(),
        confusion_matrix=confusion_matrix.cpu(),
        similarities=similarities.cpu(),
    )


def evaluate_semantic_image_text_retrieval(
    model: VisionLanguageDualEncoder,
    images: Tensor,
    concept_labels: Tensor,
    concept_token_ids: Tensor,
    device: torch.device,
    temperature: float,
    image_batch_size: int,
) -> SemanticRetrievalEvaluation:
    """Encode a dataset and evaluate it against canonical concept texts."""
    if images.ndim != 4 or images.shape[1] != 3 or images.shape[0] == 0:
        raise ValueError("images must have non-empty shape (N, 3, H, W)")
    if concept_labels.shape[0] != images.shape[0]:
        raise ValueError("images and labels must contain equal samples")
    if type(image_batch_size) is not int or image_batch_size <= 0:
        raise ValueError("image_batch_size must be a positive integer")

    model.eval()
    image_embedding_batches: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, images.shape[0], image_batch_size):
            image_embedding_batches.append(
                model.image_encoder(
                    images[start:start + image_batch_size].to(device)
                ).cpu()
            )
        text_embeddings = model.text_encoder(
            concept_token_ids.to(device)
        ).cpu()
    image_embeddings = torch.cat(image_embedding_batches, dim=0)
    similarities = calculate_image_text_similarities(
        image_embeddings,
        text_embeddings,
        temperature,
    )
    return evaluate_semantic_similarity_matrix(
        similarities,
        concept_labels.cpu(),
    )
