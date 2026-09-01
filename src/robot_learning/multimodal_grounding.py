"""Synthetic scenes for learning language-conditioned visual grounding.

The module intentionally contains data generation only.  A sample follows this
contract:

1. ``image`` shows several color-shape objects.
2. ``instruction`` names exactly one of those objects.
3. ``target_center`` gives that object's normalized image coordinate.

Keeping this contract separate from the model makes label errors easier to find
before a training loop is introduced.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from robot_learning.vision_language import SmallTextEncoder


COLOR_NAMES = ("red", "green", "blue")
SHAPE_NAMES = ("square", "circle")


# ---------------------------------------------------------------------------
# 1. Dataset contract and synthetic scene generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundingDataset:
    """A batch of images, instructions, scene objects, and target locations.

    Tensor shapes:
    - ``images``: ``(N, 3, H, W)``
    - ``scene_concept_labels``: ``(N, K)``
    - ``scene_centers``: ``(N, K, 2)`` in normalized ``(x, y)`` coordinates
    - ``target_indices``: ``(N,)``; index of the requested object in each scene
    - ``target_centers``: ``(N, 2)`` in normalized ``(x, y)`` coordinates
    """

    images: Tensor
    instructions: tuple[str, ...]
    scene_concept_labels: Tensor
    scene_centers: Tensor
    target_indices: Tensor
    target_centers: Tensor
    concept_descriptions: tuple[str, ...]


def generate_grounding_dataset(
    sample_count: int,
    image_size: int,
    object_count: int,
    seed: int,
) -> GroundingDataset:
    """Generate reproducible multi-object scenes and unambiguous instructions."""
    _validate_generation_arguments(
        sample_count=sample_count,
        image_size=image_size,
        object_count=object_count,
        seed=seed,
    )

    # Each concept is a unique color-shape pair, such as ``red square``.
    concept_descriptions = tuple(
        f"{color_name} {shape_name}"
        for color_name in COLOR_NAMES
        for shape_name in SHAPE_NAMES
    )
    concept_count = len(concept_descriptions)
    rng = np.random.default_rng(seed)

    # Preallocate every output so their sample dimension cannot drift apart.
    images_rgb = np.empty(
        (sample_count, image_size, image_size, 3),
        dtype=np.uint8,
    )
    scene_labels = np.empty((sample_count, object_count), dtype=np.int64)
    scene_centers = np.empty(
        (sample_count, object_count, 2),
        dtype=np.float32,
    )
    target_indices = np.empty(sample_count, dtype=np.int64)
    instructions: list[str] = []

    base_colors = {
        "red": np.array((220, 40, 40), dtype=np.int16),
        "green": np.array((40, 200, 40), dtype=np.int16),
        "blue": np.array((40, 70, 220), dtype=np.int16),
    }
    anchor_points = _make_anchor_points(image_size)

    for sample_index in range(sample_count):
        # Low-intensity noise prevents the background from being perfectly flat.
        image = rng.integers(
            0,
            26,
            size=(image_size, image_size, 3),
            dtype=np.uint8,
        )

        # Concepts are unique inside one scene, so the instruction has one answer.
        labels = rng.choice(
            concept_count,
            size=object_count,
            replace=False,
        )
        anchor_indices = rng.choice(
            len(anchor_points),
            size=object_count,
            replace=False,
        )

        for object_index, (label, anchor_index) in enumerate(
            zip(labels, anchor_indices)
        ):
            center_x, center_y = anchor_points[int(anchor_index)]
            color_name, shape_name = concept_descriptions[int(label)].split()
            half_size = int(
                rng.integers(
                    max(2, image_size // 12),
                    max(3, image_size // 9) + 1,
                )
            )
            color_jitter = rng.integers(-12, 13, size=3)
            color = tuple(
                int(value)
                for value in np.clip(
                    base_colors[color_name] + color_jitter,
                    0,
                    255,
                )
            )

            _draw_shape(
                image=image,
                shape_name=shape_name,
                center=(center_x, center_y),
                half_size=half_size,
                color=color,
            )
            scene_centers[sample_index, object_index] = (
                center_x / image_size,
                center_y / image_size,
            )

        target_index = int(rng.integers(0, object_count))
        target_label = int(labels[target_index])
        instructions.append(
            f"select the {concept_descriptions[target_label]}"
        )
        images_rgb[sample_index] = image
        scene_labels[sample_index] = labels
        target_indices[sample_index] = target_index

    # Convert channel-last uint8 images into PyTorch's channel-first float format.
    image_tensor = (
        torch.from_numpy(images_rgb)
        .permute(0, 3, 1, 2)
        .to(dtype=torch.float32)
        .div(255.0)
    )
    scene_center_tensor = torch.from_numpy(scene_centers)
    target_index_tensor = torch.from_numpy(target_indices)
    sample_indices = torch.arange(sample_count)
    target_centers = scene_center_tensor[
        sample_indices,
        target_index_tensor,
    ].clone()

    return GroundingDataset(
        images=image_tensor,
        instructions=tuple(instructions),
        scene_concept_labels=torch.from_numpy(scene_labels),
        scene_centers=scene_center_tensor,
        target_indices=target_index_tensor,
        target_centers=target_centers,
        concept_descriptions=concept_descriptions,
    )


# ---------------------------------------------------------------------------
# 2. Grounding baselines and multimodal fusion model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundingOutput:
    """Predicted target centers and optional spatial attention probabilities.

    ``predicted_centers`` always has shape ``(N, 2)``.  Models that inspect an
    image also return ``attention_weights`` with shape ``(N, P)``, where ``P``
    is the number of spatial visual tokens.
    """

    predicted_centers: Tensor
    attention_weights: Tensor | None


def predict_fixed_center(sample_count: int) -> GroundingOutput:
    """Return the image center without looking at either input modality."""
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    return GroundingOutput(
        predicted_centers=torch.full((sample_count, 2), 0.5),
        attention_weights=None,
    )


class SpatialVisualEncoder(nn.Module):
    """Convert RGB images into a sequence of spatial visual tokens.

    Shape flow for 64x64 input images:

    ``(N, 3, 64, 64) -> (N, D, 8, 8) -> (N, 64, D)``

    Unlike global average pooling, this keeps *where* each visual feature came
    from.  Grounding needs that spatial information to predict a location.
    """

    def __init__(self, embedding_dimension: int = 32) -> None:
        super().__init__()
        if type(embedding_dimension) is not int or embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be a positive integer")
        self.embedding_dimension = embedding_dimension
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                32,
                embedding_dimension,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
        )
        self.normalization = nn.LayerNorm(embedding_dimension)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Return visual tokens ``(N, P, D)`` and grid centers ``(P, 2)``."""
        _validate_images(images)
        feature_map = self.features(images)
        _, _, feature_height, feature_width = feature_map.shape

        # Flatten spatial cells into tokens while keeping channels as features.
        visual_tokens = feature_map.flatten(start_dim=2).transpose(1, 2)
        visual_tokens = self.normalization(visual_tokens)
        grid_centers = _make_normalized_grid_centers(
            feature_height=feature_height,
            feature_width=feature_width,
            device=images.device,
            dtype=images.dtype,
        )
        return visual_tokens, grid_centers


class VisionOnlyGrounder(nn.Module):
    """Attend to one salient image region without access to the instruction."""

    def __init__(self, embedding_dimension: int = 32) -> None:
        super().__init__()
        self.visual_encoder = SpatialVisualEncoder(embedding_dimension)

        # This one learned query is shared by every sample.  Because it never
        # sees language, it cannot know which of several objects was requested.
        self.query = nn.Parameter(torch.empty(embedding_dimension))
        nn.init.normal_(self.query, mean=0.0, std=0.02)

    def forward(self, images: Tensor) -> GroundingOutput:
        """Predict coordinates using only image saliency."""
        visual_tokens, grid_centers = self.visual_encoder(images)
        query = self.query.expand(images.shape[0], -1)
        return _attend_to_spatial_tokens(
            visual_tokens=visual_tokens,
            query=query,
            grid_centers=grid_centers,
        )


class LanguageOnlyGrounder(nn.Module):
    """Predict a location from text while remaining blind to the image."""

    def __init__(
        self,
        vocabulary_size: int,
        maximum_token_count: int,
        embedding_dimension: int = 32,
        text_head_count: int = 4,
        padding_id: int = 0,
    ) -> None:
        super().__init__()
        self.text_encoder = SmallTextEncoder(
            vocabulary_size=vocabulary_size,
            maximum_token_count=maximum_token_count,
            embedding_dimension=embedding_dimension,
            head_count=text_head_count,
            padding_id=padding_id,
        )
        self.coordinate_head = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.GELU(),
            nn.Linear(embedding_dimension, 2),
            nn.Sigmoid(),
        )

    def forward(self, token_ids: Tensor) -> GroundingOutput:
        """Predict coordinates from text tokens with shape ``(N, L)``."""
        text_embedding = self.text_encoder(token_ids)
        return GroundingOutput(
            predicted_centers=self.coordinate_head(text_embedding),
            attention_weights=None,
        )


class MultimodalGrounder(nn.Module):
    """Use language as a query over spatial visual tokens.

    This is a compact Cross-Attention-like grounding model:

    1. CNN produces one visual token per image region.
    2. Transformer text encoder produces one instruction vector.
    3. The instruction vector scores every visual region.
    4. A weighted average of region coordinates becomes the prediction.
    """

    def __init__(
        self,
        vocabulary_size: int,
        maximum_token_count: int,
        embedding_dimension: int = 32,
        text_head_count: int = 4,
        padding_id: int = 0,
    ) -> None:
        super().__init__()
        self.visual_encoder = SpatialVisualEncoder(embedding_dimension)
        self.text_encoder = SmallTextEncoder(
            vocabulary_size=vocabulary_size,
            maximum_token_count=maximum_token_count,
            embedding_dimension=embedding_dimension,
            head_count=text_head_count,
            padding_id=padding_id,
        )

        # Query and Key projections place language and vision in a comparable
        # feature space.  Their dot product measures regional relevance.
        self.query_projection = nn.Linear(
            embedding_dimension,
            embedding_dimension,
        )
        self.key_projection = nn.Linear(
            embedding_dimension,
            embedding_dimension,
        )

    def forward(self, images: Tensor, token_ids: Tensor) -> GroundingOutput:
        """Predict target coordinates from paired image and language inputs."""
        _validate_images(images)
        if token_ids.ndim != 2 or token_ids.shape[0] != images.shape[0]:
            raise ValueError("token_ids must have shape (N, L) matching images")

        visual_tokens, grid_centers = self.visual_encoder(images)
        text_embedding = self.text_encoder(token_ids)
        visual_keys = self.key_projection(visual_tokens)
        language_query = self.query_projection(text_embedding)
        return _attend_to_spatial_tokens(
            visual_tokens=visual_keys,
            query=language_query,
            grid_centers=grid_centers,
        )


# ---------------------------------------------------------------------------
# 3. Training loss and evaluation metric
# ---------------------------------------------------------------------------


def calculate_grounding_loss(
    predicted_centers: Tensor,
    target_centers: Tensor,
) -> Tensor:
    """Return Smooth L1 coordinate loss for normalized ``(x, y)`` centers."""
    _validate_center_pair(predicted_centers, target_centers)
    return functional.smooth_l1_loss(predicted_centers, target_centers)


def calculate_mean_center_error(
    predicted_centers: Tensor,
    target_centers: Tensor,
) -> float:
    """Return mean Euclidean target error in normalized image coordinates."""
    _validate_center_pair(predicted_centers, target_centers)
    errors = torch.linalg.vector_norm(
        predicted_centers - target_centers,
        dim=1,
    )
    return float(errors.mean().item())


# ---------------------------------------------------------------------------
# 4. Internal helpers (read only when tracing implementation details)
# ---------------------------------------------------------------------------


def _attend_to_spatial_tokens(
    visual_tokens: Tensor,
    query: Tensor,
    grid_centers: Tensor,
) -> GroundingOutput:
    """Turn query-token similarity into a spatial coordinate prediction."""
    embedding_dimension = visual_tokens.shape[-1]
    attention_scores = torch.einsum(
        "npd,nd->np",
        visual_tokens,
        query,
    ) / math.sqrt(embedding_dimension)
    attention_weights = torch.softmax(attention_scores, dim=1)

    # Soft-argmax is differentiable: training can move probability toward the
    # requested region without choosing a hard grid index during forward().
    predicted_centers = attention_weights @ grid_centers
    return GroundingOutput(
        predicted_centers=predicted_centers,
        attention_weights=attention_weights,
    )


def _make_normalized_grid_centers(
    feature_height: int,
    feature_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return normalized ``(x, y)`` centers for flattened feature-map cells."""
    x_coordinates = (
        torch.arange(feature_width, device=device, dtype=dtype) + 0.5
    ) / feature_width
    y_coordinates = (
        torch.arange(feature_height, device=device, dtype=dtype) + 0.5
    ) / feature_height
    grid_y, grid_x = torch.meshgrid(
        y_coordinates,
        x_coordinates,
        indexing="ij",
    )
    return torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1)


def _validate_images(images: Tensor) -> None:
    """Validate the RGB image batch contract used by all visual models."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape (N, 3, H, W)")
    if not images.is_floating_point():
        raise ValueError("images must use a floating-point dtype")


def _validate_center_pair(
    predicted_centers: Tensor,
    target_centers: Tensor,
) -> None:
    """Validate two matching coordinate batches before metric calculation."""
    if predicted_centers.ndim != 2 or predicted_centers.shape[1] != 2:
        raise ValueError("predicted_centers must have shape (N, 2)")
    if target_centers.shape != predicted_centers.shape:
        raise ValueError("target_centers must match predicted_centers")
    if not predicted_centers.is_floating_point():
        raise ValueError("predicted_centers must use a floating-point dtype")
    if not target_centers.is_floating_point():
        raise ValueError("target_centers must use a floating-point dtype")


def _make_anchor_points(image_size: int) -> tuple[tuple[int, int], ...]:
    """Return separated candidate centers on a three-by-three image grid."""
    fractions = (0.2, 0.5, 0.8)
    return tuple(
        (int(round(x * (image_size - 1))), int(round(y * (image_size - 1))))
        for y in fractions
        for x in fractions
    )


def _draw_shape(
    image: np.ndarray,
    shape_name: str,
    center: tuple[int, int],
    half_size: int,
    color: tuple[int, int, int],
) -> None:
    """Draw one filled square or circle into an RGB NumPy image."""
    center_x, center_y = center
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
            center,
            half_size,
            color,
            thickness=-1,
        )


def _validate_generation_arguments(
    sample_count: int,
    image_size: int,
    object_count: int,
    seed: int,
) -> None:
    """Reject invalid data settings instead of silently changing them."""
    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if type(image_size) is not int or image_size < 32:
        raise ValueError("image_size must be an integer of at least 32")
    if type(object_count) is not int or not 2 <= object_count <= 6:
        raise ValueError("object_count must be an integer from 2 to 6")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
