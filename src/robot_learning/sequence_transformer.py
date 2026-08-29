"""A small Transformer and paired token-order dataset for sequence learning."""

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from robot_learning.attention import TransformerEncoderBlock
from robot_learning.transformer_tokens import (
    DiscreteTokenEmbedding,
    LearnedPositionEmbedding,
)


@dataclass(frozen=True)
class TokenOrderDataset:
    """Paired sequences where labels depend only on A/B order."""

    token_ids: Tensor
    labels: Tensor


@dataclass(frozen=True)
class SequenceEvaluation:
    """Cross-entropy, accuracy, predictions, and labels."""

    loss: float
    accuracy: float
    predictions: Tensor
    labels: Tensor


@dataclass(frozen=True)
class SequenceTrainingResult:
    """Training history and best validation checkpoint."""

    training_losses: list[float]
    validation_losses: list[float]
    validation_accuracies: list[float]
    best_epoch: int
    best_validation_loss: float


class TokenOrderTransformer(nn.Module):
    """Classify token order with optional learned absolute positions."""

    def __init__(
        self,
        vocabulary_size: int,
        maximum_token_count: int,
        embedding_dimension: int,
        head_count: int,
        mlp_hidden_dimension: int,
        layer_count: int,
        use_position_embedding: bool,
    ) -> None:
        super().__init__()
        if type(layer_count) is not int or layer_count <= 0:
            raise ValueError("layer_count must be a positive integer")
        if type(use_position_embedding) is not bool:
            raise ValueError("use_position_embedding must be boolean")
        self.maximum_token_count = maximum_token_count
        self.embedding_dimension = embedding_dimension
        self.token_embedding = DiscreteTokenEmbedding(
            vocabulary_size,
            embedding_dimension,
        )
        self.position_embedding = (
            LearnedPositionEmbedding(maximum_token_count, embedding_dimension)
            if use_position_embedding
            else None
        )
        self.blocks = nn.ModuleList(
            TransformerEncoderBlock(
                embedding_dimension,
                head_count,
                mlp_hidden_dimension,
            )
            for _ in range(layer_count)
        )
        self.final_normalization = nn.LayerNorm(embedding_dimension)
        self.classifier = nn.Linear(embedding_dimension, 2)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return two class logits for every sequence."""
        if token_ids.ndim != 2 or token_ids.shape[1] > self.maximum_token_count:
            raise ValueError("token_ids must have shape (N, L) within maximum length")
        tokens = self.token_embedding(token_ids)
        if self.position_embedding is not None:
            tokens = self.position_embedding(tokens)
        for block in self.blocks:
            tokens = block(tokens).output
        pooled = self.final_normalization(tokens).mean(dim=1)
        return self.classifier(pooled)


def generate_token_order_dataset(
    pair_count: int,
    sequence_length: int,
    vocabulary_size: int,
    seed: int,
) -> TokenOrderDataset:
    """Generate paired permutations labeled by whether token A precedes B."""
    if type(pair_count) is not int or pair_count <= 0:
        raise ValueError("pair_count must be a positive integer")
    if type(sequence_length) is not int or sequence_length < 2:
        raise ValueError("sequence_length must be an integer of at least two")
    if type(vocabulary_size) is not int or vocabulary_size < 4:
        raise ValueError("vocabulary_size must be an integer of at least four")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(seed)
    token_ids = np.empty((pair_count * 2, sequence_length), dtype=np.int64)
    labels = np.empty(pair_count * 2, dtype=np.int64)
    for pair_index in range(pair_count):
        sequence = rng.integers(
            3,
            vocabulary_size,
            size=sequence_length,
            dtype=np.int64,
        )
        first_position, second_position = np.sort(
            rng.choice(sequence_length, size=2, replace=False)
        )
        positive = sequence.copy()
        positive[first_position] = 1
        positive[second_position] = 2
        negative = positive.copy()
        negative[first_position] = 2
        negative[second_position] = 1
        start = pair_index * 2
        token_ids[start] = positive
        token_ids[start + 1] = negative
        labels[start:start + 2] = (1, 0)
    return TokenOrderDataset(
        token_ids=torch.from_numpy(token_ids),
        labels=torch.from_numpy(labels),
    )


def create_sequence_data_loader(
    dataset: TokenOrderDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a reproducible token-sequence DataLoader."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        TensorDataset(dataset.token_ids, dataset.labels),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def train_sequence_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
) -> float:
    """Train one epoch and return sample-weighted cross-entropy."""
    model.train()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    sample_count = 0
    for token_ids, labels in data_loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(token_ids), labels)
        loss.backward()
        optimizer.step()
        batch_size = token_ids.shape[0]
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("training data loader cannot be empty")
    return total_loss / sample_count


def evaluate_sequence_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> SequenceEvaluation:
    """Evaluate token-order classification without gradients."""
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    sample_count = 0
    prediction_batches: list[Tensor] = []
    label_batches: list[Tensor] = []
    with torch.no_grad():
        for token_ids, labels in data_loader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)
            logits = model(token_ids)
            loss = loss_function(logits, labels)
            predictions = logits.argmax(dim=1)
            batch_size = token_ids.shape[0]
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size
            prediction_batches.append(predictions.cpu())
            label_batches.append(labels.cpu())
    if sample_count == 0:
        raise ValueError("evaluation data loader cannot be empty")
    predictions = torch.cat(prediction_batches)
    labels = torch.cat(label_batches)
    return SequenceEvaluation(
        loss=total_loss / sample_count,
        accuracy=float((predictions == labels).float().mean().item()),
        predictions=predictions,
        labels=labels,
    )


def train_sequence_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> SequenceTrainingResult:
    """Train a sequence classifier and restore its best validation state."""
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    for value, name, allow_zero in (
        (learning_rate, "learning_rate", False),
        (weight_decay, "weight_decay", True),
    ):
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value < 0.0
            or (not allow_zero and value == 0.0)
        ):
            raise ValueError(f"{name} has an invalid value")

    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    training_losses: list[float] = []
    validation_losses: list[float] = []
    validation_accuracies: list[float] = []
    best_epoch = 0
    best_validation_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    for epoch_index in range(epochs):
        training_losses.append(
            train_sequence_epoch(model, train_loader, optimizer, device)
        )
        validation = evaluate_sequence_model(
            model,
            validation_loader,
            device,
        )
        validation_losses.append(validation.loss)
        validation_accuracies.append(validation.accuracy)
        if validation.loss < best_validation_loss:
            best_validation_loss = validation.loss
            best_epoch = epoch_index + 1
            best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return SequenceTrainingResult(
        training_losses=training_losses,
        validation_losses=validation_losses,
        validation_accuracies=validation_accuracies,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
    )
