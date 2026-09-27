"""Time-align overlapping action chunks before temporal ensembling."""

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TemporalEnsembleResult:
    action: np.ndarray
    issued_steps: tuple[int, ...]
    chunk_offsets: tuple[int, ...]
    normalized_weights: tuple[float, ...]
    aligned_predictions: np.ndarray


def ensemble_action_at(
    chunks_by_issued_step: Mapping[int, np.ndarray],
    *,
    target_step: int,
    decay: float,
) -> TemporalEnsembleResult:
    """Average only predictions for target_step; oldest active chunk is i=0.

    Each chunk has shape (horizon, action_size), and its row j predicts the
    action for absolute step issued_step + j. Callers must keep Episodes separate.
    """
    if type(target_step) is not int or target_step < 0:
        raise ValueError("target_step must be a non-negative integer")
    if not math.isfinite(decay) or decay < 0:
        raise ValueError("decay must be finite and non-negative")
    if not chunks_by_issued_step:
        raise ValueError("at least one chunk is required")

    issued_steps: list[int] = []
    offsets: list[int] = []
    aligned: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for issued_step, value in sorted(chunks_by_issued_step.items()):
        if type(issued_step) is not int or issued_step < 0:
            raise ValueError("issued steps must be non-negative integers")
        if issued_step > target_step:
            raise ValueError("future chunks cannot contribute to a past action")
        chunk = np.asarray(value, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] == 0:
            raise ValueError("each chunk must have shape (horizon, action_size)")
        if expected_shape is None:
            expected_shape = chunk.shape
        elif chunk.shape != expected_shape:
            raise ValueError("all chunks must have the same shape")
        if not np.all(np.isfinite(chunk)):
            raise ValueError("chunk values must be finite")
        offset = target_step - issued_step
        if offset >= chunk.shape[0]:
            continue  # The chunk no longer predicts this absolute time.
        issued_steps.append(issued_step)
        offsets.append(offset)
        aligned.append(chunk[offset])

    if not aligned:
        raise ValueError("no chunk predicts the target step")
    raw_weights = np.exp(-decay * np.arange(len(aligned), dtype=np.float64))
    weights = raw_weights / raw_weights.sum()
    predictions = np.stack(aligned)
    return TemporalEnsembleResult(
        action=np.average(predictions, axis=0, weights=weights).astype(np.float32),
        issued_steps=tuple(issued_steps),
        chunk_offsets=tuple(offsets),
        normalized_weights=tuple(float(weight) for weight in weights),
        aligned_predictions=predictions.astype(np.float32),
    )
