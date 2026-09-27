"""Unit tests for absolute-time alignment and oldest-first weighting."""

import math

import numpy as np
import pytest

from robot_learning.temporal_ensemble import ensemble_action_at


def example_chunks() -> dict[int, np.ndarray]:
    return {
        0: np.array([[100.0], [101.0], [0.3]], dtype=np.float32),
        1: np.array([[200.0], [0.5], [201.0]], dtype=np.float32),
        2: np.array([[0.7], [300.0], [301.0]], dtype=np.float32),
    }


def test_only_matching_absolute_time_is_averaged() -> None:
    result = ensemble_action_at(example_chunks(), target_step=2, decay=0.0)
    assert result.issued_steps == (0, 1, 2)
    assert result.chunk_offsets == (2, 1, 0)
    np.testing.assert_allclose(result.aligned_predictions[:, 0], [0.3, 0.5, 0.7])
    np.testing.assert_allclose(result.action, [0.5])
    np.testing.assert_allclose(result.normalized_weights, [1 / 3] * 3)


def test_exponential_weights_are_oldest_first() -> None:
    result = ensemble_action_at(
        example_chunks(), target_step=2, decay=math.log(2)
    )
    np.testing.assert_allclose(result.normalized_weights, [4 / 7, 2 / 7, 1 / 7])
    np.testing.assert_allclose(result.action, [(0.3 * 4 + 0.5 * 2 + 0.7) / 7])


def test_expired_chunk_is_excluded_and_future_chunk_is_rejected() -> None:
    chunks = {
        0: np.array([[10.0], [11.0]], dtype=np.float32),
        2: np.array([[0.4], [0.5]], dtype=np.float32),
    }
    result = ensemble_action_at(chunks, target_step=2, decay=0.0)
    assert result.issued_steps == (2,)
    np.testing.assert_allclose(result.action, [0.4])
    chunks[3] = np.array([[0.6], [0.7]], dtype=np.float32)
    with pytest.raises(ValueError, match="future"):
        ensemble_action_at(chunks, target_step=2, decay=0.0)


def test_opposite_predictions_can_cancel_without_becoming_valid() -> None:
    chunks = {
        0: np.array([[0.1], [0.8]], dtype=np.float32),
        1: np.array([[-0.8], [0.2]], dtype=np.float32),
    }
    result = ensemble_action_at(chunks, target_step=1, decay=0.0)
    np.testing.assert_allclose(result.action, [0.0], atol=1e-7)


def test_bad_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="decay"):
        ensemble_action_at(example_chunks(), target_step=2, decay=-0.1)
    with pytest.raises(ValueError, match="same shape"):
        ensemble_action_at(
            {0: np.zeros((2, 1)), 1: np.zeros((3, 1))},
            target_step=1,
            decay=0.0,
        )
