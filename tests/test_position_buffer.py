import numpy as np
import pytest

from robot_learning.position_buffer import PositionBuffer


def test_buffer_calculates_mean_position():
    buffer = PositionBuffer(capacity=3)

    buffer.append([0.0, 0.0])
    buffer.append([2.0, 1.0])
    buffer.append([4.0, 2.0])

    np.testing.assert_allclose(buffer.mean(), [2.0, 1.0])


def test_buffer_keeps_only_recent_positions():
    buffer = PositionBuffer(capacity=2)

    buffer.append([0.0, 0.0])
    buffer.append([2.0, 0.0])
    buffer.append([4.0, 0.0])

    assert len(buffer) == 2
    np.testing.assert_allclose(buffer.mean(), [3.0, 0.0])


def test_buffer_copies_input_position():
    buffer = PositionBuffer(capacity=2)
    position = np.array([1.0, 2.0])

    buffer.append(position)
    position[0] = 100.0

    np.testing.assert_allclose(buffer.mean(), [1.0, 2.0])


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_buffer_rejects_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        PositionBuffer(capacity)


@pytest.mark.parametrize(
    "position",
    [
        [],
        [1.0],
        [1.0, 2.0, 3.0],
        [[1.0, 2.0]],
    ],
)
def test_buffer_rejects_invalid_position_shape(position):
    buffer = PositionBuffer(capacity=2)

    with pytest.raises(ValueError):
        buffer.append(position)


def test_empty_buffer_has_no_mean():
    buffer = PositionBuffer(capacity=2)

    with pytest.raises(ValueError):
        buffer.mean()