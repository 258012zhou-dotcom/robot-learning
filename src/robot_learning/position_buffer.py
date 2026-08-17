from collections import deque
from collections.abc import Sequence

import numpy as np


class PositionBuffer:
    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")

        self._positions: deque[np.ndarray] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._positions)

    def append(
        self,
        position: Sequence[float] | np.ndarray,
    ) -> None:
        point = np.asarray(position, dtype=float)

        if point.shape != (2,):
            raise ValueError("position must have shape (2,)")

        self._positions.append(point.copy())

    def mean(self) -> np.ndarray:
        if not self._positions:
            raise ValueError("cannot calculate mean of an empty buffer")

        positions = np.stack(self._positions)
        return positions.mean(axis=0)