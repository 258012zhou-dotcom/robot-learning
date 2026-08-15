import numpy as np
from numpy.typing import ArrayLike, NDArray


def next_position(position, velocity, dt):
    """计算一维机器人经过 dt 秒后的新位置。"""
    if dt < 0:
        raise ValueError("dt must be non-negative")

    return position + velocity * dt


def simulate_constant_velocity(
    initial_position: ArrayLike,
    velocity: ArrayLike,
    dt: float,
    steps: int,
) -> NDArray[np.float64]:
    """仿真二维点机器人以恒定速度运动时的轨迹。"""
    if dt < 0:
        raise ValueError("dt must be non-negative")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    position_array = np.asarray(initial_position, dtype=float)
    velocity_array = np.asarray(velocity, dtype=float)
    if position_array.shape != (2,) or velocity_array.shape != (2,):
        raise ValueError("initial_position and velocity must be two-dimensional")

    times = np.arange(steps + 1, dtype=float) * dt
    return position_array + times[:, np.newaxis] * velocity_array
