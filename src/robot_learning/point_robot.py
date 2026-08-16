import numpy as np
from numpy.typing import ArrayLike, NDArray
from dataclasses import dataclass

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

@dataclass(frozen=True)
class TrajectoryStats:
    displacement: np.ndarray
    displacement_distance: float
    path_length: float
    average_speed: float


def analyze_trajectory(trajectory: np.ndarray, dt: float) -> TrajectoryStats:
    points = np.asarray(trajectory, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("trajectory must have shape (T, 2)")
    if points.shape[0] < 2:
        raise ValueError("trajectory must contain at least two points")
    if dt <= 0:
        raise ValueError("dt must be positive")

    step_displacements = np.diff(points, axis=0)
    step_distances = np.linalg.norm(step_displacements, axis=1)

    displacement = points[-1] - points[0]
    displacement_distance = float(np.linalg.norm(displacement))
    path_length = float(step_distances.sum())

    duration = (points.shape[0] - 1) * dt
    average_speed = path_length / duration

    return TrajectoryStats(
        displacement=displacement,
        displacement_distance=displacement_distance,
        path_length=path_length,
        average_speed=average_speed,
    )