"""A small 2D closed-loop task with two valid routes around a circle."""

from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import numpy as np


@dataclass(frozen=True)
class NavigationScene:
    """Geometry shared by the upper- and lower-route demonstrations."""

    start_x: float
    start_y: float
    goal_x: float
    goal_y: float
    obstacle_x: float
    obstacle_y: float
    obstacle_radius: float


def sample_scene(seed: int) -> NavigationScene:
    """Return reproducible geometry; the route is deliberately not sampled here."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    return NavigationScene(
        start_x=float(rng.uniform(-1.30, -1.15)),
        start_y=float(rng.uniform(-0.06, 0.06)),
        goal_x=float(rng.uniform(1.15, 1.30)),
        goal_y=float(rng.uniform(-0.06, 0.06)),
        obstacle_x=float(rng.uniform(-0.08, 0.08)),
        obstacle_y=float(rng.uniform(-0.06, 0.06)),
        obstacle_radius=float(rng.uniform(0.34, 0.42)),
    )


def segment_clearance(
    start: np.ndarray, end: np.ndarray, center: np.ndarray, radius: float
) -> float:
    """Signed clearance of a motion segment from a circular obstacle."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    if start.shape != (2,) or end.shape != (2,) or center.shape != (2,):
        raise ValueError("points must be two-dimensional")
    if (
        not np.all(np.isfinite(start))
        or not np.all(np.isfinite(end))
        or not np.all(np.isfinite(center))
    ):
        raise ValueError("points must be finite")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    displacement = end - start
    squared_length = float(np.dot(displacement, displacement))
    fraction = (
        0.0
        if squared_length == 0.0
        else float(np.clip(np.dot(center - start, displacement) / squared_length, 0.0, 1.0))
    )
    closest = start + fraction * displacement
    return float(np.linalg.norm(closest - center) - radius)


class TwoRouteNavigationEnv(gym.Env[np.ndarray, np.ndarray]):
    """Velocity-controlled point robot; touching the obstacle ends the Episode."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        dt: float = 0.1,
        max_episode_steps: int = 80,
        goal_tolerance: float = 0.08,
    ) -> None:
        super().__init__()
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if type(max_episode_steps) is not int or max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be a positive integer")
        if not np.isfinite(goal_tolerance) or goal_tolerance <= 0:
            raise ValueError("goal_tolerance must be finite and positive")
        self.dt = float(dt)
        self.max_episode_steps = max_episode_steps
        self.goal_tolerance = float(goal_tolerance)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        position_bound = 2.0 + max_episode_steps * dt
        self.observation_space = spaces.Box(
            low=np.asarray(
                [-position_bound, -position_bound, -2.0, -2.0, -1.0, -1.0, 0.0],
                dtype=np.float32,
            ),
            high=np.asarray(
                [position_bound, position_bound, 2.0, 2.0, 1.0, 1.0, 1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.scene: NavigationScene | None = None
        self._position = np.zeros(2, dtype=np.float64)
        self._elapsed_steps = 0
        self._finished = False

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if options:
            raise ValueError("this teaching task does not accept reset options")
        scene_seed = (
            int(self.np_random.integers(0, 2**31)) if seed is None else seed
        )
        self.scene = sample_scene(scene_seed)
        self._position = np.array(
            [self.scene.start_x, self.scene.start_y], dtype=np.float64
        )
        self._elapsed_steps = 0
        self._finished = False
        return self._observation(), {
            "scene_seed": scene_seed,
            "is_success": False,
            "collision": False,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.scene is None or self._finished:
            raise RuntimeError("reset before stepping or after an Episode ends")
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (2,) or not np.all(np.isfinite(command)):
            raise ValueError("action must be a finite two-dimensional velocity")
        if np.any(command < self.action_space.low) or np.any(command > self.action_space.high):
            raise ValueError("action exceeds the declared velocity bounds")
        next_position = self._position + self.dt * command
        center = np.array(
            [self.scene.obstacle_x, self.scene.obstacle_y], dtype=np.float64
        )
        clearance = segment_clearance(
            self._position, next_position, center, self.scene.obstacle_radius
        )
        collision = clearance <= 0.0
        self._position = next_position
        self._elapsed_steps += 1
        distance = float(
            np.linalg.norm(
                self._position - [self.scene.goal_x, self.scene.goal_y]
            )
        )
        success = distance <= self.goal_tolerance and not collision
        terminated = collision or success
        truncated = self._elapsed_steps >= self.max_episode_steps and not terminated
        self._finished = terminated or truncated
        info = {
            "is_success": success,
            "collision": collision,
            "distance_to_goal": distance,
            "obstacle_clearance": clearance,
        }
        return (
            self._observation(),
            -distance - (10.0 if collision else 0.0),
            terminated,
            truncated,
            info,
        )

    def _observation(self) -> np.ndarray:
        if self.scene is None:
            raise RuntimeError("reset before requesting an observation")
        return np.asarray(
            [
                *self._position,
                self.scene.goal_x,
                self.scene.goal_y,
                self.scene.obstacle_x,
                self.scene.obstacle_y,
                self.scene.obstacle_radius,
            ],
            dtype=np.float32,
        )


class TwoRouteExpert:
    """Commit to one waypoint outside the circle, then continue to the goal."""

    def __init__(self, route: int, *, dt: float = 0.1, speed: float = 0.8) -> None:
        if route not in (-1, 1):
            raise ValueError("route must be -1 (lower) or +1 (upper)")
        if (
            not np.isfinite(dt)
            or dt <= 0
            or not np.isfinite(speed)
            or not 0 < speed <= 1
        ):
            raise ValueError("dt and speed must be positive and finite; speed <= 1")
        self.route = route
        self.dt = float(dt)
        self.speed = float(speed)
        self._past_waypoint = False

    def reset(self) -> None:
        self._past_waypoint = False

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float64)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise ValueError("observation must be a finite seven-dimensional vector")
        position = value[:2]
        waypoint = np.asarray(
            [value[4], value[5] + self.route * (value[6] + 0.45)]
        )
        if np.linalg.norm(waypoint - position) <= 1e-4:
            self._past_waypoint = True
        destination = value[2:4] if self._past_waypoint else waypoint
        error = destination - position
        distance = float(np.linalg.norm(error))
        if distance == 0.0:
            return np.zeros(2, dtype=np.float32)
        velocity = error * min(self.speed / distance, 1.0 / self.dt)
        return velocity.astype(np.float32)
