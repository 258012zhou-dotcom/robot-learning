"""一维点质量动力学与 PID 闭环控制。"""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray


def _positive_number(value: float, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True)
class PIDGains:
    """PID 的比例、积分和微分增益。"""

    kp: float
    ki: float
    kd: float

    def __post_init__(self) -> None:
        for name, value in (
            ("kp", self.kp),
            ("ki", self.ki),
            ("kd", self.kd),
        ):
            if not isinstance(value, Real) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative number")
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")


class PIDController:
    """带输出限幅和积分限幅的离散 PID 控制器。"""

    def __init__(
        self,
        gains: PIDGains,
        output_limit: float,
        integral_limit: float,
    ) -> None:
        self.gains = gains
        self.output_limit = _positive_number(output_limit, "output_limit")
        self.integral_limit = _positive_number(
            integral_limit,
            "integral_limit",
        )
        self._integral = 0.0
        self._previous_error: float | None = None

    @property
    def integral(self) -> float:
        return self._integral

    def reset(self) -> None:
        """清除控制器的历史状态。"""
        self._integral = 0.0
        self._previous_error = None

    def step(self, error: float, dt: float) -> float:
        """根据当前误差计算一次受限控制输出。"""
        if not isinstance(error, Real) or isinstance(error, bool):
            raise ValueError("error must be a real number")
        if not np.isfinite(error):
            raise ValueError("error must be finite")
        time_step = _positive_number(dt, "dt")
        error_value = float(error)

        self._integral = float(
            np.clip(
                self._integral + error_value * time_step,
                -self.integral_limit,
                self.integral_limit,
            )
        )
        derivative = (
            0.0
            if self._previous_error is None
            else (error_value - self._previous_error) / time_step
        )
        self._previous_error = error_value

        output = (
            self.gains.kp * error_value
            + self.gains.ki * self._integral
            + self.gains.kd * derivative
        )
        return float(
            np.clip(output, -self.output_limit, self.output_limit)
        )


@dataclass(frozen=True)
class PIDSimulation:
    times: NDArray[np.float64]
    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]
    controls: NDArray[np.float64]


def simulate_pid_point_mass(
    *,
    initial_position: float,
    initial_velocity: float,
    target_position: float,
    mass: float,
    dt: float,
    steps: int,
    gains: PIDGains,
    output_limit: float,
    integral_limit: float,
) -> PIDSimulation:
    """使用半隐式欧拉法仿真受 PID 控制的一维点质量。"""
    mass_value = _positive_number(mass, "mass")
    time_step = _positive_number(dt, "dt")
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")

    initial_values = np.asarray(
        [initial_position, initial_velocity, target_position],
        dtype=float,
    )
    if not np.all(np.isfinite(initial_values)):
        raise ValueError("positions and velocity must be finite")

    controller = PIDController(
        gains=gains,
        output_limit=output_limit,
        integral_limit=integral_limit,
    )
    times = np.arange(steps + 1, dtype=float) * time_step
    positions = np.empty(steps + 1, dtype=float)
    velocities = np.empty(steps + 1, dtype=float)
    controls = np.empty(steps, dtype=float)
    positions[0] = float(initial_position)
    velocities[0] = float(initial_velocity)

    for step_index in range(steps):
        error = float(target_position) - positions[step_index]
        control = controller.step(error, time_step)
        acceleration = control / mass_value

        velocities[step_index + 1] = (
            velocities[step_index] + acceleration * time_step
        )
        positions[step_index + 1] = (
            positions[step_index]
            + velocities[step_index + 1] * time_step
        )
        controls[step_index] = control

    return PIDSimulation(
        times=times,
        positions=positions,
        velocities=velocities,
        controls=controls,
    )
