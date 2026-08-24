"""Basic feedback controllers for the point robot."""

from dataclasses import dataclass, field


def proportional_command(
    *,
    target: float,
    current: float,
    kp: float,
    max_command: float,
) -> float:
    """Compute a saturated proportional control command."""
    if kp <= 0:
        raise ValueError("kp must be positive")
    if max_command <= 0:
        raise ValueError("max_command must be positive")

    error = target - current
    raw_command = kp * error

    return max(-max_command, min(max_command, raw_command))


def simulate_proportional_joint(
    *,
    target: float,
    initial_position: float,
    kp: float,
    max_velocity: float,
    dt: float,
    steps: int,
) -> list[float]:
    """Simulate a velocity-controlled joint using P control."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    current = initial_position
    positions = [current]

    for _ in range(steps):
        velocity = proportional_command(
            target=target,
            current=current,
            kp=kp,
            max_command=max_velocity,
        )
        current += velocity * dt
        positions.append(current)

    return positions


@dataclass
class PIDController:
    """A PID controller with output and integral limits."""

    kp: float
    ki: float
    kd: float
    max_command: float
    integral_limit: float

    _integral: float = field(default=0.0, init=False)
    _previous_error: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.kp < 0 or self.ki < 0 or self.kd < 0:
            raise ValueError("PID gains must be non-negative")
        if self.max_command <= 0:
            raise ValueError("max_command must be positive")
        if self.integral_limit <= 0:
            raise ValueError("integral_limit must be positive")

    def update(
        self,
        *,
        target: float,
        current: float,
        dt: float,
    ) -> float:
        """Update the controller and return a limited command."""
        if dt <= 0:
            raise ValueError("dt must be positive")

        error = target - current

        self._integral += error * dt
        self._integral = max(
            -self.integral_limit,
            min(self.integral_limit, self._integral),
        )

        if self._previous_error is None:
            derivative = 0.0
        else:
            derivative = (error - self._previous_error) / dt

        self._previous_error = error

        raw_command = (
            self.kp * error
            + self.ki * self._integral
            + self.kd * derivative
        )

        return max(
            -self.max_command,
            min(self.max_command, raw_command),
        )

    def reset(self) -> None:
        """Clear the stored integral and previous error."""
        self._integral = 0.0
        self._previous_error = None


def simulate_pid_joint(
    *,
    controller: PIDController,
    target: float,
    initial_position: float,
    velocity_disturbance: float,
    dt: float,
    steps: int,
) -> list[float]:
    """Simulate a PID-controlled joint with a velocity disturbance."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    controller.reset()

    current = initial_position
    positions = [current]

    for _ in range(steps):
        command = controller.update(
            target=target,
            current=current,
            dt=dt,
        )

        actual_velocity = command + velocity_disturbance
        current += actual_velocity * dt
        positions.append(current)

    return positions


def simulate_second_order_joint(
    *,
    controller: PIDController,
    target: float,
    initial_position: float,
    initial_velocity: float,
    inertia: float,
    damping: float,
    dt: float,
    steps: int,
) -> list[float]:
    """Simulate an inertial joint controlled by PID."""
    if inertia <= 0:
        raise ValueError("inertia must be positive")
    if damping < 0:
        raise ValueError("damping must be non-negative")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    controller.reset()

    position = initial_position
    velocity = initial_velocity
    positions = [position]

    for _ in range(steps):
        command = controller.update(
            target=target,
            current=position,
            dt=dt,
        )

        acceleration = (
            command - damping * velocity
        ) / inertia

        velocity += acceleration * dt
        position += velocity * dt
        positions.append(position)

    return positions
