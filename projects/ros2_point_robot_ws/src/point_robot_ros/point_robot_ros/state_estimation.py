"""Minimal scalar Kalman filter for position estimation."""

from dataclasses import dataclass, field
import math


@dataclass
class ScalarKalmanFilter:
    """Estimate one-dimensional position and uncertainty."""

    estimate: float
    variance: float
    process_variance: float

    last_gain: float = field(
        default=0.0,
        init=False,
    )
    last_innovation: float = field(
        default=0.0,
        init=False,
    )

    def __post_init__(self) -> None:
        if not math.isfinite(self.estimate):
            raise ValueError("estimate must be finite")
        if self.variance < 0.0:
            raise ValueError("variance must be non-negative")
        if self.process_variance < 0.0:
            raise ValueError(
                "process_variance must be non-negative"
            )

    def predict(
        self,
        *,
        velocity: float,
        dt: float,
    ) -> float:
        """Predict the next position using a velocity model."""
        if not math.isfinite(velocity):
            raise ValueError("velocity must be finite")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")

        self.estimate += velocity * dt
        self.variance += self.process_variance

        return self.estimate

    def update(
        self,
        *,
        measurement: float,
        measurement_variance: float,
    ) -> float:
        """Correct the estimate using one noisy measurement."""
        if not math.isfinite(measurement):
            raise ValueError("measurement must be finite")
        if (
            not math.isfinite(measurement_variance)
            or measurement_variance <= 0.0
        ):
            raise ValueError(
                "measurement_variance must be positive"
            )

        self.last_innovation = (
            measurement - self.estimate
        )
        self.last_gain = (
            self.variance
            / (self.variance + measurement_variance)
        )

        self.estimate += (
            self.last_gain * self.last_innovation
        )
        self.variance *= 1.0 - self.last_gain

        return self.estimate
