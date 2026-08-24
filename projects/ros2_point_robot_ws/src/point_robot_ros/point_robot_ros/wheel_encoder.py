"""Wheel encoder conversion utilities."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WheelEncoderModel:
    """Convert encoder ticks into wheel angle and distance."""

    wheel_radius: float
    ticks_per_revolution: int

    def __post_init__(self) -> None:
        if self.wheel_radius <= 0:
            raise ValueError("wheel_radius must be positive")
        if self.ticks_per_revolution <= 0:
            raise ValueError(
                "ticks_per_revolution must be positive"
            )

    @property
    def distance_per_tick(self) -> float:
        """Return the wheel travel represented by one tick."""
        return (
            2.0
            * math.pi
            * self.wheel_radius
            / self.ticks_per_revolution
        )

    def ticks_to_angle(self, delta_ticks: int) -> float:
        """Convert a tick change into wheel rotation in radians."""
        return (
            2.0
            * math.pi
            * delta_ticks
            / self.ticks_per_revolution
        )

    def ticks_to_distance(self, delta_ticks: int) -> float:
        """Convert a tick change into signed wheel travel."""
        return delta_ticks * self.distance_per_tick

    def count_change_to_distance(
        self,
        *,
        previous_count: int,
        current_count: int,
    ) -> float:
        """Convert two cumulative counts into traveled distance."""
        delta_ticks = current_count - previous_count
        return self.ticks_to_distance(delta_ticks)
