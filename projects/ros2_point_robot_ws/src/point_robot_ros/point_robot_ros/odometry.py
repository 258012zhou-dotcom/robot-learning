"""Differential-drive odometry calculations."""

from dataclasses import dataclass
import math
from point_robot_ros.wheel_encoder import WheelEncoderModel


@dataclass(frozen=True)
class Pose2D:
    """A planar position and heading estimate."""

    x: float
    y: float
    heading: float


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the range [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def update_odometry(
    *,
    pose: Pose2D,
    left_distance: float,
    right_distance: float,
    wheel_base: float,
) -> Pose2D:
    """Update a differential-drive pose from wheel travel."""
    if wheel_base <= 0:
        raise ValueError("wheel_base must be positive")

    center_distance = (
        left_distance + right_distance
    ) / 2.0
    heading_change = (
        right_distance - left_distance
    ) / wheel_base

    if abs(heading_change) < 1e-12:
        body_x = center_distance
        body_y = 0.0
    else:
        turn_radius = center_distance / heading_change
        body_x = turn_radius * math.sin(heading_change)
        body_y = turn_radius * (
            1.0 - math.cos(heading_change)
        )

    world_x = (
        math.cos(pose.heading) * body_x
        - math.sin(pose.heading) * body_y
    )
    world_y = (
        math.sin(pose.heading) * body_x
        + math.cos(pose.heading) * body_y
    )

    return Pose2D(
        x=pose.x + world_x,
        y=pose.y + world_y,
        heading=normalize_angle(
            pose.heading + heading_change
        ),
    )


class DifferentialDriveOdometry:
    """Estimate robot pose from cumulative wheel encoder counts."""

    def __init__(
        self,
        *,
        left_encoder: WheelEncoderModel,
        right_encoder: WheelEncoderModel,
        wheel_base: float,
        initial_pose: Pose2D = Pose2D(0.0, 0.0, 0.0),
    ) -> None:
        if wheel_base <= 0:
            raise ValueError("wheel_base must be positive")

        self.left_encoder = left_encoder
        self.right_encoder = right_encoder
        self.wheel_base = wheel_base
        self.pose = initial_pose

        self._previous_left_count: int | None = None
        self._previous_right_count: int | None = None

    def update(
        self,
        *,
        left_count: int,
        right_count: int,
    ) -> Pose2D:
        """Update the pose using cumulative encoder counts."""
        if (
            self._previous_left_count is None
            or self._previous_right_count is None
        ):
            self._previous_left_count = left_count
            self._previous_right_count = right_count
            return self.pose

        left_distance = (
            self.left_encoder.count_change_to_distance(
                previous_count=self._previous_left_count,
                current_count=left_count,
            )
        )
        right_distance = (
            self.right_encoder.count_change_to_distance(
                previous_count=self._previous_right_count,
                current_count=right_count,
            )
        )

        self.pose = update_odometry(
            pose=self.pose,
            left_distance=left_distance,
            right_distance=right_distance,
            wheel_base=self.wheel_base,
        )

        self._previous_left_count = left_count
        self._previous_right_count = right_count

        return self.pose

    def reset(
        self,
        pose: Pose2D = Pose2D(0.0, 0.0, 0.0),
    ) -> None:
        """Reset the pose and forget previous encoder counts."""
        self.pose = pose
        self._previous_left_count = None
        self._previous_right_count = None
