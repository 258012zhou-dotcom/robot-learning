"""Planar kinematics for the point robot camera joint."""

import math


CAMERA_MOUNT_X = 0.25
CAMERA_MOUNT_Y = 0.0
CAMERA_MOUNT_Z = 0.28
CAMERA_YAW_LIMIT = 1.5708


def camera_forward_point(
    yaw: float,
    distance: float,
) -> tuple[float, float, float]:
    """Return a forward camera point expressed in base_link."""
    if not math.isfinite(yaw):
        raise ValueError("yaw must be finite")
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("distance must be a non-negative finite number")

    x = CAMERA_MOUNT_X + distance * math.cos(yaw)
    y = CAMERA_MOUNT_Y + distance * math.sin(yaw)
    z = CAMERA_MOUNT_Z

    return x, y, z


def camera_yaw_to_target(
    target_x: float,
    target_y: float,
) -> float:
    """Return the camera yaw needed to face a target in base_link."""
    if not math.isfinite(target_x) or not math.isfinite(target_y):
        raise ValueError("target coordinates must be finite")

    delta_x = target_x - CAMERA_MOUNT_X
    delta_y = target_y - CAMERA_MOUNT_Y

    if math.hypot(delta_x, delta_y) <= 1e-12:
        raise ValueError("target cannot coincide with the camera mount")

    yaw = math.atan2(delta_y, delta_x)

    if not -CAMERA_YAW_LIMIT <= yaw <= CAMERA_YAW_LIMIT:
        raise ValueError("target is outside the camera joint limits")

    return yaw
