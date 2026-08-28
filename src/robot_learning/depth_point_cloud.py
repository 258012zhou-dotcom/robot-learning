"""Depth-image backprojection and pinhole point-cloud geometry."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from robot_learning.vision_preprocessing import CameraIntrinsics


@dataclass(frozen=True)
class PointCloud:
    """Unorganized camera-frame points and their source image pixels."""

    points_m: Tensor
    pixels_uv: Tensor

    def __post_init__(self) -> None:
        if self.points_m.ndim != 2 or self.points_m.shape[1] != 3:
            raise ValueError("points_m must have shape (N, 3)")
        if self.pixels_uv.ndim != 2 or self.pixels_uv.shape[1] != 2:
            raise ValueError("pixels_uv must have shape (N, 2)")
        if self.points_m.shape[0] != self.pixels_uv.shape[0]:
            raise ValueError("points and pixels must contain equal counts")
        if not torch.is_floating_point(self.points_m):
            raise ValueError("points_m must use a floating-point dtype")
        if self.pixels_uv.dtype != torch.int64:
            raise ValueError("pixels_uv must use torch.int64")


def pixels_to_camera_points(
    pixels_uv: Tensor,
    depths_m: Tensor,
    intrinsics: CameraIntrinsics,
) -> Tensor:
    """Backproject pixel coordinates and Z-depths into camera optical frame."""
    if pixels_uv.ndim != 2 or pixels_uv.shape[1] != 2:
        raise ValueError("pixels_uv must have shape (N, 2)")
    if depths_m.ndim != 1 or depths_m.shape[0] != pixels_uv.shape[0]:
        raise ValueError("depths_m must have shape (N,)")
    if pixels_uv.shape[0] == 0:
        return torch.empty(
            (0, 3),
            dtype=torch.float32,
            device=pixels_uv.device,
        )

    pixels = pixels_uv.to(dtype=torch.float32)
    depths = depths_m.to(dtype=torch.float32)
    if not torch.isfinite(pixels).all():
        raise ValueError("pixel coordinates must be finite")
    if not torch.isfinite(depths).all() or torch.any(depths <= 0.0):
        raise ValueError("depths_m must contain finite positive Z-depths")

    horizontal = (
        (pixels[:, 0] - intrinsics.cx)
        * depths
        / intrinsics.fx
    )
    vertical = (
        (pixels[:, 1] - intrinsics.cy)
        * depths
        / intrinsics.fy
    )
    return torch.stack((horizontal, vertical, depths), dim=1)


def camera_points_to_pixels(
    points_m: Tensor,
    intrinsics: CameraIntrinsics,
) -> Tensor:
    """Project camera-frame XYZ points to floating-point image pixels."""
    if points_m.ndim != 2 or points_m.shape[1] != 3:
        raise ValueError("points_m must have shape (N, 3)")
    if points_m.shape[0] == 0:
        return torch.empty(
            (0, 2),
            dtype=torch.float32,
            device=points_m.device,
        )

    points = points_m.to(dtype=torch.float32)
    if not torch.isfinite(points).all() or torch.any(points[:, 2] <= 0.0):
        raise ValueError("points must be finite and have positive Z")
    horizontal = intrinsics.fx * points[:, 0] / points[:, 2] + intrinsics.cx
    vertical = intrinsics.fy * points[:, 1] / points[:, 2] + intrinsics.cy
    return torch.stack((horizontal, vertical), dim=1)


def depth_image_to_point_cloud(
    depth_image: Tensor,
    intrinsics: CameraIntrinsics,
    *,
    depth_scale_to_meters: float = 1.0,
    minimum_depth_m: float = 0.0,
    maximum_depth_m: float = float("inf"),
    selection_mask: Tensor | None = None,
) -> PointCloud:
    """Convert valid selected depth pixels to an unorganized point cloud."""
    if depth_image.ndim != 2:
        raise ValueError("depth_image must have shape (H, W)")
    if depth_image.shape[0] == 0 or depth_image.shape[1] == 0:
        raise ValueError("depth_image must not be empty")
    if (
        not math.isfinite(depth_scale_to_meters)
        or depth_scale_to_meters <= 0.0
    ):
        raise ValueError("depth_scale_to_meters must be positive and finite")
    if (
        not math.isfinite(minimum_depth_m)
        or minimum_depth_m < 0.0
        or math.isnan(maximum_depth_m)
        or maximum_depth_m <= minimum_depth_m
    ):
        raise ValueError("depth limits must satisfy 0 <= minimum < maximum")
    if selection_mask is not None:
        if selection_mask.shape != depth_image.shape:
            raise ValueError("selection_mask must match the depth image shape")
        if selection_mask.dtype != torch.bool:
            raise ValueError("selection_mask must use torch.bool")

    depths_m = depth_image.to(dtype=torch.float32) * depth_scale_to_meters
    valid = torch.isfinite(depths_m)
    valid &= depths_m > 0.0
    valid &= depths_m >= minimum_depth_m
    valid &= depths_m <= maximum_depth_m
    if selection_mask is not None:
        valid &= selection_mask

    vertical_pixels, horizontal_pixels = torch.nonzero(
        valid,
        as_tuple=True,
    )
    pixels_uv = torch.stack(
        (horizontal_pixels, vertical_pixels),
        dim=1,
    ).to(dtype=torch.int64)
    selected_depths = depths_m[vertical_pixels, horizontal_pixels]
    points_m = pixels_to_camera_points(
        pixels_uv,
        selected_depths,
        intrinsics,
    )
    return PointCloud(points_m=points_m, pixels_uv=pixels_uv)
