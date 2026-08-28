"""Unit tests for depth backprojection and point-cloud filtering."""

import pytest
import torch

from robot_learning.depth_point_cloud import (
    camera_points_to_pixels,
    depth_image_to_point_cloud,
    pixels_to_camera_points,
)
from robot_learning.vision_preprocessing import CameraIntrinsics


def test_principal_point_backprojects_to_camera_optical_axis() -> None:
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    pixels = torch.tensor([[320.0, 240.0]])
    depths = torch.tensor([2.0])

    points = pixels_to_camera_points(pixels, depths, intrinsics)

    torch.testing.assert_close(points, torch.tensor([[0.0, 0.0, 2.0]]))


def test_offset_pixel_matches_pinhole_backprojection_formula() -> None:
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

    points = pixels_to_camera_points(
        torch.tensor([[420.0, 290.0]]),
        torch.tensor([2.0]),
        intrinsics,
    )

    torch.testing.assert_close(points, torch.tensor([[0.4, 0.2, 2.0]]))


def test_projection_and_backprojection_round_trip_pixels() -> None:
    intrinsics = CameraIntrinsics(fx=400.0, fy=450.0, cx=160.0, cy=120.0)
    pixels = torch.tensor(
        [[160.0, 120.0], [210.0, 80.0], [30.0, 200.0]]
    )
    depths = torch.tensor([1.0, 2.5, 4.0])

    points = pixels_to_camera_points(pixels, depths, intrinsics)
    recovered_pixels = camera_points_to_pixels(points, intrinsics)

    torch.testing.assert_close(recovered_pixels, pixels)


def test_uint16_millimeters_are_scaled_to_meters() -> None:
    depth_mm = torch.tensor([[2000]], dtype=torch.uint16)
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0)

    cloud = depth_image_to_point_cloud(
        depth_mm,
        intrinsics,
        depth_scale_to_meters=0.001,
    )

    torch.testing.assert_close(
        cloud.points_m,
        torch.tensor([[0.0, 0.0, 2.0]]),
    )
    assert cloud.pixels_uv.tolist() == [[0, 0]]


def test_invalid_and_out_of_range_depth_values_are_filtered() -> None:
    depth = torch.tensor(
        [
            [0.0, float("nan"), 0.4],
            [float("inf"), 1.0, 6.0],
        ]
    )
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=1.0, cy=0.5)

    cloud = depth_image_to_point_cloud(
        depth,
        intrinsics,
        minimum_depth_m=0.5,
        maximum_depth_m=5.0,
    )

    assert cloud.pixels_uv.tolist() == [[1, 1]]
    assert cloud.points_m[:, 2].tolist() == pytest.approx([1.0])


def test_boolean_selection_mask_keeps_only_target_pixels() -> None:
    depth = torch.full((2, 3), 2.0)
    selection = torch.tensor(
        [
            [False, True, False],
            [False, False, True],
        ]
    )
    intrinsics = CameraIntrinsics(fx=2.0, fy=2.0, cx=1.0, cy=0.0)

    cloud = depth_image_to_point_cloud(
        depth,
        intrinsics,
        selection_mask=selection,
    )

    assert cloud.pixels_uv.tolist() == [[1, 0], [2, 1]]
    torch.testing.assert_close(
        cloud.points_m,
        torch.tensor([[0.0, 0.0, 2.0], [1.0, 1.0, 2.0]]),
    )


def test_no_valid_depth_returns_empty_point_cloud() -> None:
    depth = torch.zeros((3, 4))
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=1.5, cy=1.0)

    cloud = depth_image_to_point_cloud(depth, intrinsics)

    assert cloud.points_m.shape == (0, 3)
    assert cloud.pixels_uv.shape == (0, 2)


def test_selection_mask_shape_must_match_depth_image() -> None:
    depth = torch.ones((3, 4))
    selection = torch.ones((2, 4), dtype=torch.bool)
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=1.5, cy=1.0)

    with pytest.raises(ValueError, match="match"):
        depth_image_to_point_cloud(
            depth,
            intrinsics,
            selection_mask=selection,
        )


def test_projection_rejects_points_behind_camera() -> None:
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=0.0, cy=0.0)

    with pytest.raises(ValueError, match="positive Z"):
        camera_points_to_pixels(
            torch.tensor([[0.0, 0.0, -1.0]]),
            intrinsics,
        )
