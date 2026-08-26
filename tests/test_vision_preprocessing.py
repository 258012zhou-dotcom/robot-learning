"""Unit tests for reliable visual preprocessing geometry."""

import numpy as np
import pytest
import torch

from robot_learning.vision_preprocessing import (
    CameraIntrinsics,
    bgr_image_to_batch_tensor,
    detect_largest_hsv_target,
    inverse_map_bounding_box,
    letterbox_image,
    map_bounding_box,
    transform_camera_intrinsics,
    validate_bgr_image,
)


GREEN_LOWER_HSV = (50, 100, 100)
GREEN_UPPER_HSV = (70, 255, 255)


def test_valid_bgr_image_contract() -> None:
    image = np.zeros((48, 64, 3), dtype=np.uint8)

    validate_bgr_image(image)


@pytest.mark.parametrize(
    "invalid_image",
    [
        np.zeros((48, 64), dtype=np.uint8),
        np.zeros((48, 64, 3), dtype=np.float32),
    ],
)
def test_invalid_bgr_image_contract_is_rejected(
    invalid_image: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        validate_bgr_image(invalid_image)


def test_hsv_target_detection_returns_mask_and_bounding_box() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    image[20:60, 30:70] = (0, 255, 0)

    mask, bounding_box = detect_largest_hsv_target(
        image,
        GREEN_LOWER_HSV,
        GREEN_UPPER_HSV,
    )

    assert mask.shape == (80, 100)
    assert mask.dtype == np.uint8
    assert bounding_box == (30, 20, 40, 40)


def test_hsv_target_detection_selects_largest_region() -> None:
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[10:20, 10:20] = (0, 255, 0)
    image[30:70, 50:100] = (0, 255, 0)

    _, bounding_box = detect_largest_hsv_target(
        image,
        GREEN_LOWER_HSV,
        GREEN_UPPER_HSV,
    )

    assert bounding_box == (50, 30, 50, 40)


def test_hsv_target_detection_returns_none_without_target() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)

    mask, bounding_box = detect_largest_hsv_target(
        image,
        GREEN_LOWER_HSV,
        GREEN_UPPER_HSV,
    )

    assert np.count_nonzero(mask) == 0
    assert bounding_box is None


def test_letterbox_preserves_aspect_ratio_and_adds_padding() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    output, transform = letterbox_image(image, (224, 224))

    assert output.shape == (224, 224, 3)
    assert transform.scale_x == pytest.approx(0.35)
    assert transform.scale_y == pytest.approx(0.35)
    assert transform.padding_left == 0
    assert transform.padding_top == 28


def test_bounding_box_round_trip_preserves_coordinates() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _, transform = letterbox_image(image, (224, 224))
    original_box = (400.0, 120.0, 120.0, 160.0)

    model_box = map_bounding_box(original_box, transform)
    recovered_box = inverse_map_bounding_box(model_box, transform)

    assert model_box == pytest.approx((140.0, 70.0, 42.0, 56.0))
    assert recovered_box == pytest.approx(original_box)


def test_camera_intrinsics_follow_letterbox_geometry() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _, transform = letterbox_image(image, (224, 224))
    intrinsics = CameraIntrinsics(
        fx=600.0,
        fy=600.0,
        cx=319.5,
        cy=239.5,
    )

    transformed = transform_camera_intrinsics(
        intrinsics,
        transform,
    )

    assert transformed.fx == pytest.approx(210.0)
    assert transformed.fy == pytest.approx(210.0)
    assert transformed.cx == pytest.approx(111.825)
    assert transformed.cy == pytest.approx(111.825)


def test_tensor_conversion_changes_bgr_hwc_to_rgb_nchw() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 0] = (10, 20, 30)

    tensor = bgr_image_to_batch_tensor(image)

    assert tensor.shape == (1, 3, 2, 3)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"
    assert tensor[0, :, 0, 0].tolist() == pytest.approx(
        [30 / 255, 20 / 255, 10 / 255]
    )
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0


@pytest.mark.parametrize(
    "output_size",
    [(0, 224), (224, -1), (224.0, 224)],
)
def test_invalid_output_size_is_rejected(output_size) -> None:
    image = np.zeros((48, 64, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        letterbox_image(image, output_size)
