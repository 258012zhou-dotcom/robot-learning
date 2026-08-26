"""Reliable image preprocessing and coordinate transformations."""

from dataclasses import dataclass
import math

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics in pixel units."""

    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        values = (self.fx, self.fy, self.cx, self.cy)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("camera intrinsics must be finite")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")


@dataclass(frozen=True)
class LetterboxTransform:
    """Geometry needed to map coordinates through Letterbox."""

    original_width: int
    original_height: int
    output_width: int
    output_height: int
    scale_x: float
    scale_y: float
    padding_left: int
    padding_top: int


def validate_bgr_image(image_bgr: np.ndarray) -> None:
    """Validate the image contract used by this preprocessing pipeline."""
    if not isinstance(image_bgr, np.ndarray):
        raise ValueError("image_bgr must be a NumPy array")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape (H, W, 3)")
    if image_bgr.dtype != np.uint8:
        raise ValueError("image_bgr must use uint8 pixels")
    if image_bgr.shape[0] == 0 or image_bgr.shape[1] == 0:
        raise ValueError("image_bgr cannot be empty")


def detect_largest_hsv_target(
    image_bgr: np.ndarray,
    lower_hsv: tuple[int, int, int],
    upper_hsv: tuple[int, int, int],
    *,
    minimum_area: float = 1.0,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Create an HSV mask and return its largest target bounding box."""
    validate_bgr_image(image_bgr)
    lower = _validate_hsv_bound(lower_hsv, "lower_hsv")
    upper = _validate_hsv_bound(upper_hsv, "upper_hsv")
    if np.any(lower > upper):
        raise ValueError("lower_hsv cannot exceed upper_hsv")
    if not math.isfinite(minimum_area) or minimum_area <= 0.0:
        raise ValueError("minimum_area must be a positive finite number")

    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, lower, upper)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= minimum_area
    ]
    if not valid_contours:
        return mask, None

    largest_contour = max(valid_contours, key=cv2.contourArea)
    return mask, cv2.boundingRect(largest_contour)


def letterbox_image(
    image_bgr: np.ndarray,
    output_size: tuple[int, int],
    fill_bgr: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize without distortion and pad to ``(width, height)``."""
    validate_bgr_image(image_bgr)
    output_width, output_height = _validate_output_size(output_size)
    fill = _validate_fill_color(fill_bgr)
    original_height, original_width = image_bgr.shape[:2]

    requested_scale = min(
        output_width / original_width,
        output_height / original_height,
    )
    resized_width = max(1, round(original_width * requested_scale))
    resized_height = max(1, round(original_height * requested_scale))
    resized = cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    horizontal_padding = output_width - resized_width
    vertical_padding = output_height - resized_height
    padding_left = horizontal_padding // 2
    padding_right = horizontal_padding - padding_left
    padding_top = vertical_padding // 2
    padding_bottom = vertical_padding - padding_top

    output = cv2.copyMakeBorder(
        resized,
        padding_top,
        padding_bottom,
        padding_left,
        padding_right,
        cv2.BORDER_CONSTANT,
        value=fill,
    )
    transform = LetterboxTransform(
        original_width=original_width,
        original_height=original_height,
        output_width=output_width,
        output_height=output_height,
        scale_x=resized_width / original_width,
        scale_y=resized_height / original_height,
        padding_left=padding_left,
        padding_top=padding_top,
    )
    return output, transform


def map_bounding_box(
    bounding_box: tuple[float, float, float, float],
    transform: LetterboxTransform,
) -> tuple[float, float, float, float]:
    """Map an ``(x, y, width, height)`` box to Letterbox coordinates."""
    x, y, width, height = _validate_bounding_box(bounding_box)
    return (
        x * transform.scale_x + transform.padding_left,
        y * transform.scale_y + transform.padding_top,
        width * transform.scale_x,
        height * transform.scale_y,
    )


def inverse_map_bounding_box(
    bounding_box: tuple[float, float, float, float],
    transform: LetterboxTransform,
) -> tuple[float, float, float, float]:
    """Map a Letterbox box back to original image coordinates."""
    x, y, width, height = _validate_bounding_box(bounding_box)
    return (
        (x - transform.padding_left) / transform.scale_x,
        (y - transform.padding_top) / transform.scale_y,
        width / transform.scale_x,
        height / transform.scale_y,
    )


def transform_camera_intrinsics(
    intrinsics: CameraIntrinsics,
    transform: LetterboxTransform,
) -> CameraIntrinsics:
    """Update pinhole intrinsics after Resize and Padding."""
    return CameraIntrinsics(
        fx=intrinsics.fx * transform.scale_x,
        fy=intrinsics.fy * transform.scale_y,
        cx=(
            intrinsics.cx * transform.scale_x
            + transform.padding_left
        ),
        cy=(
            intrinsics.cy * transform.scale_y
            + transform.padding_top
        ),
    )


def bgr_image_to_batch_tensor(
    image_bgr: np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Convert BGR uint8 HWC data to RGB float32 NCHW data."""
    validate_bgr_image(image_bgr)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)
    tensor = tensor.to(dtype=torch.float32).div(255.0).unsqueeze(0)
    return tensor.to(device=device)


def _validate_output_size(
    output_size: tuple[int, int],
) -> tuple[int, int]:
    if (
        len(output_size) != 2
        or any(type(value) is not int for value in output_size)
        or any(value <= 0 for value in output_size)
    ):
        raise ValueError("output_size must contain positive integer width and height")
    return output_size


def _validate_fill_color(
    fill_bgr: tuple[int, int, int],
) -> tuple[int, int, int]:
    if (
        len(fill_bgr) != 3
        or any(type(value) is not int for value in fill_bgr)
        or any(value < 0 or value > 255 for value in fill_bgr)
    ):
        raise ValueError("fill_bgr must contain three integers from 0 to 255")
    return fill_bgr


def _validate_hsv_bound(
    bound: tuple[int, int, int],
    name: str,
) -> np.ndarray:
    if (
        len(bound) != 3
        or any(type(value) is not int for value in bound)
        or bound[0] < 0
        or bound[0] > 179
        or any(value < 0 or value > 255 for value in bound[1:])
    ):
        raise ValueError(
            f"{name} must contain H in [0, 179] and S/V in [0, 255]"
        )
    return np.asarray(bound, dtype=np.uint8)


def _validate_bounding_box(
    bounding_box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if len(bounding_box) != 4:
        raise ValueError("bounding_box must contain x, y, width, and height")
    values = tuple(float(value) for value in bounding_box)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bounding_box must be finite")
    if values[2] <= 0.0 or values[3] <= 0.0:
        raise ValueError("bounding_box width and height must be positive")
    return values
