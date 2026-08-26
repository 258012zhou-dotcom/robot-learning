"""Bounding-box geometry and post-processing for object detection."""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Sequence


BoxXYXY = tuple[float, float, float, float]


def _as_box_xyxy(box: Sequence[Real]) -> BoxXYXY:
    """Validate and convert a four-value box to floating-point xyxy form."""
    if len(box) != 4:
        raise ValueError("box must contain four coordinates")
    if any(
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in box
    ):
        raise ValueError("box coordinates must be finite numbers")

    x_min, y_min, x_max, y_max = (float(value) for value in box)
    if x_max < x_min or y_max < y_min:
        raise ValueError("box maximum coordinates must not be smaller than minima")
    return x_min, y_min, x_max, y_max


@dataclass(frozen=True)
class Detection:
    """One class-labelled detection in continuous xyxy coordinates."""

    box_xyxy: BoxXYXY
    score: float
    class_id: int

    def __post_init__(self) -> None:
        validated_box = _as_box_xyxy(self.box_xyxy)
        if (
            validated_box[2] == validated_box[0]
            or validated_box[3] == validated_box[1]
        ):
            raise ValueError("detection box must have positive area")
        if (
            not isinstance(self.score, Real)
            or isinstance(self.score, bool)
            or not math.isfinite(float(self.score))
            or not 0.0 <= float(self.score) <= 1.0
        ):
            raise ValueError("score must be a finite number in [0, 1]")
        if (
            not isinstance(self.class_id, Integral)
            or isinstance(self.class_id, bool)
        ):
            raise ValueError("class_id must be a non-negative integer")
        if int(self.class_id) < 0:
            raise ValueError("class_id must be a non-negative integer")

        object.__setattr__(self, "box_xyxy", validated_box)
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "class_id", int(self.class_id))


def xywh_to_xyxy(box_xywh: Sequence[Real]) -> BoxXYXY:
    """Convert top-left xywh coordinates to xyxy coordinates."""
    if len(box_xywh) != 4:
        raise ValueError("box must contain four values")
    if any(
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in box_xywh
    ):
        raise ValueError("box values must be finite numbers")

    x_min, y_min, width, height = (float(value) for value in box_xywh)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("box width and height must be positive")
    return x_min, y_min, x_min + width, y_min + height


def xyxy_to_xywh(box_xyxy: Sequence[Real]) -> BoxXYXY:
    """Convert xyxy coordinates to top-left xywh coordinates."""
    x_min, y_min, x_max, y_max = _as_box_xyxy(box_xyxy)
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0.0 or height <= 0.0:
        raise ValueError("box width and height must be positive")
    return x_min, y_min, width, height


def box_area_xyxy(box_xyxy: Sequence[Real]) -> float:
    """Return the area of a valid or degenerate xyxy box."""
    x_min, y_min, x_max, y_max = _as_box_xyxy(box_xyxy)
    return (x_max - x_min) * (y_max - y_min)


def clip_box_xyxy(
    box_xyxy: Sequence[Real],
    image_size: tuple[int, int],
) -> BoxXYXY:
    """Clip an xyxy box to an image whose size is (width, height)."""
    x_min, y_min, x_max, y_max = _as_box_xyxy(box_xyxy)
    if (
        len(image_size) != 2
        or any(type(value) is not int or value <= 0 for value in image_size)
    ):
        raise ValueError("image_size must be positive integer (width, height)")

    image_width, image_height = image_size
    clipped = (
        min(max(x_min, 0.0), float(image_width)),
        min(max(y_min, 0.0), float(image_height)),
        min(max(x_max, 0.0), float(image_width)),
        min(max(y_max, 0.0), float(image_height)),
    )
    if box_area_xyxy(clipped) == 0.0:
        raise ValueError("clipped box has no area inside the image")
    return clipped


def intersection_over_union(
    first_box: Sequence[Real],
    second_box: Sequence[Real],
) -> float:
    """Compute intersection over union (IoU) for two xyxy boxes."""
    first = _as_box_xyxy(first_box)
    second = _as_box_xyxy(second_box)
    first_area = box_area_xyxy(first)
    second_area = box_area_xyxy(second)
    if first_area == 0.0 or second_area == 0.0:
        raise ValueError("IoU requires boxes with positive area")

    intersection_width = max(
        0.0,
        min(first[2], second[2]) - max(first[0], second[0]),
    )
    intersection_height = max(
        0.0,
        min(first[3], second[3]) - max(first[1], second[1]),
    )
    intersection_area = intersection_width * intersection_height
    union_area = first_area + second_area - intersection_area
    return intersection_area / union_area


def filter_detections(
    detections: Sequence[Detection],
    minimum_score: float,
) -> list[Detection]:
    """Keep detections whose confidence score reaches the threshold."""
    if (
        not isinstance(minimum_score, Real)
        or isinstance(minimum_score, bool)
        or not 0.0 <= float(minimum_score) <= 1.0
    ):
        raise ValueError("minimum_score must be in [0, 1]")
    return [
        detection
        for detection in detections
        if detection.score >= float(minimum_score)
    ]


def non_maximum_suppression(
    detections: Sequence[Detection],
    iou_threshold: float,
) -> list[Detection]:
    """Apply greedy class-aware non-maximum suppression."""
    if (
        not isinstance(iou_threshold, Real)
        or isinstance(iou_threshold, bool)
        or not 0.0 <= float(iou_threshold) <= 1.0
    ):
        raise ValueError("iou_threshold must be in [0, 1]")

    remaining = sorted(
        detections,
        key=lambda detection: detection.score,
        reverse=True,
    )
    kept: list[Detection] = []
    while remaining:
        selected = remaining.pop(0)
        kept.append(selected)
        remaining = [
            candidate
            for candidate in remaining
            if candidate.class_id != selected.class_id
            or intersection_over_union(
                selected.box_xyxy,
                candidate.box_xyxy,
            ) <= float(iou_threshold)
        ]
    return kept
