"""Unit tests for object-detection box geometry and post-processing."""

import pytest

from robot_learning.detection_geometry import (
    Detection,
    box_area_xyxy,
    clip_box_xyxy,
    filter_detections,
    intersection_over_union,
    non_maximum_suppression,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


def test_box_coordinate_conversion_round_trip() -> None:
    box_xywh = (10.0, 20.0, 30.0, 40.0)

    box_xyxy = xywh_to_xyxy(box_xywh)

    assert box_xyxy == (10.0, 20.0, 40.0, 60.0)
    assert xyxy_to_xywh(box_xyxy) == box_xywh


def test_box_area_uses_continuous_coordinate_convention() -> None:
    assert box_area_xyxy((10.0, 20.0, 40.0, 60.0)) == 1200.0


def test_box_is_clipped_to_image_boundaries() -> None:
    clipped = clip_box_xyxy(
        (-5.0, 10.0, 110.0, 90.0),
        image_size=(100, 80),
    )

    assert clipped == (0.0, 10.0, 100.0, 80.0)


def test_identical_boxes_have_iou_one() -> None:
    box = (0.0, 0.0, 10.0, 10.0)

    assert intersection_over_union(box, box) == pytest.approx(1.0)


def test_non_overlapping_boxes_have_iou_zero() -> None:
    assert intersection_over_union(
        (0.0, 0.0, 10.0, 10.0),
        (20.0, 20.0, 30.0, 30.0),
    ) == pytest.approx(0.0)


def test_partially_overlapping_boxes_have_expected_iou() -> None:
    # Each box has area 100; their 5x5 intersection has area 25.
    # IoU = 25 / (100 + 100 - 25) = 1/7.
    iou = intersection_over_union(
        (0.0, 0.0, 10.0, 10.0),
        (5.0, 5.0, 15.0, 15.0),
    )

    assert iou == pytest.approx(1.0 / 7.0)


def test_confidence_filter_keeps_threshold_boundary() -> None:
    detections = [
        Detection((0.0, 0.0, 10.0, 10.0), score=0.49, class_id=0),
        Detection((20.0, 20.0, 30.0, 30.0), score=0.50, class_id=1),
    ]

    filtered = filter_detections(detections, minimum_score=0.5)

    assert filtered == [detections[1]]


def test_nms_suppresses_lower_score_overlapping_box_of_same_class() -> None:
    best = Detection((0.0, 0.0, 10.0, 10.0), score=0.9, class_id=0)
    duplicate = Detection((1.0, 1.0, 11.0, 11.0), score=0.7, class_id=0)
    separate = Detection((20.0, 20.0, 30.0, 30.0), score=0.6, class_id=0)

    kept = non_maximum_suppression(
        [duplicate, separate, best],
        iou_threshold=0.5,
    )

    assert kept == [best, separate]


def test_nms_keeps_overlapping_boxes_from_different_classes() -> None:
    square = Detection((0.0, 0.0, 10.0, 10.0), score=0.9, class_id=0)
    circle = Detection((1.0, 1.0, 11.0, 11.0), score=0.8, class_id=1)

    kept = non_maximum_suppression(
        [circle, square],
        iou_threshold=0.5,
    )

    assert kept == [square, circle]


@pytest.mark.parametrize(
    "box",
    [
        (0.0, 0.0, 0.0, 10.0),
        (10.0, 0.0, 0.0, 10.0),
        (0.0, 0.0, float("nan"), 10.0),
    ],
)
def test_detection_rejects_invalid_boxes(box) -> None:
    with pytest.raises(ValueError):
        Detection(box, score=0.8, class_id=0)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, True])
def test_nms_rejects_invalid_threshold(threshold) -> None:
    with pytest.raises(ValueError):
        non_maximum_suppression([], threshold)
