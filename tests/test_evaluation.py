import pytest
import torch

from robot_learning.evaluation import (
    euclidean_errors,
    summarize_errors_by_speed,
    summarize_prediction_errors,
)


def test_error_summary_uses_distinct_coordinate_and_euclidean_metrics():
    predictions = torch.zeros(2, 2)
    targets = torch.tensor([[3.0, 4.0], [0.0, 0.0]])

    summary = summarize_prediction_errors(predictions, targets)

    assert summary.coordinate_rmse == pytest.approx(2.5)
    assert summary.median_euclidean_error == pytest.approx(0.0)
    assert summary.p95_euclidean_error == pytest.approx(4.75)
    assert summary.maximum_euclidean_error == pytest.approx(5.0)


def test_euclidean_errors_are_returned_per_sample():
    predictions = torch.tensor([[1.0, 1.0], [3.0, 4.0]])
    targets = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

    errors = euclidean_errors(predictions, targets)

    torch.testing.assert_close(errors, torch.tensor([0.0, 5.0]))


def test_speed_groups_have_expected_counts_and_rmse():
    predictions = torch.zeros(4, 2)
    targets = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 0.0],
        ]
    )
    velocities = torch.tensor(
        [
            [0.1, 0.0],
            [0.2, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )

    groups = summarize_errors_by_speed(
        predictions,
        targets,
        velocities,
    )

    assert groups.speed_threshold == pytest.approx(0.2)
    assert groups.low_speed_count == 2
    assert groups.high_speed_count == 2
    assert groups.low_speed_rmse == pytest.approx(2 ** -0.5)
    assert groups.high_speed_rmse == pytest.approx(2 ** 0.5)


@pytest.mark.parametrize(
    ("predictions", "targets"),
    [
        (torch.empty(0, 2), torch.empty(0, 2)),
        (torch.zeros(3, 3), torch.zeros(3, 3)),
        (torch.zeros(3, 2), torch.zeros(2, 2)),
    ],
)
def test_error_summary_rejects_invalid_shapes(predictions, targets):
    with pytest.raises(ValueError):
        summarize_prediction_errors(predictions, targets)


def test_speed_analysis_rejects_group_without_variation():
    predictions = torch.zeros(3, 2)
    targets = torch.zeros(3, 2)
    velocities = torch.ones(3, 2)

    with pytest.raises(ValueError, match="both contain"):
        summarize_errors_by_speed(predictions, targets, velocities)
