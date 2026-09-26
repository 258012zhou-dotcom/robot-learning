"""Unit tests for the multi-seed PPO evaluation protocol and summary."""

import pytest

from robot_learning.ppo_multiseed import (
    summarize_success_variation,
    training_seed_ranges,
)


def test_training_seed_ranges_match_predeclared_protocol() -> None:
    ranges = training_seed_ranges(
        [37, 197, 357], episodes_per_run=160,
        evaluation_seeds=range(100037, 100057),
    )

    assert [(items.start, items.stop - 1) for items in ranges] == [
        (37, 196), (197, 356), (357, 516),
    ]


@pytest.mark.parametrize(
    ("starts", "evaluation_seeds", "message"),
    [
        ([37, 100], range(100037, 100057), "training seed ranges overlap"),
        ([37, 37], range(100037, 100057), "distinct"),
        ([37, 197], range(200, 210), "training and evaluation seeds overlap"),
    ],
)
def test_training_seed_ranges_reject_leakage(
    starts: list[int], evaluation_seeds: range, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        training_seed_ranges(starts, episodes_per_run=160, evaluation_seeds=evaluation_seeds)


def test_summary_keeps_all_training_runs_not_just_best() -> None:
    summary = summarize_success_variation([6, 0, 9], episodes_per_run=20)

    assert summary == {
        "training_run_count": 3,
        "evaluation_episodes_per_run": 20,
        "success_counts": [6, 0, 9],
        "minimum_success_count": 0,
        "maximum_success_count": 9,
        "mean_success_rate": 0.25,
    }


def test_summary_rejects_impossible_success_count() -> None:
    with pytest.raises(ValueError, match="within"):
        summarize_success_variation([6, 21], episodes_per_run=20)
