"""Unit tests for one-step TD targets and updates."""

import pytest

from robot_learning.temporal_difference import update_state_value_td


def test_non_terminal_td_update_bootstraps_from_next_value() -> None:
    """The lesson's numeric example must produce target 4.4 and value 4.08."""
    result = update_state_value_td(
        4.0,
        reward=-1.0,
        next_value=6.0,
        discount_factor=0.9,
        learning_rate=0.2,
        terminated=False,
    )

    assert result.target == pytest.approx(4.4)
    assert result.error == pytest.approx(0.4)
    assert result.updated_value == pytest.approx(4.08)


def test_terminal_td_target_does_not_bootstrap() -> None:
    """A terminal transition has no future value, even if one is supplied."""
    result = update_state_value_td(
        3.0,
        reward=1.0,
        next_value=1000.0,
        discount_factor=0.9,
        learning_rate=0.5,
        terminated=True,
    )

    assert result.target == 1.0
    assert result.error == -2.0
    assert result.updated_value == 2.0


@pytest.mark.parametrize(
    ("discount_factor", "learning_rate"),
    [(-0.1, 0.1), (1.1, 0.1), (0.9, 0.0), (0.9, 1.1)],
)
def test_invalid_td_hyperparameters_are_rejected(
    discount_factor: float,
    learning_rate: float,
) -> None:
    """Invalid gamma or alpha must not silently create value targets."""
    with pytest.raises(ValueError):
        update_state_value_td(
            0.0,
            reward=0.0,
            next_value=0.0,
            discount_factor=discount_factor,
            learning_rate=learning_rate,
            terminated=False,
        )


def test_td_rejects_non_boolean_termination_flag() -> None:
    """Integer flags must not accidentally change terminal bootstrapping."""
    with pytest.raises(ValueError, match="boolean"):
        update_state_value_td(
            0.0,
            reward=0.0,
            next_value=1.0,
            discount_factor=0.9,
            learning_rate=0.1,
            terminated=1,
        )
