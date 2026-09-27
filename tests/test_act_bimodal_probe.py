"""Unit tests for the controlled two-mode teaching input."""

import pytest
import torch

from robot_learning.act_bimodal_probe import make_opposite_action_pair


def test_same_observation_has_opposite_valid_action_chunks() -> None:
    observations, actions, mask = make_opposite_action_pair(
        horizon=4, magnitude=0.8
    )
    assert observations.shape == (2, 2)
    torch.testing.assert_close(observations[0], observations[1])
    assert actions.shape == (2, 4, 1)
    assert torch.all(actions[0] == -0.8)
    assert torch.all(actions[1] == 0.8)
    assert torch.all(mask)


def test_invalid_probe_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="horizon"):
        make_opposite_action_pair(horizon=0, magnitude=0.8)
    with pytest.raises(ValueError, match="magnitude"):
        make_opposite_action_pair(horizon=4, magnitude=1.1)
