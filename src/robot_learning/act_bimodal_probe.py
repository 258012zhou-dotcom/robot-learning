"""Two deliberately contradictory demonstrations for an ACT mechanism check."""

import torch
from torch import Tensor


def make_opposite_action_pair(
    *, horizon: int, magnitude: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Return identical observations with left/right action chunks."""
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if not 0.0 < magnitude <= 1.0:
        raise ValueError("magnitude must be in (0, 1]")
    observations = torch.zeros((2, 2), dtype=torch.float32)
    actions = torch.empty((2, horizon, 1), dtype=torch.float32)
    actions[0].fill_(-magnitude)
    actions[1].fill_(magnitude)
    valid_mask = torch.ones((2, horizon), dtype=torch.bool)
    return observations, actions, valid_mask
