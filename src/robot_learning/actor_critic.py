"""Value-prediction loss used to train a Critic."""

import torch
from torch import Tensor


def value_prediction_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Fit V to a return or TD target without updating the target itself."""
    if prediction.shape != target.shape or prediction.numel() == 0:
        raise ValueError("prediction and target must have equal non-empty shapes")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("value-prediction inputs must be finite")
    return torch.square(prediction - target.detach()).mean()
