"""Unit tests for reproducible sampled-action evaluation."""

import numpy as np
import pytest
import torch

from robot_learning.ppo_action_selection import SampledDiscretePolicy
from robot_learning.ppo_training import ACTION_VALUES, DiscreteActorCritic


def test_sampled_actions_repeat_with_same_seed_and_stay_in_action_set() -> None:
    model = DiscreteActorCritic()
    for parameter in model.actor.parameters():
        torch.nn.init.zeros_(parameter)
    observation = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    first = SampledDiscretePolicy(model, seed=19)
    second = SampledDiscretePolicy(model, seed=19)

    first_actions = [float(first(observation)[0]) for _ in range(30)]
    second_actions = [float(second(observation)[0]) for _ in range(30)]

    assert first_actions == second_actions
    assert set(first_actions) <= set(ACTION_VALUES)
    assert len(set(first_actions)) > 1


def test_sampling_does_not_advance_global_torch_rng() -> None:
    model = DiscreteActorCritic()
    policy = SampledDiscretePolicy(model, seed=7)
    observation = np.asarray([0.0, 0.0, -1.0, -1.0], dtype=np.float32)
    before = torch.get_rng_state().clone()

    policy(observation)

    assert torch.equal(torch.get_rng_state(), before)


def test_sampling_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SampledDiscretePolicy(DiscreteActorCritic(), seed=-1)
