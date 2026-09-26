"""Unit tests for one SAC update; no environment is stepped."""

import numpy as np
import pytest
import torch

from robot_learning.sac_dataflow import ReplayBatch
from robot_learning.sac_models import SquashedGaussianActor, TwinQCritic
from robot_learning.sac_training import (
    make_target_critic,
    sac_update,
    soft_update_target,
)


def test_target_critic_starts_equal_and_soft_update_interpolates() -> None:
    online = TwinQCritic()
    target = make_target_critic(online)
    assert all(not parameter.requires_grad for parameter in target.parameters())
    for online_parameter, target_parameter in zip(
        online.parameters(), target.parameters(), strict=True,
    ):
        torch.testing.assert_close(online_parameter, target_parameter)

    with torch.no_grad():
        for online_parameter in online.parameters():
            online_parameter.fill_(1.0)
        for target_parameter in target.parameters():
            target_parameter.zero_()
    soft_update_target(target, online, tau=0.25)
    for target_parameter in target.parameters():
        torch.testing.assert_close(target_parameter, torch.full_like(target_parameter, 0.25))

    with pytest.raises(ValueError, match="tau"):
        soft_update_target(target, online, tau=1.5)


def test_one_update_changes_online_networks_but_target_only_tracks_softly() -> None:
    torch.manual_seed(8)
    actor = SquashedGaussianActor()
    critic = TwinQCritic()
    target = make_target_critic(critic)
    actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in critic.parameters()]
    target_before = [parameter.detach().clone() for parameter in target.parameters()]

    rng = np.random.default_rng(8)
    batch = ReplayBatch(
        indices=np.arange(8),
        observations=rng.normal(size=(8, 4)).astype(np.float32),
        actions=rng.uniform(-1, 1, size=(8, 1)).astype(np.float32),
        rewards=rng.normal(size=8).astype(np.float32),
        next_observations=rng.normal(size=(8, 4)).astype(np.float32),
        terminated=np.asarray([False] * 7 + [True]),
        truncated=np.zeros(8, dtype=np.bool_),
    )
    metrics = sac_update(
        actor, critic, target,
        torch.optim.Adam(actor.parameters(), lr=1e-3),
        torch.optim.Adam(critic.parameters(), lr=1e-3),
        batch, gamma=0.99, alpha=0.2, tau=0.1,
    )

    assert np.isfinite(metrics.critic_loss) and np.isfinite(metrics.actor_loss)
    assert metrics.critic_loss >= 0.0
    assert any(not torch.equal(before, after) for before, after in zip(
        actor_before, actor.parameters(), strict=True,
    ))
    assert any(not torch.equal(before, after) for before, after in zip(
        critic_before, critic.parameters(), strict=True,
    ))
    for before, after, online in zip(
        target_before, target.parameters(), critic.parameters(), strict=True,
    ):
        torch.testing.assert_close(after, before * 0.9 + online.detach() * 0.1)
        assert after.grad is None
    assert all(parameter.grad is None for parameter in critic.parameters())
    assert all(parameter.requires_grad for parameter in critic.parameters())
